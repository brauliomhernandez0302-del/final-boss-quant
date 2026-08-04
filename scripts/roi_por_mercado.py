#!/usr/bin/env python3
"""ROI por umbral de edge, en los TRES mercados — el instrumento que faltaba.

EL VACÍO QUE CIERRA
===================
`backtest_and_retrain.py` puntúa **exclusivamente moneyline**: Brier, accuracy y
ROI sobre `home_won`. Pero de los picks realmente publicados, **298 de 431 son
runline o total** — el 69 %. O sea que el instrumento que valida el modelo mide
un mercado y el sistema apuesta tres.

No era un vacío de modelado: el Monte Carlo YA calcula `p_rl_home`/`p_rl_away`
en cada corrida del backtest y las descarta, y `p_over`/`p_under` salen con sólo
pasarle la línea. Era un vacío de DATOS — `historical_odds` no tenía ni una
columna de total o runline, así que no había contra qué comparar. Eso se
rellenó el 2026-08-04.

CÓMO MIDE
=========
Re-simula cada juego desde las λ finales ya guardadas en
`game_outcomes.backtest_lambda_*` (mismo enfoque que `derived_eval`), con seed
determinístico por `game_pk`, y compara contra el resultado real.

  probabilidad justa   devig del par de PINNACLE de ese mismo mercado. Es la
                       referencia más limpia disponible y es lo que decide si
                       hay ventaja.
  precio de la apuesta MEJOR par disponible. Es a lo que realmente se apostaría
                       y es lo que decide cuánto rinde.

Los dos, separados a propósito: usar el mejor precio también como referencia
justa inventa ventaja donde sólo hay line shopping.

EL PUNTO FIRMADO NO ES OPCIONAL
===============================
Para el runline se usa `rl_home_point_*`. Un precio de runline sin saber quién
pone el −1.5 no identifica ningún evento: si el local es el favorito, cubrir es
ganar por 2+; si es el no-favorito, cubrir es perder por menos de 2. Asumir lo
primero siempre es exactamente PURP-1, que publicó 55 picks con EV falso. Un
juego sin punto firmado se excluye, no se adivina.

QUÉ NO ES
=========
No es un backtest del pipeline: las λ son entrada, no se recalculan. Mide la
capa de decisión —probabilidad contra precio— sobre λ ya fijadas. Si las λ
cambian, hay que re-correr el backtest antes que esto.

Uso:
    python scripts/roi_por_mercado.py
    python scripts/roi_por_mercado.py --season 2025 --n-sims 200000
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced  # noqa: E402

DB = RAIZ / "data" / "predictions_history.db"
UMBRALES = [0.00, 0.02, 0.05, 0.08, 0.10]
# Mismo criterio que el ROI de moneyline del backtest: una cuota extrema es un
# mercado que en la práctica no se apuesta y domina el ROI por puro tamaño.
CUOTA_MAXIMA = 6.0


def _devig(a: float, b: float) -> tuple[float, float]:
    """Multiplicativo — el mismo que usa el motor (`remove_vig_multiplicative`)."""
    ia, ib = 1.0 / a, 1.0 / b
    t = ia + ib
    return ia / t, ib / t


def cargar(con: sqlite3.Connection, season: int | None) -> list[dict]:
    filtro = "AND o.season = ?" if season else ""
    params = (season,) if season else ()
    sql = f"""
        SELECT o.game_pk, o.season, o.game_date, o.home_won,
               o.actual_home_runs, o.actual_away_runs,
               o.backtest_lambda_home AS lh, o.backtest_lambda_away AS la,
               h.ml_home_pin, h.ml_away_pin, h.ml_home_best, h.ml_away_best,
               h.total_point_pin, h.total_over_pin, h.total_under_pin,
               h.total_point_best, h.total_over_best, h.total_under_best,
               h.rl_home_point_pin, h.rl_home_pin, h.rl_away_pin,
               h.rl_home_point_best, h.rl_home_best, h.rl_away_best
        FROM game_outcomes o
        JOIN historical_odds h USING (game_pk)
        WHERE o.backtest_lambda_home IS NOT NULL
          AND o.actual_home_runs IS NOT NULL
          {filtro}
        ORDER BY o.game_date
    """
    return [dict(r) for r in con.execute(sql, params)]


def apuestas_del_juego(g: dict, home: np.ndarray, away: np.ndarray,
                       p_home_sim: float) -> list[dict]:
    """Todas las apuestas evaluables de un juego, con su edge y su resultado."""
    n = len(home)
    margen = home - away
    total = home + away
    out: list[dict] = []

    # ── Moneyline ──────────────────────────────────────────────────────────
    # `p_home` del simulador, NO `(home > away).mean()`. El 10.9 % de las
    # simulaciones terminan empatadas —juegos que en la realidad irían a
    # entradas extra— y el simulador las reparte proporcionalmente al ruido de
    # λ de esa misma simulación. Contarlas todas como derrota del local
    # sub-estimaba su probabilidad en 5.6 pp, que a su vez fabricaba un "edge"
    # sistemático hacia el visitante en el 84 % de los juegos. Ese era un bug
    # de ESTE script, no del modelo — encontrado comparando contra
    # `game_outcomes.backtest_p_home_raw`, que daba 0.5207 donde este cálculo
    # daba 0.4674.
    if all(g[k] for k in ("ml_home_pin", "ml_away_pin", "ml_home_best", "ml_away_best")):
        p_mod = float(p_home_sim)
        fair_h, fair_a = _devig(g["ml_home_pin"], g["ml_away_pin"])
        gano_local = bool(g["home_won"])
        out += [
            {"mercado": "MONEYLINE", "lado": "HOME", "p": p_mod, "fair": fair_h,
             "cuota": g["ml_home_best"], "gano": gano_local},
            {"mercado": "MONEYLINE", "lado": "AWAY", "p": 1 - p_mod, "fair": fair_a,
             "cuota": g["ml_away_best"], "gano": not gano_local},
        ]

    # ── Total ──────────────────────────────────────────────────────────────
    # La probabilidad se calcula en la línea de PINNACLE, que es la que define
    # el fair. Si el mejor par está en otra línea son mercados distintos y no
    # se pueden comparar — se descarta.
    if (g["total_point_pin"] is not None and g["total_over_pin"] and g["total_under_pin"]
            and g["total_over_best"] and g["total_point_best"] == g["total_point_pin"]):
        linea = float(g["total_point_pin"])
        real_total = g["actual_home_runs"] + g["actual_away_runs"]
        if real_total != linea:  # push: ni gana ni pierde, se excluye
            # Con línea entera (9.0) el empate es PUSH, no under: se saca del
            # denominador en vez de contarlo como acierto del under. Mismo
            # cuidado que con los empates del moneyline.
            no_push = int(np.count_nonzero(total != linea))
            if no_push == 0:
                return out
            p_over = float(np.count_nonzero(total > linea) / no_push)
            fair_o, fair_u = _devig(g["total_over_pin"], g["total_under_pin"])
            out += [
                {"mercado": "TOTAL", "lado": "OVER", "p": p_over, "fair": fair_o,
                 "cuota": g["total_over_best"], "gano": real_total > linea},
                {"mercado": "TOTAL", "lado": "UNDER", "p": 1 - p_over, "fair": fair_u,
                 "cuota": g["total_under_best"], "gano": real_total < linea},
            ]

    # ── Runline ────────────────────────────────────────────────────────────
    if (g["rl_home_point_pin"] is not None and g["rl_home_pin"] and g["rl_away_pin"]
            and g["rl_home_best"] and g["rl_home_point_best"] == g["rl_home_point_pin"]):
        punto = float(g["rl_home_point_pin"])   # FIRMADO: −1.5 local favorito
        real_margen = g["actual_home_runs"] - g["actual_away_runs"]
        # El local cubre si su margen supera −punto. Con punto=−1.5 eso es
        # margen > 1.5 (ganar por 2+); con punto=+1.5, margen > −1.5 (perder
        # por menos de 2, o ganar). Es la corrección de PURP-1.
        umbral = -punto
        if real_margen != umbral:
            # Ídem: con punto entero (±1.0, ±2.0 — 25 juegos de la muestra) un
            # margen igual al umbral es push.
            no_push_rl = int(np.count_nonzero(margen != umbral))
            if no_push_rl == 0:
                return out
            p_home_cubre = float(np.count_nonzero(margen > umbral) / no_push_rl)
            fair_h, fair_a = _devig(g["rl_home_pin"], g["rl_away_pin"])
            out += [
                {"mercado": "RUNLINE", "lado": f"HOME {punto:+.1f}", "p": p_home_cubre,
                 "fair": fair_h, "cuota": g["rl_home_best"],
                 "gano": real_margen > umbral},
                {"mercado": "RUNLINE", "lado": f"AWAY {-punto:+.1f}", "p": 1 - p_home_cubre,
                 "fair": fair_a, "cuota": g["rl_away_best"],
                 "gano": real_margen < umbral},
            ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--n-sims", type=int, default=200_000)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if not args.db.exists():
        print(f"✗ no existe {args.db}", file=sys.stderr)
        return 2
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    juegos = cargar(con, args.season)
    con.close()
    if not juegos:
        print("Sin juegos con λ guardadas Y cuotas históricas. ¿Corriste el "
              "backfill de derivados (fetch_historical_odds.py --derivados)?")
        return 1

    print(f"Juegos con λ y cuotas: {len(juegos)}  "
          f"({juegos[0]['game_date']} a {juegos[-1]['game_date']})")
    print(f"Simulaciones por juego: {args.n_sims:,}\n")

    todas: list[dict] = []
    for i, g in enumerate(juegos, 1):
        mc = monte_carlo_advanced(
            lh=float(g["lh"]), la=float(g["la"]), n_max=args.n_sims,
            block=min(50_000, args.n_sims), store_samples=True,
            analyze_f5=False, rng_seed=int(g["game_pk"]) % (2**32),
        )
        todas += apuestas_del_juego(g, np.asarray(mc["home_samples"]),
                                    np.asarray(mc["away_samples"]),
                                    float(mc["p_home"]))
        if i % 200 == 0 or i == len(juegos):
            print(f"  simulados {i}/{len(juegos)}", end="\r")
    print(" " * 40, end="\r")

    # ── ROI por mercado y umbral ───────────────────────────────────────────
    resultado: dict = {}
    print(f"{'mercado':10s} {'umbral':>7s} {'apuestas':>9s} {'aciertos':>9s} "
          f"{'ROI':>8s} {'±SE':>5s} {'t':>6s} {'edge medio':>11s}")
    print("─" * 78)
    for mercado in ("MONEYLINE", "TOTAL", "RUNLINE"):
        delm = [a for a in todas if a["mercado"] == mercado
                and a["cuota"] and a["cuota"] <= CUOTA_MAXIMA]
        resultado[mercado] = {}
        for thr in UMBRALES:
            sel = [a for a in delm if (a["p"] - a["fair"]) >= thr]
            if not sel:
                print(f"{mercado:10s} {thr:6.0%} {0:9d} {'—':>9s} {'—':>8s}")
                resultado[mercado][f"edge>={thr:.0%}"] = {"apuestas": 0}
                continue
            ganadas = sum(1 for a in sel if a["gano"])
            pagos = np.array([(a["cuota"] - 1) if a["gano"] else -1.0 for a in sel])
            profit = float(pagos.sum())
            roi = 100 * profit / len(sel)
            edge = 100 * sum(a["p"] - a["fair"] for a in sel) / len(sel)
            # Error estándar del ROI medio. Una apuesta paga (cuota−1) o −1, así
            # que su varianza es grande y con pocas apuestas el ROI se mueve
            # varios puntos por puro azar. Sin esto, un +6 % sobre 900 apuestas
            # se lee como señal cuando puede ser ruido — que es justo el error
            # que este proyecto ya cometió con el CLV medido contra el crudo.
            se = 100 * float(pagos.std(ddof=1)) / np.sqrt(len(sel)) if len(sel) > 1 else float("nan")
            t = roi / se if se and np.isfinite(se) and se > 0 else float("nan")
            marca = "  *" if abs(t) >= 1.96 else ""
            print(f"{mercado:10s} {thr:6.0%} {len(sel):9d} "
                  f"{100*ganadas/len(sel):8.1f}% {roi:7.2f}% ±{se:4.2f} "
                  f"{t:+6.2f} {edge:9.2f}pp{marca}")
            resultado[mercado][f"edge>={thr:.0%}"] = {
                "apuestas": len(sel), "aciertos_pct": round(100*ganadas/len(sel), 2),
                "profit_u": round(profit, 3), "roi_pct": round(roi, 3),
                "roi_se_pp": round(se, 3), "t": round(t, 3),
                "ic95": [round(roi - 1.96*se, 2), round(roi + 1.96*se, 2)],
                "edge_medio_pp": round(edge, 3),
            }
        print()

    # ── Diagnóstico de sesgo direccional ───────────────────────────────────
    # Si el modelo elige casi siempre el mismo lado de un mercado, lo que se
    # está midiendo no es habilidad juego a juego sino un sesgo sistemático que
    # esta muestra premió. Un sesgo puede ser rentable por un tiempo y no
    # sobrevivir a otra temporada, así que se reporta aparte y explícito.
    print("SESGO DIRECCIONAL — ¿elige lados o elige juegos?")
    print(f"  {'mercado':10s} {'umbral':>7s} {'lado dominante':>28s} {'reparto':>9s}")
    for mercado in ("MONEYLINE", "TOTAL", "RUNLINE"):
        delm = [a for a in todas if a["mercado"] == mercado
                and a["cuota"] and a["cuota"] <= CUOTA_MAXIMA]
        for thr in (0.00, 0.10):
            sel = [a for a in delm if (a["p"] - a["fair"]) >= thr]
            if not sel:
                continue
            # Para runline se agrupa por si el lado elegido es el que PONE o el
            # que RECIBE las carreras, no por local/visitante.
            def familia(a):
                if a["mercado"] != "RUNLINE":
                    return a["lado"]
                return "pone (favorito)" if "-" in a["lado"] else "recibe (no-favorito)"
            cuenta = defaultdict(int)
            for a in sel:
                cuenta[familia(a)] += 1
            dom, n_dom = max(cuenta.items(), key=lambda kv: kv[1])
            pct = 100 * n_dom / len(sel)
            aviso = "  ← sesgo fuerte" if pct >= 70 else ""
            print(f"  {mercado:10s} {thr:6.0%} {dom:>28s} {pct:8.1f}%{aviso}")
            resultado[mercado].setdefault("sesgo", {})[f"edge>={thr:.0%}"] = {
                "lado_dominante": dom, "pct": round(pct, 1),
                "reparto": dict(cuenta),
            }
    print()

    n_celdas = 3 * len(UMBRALES)
    print("CÓMO LEER ESTO")
    print(f"  ⚠ COMPARACIONES MÚLTIPLES: se prueban {n_celdas} celdas a la vez. Con "
          f"{n_celdas} pruebas")
    print(f"      independientes, ver al menos una con |t|>=1.96 por puro azar tiene "
          f"probabilidad ~{100*(1-0.95**n_celdas):.0f} %.")
    print("      Un solo asterisco NO es un hallazgo. Lo que sí lo sería: un patrón")
    print("      monótono —el ROI sube al subir el umbral— sostenido en un mercado y")
    print("      reproducido en otra temporada. Un asterisco aislado es ruido.")
    print("  * = |t| >= 1.96. Sin asterisco, el ROI no se distingue de cero por más")
    print("      llamativo que sea el número: una apuesta paga (cuota−1) o −1, y esa")
    print("      varianza mueve el ROI varios puntos con unos cientos de apuestas.")
    print("  El fair sale del devig de PINNACLE y el pago va al MEJOR par disponible,")
    print("      así que parte de cualquier ROI positivo es line shopping y no")
    print("      habilidad del modelo. El overround del mejor par es ~1-2pp menor.")
    print("  Las λ vienen de `game_outcomes.backtest_lambda_*`, o sea de corridas de")
    print("      backtest anteriores. Si esas corridas están contaminadas, esto mide")
    print("      su contaminación con precisión. Re-correr el backtest limpio ANTES")
    print("      de tomar cualquier decisión con estos números.")

    if args.json:
        args.json.write_text(json.dumps(
            {"n_juegos": len(juegos), "n_sims": args.n_sims,
             "season": args.season, "resultado": resultado}, indent=1))
        print(f"\n→ {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
