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


def apuestas_del_juego(g: dict, home: np.ndarray, away: np.ndarray) -> list[dict]:
    """Todas las apuestas evaluables de un juego, con su edge y su resultado."""
    n = len(home)
    margen = home - away
    total = home + away
    out: list[dict] = []

    # ── Moneyline ──────────────────────────────────────────────────────────
    if all(g[k] for k in ("ml_home_pin", "ml_away_pin", "ml_home_best", "ml_away_best")):
        p_mod = float(np.count_nonzero(margen > 0) / n)
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
            p_over = float(np.count_nonzero(total > linea) / n)
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
            p_home_cubre = float(np.count_nonzero(margen > umbral) / n)
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
                                    np.asarray(mc["away_samples"]))
        if i % 200 == 0 or i == len(juegos):
            print(f"  simulados {i}/{len(juegos)}", end="\r")
    print(" " * 40, end="\r")

    # ── ROI por mercado y umbral ───────────────────────────────────────────
    resultado: dict = {}
    print(f"{'mercado':10s} {'umbral':>7s} {'apuestas':>9s} {'aciertos':>9s} "
          f"{'ROI':>9s} {'edge medio':>11s}")
    print("─" * 62)
    for mercado in ("MONEYLINE", "TOTAL", "RUNLINE"):
        delm = [a for a in todas if a["mercado"] == mercado
                and a["cuota"] and a["cuota"] <= CUOTA_MAXIMA]
        resultado[mercado] = {}
        for thr in UMBRALES:
            sel = [a for a in delm if (a["p"] - a["fair"]) >= thr]
            if not sel:
                print(f"{mercado:10s} {thr:6.0%} {0:9d} {'—':>9s} {'—':>9s} {'—':>11s}")
                resultado[mercado][f"edge>={thr:.0%}"] = {"apuestas": 0}
                continue
            ganadas = sum(1 for a in sel if a["gano"])
            profit = sum((a["cuota"] - 1) if a["gano"] else -1.0 for a in sel)
            roi = 100 * profit / len(sel)
            edge = 100 * sum(a["p"] - a["fair"] for a in sel) / len(sel)
            print(f"{mercado:10s} {thr:6.0%} {len(sel):9d} "
                  f"{100*ganadas/len(sel):8.1f}% {roi:8.2f}% {edge:10.2f}pp")
            resultado[mercado][f"edge>={thr:.0%}"] = {
                "apuestas": len(sel), "aciertos_pct": round(100*ganadas/len(sel), 2),
                "profit_u": round(profit, 3), "roi_pct": round(roi, 3),
                "edge_medio_pp": round(edge, 3),
            }
        print()

    print("Lectura honesta: el ROI de un umbral alto se mide sobre pocas apuestas.")
    print("Un ROI llamativo con n<200 es ruido, no una señal — mirar la columna.")
    print("\nY el fair sale de Pinnacle mientras la apuesta se paga al mejor precio:")
    print("parte del ROI positivo es line shopping, no habilidad del modelo.")

    if args.json:
        args.json.write_text(json.dumps(
            {"n_juegos": len(juegos), "n_sims": args.n_sims,
             "season": args.season, "resultado": resultado}, indent=1))
        print(f"\n→ {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
