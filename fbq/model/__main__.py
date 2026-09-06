"""Entrena y evalúa el candidato v1 de P(gana el local), y exporta el detalle.

    python3 -m fbq.model --salida docs/candidato_v1_2026-09-06

Escribe `<salida>.csv` (una fila por juego evaluado) y `<salida>.json`
(cobertura, exclusiones, métricas, veredicto del evaluador y versiones).

⚠️ La evaluación de 2025 y 2026 es HISTÓRICA, no fuera de muestra: esos años ya
fueron explorados por este proyecto. Ver el preregistro.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from fbq.evaluator.frame import DB_MERCADO, DB_RESULTADOS, load_frame_propio
from fbq.evaluator.score import brier, evaluate, log_loss, roi_con_ic
from fbq.model import features as F
from fbq.model.candidato import CONFIG, construir, evaluar_expansivo

CAMPOS = [
    "game_pk", "official_date", "season", "home_team", "away_team",
    "corte", "inicio_utc", *F.TODAS, "n_previos_local", "n_previos_visita",
    "p_v12", "p_v13", "delta_p", "p_tasa_base", "p_mercado", "resultado_local",
    "brier_v12", "brier_v13", "delta_brier", "brier_tasa_base", "brier_mercado",
]


def _version_codigo() -> Dict[str, Any]:
    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": git("rev-parse", "HEAD"),
            "rama": git("rev-parse", "--abbrev-ref", "HEAD"),
            "arbol_limpio": git("status", "--porcelain") == ""}


def _version_datos() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m = sqlite3.connect(f"file:{DB_MERCADO}?mode=ro", uri=True)
    out["market.db"] = dict(zip(
        ("filas", "primera_captura", "ultima_captura", "eventos_enlazados"),
        m.execute("SELECT COUNT(*), MIN(captured_at), MAX(captured_at), "
                  "(SELECT COUNT(*) FROM event_link WHERE game_pk IS NOT NULL) "
                  "FROM odds_snapshot").fetchone()))
    m.close()
    r = sqlite3.connect(f"file:{DB_RESULTADOS}?mode=ro", uri=True)
    out["results.db"] = dict(zip(
        ("juegos", "primer_dia", "ultimo_dia"),
        r.execute("SELECT COUNT(*), MIN(official_date), MAX(official_date) "
                  "FROM resultado").fetchone()))
    r.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--evaluar", nargs="+", type=int, default=[2025, 2026])
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=400)
    args = ap.parse_args()

    filas, excl = construir(args.seasons)
    # Las dos versiones sobre las MISMAS filas y los MISMOS cortes: lo único
    # que cambia entre ellas es qué columnas entran al ajuste.
    por_version = {
        "v1.2": evaluar_expansivo(filas, args.evaluar, nombres=F.NOMBRES),
        "v1.3": evaluar_expansivo(filas, args.evaluar, nombres=F.NOMBRES_V13),
    }
    candidatos = por_version["v1.3"]
    base = {c.temporada: c for c in por_version["v1.2"]}

    detalle: List[Dict[str, Any]] = []
    resumen: Dict[str, Any] = {
        "config": CONFIG,
        "advertencia": ("Evaluación HISTÓRICA, no fuera de muestra: 2025 y 2026 "
                        "ya fueron explorados por este proyecto. El diseño "
                        "temporal impide que el modelo vea el futuro, no que lo "
                        "haya visto quien eligió las variables."),
        "cobertura": {
            "filas_predecibles": len(filas),
            "por_temporada": {int(s): sum(1 for f in filas if f.season == s)
                              for s in sorted({f.season for f in filas})},
            "exclusiones": dict(excl),
        },
        "temporadas": {},
        "version_codigo": _version_codigo(),
        "version_datos": _version_datos(),
    }

    for c in candidatos:
        b = base[c.temporada]
        assert [f.game_pk for f in b.filas] == [f.game_pk for f in c.filas], \
            "las dos versiones tienen que evaluarse sobre las MISMAS filas"
        for i, f in enumerate(c.filas):
            y = int(c.y[i])
            p12, p13 = float(b.p_v1[i]), float(c.p_v1[i])
            detalle.append({
                "game_pk": f.game_pk, "official_date": f.official_date,
                "season": f.season, "home_team": f.home_team,
                "away_team": f.away_team, "corte": f.corte,
                "inicio_utc": f.inicio_utc,
                **dict(zip(F.TODAS, f.x)),
                "n_previos_local": f.extra.get("n_local"),
                "n_previos_visita": f.extra.get("n_visita"),
                "p_v12": p12, "p_v13": p13, "delta_p": p13 - p12,
                "p_tasa_base": float(c.p_tasa_base[i]),
                "p_mercado": float(c.p_mercado[i]),
                "resultado_local": y,
                "brier_v12": (p12 - y) ** 2, "brier_v13": (p13 - y) ** 2,
                "delta_brier": (p13 - y) ** 2 - (p12 - y) ** 2,
                "brier_tasa_base": float((c.p_tasa_base[i] - c.y[i]) ** 2),
                "brier_mercado": float((c.p_mercado[i] - c.y[i]) ** 2),
            })

        # El nulo se recalcula SOBRE LA INTERSECCIÓN evaluable. El 0,242124067
        # publicado corresponde al marco completo de 6.106 juegos y compararse
        # contra él sería comparar dos muestras distintas.
        marco = load_frame_propio((c.temporada,))
        evaluables = {f.game_pk for f in c.filas}
        m_marco = np.isin(marco.game_pk, list(evaluables))
        sub = marco.subset(m_marco)
        por_juego = {int(f.game_pk): float(p) for f, p in zip(c.filas, c.p_v1)}
        # ALINEAR: `p_v1` va en el orden de las filas del modelo y el marco va
        # ordenado por fecha. Pasarlo como array sin alinear emparejaría la
        # probabilidad de un juego con el precio de otro — PURP-1 otra vez.
        p_alineada = sub.candidate_from_column(por_juego)
        assert np.isfinite(p_alineada).all(), "hay juegos del marco sin predicción"
        rep = evaluate(sub, por_juego, nombre=f"v1 {c.temporada}",
                       n_bootstrap=args.bootstrap)

        resumen["temporadas"][int(c.temporada)] = {
            "entrenamiento": {"temporadas": c.temporadas_entrenamiento,
                              "n": c.modelo.n_entrenamiento,
                              "tasa_base": c.tasa_base_entrenamiento},
            "n_evaluables": len(c.y),
            "n_marco_completo": len(marco),
            "coeficientes": c.modelo.coeficientes,
            "lambda_l2": c.modelo.lambda_l2,
            "metricas": {
                nom: {"brier": brier(p, c.y), "log_loss": log_loss(p, c.y)}
                for nom, p in (("v1.3", c.p_v1), ("v1.2", b.p_v1),
                               ("tasa_base", c.p_tasa_base),
                               ("mercado_interseccion", c.p_mercado))
            },
            "aporte_del_componente": {
                "delta_brier": brier(c.p_v1, c.y) - brier(b.p_v1, c.y),
                "delta_log_loss": log_loss(c.p_v1, c.y) - log_loss(b.p_v1, c.y),
                "coef_b2b_visita": c.modelo.coeficientes["b2b_visita"],
                "juegos_con_prediccion_distinta": int(
                    sum(1 for i in range(len(c.y))
                        if abs(float(c.p_v1[i]) - float(b.p_v1[i])) > 1e-12)),
            },
            "coeficientes_v12": b.modelo.coeficientes,
            "brier_mercado_marco_completo": brier(marco.p_market, marco.y),
            "veredicto_evaluador": {
                "n": rep.n,
                "brier_candidato": rep.brier_candidato,
                "brier_mercado": rep.brier_mercado,
                "brecha": rep.brecha,
                "b_candidato": rep.b_candidato,
                "ic95": list(rep.ic_b_candidato),
                "p_b_positivo": rep.p_b_candidato_positivo,
                "aporta_sobre_el_precio": rep.aporta_sobre_el_precio,
                "roi": roi_con_ic(p_alineada, sub, n_bootstrap=args.bootstrap),
                "mezcla_fuera_de_muestra": rep.mezcla,
            },
        }

    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        csv_path = args.salida.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CAMPOS)
            w.writeheader()
            w.writerows(detalle)
        json_path = args.salida.with_suffix(".json")
        json_path.write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"detalle: {len(detalle)} juegos → {csv_path}")
        print(f"resumen: {json_path}")

    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
