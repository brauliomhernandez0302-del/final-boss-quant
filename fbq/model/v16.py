"""fbq/model/v16.py — evaluación experimental de v1.6 = v1.2 + concentración.

    python3 -m fbq.model.v16 --salida docs/v16_concentracion_2026-09-08

Ejecuta `docs/PREREGISTRO_V1_6_CONCENTRACION_2026-09-08.md`, commiteado en
`7508cb6` antes de medir rendimiento.

Reutiliza entero el arnés de v1.5: mismos pliegues, misma ridge congelada, misma
re-estimación de v1.2 sobre la intersección, mismo bootstrap agrupado. Lo único
que cambia es **qué columna** entra al ajuste — que es la única forma de que la
diferencia signifique «esta variable aporta» y no «esta corrida usó otra
muestra».

⚠️ EXPLORATORIA: 2025 y 2026 ya fueron explorados por este proyecto.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from fbq.evaluator.score import brier, dif_pareada_ic
from fbq.model import features as F
from fbq.model.candidato import _matriz
from fbq.model.carga import VENTANA_HORAS
from fbq.model.detector import verificar_plausibilidad
from fbq.model.logistica import LAMBDA_L2, ajustar
from fbq.model.v15 import filas_con_carga

log = logging.getLogger(__name__)

CONFIG = {
    "preregistro": "docs/PREREGISTRO_V1_6_CONCENTRACION_2026-09-08.md",
    "v1_2": list(F.NOMBRES),
    "v1_6": list(F.NOMBRES_V16),
    "ventana_horas": VENTANA_HORAS,
    "estadistico": "HHI de los lanzamientos de relevo entre brazos",
    "lambda_l2": LAMBDA_L2,
    "evaluacion": "EXPLORATORIA — 2025 y 2026 ya fueron explorados",
    "dato": ("carga OBSERVADA por relevista; no demuestra disponibilidad, "
             "lesión ni decisión del entrenador"),
}


def evaluar(filas, temporadas=(2025, 2026), *, n_bootstrap: int = 2000):
    aptas = [f for f in filas if f.extra.get("concentracion_ok") == 1]
    salida = []
    for temporada in temporadas:
        tr = [f for f in aptas if f.season < temporada]
        te = [f for f in aptas if f.season == temporada]
        if not tr or not te:
            continue
        assert max(f.season for f in tr) < temporada, "el entrenamiento mira el futuro"

        ytr = np.array([f.y for f in tr], float)
        yte = np.array([f.y for f in te], float)
        modelos, p = {}, {}
        for nombre, cols in (("v1.2", F.NOMBRES), ("v1.6", F.NOMBRES_V16)):
            Xtr, _ = _matriz(tr, cols)
            Xte, _ = _matriz(te, cols)
            m = ajustar(Xtr, ytr, tuple(cols))     # sólo entrenamiento
            modelos[nombre] = m
            p[nombre] = m.predecir(Xte)
            verificar_plausibilidad(p[nombre], yte, nombre=f"{nombre} {temporada}")

        p_mer = np.array([f.p_mercado for f in te], float)
        b12 = (p["v1.2"] - yte) ** 2
        b16 = (p["v1.6"] - yte) ** 2
        bmer = (p_mer - yte) ** 2
        local = np.array([f.home_team for f in te])

        salida.append({
            "temporada": temporada,
            "n_entrenamiento": len(tr), "n_evaluacion": len(te),
            "temporadas_entrenamiento": sorted({f.season for f in tr}),
            "brier_v1_2": brier(p["v1.2"], yte),
            "brier_v1_6": brier(p["v1.6"], yte),
            "brier_mercado": brier(p_mer, yte),
            "coeficientes_v1_6": dict(zip(F.NOMBRES_V16,
                                          [float(c) for c in modelos["v1.6"].beta[1:]])),
            "coeficientes_v1_2": dict(zip(F.NOMBRES,
                                          [float(c) for c in modelos["v1.2"].beta[1:]])),
            "desvio_entrenamiento": dict(zip(F.NOMBRES_V16,
                                             [float(c) for c in modelos["v1.6"].sd])),
            "pareado_v1_6_menos_v1_2": dif_pareada_ic(b16 - b12, local, n_bootstrap=n_bootstrap),
            "pareado_v1_6_menos_mercado": dif_pareada_ic(b16 - bmer, local, n_bootstrap=n_bootstrap),
            "pareado_v1_2_menos_mercado": dif_pareada_ic(b12 - bmer, local, n_bootstrap=n_bootstrap),
            "_detalle": [
                {"game_pk": f.game_pk, "official_date": f.official_date,
                 "season": f.season, "home_team": f.home_team,
                 "away_team": f.away_team, "corte": f.corte, "y": f.y,
                 "dif_concentracion": f.x[F.TODAS.index("dif_concentracion")],
                 "hhi_local": f.extra.get("hhi_local"),
                 "hhi_visita": f.extra.get("hhi_visita"),
                 "brazos_local": f.extra.get("brazos_local"),
                 "brazos_visita": f.extra.get("brazos_visita"),
                 "dif_carga_relevo": f.x[F.TODAS.index("dif_carga_relevo")],
                 "p_v12": float(p["v1.2"][i]), "p_v16": float(p["v1.6"][i]),
                 "p_mercado": float(p_mer[i]),
                 "brier_v12": float(b12[i]), "brier_v16": float(b16[i]),
                 "brier_mercado": float(bmer[i]),
                 "delta_brier_v16_v12": float(b16[i] - b12[i])}
                for i, f in enumerate(te)],
        })
    return salida


def cobertura(filas) -> Dict[str, Any]:
    from collections import Counter
    aptas = [f for f in filas if f.extra.get("concentracion_ok") == 1]
    hhi = np.array([f.extra["hhi_local"] for f in aptas], float)
    br = np.array([f.extra["brazos_local"] for f in aptas], float)
    return {
        "filas_predecibles": len(filas),
        "con_concentracion": len(aptas),
        "sin_concentracion": len(filas) - len(aptas),
        "motivos": dict(Counter(f.extra.get("concentracion_motivo") for f in filas
                                if f.extra.get("concentracion_ok") != 1)),
        "por_temporada": {s: {"filas": sum(1 for f in filas if f.season == s),
                              "con_concentracion": sum(1 for f in aptas if f.season == s)}
                          for s in sorted({f.season for f in filas})},
        "hhi_local": {"media": float(hhi.mean()), "mediana": float(np.median(hhi)),
                      "min": float(hhi.min()), "max": float(hhi.max())},
        "brazos_local": {"media": float(br.mean()), "min": int(br.min()),
                         "max": int(br.max())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--salida", default="")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    filas, excl, _ = filas_con_carga(a.seasons, con_concentracion=True)
    cob = cobertura(filas)
    log.info("cobertura: %s", json.dumps(cob, ensure_ascii=False))
    res = evaluar(filas, n_bootstrap=a.bootstrap)

    for r in res:
        pv = r["pareado_v1_6_menos_v1_2"]
        print(f"\n  ══ {r['temporada']} · n={r['n_evaluacion']} "
              f"(entrena {r['temporadas_entrenamiento']}, {r['n_entrenamiento']}) ══")
        print(f"  Brier  v1.2 {r['brier_v1_2']:.6f} · v1.6 {r['brier_v1_6']:.6f} "
              f"· mercado {r['brier_mercado']:.6f}")
        print(f"  pareado v1.6−v1.2  {pv['media']:+.6f}  "
              f"IC95 [{pv['ic95'][0]:+.6f}, {pv['ic95'][1]:+.6f}]  "
              f"(deff {pv['efecto_diseno']:.3f}, {pv['clusters']} clústeres)")
        print("  coef v1.6: " + " · ".join(
            f"{k} {v:+.5f}" for k, v in r["coeficientes_v1_6"].items()))

    if a.salida:
        base = Path(a.salida)
        base.parent.mkdir(parents=True, exist_ok=True)
        detalle = [d for r in res for d in r["_detalle"]]
        if detalle:
            with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(detalle[0]))
                w.writeheader(); w.writerows(detalle)
        base.with_suffix(".json").write_text(json.dumps({
            "config": CONFIG, "cobertura": cob,
            "exclusiones_de_fila": dict(excl),
            "pliegues": [{k: v for k, v in r.items() if k != "_detalle"} for r in res],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  escrito: {base.with_suffix('.csv')} y {base.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
