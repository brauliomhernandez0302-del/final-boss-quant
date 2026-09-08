"""fbq/model/v15.py — evaluación experimental de v1.5 = v1.2 + carga de bullpen.

    python3 -m fbq.model.v15 --salida docs/v15_bullpen_2026-09-07

Ejecuta `docs/PREREGISTRO_V1_5_BULLPEN_2026-09-07.md`, commiteado antes de medir.

⚠️ **EXPLORATORIA.** 2025 y 2026 ya fueron explorados por este proyecto —siete
baselines, dieciséis informes de auditoría, nueve motores medidos sobre esos
mismos años—. El diseño temporal impide que el MODELO vea el futuro; no impide
que lo haya visto quien eligió la variable. Ninguna cifra de acá acredita nada.

## La intersección, que es lo que hace comparable la comparación

La carga es la única variable que puede faltar. Una fila sin ella se cae de
v1.5, **y también de v1.2 y del mercado**: las tres columnas se miden sobre
exactamente las mismas filas o no se comparan. v1.2 se RE-AJUSTA sobre esa
intersección — comparar el v1.2 publicado (ajustado sobre un conjunto mayor)
contra un v1.5 ajustado sobre la intersección compararía dos muestras, que es el
error que tiró siete baselines de este proyecto.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from fbq.evaluator.score import brier, dif_pareada_ic
from fbq.model import features as F
from fbq.model.candidato import construir, _matriz
from fbq.model.carga import ESCALA_CARGA, VENTANA_HORAS, cargar_indice
from fbq.model.detector import verificar_plausibilidad
from fbq.model.logistica import LAMBDA_L2, ajustar
from fbq.model.pit import cargar_partidos

log = logging.getLogger(__name__)

CONFIG = {
    "preregistro": "docs/PREREGISTRO_V1_5_BULLPEN_2026-09-07.md",
    "v1_2": list(F.NOMBRES),
    "v1_5": list(F.NOMBRES_V15),
    "ventana_horas": VENTANA_HORAS,
    "escala_carga": ESCALA_CARGA,
    "lambda_l2": LAMBDA_L2,
    "pliegues": "expansivos: entrena con temporadas < T, evalúa T",
    "evaluacion": "EXPLORATORIA — 2025 y 2026 ya fueron explorados",
}


def filas_con_carga(seasons: Sequence[int] = (2024, 2025, 2026),
                    *, con_concentracion: bool = False):
    """Todas las filas predecibles, con la carga ya adjunta en `extra`."""
    partidos = cargar_partidos(seasons)
    indice = cargar_indice(partidos)
    conc = None
    if con_concentracion:
        from fbq.model.carga import cargar_concentracion
        conc = cargar_concentracion()
    filas, excl = construir(seasons, indice_carga=indice, indice_concentracion=conc)
    return filas, excl, indice


def evaluar(filas, temporadas=(2025, 2026), *, n_bootstrap: int = 2000):
    """v1.2, v1.5 y el mercado sobre la MISMA intersección, pliegue a pliegue."""
    aptas = [f for f in filas if f.extra.get("carga_ok") == 1]
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
        for nombre, cols in (("v1.2", F.NOMBRES), ("v1.5", F.NOMBRES_V15)):
            Xtr, _ = _matriz(tr, cols)
            Xte, _ = _matriz(te, cols)
            # El peso sale SÓLO del entrenamiento. `ajustar` no ve `te`.
            m = ajustar(Xtr, ytr, tuple(cols))
            modelos[nombre] = m
            p[nombre] = m.predecir(Xte)
            verificar_plausibilidad(p[nombre], yte, nombre=f"{nombre} {temporada}")

        p_mer = np.array([f.p_mercado for f in te], float)
        b12 = (p["v1.2"] - yte) ** 2
        b15 = (p["v1.5"] - yte) ** 2
        bmer = (p_mer - yte) ** 2
        local = np.array([f.home_team for f in te])

        salida.append({
            "temporada": temporada,
            "n_entrenamiento": len(tr), "n_evaluacion": len(te),
            "temporadas_entrenamiento": sorted({f.season for f in tr}),
            "brier_v1_2": brier(p["v1.2"], yte),
            "brier_v1_5": brier(p["v1.5"], yte),
            "brier_mercado": brier(p_mer, yte),
            # `ajustar` estandariza con la media y el desvío del pliegue de
            # ENTRENAMIENTO, así que los coeficientes son por desvío típico y
            # son comparables entre sí. La estandarización viaja dentro del
            # modelo: no hay forma de aplicarlo con otra.
            "coeficientes_v1_5": dict(zip(F.NOMBRES_V15,
                                          [float(c) for c in modelos["v1.5"].beta[1:]])),
            "intercepto_v1_5": float(modelos["v1.5"].beta[0]),
            "coeficientes_v1_2": dict(zip(F.NOMBRES,
                                          [float(c) for c in modelos["v1.2"].beta[1:]])),
            "desvio_entrenamiento_v1_5": dict(zip(F.NOMBRES_V15,
                                                  [float(c) for c in modelos["v1.5"].sd])),
            "pareado_v1_5_menos_v1_2": dif_pareada_ic(b15 - b12, local,
                                                      n_bootstrap=n_bootstrap),
            "pareado_v1_5_menos_mercado": dif_pareada_ic(b15 - bmer, local,
                                                         n_bootstrap=n_bootstrap),
            "pareado_v1_2_menos_mercado": dif_pareada_ic(b12 - bmer, local,
                                                         n_bootstrap=n_bootstrap),
            "_detalle": [
                {"game_pk": f.game_pk, "official_date": f.official_date,
                 "season": f.season, "home_team": f.home_team,
                 "away_team": f.away_team, "corte": f.corte, "y": f.y,
                 "dif_carga_relevo": f.x[F.TODAS.index("dif_carga_relevo")],
                 "pitches_relevo_local": f.extra.get("pitches_relevo_local"),
                 "pitches_relevo_visita": f.extra.get("pitches_relevo_visita"),
                 "juegos_ventana_local": f.extra.get("juegos_ventana_local"),
                 "juegos_ventana_visita": f.extra.get("juegos_ventana_visita"),
                 "p_v12": float(p["v1.2"][i]), "p_v15": float(p["v1.5"][i]),
                 "p_mercado": float(p_mer[i]),
                 "brier_v12": float(b12[i]), "brier_v15": float(b15[i]),
                 "brier_mercado": float(bmer[i]),
                 "delta_brier_v15_v12": float(b15[i] - b12[i])}
                for i, f in enumerate(te)],
        })
    return salida


def cobertura(filas, indice) -> Dict[str, Any]:
    from collections import Counter
    motivos = Counter(f.extra.get("carga_motivo", "") for f in filas
                      if f.extra.get("carga_ok") != 1)
    aptas = [f for f in filas if f.extra.get("carga_ok") == 1]
    juegos = [int(f.extra.get("juegos_ventana_local") or 0) for f in aptas]
    vis = [int(f.extra.get("juegos_ventana_visita") or 0) for f in aptas]
    dif = [f.x[F.TODAS.index("dif_carga_relevo")] * ESCALA_CARGA for f in aptas]
    return {
        "filas_predecibles": len(filas),
        "con_carga_computable": len(aptas),
        "sin_carga": len(filas) - len(aptas),
        "motivos_sin_carga": dict(motivos),
        "por_temporada": {
            s: {"filas": sum(1 for f in filas if f.season == s),
                "con_carga": sum(1 for f in aptas if f.season == s)}
            for s in sorted({f.season for f in filas})},
        "juegos_en_ventana_local": {
            "media": float(np.mean(juegos)) if juegos else None,
            "distribucion": {int(k): int(v) for k, v in
                             zip(*np.unique(juegos, return_counts=True))}},
        "juegos_en_ventana_visita": {
            "media": float(np.mean(vis)) if vis else None},
        "dif_lanzamientos_sin_escalar": {
            "media": float(np.mean(dif)) if dif else None,
            "desvio": float(np.std(dif, ddof=1)) if len(dif) > 1 else None,
            "min": float(np.min(dif)) if dif else None,
            "max": float(np.max(dif)) if dif else None},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--salida", default="")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    filas, excl, indice = filas_con_carga(a.seasons)
    cob = cobertura(filas, indice)
    log.info("cobertura: %s", json.dumps(cob, ensure_ascii=False))
    res = evaluar(filas, n_bootstrap=a.bootstrap)

    for r in res:
        pv = r["pareado_v1_5_menos_v1_2"]
        print(f"\n  ══ {r['temporada']} · n={r['n_evaluacion']} "
              f"(entrena {r['temporadas_entrenamiento']}, {r['n_entrenamiento']}) ══")
        print(f"  Brier  v1.2 {r['brier_v1_2']:.6f} · v1.5 {r['brier_v1_5']:.6f} "
              f"· mercado {r['brier_mercado']:.6f}")
        print(f"  pareado v1.5−v1.2  {pv['media']:+.6f}  "
              f"IC95 [{pv['ic95'][0]:+.6f}, {pv['ic95'][1]:+.6f}]  "
              f"(deff {pv['efecto_diseno']:.3f}, {pv['clusters']} clústeres)")
        print(f"  coef v1.5: " + " · ".join(
            f"{k} {v:+.5f}" for k, v in r["coeficientes_v1_5"].items()))

    if a.salida:
        base = Path(a.salida)
        base.parent.mkdir(parents=True, exist_ok=True)
        detalle = [d for r in res for d in r["_detalle"]]
        if detalle:
            with (base.with_suffix(".csv")).open("w", newline="", encoding="utf-8") as fh:
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
