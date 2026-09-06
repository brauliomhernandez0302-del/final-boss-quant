"""Entrena v1.2 y v1.4 una sola vez y CONGELA los dos ajustes.

    python3 -m fbq.model.congelar

Un modelo que se re-entrena cada noche no produce una serie prospectiva:
produce una sucesión de modelos distintos evaluados una vez cada uno. Acá los
coeficientes se calculan una vez, se escriben con su fecha, su muestra y una
huella, y de ahí los lee `fbq.model.prospectiva`.

## Con qué se entrena cada versión

**Las dos, con las temporadas 2024 y 2025 completas**, sobre los juegos del
marco evaluable (con par de Pinnacle pre-juego y resultado). v1.4 se restringe
además a los juegos donde los dos abridores tienen instantánea PIT con
suficiente muestra.

**El corte de cada fila de entrenamiento sigue siendo el `captured_at` de su
precio**, igual que en toda la línea v1.x. Las estadísticas del abridor salen de
la instantánea PIT anterior a ese día.

## La concesión declarada del entrenamiento de v1.4

Para entrenar hace falta saber quién abrió en 2024-2025, y **el anuncio de
entonces no existe**: la API rellena ese campo con quien efectivamente abrió
(medido: 0 discrepancias en 800 equipo-juego, contra 7,1% en un registro
prospectivo real). Así que el entrenamiento usa el **abridor REAL del boxscore**.

Es una asimetría entre entrenar y aplicar, y se declara en vez de esconderse:

- **no es una fuga en la evaluación**, porque la evaluación es prospectiva y
  usa exclusivamente el anuncio previo al corte;
- su magnitud está acotada por la tasa de cambio de anuncio, **7,1 %** medida
  sobre 84 equipo-juego;
- el sesgo que introduce va en contra del modelo, no a favor: entrena con
  identidades ligeramente mejores que las que tendrá al aplicar.

Alternativa descartada: entrenar sólo con juegos donde anuncio y real
coincidieran. No se puede — para 2024-2025 no existe el anuncio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from fbq.model import abridores as AB
from fbq.model import features as F
from fbq.model.candidato import Fila, _matriz, construir
from fbq.model.logistica import LAMBDA_L2, ajustar
from fbq.model.prospectiva import CONGELADO

log = logging.getLogger(__name__)
REALES = Path(__file__).parent.parent.parent / "data" / "abridores_reales.json"

NOMBRES_V14 = F.NOMBRES + ("dif_calidad_abridor",)


def _abridores_reales(ruta: Path = REALES) -> Dict[int, Dict[str, Optional[int]]]:
    if not ruta.exists():
        raise FileNotFoundError(f"falta {ruta}: hace falta el abridor real para entrenar")
    return {int(k): v for k, v in json.loads(ruta.read_text(encoding="utf-8"))["juegos"].items()}


def variable_abridor(filas: List[Fila], reales: Dict[int, Dict[str, Optional[int]]]
                     ) -> Dict[int, float]:
    """`dif_calidad_abridor` por juego, con la instantánea PIT previa al día."""
    salida: Dict[int, float] = {}
    motivos: Dict[str, int] = {}
    for f in filas:
        r = reales.get(f.game_pk) or {}
        if not r.get("home") or not r.get("away"):
            motivos["sin_abridor_real"] = motivos.get("sin_abridor_real", 0) + 1
            continue
        dia = f.official_date
        loc = AB.desde_pit(r["home"], dia)
        vis = AB.desde_pit(r["away"], dia)
        val, motivo = AB.diferencia(loc, vis)
        if val is None:
            motivos[motivo] = motivos.get(motivo, 0) + 1
            continue
        salida[f.game_pk] = val
    log.info("variable de abridor: %s juegos · exclusiones %s", len(salida), motivos)
    return salida


def congelar(*, seasons=(2024, 2025, 2026), entrenar_con=(2024, 2025),
             salida: Path = CONGELADO, precios=None) -> Dict[str, Any]:
    filas, _ = construir(seasons, precios=precios)
    tr = [f for f in filas if f.season in entrenar_con]
    reales = _abridores_reales()
    dif = variable_abridor(tr, reales)

    modelos = []
    # v1.2 — sobre TODOS los juegos de entrenamiento
    X, y = _matriz(tr, F.NOMBRES)
    m12 = ajustar(X, y, F.NOMBRES)
    modelos.append(("v1.2", m12, F.NOMBRES, len(tr)))

    # v1.4 — sólo donde la variable de abridor existe
    tr14 = [f for f in tr if f.game_pk in dif]
    X14 = np.column_stack([
        np.array([[f.x[F.TODAS.index(n)] for n in F.NOMBRES] for f in tr14], float),
        np.array([dif[f.game_pk] for f in tr14], float)])
    y14 = np.array([f.y for f in tr14], float)
    m14 = ajustar(X14, y14, NOMBRES_V14)
    modelos.append(("v1.4", m14, NOMBRES_V14, len(tr14)))

    doc = {
        "congelado_en": date.today().isoformat(),
        "entrenado_con": list(entrenar_con),
        "preregistros": ["docs/PREREGISTRO_MODELO_V1_2026-09-06.md",
                         "docs/PREREGISTRO_V1_4_ABRIDORES_2026-09-06.md"],
        "concesion_declarada": (
            "v1.4 se ENTRENA con el abridor real del boxscore porque el anuncio "
            "de 2024-2025 no existe; se APLICA con el anuncio previo al corte. "
            "Magnitud acotada: 7,1% de cambio de anuncio medido sobre 84 "
            "equipo-juego. No es fuga en la evaluación, que es prospectiva."),
        "constantes": {"LAMBDA_L2": LAMBDA_L2, "K_BF": AB.K_BF,
                       "MIN_BF_ABRIDOR": AB.MIN_BF_ABRIDOR,
                       "LIGA_K_MENOS_BB": AB.LIGA_K_MENOS_BB,
                       "VENTANA": F.VENTANA, "K_REGRESION": F.K_REGRESION},
        "modelos": {},
    }
    for version, m, nombres, n in modelos:
        cuerpo = {"version": version, "variables": list(nombres),
                  "beta": [float(b) for b in m.beta],
                  "mu": [float(v) for v in m.mu], "sd": [float(v) for v in m.sd],
                  "n_entrenamiento": n, "lambda_l2": m.lambda_l2,
                  "coeficientes": m.coeficientes}
        cuerpo["sha"] = hashlib.sha256(
            json.dumps(cuerpo, sort_keys=True).encode()).hexdigest()[:16]
        doc["modelos"][version] = cuerpo

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precios-cache", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    precios = pickle.load(open(args.precios_cache, "rb")) if args.precios_cache else None
    doc = congelar(precios=precios)
    for v, m in doc["modelos"].items():
        log.info("%s · n=%s · sha=%s · %s", v, m["n_entrenamiento"], m["sha"],
                 {k: round(x, 4) for k, x in m["coeficientes"].items()})


if __name__ == "__main__":
    main()
