"""fbq/features/market_shape.py — señales de la FORMA del mercado, no de su nivel.

Son las primeras features del sistema, y se eligieron por una razón medida: en
el sistema anterior las nueve señales correlacionaban 0.28-0.65 con el precio y
sólo 0.05-0.10 con el resultado. Estaban re-derivando lo que el precio ya sabía.
Una feature con chance tiene que ser **ortogonal al precio**, y lo único que
tenemos con historia y es ortogonal por construcción no es el NIVEL del precio
sino su FORMA: cuánto discrepan las casas entre sí, y cuánto se aparta la más
generosa de la sharp.

VEREDICTO (medido el 2026-08-04 sobre 5.429 juegos de 2024-2026): **ninguna
cruza el portón.** Los signos de los coeficientes son estables en las tres
temporadas —cosa que ninguno de los nueve motores del sistema anterior logró—
pero la magnitud no alcanza a cubrir el 2.05% de vig por lado. El detalle está
en el docstring de cada una.

Se dejan registradas, con su veredicto, precisamente porque no funcionaron:
borrarlas garantizaría que alguien las vuelva a intentar dentro de seis meses
sin saber que ya se midieron.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from fbq.features.base import Feature, registrar

# Las tres series de precio viven en el almacén del sistema anterior mientras
# `market/` acumula historia propia: hoy tiene un día y éstas necesitan
# temporadas. Es una dependencia de DATOS, no de código.
DB_HISTORICA = Path(__file__).parent.parent.parent / "data" / "predictions_history.db"


def _series(game_pks: Sequence[int]) -> Dict[int, sqlite3.Row]:
    if not game_pks:
        return {}
    marcas = ",".join("?" * len(game_pks))
    con = sqlite3.connect(f"file:{DB_HISTORICA}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        filas = con.execute(
            f"""SELECT game_pk, ml_home_pin, ml_away_pin, ml_home_cons,
                       ml_away_cons, ml_home_best, ml_away_best, n_bookmakers
                FROM historical_odds WHERE game_pk IN ({marcas})""",
            tuple(int(g) for g in game_pks),
        ).fetchall()
    finally:
        con.close()
    return {r["game_pk"]: r for r in filas}


def _logit_devig(precio_a: float, precio_b: float) -> float:
    p = (1 / precio_a) / ((1 / precio_a) + (1 / precio_b))
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _desacuerdo(game_pks: Sequence[int]) -> Dict[int, float]:
    salida = {}
    for pk, r in _series(game_pks).items():
        if not all([r["ml_home_pin"], r["ml_away_pin"],
                    r["ml_home_cons"], r["ml_away_cons"]]):
            continue
        salida[pk] = (_logit_devig(r["ml_home_cons"], r["ml_away_cons"])
                      - _logit_devig(r["ml_home_pin"], r["ml_away_pin"]))
    return salida


def _prima(game_pks: Sequence[int]) -> Dict[int, float]:
    salida = {}
    for pk, r in _series(game_pks).items():
        if not all([r["ml_home_pin"], r["ml_away_pin"],
                    r["ml_home_best"], r["ml_away_best"]]):
            continue
        salida[pk] = float(np.log(r["ml_home_best"] / r["ml_home_pin"])
                           - np.log(r["ml_away_best"] / r["ml_away_pin"]))
    return salida


def _profundidad(game_pks: Sequence[int]) -> Dict[int, float]:
    return {pk: float(r["n_bookmakers"]) for pk, r in _series(game_pks).items()
            if r["n_bookmakers"] is not None}


DESACUERDO = registrar(Feature(
    nombre="desacuerdo_cons_pin",
    descripcion="logit(consenso desvigorizado) − logit(Pinnacle desvigorizado)",
    calcular=_desacuerdo,
    hipotesis=(
        "Cuando el consenso de ~28 casas se aparta del libro sharp, el que "
        "suele tener razón es el sharp. La diferencia entre los dos es "
        "ortogonal al NIVEL del precio de Pinnacle por construcción, así que "
        "puede aportar algo que ese nivel no contiene."),
    veredicto="NO CRUZA",
    nota=(
        "Coeficiente POSITIVO y estable en las tres temporadas (+0.0387, "
        "+0.0755, +0.0554) — estabilidad de signo que ninguno de los nueve "
        "motores del sistema anterior tuvo. Pero el Brier empeora en 2024 y "
        "2025, y el ROI combinado con `prima` es negativo en los dos años con "
        "muestra grande. La señal existe; su magnitud no cubre el vig."),
))

PRIMA = registrar(Feature(
    nombre="prima_mejor_precio",
    descripcion="ventaja logarítmica del mejor precio sobre el de Pinnacle, por lado",
    calcular=_prima,
    hipotesis=(
        "Una prima grande del mejor precio sobre el sharp suele ser una "
        "cotización rezagada o errónea, no una oportunidad: el lado con el "
        "precio atípico debería ganar MENOS de lo que ese precio implica."),
    veredicto="NO CRUZA",
    nota=(
        "Coeficiente NEGATIVO y estable en las tres temporadas (−0.059, "
        "−0.055, −0.043), o sea que la hipótesis apunta en la dirección "
        "correcta. Misma conclusión que `desacuerdo`: no alcanza para el vig."),
))

PROFUNDIDAD = registrar(Feature(
    nombre="profundidad_mercado",
    descripcion="cuántas casas cotizan el juego",
    calcular=_profundidad,
    hipotesis=(
        "Un juego con pocas casas cotizando es un mercado menos eficiente, "
        "donde el precio debería tener menos información."),
    veredicto="NO CRUZA",
    nota=(
        "El coeficiente CAMBIA DE SIGNO entre temporadas (−0.0662, +0.0267, "
        "+0.0210) y el Brier empeora en las tres. Es la firma de una señal "
        "inexistente, distinta de las otras dos — que al menos son estables."),
))
