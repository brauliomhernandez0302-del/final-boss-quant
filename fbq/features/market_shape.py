"""fbq/features/market_shape.py — señales de la FORMA del mercado, no de su nivel.

Son las primeras features del sistema, y se eligieron por una razón medida: en
el sistema anterior las nueve señales correlacionaban 0.28-0.65 con el precio y
sólo 0.05-0.10 con el resultado. Estaban re-derivando lo que el precio ya sabía.
Una feature con chance tiene que ser **ortogonal al precio**, y lo único que
tenemos con historia y es ortogonal por construcción no es el NIVEL del precio
sino su FORMA: cuánto discrepan las casas entre sí, y cuánto se aparta la más
generosa de la sharp.

VEREDICTO v1 (2026-08-04, almacén legado, 5.429 juegos): **ninguna cruza el
portón.** Los signos de los coeficientes son estables en las tres temporadas
—cosa que ninguno de los nueve motores del sistema anterior logró— pero la
magnitud no alcanza a cubrir el vig.

VEREDICTO v2 (2026-09-06, almacén propio, 6.106 juegos, sin precios en vivo):
ver `veredictos` de cada feature. La v1 NO se borra: se midió contra un nulo
distinto y sigue siendo cierta sobre esa referencia.

Se dejan registradas, con sus veredictos, precisamente porque no funcionaron:
borrarlas garantizaría que alguien las vuelva a intentar dentro de seis meses
sin saber que ya se midieron.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from fbq.features.base import Feature, Veredicto, registrar

# Las tres series salen del almacén PROPIO desde el 2026-09-06. Antes venían de
# `predictions_history.db`, que era la última dependencia de datos con el
# sistema anterior; la importación del histórico la cortó.
DB_MERCADO = Path(__file__).parent.parent.parent / "data" / "market.db"

# Agregado derivado sobre ~28 libros, no una casa. Es lo que `desacuerdo` mide
# contra el sharp, y lo que `prima` tiene que excluir del "mejor precio":
# nadie puede apostar contra un promedio.
CONSENSO = "_consensus"


def _series(game_pks: Sequence[int], *,
            db_mercado: Optional[Path] = None,
            solo_pregame: bool = True) -> Dict[int, Dict[str, Optional[float]]]:
    """Los precios de cada juego, del almacén propio.

    `solo_pregame=True` por la misma razón que en el marco de evaluación: una
    cotización posterior al primer lanzamiento no es una observación del
    mercado pre-juego, y una feature calculada sobre ella mediría información
    que el modelo no puede tener.
    """
    if not game_pks:
        return {}
    marcas = ",".join("?" * len(game_pks))
    con = sqlite3.connect(f"file:{db_mercado or DB_MERCADO}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        filas = con.execute(
            f"""
            WITH ultima AS (
                SELECT s.event_id, s.book, s.side, s.price_dec,
                       ROW_NUMBER() OVER (PARTITION BY s.event_id, s.side, s.book
                                          ORDER BY s.captured_at DESC, s.id DESC) AS rn
                FROM odds_snapshot s
                WHERE s.market = 'h2h'
                  {"AND s.captured_at < s.commence_time" if solo_pregame else ""}
            )
            SELECT l.game_pk,
                   MAX(CASE WHEN u.book='pinnacle'  AND u.side='home' THEN u.price_dec END) AS pin_home,
                   MAX(CASE WHEN u.book='pinnacle'  AND u.side='away' THEN u.price_dec END) AS pin_away,
                   MAX(CASE WHEN u.book=?           AND u.side='home' THEN u.price_dec END) AS cons_home,
                   MAX(CASE WHEN u.book=?           AND u.side='away' THEN u.price_dec END) AS cons_away,
                   MAX(CASE WHEN u.book<>?          AND u.side='home' THEN u.price_dec END) AS best_home,
                   MAX(CASE WHEN u.book<>?          AND u.side='away' THEN u.price_dec END) AS best_away,
                   COUNT(DISTINCT CASE WHEN u.book<>? THEN u.book END)                      AS n_libros,
                   MAX(l.method)                                                            AS metodo
            FROM ultima u
            JOIN event_link l ON l.event_id = u.event_id
            WHERE u.rn = 1 AND l.game_pk IN ({marcas})
            GROUP BY l.game_pk
            """,
            (CONSENSO, CONSENSO, CONSENSO, CONSENSO, CONSENSO,
             *(int(g) for g in game_pks)),
        ).fetchall()
    finally:
        con.close()
    return {r["game_pk"]: dict(r) for r in filas}


def _logit_devig(precio_a: float, precio_b: float) -> float:
    p = (1 / precio_a) / ((1 / precio_a) + (1 / precio_b))
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _desacuerdo(game_pks: Sequence[int]) -> Dict[int, float]:
    salida = {}
    for pk, r in _series(game_pks).items():
        if not all([r["pin_home"], r["pin_away"], r["cons_home"], r["cons_away"]]):
            continue
        salida[pk] = (_logit_devig(r["cons_home"], r["cons_away"])
                      - _logit_devig(r["pin_home"], r["pin_away"]))
    return salida


def _prima(game_pks: Sequence[int]) -> Dict[int, float]:
    salida = {}
    for pk, r in _series(game_pks).items():
        if not all([r["pin_home"], r["pin_away"], r["best_home"], r["best_away"]]):
            continue
        salida[pk] = float(np.log(r["best_home"] / r["pin_home"])
                           - np.log(r["best_away"] / r["pin_away"]))
    return salida


def _profundidad(game_pks: Sequence[int]) -> Dict[int, float]:
    """Cuántas casas REALES cotizan el juego, contadas en el almacén propio.

    Sólo para juegos que capturamos nosotros. Para los importados del histórico
    el conteo sería un artefacto: de ese período se conservaron tres precios por
    juego (Pinnacle, el consenso y el mejor), no el board, así que "3 libros" no
    dice nada sobre la profundidad del mercado — dice qué guardó el sistema
    anterior. El dato original (`n_bookmakers`, entre 2 y 32) no se puede
    reconstruir desde el almacén propio y no se inventa.

    Consecuencia práctica: hoy la feature cubre sólo la ventana de captura
    propia y el portón va a decir que no hay muestra. Se vuelve medible sola a
    medida que la captura acumule temporadas.
    """
    return {pk: float(r["n_libros"])
            for pk, r in _series(game_pks).items()
            if r["metodo"] != "import_historico" and r["n_libros"]}


DESACUERDO = registrar(Feature(
    nombre="desacuerdo_cons_pin",
    descripcion="logit(consenso desvigorizado) − logit(Pinnacle desvigorizado)",
    calcular=_desacuerdo,
    hipotesis=(
        "Cuando el consenso de ~28 casas se aparta del libro sharp, el que "
        "suele tener razón es el sharp. La diferencia entre los dos es "
        "ortogonal al NIVEL del precio de Pinnacle por construcción, así que "
        "puede aportar algo que ese nivel no contiene."),
    veredictos=[
        Veredicto(
            version="v1", fecha="2026-08-04", almacen="legado", codigo="9b7c33a",
            referencia="n=5.429, Brier del nulo 0,241560",
            resultado="NO CRUZA",
            nota=("Coeficiente POSITIVO y estable en las tres temporadas "
                  "(+0.0387, +0.0755, +0.0554) — estabilidad de signo que "
                  "ninguno de los nueve motores del sistema anterior tuvo. "
                  "Pero el Brier empeora en 2024 y 2025, y el ROI combinado "
                  "con `prima` es negativo en los dos años con muestra grande. "
                  "La señal existe; su magnitud no cubre el vig.")),
        Veredicto(
            version="v2", fecha="2026-09-06", almacen="propio", codigo="HEAD@2026-09-06",
            referencia="n=6.106, Brier del nulo 0,242124",
            resultado="NO CRUZA",
            nota=("El signo YA NO es estable: +0.0298 / −0.0002 / −0.0120 "
                  "contra el +0.0387 / +0.0755 / +0.0554 de la v1. La causa "
                  "está AISLADA y no es el cambio de almacén: los valores de "
                  "la feature son idénticos bit a bit en los 5.300 juegos que "
                  "las dos versiones cubren, y el camino exacto de la v1 "
                  "reproduce sus tres coeficientes al último decimal sobre los "
                  "datos de hoy. Lo único que cambia el signo es EXCLUIR los "
                  "129 juegos cuyo precio era EN VIVO: con el mismo nulo del "
                  "legado y sólo esa exclusión, 2025 pasa de +0.0755 a −0.0164 "
                  "y 2026 de +0.0554 a −0.0120. O sea que la estabilidad de "
                  "signo que la v1 destacaba —'la que ninguno de los nueve "
                  "motores tuvo'— la sostenía un 2,4% de juegos cuyo precio ya "
                  "había visto parte del partido. El Brier empeora en las tres "
                  "temporadas y el ROI es negativo en las dos muestras "
                  "grandes.")),
    ],
))

PRIMA = registrar(Feature(
    nombre="prima_mejor_precio",
    descripcion="ventaja logarítmica del mejor precio sobre el de Pinnacle, por lado",
    calcular=_prima,
    hipotesis=(
        "Una prima grande del mejor precio sobre el sharp suele ser una "
        "cotización rezagada o errónea, no una oportunidad: el lado con el "
        "precio atípico debería ganar MENOS de lo que ese precio implica."),
    veredictos=[
        Veredicto(
            version="v1", fecha="2026-08-04", almacen="legado", codigo="9b7c33a",
            referencia="n=5.429, Brier del nulo 0,241560",
            resultado="NO CRUZA",
            nota=("Coeficiente NEGATIVO y estable en las tres temporadas "
                  "(−0.059, −0.055, −0.043), o sea que la hipótesis apunta en "
                  "la dirección correcta. Misma conclusión que `desacuerdo`: "
                  "no alcanza para el vig.")),
        Veredicto(
            version="v2", fecha="2026-09-06", almacen="propio", codigo="HEAD@2026-09-06",
            referencia="n=6.106, Brier del nulo 0,242124",
            resultado="NO CRUZA",
            nota=("Mismo desenlace que `desacuerdo` y por la misma causa, "
                  "también aislada: el camino exacto de la v1 reproduce "
                  "−0.0594 / −0.0547 / −0.0432 sobre los datos de hoy, y basta "
                  "excluir los 129 juegos con precio EN VIVO —mismo nulo del "
                  "legado, nada más— para que 2025 pase a +0.0304 y 2026 a "
                  "+0.0267. La 'estabilidad de signo' de la v1 vivía en esos "
                  "juegos. El Brier empeora en las tres temporadas; el ROI es "
                  "negativo en todos los umbrales con muestra.")),
    ],
))

PROFUNDIDAD = registrar(Feature(
    nombre="profundidad_mercado",
    descripcion="cuántas casas cotizan el juego",
    calcular=_profundidad,
    hipotesis=(
        "Un juego con pocas casas cotizando es un mercado menos eficiente, "
        "donde el precio debería tener menos información."),
    veredictos=[
        Veredicto(
            version="v1", fecha="2026-08-04", almacen="legado", codigo="9b7c33a",
            referencia="n=5.429, Brier del nulo 0,241560",
            resultado="NO CRUZA",
            nota=("El coeficiente CAMBIA DE SIGNO entre temporadas (−0.0662, "
                  "+0.0267, +0.0210) y el Brier empeora en las tres. Es la "
                  "firma de una señal inexistente, distinta de las otras dos "
                  "— que al menos son estables.")),
        Veredicto(
            version="v2", fecha="2026-09-06", almacen="propio", codigo="HEAD@2026-09-06",
            referencia="n=6.106, Brier del nulo 0,242124",
            resultado="NO MEDIBLE",
            nota=("NO es un veredicto sobre la señal: es la constatación de "
                  "que el dato no existe en el almacén propio. `n_bookmakers` "
                  "(2 a 32) sólo vivía en `historical_odds`; de ese período la "
                  "importación conservó tres precios por juego —Pinnacle, el "
                  "consenso y el mejor—, no el board, así que contar libros "
                  "daría qué guardó el sistema anterior y no la profundidad "
                  "del mercado. Cubre 25 juegos, los de la captura propia. Se "
                  "vuelve medible sola cuando la captura acumule temporadas. "
                  "El veredicto vigente sobre la SEÑAL sigue siendo el v1.")),
    ],
))
