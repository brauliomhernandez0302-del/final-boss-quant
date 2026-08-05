"""fbq/sources/statcast.py — Baseball Savant.

Devuelve filas CRUDAS, una por lanzamiento o por jugador según el endpoint. No
agrega, no promedia, no calcula ninguna métrica derivada: eso es trabajo de
`features/`, que es donde vive el contrato de "as of".

La separación importa acá más que en ningún otro sitio. En el sistema anterior
la agregación vivía pegada a la descarga, y dos defectos de conteo idénticos
—`batted_ball_count` inflado 1.905x por contar fouls no terminales— convivieron
en DOS implementaciones independientes durante meses, porque cada una había
copiado la lógica en vez de compartirla. Una capa que sólo descarga no puede
tener ese bug.

Savant no tiene clave ni cuota, pero sí un tope duro de filas por consulta y no
avisa cuando lo alcanza: devuelve las primeras N y calla. Por eso `eventos()`
falla ruidoso al acercarse al tope en vez de entregar un día truncado que
parecería completo.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

BASE = "https://baseballsavant.mlb.com"
TIMEOUT = (5, 60)

# Tope de la consulta de Savant. Al alcanzarlo devuelve las primeras filas sin
# ninguna señal, así que hay que tratarlo como error y no como resultado.
TOPE_FILAS = 24_999

_SESION = requests.Session()
_SESION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; fbq/1.0)"})


class DatoTruncado(RuntimeError):
    """La respuesta llegó al tope de filas. Lo que hay dentro es un prefijo
    arbitrario, no una muestra: agregarlo daría una métrica sesgada que se ve
    perfectamente normal."""


class FechaInesperada(RuntimeError):
    """Savant devolvió filas de un día distinto al pedido. Ocurre y hay que
    detectarlo: una instantánea etiquetada con la fecha equivocada rompe el
    contrato de tiempo aguas abajo sin dejar rastro."""


def _csv(texto: str) -> List[Dict[str, str]]:
    # `lstrip("﻿")` no es cosmético: sin quitar el BOM, la PRIMERA columna
    # del CSV queda con un nombre que no coincide con ninguna clave esperada, y
    # el fallo aparece como un campo vacío mucho más tarde.
    return list(csv.DictReader(io.StringIO(texto.lstrip("﻿"))))


def eventos(dia: str, *, tipo_jugador: str = "pitcher") -> List[Dict[str, str]]:
    """Los lanzamientos de un día, crudos, una fila por evento.

    Un día por consulta y no un rango: el rango es donde el tope de filas muerde
    en silencio, y un día de MLB completo entra con margen. Además hace que
    reintentar un día fallido no obligue a re-descargar el resto.
    """
    r = _SESION.get(
        f"{BASE}/statcast_search/csv",
        params={"all": "true", "type": "details", "player_type": tipo_jugador,
                "game_date_gt": dia, "game_date_lt": dia},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    filas = _csv(r.text)

    if len(filas) >= TOPE_FILAS:
        raise DatoTruncado(
            f"{dia}: {len(filas)} filas, en el tope de {TOPE_FILAS}. Savant no "
            f"avisa cuando trunca; hay que partir la consulta antes de usarlas."
        )
    for f in filas:
        actual = f.get("game_date")
        if actual and actual != dia:
            raise FechaInesperada(
                f"se pidió {dia} y llegó una fila de {actual}")
    return filas


def leaderboard_esperadas(temporada: int, *, tipo: str = "batter") -> List[Dict[str, Any]]:
    """xwOBA, xBA y compañía por jugador para una temporada.

    ACUMULADO DE TEMPORADA: no tiene corte por fecha. Sirve para una línea base
    previa, nunca para predecir un juego de esa misma temporada — para eso hace
    falta reconstruir la instantánea desde `eventos()`, que es lo que el
    contrato de tiempo exige.
    """
    return _json(f"{BASE}/leaderboard/expected_statistics",
                 {"type": tipo, "year": temporada, "csv": "false"})


def leaderboard_oaa(temporada: int) -> List[Dict[str, Any]]:
    """Outs Above Average por jugador. Mismo aviso de acumulado que arriba."""
    return _json(f"{BASE}/leaderboard/outs_above_average",
                 {"type": "Fielder", "year": temporada, "csv": "false"})


def _json(url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    r = _SESION.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    datos = r.json()
    # Savant devuelve a veces la lista pelada y a veces envuelta. Normalizar
    # sólo el CONTENEDOR es aceptable acá; el contenido no se toca.
    if isinstance(datos, dict):
        for clave in ("data", "leaderboard", "rows"):
            if isinstance(datos.get(clave), list):
                return datos[clave]
        return []
    return datos if isinstance(datos, list) else []
