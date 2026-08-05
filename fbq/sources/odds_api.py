"""fbq/sources/odds_api.py — The Odds API.

Devuelve los eventos **CRUDOS**, con su lista `bookmakers` intacta. No
normaliza, no aplana, no elige "el mejor precio": eso lo decide quien consume.

Esa disciplina no es purismo. La versión anterior normalizaba en el fetcher y
guardaba sólo lo normalizado, así que los precios de Pinnacle para total y
runline —que llegaban en cada respuesta desde siempre— se descartaban en el
parseo. Cuando hicieron falta para desvigorizar esos mercados, no existía ni un
día de histórico y hubo que empezar a capturarlos de cero.

La caché es de archivo y compartida entre procesos a propósito: varias corridas
del cron caen dentro de la misma ventana y sirven la misma respuesta sin gastar
cuota. Una llamada histórica cuesta (nº mercados × nº regiones × 10) créditos,
así que la combinación importa mucho más que la frecuencia.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

# `eu` es obligatorio: Pinnacle no está licenciado en EE.UU. y sin él no llega
# su par de precios, que es contra el que se desvigoriza para obtener la línea
# justa. Pedir sólo `us` deja al sistema sin referencia sharp — ya pasó una
# temporada entera así, y eso desactivó en silencio la calibración que depende
# de tener una línea de Pinnacle.
REGIONES = "us,eu"

# El endpoint masivo NO soporta mercados de período (h2h_h1 y compañía):
# devuelve 422 para la petición ENTERA si se incluye alguno, rompiendo la
# captura de todos los deportes a la vez. Los períodos necesitan el endpoint
# por evento.
MERCADOS = ("h2h", "totals", "spreads")

DEPORTES = (
    "baseball_mlb",
    "basketball_nba",
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_usa_mls",
    "mma_mixed_martial_arts",
)

CACHE = Path(__file__).parent.parent.parent / ".cache" / "fbq_odds_board.json"
CACHE_TTL = 600  # segundos

PINNACLE = "pinnacle"


class SinClave(RuntimeError):
    """No hay ODDS_API_KEY. Se levanta en vez de devolver una lista vacía:
    un board vacío es indistinguible de 'no hay juegos hoy' y el cron lo
    ocultaría para siempre."""


def _clave() -> str:
    k = os.getenv("ODDS_API_KEY", "").strip()
    if not k:
        raise SinClave("ODDS_API_KEY no está definida")
    return k


def _leer_cache(ttl: int, deportes: tuple[str, ...]) -> Optional[List[Dict[str, Any]]]:
    """La caché sirve sólo si CUBRE lo pedido.

    Guardar qué deportes contiene no es un detalle: una caché escrita por una
    corrida de MLB serviría, sin esa comprobación, a una petición de los nueve
    deportes, y ocho desaparecerían del board sin ninguna señal.
    """
    if not CACHE.exists():
        return None
    try:
        contenido = json.loads(CACHE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    edad = time.time() - contenido.get("guardado_en", 0)
    if edad > ttl:
        return None
    cubiertos = set(contenido.get("deportes", []))
    if not set(deportes) <= cubiertos:
        log.debug("caché no cubre %s — se pide fresco",
                  sorted(set(deportes) - cubiertos))
        return None
    log.debug("caché de board: %.1f min de antigüedad", edad / 60)
    return [e for e in contenido.get("eventos", [])
            if e.get("sport_key") in set(deportes)]


def _escribir_cache(eventos: List[Dict[str, Any]], deportes: tuple[str, ...]) -> None:
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(
            {"guardado_en": time.time(), "deportes": list(deportes),
             "eventos": eventos},
            ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        log.warning("no se pudo guardar la caché del board: %s", exc)


def board(
    *,
    deportes: tuple[str, ...] = DEPORTES,
    ttl: int = CACHE_TTL,
    forzar: bool = False,
) -> List[Dict[str, Any]]:
    """El board completo, crudo. Levanta si no hay clave.

    Si un deporte falla, el resto se devuelve igual pero **no se cachea**: una
    respuesta parcial guardada como si fuera completa se sirve durante la
    ventana entera y hace desaparecer juegos sin ninguna señal.

    Una lista vacía significa que el proveedor devolvió cero eventos, no que
    algo falló — los fallos levantan o quedan en el log.
    """
    if not forzar:
        cacheado = _leer_cache(ttl, deportes)
        if cacheado is not None:
            return cacheado

    clave = _clave()
    eventos: List[Dict[str, Any]] = []
    completo = True
    for deporte in deportes:
        url = (f"{BASE_URL}/sports/{deporte}/odds/"
               f"?apiKey={clave}&regions={REGIONES}"
               f"&markets={','.join(MERCADOS)}&oddsFormat=decimal")
        try:
            r = requests.get(url, timeout=(5, 20))
            r.raise_for_status()
            datos = r.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            log.warning("%s: fallo al pedir el board — %s", deporte, exc)
            completo = False
            continue
        eventos.extend(datos)
        log.debug("%s: %d eventos", deporte, len(datos))
        time.sleep(0.3)

    if completo and eventos:
        _escribir_cache(eventos, deportes)
    elif not completo:
        log.warning("board parcial: no se cachea, se reintenta en la próxima corrida")
    return eventos


def cuota_restante() -> Optional[int]:
    """Créditos que quedan, según la cabecera del proveedor.

    Es una llamada barata contra `/sports/`, que no consume cuota. Vale
    consultarla antes de cualquier backfill: una descarga histórica cuesta
    (mercados × regiones × 10) por día pedido, y es fácil gastar la mitad del
    mes en una corrida distraída.
    """
    try:
        r = requests.get(f"{BASE_URL}/sports/?apiKey={_clave()}", timeout=20)
        r.raise_for_status()
    except (requests.RequestException, SinClave) as exc:
        log.warning("no se pudo consultar la cuota: %s", exc)
        return None
    valor = r.headers.get("x-requests-remaining")
    return int(valor) if valor is not None else None
