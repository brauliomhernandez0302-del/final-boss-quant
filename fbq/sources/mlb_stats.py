"""fbq/sources/mlb_stats.py — la API de estadísticas de MLB.

Gratis, sin clave, sin cuota. Devuelve lo que el proveedor devuelve; no
interpreta.

La única decisión que toma es de FORMA, no de contenido: `schedule()` aplana
los bloques de fecha a una lista de juegos, conservando `officialDate` de cada
uno. Acepta un día (`fecha`) o un rango (`desde`/`hasta`); el rango existe
porque reconstruir tres temporadas de horas de inicio día por día son ~500
llamadas y en tres. Esa forma anidada es la que hizo que un consumidor leyera `dates[0]` y se
quedara con el cascarón de un juego pospuesto en vez del jugado.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import requests

BASE = "https://statsapi.mlb.com/api/v1"
TIMEOUT = (5, 20)

# Cuántos días pide como máximo cada llamada de rango.
#
# Medido el 2026-09-06: un `startDate`/`endDate` que abarca 2024-03-20 →
# 2026-08-03 devuelve 3.023 juegos y se corta en 2025-03-20 — exactamente un
# año — SIN error, sin aviso y con HTTP 200. Pedirlo en tres tramos anuales
# devuelve 7.886. Es el mismo modo de fallo que Savant: el proveedor entrega
# las primeras N filas y calla, y quien recibe la lista no tiene forma de
# distinguir "no hay más juegos" de "no me diste más juegos".
#
# 300 deja margen bajo el tope observado sin multiplicar las llamadas.
MAX_DIAS_POR_LLAMADA = 300


def _get(ruta: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(f"{BASE}/{ruta}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def schedule(
    *,
    fecha: Optional[str] = None,
    desde: Optional[str] = None,
    hasta: Optional[str] = None,
    game_pk: Optional[int] = None,
    hidratar: str = "linescore",
) -> List[Dict[str, Any]]:
    """Los juegos del schedule, aplanados, con su bloque de fecha adjunto.

    Un `game_pk` pospuesto y rejugado aparece en DOS bloques de fecha, y los dos
    conservan el mismo `gamePk`. Por eso cada juego devuelto lleva
    `_bloque_fecha`: sin él, quien reciba la lista no puede distinguir el
    cascarón del día original del juego que realmente se disputó.

    No filtra por estado. Filtrar es una decisión del consumidor y `results/`
    la toma con su propia lista de estados permitidos.
    """
    if (desde is None) != (hasta is None):
        raise ValueError("un rango necesita `desde` Y `hasta`; medio rango "
                         "devolvería el schedule entero sin avisar")

    base: Dict[str, Any] = {"sportId": 1, "hydrate": hidratar}
    if fecha:
        base["date"] = fecha
    if game_pk:
        base["gamePk"] = game_pk

    if desde and hasta:
        tramos = _tramos(desde, hasta)
    else:
        tramos = [None]

    juegos: List[Dict[str, Any]] = []
    for tramo in tramos:
        params = dict(base)
        if tramo:
            params["startDate"], params["endDate"] = tramo
        datos = _get("schedule", params)
        for bloque in datos.get("dates", []):
            for g in bloque.get("games", []):
                g["_bloque_fecha"] = bloque.get("date")
                juegos.append(g)
    return juegos


def _tramos(desde: str, hasta: str) -> List[tuple[str, str]]:
    """Parte un rango en ventanas que el proveedor sí devuelve enteras.

    Ver la nota de `MAX_DIAS_POR_LLAMADA`: un rango largo se trunca en silencio.
    """
    a, b = date.fromisoformat(desde), date.fromisoformat(hasta)
    if b < a:
        raise ValueError(f"rango invertido: {desde} → {hasta}")
    out: List[tuple[str, str]] = []
    while a <= b:
        fin = min(a + timedelta(days=MAX_DIAS_POR_LLAMADA - 1), b)
        out.append((a.isoformat(), fin.isoformat()))
        a = fin + timedelta(days=1)
    return out


def linescore_final(juego: Dict[str, Any]) -> Optional[tuple[int, int, int | None]]:
    """`(carreras_local, carreras_visita, innings)` de un juego del schedule.

    Devuelve None si el juego no trae marcador — un cascarón pospuesto tiene
    `linescore.teams` vacío, y ésa es la señal de que no hay nada que leer.
    """
    ls = juego.get("linescore") or {}
    equipos = ls.get("teams") or {}
    local = (equipos.get("home") or {}).get("runs")
    visita = (equipos.get("away") or {}).get("runs")
    if local is None or visita is None:
        return None
    return int(local), int(visita), ls.get("currentInning")

# Campos mínimos del feed en vivo para fechar el FIN de un partido. La API
# acepta `fields` y recorta la respuesta: sin esto cada feed pesa megabytes y
# fechar una temporada sería inviable; con esto son ~1,3 KB por juego.
CAMPOS_FIN = ("gameData,datetime,dateTime,resumeDateTime,officialDate,"
              "liveData,plays,currentPlay,about,endTime,"
              "boxscore,info,label,value,status,detailedState")


def fin_de_juego(game_pk: int) -> Dict[str, Any]:
    """Cuándo terminó REALMENTE un partido, según el feed en vivo.

    Devuelve el instante de la ÚLTIMA JUGADA (`currentPlay.about.endTime` en un
    partido terminado), que es una observación y no una estimación. Es el dato
    que permite dejar de aproximar el fin con una cota sobre el inicio.

    Trae además `duracion` tal como la publica el boxscore —el campo `T`, que
    incluye entre paréntesis los minutos de demora cuando los hubo— porque un
    partido de 2:08 con 1:31 de demora ocupa 3:39 de reloj de pared, y lo que
    importa para un corte es el reloj de pared, no el tiempo de juego.

    Para un suspendido, `dateTime` ya viene siendo la REANUDACIÓN y
    `resumeDateTime` lo confirma.
    """
    r = requests.get(f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live",
                     params={"fields": CAMPOS_FIN}, timeout=TIMEOUT)
    r.raise_for_status()
    d = r.json()
    gd = (d.get("gameData") or {}).get("datetime") or {}
    live = d.get("liveData") or {}
    fin = (((live.get("plays") or {}).get("currentPlay") or {})
           .get("about") or {}).get("endTime")
    duracion = next(
        (i.get("value") for i in ((live.get("boxscore") or {}).get("info") or [])
         if i.get("label") == "T"), None)
    return {
        "game_pk": int(game_pk),
        "inicio": gd.get("dateTime"),
        "reanudacion": gd.get("resumeDateTime"),
        "official_date": gd.get("officialDate"),
        "fin": fin,
        "duracion": duracion,
        "estado": ((d.get("gameData") or {}).get("status") or {}).get("detailedState"),
    }
