"""fbq/sources/mlb_stats.py — la API de estadísticas de MLB.

Gratis, sin clave, sin cuota. Devuelve lo que el proveedor devuelve; no
interpreta.

La única decisión que toma es de FORMA, no de contenido: `schedule()` aplana
los bloques de fecha a una lista de juegos, conservando `officialDate` de cada
uno. Esa forma anidada es la que hizo que un consumidor leyera `dates[0]` y se
quedara con el cascarón de un juego pospuesto en vez del jugado.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

BASE = "https://statsapi.mlb.com/api/v1"
TIMEOUT = (5, 20)


def _get(ruta: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(f"{BASE}/{ruta}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def schedule(
    *,
    fecha: Optional[str] = None,
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
    params: Dict[str, Any] = {"sportId": 1, "hydrate": hidratar}
    if fecha:
        params["date"] = fecha
    if game_pk:
        params["gamePk"] = game_pk

    datos = _get("schedule", params)
    juegos: List[Dict[str, Any]] = []
    for bloque in datos.get("dates", []):
        for g in bloque.get("games", []):
            g["_bloque_fecha"] = bloque.get("date")
            juegos.append(g)
    return juegos


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
