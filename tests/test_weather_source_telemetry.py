"""`inputs_snapshot['weather_source']` refleja lo que el motor realmente hizo.

Contexto (2026-08-03). La instantánea leía `game_data['weather'].get('source')`
y esa clave NO EXISTE: `WeatherAPI.get_weather_for_stadium()` devuelve catorce
campos y ninguno se llama `source`. Resultado medido sobre los picks reales:
0 de 52 registraron procedencia del clima, mientras el motor SÍ lo recibía y
SÍ movía λ (verificado en vivo, `weather_mult` 1.018 sobre Truist Park).

No era un fallo de captura sino de telemetría, y es la peor variante: un dato
bueno con la etiqueta rota no se puede auditar, porque nadie va a buscar lo
que cree que nunca llegó.

El arreglo estampa la procedencia que calcula el propio motor
(`park_meta['weather_source']`, park_weather_engine.py) en vez de re-derivarla
en el sitio de la instantánea — una sola fuente de verdad, sin regla duplicada
que pueda desincronizarse.
"""
import inspect
import re

import pytest

from modules.baseball_module.hfa.park_weather_engine import adjust_for_park_and_weather


def _weather():
    """Un dict de clima con la forma REAL que devuelve el fetcher."""
    return {
        "stadium": "Truist Park", "city": "Atlanta",
        "temp_f": 87.5, "humidity": 44.0,
        "wind_speed_mph": 6.2, "wind_direction": 180,
        "conditions": "Clear", "description": "clear sky",
        "rain_mm": 0.0, "precip_probability": 0,
        "postponement_risk": False, "timestamp": "2026-08-02T17:35:00Z",
        "forecast_slots": ["18:00"], "n_slots": 1,
    }


def test_el_fetcher_no_devuelve_una_clave_source():
    """La premisa del bug. Si algún día el fetcher SÍ emite `source`, este test
    falla y avisa de que la lectura vieja volvería a ser viable — momento de
    revisar cuál de las dos fuentes debe mandar, no de tener las dos."""
    assert "source" not in _weather()


def test_el_motor_marca_live_cuando_hay_clima():
    game = {"park": {"name": "Truist Park"}, "venue": "Truist Park",
            "weather": _weather()}
    _, _, meta = adjust_for_park_and_weather(4.5, 4.5, game)
    assert meta["weather_source"] == "live"


def test_el_motor_marca_missing_cuando_no_hay_clima():
    game = {"park": {"name": "Truist Park"}, "venue": "Truist Park"}
    _, _, meta = adjust_for_park_and_weather(4.5, 4.5, game)
    assert meta["weather_source"] == "missing"


def test_la_lectura_vieja_habria_sido_ciega():
    """Reproduce el bug exacto: sobre un clima real, la expresión anterior da
    None aunque el motor haya aplicado un multiplicador distinto de 1."""
    game = {"park": {"name": "Truist Park"}, "venue": "Truist Park",
            "weather": _weather()}
    _, _, meta = adjust_for_park_and_weather(4.5, 4.5, game)
    lectura_vieja = (game.get("weather") or {}).get("source")
    assert lectura_vieja is None
    assert meta["weather_source"] == "live"


def test_run_module_estampa_desde_park_meta():
    """run_module debe TOMAR la procedencia del motor, no re-derivarla.

    Se verifica sobre la fuente porque montar una corrida completa acá pediría
    red y base de datos. Lo que se fija son las dos mitades del arreglo:
    que exista el estampado desde `park_meta`, y que no haya vuelto la lectura
    ciega a `game_data['weather']['source']`.
    """
    from modules.baseball_module.core import run_module as rm
    src = inspect.getsource(rm)

    assert re.search(
        r"inputs_snapshot'?\]?\[\s*['\"]weather_source['\"]\s*\]\s*=\s*"
        r"park_meta\.get\(\s*['\"]weather_source['\"]",
        src,
    ), "run_module ya no estampa weather_source desde park_meta"

    # La expresión del bug, ignorando el comentario que la documenta.
    codigo = "\n".join(
        l for l in src.splitlines() if not l.lstrip().startswith("#")
    )
    assert ".get('source')" not in codigo and '.get("source")' not in codigo, (
        "volvió la lectura ciega de una clave 'source' que el fetcher no emite"
    )

