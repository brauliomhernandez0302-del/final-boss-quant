"""fbq.sources — la frontera con el exterior.

Los tests de acá no golpean ninguna API: fijan las decisiones de FORMA y de
caché, que son donde esta capa puede fallar en silencio.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbq.sources import mlb_stats, odds_api


# ── La caché del board ───────────────────────────────────────────────────


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    ruta = tmp_path / "board.json"
    monkeypatch.setattr(odds_api, "CACHE", ruta)
    return ruta


def _guardar(cache, deportes, eventos, edad_s=0):
    cache.write_text(json.dumps({
        "guardado_en": time.time() - edad_s,
        "deportes": list(deportes),
        "eventos": eventos,
    }))


def test_la_cache_sirve_lo_que_cubre(cache):
    _guardar(cache, ["baseball_mlb"], [{"sport_key": "baseball_mlb", "id": "a"}])
    assert odds_api._leer_cache(600, ("baseball_mlb",)) is not None


def test_la_cache_no_sirve_lo_que_no_cubre(cache):
    """Una caché escrita por una corrida de MLB no puede servir a una petición
    de nueve deportes: ocho desaparecerían del board sin ninguna señal."""
    _guardar(cache, ["baseball_mlb"], [{"sport_key": "baseball_mlb", "id": "a"}])
    assert odds_api._leer_cache(600, ("baseball_mlb", "soccer_epl")) is None


def test_la_cache_filtra_a_lo_pedido(cache):
    """Cubrir de más es válido; devolver de más, no."""
    _guardar(cache, ["baseball_mlb", "soccer_epl"], [
        {"sport_key": "baseball_mlb", "id": "a"},
        {"sport_key": "soccer_epl", "id": "b"},
    ])
    ev = odds_api._leer_cache(600, ("baseball_mlb",))
    assert [e["id"] for e in ev] == ["a"]


def test_la_cache_caduca(cache):
    _guardar(cache, ["baseball_mlb"], [{"sport_key": "baseball_mlb"}], edad_s=1200)
    assert odds_api._leer_cache(600, ("baseball_mlb",)) is None


def test_sin_clave_levanta_en_vez_de_devolver_vacio(monkeypatch, cache):
    """Un board vacío es indistinguible de 'no hay juegos hoy', y el cron lo
    ocultaría para siempre."""
    monkeypatch.setenv("ODDS_API_KEY", "")
    with pytest.raises(odds_api.SinClave):
        odds_api.board(deportes=("baseball_mlb",))


def test_las_regiones_incluyen_eu_para_pinnacle():
    """Pinnacle no está licenciado en EE.UU. Sin `eu` no llega su par, que es
    contra el que se desvigoriza — ya pasó una temporada entera así."""
    assert "eu" in odds_api.REGIONES


def test_los_mercados_no_incluyen_periodos():
    """El endpoint masivo devuelve 422 para la petición ENTERA si se incluye
    un mercado de período, rompiendo la captura de todos los deportes."""
    assert not any(m.endswith("_h1") for m in odds_api.MERCADOS)


# ── La forma del schedule ────────────────────────────────────────────────


def test_el_linescore_de_un_cascaron_pospuesto_no_trae_marcador():
    """Un juego pospuesto tiene `linescore.teams` vacío. Ésa es la señal de
    que no hay nada que leer, y distinguirla evita escribir un no-final."""
    assert mlb_stats.linescore_final(
        {"linescore": {"teams": {"home": {}, "away": {}}}}) is None
    assert mlb_stats.linescore_final({}) is None


def test_el_linescore_de_un_final_trae_las_carreras():
    juego = {"linescore": {"currentInning": 9,
                           "teams": {"home": {"runs": 5}, "away": {"runs": 6}}}}
    assert mlb_stats.linescore_final(juego) == (5, 6, 9)
