"""El marco de evaluación armado desde los almacenes PROPIOS de fbq.

Hechos de `results.db`, precios de `market.db`. Ninguna lectura del sistema
anterior. Cada test fija una regla que, si se rompe, cambia el nulo contra el
que se mide todo lo demás.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fbq.evaluator.frame import load_frame_propio
from fbq.market.store import MarketStore
from fbq.results.store import Final, ResultsStore

FUTURO = "2099-01-01T00:00:00+00:00"


def _almacenes(tmp_path: Path, cotizaciones, *, inicio="2024-04-24T22:05:00+00:00"):
    """Un par de almacenes propios con un juego y las cotizaciones que se pidan.

    `cotizaciones` son tuplas `(captured_at, book, side, price)`.
    """
    mkt = MarketStore(tmp_path / "market.db")
    with mkt._conn() as c:
        c.executemany(
            """INSERT INTO odds_snapshot
               (captured_at, book_update, sport_key, event_id, commence_time,
                home_team, away_team, book, market, side, point, price_dec)
               VALUES (?,NULL,'baseball_mlb','EV1',?,'Local','Visita',?, 'h2h',?,NULL,?)""",
            [(cap, inicio, book, side, precio) for cap, book, side, precio in cotizaciones],
        )
    mkt.link_event("EV1", sport_key="baseball_mlb", game_pk=1,
                   official_date="2024-04-24", commence_time=inicio,
                   method="test")

    res = ResultsStore(tmp_path / "results.db")
    res.registrar([Final(game_pk=1, official_date="2024-04-24", season=2024,
                         home_team="Local", away_team="Visita",
                         home_runs=5, away_runs=3, detailed_state="Final")])
    return tmp_path / "market.db", tmp_path / "results.db"


def _cargar(tmp_path, cotizaciones, **kw):
    m, r = _almacenes(tmp_path, cotizaciones, **{k: v for k, v in kw.items() if k == "inicio"})
    return load_frame_propio((2024,), db_mercado=m, db_resultados=r,
                             **{k: v for k, v in kw.items() if k != "inicio"})


PRE = "2024-04-24T17:00:00+00:00"
VIVO = "2024-04-24T23:00:00+00:00"


def test_arma_el_marco_desde_los_almacenes_propios(tmp_path):
    f = _cargar(tmp_path, [(PRE, "pinnacle", "home", 1.90),
                           (PRE, "pinnacle", "away", 2.02)])
    assert len(f) == 1
    assert f.game_pk[0] == 1 and f.y[0] == 1.0 and f.season[0] == 2024


def test_el_nulo_es_pinnacle_desvigorizado_multiplicativamente(tmp_path):
    f = _cargar(tmp_path, [(PRE, "pinnacle", "home", 1.90),
                           (PRE, "pinnacle", "away", 2.02)])
    esperado = (1 / 1.90) / (1 / 1.90 + 1 / 2.02)
    assert f.p_market[0] == pytest.approx(esperado, abs=1e-12)
    assert f.overround[0] == pytest.approx(1 / 1.90 + 1 / 2.02 - 1, abs=1e-12)


def test_una_cotizacion_posterior_al_inicio_no_puede_ser_el_nulo(tmp_path):
    """La corrección medida: sobre la foto histórica son 139 juegos cuyo
    "precio de mercado" era en realidad una cotización EN VIVO."""
    cots = [(VIVO, "pinnacle", "home", 1.20), (VIVO, "pinnacle", "away", 4.50)]
    with pytest.raises(ValueError, match="sin juegos utilizables"):
        _cargar(tmp_path, cots)
    # el dato no se perdió: existe, sólo que no se usa como nulo pre-juego
    f = _cargar(tmp_path, cots, solo_pregame=False)
    assert len(f) == 1


def test_entre_pre_juego_y_en_vivo_se_queda_con_la_ultima_PRE_juego(tmp_path):
    f = _cargar(tmp_path, [
        ("2024-04-24T12:00:00+00:00", "pinnacle", "home", 1.80),
        ("2024-04-24T12:00:00+00:00", "pinnacle", "away", 2.15),
        (PRE, "pinnacle", "home", 1.90),     # la última antes del inicio
        (PRE, "pinnacle", "away", 2.02),
        (VIVO, "pinnacle", "home", 1.20),    # en vivo: se ignora
        (VIVO, "pinnacle", "away", 4.50),
    ])
    esperado = (1 / 1.90) / (1 / 1.90 + 1 / 2.02)
    assert f.p_market[0] == pytest.approx(esperado, abs=1e-12)


def test_el_mejor_precio_del_ROI_excluye_el_consenso(tmp_path):
    """Nadie puede apostar contra un promedio de 28 casas."""
    f = _cargar(tmp_path, [
        (PRE, "pinnacle", "home", 1.90), (PRE, "pinnacle", "away", 2.02),
        (PRE, "draftkings", "home", 1.95), (PRE, "fanduel", "away", 2.10),
        (PRE, "_consensus", "home", 9.99), (PRE, "_consensus", "away", 9.99),
    ])
    assert f.best_home[0] == 1.95
    assert f.best_away[0] == 2.10


def test_un_juego_sin_par_de_pinnacle_no_entra(tmp_path):
    """Medio par no sirve: sin los dos lados no hay línea justa, y compararse
    contra un precio crudo regala el vig entero como si fuera habilidad."""
    with pytest.raises(ValueError, match="sin juegos utilizables"):
        _cargar(tmp_path, [(PRE, "pinnacle", "home", 1.90),
                           (PRE, "draftkings", "away", 2.10)])


def test_un_evento_sin_game_pk_enlazado_no_entra(tmp_path):
    m, r = _almacenes(tmp_path, [(PRE, "pinnacle", "home", 1.90),
                                 (PRE, "pinnacle", "away", 2.02)])
    mkt = MarketStore(m)
    with mkt._conn() as c:
        c.execute("UPDATE event_link SET game_pk = NULL WHERE event_id='EV1'")
    with pytest.raises(ValueError, match="sin juegos utilizables"):
        load_frame_propio((2024,), db_mercado=m, db_resultados=r)
