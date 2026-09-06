"""El detalle por juego, y la propiedad que el informe del 2026-09-06 rompió.

Un resumen que no se reconstruye desde su detalle no es un resumen: es una
afirmación. Estos tests fijan la reconstrucción como propiedad verificable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fbq.evaluator.detalle import detalle_por_juego, resumen_desde_detalle
from fbq.market.store import MarketStore
from fbq.results.store import Final, ResultsStore

INICIO = "2024-04-24T22:05:00+00:00"
PRE = "2024-04-24T17:00:00+00:00"
VIVO = "2024-04-24T23:30:00+00:00"


def _almacenes(tmp_path: Path, juegos):
    """`juegos` = [(game_pk, [(captured_at, side, price), ...], home_runs, away_runs)]"""
    mkt = MarketStore(tmp_path / "market.db")
    res = ResultsStore(tmp_path / "results.db")
    finales = []
    for pk, cots, hr, ar in juegos:
        ev = f"EV{pk}"
        with mkt._conn() as c:
            c.executemany(
                """INSERT INTO odds_snapshot
                   (captured_at, book_update, sport_key, event_id, commence_time,
                    home_team, away_team, book, market, side, point, price_dec)
                   VALUES (?,NULL,'baseball_mlb',?,?,'Local','Visita','pinnacle','h2h',?,NULL,?)""",
                [(cap, ev, INICIO, side, precio) for cap, side, precio in cots])
        mkt.link_event(ev, sport_key="baseball_mlb", game_pk=pk,
                       official_date="2024-04-24", commence_time=INICIO,
                       method="test")
        finales.append(Final(game_pk=pk, official_date="2024-04-24", season=2024,
                             home_team="Local", away_team="Visita",
                             home_runs=hr, away_runs=ar, detailed_state="Final"))
    res.registrar(finales)
    return tmp_path / "market.db", tmp_path / "results.db"


SOLO_PRE = [(PRE, "home", 1.90), (PRE, "away", 2.02)]
SOLO_VIVO = [(VIVO, "home", 1.10), (VIVO, "away", 8.00)]
AMBOS = SOLO_PRE + SOLO_VIVO


def _detalle(tmp_path, juegos):
    m, r = _almacenes(tmp_path, juegos)
    return detalle_por_juego((2024,), db_mercado=m, db_resultados=r)


def test_clasifica_cada_juego_por_el_precio_que_realmente_usa(tmp_path):
    d = {f["game_pk"]: f for f in _detalle(tmp_path, [
        (1, SOLO_PRE, 5, 3), (2, SOLO_VIVO, 5, 3), (3, AMBOS, 5, 3)])}
    assert d[1]["clasificacion_temporal"] == "prepartido"
    assert d[2]["clasificacion_temporal"] == "en_vivo"
    # el juego 3 tiene los dos: sin filtrar gana el ÚLTIMO, que es el de en vivo
    assert d[3]["clasificacion_temporal"] == "en_vivo"
    assert d[3]["en_marco_prepartido"] is True
    assert d[3]["precio_cambia_al_filtrar"] is True
    assert d[2]["en_marco_prepartido"] is False


def test_un_juego_con_las_dos_capturas_lleva_los_dos_precios(tmp_path):
    f = _detalle(tmp_path, [(3, AMBOS, 5, 3)])[0]
    assert f["snapshot_ts"] == VIVO and f["snapshot_ts_prepartido"] == PRE
    assert f["p_home"] == pytest.approx(1 / 1.10 / (1 / 1.10 + 1 / 8.00))
    assert f["p_home_prepartido"] == pytest.approx(1 / 1.90 / (1 / 1.90 + 1 / 2.02))
    assert f["brier"] != f["brier_prepartido"]


def test_el_resumen_de_todos_se_reconstruye_desde_su_particion(tmp_path):
    """La propiedad que faltaba. `todos` SÍ se parte exactamente por
    `clasificacion_temporal`, que es la partición verdadera."""
    filas = _detalle(tmp_path, [(1, SOLO_PRE, 5, 3), (2, SOLO_VIVO, 3, 5),
                                (3, AMBOS, 5, 3), (4, SOLO_PRE, 3, 5)])
    r = resumen_desde_detalle(filas)
    a, b, t = r["solo_prepartido_en_todos"], r["solo_en_vivo_en_todos"], r["todos"]
    assert a["n"] + b["n"] == t["n"]
    assert (a["n"] * a["brier"] + b["n"] * b["brier"]) / t["n"] == pytest.approx(
        t["brier"], abs=1e-12)


def test_prepartido_y_en_vivo_NO_son_una_particion_de_todos(tmp_path):
    """El error del informe, fijado como test.

    `marco_prepartido` y `sin_precio_prepartido` NO reconstruyen `todos`,
    porque los juegos con las dos capturas están en el primero con un precio y
    en `todos` con otro. El puente publica esa diferencia en vez de esconderla.
    """
    filas = _detalle(tmp_path, [(1, SOLO_PRE, 5, 3), (2, SOLO_VIVO, 3, 5),
                                (3, AMBOS, 5, 3)])
    r = resumen_desde_detalle(filas)
    assert r["juegos_en_ambos_con_precio_distinto"]["n"] == 1
    puente = r["puente"]
    assert puente["reconstruccion_de_las_dos_cohortes"] != pytest.approx(
        puente["brier_todos_publicado"], abs=1e-9)
    # y la diferencia está explicada, juego por juego, por los que cambian de precio
    cambian = [f for f in filas if f["precio_cambia_al_filtrar"]]
    efecto = sum(f["brier"] - f["brier_prepartido"] for f in cambian) / len(filas)
    assert puente["diferencia"] == pytest.approx(efecto, abs=1e-12)


def test_la_media_es_simple_por_juego_y_no_pondera_temporadas(tmp_path):
    """Sin pesos de ninguna clase: una temporada pesa lo que pesa por su número
    de juegos. Queda fijado para que nadie lo asuma al revés."""
    m, r = _almacenes(tmp_path, [(1, SOLO_PRE, 5, 3), (2, SOLO_PRE, 3, 5)])
    filas = detalle_por_juego((2024,), db_mercado=m, db_resultados=r)
    res = resumen_desde_detalle(filas)
    assert res["media"].startswith("simple por juego")
    assert res["todos"]["brier"] == pytest.approx(
        sum(f["brier"] for f in filas) / len(filas), abs=1e-15)
