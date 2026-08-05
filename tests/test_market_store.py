"""Invariantes del almacén de precios (`market/store.py`).

Los cuatro primeros tests fijan la regla que este módulo existe para imponer:
un precio observado no se corrige ni se borra. El resto cubre las tres formas
en que este proyecto ya perdió dinero o tiempo con datos de mercado —
el punto separado de su lado (PURP-1), medio par sin desvigorizar, y un corte
temporal que incluye el propio momento que se está prediciendo.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbq.market.store import MarketStore


def _event(event_id="evt1", *, h2h_home=1.80, h2h_away=2.10,
           total_point=8.5, over=1.95, under=1.90,
           rl_home_point=-1.5, rl_home=2.05, rl_away=1.80,
           book="pinnacle"):
    """Un evento crudo con la forma exacta que devuelve The Odds API."""
    return {
        "id": event_id,
        "sport_key": "baseball_mlb",
        "commence_time": "2099-08-04T23:05:00Z",
        "home_team": "Los Angeles Dodgers",
        "away_team": "San Diego Padres",
        "bookmakers": [{
            "key": book,
            "title": book.title(),
            "last_update": "2026-08-04T18:00:00Z",
            "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Los Angeles Dodgers", "price": h2h_home},
                    {"name": "San Diego Padres", "price": h2h_away},
                ]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "price": over, "point": total_point},
                    {"name": "Under", "price": under, "point": total_point},
                ]},
                {"key": "spreads", "outcomes": [
                    {"name": "Los Angeles Dodgers", "price": rl_home,
                     "point": rl_home_point},
                    {"name": "San Diego Padres", "price": rl_away,
                     "point": -rl_home_point},
                ]},
            ],
        }],
    }


@pytest.fixture()
def store(tmp_path):
    return MarketStore(db_path=tmp_path / "market.db")


# ── Append-only ──────────────────────────────────────────────────────────


def test_update_sobre_odds_snapshot_aborta(store):
    """El motor rechaza el UPDATE, no una convención del código.

    Es exactamente lo que `track_record.db.capture_closing_line()` hace doce
    veces por día sobre `picks`, borrando cada captura con la siguiente.
    """
    store.record_board([_event()])
    with pytest.raises(sqlite3.IntegrityError):
        with store._conn() as conn:
            conn.execute("UPDATE odds_snapshot SET price_dec = 9.99")


def test_delete_sobre_odds_snapshot_aborta(store):
    store.record_board([_event()])
    with pytest.raises(sqlite3.IntegrityError):
        with store._conn() as conn:
            conn.execute("DELETE FROM odds_snapshot")


def test_precio_que_se_mueve_conserva_las_dos_observaciones(store):
    store.record_board([_event(h2h_home=1.80)], captured_at="2026-08-04T10:00:00+00:00")
    store.record_board([_event(h2h_home=1.95)], captured_at="2026-08-04T11:00:00+00:00")
    serie = store.trajectory("evt1", "h2h", "home")
    assert [r["price_dec"] for r in serie] == [1.80, 1.95]


def test_precio_que_vuelve_al_valor_previo_es_una_observacion_nueva(store):
    """Ida y vuelta al mismo número son dos movimientos, no un duplicado."""
    for ts, price in [("10:00", 1.80), ("11:00", 1.95), ("12:00", 1.80)]:
        store.record_board([_event(h2h_home=price)],
                           captured_at=f"2026-08-04T{ts}:00+00:00")
    serie = store.trajectory("evt1", "h2h", "home")
    assert [r["price_dec"] for r in serie] == [1.80, 1.95, 1.80]


# ── Deduplicado ──────────────────────────────────────────────────────────


def test_cotizacion_sin_movimiento_no_genera_fila_pero_si_barrida(store):
    """"El mercado no se movió" y "no miramos" tienen que ser distinguibles."""
    store.record_board([_event()], captured_at="2026-08-04T10:00:00+00:00")
    resumen = store.record_board([_event()], captured_at="2026-08-04T11:00:00+00:00")
    assert resumen["rows_new"] == 0
    assert resumen["unchanged"] == 6  # 2 lados × 3 mercados
    with store._conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM sweep").fetchone()[0] == 2


# ── El punto, pegado a su lado ───────────────────────────────────────────


def test_el_punto_del_runline_conserva_el_signo_de_cada_lado(store):
    """PURP-1 estructural: el punto nunca viaja separado de su lado.

    El local favorito cotiza −1.5 y el visitante +1.5. Guardar la magnitud
    (1.5) para ambos es lo que permitió emparejar la probabilidad del evento
    fácil con el precio del difícil durante 134 picks.
    """
    store.record_board([_event(rl_home_point=-1.5)])
    home = store.latest_before("evt1", "spreads", "home", "2099-01-01")
    away = store.latest_before("evt1", "spreads", "away", "2099-01-01")
    assert home["point"] == -1.5
    assert away["point"] == +1.5


def test_el_punto_del_no_favorito_local_tambien_conserva_signo(store):
    store.record_board([_event(rl_home_point=+1.5)])
    assert store.latest_before("evt1", "spreads", "home", "2099-01-01")["point"] == +1.5
    assert store.latest_before("evt1", "spreads", "away", "2099-01-01")["point"] == -1.5


# ── El par, o nada ───────────────────────────────────────────────────────


def test_pair_before_devuelve_los_dos_lados_del_mismo_libro(store):
    store.record_board([_event(h2h_home=1.80, h2h_away=2.10)])
    par = store.pair_before("evt1", "h2h", "home", "2099-01-01", book="pinnacle")
    assert par["price_side"] == 1.80
    assert par["price_opposite"] == 2.10


def test_pair_before_se_abstiene_con_medio_par(store):
    """Medio par no se puede desvigorizar; devolverlo invita a compararlo
    contra un precio crudo, que es el error que hacía ver habilidad donde
    solo había margen de la casa."""
    ev = _event()
    ev["bookmakers"][0]["markets"][0]["outcomes"] = [
        {"name": "Los Angeles Dodgers", "price": 1.80},
    ]
    store.record_board([ev])
    assert store.pair_before("evt1", "h2h", "home", "2099-01-01") is None


def test_pair_before_se_abstiene_si_los_puntos_no_coinciden(store):
    """Un total de 8.5 y uno de 9.0 no son dos lados del mismo mercado."""
    ev = _event()
    ev["bookmakers"][0]["markets"][1]["outcomes"] = [
        {"name": "Over", "price": 1.95, "point": 8.5},
        {"name": "Under", "price": 1.90, "point": 9.0},
    ]
    store.record_board([ev])
    assert store.pair_before("evt1", "totals", "over", "2099-01-01") is None


def test_pair_before_no_mezcla_libros(store):
    """Desvigorizar con un lado de una casa y el otro de otra da un número
    que no corresponde a ningún mercado real."""
    store.record_board([_event(book="pinnacle", h2h_home=1.80, h2h_away=2.10)])
    par = store.pair_before("evt1", "h2h", "home", "2099-01-01", book="fanduel")
    assert par is None


# ── Corte temporal ───────────────────────────────────────────────────────


def test_latest_before_es_estrictamente_anterior(store):
    """El corte excluye su propio instante. Un `<=` inclusivo sobre el corte
    es literalmente el leak que la Fase 2B tuvo que remediar en el camino PIT.
    """
    store.record_board([_event(h2h_home=1.80)], captured_at="2026-08-04T10:00:00+00:00")
    store.record_board([_event(h2h_home=1.95)], captured_at="2026-08-04T11:00:00+00:00")
    justo = store.latest_before("evt1", "h2h", "home", "2026-08-04T11:00:00+00:00")
    assert justo["price_dec"] == 1.80


def test_latest_before_sin_observacion_previa_devuelve_none(store):
    store.record_board([_event()], captured_at="2026-08-04T10:00:00+00:00")
    assert store.latest_before("evt1", "h2h", "home", "2026-08-04T09:00:00+00:00") is None


# ── Normalización que se abstiene ────────────────────────────────────────


def test_outcome_con_nombre_desconocido_se_cuenta_como_salto(store):
    """No se adivina a qué lado pertenece: se cuenta y queda en el resumen."""
    ev = _event()
    ev["bookmakers"][0]["markets"][0]["outcomes"] = [
        {"name": "Equipo Que No Juega", "price": 1.80},
        {"name": "San Diego Padres", "price": 2.10},
    ]
    resumen = store.record_board([ev])
    assert resumen["skipped"] == 1
    assert store.latest_before("evt1", "h2h", "home", "2099-01-01") is None


def test_evento_sin_id_del_proveedor_no_se_guarda(store):
    """Sin identidad no hay serie temporal posible, y una llave sintética
    armada con nombres de equipo falla en silencio en doubleheaders."""
    ev = _event()
    del ev["id"]
    resumen = store.record_board([ev])
    assert resumen["events"] == 0
    assert resumen["skipped"] == 1
    assert resumen["rows_new"] == 0


def test_el_libro_es_parte_del_dato(store):
    """324 de 464 picks del ledger vivo no tienen de qué casa salió su
    precio. Acá `book` es NOT NULL por esquema."""
    store.record_board([_event(book="pinnacle")])
    fila = store.latest_before("evt1", "h2h", "home", "2099-01-01")
    assert fila["book"] == "pinnacle"


# ── Pre-juego vs en vivo ─────────────────────────────────────────────────


def _event_en_vivo(**kw):
    """Mismo evento, pero la captura ocurre DESPUÉS del primer lanzamiento."""
    return _event(**kw)


def test_las_cotizaciones_en_vivo_no_se_mezclan_con_las_pre_juego(store):
    """El proveedor sigue devolviendo el evento después del primer
    lanzamiento, con precios EN VIVO. Sobre la captura real del 2026-08-04 el
    31.5% de las cotizaciones de MLB eran post-inicio, con moneylines de hasta
    150.0 a dos horas y media del comienzo. Mezclarlas produce pares con
    overround negativo que parecen arbitraje y son dos mercados distintos.
    """
    # commence_time del fixture: 2099-08-04T23:05:00Z
    store.record_board([_event(h2h_home=1.80)], captured_at="2099-08-04T22:00:00+00:00")
    store.record_board([_event(h2h_home=9.50)], captured_at="2099-08-05T01:00:00+00:00")

    pre = store.latest_before("evt1", "h2h", "home", "2099-12-31")
    assert pre["price_dec"] == 1.80

    con_vivo = store.latest_before("evt1", "h2h", "home", "2099-12-31",
                                   solo_pregame=False)
    assert con_vivo["price_dec"] == 9.50


def test_la_serie_pre_juego_excluye_lo_posterior_al_inicio(store):
    store.record_board([_event(h2h_home=1.80)], captured_at="2099-08-04T21:00:00+00:00")
    store.record_board([_event(h2h_home=1.95)], captured_at="2099-08-04T22:00:00+00:00")
    store.record_board([_event(h2h_home=9.50)], captured_at="2099-08-05T01:00:00+00:00")
    assert [r["price_dec"] for r in store.trajectory("evt1", "h2h", "home")] == [1.80, 1.95]
    assert len(store.trajectory("evt1", "h2h", "home", solo_pregame=False)) == 3


def test_el_precio_en_vivo_no_se_borra_solo_se_pide_aparte(store):
    """Un precio en vivo es un dato legítimo de otro producto."""
    store.record_board([_event(h2h_home=9.50)], captured_at="2099-08-05T01:00:00+00:00")
    assert store.latest_before("evt1", "h2h", "home", "2099-12-31") is None
    assert store.latest_before("evt1", "h2h", "home", "2099-12-31",
                               solo_pregame=False) is not None
