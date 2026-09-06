"""La importación del histórico: hora verificada, nada inventado, nada duplicado.

Cada test de acá corresponde a un defecto concreto que la versión anterior del
script tenía o habría tenido. No hay red: el índice del schedule se inyecta.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fbq.market import importar_historico as imp
from fbq.market.store import MarketStore

COLUMNAS = [
    "game_pk", "game_date", "season", "odds_api_id", "home_team", "away_team",
    "snapshot_ts", "ml_home_best", "ml_away_best", "ml_home_best_bk",
    "ml_away_best_bk", "ml_home_cons", "ml_away_cons", "ml_home_pin",
    "ml_away_pin", "total_point_pin", "total_over_pin", "total_under_pin",
    "total_point_best", "total_over_best", "total_under_best",
    "rl_home_point_pin", "rl_home_pin", "rl_away_pin", "rl_home_point_best",
    "rl_home_best", "rl_away_best",
]


def _origen(tmp_path: Path, filas) -> Path:
    """Un `historical_odds` mínimo con la forma del sistema anterior."""
    ruta = tmp_path / "origen.db"
    con = sqlite3.connect(ruta)
    con.execute(f"CREATE TABLE historical_odds ({','.join(COLUMNAS)})")
    con.executemany(
        f"INSERT INTO historical_odds ({','.join(COLUMNAS)}) "
        f"VALUES ({','.join('?' * len(COLUMNAS))})",
        [tuple(f.get(c) for c in COLUMNAS) for f in filas],
    )
    con.commit()
    con.close()
    return ruta


def _fila(**kw):
    base = dict(
        game_pk=745000, game_date="2024-04-24", season=2024,
        odds_api_id="EV1", home_team="Local", away_team="Visita",
        snapshot_ts="2024-04-24T17:00:00Z",
        ml_home_pin=1.90, ml_away_pin=2.02,
        ml_home_cons=1.88, ml_away_cons=2.00,
        ml_home_best=1.95, ml_away_best=2.10,
        ml_home_best_bk="draftkings", ml_away_best_bk="fanduel",
        total_point_pin=8.5, total_over_pin=1.91, total_under_pin=1.95,
        rl_home_point_pin=-1.5, rl_home_pin=2.30, rl_away_pin=1.65,
    )
    base.update(kw)
    return base


def _indice(*entradas):
    idx = {}
    for e in entradas:
        idx.setdefault(int(e["game_pk"]), []).append(e)
    return idx


JUEGO_NORMAL = {"game_pk": 745000, "game_date": "2024-04-24T22:05:00Z",
                "official_date": "2024-04-24", "estado": "Final"}


# ── La corrección central ────────────────────────────────────────────────

def test_commence_time_es_un_instante_completo_y_no_una_fecha(tmp_path):
    """El defecto que motivó la reescritura.

    La versión anterior escribía `commence_time='2024-04-24'` (de
    `game_outcomes.game_date`, 10 caracteres). El filtro pre-juego compara
    cadenas en SQL, así que `'2024-04-24T17:00:00Z' < '2024-04-24'` daba FALSO
    y todo lo importado quedaba invisible.
    """
    store = MarketStore(tmp_path / "m.db")
    imp.importar(store, origen=_origen(tmp_path, [_fila()]),
                 pendientes=tmp_path / "p.csv", indice=_indice(JUEGO_NORMAL))
    with store._conn() as c:
        commence = c.execute("SELECT DISTINCT commence_time FROM odds_snapshot").fetchone()[0]
    assert commence == "2024-04-24T22:05:00+00:00"
    assert len(commence) > 10


def test_las_filas_importadas_se_ven_con_el_filtro_pre_juego_por_defecto(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    imp.importar(store, origen=_origen(tmp_path, [_fila()]),
                 pendientes=tmp_path / "p.csv", indice=_indice(JUEGO_NORMAL))
    futuro = "2099-01-01T00:00:00+00:00"

    assert store.latest_before("EV1", "h2h", "home", futuro) is not None
    par = store.pair_before("EV1", "h2h", "home", futuro)
    assert par is not None and par["price_side"] == 1.90 and par["price_opposite"] == 2.02
    assert store.pair_before("EV1", "totals", "over", futuro) is not None
    assert store.pair_before("EV1", "spreads", "home", futuro) is not None


def test_los_instantes_quedan_normalizados_a_utc_explicito(tmp_path):
    """`...Z` y `...+00:00` son el mismo instante y ordenan distinto como texto.
    El almacén compara cadenas, así que la forma tiene que ser una sola."""
    store = MarketStore(tmp_path / "m.db")
    imp.importar(store, origen=_origen(tmp_path, [_fila()]),
                 pendientes=tmp_path / "p.csv", indice=_indice(JUEGO_NORMAL))
    with store._conn() as c:
        for captured, commence in c.execute(
                "SELECT captured_at, commence_time FROM odds_snapshot"):
            assert captured.endswith("+00:00")
            assert commence.endswith("+00:00")


# ── Nada se inventa, nada se descarta en silencio ────────────────────────

def test_un_juego_sin_hora_no_se_inventa_y_queda_en_el_informe(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    r = imp.importar(store, origen=_origen(tmp_path, [_fila()]),
                     pendientes=tmp_path / "p.csv", indice={})
    assert r["juegos_resueltos"] == 0
    assert r["por_motivo"] == {"sin_juego_en_schedule": 1}
    assert store.summary()["rows"] == 0
    texto = (tmp_path / "p.csv").read_text(encoding="utf-8")
    assert "745000" in texto and "sin_juego_en_schedule" in texto


def test_una_fila_sin_event_id_queda_registrada(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    r = imp.importar(store, origen=_origen(tmp_path, [_fila(odds_api_id=None)]),
                     pendientes=tmp_path / "p.csv", indice=_indice(JUEGO_NORMAL))
    assert r["por_motivo"] == {"sin_event_id": 1}
    assert "sin_event_id" in (tmp_path / "p.csv").read_text(encoding="utf-8")


def test_el_informe_lleva_los_candidatos_para_poder_conciliar(tmp_path):
    """Un pendiente sin sus candidatos obliga a rehacer el análisis a mano."""
    suspendido_a = {"game_pk": 745000, "game_date": "2024-04-23T23:45:00Z",
                    "official_date": "2024-04-23", "estado": "Final"}
    suspendido_b = {"game_pk": 745000, "game_date": "2024-04-24T16:15:00Z",
                    "official_date": "2024-04-23", "estado": "Final"}
    store = MarketStore(tmp_path / "m.db")
    r = imp.importar(store, origen=_origen(tmp_path, [_fila()]),
                     pendientes=tmp_path / "p.csv",
                     indice=_indice(suspendido_a, suspendido_b))
    assert r["por_motivo"] == {"fecha_ambigua": 1}
    texto = (tmp_path / "p.csv").read_text(encoding="utf-8")
    assert "2024-04-23T23:45:00Z" in texto and "2024-04-24T16:15:00Z" in texto


# ── Desempate de pospuestos ──────────────────────────────────────────────

def test_entre_el_cascaron_pospuesto_y_el_partido_gana_el_que_termino(tmp_path):
    cascaron = {"game_pk": 745000, "game_date": "2024-04-23T19:05:00Z",
                "official_date": "2024-04-24", "estado": "Postponed"}
    jugado = {"game_pk": 745000, "game_date": "2024-04-24T22:05:00Z",
              "official_date": "2024-04-24", "estado": "Final"}
    assert imp.resolver_inicio(745000, "2024-04-24", _indice(cascaron, jugado)) == \
        ("2024-04-24T22:05:00+00:00", "ok")


def test_dos_partidos_terminados_el_mismo_dia_oficial_no_se_desempatan(tmp_path):
    """Un juego suspendido y reanudado deja dos entradas `Final`. Elegir una al
    azar metería la hora del partido que no se jugó."""
    a = {"game_pk": 745000, "game_date": "2024-05-21T23:45:00Z",
         "official_date": "2024-05-21", "estado": "Final"}
    b = {"game_pk": 745000, "game_date": "2024-05-22T16:15:00Z",
         "official_date": "2024-05-21", "estado": "Final"}
    assert imp.resolver_inicio(745000, "2024-05-22", _indice(a, b))[0] is None


# ── Idempotencia ─────────────────────────────────────────────────────────

def test_importar_dos_veces_no_duplica_ninguna_fila(tmp_path):
    """`odds_snapshot` es append-only por trigger: un INSERT repetido NO se
    puede deshacer después con un DELETE. La única defensa es no insertarlo."""
    origen = _origen(tmp_path, [_fila()])
    store = MarketStore(tmp_path / "m.db")
    idx = _indice(JUEGO_NORMAL)

    r1 = imp.importar(store, origen=origen, pendientes=tmp_path / "p.csv", indice=idx)
    n1 = store.summary()["rows"]
    r2 = imp.importar(store, origen=origen, pendientes=tmp_path / "p.csv", indice=idx)
    n2 = store.summary()["rows"]

    assert r1["cotizaciones_nuevas"] > 0 and r1["cotizaciones_ya_estaban"] == 0
    assert r2["cotizaciones_nuevas"] == 0
    assert r2["cotizaciones_ya_estaban"] == r1["cotizaciones_nuevas"]
    assert n1 == n2


def test_el_enlace_evento_juego_tambien_es_idempotente(tmp_path):
    origen = _origen(tmp_path, [_fila()])
    store = MarketStore(tmp_path / "m.db")
    idx = _indice(JUEGO_NORMAL)
    imp.importar(store, origen=origen, pendientes=tmp_path / "p.csv", indice=idx)
    imp.importar(store, origen=origen, pendientes=tmp_path / "p.csv", indice=idx)
    assert store.summary()["linked"] == 1
    assert store.event_for_game(745000)["event_id"] == "EV1"


# ── Los precios en vivo se guardan, pero no se leen por defecto ──────────

def test_una_cotizacion_posterior_al_inicio_se_guarda_y_queda_fuera_del_pre_juego(tmp_path):
    """Un precio en vivo es un dato legítimo de otro producto: no se borra.
    Pero el nulo pre-juego no puede salir de él."""
    temprano = {"game_pk": 745000, "game_date": "2024-04-24T16:35:00Z",
                "official_date": "2024-04-24", "estado": "Final"}
    store = MarketStore(tmp_path / "m.db")
    r = imp.importar(store, origen=_origen(tmp_path, [_fila()]),
                     pendientes=tmp_path / "p.csv", indice=_indice(temprano))
    futuro = "2099-01-01T00:00:00+00:00"

    assert r["cotizaciones_en_vivo"] == 1
    assert store.summary()["rows"] > 0
    assert store.pair_before("EV1", "h2h", "home", futuro) is None
    assert store.pair_before("EV1", "h2h", "home", futuro, solo_pregame=False) is not None
