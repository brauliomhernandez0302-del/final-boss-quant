"""El almacén de anuncios: lo valioso es CUÁNDO se supo, no qué dice hoy."""

from __future__ import annotations

import pytest

from fbq.anuncios.store import AnunciosStore


def _fila(pk=1, h=100, a=200):
    return {"game_pk": pk, "official_date": "2026-09-10",
            "commence_time": "2026-09-10T23:05:00Z", "estado": "Scheduled",
            "home_pitcher_id": h, "home_pitcher": "H",
            "away_pitcher_id": a, "away_pitcher": "A"}


def test_es_append_only_por_trigger(tmp_path):
    """Un anuncio que cambia es el dato más caro de todos; pisarlo lo destruye."""
    import sqlite3
    s = AnunciosStore(tmp_path / "a.db")
    s.registrar([_fila()], fecha="2026-09-10")
    with s._conn() as c:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            c.execute("UPDATE anuncio SET pitcher_id = 999")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            c.execute("DELETE FROM anuncio")


def test_solo_inserta_lo_que_cambio_pero_registra_la_barrida(tmp_path):
    """"No cambió" y "no miramos" tienen que ser distinguibles."""
    s = AnunciosStore(tmp_path / "a.db")
    assert s.registrar([_fila()], fecha="2026-09-10")["nuevos"] == 2
    r = s.registrar([_fila()], fecha="2026-09-10")
    assert r["nuevos"] == 0 and r["sin_cambio"] == 2
    assert s.resumen()["anuncios"] == 2 and s.resumen()["barridas"] == 2


def test_un_cambio_de_abridor_deja_las_DOS_observaciones(tmp_path):
    s = AnunciosStore(tmp_path / "a.db")
    s.registrar([_fila(h=100)], fecha="2026-09-10", observado_en="2026-09-09T12:00:00Z")
    s.registrar([_fila(h=555)], fecha="2026-09-10", observado_en="2026-09-10T12:00:00Z")
    with s._conn() as c:
        ids = [r["pitcher_id"] for r in c.execute(
            "SELECT pitcher_id FROM anuncio WHERE lado='home' ORDER BY id")]
    assert ids == [100, 555]


def test_vigente_antes_devuelve_lo_que_se_sabia_no_lo_que_paso(tmp_path):
    """La única lectura permitida a un modelo."""
    s = AnunciosStore(tmp_path / "a.db")
    s.registrar([_fila(h=100)], fecha="2026-09-10", observado_en="2026-09-09T12:00:00Z")
    s.registrar([_fila(h=555)], fecha="2026-09-10", observado_en="2026-09-10T20:00:00Z")
    corte = "2026-09-10T17:00:00+00:00"
    assert s.vigente_antes(1, "home", corte)["pitcher_id"] == 100   # no el 555
    # sin observación previa al corte no se rellena: se devuelve None
    assert s.vigente_antes(1, "home", "2026-09-01T00:00:00+00:00") is None


def test_sin_anuncio_todavia_se_guarda_como_None_y_eso_es_informacion(tmp_path):
    s = AnunciosStore(tmp_path / "a.db")
    s.registrar([_fila(h=None)], fecha="2026-09-10")
    r = s.vigente_antes(1, "home", "2099-01-01T00:00:00+00:00")
    assert r is not None and r["pitcher_id"] is None


# ── Conservar lo que se sabía ENTONCES (2026-09-06) ──────────────────────

def test_el_anunciado_al_corte_se_conserva_aunque_haya_abierto_otro(tmp_path):
    """El caso que da sentido a todo el almacén.

    Observado con datos reales: `game_pk` 824402 tenía anunciados 677944 y
    690997 el 2026-08-04; el 2026-08-05 a las 20:00 se anunciaron otros dos, y
    ésos abrieron. Pero el corte del precio fue a las 18:51 de ese mismo día, o
    sea ANTES del segundo anuncio. Al corte, lo que se sabía eran los primeros.
    """
    s = AnunciosStore(tmp_path / "a.db")
    s.registrar([_fila(pk=824402, h=677944, a=690997)], fecha="2026-08-06",
                observado_en="2026-08-04T20:02:05Z")
    s.registrar([_fila(pk=824402, h=676440, a=681035)], fecha="2026-08-06",
                observado_en="2026-08-05T20:00:38Z")
    corte = "2026-08-05T18:51:02+00:00"
    assert s.vigente_antes(824402, "home", corte)["pitcher_id"] == 677944
    assert s.vigente_antes(824402, "away", corte)["pitcher_id"] == 690997
    # después del segundo anuncio, manda el segundo
    tarde = "2026-08-06T00:00:00+00:00"
    assert s.vigente_antes(824402, "home", tarde)["pitcher_id"] == 676440


def test_cada_reemplazo_lleva_su_propia_hora_de_disponibilidad(tmp_path):
    """Un reemplazo sin su hora no sirve: no se puede afirmar cuándo se supo."""
    s = AnunciosStore(tmp_path / "a.db")
    s.registrar([_fila(h=100)], fecha="2026-08-06", observado_en="2026-08-04T20:00:00Z")
    s.registrar([_fila(h=200)], fecha="2026-08-06", observado_en="2026-08-05T14:00:00Z")
    with s._conn() as c:
        obs = c.execute("SELECT pitcher_id, observado_en FROM anuncio "
                        "WHERE lado='home' ORDER BY id").fetchall()
    assert [r["pitcher_id"] for r in obs] == [100, 200]
    assert obs[0]["observado_en"] != obs[1]["observado_en"]
    assert all(r["observado_en"].endswith("+00:00") for r in obs)


def test_un_anuncio_que_aparece_donde_no_habia_tambien_es_un_cambio(tmp_path):
    """Pasar de "sin anuncio" a "anunciado" es información con su propia hora."""
    s = AnunciosStore(tmp_path / "a.db")
    s.registrar([_fila(h=None)], fecha="2026-08-06", observado_en="2026-08-02T20:01:59Z")
    s.registrar([_fila(h=615698)], fecha="2026-08-06", observado_en="2026-08-03T20:00:47Z")
    with s._conn() as c:
        ids = [r["pitcher_id"] for r in c.execute(
            "SELECT pitcher_id FROM anuncio WHERE lado='home' ORDER BY id")]
    assert ids == [None, 615698]
    # antes del segundo anuncio, lo que se sabía era "no se sabe"
    r = s.vigente_antes(1, "home", "2026-08-03T00:00:00+00:00")
    assert r is not None and r["pitcher_id"] is None


def test_la_importacion_del_track_record_trae_el_instante_no_solo_la_identidad(tmp_path):
    """Un anuncio sin su hora de observación no sirve para predecir."""
    import json as _json
    import sqlite3
    from fbq.anuncios.importar_track_record import _observaciones

    origen = tmp_path / "tr.db"
    con = sqlite3.connect(origen)
    con.execute("CREATE TABLE picks (game_pk INT, published_at TEXT, "
                "commence_time TEXT, game_date TEXT, pipeline_json TEXT)")
    con.execute("INSERT INTO picks VALUES (?,?,?,?,?)", (
        7, "2026-08-04T20:02:05.123456+00:00", "2026-08-06T23:05:00Z", "2026-08-06",
        _json.dumps({"inputs": {"home_pitcher_id": 111, "away_pitcher_id": 222,
                                "official_date": "2026-08-06"}})))
    # sin published_at utilizable no entra: no se le inventa una hora
    con.execute("INSERT INTO picks VALUES (?,?,?,?,?)", (
        8, "2026-08-04", "x", "2026-08-06",
        _json.dumps({"inputs": {"home_pitcher_id": 333}})))
    con.commit(); con.close()

    obs = _observaciones(origen)
    assert len(obs) == 1
    assert obs[0]["game_pk"] == 7
    assert obs[0]["observado_en"] == "2026-08-04T20:02:05.123456+00:00"
