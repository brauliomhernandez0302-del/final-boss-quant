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
