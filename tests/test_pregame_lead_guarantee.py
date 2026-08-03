"""La garantía pre-juego del publisher: sin horario demostrable, no se publica.

`track_record/publisher.py` promete en su docstring que todo pick lleva un
`published_at` anterior al primer pitcheo. Esa promesa vale lo que valga el
bloque de MIN_LEAD_MINUTES, y ese bloque ya estuvo vacío una vez: hasta el
2026-07-19 leía `commence_time`/`game_datetime`, claves que `_parse_game` nunca
puebla para MLB, así que `commence_raw` era siempre "" y el gate **nunca
disparó** — meses con la garantía muerta y ningún test que lo notara.

Estos tests existen para que no vuelva a pasar en silencio. Cubren los dos
bordes que hasta el 2026-07-26 fallaban ABIERTOS (`except Exception: pass —
proceed`) y ahora fallan cerrados, y la deriva del reloj.
"""
from datetime import datetime, timedelta, timezone

import pytest

from track_record import publisher
from track_record.db import TrackRecordDB


def _futuro(minutos: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutos)).isoformat()


def _juego(commence: str | None, **extra):
    g = {
        "game_pk": 999001, "home_team": "New York Yankees",
        "away_team": "Pittsburgh Pirates", "official_date": "2026-07-27",
        "status": "Scheduled", "game_type": "R",
    }
    if commence is not None:
        g["game_date"] = commence
    g.update(extra)
    return g


# ── el parser ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("crudo,legible", [
    ("2026-07-27T18:35:00Z", True),
    ("2026-07-27T18:35:00+00:00", True),
    ("2026-07-27T18:35:00", True),          # naive ⇒ se asume UTC
    ("", False),
    (None, False),
    ("mañana a la tarde", False),
    ("2026-13-45T99:99:99Z", False),
])
def test_parseo_de_horario(crudo, legible):
    assert (publisher._parse_commence_utc(crudo) is not None) is legible


def test_naive_se_interpreta_como_utc():
    dt = publisher._parse_commence_utc("2026-07-27T18:35:00")
    assert dt.tzinfo == timezone.utc


# ── la garantía ───────────────────────────────────────────────────────────────

def _corre(db, juego, monkeypatch):
    llamadas = []
    import modules.baseball_module.core.run_module as rm
    monkeypatch.setattr(rm, "run_module",
                        lambda **k: llamadas.append(k) or {"status": "error"})
    picks = publisher.publish_mlb_picks(db, games=[juego], dry_run=True)
    return picks, llamadas


def test_sin_horario_no_publica(tmp_path, monkeypatch):
    """Antes seguía de largo: el `if commence_raw:` dejaba pasar el juego entero."""
    db = TrackRecordDB(db_path=tmp_path / "tr.db")
    picks, llamadas = _corre(db, _juego(None), monkeypatch)
    assert picks == []
    assert llamadas == [], "sin horario no se puede demostrar pre-juego: ni se analiza"


def test_horario_ilegible_no_publica(tmp_path, monkeypatch):
    """Antes: `except Exception: pass  # can't parse time → proceed`."""
    db = TrackRecordDB(db_path=tmp_path / "tr.db")
    picks, llamadas = _corre(db, _juego("no-es-una-fecha"), monkeypatch)
    assert picks == []
    assert llamadas == []


def test_juego_demasiado_cerca_no_publica(tmp_path, monkeypatch):
    db = TrackRecordDB(db_path=tmp_path / "tr.db")
    picks, llamadas = _corre(db, _juego(_futuro(publisher.MIN_LEAD_MINUTES - 5)), monkeypatch)
    assert picks == []
    assert llamadas == []


def test_juego_con_margen_si_se_analiza(tmp_path, monkeypatch):
    db = TrackRecordDB(db_path=tmp_path / "tr.db")
    picks, llamadas = _corre(db, _juego(_futuro(publisher.MIN_LEAD_MINUTES + 120)), monkeypatch)
    assert len(llamadas) == 1, "con margen suficiente, el juego sí se analiza"


def test_el_umbral_se_mide_contra_el_reloj_del_momento(tmp_path, monkeypatch):
    """El `now` del inicio del bucle quedaba hasta 9 min atrasado con 27 juegos
    (cada uno corre el pipeline completo). La anticipación verificada tiene que
    ser la del momento de publicar, no la de cuando arrancó la corrida."""
    db = TrackRecordDB(db_path=tmp_path / "tr.db")
    # Justo por debajo del umbral AHORA: con un reloj viejo habría pasado.
    picks, llamadas = _corre(db, _juego(_futuro(publisher.MIN_LEAD_MINUTES - 1)), monkeypatch)
    assert llamadas == []
