"""Juegos suspendidos: no publicar sobre ellos, y no graduarlos contra otro partido.

Caso real que motivó estos tests (2026-07-26): `gamePk=823519`, PIT@NYY del
2026-07-21. El pick `OVER 9.0` se publicó el 20/07 contra el abridor **Will
Warren**; el juego se suspendió por lluvia y MLB lo repuso el 22/07 **bajo el
mismo gamePk**, esta vez con **Max Fried**. Terminó 2-0 y el reconciliador lo
graduó LOSS con −1.28u, porque buscaba por gamePk y verificaba `status == Final`
sin mirar nunca la fecha.

Dos agujeros distintos, dos defensas distintas:

  publicación   — un juego YA suspendido conserva su horario original, así que
                  pasaba MIN_LEAD_MINUTES sin objeción. Ahora hay lista de
                  estados publicables.
  reconciliación— un juego suspendido DESPUÉS de publicar el pick no lo salva
                  ningún filtro de publicación: hay que detectarlo al graduar.

La segunda es la que habría atrapado el caso real: al momento de publicar, el
juego figuraba `Scheduled`.

Tasa medida sobre la temporada 2026: 25 suspensiones en 1,612 juegos (1.55%),
repartidas parejo entre abril y julio — proyecta ~10 juegos por ventana de CLV
de 6 semanas.
"""
import pytest

from track_record import publisher, reconciler
from track_record.db import TrackRecordDB


# ── Publicación ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("estado,publicable", [
    ("Scheduled", True),
    ("Pre-Game", True),
    ("Warmup", True),
    ("Postponed", False),
    ("Final", False),
    ("Game Over", False),
    ("In Progress", False),
    ("Suspended", False),
    ("Cancelled", False),
    ("Estado Nuevo Que MLB Invente", False),   # desconocido ⇒ abstenerse
])
def test_estados_publicables(estado, publicable):
    assert (estado in publisher.PUBLISHABLE_STATUSES) is publicable


@pytest.mark.parametrize("tipo,publicable", [
    ("R", True),    # temporada regular
    ("F", True), ("D", True), ("L", True), ("W", True),   # playoffs
    ("S", False),   # spring training
    ("E", False),   # exhibición
    ("A", False),   # all-star
])
def test_tipos_publicables(tipo, publicable):
    assert (tipo in publisher.PUBLISHABLE_GAME_TYPES) is publicable


def test_un_juego_suspendido_no_se_publica(tmp_path, monkeypatch):
    """El juego suspendido conserva horario futuro: sin filtro de estado pasaba."""
    db = TrackRecordDB(db_path=tmp_path / "tr.db")
    llamadas = []

    juego = {
        "game_pk": 823519, "home_team": "New York Yankees",
        "away_team": "Pittsburgh Pirates", "official_date": "2026-07-21",
        "game_date": "2999-07-21T23:05:00Z",     # futuro: pasaría MIN_LEAD_MINUTES
        "status": "Postponed", "game_type": "R",
    }

    import modules.baseball_module.core.run_module as rm
    monkeypatch.setattr(rm, "run_module", lambda **k: llamadas.append(k) or {"status": "error"})

    picks = publisher.publish_mlb_picks(db, games=[juego], dry_run=True)
    assert picks == []
    assert llamadas == [], "no debe correr el pipeline sobre un juego suspendido"


# ── Reconciliación ────────────────────────────────────────────────────────────

def _pick(db, game_pk=823519, game_date="2026-07-21"):
    uid = f"MLB:{game_pk}:OVER:{game_date}"
    db.publish_pick(
        pick_uid=uid, game_date=game_date, sport="MLB", game_pk=game_pk,
        home_team="New York Yankees", away_team="Pittsburgh Pirates",
        market="OVER", model_prob=0.55, ev_pct=14.24, odds_decimal=1.95,
        total_line=9.0, commence_time=f"{game_date}T23:05:00+00:00",
    )
    return uid


def test_void_cuando_el_juego_se_jugo_otro_dia(tmp_path, monkeypatch):
    """El caso real: mismo gamePk, partido repuesto al día siguiente."""
    db = TrackRecordDB(db_path=tmp_path / "tr.db")
    uid = _pick(db)

    monkeypatch.setattr(reconciler, "_get_final_score", lambda s, pk: (2, 0))
    monkeypatch.setattr(reconciler, "_fetch_official_date", lambda pk: "2026-07-22")

    stats = reconciler.reconcile_pending(db=db, lookback_days=3650)

    fila = next(r for r in db.get_picks() if r["pick_uid"] == uid)
    assert fila["result"] == "VOID"
    assert fila["profit_loss_units"] == 0.0
    assert stats["voided"] == 1
    assert stats["resolved"] == 0


def test_se_gradua_normal_cuando_la_fecha_coincide(tmp_path, monkeypatch):
    db = TrackRecordDB(db_path=tmp_path / "tr.db")
    uid = _pick(db)

    monkeypatch.setattr(reconciler, "_get_final_score", lambda s, pk: (2, 0))
    monkeypatch.setattr(reconciler, "_fetch_official_date", lambda pk: "2026-07-21")

    reconciler.reconcile_pending(db=db, lookback_days=3650)

    fila = next(r for r in db.get_picks() if r["pick_uid"] == uid)
    assert fila["result"] == "LOSS", "2+0=2 < 9.0 ⇒ el OVER pierde, como corresponde"


def test_sin_fecha_disponible_no_bloquea_la_graduacion(tmp_path, monkeypatch):
    """Si la API no devuelve officialDate, se gradúa igual: la defensa no debe
    convertirse en un motivo nuevo de picks eternamente pendientes."""
    db = TrackRecordDB(db_path=tmp_path / "tr.db")
    uid = _pick(db)

    monkeypatch.setattr(reconciler, "_get_final_score", lambda s, pk: (2, 0))
    monkeypatch.setattr(reconciler, "_fetch_official_date", lambda pk: None)

    reconciler.reconcile_pending(db=db, lookback_days=3650)

    fila = next(r for r in db.get_picks() if r["pick_uid"] == uid)
    assert fila["result"] == "LOSS"
