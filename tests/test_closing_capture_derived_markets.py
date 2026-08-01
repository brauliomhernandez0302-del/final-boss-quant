"""Closing capture must cover DERIVED markets, not just h2h.

Until 2026-07-26 `capture_closing_lines.py` had branches for ML_HOME/ML_AWAY
only. Every runline and total pick therefore stored a NULL closing price next
to the MONEYLINE Pinnacle pair — 100% attrition for those markets, and the
close is unrecoverable once the game starts. The market data was already in
every sweep's response (`odds_fetcher.MARKETS` = h2h/totals/spreads); it was
parsed and dropped.

These tests pin the end-to-end path (sweep → db) for one pick of each family,
plus the line-movement flag, which is what makes a captured close usable: a
total that closed at 9.0 does not grade a pick taken at 8.5.
"""
import pytest

from track_record import capture_closing_lines as ccl
from track_record.db import TrackRecordDB

GAME_PK = 990001
COMMENCE = "2999-01-01T00:00:00+00:00"   # far future: never rejected as post-start

# One sweep's worth of market data, shaped exactly like
# odds_fetcher.get_best_odds_for_teams() returns it.
MARKET_ODDS = {
    "home_team": "New York Yankees",
    "away_team": "Boston Red Sox",
    "ml_home": 1.80, "ml_away": 2.10,
    "pin_home": 1.78, "pin_away": 2.12,
    "all_books_h2h": [],
    "total_line": 8.5, "total_over": 1.95, "total_under": 1.92,
    "pin_total_over": 1.93, "pin_total_under": 1.90, "pin_total_point": 8.5,
    "runline_home": 1.60, "runline_away": 2.45,
    "pin_runline_home": 1.57, "pin_runline_away": 2.40,
    "pin_runline_home_point": -1.5, "pin_runline_away_point": 1.5,
    "runline_home_point": -1.5, "runline_away_point": 1.5,
}


@pytest.fixture
def db(tmp_path):
    return TrackRecordDB(db_path=tmp_path / "tr.db")


def _publish(db, market, odds_decimal, *, total_line=None, runline_point=None):
    uid = f"MLB:{GAME_PK}:{market}:2026-07-26"
    db.publish_pick(
        pick_uid=uid, game_date="2026-07-26", sport="MLB", game_pk=GAME_PK,
        home_team="New York Yankees", away_team="Boston Red Sox",
        market=market, model_prob=0.55, ev_pct=5.0, odds_decimal=odds_decimal,
        commence_time=COMMENCE, total_line=total_line, runline_point=runline_point,
    )
    return uid


def _sweep(db, monkeypatch, odds=None):
    # capture_closing_lines() imports the fetcher inside the function, so the
    # patch has to land on the source module.
    import odds_fetcher
    monkeypatch.setattr(odds_fetcher, "get_best_odds_for_teams", lambda *a, **k: odds or MARKET_ODDS)
    return ccl.capture_closing_lines(db=db, dry_run=False)


def _row(db, uid):
    return next(r for r in db.get_picks() if r["pick_uid"] == uid)


@pytest.mark.parametrize(
    "market,odds,kwargs,expected_close,expected_pin_side,expected_pin_opp",
    [
        ("ML_HOME", 1.90, {}, 1.80, 1.78, 2.12),
        ("OVER", 2.05, {"total_line": 8.5}, 1.95, 1.93, 1.90),
        ("UNDER", 2.00, {"total_line": 8.5}, 1.92, 1.90, 1.93),
        ("RL_HOME", 1.70, {"runline_point": -1.5}, 1.60, 1.57, 2.40),
        ("RL_AWAY", 2.60, {"runline_point": 1.5}, 2.45, 2.40, 1.57),
    ],
)
def test_every_family_captures_a_non_null_closing_price(
    db, monkeypatch, market, odds, kwargs, expected_close, expected_pin_side, expected_pin_opp
):
    uid = _publish(db, market, odds, **kwargs)
    summary = _sweep(db, monkeypatch)
    assert summary["captured"] == 1

    row = _row(db, uid)
    assert row["closing_odds_decimal"] == expected_close, "el precio del lado del pick"
    assert row["closing_pin_side"] == expected_pin_side
    assert row["closing_pin_opposite"] == expected_pin_opp
    assert row["closing_captured_at"] is not None


def test_clv_pct_ahora_cubre_derivados_con_el_punto_QUIETO(db, monkeypatch):
    """Actualizado 2026-08-01: el CLV se extendió a derivados cuando el punto NO
    se movió. La objeción original —un total tomado a 8.5 que cierra en 9.0 no
    es el mismo mercado— sigue vigente y descarta ESE caso, no el mercado entero.

    Razón para definirlo ahora: elegir el filtro del producto por ROI necesita
    ~14.800 picks (740 días); por CLV, ~117 (6 días). Extenderlo duplicó la
    muestra con CLV de 105 a 228 picks del ledger real.
    """
    ml = _publish(db, "ML_HOME", 1.90)
    over = _publish(db, "OVER", 2.05, total_line=8.5)
    _sweep(db, monkeypatch)

    assert _row(db, ml)["clv_pct"] is not None
    over_row = _row(db, over)
    # El barrido cierra este total en el mismo punto, así que sí tiene CLV.
    assert over_row["closing_point_moved"] == 0
    assert over_row["clv_pct"] is not None
    # ...y sin embargo el cierre quedó guardado y es computable después:
    assert over_row["closing_odds_decimal"] == 1.95
    assert over_row["closing_pin_side"] == 1.93


def test_line_movement_is_flagged(db, monkeypatch):
    """Un total que cerró en 9.0 no gradúa un pick tomado a 8.5."""
    moved = _publish(db, "OVER", 2.05, total_line=8.5)
    odds = {**MARKET_ODDS, "pin_total_point": 9.0, "total_line": 9.0}
    _sweep(db, monkeypatch, odds)

    row = _row(db, moved)
    assert row["closing_point"] == 9.0
    assert row["closing_point_moved"] == 1


def test_unmoved_line_is_flagged_as_such(db, monkeypatch):
    same = _publish(db, "OVER", 2.05, total_line=8.5)
    _sweep(db, monkeypatch)

    row = _row(db, same)
    assert row["closing_point"] == 8.5
    assert row["closing_point_moved"] == 0


def test_runline_point_sign_disagreement_is_not_a_false_move(db, monkeypatch):
    """Los libros no coinciden en el signo del runline (ver odds_fetcher);
    la comparación es por magnitud, así que -1.5 vs +1.5 no es movimiento."""
    uid = _publish(db, "RL_AWAY", 2.60, runline_point=1.5)
    odds = {**MARKET_ODDS, "pin_runline_away_point": -1.5}
    _sweep(db, monkeypatch, odds)

    assert _row(db, uid)["closing_point_moved"] == 0


def test_moneyline_never_gets_a_point(db, monkeypatch):
    uid = _publish(db, "ML_HOME", 1.90)
    _sweep(db, monkeypatch)

    row = _row(db, uid)
    assert row["closing_point"] is None
    assert row["closing_point_moved"] is None
