"""Tests for track_record/db.py's closing-line (CLV) capture — added
2026-07-12, see db.py's schema-migration comment and capture_closing_lines.py
for the full rationale (this is the project's first REAL closing-line
value, distinct from the edge-as-ratio "clv_home"/"clv_away" fields in
backtest_and_retrain.py — see feedback_clv_misnomer memory)."""
import pytest
from pathlib import Path

from track_record.db import TrackRecordDB


@pytest.fixture
def db(tmp_path):
    return TrackRecordDB(db_path=tmp_path / "test_track_record.db")


def _publish(db, pick_uid="778000:ML_HOME", market="ML_HOME", odds_decimal=2.00):
    db.publish_pick(
        pick_uid=pick_uid,
        game_date="2026-07-12",
        sport="MLB",
        game_pk=778000,
        home_team="New York Yankees",
        away_team="Boston Red Sox",
        market=market,
        model_prob=0.55,
        ev_pct=0.05,
        odds_decimal=odds_decimal,
    )


class TestClosingLineCapture:

    def test_schema_migration_idempotent(self, db):
        # Re-init on the same DB must not raise (ALTER TABLE ADD COLUMN twice).
        TrackRecordDB(db_path=db.db_path)
        TrackRecordDB(db_path=db.db_path)

    def test_ml_home_beats_close_gives_positive_clv(self, db):
        # Published at 2.00, closed at 1.90 (price shortened) -> we beat the close.
        _publish(db, odds_decimal=2.00)
        ok = db.capture_closing_line(
            "778000:ML_HOME", closing_odds_decimal=1.90,
            closing_pin_home=1.90, closing_pin_away=2.05,
        )
        assert ok
        row = db.get_picks()[0]
        assert row["clv_pct"] == pytest.approx((2.00 / 1.90 - 1) * 100, abs=1e-3)
        assert row["clv_pct"] > 0

    def test_ml_home_worse_than_close_gives_negative_clv(self, db):
        # Published at 1.90, closed at 2.00 (price lengthened) -> we did worse than the close.
        _publish(db, odds_decimal=1.90)
        db.capture_closing_line(
            "778000:ML_HOME", closing_odds_decimal=2.00,
            closing_pin_home=2.00, closing_pin_away=1.95,
        )
        row = db.get_picks()[0]
        assert row["clv_pct"] < 0

    def test_ml_away_uses_pin_away_not_pin_home(self, db):
        _publish(db, pick_uid="778000:ML_AWAY", market="ML_AWAY", odds_decimal=2.10)
        db.capture_closing_line(
            "778000:ML_AWAY", closing_odds_decimal=2.00,
            closing_pin_home=1.80, closing_pin_away=2.00,
        )
        row = db.get_picks()[0]
        assert row["clv_pct"] == pytest.approx((2.10 / 2.00 - 1) * 100, abs=1e-3)

    def test_non_ml_market_stores_snapshot_but_no_clv(self, db):
        _publish(db, pick_uid="778000:OVER", market="OVER", odds_decimal=1.91)
        db.capture_closing_line(
            "778000:OVER", closing_odds_decimal=1.87,
            closing_pin_home=1.80, closing_pin_away=2.05,
        )
        row = db.get_picks()[0]
        assert row["closing_odds_decimal"] == 1.87
        assert row["clv_pct"] is None

    def test_capture_is_idempotent_second_call_noop(self, db):
        _publish(db)
        first = db.capture_closing_line(
            "778000:ML_HOME", closing_odds_decimal=1.90,
            closing_pin_home=1.90, closing_pin_away=2.05,
        )
        second = db.capture_closing_line(
            "778000:ML_HOME", closing_odds_decimal=1.50,
            closing_pin_home=1.50, closing_pin_away=2.50,
        )
        assert first is True
        assert second is False  # already captured, WHERE closing_captured_at IS NULL blocks it
        row = db.get_picks()[0]
        assert row["closing_odds_decimal"] == 1.90  # unchanged by the second call

    def test_capture_unknown_pick_uid_returns_false(self, db):
        assert db.capture_closing_line("does-not-exist", closing_odds_decimal=1.9) is False

    def test_get_picks_needing_closing_capture_filters_correctly(self, db):
        _publish(db, pick_uid="778000:ML_HOME", market="ML_HOME")
        _publish(db, pick_uid="778001:ML_HOME", market="ML_HOME")
        db.capture_closing_line(
            "778000:ML_HOME", closing_odds_decimal=1.90,
            closing_pin_home=1.90, closing_pin_away=2.05,
        )
        pending = db.get_picks_needing_closing_capture(sport="MLB")
        assert len(pending) == 1
        assert pending[0]["pick_uid"] == "778001:ML_HOME"

    def test_clv_stats_aggregate(self, db):
        _publish(db, pick_uid="778000:ML_HOME", market="ML_HOME", odds_decimal=2.00)
        _publish(db, pick_uid="778001:ML_HOME", market="ML_HOME", odds_decimal=1.80)
        db.capture_closing_line(  # positive CLV
            "778000:ML_HOME", closing_odds_decimal=1.90,
            closing_pin_home=1.90, closing_pin_away=2.05,
        )
        db.capture_closing_line(  # negative CLV (1.80 -> 1.95 is worse)
            "778001:ML_HOME", closing_odds_decimal=1.95,
            closing_pin_home=1.95, closing_pin_away=1.90,
        )
        stats = db.get_clv_stats(sport="MLB")
        assert stats["n"] == 2
        assert stats["n_positive"] == 1
        assert stats["pct_positive"] == 50.0
