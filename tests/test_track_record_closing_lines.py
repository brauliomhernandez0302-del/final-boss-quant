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


def _publish(db, pick_uid="778000:ML_HOME", market="ML_HOME", odds_decimal=2.00, commence_time=None):
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
        commence_time=commence_time,
    )


class TestClosingLineCapture:

    def test_schema_migration_idempotent(self, db):
        # Re-init on the same DB must not raise (ALTER TABLE ADD COLUMN twice).
        TrackRecordDB(db_path=db.db_path)
        TrackRecordDB(db_path=db.db_path)

    def test_ml_home_beats_close_gives_positive_clv(self, db):
        # Published at 2.00, closed at 1.90 (price shortened) -> we beat the close.
        # El CLV se mide contra el precio JUSTO devigged, no contra el crudo
        # (2026-07-31): el crudo lleva el margen adentro y regalaba el vig
        # entero como si fuera habilidad. Acá el par 1.90/2.05 tiene overround
        # 1.0146, así que el justo es 1.90*1.0146 y el CLV real es ~1.5pp menor
        # que el 5.26% que daba la cuenta vieja — pero sigue siendo positivo,
        # porque este pick SÍ le ganó al cierre.
        _publish(db, odds_decimal=2.00)
        ok = db.capture_closing_line(
            "778000:ML_HOME", closing_odds_decimal=1.90,
            closing_pin_home=1.90, closing_pin_away=2.05,
        )
        assert ok
        row = db.get_picks()[0]
        justo = 1.90 * (1 / 1.90 + 1 / 2.05)
        assert row["clv_pct"] == pytest.approx((2.00 / justo - 1) * 100, abs=1e-3)
        assert row["clv_pct"] > 0
        assert row["clv_pct"] < (2.00 / 1.90 - 1) * 100, "el crudo sobreestimaba" 

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
        justo = 2.00 * (1 / 1.80 + 1 / 2.00)
        assert row["clv_pct"] == pytest.approx((2.10 / justo - 1) * 100, abs=1e-3)

    def test_non_ml_market_stores_snapshot_but_no_clv(self, db):
        _publish(db, pick_uid="778000:OVER", market="OVER", odds_decimal=1.91)
        db.capture_closing_line(
            "778000:OVER", closing_odds_decimal=1.87,
            closing_pin_home=1.80, closing_pin_away=2.05,
        )
        row = db.get_picks()[0]
        assert row["closing_odds_decimal"] == 1.87
        assert row["clv_pct"] is None

    def test_capture_unknown_pick_uid_returns_false(self, db):
        assert db.capture_closing_line("does-not-exist", closing_odds_decimal=1.9) is False

    def test_get_picks_needing_closing_capture_returns_not_yet_started_games(self, db):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        _publish(db, pick_uid="778000:ML_HOME", market="ML_HOME",
                 commence_time=(now + timedelta(hours=3)).isoformat())
        _publish(db, pick_uid="778001:ML_HOME", market="ML_HOME",
                 commence_time=(now - timedelta(minutes=5)).isoformat())  # already started

        pending = db.get_picks_needing_closing_capture(sport="MLB")
        assert {p["pick_uid"] for p in pending} == {"778000:ML_HOME"}

    def test_get_picks_needing_closing_capture_never_excludes_null_commence_time(self, db):
        # Rows with no commence_time (a sport/publisher that doesn't
        # populate it) must never be silently stranded by this query — the
        # caller (capture_closing_lines.py) decides to skip them with a
        # warning instead.
        _publish(db, pick_uid="778000:ML_HOME", market="ML_HOME", commence_time=None)
        pending = db.get_picks_needing_closing_capture(sport="MLB")
        assert {p["pick_uid"] for p in pending} == {"778000:ML_HOME"}

    def test_get_picks_needing_closing_capture_returns_already_captured_pre_start_picks(self, db):
        # "Last pre-start capture wins" means an already-captured pick must
        # still be returned (so a later sweep can overwrite it) as long as
        # its game hasn't started — this is the opposite of the old
        # closing_captured_at IS NULL one-shot design.
        from datetime import datetime, timedelta, timezone
        commence = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        _publish(db, pick_uid="778000:ML_HOME", market="ML_HOME", commence_time=commence)
        db.capture_closing_line(
            "778000:ML_HOME", closing_odds_decimal=1.90,
            closing_pin_home=1.90, closing_pin_away=2.05,
        )
        pending = db.get_picks_needing_closing_capture(sport="MLB")
        assert {p["pick_uid"] for p in pending} == {"778000:ML_HOME"}

    def test_later_pre_start_capture_overwrites_earlier(self, db):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        commence = (now + timedelta(hours=2)).isoformat()
        _publish(db, commence_time=commence)

        early = db.capture_closing_line(
            "778000:ML_HOME", closing_odds_decimal=2.10,
            closing_pin_home=2.10, closing_pin_away=1.80,
            captured_at=(now + timedelta(minutes=10)).isoformat(),
        )
        late = db.capture_closing_line(
            "778000:ML_HOME", closing_odds_decimal=1.90,
            closing_pin_home=1.90, closing_pin_away=2.05,
            captured_at=(now + timedelta(hours=1, minutes=50)).isoformat(),
        )
        assert early is True
        assert late is True  # NOT idempotent-blocked — the later pre-start capture wins
        row = db.get_picks()[0]
        assert row["closing_odds_decimal"] == 1.90  # the later value, not the earlier one
        # minutes_before_start reflects the LATEST capture's own staleness,
        # not the first one's.
        assert row["minutes_before_start"] == pytest.approx(10.0, abs=0.1)

    def test_capture_at_or_after_commence_time_is_rejected(self, db):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        commence = (now + timedelta(minutes=30)).isoformat()
        _publish(db, commence_time=commence)

        # Attempt to capture 1 minute AFTER the game already started.
        rejected = db.capture_closing_line(
            "778000:ML_HOME", closing_odds_decimal=1.90,
            closing_pin_home=1.90, closing_pin_away=2.05,
            captured_at=(now + timedelta(minutes=31)).isoformat(),
        )
        assert rejected is False
        row = db.get_picks()[0]
        assert row["closing_odds_decimal"] is None  # untouched
        assert row["clv_pct"] is None

    def test_capture_exactly_at_commence_time_is_rejected(self, db):
        commence = "2026-07-19T20:00:00+00:00"
        _publish(db, commence_time=commence)
        rejected = db.capture_closing_line(
            "778000:ML_HOME", closing_odds_decimal=1.90,
            closing_pin_home=1.90, closing_pin_away=2.05,
            captured_at=commence,  # exactly at commence_time, not after
        )
        assert rejected is False

    def test_capture_without_commence_time_is_never_rejected(self, db):
        # No commence_time on the pick at all — capture_closing_line() has
        # no basis to reject on timing, so it must proceed normally.
        _publish(db, commence_time=None)
        ok = db.capture_closing_line(
            "778000:ML_HOME", closing_odds_decimal=1.90,
            closing_pin_home=1.90, closing_pin_away=2.05,
        )
        assert ok is True
        row = db.get_picks()[0]
        assert row["closing_odds_decimal"] == 1.90
        assert row["minutes_before_start"] is None

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
