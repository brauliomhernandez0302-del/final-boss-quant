"""Tests for track_record/capture_closing_lines.py's orchestration function
(capture_closing_lines()) — distinct from tests/test_track_record_closing_lines.py,
which tests TrackRecordDB's methods directly. These exercise the sweep loop
itself: skipping picks with no commence_time (with a warning), and passing
commence_time through to get_best_odds_for_teams() so it can disambiguate a
multi-game series (Fase 2A commit 2/3)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import odds_fetcher
from track_record.capture_closing_lines import capture_closing_lines
from track_record.db import TrackRecordDB


def _publish(db, pick_uid="778000:ML_HOME", market="ML_HOME", commence_time=None):
    db.publish_pick(
        pick_uid=pick_uid, game_date="2026-07-19", sport="MLB", game_pk=778000,
        home_team="New York Yankees", away_team="Boston Red Sox",
        market=market, model_prob=0.55, ev_pct=0.05, odds_decimal=2.0,
        commence_time=commence_time,
    )


def test_pick_without_commence_time_is_skipped_with_warning(tmp_path, caplog):
    db = TrackRecordDB(db_path=tmp_path / "t.db")
    _publish(db, commence_time=None)

    summary = capture_closing_lines(db, sport="MLB")

    assert summary["skipped_no_commence_time"] == 1
    assert summary["captured"] == 0
    assert any("no commence_time" in rec.message for rec in caplog.records)


def test_commence_time_is_passed_through_to_odds_lookup(monkeypatch, tmp_path):
    db = TrackRecordDB(db_path=tmp_path / "t.db")
    commence = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    _publish(db, commence_time=commence)

    seen_kwargs = {}

    def fake_get_best_odds_for_teams(**kwargs):
        seen_kwargs.update(kwargs)
        return {"pin_home": 1.85, "pin_away": 2.05, "ml_home": 1.85, "ml_away": 2.05}

    monkeypatch.setattr(odds_fetcher, "get_best_odds_for_teams", fake_get_best_odds_for_teams)

    summary = capture_closing_lines(db, sport="MLB")

    assert seen_kwargs.get("commence_time") == commence
    assert summary["captured"] == 1


def test_pick_with_commence_time_but_no_market_match_is_not_captured(monkeypatch, tmp_path):
    db = TrackRecordDB(db_path=tmp_path / "t.db")
    commence = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    _publish(db, commence_time=commence)

    monkeypatch.setattr(odds_fetcher, "get_best_odds_for_teams", lambda **kw: {})

    summary = capture_closing_lines(db, sport="MLB")

    assert summary["no_market_data"] == 1
    assert summary["captured"] == 0
