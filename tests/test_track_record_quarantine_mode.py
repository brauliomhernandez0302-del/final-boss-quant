"""Tests for Fase 2A commit 4's quarantine mode: every pick publishes
tagged 'quarantine' and bypasses the MIN_TIER filter while
config.QUARANTINE_MODE is true (the default) — the public band is Fase 2C's
job, fixed against a re-measured baseline, not decided by this commit."""
from __future__ import annotations

import config
import modules.baseball_module.core.run_module as run_module_mod
import odds_fetcher as odds_fetcher_mod
from track_record.db import TrackRecordDB
from track_record.publisher import publish_mlb_picks


def _game(game_pk: int = 1) -> dict:
    from datetime import datetime, timedelta, timezone
    commence = datetime.now(timezone.utc) + timedelta(hours=3)
    return {
        "game_pk": game_pk,
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "game_date": commence.isoformat(),
    }


def _fake_result_with_low_tier_bet():
    return {
        "status": "success",
        "probabilities": {"p_home": 0.55, "p_away": 0.45},
        "best_bets": [{
            "market": "ML_HOME",
            "probability": 0.55,
            "ev_pct": 0.5,          # below any real tier threshold
            "confidence_tier": "",  # unranked — _tier_rank() returns 0
            "odds": 1.85,
            "kelly_fraction": 0.01,
        }],
    }


def _patch_pipeline(monkeypatch):
    monkeypatch.setattr(run_module_mod, "run_module", lambda **kw: _fake_result_with_low_tier_bet())
    monkeypatch.setattr(odds_fetcher_mod, "get_best_odds_for_teams", lambda **kw: {})


def test_low_tier_bet_publishes_in_quarantine_mode(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(config, "QUARANTINE_MODE", True)
    db = TrackRecordDB(db_path=tmp_path / "t.db")

    published = publish_mlb_picks(db, games=[_game()], dry_run=False)

    assert len(published) == 1
    row = db.get_picks()[0]
    assert row["publish_mode"] == "quarantine"


def test_low_tier_bet_filtered_when_quarantine_mode_off(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(config, "QUARANTINE_MODE", False)
    db = TrackRecordDB(db_path=tmp_path / "t.db")

    published = publish_mlb_picks(db, games=[_game()], dry_run=False)

    assert published == []
    assert db.get_picks() == []


def test_publish_pick_defaults_to_quarantine_mode():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        db = TrackRecordDB(db_path=Path(tmp) / "t.db")
        db.publish_pick(
            pick_uid="1:ML_HOME", game_date="2026-07-19", sport="MLB", game_pk=1,
            home_team="A", away_team="B", market="ML_HOME",
            model_prob=0.55, ev_pct=5.0,
        )
        row = db.get_picks()[0]
        assert row["publish_mode"] == "quarantine"
