"""Tests for track_record/publisher.py stamping the signed runline_point
onto RL_HOME/RL_AWAY picks (needed by reconciler.py's post-game grading —
see tests/test_reconciler_runline.py and tests/test_odds_fetcher_runline_sign.py
for the rest of this fix)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import modules.baseball_module.core.run_module as run_module_mod
import odds_fetcher as odds_fetcher_mod
from track_record.db import TrackRecordDB
from track_record.publisher import publish_mlb_picks


def _game(game_pk: int = 1) -> dict:
    commence = datetime.now(timezone.utc) + timedelta(hours=3)
    return {
        "game_pk": game_pk,
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "game_date": commence.isoformat(),
    }


def _fake_result(market: str, odds: float = 1.91):
    return {
        "status": "success",
        "probabilities": {"p_home": 0.55, "p_away": 0.45},
        "best_bets": [{
            "market": market, "probability": 0.55, "ev_pct": 5.0,
            "confidence_tier": "🔥 ULTRA VALUE", "odds": odds, "kelly_fraction": 0.02,
        }],
    }


def test_rl_home_pick_gets_home_runline_point(monkeypatch, tmp_path):
    monkeypatch.setattr(run_module_mod, "run_module", lambda **kw: _fake_result("RL_HOME"))
    monkeypatch.setattr(
        odds_fetcher_mod, "get_best_odds_for_teams",
        lambda **kw: {"runline_home_point": -1.5, "runline_away_point": 1.5},
    )

    db = TrackRecordDB(db_path=tmp_path / "t.db")
    publish_mlb_picks(db, games=[_game(1)], dry_run=False)

    row = db.get_picks()[0]
    assert row["runline_point"] == -1.5


def test_rl_away_pick_gets_away_runline_point(monkeypatch, tmp_path):
    monkeypatch.setattr(run_module_mod, "run_module", lambda **kw: _fake_result("RL_AWAY"))
    monkeypatch.setattr(
        odds_fetcher_mod, "get_best_odds_for_teams",
        lambda **kw: {"runline_home_point": -1.5, "runline_away_point": 1.5},
    )

    db = TrackRecordDB(db_path=tmp_path / "t.db")
    publish_mlb_picks(db, games=[_game(1)], dry_run=False)

    row = db.get_picks()[0]
    assert row["runline_point"] == 1.5


def test_moneyline_pick_has_null_runline_point(monkeypatch, tmp_path):
    monkeypatch.setattr(run_module_mod, "run_module", lambda **kw: _fake_result("ML_HOME"))
    monkeypatch.setattr(
        odds_fetcher_mod, "get_best_odds_for_teams",
        lambda **kw: {"runline_home_point": -1.5, "runline_away_point": 1.5},
    )

    db = TrackRecordDB(db_path=tmp_path / "t.db")
    publish_mlb_picks(db, games=[_game(1)], dry_run=False)

    row = db.get_picks()[0]
    assert row["runline_point"] is None
