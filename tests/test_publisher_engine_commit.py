"""Tests for docs/PROTOCOLO_CLV_V1.md's engine_commit stamping.

Every published pick must record which prediction-engine commit it was
made under (the protocol's engine-freeze validity condition needs this to
detect a mid-window change) — cached so a batch publish run does one git
subprocess call, not one per pick.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import modules.baseball_module.core.run_module as run_module_mod
import odds_fetcher as odds_fetcher_mod
from track_record.db import TrackRecordDB
from track_record.publisher import _get_engine_commit, publish_mlb_picks


def _game(game_pk: int) -> dict:
    commence = datetime.now(timezone.utc) + timedelta(hours=3)
    return {
        "game_pk": game_pk,
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "game_date": commence.isoformat(),
    }


def _fake_result():
    return {
        "status": "success",
        "probabilities": {"p_home": 0.55, "p_away": 0.45},
        "best_bets": [{
            "market": "ML_HOME", "probability": 0.55, "ev_pct": 5.0,
            "confidence_tier": "🔥 ULTRA VALUE", "odds": 1.85, "kelly_fraction": 0.02,
        }],
    }


def test_get_engine_commit_returns_a_real_git_hash():
    _get_engine_commit.cache_clear()
    commit = _get_engine_commit()
    assert commit is not None
    assert len(commit) == 40  # full git SHA
    assert all(c in "0123456789abcdef" for c in commit)


def test_get_engine_commit_is_cached_across_calls(monkeypatch):
    _get_engine_commit.cache_clear()
    calls = []
    real_run = __import__("subprocess").run

    def counting_run(*args, **kwargs):
        calls.append(1)
        return real_run(*args, **kwargs)

    monkeypatch.setattr("track_record.publisher.subprocess.run", counting_run)
    _get_engine_commit()
    _get_engine_commit()
    _get_engine_commit()
    assert len(calls) == 1


def test_published_picks_are_stamped_with_engine_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(run_module_mod, "run_module", lambda **kw: _fake_result())
    monkeypatch.setattr(odds_fetcher_mod, "get_best_odds_for_teams", lambda **kw: {})
    _get_engine_commit.cache_clear()

    db = TrackRecordDB(db_path=tmp_path / "t.db")
    publish_mlb_picks(db, games=[_game(1)], dry_run=False)

    row = db.get_picks()[0]
    assert row["engine_commit"] == _get_engine_commit()
