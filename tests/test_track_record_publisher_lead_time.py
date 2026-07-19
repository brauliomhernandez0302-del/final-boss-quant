"""Regression test for track_record/publisher.py's pre-game lead-time gate.

2026-07-19: found live (while dry-running run_daily_picks.py before its
first cron install) that MIN_LEAD_MINUTES's skip check was silently
NEVER firing for real MLB games — commence_raw read game.get("commence_time")
or game.get("game_datetime"), keys data_fetchers.py's _parse_game() never
populates (only "game_date"). This undermined the module's own core
guarantee ("every pick has a pre-game published_at timestamp", per its
docstring) for any game close to or even past first pitch. Fixed by adding
game.get("game_date") to the fallback chain. This test exercises
publish_mlb_picks() directly (it accepts a `games` list, bypassing the real
schedule fetch) so the gate is verified without any network access.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import modules.baseball_module.core.run_module as run_module_mod
import odds_fetcher as odds_fetcher_mod
from track_record.db import TrackRecordDB
from track_record.publisher import MIN_LEAD_MINUTES, publish_mlb_picks


def _game(minutes_until_first_pitch: float, game_pk: int = 1) -> dict:
    commence = datetime.now(timezone.utc) + timedelta(minutes=minutes_until_first_pitch)
    return {
        "game_pk": game_pk,
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        # Real MLB game dicts (data_fetchers.py::_parse_game()) only ever
        # populate "game_date", not "commence_time"/"game_datetime" — using
        # that same key here is what makes this test exercise the real bug.
        "game_date": commence.isoformat(),
    }


def _patch_pipeline(monkeypatch, run_calls: list):
    def fake_run_module(**kwargs):
        run_calls.append(kwargs.get("game_id"))
        return {"status": "success", "probabilities": {}, "best_bets": []}

    monkeypatch.setattr(run_module_mod, "run_module", fake_run_module)
    monkeypatch.setattr(odds_fetcher_mod, "get_best_odds_for_teams", lambda **kw: {})


def test_game_starting_too_soon_is_skipped(monkeypatch, tmp_path):
    db = TrackRecordDB(db_path=tmp_path / "t.db")
    run_calls: list = []
    _patch_pipeline(monkeypatch, run_calls)

    game = _game(minutes_until_first_pitch=MIN_LEAD_MINUTES / 2)
    publish_mlb_picks(db, games=[game], dry_run=True)

    assert run_calls == [], (
        "a game starting in under MIN_LEAD_MINUTES must be skipped before "
        "ever reaching the prediction pipeline"
    )


def test_game_far_enough_out_is_analyzed(monkeypatch, tmp_path):
    db = TrackRecordDB(db_path=tmp_path / "t.db")
    run_calls: list = []
    _patch_pipeline(monkeypatch, run_calls)

    game = _game(minutes_until_first_pitch=MIN_LEAD_MINUTES * 4)
    publish_mlb_picks(db, games=[game], dry_run=True)

    assert run_calls == [game["game_pk"]]


def test_game_already_started_is_skipped(monkeypatch, tmp_path):
    db = TrackRecordDB(db_path=tmp_path / "t.db")
    run_calls: list = []
    _patch_pipeline(monkeypatch, run_calls)

    game = _game(minutes_until_first_pitch=-30)
    publish_mlb_picks(db, games=[game], dry_run=True)

    assert run_calls == []
