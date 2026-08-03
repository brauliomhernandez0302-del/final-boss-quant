"""Regression test: track_record/publisher.py's `dry_run` must actually
propagate to run_module()'s `persist` flag.

2026-07-19 (Fase 2A commit 4): found live, while running the first real
(non-dry-run) publish after commit 1 shipped run_module(persist=...), that
publish_mlb_picks() never passed `persist` through to run_mlb() at all — it
always used the default persist=True regardless of the publisher's own
dry_run flag. A dry-run at the publisher level (which only skips
db.publish_pick()) still silently wrote real rows to game_outcomes
(source='live') through the unpatched run_mlb() call underneath it —
re-contaminating the ledger with a fresh dry-run batch after commit 1 was
already supposed to have closed this exact hole. This test exercises the
real wiring (not a fake that ignores its own kwargs) so a future edit that
drops `persist=not dry_run` again fails immediately.
"""
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


def _patch_pipeline(monkeypatch, seen_kwargs: dict):
    def fake_run_module(**kwargs):
        seen_kwargs.update(kwargs)
        return {"status": "success", "probabilities": {}, "best_bets": []}

    monkeypatch.setattr(run_module_mod, "run_module", fake_run_module)
    monkeypatch.setattr(odds_fetcher_mod, "get_best_odds_for_teams", lambda **kw: {})


def test_dry_run_true_passes_persist_false(monkeypatch, tmp_path):
    db = TrackRecordDB(db_path=tmp_path / "t.db")
    seen: dict = {}
    _patch_pipeline(monkeypatch, seen)

    publish_mlb_picks(db, games=[_game()], dry_run=True)

    assert seen.get("persist") is False, (
        "publish_mlb_picks(dry_run=True) must call run_mlb(persist=False) — "
        "otherwise a 'dry run' still writes real rows to the live "
        "calibration ledger through the unpatched pipeline call"
    )


def test_dry_run_false_passes_persist_true(monkeypatch, tmp_path):
    db = TrackRecordDB(db_path=tmp_path / "t.db")
    seen: dict = {}
    _patch_pipeline(monkeypatch, seen)

    publish_mlb_picks(db, games=[_game()], dry_run=False)

    assert seen.get("persist") is True
