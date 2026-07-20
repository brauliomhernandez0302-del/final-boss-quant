"""Tests for track_record/publisher.py's odds_book stamping
(docs/PROTOCOLO_CLV_V1.md cut (c), "por libro de O_taken").

odds_book is only ever set when the published odds_decimal is an EXACT
match against get_best_odds_for_teams()'s own ml_home_book/ml_away_book —
never guessed when the bet's own odds came from a different/unverified
source (this project's never-fabricate-a-plausible-value rule, see
FALL-001/FALL-002).
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


def _fake_result(odds: float):
    return {
        "status": "success",
        "probabilities": {"p_home": 0.55, "p_away": 0.45},
        "best_bets": [{
            "market": "ML_HOME", "probability": 0.55, "ev_pct": 5.0,
            "confidence_tier": "🔥 ULTRA VALUE", "odds": odds, "kelly_fraction": 0.02,
        }],
    }


def test_odds_book_stamped_on_exact_match(monkeypatch, tmp_path):
    monkeypatch.setattr(run_module_mod, "run_module", lambda **kw: _fake_result(1.85))
    monkeypatch.setattr(
        odds_fetcher_mod, "get_best_odds_for_teams",
        lambda **kw: {"ml_home": 1.85, "ml_home_book": "Pinnacle",
                       "ml_away": 2.05, "ml_away_book": "DraftKings"},
    )

    db = TrackRecordDB(db_path=tmp_path / "t.db")
    publish_mlb_picks(db, games=[_game(1)], dry_run=False)

    row = db.get_picks()[0]
    assert row["odds_book"] == "Pinnacle"


def test_odds_book_left_null_when_bet_odds_dont_match_market_odds(monkeypatch, tmp_path):
    # The pipeline's own odds (1.90) differ from what get_best_odds_for_teams
    # reports as the current best (1.85) — e.g. a stale snapshot inside the
    # pipeline. Must not guess a book for a price it didn't verify.
    monkeypatch.setattr(run_module_mod, "run_module", lambda **kw: _fake_result(1.90))
    monkeypatch.setattr(
        odds_fetcher_mod, "get_best_odds_for_teams",
        lambda **kw: {"ml_home": 1.85, "ml_home_book": "Pinnacle"},
    )

    db = TrackRecordDB(db_path=tmp_path / "t.db")
    publish_mlb_picks(db, games=[_game(1)], dry_run=False)

    row = db.get_picks()[0]
    assert row["odds_book"] is None


def test_odds_book_left_null_when_market_odds_has_no_book(monkeypatch, tmp_path):
    monkeypatch.setattr(run_module_mod, "run_module", lambda **kw: _fake_result(1.85))
    monkeypatch.setattr(
        odds_fetcher_mod, "get_best_odds_for_teams",
        lambda **kw: {"ml_home": 1.85},  # no ml_home_book key at all
    )

    db = TrackRecordDB(db_path=tmp_path / "t.db")
    publish_mlb_picks(db, games=[_game(1)], dry_run=False)

    row = db.get_picks()[0]
    assert row["odds_book"] is None
