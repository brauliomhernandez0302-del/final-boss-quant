"""Tests for track_record/publisher.py's decision_prob stamping
(docs/PROTOCOLO_CLV_V1.md auditability gap): the pick is decided on the
Platt-2D-corrected probability (core/value_detector.py's per-market
correction, applied only when fair_source=="pinnacle"), but that value was
never persisted anywhere on its own — only game_outcomes.p_home (Platt-1D)
was. decision_prob is that value, read straight off the same best_bets bet
dict ev_pct comes from, and must never be left NULL.
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


def _fake_result(probability: float):
    # 'probability' here stands in for whatever core/value_detector.py's
    # analyze_market_generic() already put in the bet dict — Platt-2D-
    # corrected when fair_source=="pinnacle", raw Platt-1D otherwise. The
    # publisher never re-derives Platt math itself; it only reads this field.
    return {
        "status": "success",
        "probabilities": {"p_home": 0.55, "p_away": 0.45},
        "best_bets": [{
            "market": "ML_HOME", "probability": probability, "ev_pct": 5.0,
            "confidence_tier": "🔥 ULTRA VALUE", "odds": 1.85, "kelly_fraction": 0.02,
        }],
    }


def test_decision_prob_persists_platt2d_value_when_pinnacle_available(monkeypatch, tmp_path):
    # 0.58 stands in for a Platt-2D-corrected probability that differs from
    # the raw pipeline p_home (0.55) — proves decision_prob isn't silently
    # reverting to the uncorrected value.
    monkeypatch.setattr(run_module_mod, "run_module", lambda **kw: _fake_result(0.58))
    monkeypatch.setattr(odds_fetcher_mod, "get_best_odds_for_teams", lambda **kw: {})

    db = TrackRecordDB(db_path=tmp_path / "t.db")
    publish_mlb_picks(db, games=[_game(1)], dry_run=False)

    row = db.get_picks()[0]
    assert row["decision_prob"] == 0.58
    assert row["decision_prob"] is not None


def test_decision_prob_persists_platt1d_value_when_no_pinnacle_line(monkeypatch, tmp_path):
    # No Pinnacle fair line to correct against — value_detector never applies
    # Platt-2D, so the bet dict's 'probability' IS the Platt-1D value. Must
    # still be stamped, never NULL.
    monkeypatch.setattr(run_module_mod, "run_module", lambda **kw: _fake_result(0.55))
    monkeypatch.setattr(odds_fetcher_mod, "get_best_odds_for_teams", lambda **kw: {})

    db = TrackRecordDB(db_path=tmp_path / "t.db")
    publish_mlb_picks(db, games=[_game(1)], dry_run=False)

    row = db.get_picks()[0]
    assert row["decision_prob"] == 0.55
    assert row["decision_prob"] is not None


def test_decision_prob_never_null_even_without_pipeline_probability_key(monkeypatch, tmp_path):
    # Synthesised fallback pick path (best_bets empty, minimal ML pick built
    # from market odds) doesn't set a 'probability' key at all — model_prob
    # falls back to top-level p_home, and decision_prob must follow it rather
    # than landing NULL.
    def _fake_no_best_bets(**kw):
        return {
            "status": "success",
            "probabilities": {"p_home": 0.6, "p_away": 0.4},
            "best_bets": [],
        }

    monkeypatch.setattr(run_module_mod, "run_module", _fake_no_best_bets)
    monkeypatch.setattr(
        odds_fetcher_mod, "get_best_odds_for_teams",
        lambda **kw: {"ml_home": 1.85},
    )

    db = TrackRecordDB(db_path=tmp_path / "t.db")
    publish_mlb_picks(db, games=[_game(1)], dry_run=False)

    rows = db.get_picks()
    assert len(rows) == 1
    assert rows[0]["decision_prob"] is not None
    assert rows[0]["decision_prob"] == rows[0]["model_prob"]
