"""Tests for track_record/reconciler.py's run-line (RL_HOME/RL_AWAY) grading.

Found during a run-line/totals detection review (2026-07-20): _resolve_market
hardcoded "home is always favored by -1.5" (runline defaulted to a bare
magnitude, never even threaded through from a real pick), AND its RL_AWAY
formula (`cover = -diff - runline`) had an independent sign error that
flipped WIN/LOSS in the diff∈{0,1} region even under that same flawed
assumption. Fixed by storing a SIGNED runline_point per pick (the actual
line for whichever side that pick is on) and using the general formula
cover = (that side's own run differential) + point > 0 — no favorite
assumption needed at all. No reconciler tests existed before this file.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from track_record.db import TrackRecordDB
from track_record.reconciler import _resolve_market, reconcile_pending


# ------------------------------------------------------- _resolve_market unit

def test_rl_home_favored_covers():
    # Home favored -1.5, wins by 2 -> covers.
    assert _resolve_market("RL_HOME", 5, 3, runline_point=-1.5) == "WIN"


def test_rl_home_favored_fails_to_cover():
    # Home favored -1.5, wins by only 1 -> does not cover.
    assert _resolve_market("RL_HOME", 4, 3, runline_point=-1.5) == "LOSS"


def test_rl_home_underdog_covers_on_close_loss():
    # Home is the UNDERDOG (+1.5) -- loses by 1, still covers the spread.
    assert _resolve_market("RL_HOME", 3, 4, runline_point=1.5) == "WIN"


def test_rl_home_underdog_fails_on_big_loss():
    # Home +1.5, loses by 3 -> fails to cover.
    assert _resolve_market("RL_HOME", 2, 5, runline_point=1.5) == "LOSS"


def test_rl_away_underdog_covers_on_close_home_win():
    # This is the exact regression case: away getting +1.5, home wins by
    # only 1 (diff=1). Away should cover (WIN) -- the old formula
    # (`-diff - runline`) incorrectly returned LOSS here.
    assert _resolve_market("RL_AWAY", 4, 3, runline_point=1.5) == "WIN"


def test_rl_away_underdog_fails_when_home_covers():
    # Away +1.5, home wins by 2 -> away fails to cover.
    assert _resolve_market("RL_AWAY", 5, 3, runline_point=1.5) == "LOSS"


def test_rl_away_favored_covers():
    # Away is the FAVORITE (-1.5) -- wins by 2 -> covers.
    assert _resolve_market("RL_AWAY", 3, 5, runline_point=-1.5) == "WIN"


def test_rl_away_favored_fails_to_cover():
    # Away favored -1.5, wins by only 1 -> fails to cover.
    assert _resolve_market("RL_AWAY", 3, 4, runline_point=-1.5) == "LOSS"


def test_rl_home_push_on_exact_integer_line():
    # An alternate whole-number line (-1) landing exactly -> PUSH.
    assert _resolve_market("RL_HOME", 5, 4, runline_point=-1.0) == "PUSH"


def test_rl_away_push_on_exact_integer_line():
    assert _resolve_market("RL_AWAY", 4, 5, runline_point=-1.0) == "PUSH"


def test_rl_home_void_without_runline_point():
    # Never guess a line for a pick that doesn't have one recorded (e.g.
    # published before this column existed).
    assert _resolve_market("RL_HOME", 5, 3, runline_point=None) == "VOID"


def test_rl_away_void_without_runline_point():
    assert _resolve_market("RL_AWAY", 5, 3, runline_point=None) == "VOID"


# ------------------------------------------------------- reconcile_pending wiring

def _publish_rl_pick(db, market, runline_point, game_pk=900001):
    # reconcile_pending() compares game_date against datetime.now() (naive
    # local), not UTC — match that here so the pick reliably falls into the
    # "game already played" window regardless of the local/UTC offset.
    game_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    db.publish_pick(
        pick_uid=f"{game_pk}:{market}", game_date=game_date, sport="MLB",
        game_pk=game_pk, home_team="New York Yankees", away_team="Boston Red Sox",
        market=market, model_prob=0.55, ev_pct=5.0, odds_decimal=1.91,
        runline_point=runline_point,
    )


def test_reconcile_pending_grades_rl_away_regression_case_correctly(monkeypatch, tmp_path):
    db = TrackRecordDB(db_path=tmp_path / "t.db")
    _publish_rl_pick(db, "RL_AWAY", runline_point=1.5, game_pk=900001)

    import track_record.reconciler as reconciler_mod
    monkeypatch.setattr(reconciler_mod, "_get_final_score", lambda sport, pk: (4, 3))  # home wins by 1

    stats = reconcile_pending(db, lookback_days=7, sport="MLB")
    assert stats["resolved"] == 1

    row = db.get_picks(sport="MLB")[0]
    assert row["result"] == "WIN"


def test_reconcile_pending_voids_rl_pick_with_no_runline_point(monkeypatch, tmp_path):
    db = TrackRecordDB(db_path=tmp_path / "t.db")
    _publish_rl_pick(db, "RL_HOME", runline_point=None, game_pk=900002)

    import track_record.reconciler as reconciler_mod
    monkeypatch.setattr(reconciler_mod, "_get_final_score", lambda sport, pk: (5, 3))

    stats = reconcile_pending(db, lookback_days=7, sport="MLB")
    assert stats["voided"] == 1

    row = db.get_picks(sport="MLB")[0]
    assert row["result"] == "VOID"
