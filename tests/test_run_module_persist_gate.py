"""Regression tests for run_module()'s `persist` gate (Fase 2A commit 1).

2026-07-19: found live that run_module() had no test/debug mode — every
call, for any purpose, wrote a permanent row to game_outcomes(source='live')
via LearningEngine.record_prediction(), and could also resolve OTHER
pending predictions' real scores via fetch_pending_outcomes(). A session of
diagnostic calls during this investigation wrote 46 real rows in one
afternoon (see audit_20260714/verificacion_operativa/nota_46_rows.md).

These are real end-to-end tests (real network calls, real pipeline, real
Monte Carlo) — the whole point is verifying that with persist=False, NO row
appears in game_outcomes, which can't be proven by mocking the write call
itself (that would just prove the mock was correctly wired, not that the
real function respects the flag). Writes are redirected to a scratch DB via
monkeypatching config.DATA_DIR (run_module() reads it via a local import),
never touching the real production database.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta

import config
import pytest


def _first_upcoming_game_pk() -> int:
    """A real, currently-scheduled MLB game_pk — today or tomorrow."""
    from data_fetchers import MLBStatsAPI

    api = MLBStatsAPI()
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    games = (api.get_todays_games(date=today) or []) + (api.get_todays_games(date=tomorrow) or [])
    if not games:
        pytest.skip("no real scheduled MLB games available to test against")
    return games[0]["game_pk"]


def _game_outcomes_row_count(db_path, game_pk: int) -> int:
    if not db_path.exists():
        return 0
    con = sqlite3.connect(db_path)
    try:
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_outcomes'"
        ).fetchall()
        if not tables:
            return 0
        return con.execute(
            "SELECT COUNT(*) FROM game_outcomes WHERE game_pk = ?", (game_pk,)
        ).fetchone()[0]
    finally:
        con.close()


@pytest.mark.integration
def test_persist_false_writes_no_row(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    from modules.baseball_module.core.run_module import run_module

    game_pk = _first_upcoming_game_pk()
    db_path = tmp_path / "predictions_history.db"

    run_module(game_id=game_pk, persist=False)

    assert _game_outcomes_row_count(db_path, game_pk) == 0


@pytest.mark.integration
def test_fbq_no_persist_env_overrides_persist_true(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setenv("FBQ_NO_PERSIST", "1")
    from modules.baseball_module.core.run_module import run_module

    game_pk = _first_upcoming_game_pk()
    db_path = tmp_path / "predictions_history.db"

    # persist=True explicitly, but the env var must win
    run_module(game_id=game_pk, persist=True)

    assert _game_outcomes_row_count(db_path, game_pk) == 0


@pytest.mark.integration
def test_persist_true_without_env_writes_a_row(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.delenv("FBQ_NO_PERSIST", raising=False)
    from modules.baseball_module.core.run_module import run_module

    game_pk = _first_upcoming_game_pk()
    db_path = tmp_path / "predictions_history.db"

    result = run_module(game_id=game_pk, persist=True)
    if result.get("status") != "success":
        pytest.skip(f"pipeline did not complete for this game (status={result.get('status')}), "
                    "can't assert the write side without a successful run")

    assert _game_outcomes_row_count(db_path, game_pk) == 1


def test_pin_backfill_never_touches_prediction_fields(tmp_path):
    """record_prediction()'s COALESCE backfill must only ever fill
    ml_home_pin/ml_away_pin/stage_factors_json/p_home_raw/p_away_raw/source
    on a pre-existing row — never overwrite p_home/p_away/lambda_home/
    lambda_away, which are immutable once a prediction is recorded."""
    from modules.baseball_module.calibration.learning_engine import LearningEngine

    db_path = tmp_path / "test_predictions.db"
    engine = LearningEngine(db_path=db_path)

    inserted = engine.record_prediction(
        game_pk=999001, game_date="2026-07-20", season=2026,
        home_team="Team A", away_team="Team B",
        lambda_home=4.5, lambda_away=4.2,
        p_home=0.55, p_away=0.45,
        p_home_raw=0.53, p_away_raw=0.47,
        ml_home_pin=None, ml_away_pin=None,
    )
    assert inserted is True

    # Re-run later in the day, now with pins available (and, hypothetically,
    # a different lambda/p_home if this were called again with fresh model
    # output) — the backfill must take the pins but leave everything else
    # from the first call untouched.
    second = engine.record_prediction(
        game_pk=999001, game_date="2026-07-20", season=2026,
        home_team="Team A", away_team="Team B",
        lambda_home=9.9, lambda_away=9.9,  # deliberately different — must NOT land
        p_home=0.99, p_away=0.01,          # deliberately different — must NOT land
        p_home_raw=0.98, p_away_raw=0.02,  # must NOT land (already non-NULL)
        ml_home_pin=1.85, ml_away_pin=2.05,
    )
    assert second is False  # not a new insert

    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT lambda_home, lambda_away, p_home, p_away, p_home_raw, p_away_raw, "
        "ml_home_pin, ml_away_pin FROM game_outcomes WHERE game_pk = 999001"
    ).fetchone()
    lambda_home, lambda_away, p_home, p_away, p_home_raw, p_away_raw, ml_home_pin, ml_away_pin = row

    assert lambda_home == 4.5 and lambda_away == 4.2  # untouched
    assert p_home == 0.55 and p_away == 0.45          # untouched
    assert p_home_raw == 0.53 and p_away_raw == 0.47  # untouched (already non-NULL)
    assert ml_home_pin == 1.85 and ml_away_pin == 2.05  # backfilled (was NULL)
