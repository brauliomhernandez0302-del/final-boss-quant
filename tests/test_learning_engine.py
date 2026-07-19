"""
Tests for the LearningEngine bias computation and clamp.

The engine computes mean(actual_runs / predicted_lambda) per team and
clamps the result to [1 − _BIAS_CLAMP, 1 + _BIAS_CLAMP] = [0.70, 1.30].

Key invariants:
  - Fewer than _MIN_SAMPLES finished games → returns 1.0 (neutral)
  - Perfect predictions (actual = lambda) → bias = 1.0
  - Systematic under-prediction → bias > 1.0
  - Systematic over-prediction  → bias < 1.0
  - Raw bias of 1.50 is clamped to 1.30
  - Raw bias of 0.50 is clamped to 0.70
  - record_prediction is idempotent (same game_pk inserted twice → no duplicate)
  - update_outcome fills actual runs and home_won correctly
"""
import pytest
from pathlib import Path
from modules.baseball_module.calibration.learning_engine import (
    LearningEngine, _BIAS_CLAMP, _MIN_SAMPLES,
)


@pytest.fixture
def engine(tmp_path):
    """Fresh in-memory-equivalent engine backed by a temp file."""
    db = tmp_path / "test_predictions.db"
    return LearningEngine(db_path=db)


def _insert_games(engine, team, season, actual_over_lambda_ratios, lambda_val=4.5):
    """
    Helper: insert finished games where actual = ratio × lambda,
    for a known list of ratios (all as home team for simplicity).
    """
    for i, ratio in enumerate(actual_over_lambda_ratios):
        actual = round(ratio * lambda_val)
        engine.record_prediction(
            game_pk=1000 + i,
            game_date=f"{season}-05-01",
            season=season,
            home_team=team,
            away_team="Opponent",
            lambda_home=lambda_val,
            lambda_away=3.8,
            p_home=0.55,
            p_away=0.45,
        )
        engine.update_outcome(1000 + i, actual_home_runs=actual, actual_away_runs=3)


class TestBiasNeutral:

    def test_below_min_samples_returns_one(self, engine):
        # Insert fewer than _MIN_SAMPLES games → neutral bias
        _insert_games(engine, "Yankees", 2026, [1.0] * (_MIN_SAMPLES - 1))
        bias = engine.compute_team_bias("Yankees", 2026)
        assert bias == 1.0

    def test_exactly_min_samples_applies_bias(self, engine):
        # Exactly _MIN_SAMPLES games with a consistent ratio
        _insert_games(engine, "RedSox", 2026, [1.10] * _MIN_SAMPLES)
        bias = engine.compute_team_bias("RedSox", 2026)
        assert bias != 1.0  # should now compute a real bias

    def test_perfect_calibration_is_one(self, engine):
        # 2026-07-11: compute_team_bias now applies untruncate_home_runs()
        # to actual_home_runs before dividing by lambda_home (walk-off
        # truncation correction — see learning_engine.py's constant
        # docstring). A raw box-score ratio of exactly 1.0 (actual==lambda)
        # is no longer "neutral" — it now means the team scored MORE than
        # the latent rate would predict once truncation is undone. True
        # neutral input for this test is a raw ratio equal to the
        # truncation factor itself, which untruncate_home_runs() maps back
        # to exactly 1.0. actual_home_runs must be an integer (box scores
        # are), so pick lambda=30.0 and ratio=29/30 — round(29/30 * 30.0)
        # == 29 exactly, no rounding slop, and 29 ≈ 0.967 × 30.
        _insert_games(
            engine, "Dodgers", 2026,
            [29.0 / 30.0] * _MIN_SAMPLES, lambda_val=30.0,
        )
        bias = engine.compute_team_bias("Dodgers", 2026)
        assert bias == pytest.approx(1.0, abs=0.01)

    def test_unknown_team_returns_one(self, engine):
        # Team with zero games → neutral
        bias = engine.compute_team_bias("NoTeam", 2026)
        assert bias == 1.0

    def test_wrong_season_returns_one(self, engine):
        _insert_games(engine, "Cubs", 2025, [1.20] * _MIN_SAMPLES)
        bias = engine.compute_team_bias("Cubs", 2026)
        assert bias == 1.0


class TestBiasDirection:

    def test_under_prediction_bias_above_one(self, engine):
        # Model consistently predicts 4.5 but team scores 6.0 → ratio = 1.33
        _insert_games(engine, "Giants", 2026, [1.33] * _MIN_SAMPLES)
        bias = engine.compute_team_bias("Giants", 2026)
        assert bias > 1.0

    def test_over_prediction_bias_below_one(self, engine):
        # Model consistently predicts 4.5 but team scores 3.0 → ratio = 0.67
        _insert_games(engine, "Padres", 2026, [0.67] * _MIN_SAMPLES)
        bias = engine.compute_team_bias("Padres", 2026)
        assert bias < 1.0

    def test_bias_is_mean_of_ratios(self, engine):
        ratios = [0.80, 1.00, 1.20, 1.10, 0.90,
                  1.05, 0.95, 1.15, 1.00, 0.85]
        assert len(ratios) >= _MIN_SAMPLES
        lambda_val = 4.5
        for i, ratio in enumerate(ratios):
            actual = ratio * lambda_val
            engine.record_prediction(
                game_pk=2000 + i, game_date="2026-06-01", season=2026,
                home_team="Mets", away_team="Braves",
                lambda_home=lambda_val, lambda_away=3.8,
                p_home=0.55, p_away=0.45,
            )
            engine.update_outcome(2000 + i, actual_home_runs=round(actual), actual_away_runs=3)

        bias = engine.compute_team_bias("Mets", 2026)
        raw_mean = sum(round(r * lambda_val) / lambda_val for r in ratios) / len(ratios)
        clamped = max(1.0 - _BIAS_CLAMP, min(1.0 + _BIAS_CLAMP, raw_mean))
        assert bias == pytest.approx(clamped, abs=0.05)


class TestBiasClamp:

    def test_extreme_over_prediction_clamped_at_0_70(self, engine):
        # actual always 1, lambda always 4.5 → ratio ≈ 0.22 → clamped to 0.70
        for i in range(_MIN_SAMPLES):
            engine.record_prediction(
                game_pk=3000 + i, game_date="2026-04-01", season=2026,
                home_team="LowScorers", away_team="Opponent",
                lambda_home=4.5, lambda_away=3.8,
                p_home=0.55, p_away=0.45,
            )
            engine.update_outcome(3000 + i, actual_home_runs=1, actual_away_runs=3)

        bias = engine.compute_team_bias("LowScorers", 2026)
        assert bias == pytest.approx(1.0 - _BIAS_CLAMP)

    def test_extreme_under_prediction_clamped_at_1_30(self, engine):
        # actual always 20, lambda = 4.5 → ratio ≈ 4.44 → clamped to 1.30
        for i in range(_MIN_SAMPLES):
            engine.record_prediction(
                game_pk=4000 + i, game_date="2026-04-01", season=2026,
                home_team="HighScorers", away_team="Opponent",
                lambda_home=4.5, lambda_away=3.8,
                p_home=0.55, p_away=0.45,
            )
            engine.update_outcome(4000 + i, actual_home_runs=20, actual_away_runs=3)

        bias = engine.compute_team_bias("HighScorers", 2026)
        assert bias == pytest.approx(1.0 + _BIAS_CLAMP)

    def test_clamp_constants(self):
        # Verify the constants themselves match what the code promises.
        assert _BIAS_CLAMP == 0.30
        lower = 1.0 - _BIAS_CLAMP
        upper = 1.0 + _BIAS_CLAMP
        assert lower == pytest.approx(0.70)
        assert upper == pytest.approx(1.30)


class TestPersistence:

    def test_record_prediction_idempotent(self, engine):
        # Inserting the same game_pk twice should not create a duplicate.
        engine.record_prediction(
            game_pk=9999, game_date="2026-05-01", season=2026,
            home_team="Twins", away_team="Tigers",
            lambda_home=4.5, lambda_away=3.8,
            p_home=0.55, p_away=0.45,
        )
        engine.record_prediction(
            game_pk=9999, game_date="2026-05-01", season=2026,
            home_team="Twins", away_team="Tigers",
            lambda_home=5.0, lambda_away=3.2,  # different values
            p_home=0.60, p_away=0.40,
        )
        # No crash; only one row should exist (IGNORE on conflict)
        with engine._get_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM game_outcomes WHERE game_pk=9999"
            ).fetchone()[0]
        assert count == 1

    def test_update_outcome_sets_home_won(self, engine):
        engine.record_prediction(
            game_pk=8888, game_date="2026-05-01", season=2026,
            home_team="A", away_team="B",
            lambda_home=4.5, lambda_away=3.8,
            p_home=0.55, p_away=0.45,
        )
        engine.update_outcome(8888, actual_home_runs=5, actual_away_runs=3)
        with engine._get_conn() as conn:
            row = conn.execute(
                "SELECT home_won, actual_home_runs, actual_away_runs "
                "FROM game_outcomes WHERE game_pk=8888"
            ).fetchone()
        assert row["home_won"] == 1
        assert row["actual_home_runs"] == 5
        assert row["actual_away_runs"] == 3


class TestOfficialDate:
    """Fase 2B commit B1 (audit_20260714/fase2b/): official_date is the MLB
    schedule's own day, distinct from game_date's raw UTC timestamp (which
    is one day ahead for any night game crossing midnight UTC). Every PIT
    walk-forward cutoff and bias-window comparison should key off this
    field going forward — these tests just confirm record_prediction()
    stores and backfills it correctly; the actual cutoff cutover is Fase 2B
    commit B2, not this one."""

    def test_record_prediction_stores_official_date(self, engine):
        engine.record_prediction(
            game_pk=7001, game_date="2026-07-19", season=2026,
            home_team="A", away_team="B",
            lambda_home=4.5, lambda_away=3.8,
            p_home=0.55, p_away=0.45,
            official_date="2026-07-18",  # e.g. a night game crossing midnight UTC
        )
        with engine._get_conn() as conn:
            row = conn.execute(
                "SELECT game_date, official_date FROM game_outcomes WHERE game_pk=7001"
            ).fetchone()
        assert row["game_date"] == "2026-07-19"
        assert row["official_date"] == "2026-07-18"

    def test_official_date_backfills_without_touching_prediction_fields(self, engine):
        engine.record_prediction(
            game_pk=7002, game_date="2026-07-19", season=2026,
            home_team="A", away_team="B",
            lambda_home=4.5, lambda_away=3.8,
            p_home=0.55, p_away=0.45,
            official_date=None,
        )
        # Second call (e.g. a later same-day re-run) now has official_date —
        # backfilled via COALESCE, same pattern as ml_home_pin.
        engine.record_prediction(
            game_pk=7002, game_date="2026-07-19", season=2026,
            home_team="A", away_team="B",
            lambda_home=9.9, lambda_away=9.9,  # must NOT land
            p_home=0.99, p_away=0.01,          # must NOT land
            official_date="2026-07-18",
        )
        with engine._get_conn() as conn:
            row = conn.execute(
                "SELECT lambda_home, p_home, official_date FROM game_outcomes WHERE game_pk=7002"
            ).fetchone()
        assert row["lambda_home"] == 4.5  # untouched
        assert row["p_home"] == 0.55      # untouched
        assert row["official_date"] == "2026-07-18"  # backfilled

    def test_update_outcome_home_loss(self, engine):
        engine.record_prediction(
            game_pk=7777, game_date="2026-05-01", season=2026,
            home_team="A", away_team="B",
            lambda_home=4.5, lambda_away=3.8,
            p_home=0.55, p_away=0.45,
        )
        engine.update_outcome(7777, actual_home_runs=2, actual_away_runs=6)
        with engine._get_conn() as conn:
            row = conn.execute(
                "SELECT home_won FROM game_outcomes WHERE game_pk=7777"
            ).fetchone()
        assert row["home_won"] == 0

    def test_away_team_bias_also_computed(self, engine):
        # Team appears as away team; bias should still be computed.
        lambda_val = 4.0
        for i in range(_MIN_SAMPLES):
            engine.record_prediction(
                game_pk=5000 + i, game_date="2026-05-01", season=2026,
                home_team="HomeTeam", away_team="AwayTeam",
                lambda_home=4.5, lambda_away=lambda_val,
                p_home=0.55, p_away=0.45,
            )
            engine.update_outcome(5000 + i, actual_home_runs=4, actual_away_runs=6)

        bias = engine.compute_team_bias("AwayTeam", 2026)
        # actual_away=6 / lambda_away=4.0 = 1.50 → clamped to 1.30
        assert bias == pytest.approx(1.0 + _BIAS_CLAMP)
