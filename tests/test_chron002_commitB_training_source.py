"""
Roadmap Step 2, Commit B tests (audit_20260714/14_remediation_roadmap.md).

recalibrate_platt_2d() is the only cross-season reader of game_outcomes
prediction columns anywhere in the live-reachable code path (enumeration:
audit_20260714/chron002_commitB_enumeration.md). Since CHRON-001, a
season's live prediction columns are frozen forever once a backtest has
touched that row; the backtest_* columns are what a validated backtest
re-run actually refreshes. `training_columns='backtest_preferred'`
(the new default) reads COALESCE(backtest_p_home, p_home) instead of the
plain live column, so calibration trains on the current model's output,
not a frozen historical snapshot.

Three tests, matching FASE B.3 exactly:
  1. Equality freeze — on data shaped like today's real DB (backtest_p_home
     == p_home everywhere a backtest has touched), 'backtest_preferred'
     and 'live_only' produce IDENTICAL fitted params. Locks in that this
     commit is a behavioral no-op today.
  2. Synthetic rows where live and backtest values differ — the fit uses
     the backtest values.
  3. COALESCE fallback — rows with no backtest_p_home populated fall back
     to the live p_home value.
"""
import pytest

from modules.baseball_module.calibration.learning_engine import LearningEngine, _PLATT2D_MIN_SAMPLES


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test_predictions.db"
    return LearningEngine(db_path=db)


def _insert_row(conn, game_pk, season, home_won, p_home, backtest_p_home,
                 ml_home_pin=1.90, ml_away_pin=1.95):
    conn.execute(
        """
        INSERT INTO game_outcomes
            (game_pk, game_date, season, home_team, away_team,
             home_won, p_home, backtest_p_home, ml_home_pin, ml_away_pin, source)
        VALUES (?, ?, ?, 'A', 'B', ?, ?, ?, ?, ?, 'backtest')
        """,
        (game_pk, f"{season}-05-01", season, home_won, p_home, backtest_p_home,
         ml_home_pin, ml_away_pin),
    )


def _n_rows() -> int:
    return _PLATT2D_MIN_SAMPLES + 10


# ── 1. Equality freeze: today's real-world shape produces identical fits ──


def test_backtest_preferred_equals_live_only_when_columns_match(engine):
    """Mirrors the current production DB: every game_outcomes row a
    backtest has touched has backtest_p_home == p_home (CHRON-001's own
    migration copied them identically). Both training_columns modes must
    therefore produce byte-identical fitted params — proof this commit
    doesn't move any current number."""
    season = 2040
    with engine._get_conn() as conn:
        for i in range(_n_rows()):
            home_won = i % 2
            # Realistic-ish, noisy but genuinely predictive signal.
            p = 0.65 if home_won == 1 else 0.35
            p = min(0.95, max(0.05, p + (0.03 if i % 5 == 0 else -0.02)))
            _insert_row(conn, 95000 + i, season - 1, home_won, p, p)  # backtest == live

    result_preferred = engine.recalibrate_platt_2d(season, training_columns="backtest_preferred")
    result_live_only = engine.recalibrate_platt_2d(season, training_columns="live_only")

    assert result_preferred is not None
    assert result_live_only is not None
    assert result_preferred == result_live_only


# ── 2. Backtest-preferred: fit uses backtest values when they differ ──────


def test_backtest_preferred_uses_backtest_values_when_they_differ(engine):
    """live p_home is deliberately anti-correlated with home_won;
    backtest_p_home is correctly correlated. 'backtest_preferred' must fit
    against the backtest signal (positive slope on p_home), while
    'live_only' fits against the live signal (negative slope)."""
    season = 2041
    with engine._get_conn() as conn:
        for i in range(_n_rows()):
            home_won = i % 2
            p_live = 0.15 if home_won == 1 else 0.85       # anti-correlated
            p_backtest = 0.85 if home_won == 1 else 0.15   # correctly correlated
            _insert_row(conn, 96000 + i, season - 1, home_won, p_live, p_backtest)

    a_pref, b_pref, c_pref = engine.recalibrate_platt_2d(season, training_columns="backtest_preferred")
    a_live, b_live, c_live = engine.recalibrate_platt_2d(season, training_columns="live_only")

    # b is the coefficient on logit(p_home) — see recalibrate_platt()'s own
    # a/b-naming note; Platt-2D's b here is genuinely the p_home slope.
    assert b_pref > 0, f"expected positive p_home slope from backtest-preferred fit, got {b_pref}"
    assert b_live < 0, f"expected negative p_home slope from live-only fit, got {b_live}"


# ── 3. COALESCE fallback for rows never touched by a backtest ─────────────


def test_backtest_preferred_falls_back_to_live_when_backtest_missing(engine):
    """Rows with backtest_p_home left NULL (never processed by a backtest —
    e.g. very recent live games) must fall back to the live p_home value
    under 'backtest_preferred', not be silently excluded."""
    season = 2042
    with engine._get_conn() as conn:
        for i in range(_n_rows()):
            home_won = i % 2
            p_live = 0.70 if home_won == 1 else 0.30
            conn.execute(
                """
                INSERT INTO game_outcomes
                    (game_pk, game_date, season, home_team, away_team,
                     home_won, p_home, backtest_p_home, ml_home_pin, ml_away_pin, source)
                VALUES (?, ?, ?, 'A', 'B', ?, ?, NULL, 1.90, 1.95, 'live')
                """,
                (97000 + i, f"{season - 1}-05-01", season - 1, home_won, p_live),
            )

    result = engine.recalibrate_platt_2d(season, training_columns="backtest_preferred")
    assert result is not None
    a, b, c = result
    # Same signal as a positively-correlated fit — the fallback must have
    # actually picked up the live values (b > 0), not silently dropped
    # every row (which would mean n < _PLATT2D_MIN_SAMPLES -> None).
    assert b > 0
