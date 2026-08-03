"""
CHRON-001 regression tests (audit_20260714/, roadmap Step 1).

game_outcomes used to be shared, unprotected, between live-production writes
(record_prediction()/update_outcome()) and backtest overwrites
(backtest_and_retrain.py::update_game_outcomes(), a plain UPDATE...WHERE
game_pk=? with no provenance guard) — a routine backtest run touching a
reconciled live game silently destroyed the live prediction record with no
audit trail. See audit_20260714/08_chronology_audit.md and
audit_20260714/chron001_forensics_report.md (0 of 563 already-overwritten
live rows were recoverable from any backup on disk).

Fix: a `source` column records who authored the live prediction columns
(set once at INSERT, never flipped); backtest_and_retrain.py writes its
recomputed values to backtest_lambda_home/backtest_lambda_away/
backtest_p_home/backtest_p_away/backtest_p_home_raw/backtest_p_away_raw/
backtest_stage_factors_json instead of the live columns; every learning
function that reads a prediction column takes `prediction_source:
'live'|'backtest' = 'live'`.

Six tests below, matching audit_20260714/14_remediation_roadmap.md Step 1
FASE 5 exactly:
  1. The audit's own regression test — insert live, backtest-overwrite,
     assert live columns untouched byte-for-byte.
  2. record_prediction()'s backfill branch never flips an already-set
     `source` and never touches backtest_* columns.
  3. A Platt refit in backtest mode trains on backtest_* columns, not live.
  4. The live path with all defaults reads live columns unchanged.
  5. The schema migration is idempotent (run twice, no failure/duplication).
  6. The one-time `source` backfill assigns the correct category to each
     of the three row types.
"""
import sqlite3

import pytest

import backtest_and_retrain as backtest
from modules.baseball_module.calibration.learning_engine import LearningEngine


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test_predictions.db"
    return LearningEngine(db_path=db)


def _row(conn: sqlite3.Connection, game_pk: int) -> sqlite3.Row:
    conn.row_factory = sqlite3.Row
    return conn.execute("SELECT * FROM game_outcomes WHERE game_pk = ?", (game_pk,)).fetchone()


# ── 1. The audit's own regression test ──────────────────────────────────────


def test_backtest_overwrite_never_touches_live_columns(engine):
    """record_prediction() authors a live row; update_game_outcomes() then
    "reprocesses" the same game_pk (as a batched backtest run touching a
    reconciled live game would) — the live prediction columns and `source`
    must come out byte-for-byte identical, and the backtest_* shadow columns
    must hold the new values instead."""
    inserted = engine.record_prediction(
        game_pk=90001, game_date="2026-06-01", season=2026,
        home_team="Dodgers", away_team="Giants",
        lambda_home=5.2259, lambda_away=3.8123,
        p_home=0.81749, p_away=0.18251,
        p_home_raw=0.7912, p_away_raw=0.2088,
        stage_factors={"pitcher_on_home_lambda": 0.94},
    )
    assert inserted is True

    with engine._get_conn() as conn:
        before = _row(conn, 90001)
    assert before["source"] == "live"
    assert before["backtest_run_at"] is None

    conn = backtest.get_conn(engine.db_path)
    backtest.update_game_outcomes(
        conn, 90001,
        lh=9.999, la=8.888, p_home=0.7777, p_away=0.2223,
        p_home_raw=0.75, p_away_raw=0.25,
        stage_factors={"pitcher_on_home_lambda": 1.10},
    )
    conn.commit()
    after = _row(conn, 90001)

    # Live prediction columns: byte-for-byte identical to what record_prediction() wrote.
    assert after["lambda_home"] == before["lambda_home"]
    assert after["lambda_away"] == before["lambda_away"]
    assert after["p_home"] == before["p_home"]
    assert after["p_away"] == before["p_away"]
    assert after["p_home_raw"] == before["p_home_raw"]
    assert after["p_away_raw"] == before["p_away_raw"]
    assert after["stage_factors_json"] == before["stage_factors_json"]
    assert after["source"] == "live"

    # backtest_* shadow columns: hold the new, recomputed values.
    assert after["backtest_lambda_home"] == 9.999
    assert after["backtest_lambda_away"] == 8.888
    assert after["backtest_p_home"] == 0.7777
    assert after["backtest_p_away"] == 0.2223
    assert after["backtest_p_home_raw"] == 0.75
    assert after["backtest_p_away_raw"] == 0.25
    assert after["backtest_run_at"] is not None


# ── 2. record_prediction()'s backfill branch ────────────────────────────────


def test_record_prediction_backfill_does_not_flip_source(engine):
    """Calling record_prediction() again for a game_pk that already has a
    `source` must not change it, and must never write to backtest_* columns
    — that branch only backfills stage_factors_json/p_home_raw/p_away_raw/
    ml_home_pin/ml_away_pin, exactly as before this fix."""
    engine.record_prediction(
        game_pk=90002, game_date="2026-06-02", season=2026,
        home_team="Yankees", away_team="Red Sox",
        lambda_home=4.5, lambda_away=4.1, p_home=0.55, p_away=0.45,
    )
    with engine._get_conn() as conn:
        assert _row(conn, 90002)["source"] == "live"

    # Second call for the same game_pk — INSERT OR IGNORE no-ops, backfill fires.
    inserted_again = engine.record_prediction(
        game_pk=90002, game_date="2026-06-02", season=2026,
        home_team="Yankees", away_team="Red Sox",
        lambda_home=4.5, lambda_away=4.1, p_home=0.55, p_away=0.45,
        stage_factors={"defense_on_away_lambda": 1.02},
        p_home_raw=0.53, p_away_raw=0.47,
    )
    assert inserted_again is False

    with engine._get_conn() as conn:
        row = _row(conn, 90002)
    assert row["source"] == "live"
    assert row["backtest_lambda_home"] is None
    assert row["backtest_p_home"] is None
    # The backfill DID fill previously-NULL fields.
    assert row["p_home_raw"] == 0.53
    assert row["stage_factors_json"] is not None


def test_record_prediction_backfill_sets_live_when_source_was_null(engine):
    """A row inserted by some path that predates this fix (source left
    NULL) gets 'live' the first time record_prediction() backfills it —
    conservative, since record_prediction() is exclusively the live writer
    (backtest_and_retrain.py never calls it)."""
    with engine._get_conn() as conn:
        conn.execute(
            """
            INSERT INTO game_outcomes
                (game_pk, game_date, season, home_team, away_team,
                 lambda_home, lambda_away, p_home, p_away)
            VALUES (90003, '2026-06-03', 2026, 'Mets', 'Braves', 4.4, 4.2, 0.5, 0.5)
            """
        )
    with engine._get_conn() as conn:
        assert _row(conn, 90003)["source"] is None

    engine.record_prediction(
        game_pk=90003, game_date="2026-06-03", season=2026,
        home_team="Mets", away_team="Braves",
        lambda_home=4.4, lambda_away=4.2, p_home=0.5, p_away=0.5,
        p_home_raw=0.51, p_away_raw=0.49,
    )
    with engine._get_conn() as conn:
        assert _row(conn, 90003)["source"] == "live"


# ── 3. Platt refit in backtest mode trains on backtest_* columns ───────────


def test_recalibrate_platt_backtest_mode_reads_backtest_columns(engine):
    """Feed live columns an inverted (anti-correlated) signal and backtest_*
    columns a correctly-correlated one for the same 60 rows. The two fits
    must disagree in sign — proof recalibrate_platt(prediction_source=
    'backtest') is not reading the live columns at all."""
    season = 2031  # unused season, isolated from any other test's data
    with engine._get_conn() as conn:
        for i in range(60):
            home_won = 1 if i % 2 == 0 else 0
            # Live: anti-correlated with the real outcome (deliberately "wrong").
            p_home_raw_live = 0.10 if home_won == 1 else 0.90
            # Backtest: correctly correlated with the real outcome.
            p_home_raw_bt = 0.90 if home_won == 1 else 0.10
            conn.execute(
                """
                INSERT INTO game_outcomes
                    (game_pk, game_date, season, home_team, away_team,
                     home_won, p_home, p_home_raw,
                     backtest_p_home, backtest_p_home_raw, source, backtest_run_at)
                VALUES (?, ?, ?, 'A', 'B', ?, ?, ?, ?, ?, 'backtest', '2026-01-01T00:00:00Z')
                """,
                (91000 + i, "2031-05-01", season, home_won,
                 p_home_raw_live, p_home_raw_live, p_home_raw_bt, p_home_raw_bt),
            )

    # Note: recalibrate_platt()'s internal naming is `a` = fitted slope
    # (lr.coef_), `b` = fitted intercept (lr.intercept_) — unusual vs. the
    # typical "y = a + bx" convention, but that's the existing, unmodified
    # code this fix builds on (see its docstring / _platt() application).
    a_live, b_live = engine.recalibrate_platt(season, prediction_source="live")
    a_bt, b_bt = engine.recalibrate_platt(season, prediction_source="backtest")

    assert a_live < 0, f"expected negative slope from anti-correlated live data, got {a_live}"
    assert a_bt > 0, f"expected positive slope from correlated backtest data, got {a_bt}"


# ── 4. Live path with defaults is unchanged ─────────────────────────────────


def test_live_defaults_read_live_columns_unchanged(engine):
    """compute_team_bias() called with no prediction_source argument must
    read the live lambda_home/lambda_away/stage_factors_json columns —
    exactly the pre-fix behavior — even when backtest_* columns are present
    and wildly different.

    Uses `before_date` (as backtest_and_retrain.py's real per-game
    walk-forward calls always do — see run_pipeline()) so the season-wide
    ml_state cache is bypassed and every call recomputes fresh from the DB;
    without it, a 'live' call and a 'backtest' call for the same
    team/season would collide on the same (source-unaware) cache key — a
    pre-existing characteristic unrelated to this fix (see the comment on
    backtest_and_retrain.py's "step 4" bias-refresh call site), not
    something this test is meant to exercise.
    """
    season = 2032
    team = "Cubs"
    cutoff = "2032-05-02"
    with engine._get_conn() as conn:
        for i in range(10):
            # actual = 2x lambda on the LIVE column (real, strong under-prediction bias)
            # but actual = 1x lambda on the backtest column (perfectly calibrated)
            conn.execute(
                """
                INSERT INTO game_outcomes
                    (game_pk, game_date, season, home_team, away_team,
                     lambda_home, backtest_lambda_home,
                     actual_home_runs, actual_away_runs, source, backtest_run_at)
                VALUES (?, ?, ?, ?, 'Opponent', 4.0, 8.0, 8, 3, 'backtest', '2026-01-01T00:00:00Z')
                """,
                (92000 + i, "2032-05-01", season, team),
            )

    bias_default = engine.compute_team_bias(team, season, min_samples=5, before_date=cutoff)
    bias_explicit_live = engine.compute_team_bias(
        team, season, min_samples=5, before_date=cutoff, prediction_source="live",
    )
    bias_backtest = engine.compute_team_bias(
        team, season, min_samples=5, before_date=cutoff, prediction_source="backtest",
    )

    assert bias_default == bias_explicit_live
    # Live: untruncate_home_runs(8)/4.0 ~= 2.07 -> clamped bias ceiling (1.3)
    assert bias_default > 1.2
    # Backtest: untruncate_home_runs(8)/8.0 ~= 1.03 -> close to neutral, well below the clamp
    assert bias_backtest < 1.1
    assert bias_default != bias_backtest


# ── 5. Schema migration is idempotent ───────────────────────────────────────


def test_migration_idempotent(tmp_path):
    db = tmp_path / "idempotent.db"
    engine1 = LearningEngine(db_path=db)
    engine1.record_prediction(
        game_pk=93001, game_date="2026-06-01", season=2026,
        home_team="A", away_team="B", lambda_home=4.5, lambda_away=4.0,
        p_home=0.55, p_away=0.45,
    )

    # Re-instantiating (as every live run_module() call and every backtest
    # run does) must not fail and must not duplicate or corrupt rows.
    engine2 = LearningEngine(db_path=db)
    engine3 = LearningEngine(db_path=db)

    with engine3._get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM game_outcomes WHERE game_pk = 93001").fetchone()[0]
        row = _row(conn, 93001)
    assert count == 1
    assert row["source"] == "live"


# ── 6. One-time `source` backfill assigns the correct category ────────────


def test_source_backfill_three_categories(tmp_path):
    db = tmp_path / "backfill.db"
    engine = LearningEngine(db_path=db)  # creates schema, backfill no-ops (table empty)

    with engine._get_conn() as conn:
        # Category 1: season=2026, backtest_run_at NULL -> 'live'
        conn.execute(
            "INSERT INTO game_outcomes (game_pk, game_date, season, home_team, away_team) "
            "VALUES (94001, '2026-06-01', 2026, 'A', 'B')"
        )
        # Category 2: backtest_run_at NOT NULL (any season) -> 'backtest',
        # and its live-column values get copied into backtest_*.
        conn.execute(
            "INSERT INTO game_outcomes "
            "(game_pk, game_date, season, home_team, away_team, "
            " lambda_home, p_home, backtest_run_at) "
            "VALUES (94002, '2025-06-01', 2025, 'C', 'D', 4.5, 0.55, '2026-06-28T22:33:48Z')"
        )
        # Category 3: neither -> 'import' (e.g. an older bulk-imported season
        # never touched by a backtest run).
        conn.execute(
            "INSERT INTO game_outcomes (game_pk, game_date, season, home_team, away_team) "
            "VALUES (94003, '2023-06-01', 2023, 'E', 'F')"
        )

    # Re-instantiate to trigger the backfill against these freshly-inserted,
    # source-less rows (mirrors: rows existed before this fix shipped).
    engine2 = LearningEngine(db_path=db)

    with engine2._get_conn() as conn:
        r1 = _row(conn, 94001)
        r2 = _row(conn, 94002)
        r3 = _row(conn, 94003)

    assert r1["source"] == "live"
    assert r2["source"] == "backtest"
    assert r2["backtest_lambda_home"] == 4.5
    assert r2["backtest_p_home"] == 0.55
    assert r3["source"] == "import"
