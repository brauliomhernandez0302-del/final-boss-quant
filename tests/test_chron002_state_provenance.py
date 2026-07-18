"""
CHRON-002 regression tests (audit_20260714/14_remediation_roadmap.md,
roadmap Step 2, Commit A).

ml_state and kalman_state used to be shared, unprotected, between live
production and backtest refits — the residual CHRON-001 explicitly left
open (a backtest touching a live-active season could silently overwrite
the exact Platt/team-bias/pipeline-weight/Kalman keys production reads).

Fix: both tables now carry a `state_source` column ('live'|'backtest') in
their PRIMARY KEY. Every learning-engine function that reads/writes
ml_state or kalman_state takes `prediction_source: str = "live"`.
backtest_and_retrain.py passes 'backtest' explicitly at every call site.
scripts/promote_calibration.py is the only sanctioned way to move a
backtest's fitted state into the live namespace.

Four tests below, matching the roadmap's Commit A FASE A.5 exactly:
  1. Isolation — writing 'backtest' state never alters what 'live' reads,
     and vice versa.
  2. Migration is idempotent (run twice, no failure/duplication).
  3. The copy-to-both migration produces identical values in both namespaces.
  4. promote_calibration.py copies correctly and never writes without
     --confirm.
"""
import runpy
import sqlite3
import sys
from pathlib import Path

import pytest

from modules.baseball_module.calibration.learning_engine import LearningEngine

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test_predictions.db"
    return LearningEngine(db_path=db)


# ── 1. Isolation between 'live' and 'backtest' namespaces ─────────────────


def test_save_state_isolation(engine):
    engine.save_state("platt_params", "calibration", {"a": 1.0, "b": 0.0, "n": 0}, 0, 2024,
                       prediction_source="live")
    engine.save_state("platt_params", "calibration", {"a": 9.9, "b": 9.9, "n": 999}, 999, 2024,
                       prediction_source="backtest")

    live = engine.load_state("platt_params", "calibration", 2024, prediction_source="live")
    bt = engine.load_state("platt_params", "calibration", 2024, prediction_source="backtest")

    assert live["a"] == 1.0 and live["sample_count"] == 0
    assert bt["a"] == 9.9 and bt["sample_count"] == 999

    # Overwriting the backtest value must not touch live.
    engine.save_state("platt_params", "calibration", {"a": -5.0, "b": -5.0, "n": 1}, 1, 2024,
                       prediction_source="backtest")
    live_after = engine.load_state("platt_params", "calibration", 2024, prediction_source="live")
    assert live_after["a"] == 1.0


def test_kalman_isolation(engine):
    engine.update_kalman("Cubs", "offense_home", 2024, 4.5, prediction_source="live")
    engine.update_kalman("Cubs", "offense_home", 2024, 9.9, prediction_source="backtest")

    live_est = engine.get_kalman_estimate("Cubs", "offense_home", 2024, prediction_source="live")
    bt_est = engine.get_kalman_estimate("Cubs", "offense_home", 2024, prediction_source="backtest")

    assert live_est == pytest.approx(4.5)
    assert bt_est == pytest.approx(9.9)
    assert live_est != bt_est

    # A second backtest update must not move the live estimate.
    engine.update_kalman("Cubs", "offense_home", 2024, 20.0, prediction_source="backtest")
    live_est_after = engine.get_kalman_estimate("Cubs", "offense_home", 2024, prediction_source="live")
    assert live_est_after == pytest.approx(4.5)


def test_reset_kalman_scoped_to_one_namespace(engine):
    engine.update_kalman("Mets", "offense_home", 2024, 4.0, prediction_source="live")
    engine.update_kalman("Mets", "offense_home", 2024, 8.0, prediction_source="backtest")

    deleted = engine.reset_kalman_for_seasons([2024], prediction_source="backtest")
    assert deleted == 1
    assert engine.get_kalman_estimate("Mets", "offense_home", 2024, prediction_source="live") == pytest.approx(4.0)
    assert engine.get_kalman_estimate("Mets", "offense_home", 2024, prediction_source="backtest") is None


# ── 2 & 3. Migration idempotency + copy-to-both identity ──────────────────


def test_state_source_migration_idempotent_and_copies_identical_values(tmp_path):
    db = tmp_path / "premigration.db"

    # Build a DB with pre-CHRON-002 (no state_source) ml_state/kalman_state
    # rows, mimicking what any real production DB looked like before this
    # fix — a fresh LearningEngine() first creates the CHRON-002 schema
    # from scratch (nothing to migrate), so insert rows, drop the column
    # info from a fresh instance won't help; instead build the legacy
    # shape directly.
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE ml_state (
            key TEXT NOT NULL, scope TEXT NOT NULL, season INTEGER NOT NULL,
            value_json TEXT NOT NULL, sample_count INTEGER DEFAULT 0,
            updated_at TEXT NOT NULL, PRIMARY KEY (key, scope, season)
        );
        CREATE TABLE kalman_state (
            team TEXT NOT NULL, context TEXT NOT NULL, season INTEGER NOT NULL,
            x_est REAL NOT NULL, p_est REAL NOT NULL, n_obs INTEGER DEFAULT 0,
            updated_at TEXT NOT NULL, PRIMARY KEY (team, context, season)
        );
        """
    )
    con.execute(
        "INSERT INTO ml_state VALUES ('platt_params','calibration',2024,'{\"a\":0.71,\"b\":0.02,\"n\":500}',500,'2026-01-01T00:00:00Z')"
    )
    con.execute(
        "INSERT INTO kalman_state VALUES ('Yankees','offense_home',2024,4.8,0.4,50,'2026-01-01T00:00:00Z')"
    )
    con.commit()
    con.close()

    engine = LearningEngine(db_path=db)  # triggers _migrate_chron002_state_source

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    ml_rows = con.execute(
        "SELECT state_source, value_json, sample_count FROM ml_state "
        "WHERE key='platt_params' AND scope='calibration' AND season=2024"
    ).fetchall()
    kal_rows = con.execute(
        "SELECT state_source, x_est, p_est, n_obs FROM kalman_state "
        "WHERE team='Yankees' AND context='offense_home' AND season=2024"
    ).fetchall()

    assert {r["state_source"] for r in ml_rows} == {"live", "backtest"}
    assert {r["state_source"] for r in kal_rows} == {"live", "backtest"}
    # Copy-to-both: identical payload in both namespaces.
    live_ml = next(r for r in ml_rows if r["state_source"] == "live")
    bt_ml = next(r for r in ml_rows if r["state_source"] == "backtest")
    assert live_ml["value_json"] == bt_ml["value_json"]
    assert live_ml["sample_count"] == bt_ml["sample_count"]
    live_kal = next(r for r in kal_rows if r["state_source"] == "live")
    bt_kal = next(r for r in kal_rows if r["state_source"] == "backtest")
    assert live_kal["x_est"] == bt_kal["x_est"] == 4.8
    assert live_kal["n_obs"] == bt_kal["n_obs"] == 50

    # Idempotency: instantiate twice more, row counts must not change.
    LearningEngine(db_path=db)
    LearningEngine(db_path=db)
    con2 = sqlite3.connect(db)
    ml_count = con2.execute(
        "SELECT COUNT(*) FROM ml_state WHERE key='platt_params' AND scope='calibration' AND season=2024"
    ).fetchone()[0]
    kal_count = con2.execute(
        "SELECT COUNT(*) FROM kalman_state WHERE team='Yankees' AND context='offense_home' AND season=2024"
    ).fetchone()[0]
    assert ml_count == 2
    assert kal_count == 2


# ── 4. promote_calibration.py ───────────────────────────────────────────


def _run_promote(argv):
    old_argv = sys.argv
    sys.argv = ["promote_calibration.py", *argv]
    try:
        runpy.run_path(str(REPO_ROOT / "scripts" / "promote_calibration.py"), run_name="__main__")
    finally:
        sys.argv = old_argv


def test_promote_calibration_dry_run_writes_nothing(engine, capsys):
    engine.save_state("platt_params", "calibration", {"a": 1.0, "b": 0.0, "n": 0}, 0, 2024,
                       prediction_source="live")
    engine.save_state("platt_params", "calibration", {"a": 0.75, "b": 0.05, "n": 400}, 400, 2024,
                       prediction_source="backtest")

    _run_promote(["--season", "2024", "--mechanism", "platt", "--db-path", str(engine.db_path)])

    out = capsys.readouterr().out
    assert "DRY RUN" in out
    # Live value must be untouched — no --confirm was passed.
    live = engine.load_state("platt_params", "calibration", 2024, prediction_source="live")
    assert live["a"] == 1.0


def test_promote_calibration_confirm_copies_backtest_to_live(engine, capsys):
    engine.save_state("platt_params", "calibration", {"a": 1.0, "b": 0.0, "n": 0}, 0, 2024,
                       prediction_source="live")
    engine.save_state("platt_params", "calibration", {"a": 0.75, "b": 0.05, "n": 400}, 400, 2024,
                       prediction_source="backtest")

    _run_promote([
        "--season", "2024", "--mechanism", "platt", "--db-path", str(engine.db_path), "--confirm",
    ])

    live = engine.load_state("platt_params", "calibration", 2024, prediction_source="live")
    assert live["a"] == 0.75
    assert live["sample_count"] == 400
    # The backtest source value itself must remain unchanged.
    bt = engine.load_state("platt_params", "calibration", 2024, prediction_source="backtest")
    assert bt["a"] == 0.75
