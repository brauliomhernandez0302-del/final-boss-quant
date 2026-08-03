"""
LEARN-002 regression tests (audit_20260714/14_remediation_roadmap.md,
roadmap Step 2, Commit C).

calibration_health() is the minimal "is calibration alive" production
monitor the project's own blueprint had proposed but never built, despite
three independent, confirmed silent-calibration-failure incidents
(REG-001, REG-003, REG-007). Scoped to source='live' rows only — possible
for the first time because of CHRON-001's provenance column.

Five tests, matching FASE C.3 exactly:
  1. All-identity dataset (p_home == p_home_raw everywhere) -> the alert
     fires. This is the test that would have caught REG-003/REG-007.
  2. ml_home_pin all NULL -> the alert fires.
  3. Healthy dataset -> no alert.
  4. source='backtest' rows in the window are excluded from the metrics.
  5. n_rows < 20 -> no alert (avoid noise on a small sample).
"""
import logging

import pytest
from datetime import datetime, timedelta, timezone

from modules.baseball_module.calibration.learning_engine import LearningEngine


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test_predictions.db"
    return LearningEngine(db_path=db)


def _recent_date(days_ago: int = 1) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _insert(conn, game_pk, p_home, p_home_raw, ml_home_pin, source="live", days_ago=1):
    conn.execute(
        """
        INSERT INTO game_outcomes
            (game_pk, game_date, season, home_team, away_team,
             p_home, p_home_raw, ml_home_pin, ml_away_pin, source)
        VALUES (?, ?, 2026, 'A', 'B', ?, ?, ?, 1.95, ?)
        """,
        (game_pk, _recent_date(days_ago), p_home, p_home_raw, ml_home_pin, source),
    )


# ── 1. All-identity Platt -> alert fires ────────────────────────────────


def test_identity_platt_triggers_alert(engine, caplog):
    with engine._get_conn() as conn:
        for i in range(30):
            _insert(conn, 80000 + i, p_home=0.55, p_home_raw=0.55, ml_home_pin=1.90)

    with caplog.at_level(logging.WARNING):
        result = engine.calibration_health(window_days=14)

    assert result["n_rows"] == 30
    assert result["pct_platt_active"] == 0.0
    assert any("appears DEAD" in r.message for r in caplog.records)


# ── 2. Pinnacle missing -> alert fires ──────────────────────────────────


def test_missing_pinnacle_triggers_alert(engine, caplog):
    with engine._get_conn() as conn:
        for i in range(30):
            _insert(conn, 81000 + i, p_home=0.60, p_home_raw=0.55, ml_home_pin=None)

    with caplog.at_level(logging.WARNING):
        result = engine.calibration_health(window_days=14)

    assert result["pct_pinnacle_present"] == 0.0
    assert any("appears MISSING" in r.message for r in caplog.records)


# ── 3. Healthy dataset -> no alert ──────────────────────────────────────


def test_healthy_dataset_no_alert(engine, caplog):
    with engine._get_conn() as conn:
        for i in range(30):
            _insert(conn, 82000 + i, p_home=0.55 + i * 0.001, p_home_raw=0.50, ml_home_pin=1.90)

    with caplog.at_level(logging.WARNING):
        result = engine.calibration_health(window_days=14)

    assert result["pct_platt_active"] == 100.0
    assert result["pct_pinnacle_present"] == 100.0
    assert not any(
        "appears DEAD" in r.message or "appears MISSING" in r.message for r in caplog.records
    )


# ── 4. Backtest rows in the window don't count ──────────────────────────


def test_backtest_rows_excluded_from_metrics(engine):
    with engine._get_conn() as conn:
        # 25 healthy live rows.
        for i in range(25):
            _insert(conn, 83000 + i, p_home=0.55, p_home_raw=0.50, ml_home_pin=1.90, source="live")
        # 100 unhealthy backtest rows in the same window — must not move the metric.
        for i in range(100):
            _insert(conn, 84000 + i, p_home=0.50, p_home_raw=0.50, ml_home_pin=None, source="backtest")

    result = engine.calibration_health(window_days=14)
    assert result["n_rows"] == 25  # backtest rows not counted
    assert result["pct_platt_active"] == 100.0
    assert result["pct_pinnacle_present"] == 100.0


# ── 5. n_rows < 20 -> no alert ───────────────────────────────────────────


def test_small_sample_no_alert(engine, caplog):
    with engine._get_conn() as conn:
        for i in range(10):  # below _CALIBRATION_HEALTH_MIN_ROWS=20
            _insert(conn, 85000 + i, p_home=0.50, p_home_raw=0.50, ml_home_pin=None)

    with caplog.at_level(logging.WARNING):
        result = engine.calibration_health(window_days=14)

    assert result["n_rows"] == 10
    assert result["pct_platt_active"] == 0.0
    assert not any(
        "appears DEAD" in r.message or "appears MISSING" in r.message for r in caplog.records
    )
