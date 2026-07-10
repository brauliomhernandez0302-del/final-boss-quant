"""Anti-leakage regression suite — Opening Day 2024.

FASE 0 / blueprint item 0.1: "test anti-leakage automatizado en Opening Day 2024".

Strategy: poison tests, not implementation-detail assertions. For every PIT
domain, seed a scratch store with legitimate "past" data plus a deliberately
extreme "future" record (dated after the target as-of-date), then assert the
as-of-date snapshot is byte-identical whether or not the future record exists.
This catches ANY leak mechanism — a missing WHERE clause, a wrong comparison
operator, a cache that ignores date — without hardcoding today's SQL text.

Two domains are covered:
  1. `RawSavantEventsCache` + `SavantRollingPITBuilder` — the shared raw-event
     store behind all 4 Savant-derived PIT domains (pitcher, team offense/TTE,
     team defense, bullpen all read through the same date-bounded query
     methods; the pitcher rolling builder is exercised directly as the
     representative case).
  2. `LearningEngine.compute_team_bias` / `compute_multidim_bias` /
     `compute_team_bias_kalman_adjusted` — found genuinely leaking on
     2026-07-08 (season-only WHERE clause, no date bound at all) and fixed
     with a `before_date` walk-forward cutoff. These tests pin that fix down.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from modules.baseball_module.advanced_pit_enrichment.raw_savant_events_cache import (
    RawSavantEventsCache,
)
from modules.baseball_module.advanced_pit_enrichment.savant_rolling_pit_builder import (
    SavantRollingPITBuilder,
)
from modules.baseball_module.calibration.learning_engine import LearningEngine

OPENING_DAY_2024 = "2024-03-28"
SEASON_START_2024 = "2024-03-20"  # real 2024 opener (Seoul series)
FUTURE_DATE = "2024-09-15"  # deep in-season, must be invisible as of Opening Day


def _savant_event(*, game_date: str, game_pk: int, pitcher: int, ab: int, est_woba: float) -> dict:
    return {
        "game_date": game_date,
        "game_pk": game_pk,
        "at_bat_number": ab,
        "pitch_number": 1,
        "pitcher": pitcher,
        "batter": 999,
        "events": "single",
        "launch_speed": 95.0,
        "launch_angle": 12.0,
        "estimated_woba_using_speedangle": est_woba,
        "woba_value": est_woba,
        "woba_denom": 1.0,
        "launch_speed_angle": 4,
    }


class TestRawSavantEventsAntiLeakage:
    """Poison test: an extreme future event must not move the Opening Day snapshot."""

    def _build_snapshot(self, db_path: Path, *, include_future_poison: bool) -> dict:
        cache = RawSavantEventsCache(db_path)
        events = [
            _savant_event(game_date=SEASON_START_2024, game_pk=1, pitcher=700001, ab=1, est_woba=0.320),
            _savant_event(game_date="2024-03-27", game_pk=2, pitcher=700001, ab=1, est_woba=0.310),
        ]
        if include_future_poison:
            # Wildly inflated wOBA, dated well after Opening Day. If this leaks
            # into the Opening Day rolling snapshot, est_woba will move a lot.
            events.append(
                _savant_event(game_date=FUTURE_DATE, game_pk=3, pitcher=700001, ab=1, est_woba=0.999)
            )
        cache.save_events(events, source_fingerprint="test-fixture")

        builder = SavantRollingPITBuilder(cache=cache)
        metrics = builder.build_for_as_of_date(
            season_start_date=SEASON_START_2024, as_of_date=OPENING_DAY_2024
        )
        m = metrics[700001]
        return {"est_woba": m.est_woba, "pa": m.pa, "bip": m.bip}

    def test_future_poison_does_not_change_opening_day_snapshot(self, tmp_path):
        clean = self._build_snapshot(tmp_path / "clean.db", include_future_poison=False)
        poisoned = self._build_snapshot(tmp_path / "poisoned.db", include_future_poison=True)
        assert clean == poisoned, (
            "A Savant event dated after Opening Day leaked into the "
            "as-of-Opening-Day rolling snapshot"
        )

    def test_future_event_is_visible_once_its_own_date_arrives(self, tmp_path):
        """Sanity check the poison event is real data, not silently dropped."""
        cache = RawSavantEventsCache(tmp_path / "later.db")
        events = [
            _savant_event(game_date=SEASON_START_2024, game_pk=1, pitcher=700001, ab=1, est_woba=0.320),
            _savant_event(game_date=FUTURE_DATE, game_pk=3, pitcher=700001, ab=1, est_woba=0.999),
        ]
        cache.save_events(events, source_fingerprint="test-fixture")
        builder = SavantRollingPITBuilder(cache=cache)
        metrics = builder.build_for_as_of_date(
            season_start_date=SEASON_START_2024, as_of_date=FUTURE_DATE
        )
        assert metrics[700001].pa == 2  # both events now legitimately in scope


class TestTeamBiasAntiLeakage:
    """Poison test for the real leak found+fixed 2026-07-08 in learning_engine.py."""

    TEAM = "AAA"
    OPP = "ZZZ"

    def _seed(self, conn: sqlite3.Connection, *, include_future_poison: bool) -> None:
        rows = [(1000, OPENING_DAY_2024, 2024, self.TEAM, self.OPP, 3, 4.0, 1.0, 4, 3, 1)]
        gp = 1001
        for i in range(9):
            rows.append((gp, f"2024-04-{i + 2:02d}", 2024, self.TEAM, self.OPP, 4, 4.0, 1.0, 4, 3, 1))
            gp += 1
        if include_future_poison:
            # Extreme run output, dated deep in-season — must not inform the
            # bias used to predict the Opening Day game above.
            for i in range(10):
                rows.append(
                    (gp, f"2024-09-{i + 1:02d}", 2024, self.TEAM, self.OPP, 9, 4.0, 1.0, 12, 3, 1)
                )
                gp += 1
        conn.executemany(
            """INSERT INTO game_outcomes
               (game_pk, game_date, season, home_team, away_team, month,
                lambda_home, lambda_away, actual_home_runs, actual_away_runs, home_won)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()

    def _bias_as_of_opening_day(self, db_path: Path, *, include_future_poison: bool) -> float:
        learning = LearningEngine(db_path=db_path)
        with learning._get_conn() as conn:
            self._seed(conn, include_future_poison=include_future_poison)
        return learning.compute_team_bias_kalman_adjusted(
            self.TEAM, 2024, "offense_home", month=3, before_date=OPENING_DAY_2024
        )

    def test_future_poison_does_not_change_opening_day_bias(self, tmp_path):
        clean = self._bias_as_of_opening_day(tmp_path / "clean.db", include_future_poison=False)
        poisoned = self._bias_as_of_opening_day(tmp_path / "poisoned.db", include_future_poison=True)
        assert clean == poisoned == 1.0, (
            "September actual-runs data leaked into a bias computed as of Opening Day"
        )

    def test_poison_is_visible_once_its_own_date_arrives(self, tmp_path):
        """Sanity check: the same poison rows DO move the bias once genuinely in the past."""
        learning = LearningEngine(db_path=tmp_path / "later.db")
        with learning._get_conn() as conn:
            self._seed(conn, include_future_poison=True)
        bias = learning.compute_team_bias_kalman_adjusted(
            self.TEAM, 2024, "offense_home", month=9, before_date="2024-09-05"
        )
        assert bias > 1.0, "poison rows should legitimately move the bias once they're in the past"

    def test_default_before_date_none_preserves_legacy_full_season_behavior(self, tmp_path):
        """before_date=None (the live-path default) must be unchanged: no regression
        for the one caller (run_module.py, live predictions) where full-season
        aggregation is correct because no future actual results can exist yet."""
        learning = LearningEngine(db_path=tmp_path / "live.db")
        with learning._get_conn() as conn:
            self._seed(conn, include_future_poison=True)
        bias = learning.compute_team_bias_kalman_adjusted(self.TEAM, 2024, "offense_home", month=3)
        assert bias > 1.0  # full season (including the "future" rows) is legitimately visible
