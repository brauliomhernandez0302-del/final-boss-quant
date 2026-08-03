"""Fase 2B commit B2 — regression tests for the official_date cutover
(audit_20260714/fase2b/).

game_outcomes.game_date is the raw UTC gameDate timestamp truncated to a
date — for any night game crossing midnight UTC (the norm for west-coast
teams), that's one calendar day AHEAD of the MLB schedule's own
officialDate. Every PIT walk-forward cutoff derived from game_date
therefore requested one day too late, and with PITCache.get_latest()'s
inclusive `<=` comparison, silently included the target game's own game-day
in its own PIT snapshot — a real leak (audit_20260714/verificacion_operativa/
reporte.md, V4). Confirmed empirically against game_pk=745199
(officialDate=2024-09-18, game_date=2024-09-19).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backtest_and_retrain import (
    _official_day_for_row,
    _prediction_cutoff_for_row,
    _team_tte_pit_cutoff_for_row,
    _defense_pit_cutoff_for_row,
    _bullpen_pit_cutoff_for_row,
    _experimental_pitcher_pit_cutoff_for_row,
)

PIT_CACHE_DB = Path("data/predictions_history.db")


def _row(game_date: str, official_date: str | None):
    return {"game_date": game_date, "official_date": official_date}


class TestOfficialDayForRow:

    def test_night_game_uses_official_date_not_game_date(self):
        # A night game crossing midnight UTC: game_date is one day ahead.
        row = _row(game_date="2024-09-19", official_date="2024-09-18")
        assert _official_day_for_row(row) == "2024-09-18"

    def test_day_game_official_date_equals_game_date(self):
        row = _row(game_date="2024-09-29", official_date="2024-09-29")
        assert _official_day_for_row(row) == "2024-09-29"

    def test_missing_official_date_falls_back_to_game_date(self):
        row = _row(game_date="2024-09-19", official_date=None)
        assert _official_day_for_row(row) == "2024-09-19"

    def test_missing_official_date_key_entirely_falls_back(self):
        row = {"game_date": "2024-09-19"}
        assert _official_day_for_row(row) == "2024-09-19"


class TestCutoffFunctionsUseOfficialDate:
    """Each of the 5 cutoffs must derive from official_date, not game_date —
    verified for a synthetic night game (where the two disagree) and a
    synthetic day game (where they agree and nothing should change)."""

    NIGHT_ROW = _row(game_date="2024-09-19", official_date="2024-09-18")
    DAY_ROW = _row(game_date="2024-09-29", official_date="2024-09-29")

    @pytest.mark.parametrize("cutoff_fn", [
        _team_tte_pit_cutoff_for_row,
        _defense_pit_cutoff_for_row,
        _bullpen_pit_cutoff_for_row,
    ])
    def test_night_game_cutoff_is_one_day_before_official_date(self, cutoff_fn):
        assert cutoff_fn(self.NIGHT_ROW) == "2024-09-17T23:59:59Z"

    @pytest.mark.parametrize("cutoff_fn", [
        _team_tte_pit_cutoff_for_row,
        _defense_pit_cutoff_for_row,
        _bullpen_pit_cutoff_for_row,
    ])
    def test_day_game_cutoff_unaffected(self, cutoff_fn):
        assert cutoff_fn(self.DAY_ROW) == "2024-09-28T23:59:59Z"

    def test_prediction_cutoff_is_end_of_official_date_itself(self):
        # _prediction_cutoff_for_row is NOT a "day-before" walk-forward
        # cutoff like the other 4 (it never subtracted a day, pre- or
        # post-B2 — only the date SOURCE changed here, from game_date to
        # official_date, preserving this function's original semantic).
        assert _prediction_cutoff_for_row(self.NIGHT_ROW) == "2024-09-18T23:59:59Z"
        assert _prediction_cutoff_for_row(self.DAY_ROW) == "2024-09-29T23:59:59Z"

    def test_experimental_pitcher_pit_cutoff_night_game(self):
        assert _experimental_pitcher_pit_cutoff_for_row(self.NIGHT_ROW) == "2024-09-17T23:59:59Z"

    def test_experimental_pitcher_pit_cutoff_day_game(self):
        assert _experimental_pitcher_pit_cutoff_for_row(self.DAY_ROW) == "2024-09-28T23:59:59Z"

    def test_prediction_cutoff_prefers_explicit_cutoff_field_over_dates(self):
        row = {**self.NIGHT_ROW, "prediction_cutoff_utc": "2024-09-16T12:00:00Z"}
        assert _prediction_cutoff_for_row(row) == "2024-09-16T12:00:00Z"


@pytest.mark.skipif(
    not PIT_CACHE_DB.exists(),
    reason=f"{PIT_CACHE_DB} not present (gitignored local DB) — real-game fixture needs a dev environment",
)
class TestRealGameFixture745199:
    """The exact case the Fase 2B sweep found and this commit fixes:
    game_pk=745199, officialDate=2024-09-18, game_date (raw UTC
    timestamp)=2024-09-19. Permanent regression — if a future change
    reintroduces deriving cutoffs from game_date, this fails."""

    def _row(self):
        con = sqlite3.connect(PIT_CACHE_DB)
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT game_date, official_date FROM game_outcomes WHERE game_pk = 745199"
        ).fetchone()
        assert row is not None, "game_pk=745199 not found — is this the expected DB?"
        return row

    def test_745199_official_date_is_correct(self):
        row = self._row()
        assert row["game_date"] == "2024-09-19"
        assert row["official_date"] == "2024-09-18"

    def test_745199_cutoff_is_one_day_before_official_date_not_game_date(self):
        row = self._row()
        assert _team_tte_pit_cutoff_for_row(row) == "2024-09-17T23:59:59Z"
        assert _defense_pit_cutoff_for_row(row) == "2024-09-17T23:59:59Z"
        assert _bullpen_pit_cutoff_for_row(row) == "2024-09-17T23:59:59Z"
        # The pre-fix (buggy) cutoff would have been 2024-09-18T23:59:59Z —
        # confirm we're NOT producing that.
        assert _team_tte_pit_cutoff_for_row(row) != "2024-09-18T23:59:59Z"
