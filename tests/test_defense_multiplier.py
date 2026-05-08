"""
Tests for the defense multiplier in both HFA engine and AutoCalibrator.

Critical invariant: elite defense (low ERA/WHIP/RA) → multiplier < 1.0
                    weak defense (high ERA/WHIP/RA) → multiplier > 1.0
                    league-average defense           → multiplier ≈ 1.0

The bug this guards against: previously the formula was inverted
(league_avg / stat instead of stat / league_avg), making elite pitching
look like easy scoring. These tests pin the correct direction.
"""
import pytest
from config import LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP, LEAGUE_AVG_RUNS


ELITE = {
    "team_era": 3.00,
    "team_whip": 1.05,
    "runs_allowed_per_game": 3.40,
}
AVERAGE = {
    "team_era": LEAGUE_AVG_ERA,
    "team_whip": LEAGUE_AVG_WHIP,
    "runs_allowed_per_game": LEAGUE_AVG_RUNS,
}
WEAK = {
    "team_era": 5.50,
    "team_whip": 1.65,
    "runs_allowed_per_game": 5.60,
}


class TestHFADefenseMultiplier:
    """HFAEngine._calculate_defense_multiplier — team ERA/WHIP/RA, clamped [0.78, 1.22]."""

    def _mult(self, team_dict):
        from modules.baseball_module.hfa.hfa_engine import HFAEngine
        return HFAEngine()._calculate_defense_multiplier(team_dict)

    def test_elite_defense_below_one(self):
        result = self._mult(ELITE)
        assert result < 1.0, f"Elite defense should suppress scoring: got {result:.4f}"

    def test_weak_defense_above_one(self):
        result = self._mult(WEAK)
        assert result > 1.0, f"Weak defense should inflate scoring: got {result:.4f}"

    def test_average_defense_is_one(self):
        result = self._mult(AVERAGE)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_monotone_direction(self):
        # Replacing each metric independently with a better value must reduce the multiplier.
        import copy
        base = {
            "team_era": 4.50,
            "team_whip": 1.40,
            "runs_allowed_per_game": 4.80,
        }
        better_era  = {**base, "team_era": 3.00}
        better_whip = {**base, "team_whip": 1.05}
        better_ra   = {**base, "runs_allowed_per_game": 3.50}

        b = self._mult(base)
        assert self._mult(better_era)  < b
        assert self._mult(better_whip) < b
        assert self._mult(better_ra)   < b

    def test_lower_clamp_at_0_78(self):
        # Absurdly elite pitching: ERA=1.00 → clamped floor at 0.78
        extreme = {"team_era": 1.00, "team_whip": 0.70, "runs_allowed_per_game": 1.50}
        result = self._mult(extreme)
        assert result == pytest.approx(0.78)

    def test_upper_clamp_at_1_22(self):
        # Absurdly weak pitching: ERA=9.00 → clamped ceiling at 1.22
        extreme = {"team_era": 9.00, "team_whip": 2.50, "runs_allowed_per_game": 8.00}
        result = self._mult(extreme)
        assert result == pytest.approx(1.22)

    def test_missing_fields_use_league_averages(self):
        # When fields are absent the defaults are league averages → mult ≈ 1.0
        result = self._mult({})
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_weight_era_40_whip_35_ra_25(self):
        # Verify the weighted formula directly.
        team = {"team_era": 3.00, "team_whip": 1.30, "runs_allowed_per_game": 4.50}
        era_m  = 3.00 / LEAGUE_AVG_ERA
        whip_m = 1.30 / LEAGUE_AVG_WHIP
        ra_m   = 4.50 / LEAGUE_AVG_RUNS
        expected = era_m * 0.40 + whip_m * 0.35 + ra_m * 0.25
        import numpy as np
        expected = float(np.clip(expected, 0.78, 1.22))
        assert self._mult(team) == pytest.approx(expected, abs=1e-6)


class TestCalibratorDefenseMultiplier:
    """
    LambdaCalibrator._calculate_defense_multiplier — same directional invariant.
    The calibrator takes the *opponent* dict (team that is pitching),
    so a strong opponent pitching staff should reduce the scoring team's lambda.
    """

    def _mult(self, opponent_dict):
        from modules.baseball_module.calibration.auto_calibrator import LambdaCalibrator
        return LambdaCalibrator()._calculate_defense_multiplier(opponent_dict)

    def test_elite_opponent_below_one(self):
        result = self._mult(ELITE)
        assert result < 1.0, f"Elite opponent pitching should suppress scoring: got {result:.4f}"

    def test_weak_opponent_above_one(self):
        result = self._mult(WEAK)
        assert result > 1.0, f"Weak opponent pitching should inflate scoring: got {result:.4f}"

    def test_average_opponent_near_one(self):
        result = self._mult(AVERAGE)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_elite_and_weak_are_symmetric_around_one(self):
        # The gap above 1.0 for weak should be roughly comparable
        # to the gap below 1.0 for elite (not necessarily equal, but within 2×).
        elite_gap = 1.0 - self._mult(ELITE)
        weak_gap  = self._mult(WEAK) - 1.0
        assert 0.3 <= elite_gap / weak_gap <= 3.0, (
            f"Asymmetric gaps: elite_gap={elite_gap:.4f}, weak_gap={weak_gap:.4f}"
        )
