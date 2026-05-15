"""
Tests verifying that pitcher regression corrects ERA in the pitcher dicts
BEFORE pitcher_engine reads them, not after.

Key invariants:
  - A lucky pitcher (low BABIP) should have a higher ERA when pitcher_engine runs.
  - An unlucky pitcher (high BABIP) should have a lower ERA when pitcher_engine runs.
  - Regression factor > 1 → ERA inflated upward → pitcher_engine produces higher lambda.
  - Regression factor < 1 → ERA deflated downward → pitcher_engine produces lower lambda.
  - When both regression and pitcher engine are enabled, the outcome of adjust_for_pitchers
    should differ from a run where no regression correction was applied.
"""
import copy
import pytest
from modules.baseball_module.context_engine.pitchers_regression import (
    PitcherRegressionEngine,
    calculate_pitcher_regression,
)
from modules.baseball_module.context_engine.pitcher_engine import adjust_for_pitchers


def _pitcher(era=4.20, era_last_5=4.20, **kwargs):
    base = {
        "name": "TestPitcher",
        "era": era,
        "fip": era,
        "whip": 1.30,
        "k_per_9": 8.5,
        "era_last_5": era_last_5,
        "days_rest": 4,
        "last_pitch_count": 90,
    }
    base.update(kwargs)
    return base


def _game_data(home_pitcher, away_pitcher):
    from config import LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP, LEAGUE_AVG_RUNS
    avg_team = {
        "woba": 0.320, "ops": 0.735, "wrc_plus": 100,
        "team_era": LEAGUE_AVG_ERA, "team_whip": LEAGUE_AVG_WHIP,
        "runs_allowed_per_game": LEAGUE_AVG_RUNS,
    }
    return {
        "pitcher_home": home_pitcher,
        "pitcher_away": away_pitcher,
        "home_team": {**avg_team, "name": "HomeTeam"},
        "away_team": {**avg_team, "name": "AwayTeam"},
        "park": {"name": "Neutral Park"},
        "miles_traveled_away": 0,
        "time_zones_crossed_away": 0,
        "bullpen_home": {},
        "bullpen_away": {},
    }


def _apply_regression_to_dicts(game_data, w_reg=1.0):
    """Replicate exactly what run_module does for PASO 3."""
    factor_away, _ = calculate_pitcher_regression(
        pitcher_stats=game_data.get('pitcher_away', {}),
        opponent_stats=game_data.get('home_team', {}),
    )
    factor_home, _ = calculate_pitcher_regression(
        pitcher_stats=game_data.get('pitcher_home', {}),
        opponent_stats=game_data.get('away_team', {}),
    )
    for role, factor in [('pitcher_away', factor_away), ('pitcher_home', factor_home)]:
        pd = game_data[role]
        adj = 1.0 + w_reg * (factor - 1.0)
        pd['era'] = round(pd['era'] * adj, 3)
        if pd.get('era_last_5') is not None:
            pd['era_last_5'] = round(pd['era_last_5'] * adj, 3)
    return factor_away, factor_home


class TestRegressionCorrectsPitcherDicts:

    def test_lucky_pitcher_era_increases(self):
        # Low BABIP → pitcher lucky → factor > 1 → ERA should go up
        lucky = _pitcher(era=3.50, era_last_5=3.50, babip=0.240, lob_pct=0.780, hr_fb_pct=0.080)
        avg   = _pitcher(era=4.20, era_last_5=4.20)
        gd = _game_data(home_pitcher=avg, away_pitcher=copy.deepcopy(lucky))

        original_era = gd['pitcher_away']['era']
        factor_away, _ = _apply_regression_to_dicts(gd)

        assert factor_away > 1.0, "Lucky away pitcher should have factor > 1"
        assert gd['pitcher_away']['era'] > original_era, "Lucky pitcher ERA must increase after regression"
        assert gd['pitcher_away']['era_last_5'] > original_era, "era_last_5 must also increase"

    def test_unlucky_pitcher_era_decreases(self):
        # High BABIP → pitcher unlucky → factor < 1 → ERA should go down
        unlucky = _pitcher(era=5.20, era_last_5=5.20, babip=0.340, lob_pct=0.680, hr_fb_pct=0.150)
        avg     = _pitcher(era=4.20, era_last_5=4.20)
        gd = _game_data(home_pitcher=avg, away_pitcher=copy.deepcopy(unlucky))

        original_era = gd['pitcher_away']['era']
        factor_away, _ = _apply_regression_to_dicts(gd)

        assert factor_away < 1.0, "Unlucky away pitcher should have factor < 1"
        assert gd['pitcher_away']['era'] < original_era, "Unlucky pitcher ERA must decrease after regression"

    def test_neutral_pitcher_era_unchanged(self):
        # Exactly average luck stats → factor = 1.0 → ERA unchanged
        neutral = _pitcher(era=4.20, era_last_5=4.20, babip=0.285, lob_pct=0.738, hr_fb_pct=0.106)
        gd = _game_data(home_pitcher=copy.deepcopy(neutral), away_pitcher=copy.deepcopy(neutral))

        original_home_era = gd['pitcher_home']['era']
        original_away_era = gd['pitcher_away']['era']
        _apply_regression_to_dicts(gd)

        assert gd['pitcher_home']['era'] == pytest.approx(original_home_era, abs=0.01)
        assert gd['pitcher_away']['era'] == pytest.approx(original_away_era, abs=0.01)

    def test_home_pitcher_dict_corrected_independently(self):
        # home pitcher luck affects home pitcher's ERA dict (which pitcher_engine uses for λ_away)
        lucky_home  = _pitcher(era=3.00, era_last_5=3.00, babip=0.230, lob_pct=0.800)
        avg_away    = _pitcher(era=4.20, era_last_5=4.20)
        gd = _game_data(home_pitcher=copy.deepcopy(lucky_home), away_pitcher=copy.deepcopy(avg_away))

        _, factor_home = _apply_regression_to_dicts(gd)

        assert factor_home > 1.0, "Lucky home pitcher → factor_home > 1"
        assert gd['pitcher_home']['era'] > lucky_home['era'], "Home pitcher ERA should be corrected upward"
        # Away pitcher should be nearly unchanged (average luck stats)
        assert gd['pitcher_away']['era'] == pytest.approx(avg_away['era'], abs=0.05)


class TestRegressionAffectsPitcherEngineOutput:
    """
    Pitcher regression is DISCONNECTED from the live pipeline (run_module PASO 3 removed).
    Reason: regression only corrected ERA, but pitcher_engine now uses SIERA→xFIP→FIP→ERA
    fallback, so ERA corrections have no effect when FIP is present (which it always is).

    These tests verify the new invariant: regression ERA mutation does NOT change
    pitcher_engine output when FIP is set (the primary metric supersedes ERA).
    """

    def _run_with_regression(self, away_pitcher_stats, lh=4.5, la=4.5):
        avg = _pitcher(era=4.20, era_last_5=4.20)
        gd = _game_data(home_pitcher=copy.deepcopy(avg), away_pitcher=copy.deepcopy(away_pitcher_stats))
        _apply_regression_to_dicts(gd)
        lh_out, la_out, _ = adjust_for_pitchers(lh, la, gd)
        return lh_out, la_out

    def _run_without_regression(self, away_pitcher_stats, lh=4.5, la=4.5):
        avg = _pitcher(era=4.20, era_last_5=4.20)
        gd = _game_data(home_pitcher=copy.deepcopy(avg), away_pitcher=copy.deepcopy(away_pitcher_stats))
        lh_out, la_out, _ = adjust_for_pitchers(lh, la, gd)
        return lh_out, la_out

    def test_era_correction_has_no_effect_when_fip_present(self):
        # pitcher_engine uses FIP as primary (ERA fallback only when FIP absent).
        # Regression corrects ERA but not FIP, so output is identical.
        lucky = _pitcher(era=3.00, era_last_5=3.00, babip=0.230, lob_pct=0.800)
        lh_corrected, _ = self._run_with_regression(lucky)
        lh_raw, _       = self._run_without_regression(lucky)
        assert lh_corrected == pytest.approx(lh_raw, abs=1e-6), (
            "ERA correction must not affect pitcher_engine when FIP is the primary metric"
        )

    def test_era_correction_has_no_effect_unlucky_pitcher(self):
        unlucky = _pitcher(era=5.50, era_last_5=5.50, babip=0.350, lob_pct=0.670)
        lh_corrected, _ = self._run_with_regression(unlucky)
        lh_raw, _       = self._run_without_regression(unlucky)
        assert lh_corrected == pytest.approx(lh_raw, abs=1e-6)

    def test_lambda_away_unaffected_by_away_pitcher_luck(self):
        # Away pitcher only affects λ_home directly; λ_away unchanged regardless.
        lucky = _pitcher(era=3.00, era_last_5=3.00, babip=0.230, lob_pct=0.800)
        _, la_corrected = self._run_with_regression(lucky)
        _, la_raw       = self._run_without_regression(lucky)
        assert abs(la_corrected - la_raw) < 0.15
