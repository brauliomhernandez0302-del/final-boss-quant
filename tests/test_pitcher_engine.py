"""
Tests for the pitcher engine delta-formula combiner and individual adjustments.

The combiner uses the delta formula (not a product of factors):
    total_multiplier = 1.0 + Σ((factor_i − 1.0) × weight_i)

Key invariants:
  - All factors = 1.0 → total = 1.0 (league-average neutrality)
  - Only one factor differs → delta is proportional to that factor's weight
  - Away pitcher reduces λ_home; home pitcher reduces λ_away (not swapped)
  - Elite ERA → quality_mult < 1.0; bad ERA → quality_mult > 1.0
  - Weights sum to 1.0 (internal consistency)
"""
import pytest
import numpy as np
from modules.baseball_module.context_engine.pitcher_engine import PitcherEngine


def _engine():
    return PitcherEngine()


def _avg_pitcher(**kwargs):
    defaults = {
        "name": "TestPitcher",
        "era": 4.20,
        "fip": 4.20,
        "whip": 1.30,
        "k_per_9": 8.5,
        "era_last_5": 4.20,
        "days_rest": 4,
        "last_pitch_count": 90,
    }
    defaults.update(kwargs)
    return defaults


def _game_data(home_pitcher=None, away_pitcher=None, park="Yankee Stadium"):
    from config import LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP, LEAGUE_AVG_RUNS
    avg_team = {
        "woba": 0.320, "ops": 0.735, "wrc_plus": 100,
        "team_era": LEAGUE_AVG_ERA, "team_whip": LEAGUE_AVG_WHIP,
        "runs_allowed_per_game": LEAGUE_AVG_RUNS,
    }
    return {
        "pitcher_home": home_pitcher or _avg_pitcher(name="HomePitcher"),
        "pitcher_away": away_pitcher or _avg_pitcher(name="AwayPitcher"),
        "home_team": {**avg_team, "name": "HomeTeam"},
        "away_team": {**avg_team, "name": "AwayTeam"},
        "park": {"name": park},
        "miles_traveled_away": 0,
        "time_zones_crossed_away": 0,
        "bullpen_home": {},
        "bullpen_away": {},
    }


class TestDeltaFormula:

    def test_all_factors_neutral_gives_one(self):
        # When every sub-factor returns 1.0, the combiner must produce 1.0.
        engine = _engine()
        adj = engine._calculate_pitcher_adjustment(
            _avg_pitcher(), _game_data(), is_home=True
        )
        assert adj["total_multiplier"] == pytest.approx(1.0, abs=0.05)

    def test_delta_formula_math(self):
        # Build a scenario where only quality_mult differs and verify the delta.
        engine = _engine()
        weights = engine.weights

        # Supply a pitcher that produces a known quality_mult.
        # ERA=3.00, FIP=3.00 → composite=3.00 → quality_mult = 3.00/4.20 ≈ 0.7143
        pitcher = _avg_pitcher(era=3.00, fip=3.00, era_last_5=3.00)
        adj = engine._calculate_pitcher_adjustment(
            pitcher, _game_data(), is_home=True
        )
        q = adj["quality_mult"]
        f = adj["form_mult"]
        m = adj["matchup_mult"]
        fg = adj["fatigue_mult"]
        pk = adj["park_mult"]
        tr = adj["travel_mult"]
        bp = adj["bullpen_mult"]

        expected_total = 1.0 + (
            (q  - 1) * weights["pitcher_quality"] +
            (f  - 1) * weights["pitcher_form"] +
            (m  - 1) * weights["pitcher_matchup"] +
            (fg - 1) * weights["pitcher_fatigue"] +
            (pk - 1) * weights["park_for_pitcher"] +
            (tr - 1) * weights["travel_pitcher"] +
            (bp - 1) * weights["bullpen_quality"]
        )
        assert adj["total_multiplier"] == pytest.approx(expected_total, abs=1e-6)

    def test_single_factor_scales_by_weight(self):
        # When only quality differs from 1.0 by Δ, the total changes by Δ × quality_weight.
        engine = _engine()
        # Neutral pitcher (q≈1.0, all others ≈1.0)
        base_adj = engine._calculate_pitcher_adjustment(
            _avg_pitcher(era=4.20, fip=4.20), _game_data(), is_home=True
        )
        # Elite pitcher: ERA/FIP=3.00/3.00 → q≈0.714
        elite_adj = engine._calculate_pitcher_adjustment(
            _avg_pitcher(era=3.00, fip=3.00, era_last_5=3.00), _game_data(), is_home=True
        )
        delta_total = elite_adj["total_multiplier"] - base_adj["total_multiplier"]
        delta_q     = elite_adj["quality_mult"]     - base_adj["quality_mult"]
        # The shift in total should be approximately delta_q × quality_weight
        expected = delta_q * engine.weights["pitcher_quality"]
        # Tolerance is loose because form_mult also shifts slightly with ERA changes
        assert abs(delta_total - expected) < 0.08

    def test_weights_sum_to_one(self):
        engine = _engine()
        total = sum(engine.weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestPitcherQualityMultiplier:

    def test_elite_below_one(self):
        engine = _engine()
        mult = engine._adjust_pitcher_quality({"era": 2.50, "fip": 2.50})
        assert mult < 1.0

    def test_league_avg_is_one(self):
        engine = _engine()
        mult = engine._adjust_pitcher_quality({"era": 4.20, "fip": 4.20})
        assert mult == pytest.approx(1.0, abs=1e-6)

    def test_bad_above_one(self):
        engine = _engine()
        mult = engine._adjust_pitcher_quality({"era": 6.00, "fip": 6.00})
        assert mult > 1.0

    def test_composite_weights_siera50_xfip30_era20(self):
        # When xFIP and SIERA are absent, both fall back to FIP, which falls
        # back to ERA.  So composite = ERA*0.50 + ERA*0.30 + ERA*0.20 = ERA.
        engine = _engine()
        era = 3.00
        composite = era * 0.50 + era * 0.30 + era * 0.20   # all three = ERA
        expected = float(np.clip(composite / 4.20, 0.70, 1.30))
        result = engine._adjust_pitcher_quality({"era": era})
        assert result == pytest.approx(expected, abs=1e-6)

    def test_real_xfip_siera_used_when_available(self):
        # Real xFIP/SIERA from FanGraphs should dominate ERA in the composite.
        engine = _engine()
        # Lucky pitcher: great ERA but average xFIP/SIERA
        mult_lucky   = engine._adjust_pitcher_quality({"era": 2.00, "xfip": 4.20, "siera": 4.20})
        mult_genuine = engine._adjust_pitcher_quality({"era": 2.00, "xfip": 2.00, "siera": 2.00})
        assert mult_lucky > mult_genuine, "Real xFIP/SIERA should override a lucky ERA"

    def test_xwoba_penalty_increases_lambda(self):
        # Pitcher allowing high xwOBA (0.360 vs 0.320 avg) should give mult > 1.0
        engine = _engine()
        result_high = engine._adjust_pitcher_quality({"era": 4.20, "est_woba": 0.360})
        result_avg  = engine._adjust_pitcher_quality({"era": 4.20, "est_woba": 0.320})
        assert result_high > result_avg

    def test_barrel_rate_penalty_increases_lambda(self):
        # Pitcher allowing high barrel% (14% vs 8% avg) should give mult > 1.0
        engine = _engine()
        result_high = engine._adjust_pitcher_quality({"era": 4.20, "brl_percent": 14.0})
        result_avg  = engine._adjust_pitcher_quality({"era": 4.20, "brl_percent": 8.0})
        assert result_high > result_avg

    def test_clamped_at_0_70_minimum(self):
        engine = _engine()
        result = engine._adjust_pitcher_quality({"era": 0.50, "fip": 0.50})
        assert result == pytest.approx(0.70)

    def test_clamped_at_1_30_maximum(self):
        engine = _engine()
        result = engine._adjust_pitcher_quality({"era": 10.0, "fip": 10.0})
        assert result == pytest.approx(1.30)

    def test_fip_downweighs_lucky_era(self):
        # ERA 2.00 (lucky) but FIP 4.20 (average) → composite closer to average.
        engine = _engine()
        mult_lucky = engine._adjust_pitcher_quality({"era": 2.00, "fip": 4.20})
        mult_true   = engine._adjust_pitcher_quality({"era": 2.00, "fip": 2.00})
        assert mult_lucky > mult_true, "Blending FIP should temper a lucky low ERA"


class TestPitcherEngineDirectionality:
    """Away pitcher affects λ_home; home pitcher affects λ_away — never swapped."""

    def _run(self, home_p, away_p, lh=4.5, la=4.5):
        from modules.baseball_module.context_engine.pitcher_engine import adjust_for_pitchers
        gd = _game_data(home_pitcher=home_p, away_pitcher=away_p)
        return adjust_for_pitchers(lh, la, gd)

    def test_elite_away_pitcher_reduces_home_scoring(self):
        avg_p   = _avg_pitcher(era=4.20, fip=4.20)
        elite_p = _avg_pitcher(era=2.50, fip=2.80)
        lh_elite, la_elite, _ = self._run(avg_p, elite_p)
        lh_avg,   la_avg,   _ = self._run(avg_p, avg_p)
        assert lh_elite < lh_avg, "Elite away pitcher → lower λ_home"

    def test_elite_home_pitcher_reduces_away_scoring(self):
        avg_p   = _avg_pitcher(era=4.20, fip=4.20)
        elite_p = _avg_pitcher(era=2.50, fip=2.80)
        lh_elite, la_elite, _ = self._run(elite_p, avg_p)
        lh_avg,   la_avg,   _ = self._run(avg_p, avg_p)
        assert la_elite < la_avg, "Elite home pitcher → lower λ_away"

    def test_away_pitcher_does_not_affect_away_lambda(self):
        avg_p   = _avg_pitcher(era=4.20, fip=4.20)
        elite_p = _avg_pitcher(era=2.00, fip=2.00)
        _, la_elite, _ = self._run(avg_p, elite_p)
        _, la_avg,   _ = self._run(avg_p, avg_p)
        # Away pitcher quality should not significantly change λ_away
        # (bullpen is the only cross-over, and it's small)
        assert abs(la_elite - la_avg) < 0.3, "Away pitcher should not materially affect λ_away"


class TestFatigueAdjustment:

    def test_short_rest_increases_multiplier(self):
        engine = _engine()
        normal  = engine._adjust_pitcher_fatigue({"days_rest": 4, "last_pitch_count": 90})
        short   = engine._adjust_pitcher_fatigue({"days_rest": 2, "last_pitch_count": 90})
        assert short > normal

    def test_high_pitch_count_increases_multiplier(self):
        engine = _engine()
        normal = engine._adjust_pitcher_fatigue({"days_rest": 4, "last_pitch_count": 90})
        heavy  = engine._adjust_pitcher_fatigue({"days_rest": 4, "last_pitch_count": 120})
        assert heavy > normal

    def test_normal_rest_is_baseline(self):
        engine = _engine()
        result = engine._adjust_pitcher_fatigue({"days_rest": 4, "last_pitch_count": 90})
        # Under normal rest, multiplier should be exactly 1.0
        assert result == pytest.approx(1.0, abs=1e-6)
