"""
Tests for F5 lambda derivation and MC F5 simulation.

Architecture (post-refactor):
  F5 lambda = pipeline λ after Pitcher Engine × F5_SCALE (0.575)
  The Pitcher Engine already applied FIP/xFIP/SIERA/Bayesian regression.
  Bullpen is excluded — starters pitch most or all of the first 5 innings.

The old _compute_f5_lambda (ERA-based, outside the pipeline) has been removed.
"""
import pytest
from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced, F5_SCALE


class TestF5ScaleConstant:

    def test_f5_scale_in_valid_range(self):
        # MLB empirical: 55–58% of runs score in the first 5 innings.
        assert 0.55 <= F5_SCALE <= 0.60

    def test_pipeline_f5_derivation(self):
        # Simulates what run_module now does: post-pitcher λ × F5_SCALE.
        post_pitcher_lh = 4.20
        post_pitcher_la = 3.80
        lh_f5 = round(post_pitcher_lh * F5_SCALE, 3)
        la_f5 = round(post_pitcher_la * F5_SCALE, 3)
        assert lh_f5 == pytest.approx(post_pitcher_lh * F5_SCALE, abs=0.001)
        assert la_f5 == pytest.approx(post_pitcher_la * F5_SCALE, abs=0.001)
        # F5 lambda must be strictly less than full-game lambda
        assert lh_f5 < post_pitcher_lh
        assert la_f5 < post_pitcher_la

    def test_better_pitcher_produces_lower_f5_lambda(self):
        # If the Pitcher Engine reduced λ (better pitcher), F5 lambda reflects that.
        lg_avg_lh = 4.50 * F5_SCALE        # league average after neutral pitcher
        elite_lh  = 3.20 * F5_SCALE        # λ reduced by elite pitcher
        assert elite_lh < lg_avg_lh


class TestF5InSimulator:
    """Verify that the MC handles F5 lambdas correctly."""

    def test_f5_lambdas_affect_f5_probs(self):
        r_home = monte_carlo_advanced(
            lh=4.5, la=4.5, n_max=300_000, analyze_f5=True,
            lh_f5=3.5, la_f5=1.5,
        )
        r_away = monte_carlo_advanced(
            lh=4.5, la=4.5, n_max=300_000, analyze_f5=True,
            lh_f5=1.5, la_f5=3.5,
        )
        assert r_home["f5_home"] > r_home["f5_away"]
        assert r_away["f5_away"] > r_away["f5_home"]

    def test_f5_probs_sum_to_one(self):
        r = monte_carlo_advanced(
            lh=4.3, la=3.8, n_max=300_000, analyze_f5=True,
            lh_f5=2.2, la_f5=1.9,
        )
        assert r["f5_home"] + r["f5_away"] + r["f5_draw"] == pytest.approx(1.0, abs=2e-4)

    def test_fallback_scale_used_when_no_f5_lambdas(self):
        # When lh_f5/la_f5 are None the simulator applies F5_SCALE internally.
        r = monte_carlo_advanced(lh=4.5, la=3.8, n_max=300_000, analyze_f5=True)
        assert "f5_home" in r
        assert r["f5_home"] + r["f5_away"] + r["f5_draw"] == pytest.approx(1.0, abs=2e-4)

    def test_symmetric_f5_lambdas_give_roughly_equal_probs(self):
        r = monte_carlo_advanced(
            lh=4.5, la=4.5, n_max=500_000, analyze_f5=True,
            lh_f5=2.5, la_f5=2.5,
        )
        assert abs(r["f5_home"] - r["f5_away"]) < 0.03

    def test_pipeline_derived_f5_produces_valid_probs(self):
        # Simulate what run_module does: compute lh_f5/la_f5 from post-pitcher λ.
        post_pitcher_lh, post_pitcher_la = 4.10, 3.75
        lh_f5 = round(post_pitcher_lh * F5_SCALE, 3)
        la_f5 = round(post_pitcher_la * F5_SCALE, 3)
        r = monte_carlo_advanced(
            lh=post_pitcher_lh * 1.05,  # full-game includes bullpen/context
            la=post_pitcher_la * 1.05,
            n_max=300_000, analyze_f5=True,
            lh_f5=lh_f5, la_f5=la_f5,
        )
        assert 0 < r["f5_home"] < 1
        assert 0 < r["f5_away"] < 1
        assert r["f5_home"] + r["f5_away"] + r["f5_draw"] == pytest.approx(1.0, abs=2e-4)
