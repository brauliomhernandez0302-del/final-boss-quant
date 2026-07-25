"""
Tests for the Monte Carlo engine.

Covers:
  - Probability calibration: symmetric input → p_home ≈ 0.50
  - Directional correctness: lh > la → p_home > p_away
  - p_home + p_away = 1.0 (exact, by construction)
  - Over/under probabilities bracket 0.5 around the mean total
  - Early stopping when SE < threshold
  - p_over + p_under + p_push = 1.0
  - analyze_f5 outputs (f5_home, f5_away, f5_draw) sum to 1.0
  - Auto total_line equals round(mean_total × 2) / 2
  - Input validation raises for out-of-range lambdas
"""
import math
import pytest
from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced, LIMITS


FAST = {"n_max": 500_000, "rng_seed": 42}   # deterministic, fast
PRECISE = {"n_max": 2_000_000, "rng_seed": 0}


class TestProbabilityCalibration:

    def test_symmetric_lambdas_near_fifty_fifty(self):
        r = monte_carlo_advanced(lh=4.5, la=4.5, **FAST)
        assert r["p_home"] == pytest.approx(0.50, abs=0.02)
        assert r["p_away"] == pytest.approx(0.50, abs=0.02)

    def test_higher_home_lambda_wins_more(self):
        r = monte_carlo_advanced(lh=6.0, la=3.0, **FAST)
        assert r["p_home"] > 0.65, f"Strong home team should win >65%, got {r['p_home']:.3f}"
        assert r["p_away"] < r["p_home"]

    def test_higher_away_lambda_wins_more(self):
        r = monte_carlo_advanced(lh=3.0, la=6.0, **FAST)
        assert r["p_away"] > 0.65
        assert r["p_home"] < r["p_away"]

    def test_probabilities_sum_to_one(self):
        r = monte_carlo_advanced(lh=4.5, la=3.9, **FAST)
        # p_home and p_away are computed with half-credit for ties,
        # so p_home + p_away = 1.0 exactly.
        assert r["p_home"] + r["p_away"] == pytest.approx(1.0, abs=1e-9)

    def test_mean_total_near_sum_of_lambdas(self):
        # model_walkoff=False: this tests the raw NB sampling mechanism
        # (does it preserve E[total]=lh+la), not the walk-off game rule —
        # with the rule on (default), mean_total is intentionally LOWER
        # than lh+la, since home doesn't always bat the 9th (see
        # audit_20260714/val_audit/reporte.md VAL-1.3 and the fix in
        # montecarlo/simulator.py).
        lh, la = 4.5, 3.8
        r = monte_carlo_advanced(lh=lh, la=la, model_walkoff=False, **FAST)
        expected = lh + la
        assert r["mean_total"] == pytest.approx(expected, abs=0.15)

    def test_mean_home_near_lh(self):
        # model_walkoff=False — see test_mean_total_near_sum_of_lambdas.
        r = monte_carlo_advanced(lh=5.0, la=3.5, model_walkoff=False, **FAST)
        assert r["mean_home"] == pytest.approx(5.0, abs=0.15)

    def test_mean_away_near_la(self):
        r = monte_carlo_advanced(lh=5.0, la=3.5, **FAST)
        assert r["mean_away"] == pytest.approx(3.5, abs=0.15)


class TestOverUnder:

    def test_explicit_total_line_yields_p_over(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, total_line=8.5, **FAST)
        assert "p_over" in r
        assert "p_under" in r
        assert "p_push" in r

    def test_over_under_push_sum_to_one(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, total_line=8.5, **FAST)
        total = r["p_over"] + r["p_under"] + r["p_push"]
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_line_at_mean_near_fifty_percent_over(self):
        # model_walkoff=False: with the rule on, home's runs (and hence
        # mean_total) are asymmetrically truncated only when home is ahead,
        # which skews the total distribution around its own mean away from
        # 50/50 — a real, intentional effect (see VAL-1.3), not what this
        # test is checking (that a line AT the mean bisects the distribution
        # for a walkoff-free/symmetric-noise total).
        lh, la = 4.5, 3.8
        r = monte_carlo_advanced(lh=lh, la=la, model_walkoff=False, **FAST)
        mean = r["mean_total"]
        # Use the mean as the line → p_over should be close to 50%
        r2 = monte_carlo_advanced(lh=lh, la=la, total_line=mean, model_walkoff=False, **FAST)
        assert r2["p_over"] == pytest.approx(0.50, abs=0.08)

    def test_low_line_high_p_over(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, total_line=1.5, **FAST)
        assert r["p_over"] > 0.95

    def test_high_line_low_p_over(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, total_line=30.0, **FAST)
        assert r["p_over"] < 0.01

    def test_auto_line_when_analyze_f5_true(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, **FAST, analyze_f5=True)
        expected_line = round(r["mean_total"] * 2) / 2
        assert r["total_line"] == pytest.approx(expected_line, abs=1e-9)


class TestF5Output:

    def test_f5_keys_present_when_requested(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, **FAST, analyze_f5=True)
        for key in ("f5_home", "f5_away", "f5_draw"):
            assert key in r, f"Missing key: {key}"

    def test_f5_probs_sum_to_one(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, **FAST, analyze_f5=True)
        total = r["f5_home"] + r["f5_away"] + r["f5_draw"]
        assert total == pytest.approx(1.0, abs=2e-4)

    def test_f5_absent_when_not_requested(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, **FAST, analyze_f5=False)
        assert "f5_home" not in r
        assert "f5_away" not in r

    def test_real_f5_lambdas_shift_f5_probs(self):
        # lh_f5 >> la_f5 → home should win F5 more often
        r = monte_carlo_advanced(lh=4.5, la=4.5, n_max=500_000, analyze_f5=True,
                                 lh_f5=3.5, la_f5=1.5)
        assert r["f5_home"] > r["f5_away"]

    def test_f5_draw_rate_is_plausible(self):
        # With Poisson(~2) each side, ties are common (~10-20%).
        r = monte_carlo_advanced(lh=4.0, la=4.0, n_max=500_000, analyze_f5=True)
        assert 0.10 < r["f5_draw"] < 0.35


class TestEarlyStoppingAndConvergence:

    def test_converges_early_for_large_lambda_difference(self):
        # lh=8, la=2 → very unbalanced; SE drops quickly
        r = monte_carlo_advanced(lh=8.0, la=2.0, n_max=5_000_000, rng_seed=1)
        assert r["converged_early"], "Highly unbalanced game should converge early"

    def test_n_reported_matches_actual_sims(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, **FAST)
        # n must be at least block size and at most n_max
        assert LIMITS.MIN_BLOCK <= r["n"] <= FAST["n_max"]

    def test_reproducible_with_seed(self):
        r1 = monte_carlo_advanced(lh=4.5, la=3.8, n_max=500_000, rng_seed=99)
        r2 = monte_carlo_advanced(lh=4.5, la=3.8, n_max=500_000, rng_seed=99)
        assert r1["p_home"] == r2["p_home"]
        assert r1["mean_total"] == r2["mean_total"]

    def test_different_seeds_give_similar_but_not_identical(self):
        r1 = monte_carlo_advanced(lh=4.5, la=3.8, n_max=300_000, rng_seed=1)
        r2 = monte_carlo_advanced(lh=4.5, la=3.8, n_max=300_000, rng_seed=2)
        assert abs(r1["p_home"] - r2["p_home"]) < 0.02  # statistically similar
        # Not identical (different seeds produce different samples)

    def test_more_sims_reduces_variance(self):
        # 25 seeds each → stable sample-std estimate.
        # Theoretical expectation: std ∝ 1/√n → 10× more sims → ~3.16× reduction.
        # Require only 1.5× reduction to avoid flakiness from small-sample noise.
        small_results = [
            monte_carlo_advanced(lh=4.5, la=3.8, n_max=50_000, block=10_000, rng_seed=i)["p_home"]
            for i in range(25)
        ]
        large_results = [
            monte_carlo_advanced(lh=4.5, la=3.8, n_max=500_000, rng_seed=i)["p_home"]
            for i in range(25)
        ]
        import statistics
        small_std = statistics.stdev(small_results)
        large_std = statistics.stdev(large_results)
        assert small_std > large_std * 1.5, (
            f"Expected small_std ({small_std:.5f}) > 1.5 × large_std ({large_std:.5f}); "
            "more sims should produce more consistent estimates"
        )


class TestInputValidation:

    def test_lambda_below_minimum_raises(self):
        with pytest.raises(ValueError, match="λ"):
            monte_carlo_advanced(lh=0.05, la=4.5, n_max=500_000)

    def test_lambda_above_maximum_raises(self):
        with pytest.raises(ValueError, match="λ"):
            monte_carlo_advanced(lh=4.5, la=25.0, n_max=500_000)

    def test_n_max_below_minimum_raises(self):
        with pytest.raises(ValueError):
            monte_carlo_advanced(lh=4.5, la=3.8, n_max=100, block=50)

    def test_total_line_zero_raises(self):
        with pytest.raises(ValueError):
            monte_carlo_advanced(lh=4.5, la=3.8, n_max=500_000, total_line=0.0)

    def test_valid_inputs_do_not_raise(self):
        # Should complete without error
        r = monte_carlo_advanced(lh=4.5, la=3.8, n_max=200_000, total_line=8.5)
        assert "p_home" in r


class TestPercentiles:

    def test_percentile_keys_present(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, **FAST)
        for p in [10, 25, 50, 75, 90]:
            assert f"p{p}" in r["percentiles"]

    def test_percentiles_ordered(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, **FAST)
        ps = r["percentiles"]
        assert ps["p10"] <= ps["p25"] <= ps["p50"] <= ps["p75"] <= ps["p90"]

    def test_median_near_mean(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, **FAST)
        assert abs(r["percentiles"]["p50"] - r["mean_total"]) < 1.0


class TestBivariatePoisson:
    """Bivariate Poisson: correlated λ noise between home and away scoring."""

    def test_rho_returned_in_results(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, rho_game=-0.06, **FAST)
        assert r["bivariate_rho"] == pytest.approx(-0.06)

    def test_marginal_means_preserved_with_negative_rho(self):
        # Bivariate correlation must not shift the marginal means.
        # model_walkoff=False: isolates this to the correlation mechanism
        # under test — walk-off truncation is a separate, intentional mean
        # shift (see VAL-1.3), not a correlation artifact.
        r = monte_carlo_advanced(lh=4.5, la=3.8, rho_game=-0.30, model_walkoff=False, **FAST)
        assert r["mean_home"] == pytest.approx(4.5, abs=0.15)
        assert r["mean_away"] == pytest.approx(3.8, abs=0.15)

    def test_probabilities_sum_to_one_with_rho(self):
        r = monte_carlo_advanced(lh=4.5, la=3.8, rho_game=-0.06, **FAST)
        assert r["p_home"] + r["p_away"] == pytest.approx(1.0, abs=1e-9)

    def test_negative_rho_compresses_total_variance(self):
        # With large noise, negative rho should reduce std_total vs rho=0.
        # Use large lambda_noise to make effect detectable.
        kwargs = {"lh": 4.5, "la": 4.5, "n_max": 2_000_000, "rng_seed": 7,
                  "lambda_noise": 0.25}
        r_ind = monte_carlo_advanced(**kwargs, rho_game=0.0)
        r_neg = monte_carlo_advanced(**kwargs, rho_game=-0.50)
        assert r_neg["std_total"] < r_ind["std_total"], (
            f"Negative rho should compress total std: {r_neg['std_total']:.4f} "
            f"vs {r_ind['std_total']:.4f}"
        )

    def test_positive_rho_widens_total_variance(self):
        kwargs = {"lh": 4.5, "la": 4.5, "n_max": 2_000_000, "rng_seed": 7,
                  "lambda_noise": 0.25}
        r_ind = monte_carlo_advanced(**kwargs, rho_game=0.0)
        r_pos = monte_carlo_advanced(**kwargs, rho_game=0.50)
        assert r_pos["std_total"] > r_ind["std_total"]

    def test_invalid_rho_raises(self):
        with pytest.raises(ValueError, match="rho_game"):
            monte_carlo_advanced(lh=4.5, la=3.8, rho_game=1.0, n_max=500_000)
        with pytest.raises(ValueError, match="rho_game"):
            monte_carlo_advanced(lh=4.5, la=3.8, rho_game=-1.5, n_max=500_000)

    def test_rho_zero_preserves_directional_advantage(self):
        # rho=0 must not distort win probabilities — lh > la → home should still win more.
        r = monte_carlo_advanced(lh=4.5, la=3.8, n_max=500_000, rng_seed=42, rho_game=0.0)
        assert r["p_home"] > r["p_away"], "Home advantage must hold with rho=0"
        assert r["p_home"] + r["p_away"] == pytest.approx(1.0, abs=1e-9)
