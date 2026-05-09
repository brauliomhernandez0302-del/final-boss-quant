"""
Tests for the F5 lambda computation.

_compute_f5_lambda(pitcher_stats, bullpen_era) computes expected runs
scored against a pitcher in the first 5 innings:

    starter_f5_ip = min(avg_innings_per_start, 5.0)
    bullpen_f5_ip = 5.0 - starter_f5_ip
    f5_lambda = starter_f5_ip × (f5_era / 9) + bullpen_f5_ip × (bullpen_era / 9)

Key invariants:
  - No avg_innings_per_start → returns None (simulator falls back to F5_SCALE)
  - avg_ips >= 5 → bullpen contributes 0 innings in F5
  - f5_era absent → falls back to full-game ERA
  - Result is always positive when inputs are positive
  - Deeper starter (higher avg_ips) with lower ERA → lower F5 lambda
"""
import pytest
from modules.baseball_module.core.run_module import _compute_f5_lambda


class TestComputeF5Lambda:

    def test_returns_none_when_no_avg_ips(self):
        # Simulator must get None so it falls back to F5_SCALE.
        result = _compute_f5_lambda({"era": 4.00}, bullpen_era=4.20)
        assert result is None

    def test_ace_going_6_ip_no_bullpen_contribution(self):
        # avg_ips=6.0 → starter covers all 5 innings; bullpen_f5_ip=0
        result = _compute_f5_lambda(
            {"avg_innings_per_start": 6.0, "f5_era": 2.80, "era": 3.50},
            bullpen_era=4.20,
        )
        expected = 5.0 * (2.80 / 9)
        assert result == pytest.approx(expected, abs=0.001)  # rounded to 3dp

    def test_short_starter_blends_bullpen(self):
        # avg_ips=4.0 → starter covers 4 innings; bullpen covers 1 in F5
        f5_era, bullpen_era = 5.40, 4.20
        result = _compute_f5_lambda(
            {"avg_innings_per_start": 4.0, "f5_era": f5_era, "era": 5.00},
            bullpen_era=bullpen_era,
        )
        expected = 4.0 * (f5_era / 9) + 1.0 * (bullpen_era / 9)
        assert result == pytest.approx(expected, abs=0.001)

    def test_exactly_5_ip_average_no_bullpen(self):
        result = _compute_f5_lambda(
            {"avg_innings_per_start": 5.0, "f5_era": 3.60},
            bullpen_era=4.50,
        )
        expected = 5.0 * (3.60 / 9)
        assert result == pytest.approx(expected, abs=0.001)

    def test_fallback_to_full_era_when_f5_era_absent(self):
        result = _compute_f5_lambda(
            {"avg_innings_per_start": 5.5, "era": 4.00},
            bullpen_era=4.20,
        )
        expected = 5.0 * (4.00 / 9)  # avg_ips >= 5, so no bullpen portion
        assert result == pytest.approx(expected, abs=0.001)

    def test_result_always_positive(self):
        for avg_ips in [1.5, 3.0, 4.5, 5.0, 6.5]:
            result = _compute_f5_lambda(
                {"avg_innings_per_start": avg_ips, "f5_era": 4.00, "era": 4.00},
                bullpen_era=4.20,
            )
            assert result > 0, f"F5 lambda must be positive for avg_ips={avg_ips}"

    def test_elite_starter_lower_than_replacement(self):
        elite = _compute_f5_lambda(
            {"avg_innings_per_start": 6.0, "f5_era": 2.80}, bullpen_era=4.20
        )
        replacement = _compute_f5_lambda(
            {"avg_innings_per_start": 3.0, "f5_era": 6.00}, bullpen_era=4.50
        )
        assert elite < replacement, "Elite starter should produce lower expected F5 runs"

    def test_bullpen_quality_matters_when_starter_leaves_early(self):
        good_bp = _compute_f5_lambda(
            {"avg_innings_per_start": 3.0, "f5_era": 4.20}, bullpen_era=3.00
        )
        bad_bp = _compute_f5_lambda(
            {"avg_innings_per_start": 3.0, "f5_era": 4.20}, bullpen_era=6.00
        )
        assert good_bp < bad_bp, "Better bullpen should reduce F5 expected runs"

    def test_deeper_starter_with_same_era_lowers_f5(self):
        # avg_ips=6 (all 5 covered by starter at ERA=4.20) vs
        # avg_ips=3 (only 3 covered; 2 innings of worse bullpen)
        deep     = _compute_f5_lambda({"avg_innings_per_start": 6.0, "f5_era": 4.20}, bullpen_era=5.00)
        shallow  = _compute_f5_lambda({"avg_innings_per_start": 3.0, "f5_era": 4.20}, bullpen_era=5.00)
        assert deep < shallow, "Deeper starter prevents worse bullpen exposure"

    def test_avg_ips_capped_at_5(self):
        # avg_ips=8 should give same result as avg_ips=5 (cap at 5 innings for F5)
        for ips in [5.0, 6.0, 7.0, 9.0]:
            result = _compute_f5_lambda(
                {"avg_innings_per_start": ips, "f5_era": 3.50}, bullpen_era=4.20
            )
            expected = 5.0 * (3.50 / 9)
            assert result == pytest.approx(expected, abs=0.001), f"Failed for avg_ips={ips}"

    def test_empty_pitcher_dict_returns_none(self):
        assert _compute_f5_lambda({}, bullpen_era=4.20) is None


class TestF5InSimulator:
    """Verify that real F5 lambdas are used when passed to monte_carlo_advanced."""

    def test_f5_lambdas_affect_f5_probs(self):
        from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced

        # Large home advantage in F5 (home expected more runs)
        r_home = monte_carlo_advanced(
            lh=4.5, la=4.5, n_max=300_000, analyze_f5=True,
            lh_f5=3.5, la_f5=1.5,
        )
        # Large away advantage in F5
        r_away = monte_carlo_advanced(
            lh=4.5, la=4.5, n_max=300_000, analyze_f5=True,
            lh_f5=1.5, la_f5=3.5,
        )
        assert r_home["f5_home"] > r_home["f5_away"], "Higher lh_f5 should produce more home F5 wins"
        assert r_away["f5_away"] > r_away["f5_home"], "Higher la_f5 should produce more away F5 wins"

    def test_f5_probs_sum_to_one(self):
        from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced
        r = monte_carlo_advanced(lh=4.3, la=3.8, n_max=300_000, analyze_f5=True,
                                 lh_f5=2.2, la_f5=1.9)
        total = r["f5_home"] + r["f5_away"] + r["f5_draw"]
        assert total == pytest.approx(1.0, abs=2e-4)

    def test_fallback_scale_used_when_no_f5_lambdas(self):
        from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced, F5_SCALE
        # When lh_f5/la_f5 are None, F5_SCALE is applied to full-game noise.
        # The result should still be valid probabilities summing to 1.
        r = monte_carlo_advanced(lh=4.5, la=3.8, n_max=300_000, analyze_f5=True)
        assert "f5_home" in r
        assert r["f5_home"] + r["f5_away"] + r["f5_draw"] == pytest.approx(1.0, abs=2e-4)
        # With slightly higher lh, home F5 win prob should exceed away F5 win prob
        assert r["f5_home"] > r["f5_away"]
