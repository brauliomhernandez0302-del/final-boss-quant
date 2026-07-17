"""
Tests for the HFA engine asymmetric pipeline.

Architecture (post-C2 fix, post-2026-07-11 uniform-home-mult fix):
  HFAEngine  → travel fatigue (λ_away only) + uniform home-win-prob
               correction (λ_home only); per-park crowd boost ELIMINATED (FIX C2)
  ParkWeatherEngine → park factor + weather (symmetric, separate engine)

FIX C2 (2026-05-27): PER-PARK crowd boost confirmed pure noise
(Pearson=-0.015, direction 53% random). The per-stadium lookup table stays
removed permanently.

Added 2026-07-11: a small UNIFORM (not per-park) home multiplier,
`_UNIFORM_HOME_MULT`, sized from a post-leak-fix calibration diagnostic
(both 2024 and 2025 independently under-predicted home win probability by
~1.6-1.7pp). This is a different signal from what C2 removed — a flat
constant applied identically to every game cannot reproduce the removed
feature's near-zero, park-varying correlation with real HFA variation.

The tests verify:
  1. The uniform home correction applies λ_home unchanged BY PARK (not
     per-stadium — that part of FIX C2 stays true) but is no longer 1.0
  2. Travel penalty only reduces λ_away, λ_home unaffected by travel
  3. Metadata keys match the current HFAEngine API
  4. Back-to-back is now handled by ContextualEngine (not HFAEngine)
"""
import pytest
from modules.baseball_module.hfa.hfa_engine import HFAEngine, get_adjusted_lambdas, _UNIFORM_HOME_MULT
from config import LEAGUE_AVG_RUNS


def _neutral_game(park="Unknown"):
    return {
        "park": {"name": park},
        "miles_traveled_away": 0,
        "time_zones_crossed_away": 0,
    }


class TestHFAPipelineMultiplicative:

    def test_neutral_game_uniform_home_mult_only(self):
        # Post-2026-07-11: λ_home is scaled by the uniform (not per-park) mult.
        lh, la = 4.5, 4.5
        lh_out, la_out, meta = get_adjusted_lambdas(lh, la, _neutral_game())
        assert lh_out == pytest.approx(lh * (1.0 + _UNIFORM_HOME_MULT))
        assert la_out == pytest.approx(la), "λ_away must not change without travel"
        assert meta["hfa_boost_runs"] == 0.0, "FIX C2: per-park hfa_boost_runs must still be zero"
        assert meta["hfa_mult"] == pytest.approx(1.0 + _UNIFORM_HOME_MULT)

    def test_home_advantage_uniform_not_per_park(self):
        # FIX C2 still holds for the PER-PARK component: park name doesn't
        # change the mult. But λ_home != λ_away even with no travel now,
        # because of the uniform (not per-park) home correction.
        lh, la = 4.5, 4.5
        lh_out, la_out, meta = get_adjusted_lambdas(lh, la, _neutral_game("Yankee Stadium"))
        assert lh_out == pytest.approx(la_out * (1.0 + _UNIFORM_HOME_MULT))
        assert meta["hfa_boost_runs"] == 0.0
        assert meta["hfa_mult"] == pytest.approx(1.0 + _UNIFORM_HOME_MULT)

    def test_metadata_keys_present(self):
        _, _, meta = get_adjusted_lambdas(4.5, 4.5, _neutral_game("Fenway Park"))
        for key in ("park_name", "hfa_boost_runs", "hfa_mult", "travel_penalty"):
            assert key in meta, f"Missing metadata key: {key}"

    def test_all_parks_same_lambda_no_travel(self):
        # FIX C2: crowd boost eliminated — all parks produce the same λ_home (no travel).
        lh_yankee, _, _ = get_adjusted_lambdas(4.5, 4.5, _neutral_game("Yankee Stadium"))
        lh_unknown, _, _ = get_adjusted_lambdas(4.5, 4.5, _neutral_game("Future Stadium"))
        assert lh_yankee == pytest.approx(lh_unknown), "FIX C2: park name must not affect λ_home"

    def test_all_parks_same_lambda_regardless_of_size(self):
        # FIX C2: Tropicana Field and Yankee Stadium produce identical λ_home (no travel).
        lh_yankee, _, _ = get_adjusted_lambdas(4.5, 4.5, _neutral_game("Yankee Stadium"))
        lh_trop, _, _   = get_adjusted_lambdas(4.5, 4.5, _neutral_game("Tropicana Field"))
        assert lh_yankee == pytest.approx(lh_trop), "FIX C2: park size must not affect λ_home"

    def test_travel_only_reduces_away(self):
        # Coast-to-coast (3 time zones) must reduce λ_away, not λ_home.
        game_travel = _neutral_game("Yankee Stadium")
        game_travel["miles_traveled_away"] = 3000
        game_travel["time_zones_crossed_away"] = 3

        game_none = _neutral_game("Yankee Stadium")

        lh_t, la_t, _ = get_adjusted_lambdas(4.5, 4.5, game_travel)
        lh_n, la_n, _ = get_adjusted_lambdas(4.5, 4.5, game_none)

        assert la_t < la_n, "Travel must reduce λ_away"
        assert abs(lh_t - lh_n) < 0.001, "λ_home must not be affected by away travel"

    def test_travel_penalty_metadata(self):
        game = _neutral_game()
        game["time_zones_crossed_away"] = 2
        _, _, meta = get_adjusted_lambdas(4.5, 4.5, game)
        assert meta["travel_penalty"] > 0, "travel_penalty must be positive when crossing time zones"

    def test_no_travel_penalty_zero(self):
        _, _, meta = get_adjusted_lambdas(4.5, 4.5, _neutral_game())
        assert meta["travel_penalty"] == 0.0

    def test_output_always_positive(self):
        for park in ["Coors Field", "Petco Park", "Yankee Stadium", "Unknown"]:
            lh_out, la_out, _ = get_adjusted_lambdas(1.0, 1.0, _neutral_game(park))
            assert lh_out > 0, f"λ_home must be positive at {park}"
            assert la_out > 0, f"λ_away must be positive at {park}"

    def test_hfa_mult_constant_regardless_of_lambda_scale_no_travel(self):
        # hfa_mult is the same uniform constant regardless of λ magnitude
        # (it's a ratio, applied before travel) when no travel.
        _, _, meta1 = get_adjusted_lambdas(4.0, 4.0, _neutral_game("Yankee Stadium"))
        _, _, meta2 = get_adjusted_lambdas(5.0, 5.0, _neutral_game("Yankee Stadium"))
        assert meta1["hfa_mult"] == pytest.approx(1.0 + _UNIFORM_HOME_MULT)
        assert meta2["hfa_mult"] == pytest.approx(1.0 + _UNIFORM_HOME_MULT)

    def test_back_to_back_no_longer_in_hfa(self):
        # back_to_back is now owned by ContextualEngine; HFAEngine ignores it.
        game_with = _neutral_game("Yankee Stadium")
        game_with["back_to_back_away"] = True
        game_without = _neutral_game("Yankee Stadium")

        lh_w, la_w, _ = get_adjusted_lambdas(4.5, 4.5, game_with)
        lh_n, la_n, _ = get_adjusted_lambdas(4.5, 4.5, game_without)
        assert lh_w == pytest.approx(lh_n), "HFA must not apply back_to_back to λ_home"
        assert la_w == pytest.approx(la_n), "HFA must not apply back_to_back to λ_away"
