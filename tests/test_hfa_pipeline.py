"""
Tests for the HFA engine asymmetric pipeline.

Architecture (post-C2 fix):
  HFAEngine  → travel fatigue (λ_away only); crowd boost ELIMINATED (FIX C2)
  ParkWeatherEngine → park factor + weather (symmetric, separate engine)

FIX C2 (2026-05-27): crowd boost confirmed pure noise (Pearson=-0.015,
direction 53% random). hfa_boost hard-coded to 0.0. Only travel
fatigue remains as an asymmetric HFA adjustment.

The tests verify:
  1. Crowd boost is ZERO — λ_home unchanged by HFA engine (FIX C2)
  2. Travel penalty only reduces λ_away, λ_home unaffected
  3. Metadata keys match the current HFAEngine API
  4. Back-to-back is now handled by ContextualEngine (not HFAEngine)
"""
import pytest
from modules.baseball_module.hfa.hfa_engine import HFAEngine, get_adjusted_lambdas
from config import LEAGUE_AVG_RUNS


def _neutral_game(park="Unknown"):
    return {
        "park": {"name": park},
        "miles_traveled_away": 0,
        "time_zones_crossed_away": 0,
    }


class TestHFAPipelineMultiplicative:

    def test_neutral_game_no_crowd_boost(self):
        # FIX C2: crowd boost eliminated — λ_home must NOT change without travel.
        lh, la = 4.5, 4.5
        lh_out, la_out, meta = get_adjusted_lambdas(lh, la, _neutral_game())
        assert lh_out == pytest.approx(lh), "FIX C2: crowd boost eliminated, λ_home must be unchanged"
        assert la_out == pytest.approx(la), "λ_away must not change without travel"
        assert meta["hfa_boost_runs"] == 0.0, "FIX C2: hfa_boost_runs must be zero"
        assert meta["hfa_mult"] == 1.0,       "FIX C2: hfa_mult must be exactly 1.0"

    def test_home_advantage_only_from_travel(self):
        # FIX C2: with no travel, λ_home == λ_away (crowd boost gone).
        lh, la = 4.5, 4.5
        lh_out, la_out, meta = get_adjusted_lambdas(lh, la, _neutral_game("Yankee Stadium"))
        assert lh_out == pytest.approx(la_out), "FIX C2: no crowd boost → λ_home == λ_away with no travel"
        assert meta["hfa_boost_runs"] == 0.0
        assert meta["hfa_mult"] == 1.0

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

    def test_hfa_mult_always_one_no_travel(self):
        # FIX C2: hfa_mult is always exactly 1.0 when no travel (crowd boost gone).
        _, _, meta1 = get_adjusted_lambdas(4.0, 4.0, _neutral_game("Yankee Stadium"))
        _, _, meta2 = get_adjusted_lambdas(5.0, 5.0, _neutral_game("Yankee Stadium"))
        assert meta1["hfa_mult"] == 1.0
        assert meta2["hfa_mult"] == 1.0

    def test_back_to_back_no_longer_in_hfa(self):
        # back_to_back is now owned by ContextualEngine; HFAEngine ignores it.
        game_with = _neutral_game("Yankee Stadium")
        game_with["back_to_back_away"] = True
        game_without = _neutral_game("Yankee Stadium")

        lh_w, la_w, _ = get_adjusted_lambdas(4.5, 4.5, game_with)
        lh_n, la_n, _ = get_adjusted_lambdas(4.5, 4.5, game_without)
        assert lh_w == pytest.approx(lh_n), "HFA must not apply back_to_back to λ_home"
        assert la_w == pytest.approx(la_n), "HFA must not apply back_to_back to λ_away"
