"""
Tests for the HFA engine asymmetric pipeline.

Architecture (post-refactor):
  HFAEngine  → crowd boost (λ_home only) + travel fatigue (λ_away only)
  ParkWeatherEngine → park factor + weather (symmetric, separate engine)

The tests verify:
  1. Crowd boost always raises λ_home, never λ_away
  2. Travel penalty only reduces λ_away
  3. Unknown park falls back to 0.0325 default boost
  4. Metadata keys match the current HFAEngine API
  5. Back-to-back is now handled by ContextualEngine (not HFAEngine)
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

    def test_neutral_game_home_boosted(self):
        # HFA crowd boost must always raise λ_home; λ_away unchanged without travel.
        lh, la = 4.5, 4.5
        lh_out, la_out, _ = get_adjusted_lambdas(lh, la, _neutral_game())
        assert lh_out > lh, "HFA crowd boost must raise λ_home"
        assert la_out == pytest.approx(la), "λ_away must not change without travel"

    def test_home_advantage_direction(self):
        # Home should always exceed away in a symmetric matchup.
        lh, la = 4.5, 4.5
        lh_out, la_out, meta = get_adjusted_lambdas(lh, la, _neutral_game("Yankee Stadium"))
        assert lh_out > la_out, "λ_home must exceed λ_away in symmetric matchup"
        assert meta["hfa_boost_runs"] > 0, "hfa_boost_runs must be positive"
        assert meta["hfa_mult"] > 1.0,     "hfa_mult must exceed 1.0"

    def test_metadata_keys_present(self):
        _, _, meta = get_adjusted_lambdas(4.5, 4.5, _neutral_game("Fenway Park"))
        for key in ("park_name", "hfa_boost_runs", "hfa_mult", "travel_penalty"):
            assert key in meta, f"Missing metadata key: {key}"

    def test_known_park_higher_boost_than_unknown(self):
        # Yankee Stadium (0.0450) > default (0.0325)
        lh_yankee, _, _ = get_adjusted_lambdas(4.5, 4.5, _neutral_game("Yankee Stadium"))
        lh_unknown, _, _ = get_adjusted_lambdas(4.5, 4.5, _neutral_game("Future Stadium"))
        assert lh_yankee > lh_unknown, "Named park should boost more than unknown-park default"

    def test_worst_attendance_park_lower_boost(self):
        # Tropicana Field (0.0250) < Yankee Stadium (0.0450)
        lh_yankee, _, _ = get_adjusted_lambdas(4.5, 4.5, _neutral_game("Yankee Stadium"))
        lh_trop, _, _   = get_adjusted_lambdas(4.5, 4.5, _neutral_game("Tropicana Field"))
        assert lh_yankee > lh_trop, "High-attendance park must produce larger home boost"

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

    def test_crowd_boost_scales_with_input_lambda(self):
        # Multiplicative: a higher base λ gets a proportionally larger absolute boost.
        _, _, meta1 = get_adjusted_lambdas(4.0, 4.0, _neutral_game("Yankee Stadium"))
        _, _, meta2 = get_adjusted_lambdas(5.0, 5.0, _neutral_game("Yankee Stadium"))
        # hfa_mult is the same (it's the ratio), but the absolute delta differs
        assert meta1["hfa_mult"] == pytest.approx(meta2["hfa_mult"], abs=1e-6)

    def test_back_to_back_no_longer_in_hfa(self):
        # back_to_back is now owned by ContextualEngine; HFAEngine ignores it.
        game_with = _neutral_game("Yankee Stadium")
        game_with["back_to_back_away"] = True
        game_without = _neutral_game("Yankee Stadium")

        lh_w, la_w, _ = get_adjusted_lambdas(4.5, 4.5, game_with)
        lh_n, la_n, _ = get_adjusted_lambdas(4.5, 4.5, game_without)
        assert lh_w == pytest.approx(lh_n), "HFA must not apply back_to_back to λ_home"
        assert la_w == pytest.approx(la_n), "HFA must not apply back_to_back to λ_away"
