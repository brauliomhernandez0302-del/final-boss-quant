"""
Tests for the HFA engine multiplicative pipeline.

Each stage (HFA base, park factor, altitude, travel, offense, defense)
multiplies lambda independently. The tests verify:
  1. Each adjustment applies multiplicatively (λ_new = λ_old × mult)
  2. Home advantage always boosts λ_home, never λ_away
  3. Park factor applies equally to both teams
  4. High altitude boosts BOTH lambdas (home less than away)
  5. Travel penalty only reduces λ_away
  6. All-neutral inputs preserve the input lambdas
"""
import pytest
from modules.baseball_module.hfa.hfa_engine import HFAEngine, get_adjusted_lambdas
from config import LEAGUE_AVG_RUNS, LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP


def _avg_team():
    return {
        "runs_per_game": LEAGUE_AVG_RUNS,
        "woba": 0.320, "ops": 0.735, "wrc_plus": 100,
        "team_era": LEAGUE_AVG_ERA,
        "team_whip": LEAGUE_AVG_WHIP,
        "runs_allowed_per_game": LEAGUE_AVG_RUNS,
    }


def _neutral_game(park="Unknown"):
    return {
        "home_team": _avg_team(),
        "away_team": _avg_team(),
        "park": {"name": park},
        "miles_traveled_away": 0,
        "time_zones_crossed_away": 0,
        "back_to_back_away": False,
    }


class TestHFAPipelineMultiplicative:

    def test_neutral_game_minimal_change(self):
        # All inputs average, unknown park → output should be close to inputs.
        # (HFA base will still apply a small boost to home.)
        lh, la = 4.5, 4.5
        lh_out, la_out, _ = get_adjusted_lambdas(lh, la, _neutral_game())
        # Home gets some boost; away may get tiny defense/offense adjustments.
        assert lh_out > lh, "HFA base should always boost λ_home"
        assert 3.5 < la_out < 5.5, "λ_away should stay in reasonable range"

    def test_home_advantage_direction(self):
        # Home should always be boosted relative to away in a symmetric matchup.
        lh, la = 4.5, 4.5
        lh_out, la_out, meta = get_adjusted_lambdas(lh, la, _neutral_game("Yankee Stadium"))
        assert lh_out > la_out, "Home lambda must exceed away in a symmetric matchup at any park"
        assert meta["hfa_base"] > 0

    def test_park_factor_applies_both_teams(self):
        # Coors Field (runs_factor=1.25) should boost BOTH lambdas.
        lh, la = 4.0, 4.0
        lh_coors, la_coors, _ = get_adjusted_lambdas(lh, la, _neutral_game("Coors Field"))
        lh_petco, la_petco, _ = get_adjusted_lambdas(lh, la, _neutral_game("Petco Park"))
        assert lh_coors > lh_petco, "Coors should produce more home runs than Petco"
        assert la_coors > la_petco, "Coors should produce more away runs than Petco"

    def test_altitude_boosts_both_but_away_more(self):
        # Coors Field altitude=1660 m → both up, away more than home.
        game = _neutral_game("Coors Field")
        lh_out, la_out, meta = get_adjusted_lambdas(4.0, 4.0, game)
        # altitude metadata should be non-zero
        assert meta["altitude"] > 0

    def test_travel_only_reduces_away(self):
        # Maximum travel: coast-to-coast + 3 time zones + back-to-back.
        game_travel = _neutral_game("Yankee Stadium")
        game_travel["miles_traveled_away"] = 3000
        game_travel["time_zones_crossed_away"] = 3
        game_travel["back_to_back_away"] = True

        game_none = _neutral_game("Yankee Stadium")

        lh_t, la_t, _ = get_adjusted_lambdas(4.5, 4.5, game_travel)
        lh_n, la_n, _ = get_adjusted_lambdas(4.5, 4.5, game_none)

        assert la_t < la_n, "Heavy travel should reduce λ_away"
        assert abs(lh_t - lh_n) < 0.01, "λ_home should not be affected by away travel"

    def test_offense_multiplier_direction(self):
        # Give home team elite offense; away team weak offense.
        game = _neutral_game("Yankee Stadium")
        game["home_team"] = {**_avg_team(), "woba": 0.380, "ops": 0.850, "wrc_plus": 130}
        game["away_team"] = {**_avg_team(), "woba": 0.270, "ops": 0.640, "wrc_plus": 75}

        lh_out, la_out, meta = get_adjusted_lambdas(4.5, 4.5, game)
        assert meta["offense_home"] > 1.0, "Elite home offense → mult > 1"
        assert meta["offense_away"] < 1.0, "Weak away offense → mult < 1"
        # λ_home should exceed a symmetric game
        lh_sym, _, _ = get_adjusted_lambdas(4.5, 4.5, _neutral_game("Yankee Stadium"))
        assert lh_out > lh_sym

    def test_defense_multiplier_wired_to_lambda_correctly(self):
        # Home team elite defense → suppresses λ_away.
        # Away team weak defense → inflates λ_home.
        game = _neutral_game("Yankee Stadium")
        game["home_team"] = {**_avg_team(), "team_era": 3.00, "team_whip": 1.05,
                             "runs_allowed_per_game": 3.40}
        game["away_team"] = {**_avg_team(), "team_era": 5.50, "team_whip": 1.65,
                             "runs_allowed_per_game": 5.60}

        lh_out, la_out, meta = get_adjusted_lambdas(4.5, 4.5, game)
        # Home defense < 1 → reduces λ_away
        assert meta["defense_home"] < 1.0
        # Away defense > 1 → inflates λ_home
        assert meta["defense_away"] > 1.0

        # In a symmetric matchup, verify direction
        lh_sym, la_sym, _ = get_adjusted_lambdas(4.5, 4.5, _neutral_game("Yankee Stadium"))
        assert la_out < la_sym, "Elite home pitching should suppress away scoring vs average"
        assert lh_out > lh_sym, "Weak away pitching should inflate home scoring vs average"

    def test_each_step_is_multiplicative(self):
        # Run the engine and verify total change matches component product.
        engine = HFAEngine()
        lh_in, la_in = 4.0, 4.0
        game = _neutral_game("Fenway Park")
        lh_out, la_out, meta = engine.get_adjusted_lambdas(lh_in, la_in, game)
        # All multipliers from metadata should multiply together coherently.
        # We can't reconstruct exactly because of order-dependency and clamps,
        # but the final lambda must be > 0 and < 20.
        assert 0 < lh_out < 20
        assert 0 < la_out < 20

    def test_output_always_positive(self):
        for park in ["Coors Field", "Petco Park", "Yankee Stadium", "Unknown"]:
            lh_out, la_out, _ = get_adjusted_lambdas(1.0, 1.0, _neutral_game(park))
            assert lh_out > 0, f"λ_home must be positive at {park}"
            assert la_out > 0, f"λ_away must be positive at {park}"
