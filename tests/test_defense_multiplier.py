"""
Tests for defense-related adjustments after the architecture split.

Post-refactor ownership:
  • Starting pitcher (ERA/FIP) → PitcherEngine              (PASO 2)
  • Bullpen                    → BullpenEngine              (PASO 4)
  • Fielding (DER/OAA)         → DefensiveEfficiencyEngine  (PASO 6)
  • AutoCalibrator             → recent form + season context only
                                 (defense_mult removed to eliminate triple-counting)

Critical invariant preserved across all engines:
  Strong defense → reduces OPPONENT's λ
  Weak defense   → increases OPPONENT's λ
"""
import pytest
from config import LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP, LEAGUE_AVG_RUNS
from modules.baseball_module.context_engine.defensive_efficiency_engine import _LG_DER


# ── Defensive Efficiency Engine ────────────────────────────────────────────────

class TestDefensiveEfficiencyEngine:
    """
    DefensiveEfficiencyEngine: DER → multiplier on opponent's λ.
    Convention: home defence → reduces λ_away; away defence → reduces λ_home.
    """

    def _engine(self):
        from modules.baseball_module.context_engine.defensive_efficiency_engine import (
            DefensiveEfficiencyEngine,
        )
        return DefensiveEfficiencyEngine()

    def test_elite_fielding_reduces_opponent_lambda(self):
        engine = self._engine()
        # DER=0.740 (elite) → home defence should reduce λ_away
        game_data = {
            "defense_home": {"der": 0.740, "bip": 2000, "oaa": None},
            "defense_away": {},
        }
        lh, la, meta = engine.adjust_for_defense(4.5, 4.5, game_data)
        assert la < 4.5, f"Elite home fielding must reduce λ_away, got {la:.4f}"

    def test_weak_fielding_raises_opponent_lambda(self):
        engine = self._engine()
        # DER=0.685 (weak) → home defence should raise λ_away
        game_data = {
            "defense_home": {"der": 0.685, "bip": 2000, "oaa": None},
            "defense_away": {},
        }
        lh, la, meta = engine.adjust_for_defense(4.5, 4.5, game_data)
        assert la > 4.5, f"Weak home fielding must raise λ_away, got {la:.4f}"

    def test_league_average_der_neutral(self):
        engine = self._engine()
        game_data = {
            "defense_home": {"der": _LG_DER, "bip": 2000, "oaa": None},
            "defense_away": {"der": _LG_DER, "bip": 2000, "oaa": None},
        }
        lh, la, _ = engine.adjust_for_defense(4.5, 4.5, game_data)
        assert lh == pytest.approx(4.5, abs=0.01)
        assert la == pytest.approx(4.5, abs=0.01)

    def test_home_defence_affects_only_away_lambda(self):
        engine = self._engine()
        game_data = {
            "defense_home": {"der": 0.740, "bip": 2000, "oaa": None},
            "defense_away": {},   # no away defence data
        }
        lh, la, meta = engine.adjust_for_defense(4.5, 4.5, game_data)
        # λ_home should not be affected by home defence (away fielders affect λ_home)
        assert la < 4.5, "Home elite fielding must reduce λ_away"

    def test_small_sample_bayesian_regression(self):
        engine = self._engine()
        # With BIP=50 (tiny sample), DER=0.760 → heavily regressed toward 0.715
        game_data = {
            "defense_home": {"der": 0.760, "bip": 50, "oaa": None},
            "defense_away": {},
        }
        _, la_small, _ = engine.adjust_for_defense(4.5, 4.5, game_data)

        game_data2 = {
            "defense_home": {"der": 0.760, "bip": 2000, "oaa": None},
            "defense_away": {},
        }
        _, la_large, _ = engine.adjust_for_defense(4.5, 4.5, game_data2)

        # Larger sample → more aggressive adjustment
        assert la_small > la_large, "Large BIP sample should produce stronger adjustment"

    def test_max_adjustment_capped(self):
        engine = self._engine()
        # Impossibly elite fielding — should be capped at ±8% (B2: expanded from ±5%)
        game_data = {
            "defense_home": {"der": 0.999, "bip": 9999, "oaa": 100},
            "defense_away": {"der": 0.001, "bip": 9999, "oaa": -100},
        }
        lh, la, _ = engine.adjust_for_defense(4.5, 4.5, game_data)
        assert la >= 4.5 * 0.92, "Floor cap: -8% max (B2 expanded from -5%)"
        assert lh <= 4.5 * 1.08, "Ceiling cap: +8% max (B2 expanded from +5%)"

    def test_no_data_returns_unchanged(self):
        from modules.baseball_module.context_engine.defensive_efficiency_engine import (
            adjust_for_defense,
        )
        lh, la, meta = adjust_for_defense(4.5, 4.3, {})
        assert lh == 4.5
        assert la == 4.3
        assert meta.get("skipped"), "Should return a truthy skipped value when no defense data"

    def test_oaa_increases_effect_on_elite_team(self):
        engine = self._engine()
        # Neutral DER (0.715 = league avg → der_factor = 1.0, no DER adjustment).
        # Positive OAA = 20 → oaa_factor < 1.0 → combined mult < 1.0 (OAA adds signal).
        # Without OAA, neutral DER means no adjustment at all (mult = 1.0).
        base = {"der": 0.715, "bip": 2000}
        game_no_oaa  = {"defense_home": {**base, "oaa": None},  "defense_away": {}}
        game_pos_oaa = {"defense_home": {**base, "oaa": 20.0},  "defense_away": {}}

        _, la_no,  _ = engine.adjust_for_defense(4.5, 4.5, game_no_oaa)
        _, la_pos, _ = engine.adjust_for_defense(4.5, 4.5, game_pos_oaa)
        assert la_pos < la_no, "Positive OAA with neutral DER should reduce λ_away"


