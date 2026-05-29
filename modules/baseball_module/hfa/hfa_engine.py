"""
HFA ENGINE — Asymmetric home field advantage for MLB lambda pipeline
====================================================================

Scope — asymmetric adjustments ONLY:
  Away-team travel fatigue (subtracts from λ_away only).

Removed (E3 / FIX C2):
  Home crowd/familiarity boost: confirmed pure noise (Pearson=-0.015,
  direction 53% random). hfa_boost hardcoded to 0.0. The hfa_base
  per-stadium dict has been removed entirely.

NOT in scope (handled by ParkWeatherEngine, PASO 2):
  Park run-environment factor — symmetric, belongs in its own engine.
  Weather (temp, wind, rain)  — symmetric, belongs in ParkWeatherEngine.
"""

import logging
from typing import Dict, Any, Tuple
from config import LEAGUE_AVG_RUNS

logger = logging.getLogger(__name__)


# ── HFA Engine ────────────────────────────────────────────────────────────────

class HFAEngine:
    """
    Adjusts (λ_home, λ_away) for crowd advantage and away-team travel fatigue.
    Park factor and weather are handled upstream by ParkWeatherEngine (PASO 2).
    """

    def __init__(self):
        self.name = "HFA Engine"

    # ── Public interface ───────────────────────────────────────────────────────

    def get_adjusted_lambdas(
        self,
        lh: float,
        la: float,
        game_data: Dict[str, Any],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Apply crowd advantage and travel fatigue (asymmetric adjustments only).
        Park factor and weather are handled by ParkWeatherEngine (PASO 2).

        Args:
            lh: Home expected runs after ParkWeatherEngine.
            la: Away expected runs after ParkWeatherEngine.
            game_data: Dict with keys 'park' (→ 'name'), optionally
                       'miles_traveled_away', 'time_zones_crossed_away'.

        Returns:
            (lh_adjusted, la_adjusted, metadata_dict)
        """
        park_name = (game_data.get("park") or {}).get("name", "Unknown")

        logger.debug("HFA Engine | park=%s  λh=%.3f  λa=%.3f  [crowd_boost=disabled]", park_name, lh, la)

        # Crowd boost eliminated (FIX C2 / E3): Pearson=-0.015, direction 53% random.
        # Kalman + multidim_bias cover ~95% of empirical HFA signal.
        hfa_boost = 0.0
        hfa_mult  = 1.0
        lh_new    = lh

        # ── Step 2: Away-team travel fatigue (asymmetric, λ_away only) ────────
        # Travel penalty retained: 69.3% activation rate, max 1.3% λ reduction.
        # Mechanistic basis (circadian disruption) empirically confirmed.
        la_new = la
        travel_penalty = self._calculate_travel_fatigue(game_data)
        if travel_penalty > 0.0:
            la_new *= 1.0 - travel_penalty / LEAGUE_AVG_RUNS

        metadata = {
            "park_name":       park_name,
            "hfa_boost_runs":  0.0,
            "hfa_mult":        1.0,
            "travel_penalty":  round(travel_penalty, 4),
        }

        logger.debug(
            "HFA Engine done | λh: %.3f→%.3f  λa: %.3f→%.3f",
            lh, lh_new, la, la_new,
        )

        return lh_new, la_new, metadata

    # ── Private helpers ────────────────────────────────────────────────────────

    def _calculate_travel_fatigue(self, game_data: Dict) -> float:
        """
        Away-team penalty for cross-timezone travel and long-distance trips.

        Design notes:
          • Time-zone crossings are the primary causal mechanism (circadian
            disruption), so they take precedence over distance.
          • Miles used only as a fallback when time_zones data is unavailable
            (== 0 from API) but miles are non-zero — avoids double-counting.
          • Max penalty 0.10 runs (~2.2% λ cut): empirical estimates put the
            travel effect at ~0.5–1% win-prob, equivalent to ~0.05–0.10 runs.
          • Back-to-back is intentionally excluded: owned by ContextualEngine
            (PASO 7), which applies the B2B penalty to BOTH teams symmetrically
            in their own rest_days signal. Keeping it here would double-count
            the away team's B2B.
          • In practice travel fields are rarely populated by the free MLB API;
            the penalty fires mainly when game_data is enriched externally.
        """
        _mi = game_data.get("miles_traveled_away")
        miles      = float(_mi if _mi is not None else 0)
        _tz = game_data.get("time_zones_crossed_away")
        time_zones = int(_tz   if _tz is not None else 0)

        penalty = 0.0

        if time_zones >= 3:
            penalty = 0.06   # coast-to-coast: ~1.3% λ reduction
        elif time_zones == 2:
            penalty = 0.04
        elif time_zones == 1:
            penalty = 0.02
        elif miles > 2000:
            # Fallback when time_zones not populated but trip is long
            penalty = 0.05
        elif miles > 1000:
            penalty = 0.03

        return min(penalty, 0.10)


# ── Module-level helper (public interface used by run_module.py) ───────────────

def get_adjusted_lambdas(
    lh: float,
    la: float,
    game_data: Dict[str, Any],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Convenience wrapper — matches the import used in run_module.py.

    Usage:
        from hfa.hfa_engine import get_adjusted_lambdas
        lh_hfa, la_hfa, meta = get_adjusted_lambdas(lh, la, game_data)
    """
    return HFAEngine().get_adjusted_lambdas(lh, la, game_data)
