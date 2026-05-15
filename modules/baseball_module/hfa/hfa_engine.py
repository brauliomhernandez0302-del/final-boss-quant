"""
HFA ENGINE — Asymmetric home field advantage for MLB lambda pipeline
====================================================================

Scope — asymmetric adjustments ONLY:
  1. Home crowd / familiarity boost   (adds to λ_home only)
  2. Away-team travel fatigue         (subtracts from λ_away only)

NOT in scope (handled by ParkWeatherEngine, PASO 2):
  Park run-environment factor — symmetric, belongs in its own engine.
  Weather (temp, wind, rain)  — symmetric, belongs in ParkWeatherEngine.

HFA base — empirically recalibrated to match +0.034 run/game home
advantage across 5,422 games (2024-2026). Original crowd values ÷ 4.0.

Formula: hfa_mult = 1 + hfa_boost / LEAGUE_AVG_RUNS
Applied to λ_home only.  At league-average λ=4.5 the multiplier adds
exactly hfa_boost runs; at other λ the additive effect scales slightly
(≈±0.004 runs across the MLB λ range — within calibration tolerance).
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

        # ── Crowd / familiarity boost per park (in runs, asymmetric) ──────────
        # Empirically recalibrated: original crowd-boost estimates ÷ 4.0
        # to match the +0.034 run/game home scoring advantage observed
        # across 5,422 games (2024-2026). Original values (0.10–0.18) produced
        # a +0.134 model spread — 3.9× the empirical truth.
        #
        # These values capture CROWD NOISE and HOME FAMILIARITY only.
        # Thin air (Coors), dimensions (Fenway wall), turf — all symmetric
        # and already encoded in ParkWeatherEngine's runs_factor per stadium.
        self.hfa_base = {
            "Yankee Stadium":            0.0450,  # loud, iconic; short RF porch is park factor
            "Fenway Park":               0.0425,  # hostile atmosphere; Green Monster is park factor
            "Dodger Stadium":            0.0400,
            "Wrigley Field":             0.0400,
            "Busch Stadium":             0.0375,
            "Oracle Park":               0.0375,
            "Citizens Bank Park":        0.0375,
            "Progressive Field":         0.0350,
            "Coors Field":               0.0350,  # crowd familiarity; altitude is in park factor
            "Petco Park":                0.0350,
            "Great American Ball Park":  0.0325,
            "Comerica Park":             0.0325,
            "Target Field":              0.0325,
            "PNC Park":                  0.0325,
            "T-Mobile Park":             0.0325,
            "Truist Park":               0.0325,
            "Camden Yards":              0.0325,
            "Angel Stadium":             0.0325,
            "American Family Field":     0.0325,
            "Minute Maid Park":          0.0300,  # roof dampens crowd noise
            "Globe Life Field":          0.0300,
            "Kauffman Stadium":          0.0300,
            "Chase Field":               0.0300,
            "Guaranteed Rate Field":     0.0300,
            "Rogers Centre":             0.0275,
            "RingCentral Coliseum":      0.0275,
            "Citi Field":                0.0275,  # historically low attendance
            "Nationals Park":            0.0275,
            "Tropicana Field":           0.0250,  # worst attendance in MLB
            "loanDepot park":            0.0250,
        }

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
        park_name = game_data.get("park", {}).get("name", "Unknown")

        logger.debug("HFA Engine | park=%s  λh=%.3f  λa=%.3f", park_name, lh, la)

        # ── Step 1: Home crowd / familiarity boost (asymmetric, λ_home only) ──
        hfa_boost = self.hfa_base.get(park_name, 0.0325)
        hfa_mult  = 1.0 + hfa_boost / LEAGUE_AVG_RUNS
        lh_new    = lh * hfa_mult
        la_new    = la   # away unchanged by crowd boost

        # ── Step 2: Away-team travel fatigue (asymmetric, λ_away only) ────────
        travel_penalty = self._calculate_travel_fatigue(game_data)
        if travel_penalty > 0.0:
            la_new *= 1.0 - travel_penalty / LEAGUE_AVG_RUNS

        metadata = {
            "park_name":       park_name,
            "hfa_boost_runs":  round(hfa_boost, 4),
            "hfa_mult":        round(hfa_mult, 4),
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
        miles      = game_data.get("miles_traveled_away",     0)
        time_zones = game_data.get("time_zones_crossed_away", 0)

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
