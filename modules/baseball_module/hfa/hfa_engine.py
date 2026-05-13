"""
HFA ENGINE — Home Field Advantage for MLB lambda pipeline
==========================================================

Scope (per-game adjustments in run units):
  1. Home field crowd/familiarity boost       (asymmetric: home only)
  2. Park run-environment factor              (symmetric: both teams)
  3. Away-team travel fatigue                 (asymmetric: away only)

Explicitly out of scope (handled by AutoCalibrator):
  Team offensive/defensive quality — adding them here double-counts.

Data sources:
  Park factors  — FanGraphs 2024 five-year weighted park factors (÷100 scale)
  Altitudes     — geographic elevation of each stadium in metres
  HFA base      — empirically recalibrated to match +0.034 run/game
                  observed home advantage across 5,422 game sample
                  (original crowd-boost values ÷ 4.0)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from config import LEAGUE_AVG_RUNS

logger = logging.getLogger(__name__)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class StadiumFactors:
    """
    Per-park factors stored for reference and downstream consumers.

    runs_factor : FanGraphs 5-year runs park factor (1.00 = league average).
                  Applied symmetrically to both teams' lambdas.
    hr_factor   : FanGraphs 5-year HR park factor (1.00 = league average).
                  Stored for reference; not consumed by this engine directly.
    hits_factor : Approximated from runs_factor; stored for reference only.
    altitude    : Stadium elevation in metres above sea level.
                  Not used in lambda calculations (already encoded in
                  runs_factor empirically); stored as metadata.
    """
    runs_factor: float = 1.0
    hr_factor:   float = 1.0
    hits_factor: float = 1.0
    altitude:    float = 0


@dataclass
class PitcherStats:
    name:      str
    era:       float
    fip:       Optional[float] = None
    whip:      Optional[float] = None
    k_per_9:   Optional[float] = None
    bb_per_9:  Optional[float] = None
    home_era:  Optional[float] = None
    away_era:  Optional[float] = None


@dataclass
class WeatherConditions:
    temp_f:        float = 75
    wind_mph:      float = 0
    humidity_pct:  float = 50
    precipitation: bool  = False
    roof_closed:   bool  = False


@dataclass
class TeamContext:
    team_name:    str
    home_record:  Optional[Tuple[int, int]] = None
    away_record:  Optional[Tuple[int, int]] = None
    rest_days:    int = 1


# ── Stadium database ──────────────────────────────────────────────────────────
#
# runs_factor / hr_factor: FanGraphs 2024 five-year weighted park factors,
#   100-scale divided by 100. Source: fangraphs.com/guts.aspx?type=pf
#
# hits_factor: set equal to runs_factor (FanGraphs does not publish a
#   separate hits park factor; hits_factor is not used by this engine).
#
# altitude (metres): geographic elevation of each stadium.
#   Key corrections vs prior version:
#     Great American Ball Park 1609→149  (was copy-pasted from Denver)
#     Chase Field              1669→335  (Phoenix = 1,100 ft, not Denver)
#     Nationals Park            305→2    (Washington DC waterfront)
#     Truist Park                48→301  (Cumberland GA ≈ 988 ft)
#     Rogers Centre             182→76   (Toronto ≈ 249 ft)
#     RingCentral Coliseum      194→4    (Oakland bay shore)
#     Angel Stadium             231→47   (Anaheim ≈ 154 ft)
#     loanDepot park             48→4    (Miami near sea level)
#     Coors Field              1660→1589 (5,211 ft confirmed)
#     Camden Yards               97→30   (Baltimore ≈ 98 ft)
#     Target Field              161→253  (Minneapolis ≈ 830 ft)
#     Kauffman Stadium          229→274  (Kansas City ≈ 899 ft)
#     Petco Park                144→11   (San Diego near sea level)
#     Dodger Stadium              0→155  (Elysian Park hills ≈ 509 ft)
#     Busch Stadium              14→142  (St. Louis ≈ 466 ft)

STADIUM_DATABASE = {
    # ── American League East ──────────────────────────────────────────────────
    "Fenway Park":
        StadiumFactors(runs_factor=1.04, hr_factor=0.98, hits_factor=1.04, altitude=9),
    "Yankee Stadium":
        StadiumFactors(runs_factor=0.99, hr_factor=1.04, hits_factor=0.99, altitude=11),
    "Camden Yards":
        StadiumFactors(runs_factor=0.99, hr_factor=0.99, hits_factor=0.99, altitude=30),
    "Tropicana Field":
        StadiumFactors(runs_factor=0.96, hr_factor=0.96, hits_factor=0.96, altitude=3),
    "Rogers Centre":
        StadiumFactors(runs_factor=0.99, hr_factor=1.03, hits_factor=0.99, altitude=76),
    # ── American League Central ───────────────────────────────────────────────
    "Guaranteed Rate Field":
        StadiumFactors(runs_factor=1.00, hr_factor=1.05, hits_factor=1.00, altitude=181),
    "Progressive Field":
        StadiumFactors(runs_factor=0.99, hr_factor=0.98, hits_factor=0.99, altitude=199),
    "Comerica Park":
        StadiumFactors(runs_factor=1.00, hr_factor=0.96, hits_factor=1.00, altitude=183),
    "Kauffman Stadium":
        StadiumFactors(runs_factor=1.03, hr_factor=0.95, hits_factor=1.03, altitude=274),
    "Target Field":
        StadiumFactors(runs_factor=1.01, hr_factor=0.99, hits_factor=1.01, altitude=253),
    # ── American League West ──────────────────────────────────────────────────
    "Minute Maid Park":
        StadiumFactors(runs_factor=0.99, hr_factor=1.02, hits_factor=0.99, altitude=13),
    "Globe Life Field":
        StadiumFactors(runs_factor=0.99, hr_factor=1.02, hits_factor=0.99, altitude=170),
    "Angel Stadium":
        StadiumFactors(runs_factor=1.01, hr_factor=1.05, hits_factor=1.01, altitude=47),
    "T-Mobile Park":
        StadiumFactors(runs_factor=0.94, hr_factor=0.96, hits_factor=0.94, altitude=15),
    "RingCentral Coliseum":
        StadiumFactors(runs_factor=0.96, hr_factor=0.90, hits_factor=0.96, altitude=4),
    # ── National League East ──────────────────────────────────────────────────
    "Citizens Bank Park":
        StadiumFactors(runs_factor=1.01, hr_factor=1.05, hits_factor=1.01, altitude=12),
    "Citi Field":
        StadiumFactors(runs_factor=0.96, hr_factor=0.99, hits_factor=0.96, altitude=11),
    "Nationals Park":
        StadiumFactors(runs_factor=1.00, hr_factor=1.00, hits_factor=1.00, altitude=2),
    "Truist Park":
        StadiumFactors(runs_factor=1.00, hr_factor=0.99, hits_factor=1.00, altitude=301),
    "loanDepot park":
        StadiumFactors(runs_factor=1.01, hr_factor=0.97, hits_factor=1.01, altitude=4),
    # ── National League Central ───────────────────────────────────────────────
    "Wrigley Field":
        StadiumFactors(runs_factor=0.98, hr_factor=0.98, hits_factor=0.98, altitude=181),
    "Great American Ball Park":
        StadiumFactors(runs_factor=1.05, hr_factor=1.14, hits_factor=1.05, altitude=149),
    "American Family Field":
        StadiumFactors(runs_factor=0.99, hr_factor=1.04, hits_factor=0.99, altitude=205),
    "PNC Park":
        StadiumFactors(runs_factor=1.02, hr_factor=0.93, hits_factor=1.02, altitude=229),
    "Busch Stadium":
        StadiumFactors(runs_factor=0.98, hr_factor=0.94, hits_factor=0.98, altitude=142),
    # ── National League West ──────────────────────────────────────────────────
    "Dodger Stadium":
        StadiumFactors(runs_factor=0.99, hr_factor=1.10, hits_factor=0.99, altitude=155),
    "Chase Field":
        StadiumFactors(runs_factor=1.01, hr_factor=0.91, hits_factor=1.01, altitude=335),
    "Oracle Park":
        StadiumFactors(runs_factor=0.97, hr_factor=0.91, hits_factor=0.97, altitude=3),
    "Petco Park":
        StadiumFactors(runs_factor=0.96, hr_factor=1.01, hits_factor=0.96, altitude=11),
    "Coors Field":
        StadiumFactors(runs_factor=1.13, hr_factor=1.07, hits_factor=1.13, altitude=1589),
}


# ── HFA Engine ────────────────────────────────────────────────────────────────

class HFAEngine:
    """
    Adjusts (λ_home, λ_away) for park environment, crowd advantage,
    and away-team travel fatigue.
    """

    def __init__(self):
        self.name = "HFA Engine"

        # ── Crowd / familiarity boost per park (in runs, asymmetric) ──────────
        # Empirically recalibrated: original crowd-boost estimates ÷ 4.0
        # to match the +0.034 run/game home scoring advantage observed
        # across 5,422 games (2024-2026). Original values (0.10–0.18) produced
        # a +0.134 model spread — 3.9× the empirical truth.
        self.hfa_base = {
            "Yankee Stadium":            0.0450,  # loud, iconic, short RF porch
            "Fenway Park":               0.0425,  # Green Monster, hostile atmosphere
            "Dodger Stadium":            0.0400,
            "Wrigley Field":             0.0400,
            "Busch Stadium":             0.0375,
            "Oracle Park":               0.0375,
            "Citizens Bank Park":        0.0375,
            "Great American Ball Park":  0.0325,
            "Progressive Field":         0.0350,
            "Coors Field":               0.0350,  # thin air affects away team more
            "Petco Park":                0.0350,
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

        # ── Symmetric park run-environment factors ─────────────────────────────
        # Populated from STADIUM_DATABASE.runs_factor (FanGraphs 5yr, 1.00 scale).
        self.park_factors_hitters = {
            name: sf.runs_factor for name, sf in STADIUM_DATABASE.items()
        }

    # ── Public interface ───────────────────────────────────────────────────────

    def get_adjusted_lambdas(
        self,
        lh: float,
        la: float,
        game_data: Dict[str, Any],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Apply park environment, crowd advantage, and travel fatigue to lambdas.

        Args:
            lh: Home expected runs after AutoCalibrator.
            la: Away expected runs after AutoCalibrator.
            game_data: Dict with keys 'park' (→ 'name'), optionally
                       'miles_traveled_away', 'time_zones_crossed_away',
                       'back_to_back_away'.

        Returns:
            (lh_adjusted, la_adjusted, metadata_dict)
        """
        park_name = game_data.get("park", {}).get("name", "Unknown")

        metadata = {
            "hfa_base":    0.0,
            "park_factor": 1.0,
            "travel_away": 0.0,
            "altitude":    0.0,   # kept for API compat; always 0 (encoded in park factor)
        }

        logger.debug("HFA Engine | park=%s  λh=%.3f  λa=%.3f", park_name, lh, la)

        # ── Step 1: Home crowd / familiarity boost (asymmetric) ───────────────
        # Converts run-unit boost to a proportional multiplier so that the
        # absolute boost scales with the home team's offensive context.
        # At λ = LEAGUE_AVG_RUNS the absolute boost equals hfa_boost exactly.
        hfa_boost = self.hfa_base.get(park_name, 0.0325)
        hfa_mult  = 1.0 + hfa_boost / LEAGUE_AVG_RUNS
        lh_new    = lh * hfa_mult
        metadata["hfa_base"] = hfa_boost

        # ── Step 2: Park run-environment factor (symmetric) ───────────────────
        # FanGraphs empirical factor already encodes altitude, dimensions,
        # wind patterns, and weather — applied equally to both teams.
        park_mult = self.park_factors_hitters.get(park_name, 1.00)
        lh_new   *= park_mult
        la_new    = la * park_mult
        metadata["park_factor"] = park_mult

        # NOTE: No separate altitude adjustment. The park runs_factor is measured
        # from actual game outcomes and already captures every altitude effect.
        # Adding a physics-based altitude term on top double-counts those effects
        # and was the primary source of Coors/Chase lambda over-inflation.

        # ── Step 3: Away-team travel fatigue (asymmetric) ─────────────────────
        travel_penalty = self._calculate_travel_fatigue(game_data)
        if travel_penalty > 0.0:
            la_new *= 1.0 - travel_penalty / LEAGUE_AVG_RUNS
            metadata["travel_away"] = travel_penalty

        logger.debug(
            "HFA Engine done | λh: %.3f→%.3f  λa: %.3f→%.3f",
            lh, lh_new, la, la_new,
        )

        return lh_new, la_new, metadata

    # ── Private helpers ────────────────────────────────────────────────────────

    def _calculate_travel_fatigue(self, game_data: Dict) -> float:
        """
        Away-team penalty for cross-timezone travel and back-to-back games.

        Design notes:
          • Time-zone crossings are the primary causal mechanism (circadian
            disruption), so they take precedence over distance.
          • Miles used only as a fallback when time_zones data is unavailable
            (== 0 from API) but miles are non-zero — avoids double-counting.
          • Max penalty reduced from 0.25 to 0.10 runs: prior value (5.6% λ
            cut) was far above empirical estimates (~0.5–1% win-prob effect).
          • In practice these fields are rarely populated by the free MLB API;
            the penalty fires mainly when game_data is enriched externally.
        """
        miles       = game_data.get("miles_traveled_away", 0)
        time_zones  = game_data.get("time_zones_crossed_away", 0)
        back_to_back = game_data.get("back_to_back_away", False)

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

        if back_to_back:
            penalty += 0.03

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
