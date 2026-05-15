"""
PARK + WEATHER ENGINE — Symmetric run-environment adjustment
============================================================

Answers one question per game:
    "How much does this park + today's conditions inflate or deflate runs?"

This effect is SYMMETRIC — it shifts λ_home and λ_away by the same
multiplier.  It belongs in a separate engine from HFA because HFA is
asymmetric (crowd favors home, travel penalises away).  Mixing the two
makes it impossible to isolate, debug, or weight them independently.

Factors:
  1. Park factor   — FanGraphs 5-year empirical runs factor (1.00 = avg).
                     Already encodes altitude, dimensions, historical wind,
                     and every other structural characteristic of the park.

  2. Temperature   — Each 5°F above 72°F → +0.5% more runs (ball carries
                     further in warm air, pitchers grip worse).
                     FanGraphs / Codify research: ~0.4–0.6% per 5°F.

  3. Wind          — Direction relative to field orientation matters.
                     Wind blowing OUT toward outfield → more HRs/runs.
                     Wind blowing IN toward home plate → fewer HRs/runs.
                     Crosswind → minimal effect.
                     Effect scales with speed above 5 mph baseline.

  4. Rain / precip — Wet ball, soft turf, erratic grip → pitchers gain a
                     slight advantage.  Heavy rain = postponement (not here).

  5. Domed / roof  — Retractable-roof and fixed-dome parks eliminate all
                     weather effects when the roof is closed.

Data source for weather: game_data['weather'] from OpenWeather API
    (optional — engine falls back to park factor only when absent).

Pipeline position: PASO 2 — after AutoCalibrator, before HFA Engine.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Neutral temperature for run-environment baseline (°F)
_NEUTRAL_TEMP_F = 72.0
# Run-increase per 5°F above neutral
_TEMP_RATE = 0.005


@dataclass
class StadiumFactors:
    """
    Structural characteristics of each MLB park.

    runs_factor  : FanGraphs 5-year weighted runs park factor (1.00 = avg).
    hr_factor    : FanGraphs 5-year HR park factor.
    hits_factor  : Approximated from runs_factor (reference only).
    altitude     : Metres above sea level (encoded in runs_factor; stored
                   for metadata only — no separate altitude calculation).
    cf_direction : Compass bearing FROM home plate TO centre field (0–359°).
                   Wind FROM the OPPOSITE direction (cf_direction+180°)
                   blows OUT toward the outfield.  Used for wind effect.
    has_roof     : True for fixed domes and retractable-roof parks.
                   When True (or game_data indicates roof closed), weather
                   effects are suppressed to zero.
    """
    runs_factor:  float = 1.00
    hr_factor:    float = 1.00
    hits_factor:  float = 1.00
    altitude:     float = 0.0
    cf_direction: int   = 0     # degrees, home plate → CF
    has_roof:     bool  = False


# ── Stadium database ──────────────────────────────────────────────────────────
# runs_factor / hr_factor: FanGraphs 2024 five-year weighted park factors ÷ 100
# cf_direction: compass bearing home plate → CF (approximate, ballpark-level)
#   North = 0°, East = 90°, South = 180°, West = 270°

STADIUM_DATABASE: Dict[str, StadiumFactors] = {
    # ── American League East ──────────────────────────────────────────────────
    "Fenway Park":
        StadiumFactors(runs_factor=1.04, hr_factor=0.98, hits_factor=1.04,
                       altitude=9,   cf_direction=45),   # CF roughly NE
    "Yankee Stadium":
        StadiumFactors(runs_factor=0.99, hr_factor=1.04, hits_factor=0.99,
                       altitude=11,  cf_direction=315),  # CF roughly NW
    "Camden Yards":
        StadiumFactors(runs_factor=0.99, hr_factor=0.99, hits_factor=0.99,
                       altitude=30,  cf_direction=350),
    "Tropicana Field":
        StadiumFactors(runs_factor=0.96, hr_factor=0.96, hits_factor=0.96,
                       altitude=3,   cf_direction=0,   has_roof=True),
    "Rogers Centre":
        StadiumFactors(runs_factor=0.99, hr_factor=1.03, hits_factor=0.99,
                       altitude=76,  cf_direction=350, has_roof=True),
    # ── American League Central ───────────────────────────────────────────────
    "Guaranteed Rate Field":
        StadiumFactors(runs_factor=1.00, hr_factor=1.05, hits_factor=1.00,
                       altitude=181, cf_direction=345),
    "Progressive Field":
        StadiumFactors(runs_factor=0.99, hr_factor=0.98, hits_factor=0.99,
                       altitude=199, cf_direction=5),
    "Comerica Park":
        StadiumFactors(runs_factor=1.00, hr_factor=0.96, hits_factor=1.00,
                       altitude=183, cf_direction=310),
    "Kauffman Stadium":
        StadiumFactors(runs_factor=1.03, hr_factor=0.95, hits_factor=1.03,
                       altitude=274, cf_direction=350),
    "Target Field":
        StadiumFactors(runs_factor=1.01, hr_factor=0.99, hits_factor=1.01,
                       altitude=253, cf_direction=290),
    # ── American League West ──────────────────────────────────────────────────
    "Minute Maid Park":
        StadiumFactors(runs_factor=0.99, hr_factor=1.02, hits_factor=0.99,
                       altitude=13,  cf_direction=340, has_roof=True),
    "Globe Life Field":
        StadiumFactors(runs_factor=0.99, hr_factor=1.02, hits_factor=0.99,
                       altitude=170, cf_direction=10,  has_roof=True),
    "Angel Stadium":
        StadiumFactors(runs_factor=1.01, hr_factor=1.05, hits_factor=1.01,
                       altitude=47,  cf_direction=0),
    "T-Mobile Park":
        StadiumFactors(runs_factor=0.94, hr_factor=0.96, hits_factor=0.94,
                       altitude=15,  cf_direction=355, has_roof=True),
    "RingCentral Coliseum":
        StadiumFactors(runs_factor=0.96, hr_factor=0.90, hits_factor=0.96,
                       altitude=4,   cf_direction=0),
    # ── National League East ──────────────────────────────────────────────────
    "Citizens Bank Park":
        StadiumFactors(runs_factor=1.01, hr_factor=1.05, hits_factor=1.01,
                       altitude=12,  cf_direction=355),
    "Citi Field":
        StadiumFactors(runs_factor=0.96, hr_factor=0.99, hits_factor=0.96,
                       altitude=11,  cf_direction=10),
    "Nationals Park":
        StadiumFactors(runs_factor=1.00, hr_factor=1.00, hits_factor=1.00,
                       altitude=2,   cf_direction=5),
    "Truist Park":
        StadiumFactors(runs_factor=1.00, hr_factor=0.99, hits_factor=1.00,
                       altitude=301, cf_direction=5),
    "loanDepot park":
        StadiumFactors(runs_factor=1.01, hr_factor=0.97, hits_factor=1.01,
                       altitude=4,   cf_direction=0,   has_roof=True),
    # ── National League Central ───────────────────────────────────────────────
    "Wrigley Field":
        StadiumFactors(runs_factor=0.98, hr_factor=0.98, hits_factor=0.98,
                       altitude=181, cf_direction=45),   # famous NE-facing CF
    "Great American Ball Park":
        StadiumFactors(runs_factor=1.05, hr_factor=1.14, hits_factor=1.05,
                       altitude=149, cf_direction=350),
    "American Family Field":
        StadiumFactors(runs_factor=0.99, hr_factor=1.04, hits_factor=0.99,
                       altitude=205, cf_direction=340, has_roof=True),
    "PNC Park":
        StadiumFactors(runs_factor=1.02, hr_factor=0.93, hits_factor=1.02,
                       altitude=229, cf_direction=350),
    "Busch Stadium":
        StadiumFactors(runs_factor=0.98, hr_factor=0.94, hits_factor=0.98,
                       altitude=142, cf_direction=5),
    # ── National League West ──────────────────────────────────────────────────
    "Dodger Stadium":
        StadiumFactors(runs_factor=0.99, hr_factor=1.10, hits_factor=0.99,
                       altitude=155, cf_direction=340),
    "Chase Field":
        StadiumFactors(runs_factor=1.01, hr_factor=0.91, hits_factor=1.01,
                       altitude=335, cf_direction=10,  has_roof=True),
    "Oracle Park":
        StadiumFactors(runs_factor=0.97, hr_factor=0.91, hits_factor=0.97,
                       altitude=3,   cf_direction=0),   # marine west wind usually IN
    "Petco Park":
        StadiumFactors(runs_factor=0.96, hr_factor=1.01, hits_factor=0.96,
                       altitude=11,  cf_direction=350),
    "Coors Field":
        StadiumFactors(runs_factor=1.13, hr_factor=1.07, hits_factor=1.13,
                       altitude=1589, cf_direction=315),
}


class ParkWeatherEngine:
    """
    Applies the symmetric run-environment multiplier:
        total_mult = park_factor × weather_mult
    Applied identically to λ_home and λ_away.
    """

    def adjust_for_park_and_weather(
        self,
        lh: float,
        la: float,
        game_data: Dict[str, Any],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Returns (lh_adjusted, la_adjusted, metadata)."""

        park_name = game_data.get("park", {}).get("name", "Unknown")
        stadium   = STADIUM_DATABASE.get(park_name)
        park_mult = stadium.runs_factor if stadium else 1.00

        # Determine if park is sheltered today
        roof_closed = bool(game_data.get("roof_closed", False))
        if stadium and stadium.has_roof and not game_data.get("roof_open", False):
            roof_closed = True   # default retractable parks to closed

        weather     = game_data.get("weather", {})
        weather_mult, wx_meta = self._weather_mult(weather, stadium, roof_closed)

        total_mult = park_mult * weather_mult
        lh_new     = lh * total_mult
        la_new     = la * total_mult

        meta = {
            "park_name":    park_name,
            "park_factor":  round(park_mult,    4),
            "weather_mult": round(weather_mult, 4),
            "total_mult":   round(total_mult,   4),
            "roof_closed":  roof_closed,
            **wx_meta,
        }

        log.info(
            "Park+Weather | %s  park=%.3f  weather=%.3f  total=%.3f  "
            "λ_h %.3f→%.3f  λ_a %.3f→%.3f",
            park_name, park_mult, weather_mult, total_mult,
            lh, lh_new, la, la_new,
        )
        return lh_new, la_new, meta

    # ── Weather ───────────────────────────────────────────────────────────────

    def _weather_mult(
        self,
        weather:     Dict[str, Any],
        stadium:     Optional[StadiumFactors],
        roof_closed: bool,
    ) -> Tuple[float, Dict]:
        """
        Compute symmetric weather multiplier from temperature, wind, and
        precipitation.  Returns (mult, detail_dict).
        """
        if roof_closed or not weather:
            return 1.0, {"temp_mult": 1.0, "wind_mult": 1.0, "rain_mult": 1.0}

        temp_mult = self._temp_mult(weather.get("temp_f", _NEUTRAL_TEMP_F))
        wind_mult = self._wind_mult(
            weather.get("wind_speed_mph", 0),
            weather.get("wind_direction",  0),
            stadium,
        )
        rain_mult = self._rain_mult(weather.get("conditions", ""))

        combined = float(np.clip(temp_mult * wind_mult * rain_mult, 0.90, 1.12))

        return combined, {
            "temp_f":    weather.get("temp_f"),
            "temp_mult": round(temp_mult, 4),
            "wind_mph":  weather.get("wind_speed_mph"),
            "wind_dir":  weather.get("wind_direction"),
            "wind_mult": round(wind_mult, 4),
            "conditions": weather.get("conditions"),
            "rain_mult": round(rain_mult, 4),
        }

    @staticmethod
    def _temp_mult(temp_f: float) -> float:
        """
        +0.5% per 5°F above 72°F (FanGraphs research).
        Neutral: 72°F. Range: ~[0.96, 1.05] for typical game-day temps.
        """
        delta = (float(temp_f) - _NEUTRAL_TEMP_F) / 5.0
        return float(np.clip(1.0 + delta * _TEMP_RATE, 0.94, 1.06))

    @staticmethod
    def _wind_mult(
        wind_mph:  float,
        wind_from: float,   # meteorological degrees: FROM which direction
        stadium:   Optional[StadiumFactors],
    ) -> float:
        """
        Wind effect based on speed and direction relative to CF orientation.

        wind_from: direction wind is blowing FROM (0° = N, 90° = E, etc.)
        For wind to blow OUT (toward outfield), it must blow FROM the
        direction BEHIND home plate, i.e. FROM (cf_direction + 180°) ± 60°.

        Effect per 5 mph above 5 mph baseline:
            Out  → +2.0% (ball carries, more HRs)
            In   → −1.5% (batters fight the wind)
            Cross→ ±0.3% (very small variance effect)

        Clip: [0.94, 1.10]
        """
        wind_mph = float(wind_mph)
        if wind_mph < 5.0:
            return 1.0

        cf_dir = stadium.cf_direction if stadium else 0
        # Direction wind blows FROM to push ball OUT toward CF:
        # wind must come from BEHIND home plate = opposite of CF direction
        out_from = (cf_dir + 180) % 360

        # Angular distance between today's wind and "out" direction
        diff = abs((float(wind_from) - out_from + 180) % 360 - 180)

        speed_factor = (wind_mph - 5.0) / 5.0   # units of "5 mph above baseline"

        if diff <= 60:          # blowing out (±60° of ideal out direction)
            rate = 0.020
        elif diff >= 120:       # blowing in (±60° of ideal in direction)
            rate = -0.015
        else:                   # crosswind
            rate = 0.003

        return float(np.clip(1.0 + speed_factor * rate, 0.94, 1.10))

    @staticmethod
    def _rain_mult(conditions: str) -> float:
        """
        Light rain / drizzle → pitchers gain slight grip advantage.
        Heavy rain = postponement (not modelled here).
        """
        cond = str(conditions).lower()
        if any(c in cond for c in ("rain", "drizzle", "shower")):
            return 0.97
        if any(c in cond for c in ("thunderstorm", "storm")):
            return 0.96
        return 1.0


# ── Module-level helper ────────────────────────────────────────────────────────

def adjust_for_park_and_weather(
    lh: float,
    la: float,
    game_data: Dict[str, Any],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Convenience wrapper for run_module.py:
        from hfa.park_weather_engine import adjust_for_park_and_weather
        lh, la, meta = adjust_for_park_and_weather(lh, la, game_data)
    """
    return ParkWeatherEngine().adjust_for_park_and_weather(lh, la, game_data)
