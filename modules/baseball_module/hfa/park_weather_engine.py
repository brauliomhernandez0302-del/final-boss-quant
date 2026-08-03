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

Pipeline position: PASO 5 — after Bullpen Engine, before Defensive Efficiency Engine.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

# Neutral temperature for run-environment baseline (°F)
_NEUTRAL_TEMP_F = 72.0
# Run-increase per 5°F above neutral
_TEMP_RATE = 0.005

# Wind × handedness asymmetry scale.
# A team with 60% LHB (vs 45% avg = +15% deviation) in a perfect 20 mph
# RF cross-wind gets a ~0.6% scoring boost relative to the symmetric baseline.
# Small by design — this is a second-order correction on top of the primary
# in/out wind effect (which is already 2% per 5 mph unit).
_HANDEDNESS_WIND_SCALE = 0.040
# Minimum wind speed for handedness correction to activate (mph)
_HANDEDNESS_WIND_MIN_MPH = 10.0
# Fracción zurda media — fuente única en config.py, ya no un 0.45 duplicado acá
# y en pitcher_engine. Ver config.LEAGUE_AVG_LHB_PCT para la medición.
from config import LEAGUE_AVG_LHB_PCT as _AVG_LHB_PCT


def _first_present(*values: Optional[float]) -> float:
    """Return the first value that is not None (0.0 is a valid, kept value)."""
    for v in values:
        if v is not None:
            return float(v)
    return 0.0


def _signed_crosswind(wind_from: float, cf_direction: int) -> float:
    """
    Signed crosswind component relative to the LF/RF split.

    Returns +1.0 when wind blows fully toward RF (LHB advantage — lefties pull to RF).
    Returns -1.0 when wind blows fully toward LF (RHB advantage — righties pull to LF).
    Returns  0.0 for pure in/out winds (no lateral component).

    CF is at `cf_direction` from home plate.
    LF is at cf_direction − 45°, RF is at cf_direction + 45° (approximate).
    Wind pushes toward RF when it blows FROM the LF side (cf_direction − 45°).
    Wind pushes toward LF when it blows FROM the RF side (cf_direction + 45°).
    """
    lf_side_from = (cf_direction - 45) % 360   # wind from here → pushes toward RF (+)
    rf_side_from = (cf_direction + 45) % 360   # wind from here → pushes toward LF (−)

    diff_lf = abs((wind_from - lf_side_from + 180) % 360 - 180)  # 0 = perfect RF push
    diff_rf = abs((wind_from - rf_side_from + 180) % 360 - 180)  # 0 = perfect LF push

    rf_component = max(0.0, 1.0 - diff_lf / 90.0)  # 1 at 0°, 0 at 90°+
    lf_component = max(0.0, 1.0 - diff_rf / 90.0)

    return rf_component - lf_component  # + = RF wind (LHB adv), − = LF wind (RHB adv)


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
    # Same sponsorship/naming-drift situation as Daikin Park / UNIQLO Field
    # below: live MLB API reports "Oriole Park at Camden Yards" (confirmed
    # 2026-07-13 via game_outcomes), historical game_data uses "Camden
    # Yards". Without this key, live Orioles home games fell back to
    # neutral park_mult=1.00 and lost cf_direction.
    "Oriole Park at Camden Yards":
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
    # Renamed "Rate Field" for 2026 (sponsorship change) — live MLB API
    # confirmed reporting the new name via game_outcomes 2026-07-13. Same
    # building/factors as Guaranteed Rate Field; kept both keys.
    "Rate Field":
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
    # Astros' park renamed Minute Maid Park -> Daikin Park (sponsorship change).
    # Same building/factors — kept both keys: live MLB API now reports "Daikin
    # Park" (confirmed 2026-07-05, e.g. game_pk 824172), but historical
    # game_data (2024/2025 backtest) still reports the old name. Without the
    # new key, every live Astros home game fell back to STADIUM_DATABASE.get()
    # returning None -> neutral park_mult=1.00 and lost cf_direction, silently
    # discarding this park's real (0.99 runs, 1.02 HR, has_roof) factors.
    "Minute Maid Park":
        StadiumFactors(runs_factor=0.99, hr_factor=1.02, hits_factor=0.99,
                       altitude=13,  cf_direction=340, has_roof=True),
    "Daikin Park":
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
    # A's moved to Sacramento for 2025 season (new stadium, neutral prior)
    "Sutter Health Park":
        StadiumFactors(runs_factor=1.00, hr_factor=1.00, hits_factor=1.00,
                       altitude=12,  cf_direction=0),
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
    # Same sponsorship-rename situation as Daikin Park above: live MLB API now
    # reports "UNIQLO Field at Dodger Stadium" (confirmed 2026-07-05), while
    # historical game_data (2024/2025) still reports "Dodger Stadium". Kept
    # both keys pointing at the same real factors for the same reason.
    "Dodger Stadium":
        StadiumFactors(runs_factor=0.99, hr_factor=1.10, hits_factor=0.99,
                       altitude=155, cf_direction=340),
    "UNIQLO Field at Dodger Stadium":
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

        park_name = (game_data.get("park") or {}).get("name", "Unknown")
        stadium   = STADIUM_DATABASE.get(park_name)
        if stadium is None:
            log.warning("Park '%s' not in STADIUM_DATABASE — using neutral factor 1.00", park_name)
        park_mult = stadium.runs_factor if stadium else 1.00

        # Determine if park is sheltered today
        roof_closed = bool(game_data.get("roof_closed", False))
        if stadium and stadium.has_roof and not game_data.get("roof_open", False):
            roof_closed = True   # default retractable parks to closed

        weather     = game_data.get("weather", {})
        # FALL-001 fix (roadmap Step 4, audit_20260714/): purely additive
        # provenance marker — does NOT change weather_mult's value or how
        # it's computed (that stays _weather_mult()'s unmodified neutral
        # path when `weather` is empty). Distinguishes "genuinely neutral
        # conditions" from "we don't know" (an unmapped/renamed venue —
        # REG-015's exact failure mode — an API fetch failure, or the
        # backtest, which is weather-blind by design and will therefore
        # report 'missing' on every single game; see CONTRACTS.md).
        weather_source = "live" if weather else "missing"
        weather_mult, wx_meta = self._weather_mult(weather, stadium, roof_closed)

        total_mult = park_mult * weather_mult
        lh_new     = lh * total_mult
        la_new     = la * total_mult

        # ── Wind × handedness asymmetry ────────────────────────────────────────
        # Crosswind toward RF benefits LHB-heavy lineups; toward LF benefits RHB.
        # Only active when wind ≥ 10 mph, roof is open, and handedness data exists.
        hnd_home = hnd_away = 0.0
        if (
            not roof_closed
            and stadium
            and float(weather.get("wind_speed_mph") or 0) >= _HANDEDNESS_WIND_MIN_MPH
        ):
            wind_mph = float(weather.get("wind_speed_mph", 0))
            wind_dir = float(weather.get("wind_direction", 0))
            # Prefer today's confirmed-lineup handedness (same field pitcher_engine
            # uses for platoon) over the team-season average — falls back cleanly
            # when called without it (e.g. from the backtest, which doesn't set it).
            home_lhb = _first_present(
                game_data.get("home_lineup_lhb_pct"), game_data.get("home_lhb_pct"), _AVG_LHB_PCT,
            )
            away_lhb = _first_present(
                game_data.get("away_lineup_lhb_pct"), game_data.get("away_lhb_pct"), _AVG_LHB_PCT,
            )

            speed_factor  = (wind_mph - _HANDEDNESS_WIND_MIN_MPH) / 10.0
            signed_cross  = _signed_crosswind(wind_dir, stadium.cf_direction)

            # delta: how much batting team's LHB% deviates from average
            # positive signed_cross (RF wind) × positive lhb_delta → boost
            home_lhb_delta = home_lhb - _AVG_LHB_PCT
            away_lhb_delta = away_lhb - _AVG_LHB_PCT
            hnd_home = signed_cross * home_lhb_delta * speed_factor * _HANDEDNESS_WIND_SCALE
            hnd_away = signed_cross * away_lhb_delta * speed_factor * _HANDEDNESS_WIND_SCALE
            hnd_home = max(-0.015, min(0.015, hnd_home))
            hnd_away = max(-0.015, min(0.015, hnd_away))

            lh_new *= (1.0 + hnd_home)
            la_new *= (1.0 + hnd_away)

        meta = {
            "park_name":              park_name,
            "park_factor":            round(park_mult,    4),
            "weather_mult":           round(weather_mult, 4),
            "weather_source":         weather_source,
            "total_mult":             round(total_mult,   4),
            "roof_closed":            roof_closed,
            "postponement_risk":      weather.get("postponement_risk",
                                                  float(weather.get("rain_mm", 0)) > 10.0),
            "wind_handedness_home":   round(hnd_home, 4),
            "wind_handedness_away":   round(hnd_away, 4),
            **wx_meta,
        }

        hnd_str = (
            f"  hnd_h={hnd_home:+.4f} hnd_a={hnd_away:+.4f}"
            if (hnd_home or hnd_away) else ""
        )
        log.info(
            "Park+Weather | %s  park=%.3f  weather=%.3f  total=%.3f%s  "
            "λ_h %.3f→%.3f  λ_a %.3f→%.3f",
            park_name, park_mult, weather_mult, total_mult, hnd_str,
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

        _tf  = weather.get("temp_f");         tf  = float(_tf  if _tf  is not None else _NEUTRAL_TEMP_F)
        _ws  = weather.get("wind_speed_mph"); ws  = float(_ws  if _ws  is not None else 0.0)
        _wd  = weather.get("wind_direction"); wd  = float(_wd  if _wd  is not None else 0.0)
        temp_mult = self._temp_mult(tf)
        wind_mult = self._wind_mult(ws, wd, stadium)
        rain_mult = self._rain_mult(weather)  # full dict — uses rain_mm + pop when available

        combined = max(0.90, min(1.12, temp_mult * wind_mult * rain_mult))

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
        return max(0.94, min(1.06, 1.0 + delta * _TEMP_RATE))

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

        return max(0.94, min(1.10, 1.0 + speed_factor * rate))

    @staticmethod
    def _rain_mult(weather: Dict[str, Any]) -> float:
        """
        Precipitation effect on run scoring, scaled by actual intensity.

        Uses rain_mm (total mm across game window) when available;
        falls back to conditions string for backwards compatibility.

        Effect: wet ball + slick grip → pitchers lose movement on breaking
        balls but batters also struggle.  Net: slight scoring reduction.
        Intensity tiers (mm across ~3h game window):
            0–1  mm  → -1%  (drizzle / light mist)
            1–5  mm  → -3%  (light rain, playable)
            5–10 mm  → -5%  (moderate rain, pitcher severely hampered)
            >10  mm  → postponement — should not reach this engine
        Probability-weighted: effect × precip_probability so a 30% chance
        of light rain contributes less than a 90% chance.
        """
        rain_mm = float(weather.get("rain_mm", 0.0))
        pop     = float(weather.get("precip_probability", 1.0))  # default 1.0 for string fallback

        if rain_mm > 0:
            if rain_mm <= 1.0:
                raw = 0.99
            elif rain_mm <= 5.0:
                raw = 0.97
            else:
                raw = 0.95
            # Weight effect by precipitation probability
            return 1.0 + (raw - 1.0) * pop

        # Fallback: conditions string (no rain_mm available)
        cond = str(weather.get("conditions", "")).lower()
        if any(c in cond for c in ("thunderstorm", "storm")):
            return 0.95
        if any(c in cond for c in ("rain", "shower")):
            return 0.97
        if "drizzle" in cond:
            return 0.99
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
