#!/usr/bin/env python3
"""
backtest_and_retrain.py
=======================
Backtests the full MLB model pipeline against every game in game_outcomes
(4,695 historical games from 2024-2025).

For each game:
  1. Fetches season-level team and pitcher stats from MLB Stats API
     (disk-cached in .cache/backtest/ — fast on re-runs)
  2. Runs the complete pipeline:
       base_lambda → AutoCalibrator → HFA → PitcherEngine → Regression
       → MonteCarlo(50 000 sims)
  3. Writes real model λ_home, λ_away, p_home, p_away back to game_outcomes,
     replacing the 4.5 priors

After all games:
  4. Recalibrates LearningEngine bias cache (compute_team_bias per team)
  5. Prints + saves a JSON/text summary report

NOTE: Season-level stats are used (no point-in-time cutoff), which introduces
look-ahead bias for the first ~35 games of each season. Results for the
remaining ~127 games per team are free of this bias. The report tags each
game with an early-season flag so you can filter if needed.

Usage:
  python3 backtest_and_retrain.py                 # all 4 695 games
  python3 backtest_and_retrain.py --limit 100     # quick smoke-test
  python3 backtest_and_retrain.py --season 2025   # one season only
  python3 backtest_and_retrain.py --report-only   # skip pipeline, just report
  python3 backtest_and_retrain.py --prefetch       # pre-warm cache, then exit
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ── project root on sys.path ───────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from math import log as _log, exp as _exp
from config import DATA_DIR, LEAGUE_AVG_ERA, LEAGUE_AVG_RUNS, LEAGUE_AVG_WHIP
from data_fetchers import MLBDataIntegrator, MLBStatsAPI
from modules.baseball_module.hfa.park_weather_engine import STADIUM_DATABASE as _STADIUM_DB
from modules.baseball_module.calibration.learning_engine import LearningEngine
from modules.baseball_module.hfa.park_weather_engine import adjust_for_park_and_weather
from modules.baseball_module.hfa.hfa_engine import get_adjusted_lambdas
from modules.baseball_module.context_engine.defensive_efficiency_engine import adjust_for_defense
from modules.baseball_module.context_engine.pitcher_engine import adjust_for_pitchers
from modules.baseball_module.context_engine.bullpen_engine import adjust_for_bullpen
from modules.baseball_module.context_engine.contextual_engine import adjust_for_context
from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced, F5_SCALE

# True Talent Offense Engine
try:
    from modules.baseball_module.offense.true_talent_engine import get_true_talent_lambda as _get_tte_lambda
    _TTE_AVAILABLE = True
except ImportError:
    _TTE_AVAILABLE = False

# External enrichment — Savant + FanGraphs (graceful degradation if unavailable)
try:
    from modules.baseball_module.data_enrichment.savant_fetcher import SavantFetcher
    from modules.baseball_module.data_enrichment.fangraphs_fetcher import FanGraphsFetcher
    _ENRICHMENT_AVAILABLE = True
except ImportError:
    _ENRICHMENT_AVAILABLE = False

# ── paths ──────────────────────────────────────────────────────────────────
DB_PATH = ROOT / "data" / "predictions_history.db"
REPORT_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache" / "backtest"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
N_MC = 50_000

# ── Platt calibration constants (fitted on 4 859-game backtest) ────────────
# Global logistic slope was 0.534 (should be 1.0), intercept 0.144.
# Applying these shrinks over-confident extremes: 80% → 70.8%, 20% → 35.5%.
_PLATT_A = 0.547
_PLATT_B = 0.098


def _platt(p: float, a: float = _PLATT_A, b: float = _PLATT_B) -> float:
    """Shrink an over-confident probability toward calibrated range."""
    p = max(0.01, min(p, 0.99))
    logit = _log(p / (1.0 - p))
    return 1.0 / (1.0 + _exp(-(a * logit + b)))


logging.basicConfig(
    level=logging.WARNING,          # suppress engine chatter during batch
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtest")
log.setLevel(logging.INFO)

# ── static mappings ────────────────────────────────────────────────────────
TEAM_IDS: Dict[str, int] = {
    "Arizona Diamondbacks": 109,
    "Atlanta Braves": 144,
    "Baltimore Orioles": 110,
    "Boston Red Sox": 111,
    "Chicago Cubs": 112,
    "Chicago White Sox": 145,
    "Cincinnati Reds": 113,
    "Cleveland Guardians": 114,
    "Colorado Rockies": 115,
    "Detroit Tigers": 116,
    "Houston Astros": 117,
    "Kansas City Royals": 118,
    "Los Angeles Angels": 108,
    "Los Angeles Dodgers": 119,
    "Miami Marlins": 146,
    "Milwaukee Brewers": 158,
    "Minnesota Twins": 142,
    "New York Mets": 121,
    "New York Yankees": 147,
    "Athletics": 133,
    "Oakland Athletics": 133,
    "Sacramento Athletics": 133,
    "Philadelphia Phillies": 143,
    "Pittsburgh Pirates": 134,
    "San Diego Padres": 135,
    "San Francisco Giants": 137,
    "Seattle Mariners": 136,
    "St. Louis Cardinals": 138,
    "Tampa Bay Rays": 139,
    "Texas Rangers": 140,
    "Toronto Blue Jays": 141,
    "Washington Nationals": 120,
}

# ── F2: Team city coordinates (lat, lon) and timezone offset (UTC hours) ──────
# Used to compute geodesic travel distance and time-zone crossings for the
# away team before each game.  Coordinates are city-centre; tz_offset is the
# standard (non-DST) UTC offset — crossing is always the absolute difference,
# which under-counts by at most 1 when one city observes DST and the other
# doesn't (acceptable approximation).
TEAM_CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "Arizona Diamondbacks":   (33.445, -112.067),   # Phoenix — MST (no DST)
    "Atlanta Braves":         (33.891, -84.468),    # Cumberland/Atlanta — EST
    "Baltimore Orioles":      (39.284, -76.622),    # Baltimore — EST
    "Boston Red Sox":         (42.347, -71.097),    # Boston — EST
    "Chicago Cubs":           (41.948, -87.656),    # Chicago — CST
    "Chicago White Sox":      (41.830, -87.634),    # Chicago — CST
    "Cincinnati Reds":        (39.097, -84.506),    # Cincinnati — EST
    "Cleveland Guardians":    (41.496, -81.685),    # Cleveland — EST
    "Colorado Rockies":       (39.756, -104.994),   # Denver — MST
    "Detroit Tigers":         (42.339, -83.049),    # Detroit — EST
    "Houston Astros":         (29.757, -95.355),    # Houston — CST
    "Kansas City Royals":     (39.051, -94.480),    # Kansas City — CST
    "Los Angeles Angels":     (33.800, -117.883),   # Anaheim — PST
    "Los Angeles Dodgers":    (34.074, -118.240),   # Los Angeles — PST
    "Miami Marlins":          (25.778, -80.220),    # Miami — EST
    "Milwaukee Brewers":      (43.028, -87.971),    # Milwaukee — CST
    "Minnesota Twins":        (44.982, -93.278),    # Minneapolis — CST
    "New York Mets":          (40.757, -73.846),    # New York — EST
    "New York Yankees":       (40.829, -73.926),    # New York — EST
    "Athletics":              (37.752, -122.201),   # Oakland — PST
    "Oakland Athletics":      (37.752, -122.201),   # Oakland — PST
    "Sacramento Athletics":   (38.572, -121.467),   # Sacramento — PST
    "Philadelphia Phillies":  (39.906, -75.167),    # Philadelphia — EST
    "Pittsburgh Pirates":     (40.447, -80.006),    # Pittsburgh — EST
    "San Diego Padres":       (32.707, -117.157),   # San Diego — PST
    "San Francisco Giants":   (37.778, -122.389),   # San Francisco — PST
    "Seattle Mariners":       (47.591, -122.332),   # Seattle — PST
    "St. Louis Cardinals":    (38.623, -90.193),    # St. Louis — CST
    "Tampa Bay Rays":         (27.768, -82.653),    # St. Petersburg — EST
    "Texas Rangers":          (32.751, -97.083),    # Arlington — CST
    "Toronto Blue Jays":      (43.641, -79.389),    # Toronto — EST
    "Washington Nationals":   (38.873, -77.008),    # Washington DC — EST
}

# Standard UTC offset (hours, non-DST) — used for timezone-crossing count only
TEAM_TZ_OFFSET: Dict[str, int] = {
    "Arizona Diamondbacks":   -7,   # MST — no DST year-round
    "Atlanta Braves":         -5,   "Baltimore Orioles":      -5,
    "Boston Red Sox":         -5,   "Chicago Cubs":           -6,
    "Chicago White Sox":      -6,   "Cincinnati Reds":        -5,
    "Cleveland Guardians":    -5,   "Colorado Rockies":       -7,
    "Detroit Tigers":         -5,   "Houston Astros":         -6,
    "Kansas City Royals":     -6,   "Los Angeles Angels":     -8,
    "Los Angeles Dodgers":    -8,   "Miami Marlins":          -5,
    "Milwaukee Brewers":      -6,   "Minnesota Twins":        -6,
    "New York Mets":          -5,   "New York Yankees":       -5,
    "Athletics":              -8,   "Oakland Athletics":      -8,
    "Sacramento Athletics":   -8,   "Philadelphia Phillies":  -5,
    "Pittsburgh Pirates":     -5,   "San Diego Padres":       -8,
    "San Francisco Giants":   -8,   "Seattle Mariners":       -8,
    "St. Louis Cardinals":    -6,   "Tampa Bay Rays":         -5,
    "Texas Rangers":          -6,   "Toronto Blue Jays":      -5,
    "Washington Nationals":   -5,
}


def _geodesic_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in miles."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _travel_stats(away_team: str, home_team: str) -> Tuple[float, int]:
    """
    Returns (miles_traveled, time_zones_crossed) for the away team traveling
    to the home team's city.  Falls back to (0, 0) for unknown teams.
    """
    ac = TEAM_CITY_COORDS.get(away_team)
    hc = TEAM_CITY_COORDS.get(home_team)
    if not ac or not hc:
        return 0.0, 0
    miles = _geodesic_miles(ac[0], ac[1], hc[0], hc[1])
    atz = TEAM_TZ_OFFSET.get(away_team, 0)
    htz = TEAM_TZ_OFFSET.get(home_team, 0)
    tz_cross = abs(htz - atz)
    return round(miles, 1), tz_cross


TEAM_VENUES: Dict[str, str] = {
    "Arizona Diamondbacks": "Chase Field",
    "Atlanta Braves": "Truist Park",
    "Baltimore Orioles": "Camden Yards",
    "Boston Red Sox": "Fenway Park",
    "Chicago Cubs": "Wrigley Field",
    "Chicago White Sox": "Guaranteed Rate Field",
    "Cincinnati Reds": "Great American Ball Park",
    "Cleveland Guardians": "Progressive Field",
    "Colorado Rockies": "Coors Field",
    "Detroit Tigers": "Comerica Park",
    "Houston Astros": "Minute Maid Park",
    "Kansas City Royals": "Kauffman Stadium",
    "Los Angeles Angels": "Angel Stadium",
    "Los Angeles Dodgers": "Dodger Stadium",
    "Miami Marlins": "loanDepot park",
    "Milwaukee Brewers": "American Family Field",
    "Minnesota Twins": "Target Field",
    "New York Mets": "Citi Field",
    "New York Yankees": "Yankee Stadium",
    "Athletics": "RingCentral Coliseum",
    "Oakland Athletics": "RingCentral Coliseum",
    "Sacramento Athletics": "Sutter Health Park",
    "Philadelphia Phillies": "Citizens Bank Park",
    "Pittsburgh Pirates": "PNC Park",
    "San Diego Padres": "Petco Park",
    "San Francisco Giants": "Oracle Park",
    "Seattle Mariners": "T-Mobile Park",
    "St. Louis Cardinals": "Busch Stadium",
    "Tampa Bay Rays": "Tropicana Field",
    "Texas Rangers": "Globe Life Field",
    "Toronto Blue Jays": "Rogers Centre",
    "Washington Nationals": "Nationals Park",
}

# ── disk cache ─────────────────────────────────────────────────────────────

class DiskCache:
    """Thin JSON file cache.  ttl=0 → permanent (backtest data never expires)."""

    def __init__(self, cache_dir: Path, ttl: int = 0):
        self._dir = cache_dir
        self._ttl = ttl

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_")
        return self._dir / f"{safe}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if not p.exists():
            return None
        if self._ttl and (time.time() - p.stat().st_mtime) > self._ttl:
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        try:
            self._path(key).write_text(json.dumps(value))
        except Exception:
            pass


# ── API helpers ────────────────────────────────────────────────────────────

def _get(session: requests.Session, url: str, params: dict,
         retries: int = 3, delay: float = 0.15) -> Optional[dict]:
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5)
                continue
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))
    return None


def fetch_starters(game_pk: int, session: requests.Session,
                   cache: DiskCache) -> Dict[str, Optional[int]]:
    """Return actual starting pitcher IDs from the game boxscore."""
    key = f"starters_{game_pk}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    data = _get(session, f"{MLB_API_BASE}/game/{game_pk}/boxscore", {})
    if not data:
        result = {"home_pitcher_id": None, "away_pitcher_id": None}
        cache.set(key, result)
        return result
    teams = data.get("teams", {})
    hp = teams.get("home", {}).get("pitchers", [])
    ap = teams.get("away", {}).get("pitchers", [])
    result = {
        "home_pitcher_id": int(hp[0]) if hp else None,
        "away_pitcher_id": int(ap[0]) if ap else None,
    }
    cache.set(key, result)
    return result


def _team_id(name: str) -> Optional[int]:
    return TEAM_IDS.get(name) or TEAM_IDS.get(name.strip())


def prefetch_team_stats(api: MLBStatsAPI, seasons: List[int]) -> None:
    """Pre-warm the existing .cache/ team stat files for all 30 teams."""
    log.info("Pre-fetching team stats for all 30 teams × %s seasons …", seasons)
    for season in seasons:
        for name, tid in TEAM_IDS.items():
            if tid in {133}:  # Athletics already counted once
                if name != "Athletics":
                    continue
            api.get_team_offensive_stats(tid, season)
            api.get_team_pitching_stats(tid, season)
            api.get_bullpen_era(tid, season)
        log.info("  season %s team stats cached.", season)


# ── game-data builder ──────────────────────────────────────────────────────

def _default_team_dict(name: str, rpg: float = LEAGUE_AVG_RUNS) -> Dict:
    """League-average team dict used as a fallback when stats are missing."""
    return {
        "name": name,
        "runs_per_game": rpg,
        "woba": 0.320, "ops": 0.735, "wrc_plus": 100.0,
        "team_era": LEAGUE_AVG_ERA, "team_whip": LEAGUE_AVG_WHIP,
        "runs_allowed_per_game": LEAGUE_AVG_RUNS,
        "last_10": "5-5", "streak": "",
        "rest_days": 1, "wins": 81, "losses": 81,
        "home_record": "42-39", "away_record": "39-42",
        "home_runs_per_game": rpg, "away_runs_per_game": rpg,
        "miles_traveled": 0, "time_zones_crossed": 0,
    }


def build_game_data(
    game_pk: int,
    game_date: str,
    home_name: str,
    away_name: str,
    season: int,
    home_pitcher_id: Optional[int],
    away_pitcher_id: Optional[int],
    api: MLBStatsAPI,
    integrator: MLBDataIntegrator,
    park_factors: Optional[object],
    savant_stats: Optional[Dict[int, Dict]] = None,
    fg_stats: Optional[Dict[int, Dict]] = None,
) -> Tuple[Dict, float, float]:
    """
    Assemble game_data and (lh_base, la_base) from cached season-level stats.
    Returns (game_data, lh_base, la_base).
    """
    htid = _team_id(home_name)
    atid = _team_id(away_name)

    # ── team offensive stats ────────────────────────────────────────────────
    home_off = (api.get_team_offensive_stats(htid, season) or {}) if htid else {}
    away_off = (api.get_team_offensive_stats(atid, season) or {}) if atid else {}

    # ── team pitching / defense ─────────────────────────────────────────────
    home_pitch = (api.get_team_pitching_stats(htid, season) or {}) if htid else {}
    away_pitch = (api.get_team_pitching_stats(atid, season) or {}) if atid else {}

    # ── bullpen ERA ─────────────────────────────────────────────────────────
    home_bp = (api.get_bullpen_era(htid, season) or {}) if htid else {}
    away_bp = (api.get_bullpen_era(atid, season) or {}) if atid else {}

    # ── RPG for base lambda ─────────────────────────────────────────────────
    home_rpg_api = integrator._fetch_team_rpg(htid, season) if htid else LEAGUE_AVG_RUNS
    away_rpg_api = integrator._fetch_team_rpg(atid, season) if atid else LEAGUE_AVG_RUNS

    # ── base lambdas (season RPG blend — same logic as live run_module) ─────
    lh = integrator.get_team_lambda(home_name, home_rpg_api, team_id=htid)
    la = integrator.get_team_lambda(away_name, away_rpg_api, team_id=atid)

    # ── pitcher stats (5-level fallback) ────────────────────────────────────
    def _pitcher_stats(pid: Optional[int], is_home: bool, team_pitch: Dict) -> Dict:
        if not pid:
            return {}
        stats, source = api.get_pitcher_stats_full_fallback(
            pid, season,
            team_pitching=team_pitch,
            is_home=is_home,
            is_playoff=False,
        )
        if not stats:
            return {}
        # Enrich with game-log recency and F5 data (MLB levels only; skip for MiLB)
        if source not in ("aaa_current", "aa_current", "team_staff_era"):
            gl = api.get_pitcher_game_log(pid, season)
            if gl:
                stats.update(gl)
            f5 = api.get_pitcher_f5_stats(pid, season)
            if f5:
                stats.update(f5)
        return stats

    home_ps = _pitcher_stats(home_pitcher_id, is_home=True,  team_pitch=home_pitch)
    away_ps = _pitcher_stats(away_pitcher_id, is_home=False, team_pitch=away_pitch)

    # ── team sub-dicts (what calibrator + HFA + pitcher engine expect) ──────
    def _team_dict(name: str, off: dict, pitch: dict, rpg: float,
                   is_home: bool) -> Dict:
        d = _default_team_dict(name, rpg)
        d.update({
            "woba":                off.get("woba", 0.320),
            "ops":                 off.get("ops", 0.735),
            "wrc_plus":            off.get("wrc_plus", 100.0),
            "team_era":            pitch.get("team_era", LEAGUE_AVG_ERA),
            "team_whip":           pitch.get("team_whip", LEAGUE_AVG_WHIP),
            "runs_allowed_per_game": pitch.get("runs_allowed_per_game", LEAGUE_AVG_RUNS),
            "runs_per_game":       rpg,
            "home_runs_per_game":  rpg,
            "away_runs_per_game":  rpg,
        })
        return d

    home_dict = _team_dict(home_name, home_off, home_pitch, home_rpg_api, True)
    away_dict = _team_dict(away_name, away_off, away_pitch, away_rpg_api, False)

    def _pitcher_dict(name: str, ps: dict, team_pitch: dict,
                      sv: dict = None, fg: dict = None) -> Dict:
        # Use team staff ERA as fallback so we never silently inject 4.38.
        team_era = float(team_pitch.get("team_era", LEAGUE_AVG_ERA)) if team_pitch else LEAGUE_AVG_ERA
        team_whip = float(team_pitch.get("team_whip", LEAGUE_AVG_WHIP)) if team_pitch else LEAGUE_AVG_WHIP
        era = ps.get("era", team_era)
        sv  = sv or {}
        fg  = fg or {}
        return {
            "name": name or "Unknown",
            "era": era,
            "fip": ps.get("fip", era),
            "whip": ps.get("whip", team_whip),
            "k_per_9": ps.get("k_per_9", 8.5),
            "era_last_5": ps.get("era_last_5", era),
            "days_rest": ps.get("days_rest", 4),
            "last_pitch_count": ps.get("last_pitch_count", 90),
            "avg_innings_per_start": ps.get("avg_innings_per_start"),
            "f5_era": ps.get("f5_era"),
            # FanGraphs real ERA estimators
            "xfip":            fg.get("xfip"),
            "siera":           fg.get("siera"),
            "war":             fg.get("war"),
            "k_pct":           fg.get("k_pct"),
            "bb_pct":          fg.get("bb_pct"),
            "swstr_pct":       fg.get("swstr_pct"),
            "hr_fb_pct":       fg.get("hr_fb"),
            "babip":           fg.get("babip"),
            "lob_pct":         fg.get("lob_pct"),
            "innings_pitched": fg.get("ip"),   # for Bayesian regression in pitcher_engine
            # Baseball Savant contact quality
            "est_woba":        sv.get("est_woba"),
            "xera":            sv.get("xera") or fg.get("xera"),
            "brl_percent":     sv.get("brl_percent"),
            "avg_hit_speed":   sv.get("avg_hit_speed"),
            "ev95percent":     sv.get("ev95percent"),
        }

    venue = TEAM_VENUES.get(home_name, "Unknown")

    game_data: Dict[str, Any] = {
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "home_team": home_dict,
        "away_team": away_dict,
        "home_team_id": htid,
        "away_team_id": atid,
        "venue": venue,
        "park": {"name": venue},
        "pitcher_home": _pitcher_dict(
            f"Home SP ({game_pk})", home_ps, home_pitch,
            sv=(savant_stats or {}).get(home_pitcher_id or 0, {}),
            fg=(fg_stats or {}).get(home_pitcher_id or 0, {}),
        ),
        "pitcher_away": _pitcher_dict(
            f"Away SP ({game_pk})", away_ps, away_pitch,
            sv=(savant_stats or {}).get(away_pitcher_id or 0, {}),
            fg=(fg_stats or {}).get(away_pitcher_id or 0, {}),
        ),
        "home_pitcher_stats": home_ps,
        "away_pitcher_stats": away_ps,
        "bullpen_home": home_bp,
        "bullpen_away": away_bp,
        # F2: geodesic travel distance and timezone crossings for away team
        "miles_traveled_away": (_ts := _travel_stats(away_name, home_name))[0],
        "time_zones_crossed_away": _ts[1],
        # F3: B2B flags populated externally in the main loop (needs schedule context)
        "back_to_back_away": False,
        "back_to_back_home": False,
        "home_days_rest": 1,
        "away_days_rest": 1,
    }
    _stadium = _STADIUM_DB.get(venue)
    game_data["park_factor"] = _stadium.runs_factor if _stadium else 1.00

    return game_data, lh, la


# ── pipeline runner ────────────────────────────────────────────────────────

def run_pipeline(
    game_data: Dict,
    lh: float,
    la: float,
    learning: LearningEngine,
    season: int,
    n_mc: int = N_MC,
) -> Dict[str, Any]:
    """Run the full pipeline identical to run_module.py.

    Returns a dict with keys:
      lh, la            — final post-pipeline lambdas
      p_home, p_away    — Platt-calibrated win probabilities (sum to 1)
      p_home_raw        — raw Monte Carlo p_home *before* Platt (for clean Platt refitting)
      p_away_raw        — raw Monte Carlo p_away *before* Platt
      stage_factors     — per-stage raw adjustment ratios (for gradient descent)
      n_mc              — actual simulations run
    """
    home_team = game_data.get("home_team", {}).get("name", "")
    away_team = game_data.get("away_team", {}).get("name", "")
    _sf: Dict[str, float] = {}   # stage factors — raw ratios, weight-independent

    # ── TTE base lambda (fallback to legacy get_team_lambda) ─────────────────
    htid = game_data.get("home_team_id")
    atid = game_data.get("away_team_id")
    tte_active = False
    if _TTE_AVAILABLE and htid and atid:
        try:
            lh, _ = _get_tte_lambda(htid, home_team, season)
            la, _ = _get_tte_lambda(atid, away_team, season)
            tte_active = True
        except Exception:
            pass  # TTE failed — lh/la remain from build_game_data

    # ── Kalman adjustment (walk-forward: only sees games prior to this one) ───
    lh = learning.get_kalman_lambda_adjustment(home_team, "offense_home", season, lh)
    la = learning.get_kalman_lambda_adjustment(away_team, "offense_away", season, la)

    # ── Learned pipeline weights ──────────────────────────────────────────────
    _w = learning.get_pipeline_weights(season)

    # ── Team bias (LearningEngine) ────────────────────────────────────────────
    lh *= learning.compute_team_bias_kalman_adjusted(
        game_data.get("home_team", {}).get("name", ""), season, "offense_home"
    )
    la *= learning.compute_team_bias_kalman_adjusted(
        game_data.get("away_team", {}).get("name", ""), season, "offense_away"
    )

    # ── PASO 2: Park + Weather ────────────────────────────────────────────────
    _lh_pre, _la_pre = lh, la
    lh_park, la_park, _ = adjust_for_park_and_weather(lh, la, game_data)
    _raw_h = lh_park / _lh_pre if _lh_pre else 1.0
    _raw_a = la_park / _la_pre if _la_pre else 1.0
    _w_park = _w.get("park", 1.0)
    lh = _lh_pre * (1.0 + _w_park * (_raw_h - 1.0))
    la = _la_pre * (1.0 + _w_park * (_raw_a - 1.0))
    _sf["home_park"] = _raw_h
    _sf["away_park"] = _raw_a

    # ── PASO 3: HFA (crowd + travel asymmetric) ───────────────────────────────
    _lh_pre, _la_pre = lh, la
    lh_hfa, la_hfa, _ = get_adjusted_lambdas(lh, la, game_data)
    _raw_h = lh_hfa / _lh_pre if _lh_pre else 1.0
    _raw_a = la_hfa / _la_pre if _la_pre else 1.0
    _w_hfa = _w.get("hfa", 1.0)
    lh = _lh_pre * (1.0 + _w_hfa * (_raw_h - 1.0))
    la = _la_pre * (1.0 + _w_hfa * (_raw_a - 1.0))
    _sf["home_hfa"] = _raw_h
    _sf["away_hfa"] = _raw_a

    # ── PASO 4: Defensive Efficiency ──────────────────────────────────────────
    _lh_pre, _la_pre = lh, la
    if game_data.get("defense_home") or game_data.get("defense_away"):
        lh_def, la_def, _ = adjust_for_defense(lh, la, game_data)
        _raw_h = lh_def / _lh_pre if _lh_pre else 1.0
        _raw_a = la_def / _la_pre if _la_pre else 1.0
    else:
        _raw_h = _raw_a = 1.0
    _w_def = _w.get("defense", 1.0)
    lh = _lh_pre * (1.0 + _w_def * (_raw_h - 1.0))
    la = _la_pre * (1.0 + _w_def * (_raw_a - 1.0))
    _sf["home_defense"] = _raw_h
    _sf["away_defense"] = _raw_a

    # ── PASO 5: Pitcher Engine ────────────────────────────────────────────────
    _lh_pre, _la_pre = lh, la
    lh_pit, la_pit, _ = adjust_for_pitchers(lh, la, game_data)
    _raw_h = lh_pit / _lh_pre if _lh_pre else 1.0
    _raw_a = la_pit / _la_pre if _la_pre else 1.0
    _w_pit = _w.get("pitcher", 1.0)
    lh = _lh_pre * (1.0 + _w_pit * (_raw_h - 1.0))
    la = _la_pre * (1.0 + _w_pit * (_raw_a - 1.0))
    _sf["home_pitcher"] = _raw_h
    _sf["away_pitcher"] = _raw_a

    # ── F5 lambda — snapshot post-pitcher, pre-bullpen ────────────────────────
    lh_f5 = round(lh * F5_SCALE, 3)
    la_f5 = round(la * F5_SCALE, 3)

    # ── PASO 6: Bullpen Engine ────────────────────────────────────────────────
    _lh_pre, _la_pre = lh, la
    if game_data.get("bullpen_home") or game_data.get("bullpen_away"):
        lh_bp, la_bp, _ = adjust_for_bullpen(lh, la, game_data)
        _raw_h = lh_bp / _lh_pre if _lh_pre else 1.0
        _raw_a = la_bp / _la_pre if _la_pre else 1.0
    else:
        _raw_h = _raw_a = 1.0
    _w_bp = _w.get("bullpen", 1.0)
    lh = _lh_pre * (1.0 + _w_bp * (_raw_h - 1.0))
    la = _la_pre * (1.0 + _w_bp * (_raw_a - 1.0))
    _sf["home_bullpen"] = _raw_h
    _sf["away_bullpen"] = _raw_a

    # ── PASO 7: Contextual Engine (rest / B2B / umpire) ──────────────────────
    _lh_pre, _la_pre = lh, la
    lh_ctx, la_ctx, _ = adjust_for_context(lh, la, game_data)
    _raw_h = lh_ctx / _lh_pre if _lh_pre else 1.0
    _raw_a = la_ctx / _la_pre if _la_pre else 1.0
    _w_ctx = _w.get("context", 1.0)
    lh = _lh_pre * (1.0 + _w_ctx * (_raw_h - 1.0))
    la = _la_pre * (1.0 + _w_ctx * (_raw_a - 1.0))
    _sf["home_context"] = _raw_h
    _sf["away_context"] = _raw_a

    # ── Sanity clamp ──────────────────────────────────────────────────────────
    lh = max(1.5, min(lh, 12.0))
    la = max(1.5, min(la, 12.0))

    # ── PASO 8: Monte Carlo ───────────────────────────────────────────────────
    mc = monte_carlo_advanced(
        lh=lh, la=la, n_max=n_mc,
        block=min(10_000, n_mc),
        analyze_f5=False,
        lh_f5=lh_f5, la_f5=la_f5,
    )

    # Capture raw MC probabilities BEFORE Platt — needed for clean Platt refitting.
    # Using post-Platt values as training targets creates a circular dependency.
    p_home_mc = mc["p_home"]
    p_away_mc = mc["p_away"]

    # ── Platt calibration (dynamic params from learning engine) ───────────────
    # Applied asymmetrically — same logic as run_module.py. Symmetric normalization
    # cancels the b intercept for neutral games, destroying the structural home
    # advantage that recalibrate_platt() learned.
    _pa, _pb = learning.get_platt_params(season)
    _p_h     = _platt(p_home_mc, _pa, _pb)
    p_home_cal = round(_p_h, 5)
    p_away_cal = round(1.0 - _p_h, 5)

    return {
        "lh":          round(lh, 4),
        "la":          round(la, 4),
        "p_home":      p_home_cal,
        "p_away":      p_away_cal,
        "p_home_raw":  round(p_home_mc, 5),   # pre-Platt, for Platt refitting
        "p_away_raw":  round(p_away_mc, 5),
        "stage_factors": _sf,
        "n_mc":        mc["n"],
    }


# ── database helpers ────────────────────────────────────────────────────────

def get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _add_backtest_col(conn: sqlite3.Connection) -> None:
    """Add backtest_run_at column if absent (idempotent)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(game_outcomes)")}
    if "backtest_run_at" not in cols:
        conn.execute("ALTER TABLE game_outcomes ADD COLUMN backtest_run_at TEXT")
        conn.commit()


def update_game_outcomes(
    conn: sqlite3.Connection,
    game_pk: int,
    lh: float, la: float,
    p_home: float, p_away: float,
    p_home_raw: Optional[float] = None,
    p_away_raw: Optional[float] = None,
    stage_factors: Optional[Dict] = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    sf_json = json.dumps(stage_factors) if stage_factors else None
    conn.execute(
        """
        UPDATE game_outcomes
        SET lambda_home = ?, lambda_away = ?,
            p_home = ?, p_away = ?,
            p_home_raw = ?, p_away_raw = ?,
            stage_factors_json = ?,
            backtest_run_at = ?
        WHERE game_pk = ?
        """,
        (lh, la, p_home, p_away, p_home_raw, p_away_raw, sf_json, now, game_pk),
    )


# ── pinnacle devig ─────────────────────────────────────────────────────────

def _devig(o1: float, o2: float) -> Tuple[float, float]:
    """Multiplicative devig of a two-outcome market."""
    t = 1.0 / o1 + 1.0 / o2
    return (1.0 / o1) / t, (1.0 / o2) / t


# ── report ─────────────────────────────────────────────────────────────────

def generate_report(
    results: List[Dict],
    out_path: Path,
) -> None:
    """Compute and print a full backtest summary; save JSON sidecar."""
    n = len(results)
    if n == 0:
        log.warning("No results to report.")
        return

    # ── basic accuracy ──────────────────────────────────────────────────────
    correct = sum(1 for r in results if r["model_correct"])
    accuracy = correct / n

    brier_model  = sum((r["p_home"] - r["home_won"]) ** 2 for r in results) / n
    brier_random = 0.25

    def _safe_log(p: float) -> float:
        return math.log(max(p, 1e-9))

    logloss_model = -sum(
        r["home_won"] * _safe_log(r["p_home"])
        + (1 - r["home_won"]) * _safe_log(r["p_away"])
        for r in results
    ) / n

    # ── Pinnacle comparison ─────────────────────────────────────────────────
    pin_rows = [r for r in results if r["pin_fair_home"] is not None]
    n_pin = len(pin_rows)

    brier_pin = (
        sum((r["pin_fair_home"] - r["home_won"]) ** 2 for r in pin_rows) / n_pin
        if n_pin else None
    )
    logloss_pin = (
        -sum(
            r["home_won"] * _safe_log(r["pin_fair_home"])
            + (1 - r["home_won"]) * _safe_log(r["pin_fair_away"])
            for r in pin_rows
        ) / n_pin
        if n_pin else None
    )

    edges = [r["model_edge"] for r in pin_rows if r["model_edge"] is not None]
    agree_pct = (
        sum(
            1 for r in pin_rows
            if (r["p_home"] > 0.5) == (r["pin_fair_home"] > 0.5)
        ) / n_pin * 100
        if n_pin else None
    )

    # ── calibration by bucket ───────────────────────────────────────────────
    buckets: Dict[str, Dict] = {
        "<40%":   {"min": 0.00, "max": 0.40},
        "40-45%": {"min": 0.40, "max": 0.45},
        "45-50%": {"min": 0.45, "max": 0.50},
        "50-55%": {"min": 0.50, "max": 0.55},
        "55-60%": {"min": 0.55, "max": 0.60},
        "60-70%": {"min": 0.60, "max": 0.70},
        ">70%":   {"min": 0.70, "max": 1.01},
    }
    for b in buckets.values():
        b["n"] = b["wins"] = b["sum_p"] = 0

    for r in results:
        ph = r["p_home"]
        hw = r["home_won"]
        for b in buckets.values():
            if b["min"] <= ph < b["max"]:
                b["n"] += 1
                b["wins"] += hw
                b["sum_p"] += ph
                break

    # ── ROI simulation (flat 1-unit at Pinnacle odds) ──────────────────────
    # Bets the side with the highest positive edge per game (home or away).
    # Games where either Pinnacle line exceeds 4.0 decimal (+300 American) are
    # filtered out — they indicate emergency/position-player pitcher situations
    # the model cannot price, and produced phantom 15–41% edges in the audit.
    _EXTREME_ODDS_THRESHOLD = 4.0
    n_extreme_filtered = sum(
        1 for r in pin_rows
        if (r.get("ml_home_pin") or 0) > _EXTREME_ODDS_THRESHOLD
        or (r.get("ml_away_pin") or 0) > _EXTREME_ODDS_THRESHOLD
    )
    bettable_rows = [
        r for r in pin_rows
        if (r.get("ml_home_pin") or 0) <= _EXTREME_ODDS_THRESHOLD
        and (r.get("ml_away_pin") or 0) <= _EXTREME_ODDS_THRESHOLD
        and r.get("ml_home_pin") and r.get("ml_away_pin")
    ]

    thresholds = [0.00, 0.02, 0.05, 0.08, 0.10]
    roi_table: Dict[float, Dict] = {}
    for thr in thresholds:
        bets = staked = profit = 0
        clv_values: list = []
        for r in bettable_rows:
            edge_home = r.get("model_edge")
            edge_away = r.get("model_edge_away")

            # Pick the side with the larger positive edge; skip if neither clears thr
            if (edge_home or -1) >= (edge_away or -1) and (edge_home or -1) >= thr:
                best_edge = edge_home
                odds = r["ml_home_pin"]
                won = bool(r["home_won"])
                clv = r.get("clv_home")
            elif (edge_away or -1) >= thr:
                best_edge = edge_away
                odds = r["ml_away_pin"]
                won = not bool(r["home_won"])
                clv = r.get("clv_away")
            else:
                continue

            bets += 1
            staked += 1.0
            profit += (odds - 1) if won else -1.0
            if clv is not None:
                clv_values.append(clv)

        mean_clv = sum(clv_values) / len(clv_values) if clv_values else None
        pos_clv_pct = (
            sum(1 for c in clv_values if c > 0) / len(clv_values) * 100
            if clv_values else None
        )
        roi_table[thr] = {
            "bets": bets,
            "staked": round(staked, 2),
            "profit": round(profit, 4),
            "roi_pct": round(profit / staked * 100, 2) if staked > 0 else 0.0,
            "n_extreme_filtered": n_extreme_filtered,
            # CLV: model probability / Pinnacle fair probability - 1
            # Positive → model was pricing ahead of closing line (real edge signal)
            "mean_clv_pct": round(mean_clv * 100, 3) if mean_clv is not None else None,
            "pos_clv_pct":  round(pos_clv_pct, 1) if pos_clv_pct is not None else None,
        }

    # ── season breakdown ────────────────────────────────────────────────────
    by_season: Dict[int, Dict] = defaultdict(
        lambda: {"n": 0, "correct": 0, "brier": 0.0}
    )
    for r in results:
        s = r["season"]
        by_season[s]["n"] += 1
        by_season[s]["correct"] += r["model_correct"]
        by_season[s]["brier"] += (r["p_home"] - r["home_won"]) ** 2

    # ── lambda distribution ─────────────────────────────────────────────────
    lhs = sorted(r["lh"] for r in results)
    las = sorted(r["la"] for r in results)

    def _pct(lst: list, p: float) -> float:
        idx = int(len(lst) * p)
        return round(lst[min(idx, len(lst) - 1)], 3)

    # ── assemble report dict ────────────────────────────────────────────────
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_games": n,
        "seasons": sorted({r["season"] for r in results}),
        "overall": {
            "accuracy_pct":  round(accuracy * 100, 2),
            "brier_model":   round(brier_model, 5),
            "brier_random":  brier_random,
            "brier_vs_random_pct": round((brier_random - brier_model) / brier_random * 100, 2),
            "logloss_model": round(logloss_model, 5),
        },
        "vs_pinnacle": {
            "n_games_with_pin": n_pin,
            "brier_pinnacle":   round(brier_pin, 5) if brier_pin else None,
            "brier_model_vs_pin_pct": (
                round((brier_pin - brier_model) / brier_pin * 100, 2)
                if brier_pin else None
            ),
            "logloss_pinnacle": round(logloss_pin, 5) if logloss_pin else None,
            "model_pin_agreement_pct": round(agree_pct, 2) if agree_pct else None,
            "edge_vs_pin": {
                "mean":  round(sum(edges) / len(edges) * 100, 3) if edges else None,
                "std":   round(
                    math.sqrt(sum((e - sum(edges) / len(edges)) ** 2 for e in edges)
                              / len(edges)) * 100, 3
                ) if len(edges) > 1 else None,
                "p10":   round(sorted(edges)[int(len(edges) * 0.10)] * 100, 2) if edges else None,
                "p25":   round(sorted(edges)[int(len(edges) * 0.25)] * 100, 2) if edges else None,
                "p50":   round(sorted(edges)[int(len(edges) * 0.50)] * 100, 2) if edges else None,
                "p75":   round(sorted(edges)[int(len(edges) * 0.75)] * 100, 2) if edges else None,
                "p90":   round(sorted(edges)[int(len(edges) * 0.90)] * 100, 2) if edges else None,
            },
        },
        "calibration": {
            label: {
                "n": b["n"],
                "predicted_win_pct": round(b["sum_p"] / b["n"] * 100, 1) if b["n"] else None,
                "actual_win_pct":    round(b["wins"] / b["n"] * 100, 1) if b["n"] else None,
            }
            for label, b in buckets.items()
        },
        "roi_simulation": {
            f"edge>={int(thr*100)}%": roi_table[thr]
            for thr in thresholds
        },
        "by_season": {
            str(s): {
                "n": v["n"],
                "accuracy_pct": round(v["correct"] / v["n"] * 100, 2),
                "brier": round(v["brier"] / v["n"], 5),
            }
            for s, v in sorted(by_season.items())
        },
        "lambda_distribution": {
            "lh": {"p10": _pct(lhs, .10), "p25": _pct(lhs, .25), "median": _pct(lhs, .50),
                   "p75": _pct(lhs, .75), "p90": _pct(lhs, .90),
                   "mean": round(sum(lhs) / len(lhs), 3)},
            "la": {"p10": _pct(las, .10), "p25": _pct(las, .25), "median": _pct(las, .50),
                   "p75": _pct(las, .75), "p90": _pct(las, .90),
                   "mean": round(sum(las) / len(las), 3)},
        },
    }

    # ── save JSON ────────────────────────────────────────────────────────────
    report_path = out_path / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    report_path.write_text(json.dumps(report, indent=2))

    # ── print summary ────────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  BACKTEST REPORT  —  {n} games  |  seasons: {report['seasons']}")
    print(sep)
    ov = report["overall"]
    print(f"\n  OVERALL MODEL PERFORMANCE")
    print(f"  {'Accuracy':30s}  {ov['accuracy_pct']:>7.2f}%")
    print(f"  {'Brier score (model)':30s}  {ov['brier_model']:>7.5f}")
    print(f"  {'Brier score (random 50/50)':30s}  {ov['brier_random']:>7.5f}")
    print(f"  {'Improvement vs random':30s}  {ov['brier_vs_random_pct']:>7.2f}%")
    print(f"  {'Log-loss':30s}  {ov['logloss_model']:>7.5f}")

    vp = report["vs_pinnacle"]
    if vp["n_games_with_pin"]:
        print(f"\n  VS PINNACLE  ({vp['n_games_with_pin']} games)")
        print(f"  {'Brier (Pinnacle)':30s}  {vp['brier_pinnacle']:>7.5f}")
        print(f"  {'Brier (model)':30s}  {ov['brier_model']:>7.5f}")
        print(f"  {'Model vs Pinnacle Brier':30s}  {vp['brier_model_vs_pin_pct']:>+7.2f}%")
        print(f"  {'Log-loss (Pinnacle)':30s}  {vp['logloss_pinnacle']:>7.5f}")
        print(f"  {'Log-loss (model)':30s}  {ov['logloss_model']:>7.5f}")
        print(f"  {'Favorite agreement':30s}  {vp['model_pin_agreement_pct']:>7.2f}%")
        ed = vp["edge_vs_pin"]
        print(f"  {'Mean model edge vs Pin':30s}  {ed['mean']:>+7.3f}%")
        print(f"  {'Edge std dev':30s}  {ed['std']:>7.3f}%")
        print(f"  {'Edge p25/p50/p75':30s}  "
              f"{ed['p25']:>+6.2f}% / {ed['p50']:>+6.2f}% / {ed['p75']:>+6.2f}%")

    print(f"\n  CALIBRATION")
    print(f"  {'Bucket':10s}  {'N':>6s}  {'Pred%':>7s}  {'Actual%':>8s}  {'Diff':>7s}")
    for label, c in report["calibration"].items():
        if c["n"] == 0:
            continue
        diff = (c["actual_win_pct"] or 0) - (c["predicted_win_pct"] or 0)
        print(f"  {label:10s}  {c['n']:>6d}  {c['predicted_win_pct']:>7.1f}%  "
              f"{c['actual_win_pct']:>7.1f}%  {diff:>+6.1f}%")

    _n_filt = next(iter(report["roi_simulation"].values()), {}).get("n_extreme_filtered", 0)
    print(f"\n  ROI SIMULATION (flat 1-unit, best-edge side @ Pinnacle)")
    print(f"  Extreme-odds games filtered (pin>4.0): {_n_filt}")
    print(f"  {'Edge threshold':16s}  {'Bets':>6s}  {'Profit':>8s}  {'ROI':>8s}  {'Mean CLV':>9s}  {'CLV>0':>6s}")
    for label, rt in report["roi_simulation"].items():
        if rt["bets"] == 0:
            continue
        clv_str = f"{rt['mean_clv_pct']:>+8.3f}%" if rt.get("mean_clv_pct") is not None else "      N/A"
        pos_str = f"{rt['pos_clv_pct']:>5.1f}%" if rt.get("pos_clv_pct") is not None else "   N/A"
        print(f"  {label:16s}  {rt['bets']:>6d}  {rt['profit']:>+8.2f}u  "
              f"{rt['roi_pct']:>+7.2f}%  {clv_str}  {pos_str}")

    print(f"\n  BY SEASON")
    for s, sv in report["by_season"].items():
        print(f"  {s}  N={sv['n']:>4d}  accuracy={sv['accuracy_pct']:>5.2f}%  "
              f"Brier={sv['brier']:.5f}")

    print(f"\n  LAMBDA DISTRIBUTION (final, post-pipeline)")
    ld = report["lambda_distribution"]
    print(f"  λ_home  mean={ld['lh']['mean']:.3f}  "
          f"p25={ld['lh']['p25']:.3f}  median={ld['lh']['median']:.3f}  "
          f"p75={ld['lh']['p75']:.3f}")
    print(f"  λ_away  mean={ld['la']['mean']:.3f}  "
          f"p25={ld['la']['p25']:.3f}  median={ld['la']['median']:.3f}  "
          f"p75={ld['la']['p75']:.3f}")

    print(f"\n  Report saved → {report_path}")
    print(sep)


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MLB pipeline backtest + retrain")
    parser.add_argument("--limit",       type=int,   default=0,
                        help="Process only the first N games (0 = all)")
    parser.add_argument("--season", "--seasons", type=int, default=0,
                        help="Restrict to one season (0 = all)")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip pipeline; just re-generate report from current DB state")
    parser.add_argument("--prefetch",    action="store_true",
                        help="Pre-warm team and starter caches, then exit")
    parser.add_argument("--no-cache",    action="store_true",
                        help="Ignore cached starter lookups (re-fetch everything)")
    parser.add_argument("--workers",     type=int,   default=1,
                        help="Parallel starter-fetch workers (default 1 = sequential)")
    args = parser.parse_args()

    # ── load game_outcomes rows ─────────────────────────────────────────────
    conn = get_conn(DB_PATH)
    _add_backtest_col(conn)

    where = "WHERE actual_home_runs IS NOT NULL"
    if args.season:
        where += f" AND season = {args.season}"
    order = "ORDER BY game_date ASC"
    limit = f"LIMIT {args.limit}" if args.limit else ""
    rows = conn.execute(
        f"SELECT * FROM game_outcomes {where} {order} {limit}"
    ).fetchall()

    seasons = sorted({r["season"] for r in rows})
    log.info("Loaded %d games  |  seasons: %s", len(rows), seasons)

    # ── init shared objects ─────────────────────────────────────────────────
    session = requests.Session()
    api = MLBStatsAPI()
    integrator = MLBDataIntegrator()
    park_factors = None  # park factors now read directly from STADIUM_DATABASE per venue
    cache = DiskCache(CACHE_DIR)
    learning = LearningEngine(db_path=DB_PATH)

    # ── load Savant + FanGraphs data per season ──────────────────────────────
    _enrich_cache_dir = ROOT / ".cache"
    savant_by_season: Dict[int, Dict[int, Dict]] = {}
    fg_by_season: Dict[int, Dict[int, Dict]] = {}
    if _ENRICHMENT_AVAILABLE:
        _sv_fetcher = SavantFetcher(cache_dir=_enrich_cache_dir)
        _fg_fetcher = FanGraphsFetcher(cache_dir=_enrich_cache_dir)
        for _yr in seasons:
            log.info("Loading Savant + FanGraphs stats for season %d …", _yr)
            savant_by_season[_yr] = _sv_fetcher.get_all_pitcher_stats(_yr)
            fg_by_season[_yr]     = _fg_fetcher.get_all_pitcher_stats(_yr)
            log.info("  Savant: %d pitchers | FG: %d pitchers",
                     len(savant_by_season[_yr]), len(fg_by_season[_yr]))

    if args.no_cache:
        cache = DiskCache(CACHE_DIR, ttl=1)  # effectively bypasses old entries

    # ── prefetch mode ───────────────────────────────────────────────────────
    if args.prefetch:
        prefetch_team_stats(api, seasons)
        log.info("Pre-fetching starters for %d games …", len(rows))
        for i, row in enumerate(rows, 1):
            fetch_starters(row["game_pk"], session, cache)
            if i % 500 == 0:
                log.info("  starters: %d / %d", i, len(rows))
            time.sleep(0.08)
        log.info("Prefetch complete. Exiting.")
        conn.close()
        return

    # ── report-only mode ────────────────────────────────────────────────────
    if args.report_only:
        result_rows = conn.execute(
            f"""
            SELECT game_pk, season, home_team, away_team,
                   lambda_home AS lh, lambda_away AS la,
                   p_home, p_away,
                   actual_home_runs, actual_away_runs, home_won,
                   ml_home_pin, ml_away_pin,
                   market_prob_home, market_prob_away
            FROM game_outcomes {where}
            AND backtest_run_at IS NOT NULL
            {order} {limit}
            """
        ).fetchall()
        results = []
        for r in result_rows:
            pin_fh = pin_fa = None
            if r["ml_home_pin"] and r["ml_away_pin"]:
                pin_fh, pin_fa = _devig(r["ml_home_pin"], r["ml_away_pin"])
            results.append({
                "game_pk":         r["game_pk"], "season": r["season"],
                "lh":              r["lh"], "la": r["la"],
                "p_home":          r["p_home"], "p_away": r["p_away"],
                "home_won":        r["home_won"],
                "model_correct":   int((r["p_home"] > 0.5) == bool(r["home_won"])),
                "pin_fair_home":   pin_fh, "pin_fair_away": pin_fa,
                "ml_home_pin":     r["ml_home_pin"],
                "ml_away_pin":     r["ml_away_pin"],
                "model_edge":      (r["p_home"] - pin_fh) if pin_fh else None,
                "model_edge_away": (r["p_away"] - pin_fa) if pin_fa else None,
                # CLV: how much model probability exceeds Pinnacle fair probability (%).
                # Positive CLV → model was pricing ahead of where market closed.
                "clv_home": (r["p_home"] / pin_fh - 1.0) if (pin_fh and pin_fh > 0) else None,
                "clv_away": (r["p_away"] / pin_fa - 1.0) if (pin_fa and pin_fa > 0) else None,
            })
        generate_report(results, REPORT_DIR)
        conn.close()
        return

    # ── pre-warm team stats ─────────────────────────────────────────────────
    prefetch_team_stats(api, seasons)

    # ── walk-forward learning state reset ──────────────────────────────────
    # Three adaptive structures must be reset before the walk-forward loop:
    #
    #   1. Kalman states  — were built from full-season data (look-ahead bias).
    #      Each game will call update_kalman() *after* its prediction, so the
    #      filter sees only past observations.
    #
    #   2. Pipeline weights — were optimised on Kalman-look-ahead-biased
    #      predictions.  Gradient descent re-learns from clean walk-forward data.
    #
    #   3. Platt params — reset to identity (a=1.0, b=0.0) per season, then
    #      warm-started from the prior season's fitted params.  This matches
    #      how production operates: day 1 of season N uses the calibration
    #      fitted on season N-1.  Without warm-start the backtest Brier
    #      measures raw MC performance (no Platt applied), which under-reports
    #      the model's true production accuracy by ~1.4%.
    #
    log.info("Resetting walk-forward learning state for seasons %s …", list(seasons))

    # Capture prior-season Platt params BEFORE reset so warm-start can use them.
    # Seasons processed in ascending order so each N seeds from N-1's fitted params.
    _prior_platt: Dict[int, Optional[Dict]] = {}
    for season in sorted(seasons):
        _prior_platt[season] = learning.load_state("platt_params", "calibration", season - 1)

    n_kal_deleted = learning.reset_kalman_for_seasons(list(seasons))
    n_wt_reset    = learning.reset_pipeline_weights(list(seasons))
    n_plt_deleted = learning.reset_platt_params(list(seasons))
    log.info(
        "  Kalman: %d rows deleted | weights: %d seasons reset | Platt: %d rows deleted",
        n_kal_deleted, n_wt_reset, n_plt_deleted,
    )

    # Warm-start each season's Platt with the prior season's fitted params.
    # Minimum 50 samples required (same threshold as recalibrate_platt).
    _PLATT_MIN_SAMPLES = 50
    for season in sorted(seasons):
        prior = _prior_platt.get(season)
        if prior and prior.get("n", 0) >= _PLATT_MIN_SAMPLES:
            learning.save_state(
                "platt_params", "calibration",
                {"a": prior["a"], "b": prior["b"], "c": prior.get("c", 0.0), "n": 0},
                sample_count=0,
                season=season,
            )
            log.info(
                "  Platt warm-start season %d ← season %d: a=%.4f b=%.4f",
                season, season - 1, prior["a"], prior["b"],
            )
        else:
            log.info(
                "  Platt season %d: no prior-season params (n=%d) — using identity defaults",
                season, prior.get("n", 0) if prior else 0,
            )

    # ── main backtest loop ──────────────────────────────────────────────────
    results: List[Dict] = []
    n_ok = n_err = 0
    t0 = time.time()

    # F3: track last game date AND venue per team to detect meaningful B2B.
    # "Meaningful B2B" = played yesterday AND changed cities (overnight travel),
    # as opposed to within-series consecutive games in the same venue (no extra
    # fatigue beyond the normal 3-4 game road-trip rhythm).
    # Key: team name → (last_date "YYYY-MM-DD", last_venue str)
    _team_last_game: Dict[str, Tuple[str, str]] = {}

    for idx, row in enumerate(rows, 1):
        game_pk   = row["game_pk"]
        game_date = row["game_date"]
        season    = row["season"]
        home_name = row["home_team"]
        away_name = row["away_team"]
        home_won  = row["home_won"]

        # F3: detect meaningful B2B — played yesterday AND changed cities.
        # Within-series games (same venue, consecutive days) are normal MLB
        # rhythm and don't trigger extra fatigue.  The penalty applies when a
        # team flew overnight to reach the current game.
        _today_str = game_date[:10]                          # "YYYY-MM-DD"
        _yesterday = (
            datetime.strptime(_today_str, "%Y-%m-%d").date()
            - __import__('datetime').timedelta(days=1)
        ).strftime("%Y-%m-%d")
        _cur_venue = TEAM_VENUES.get(home_name, "Unknown")

        _away_last = _team_last_game.get(away_name, ("", ""))
        _home_last = _team_last_game.get(home_name, ("", ""))

        # Away B2B: played yesterday AND their last venue ≠ current venue
        # (they traveled overnight between cities)
        _b2b_away = (_away_last[0] == _yesterday and _away_last[1] != _cur_venue)

        # Home B2B: played yesterday in a different venue (away game yesterday,
        # home game today — they also had to travel/return home)
        _b2b_home = (_home_last[0] == _yesterday and _home_last[1] != _cur_venue)

        # Update last-game tracker AFTER computing B2B
        _team_last_game[away_name] = (_today_str, _cur_venue)
        _team_last_game[home_name] = (_today_str, _cur_venue)

        # starting pitchers
        starters = fetch_starters(game_pk, session, cache)
        time.sleep(0.08)           # gentle rate limit (≈12 req/s)

        try:
            game_data, lh, la = build_game_data(
                game_pk, game_date, home_name, away_name, season,
                starters["home_pitcher_id"],
                starters["away_pitcher_id"],
                api, integrator, park_factors,
                savant_stats=savant_by_season.get(season),
                fg_stats=fg_by_season.get(season),
            )
            # F3: inject B2B flags computed from schedule context above
            game_data["back_to_back_away"] = _b2b_away
            game_data["back_to_back_home"] = _b2b_home

            pred = run_pipeline(game_data, lh, la, learning, season, N_MC)

            # persist to DB
            update_game_outcomes(
                conn, game_pk,
                pred["lh"], pred["la"],
                pred["p_home"], pred["p_away"],
                p_home_raw=pred.get("p_home_raw"),
                p_away_raw=pred.get("p_away_raw"),
                stage_factors=pred.get("stage_factors"),
            )
            conn.commit()

            # Walk-forward Kalman: update *after* writing results so subsequent
            # games in this loop see the current observation — zero look-ahead.
            learning.update_kalman(home_name, "offense_home", season, float(row["actual_home_runs"]))
            learning.update_kalman(away_name, "offense_away", season, float(row["actual_away_runs"]))
            learning.update_kalman(home_name, "defense_home", season, float(row["actual_away_runs"]))
            learning.update_kalman(away_name, "defense_away", season, float(row["actual_home_runs"]))

            # Pinnacle fair prob
            pin_fh = pin_fa = None
            if row["ml_home_pin"] and row["ml_away_pin"]:
                pin_fh, pin_fa = _devig(row["ml_home_pin"], row["ml_away_pin"])

            results.append({
                "game_pk":        game_pk,
                "season":         season,
                "lh":             pred["lh"],
                "la":             pred["la"],
                "p_home":         pred["p_home"],
                "p_away":         pred["p_away"],
                "home_won":       int(home_won),
                "model_correct":  int((pred["p_home"] > 0.5) == bool(home_won)),
                "pin_fair_home":  pin_fh,
                "pin_fair_away":  pin_fa,
                "ml_home_pin":    row["ml_home_pin"],
                "ml_away_pin":    row["ml_away_pin"],
                "model_edge":     (pred["p_home"] - pin_fh) if pin_fh else None,
                "model_edge_away": (pred["p_away"] - pin_fa) if pin_fa else None,
                "clv_home": (pred["p_home"] / pin_fh - 1.0) if (pin_fh and pin_fh > 0) else None,
                "clv_away": (pred["p_away"] / pin_fa - 1.0) if (pin_fa and pin_fa > 0) else None,
            })
            n_ok += 1

        except Exception as exc:
            log.warning("game_pk=%d FAILED: %s", game_pk, exc)
            n_err += 1
            p_h = row["p_home"]
            if p_h is None:
                continue   # no prior prediction to fall back to; skip from report
            results.append({
                "game_pk":         game_pk, "season": season,
                "lh":              row["lambda_home"], "la": row["lambda_away"],
                "p_home":          p_h, "p_away": row["p_away"],
                "home_won":        int(home_won),
                "model_correct":   int((p_h > 0.5) == bool(home_won)),
                "pin_fair_home":   None, "pin_fair_away": None,
                "ml_home_pin":     row["ml_home_pin"],
                "ml_away_pin":     row["ml_away_pin"],
                "model_edge":      None, "model_edge_away": None,
                "clv_home":        None, "clv_away":        None,
            })

        # progress log every 250 games
        if idx % 250 == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            eta  = (len(rows) - idx) / rate
            log.info(
                "  [%d/%d] ok=%d err=%d  %.1f g/s  ETA %.0fs",
                idx, len(rows), n_ok, n_err, rate, eta,
            )

    conn.commit()

    elapsed_total = time.time() - t0
    log.info(
        "Pipeline complete: %d ok / %d err  (%.1fs, %.1f g/s)",
        n_ok, n_err, elapsed_total, len(rows) / elapsed_total,
    )

    # ── step 4: recalibrate learning engine bias ───────────────────────────
    log.info("Recalibrating learning engine bias …")
    teams = conn.execute(
        "SELECT DISTINCT home_team AS t FROM game_outcomes "
        "UNION SELECT DISTINCT away_team FROM game_outcomes"
    ).fetchall()

    refreshed = 0
    for t in teams:
        for season in seasons:
            bias = learning.compute_team_bias(t["t"], season)
            if bias != 1.0:
                refreshed += 1

    log.info("  %d teams | %d non-neutral biases written to ml_state", len(teams), refreshed)

    # ── step 4b: Platt recalibration per season ─────────────────────────────
    # Kalman states are now fully populated by the walk-forward loop above.
    log.info("Running Platt recalibration per season …")
    for season in seasons:
        try:
            a, b = learning.recalibrate_platt(season)
            log.info("  Season %d: Platt a=%.4f b=%.4f", season, a, b)
        except Exception as exc:
            log.warning("  Platt failed for season %d: %s", season, exc)

    # ── step 5: report ─────────────────────────────────────────────────────
    generate_report(results, REPORT_DIR)
    conn.close()


if __name__ == "__main__":
    main()
