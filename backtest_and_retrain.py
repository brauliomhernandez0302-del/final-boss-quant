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
       base_lambda → LearningEngine(Kalman+bias) → Park/Weather → HFA
       → Defense → PitcherEngine → Bullpen → Context → MonteCarlo(50 000 sims)
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
import atexit
import json
import logging
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
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
from modules.baseball_module.calibration.learning_engine import LearningEngine, untruncate_home_runs
from modules.baseball_module.hfa.park_weather_engine import adjust_for_park_and_weather
# DEFERRED F7: import kept for reactivation post-Sprint 3
# from modules.baseball_module.hfa.historical_weather import HistoricalWeatherFetcher
from modules.baseball_module.hfa.hfa_engine import get_adjusted_lambdas
from modules.baseball_module.context_engine.defensive_efficiency_engine import adjust_for_defense
from modules.baseball_module.context_engine.pitcher_engine import adjust_for_pitchers
from modules.baseball_module.context_engine.bullpen_engine import (
    adjust_for_bullpen,
    BULLPEN_CLAMP_LOW,
    BULLPEN_CLAMP_HIGH,
)
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

TEAM_TTE_PIT_ENTITY_IDS: Dict[str, str] = {
    "Arizona Diamondbacks": "AZ",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Oakland Athletics": "ATH",
    "Sacramento Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
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
    "Athletics":              (38.572, -121.467),   # Sacramento — PST
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
    "Athletics": "Sutter Health Park",
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


def prefetch_team_stats(
    api: MLBStatsAPI,
    seasons: List[int],
    *,
    include_legacy_bullpen: bool = True,
) -> None:
    """Pre-warm the existing .cache/ team stat files for all 30 teams."""
    log.info("Pre-fetching team stats for all 30 teams × %s seasons …", seasons)
    for season in seasons:
        for name, tid in TEAM_IDS.items():
            if tid in {133}:  # Athletics already counted once
                if name != "Athletics":
                    continue
            api.get_team_pitching_stats(tid, season)
            if include_legacy_bullpen:
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
    weather_fetcher: Optional["HistoricalWeatherFetcher"] = None,
    pitcher_game_log_as_of_date: Optional[str] = None,
    use_pitcher_full_season_fallback: bool = True,
    use_team_full_season_offense_base: bool = True,
    use_team_full_season_defense: bool = True,
    use_legacy_full_season_bullpen: bool = True,
    use_team_full_season_pitching_base: bool = True,
) -> Tuple[Dict, float, float]:
    """
    Assemble game_data and (lh_base, la_base) from cached season-level stats.
    Returns (game_data, lh_base, la_base).
    """
    htid = _team_id(home_name)
    atid = _team_id(away_name)

    # get_team_offensive_stats() was deleted 2026-07-06 (data_fetchers.py
    # dead-code review — its woba/ops/wrc_plus output was traced to
    # game_data['home_team']/['away_team'] and confirmed never read by any
    # engine downstream, same as the identical dead pattern found and fixed
    # in run_module.py's own team_dict construction the same day). home_off/
    # away_off now stay empty; _team_dict()'s existing .get(..., default)
    # fallbacks handle this the same way they already handle any missing data.
    home_off: Dict = {}
    away_off: Dict = {}

    # ── team pitching / defense ─────────────────────────────────────────────
    home_pitch = (api.get_team_pitching_stats(htid, season) or {}) if htid else {}
    away_pitch = (api.get_team_pitching_stats(atid, season) or {}) if atid else {}

    # ── bullpen ERA ─────────────────────────────────────────────────────────
    home_bp = (
        (api.get_bullpen_era(htid, season) or {})
        if htid and use_legacy_full_season_bullpen
        else {}
    )
    away_bp = (
        (api.get_bullpen_era(atid, season) or {})
        if atid and use_legacy_full_season_bullpen
        else {}
    )

    if use_team_full_season_offense_base:
        # ── RPG for base lambda ─────────────────────────────────────────────
        home_rpg_api = integrator._fetch_team_rpg(htid, season) if htid else LEAGUE_AVG_RUNS
        away_rpg_api = integrator._fetch_team_rpg(atid, season) if atid else LEAGUE_AVG_RUNS

        # ── base lambdas (season RPG blend — same logic as live run_module) ─
        lh = integrator.get_team_lambda(home_name, home_rpg_api, team_id=htid)
        la = integrator.get_team_lambda(away_name, away_rpg_api, team_id=atid)
    else:
        home_rpg_api = LEAGUE_AVG_RUNS
        away_rpg_api = LEAGUE_AVG_RUNS
        lh = LEAGUE_AVG_RUNS
        la = LEAGUE_AVG_RUNS

    # ── pitcher stats (5-level fallback) ────────────────────────────────────
    def _pitcher_stats(pid: Optional[int], is_home: bool, team_pitch: Dict) -> Dict:
        if not pid:
            return {}
        if not use_pitcher_full_season_fallback:
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
            if pitcher_game_log_as_of_date is None:
                gl = api.get_pitcher_game_log(pid, season)
            else:
                gl = api.get_pitcher_game_log(
                    pid,
                    season,
                    as_of_date=pitcher_game_log_as_of_date,
                )
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
        # team_era/team_whip/runs_allowed_per_game come from the same
        # season-aggregate get_team_pitching_stats() call as der/bip (which
        # use_team_full_season_defense already gates). Without this flag they
        # leaked into home_dict/away_dict unconditionally regardless of PIT
        # mode, concentrated exactly in the thin-IP fallback-of-fallback path
        # in run_module.py where a starter's own era/whip is missing.
        if use_team_full_season_pitching_base:
            team_era = pitch.get("team_era", LEAGUE_AVG_ERA)
            team_whip = pitch.get("team_whip", LEAGUE_AVG_WHIP)
            runs_allowed = pitch.get("runs_allowed_per_game", LEAGUE_AVG_RUNS)
        else:
            team_era = LEAGUE_AVG_ERA
            team_whip = LEAGUE_AVG_WHIP
            runs_allowed = LEAGUE_AVG_RUNS
        d.update({
            "woba":                off.get("woba", 0.320),
            "ops":                 off.get("ops", 0.735),
            "wrc_plus":            off.get("wrc_plus", 100.0),
            "team_era":            team_era,
            "team_whip":           team_whip,
            "runs_allowed_per_game": runs_allowed,
            "runs_per_game":       rpg,
            "home_runs_per_game":  rpg,
            "away_runs_per_game":  rpg,
        })
        return d

    home_dict = _team_dict(home_name, home_off, home_pitch, home_rpg_api, True)
    away_dict = _team_dict(away_name, away_off, away_pitch, away_rpg_api, False)

    def _pitcher_dict(name: str, ps: dict, team_pitch: dict,
                      sv: dict = None, fg: dict = None) -> Dict:
        # In experimental pitcher PIT mode, missing pitcher snapshots must not
        # inherit full-season team staff stats. Use league average until PIT
        # overlays explicit point-in-time fields.
        if use_pitcher_full_season_fallback:
            team_era = float(team_pitch.get("team_era", LEAGUE_AVG_ERA)) if team_pitch else LEAGUE_AVG_ERA
            team_whip = float(team_pitch.get("team_whip", LEAGUE_AVG_WHIP)) if team_pitch else LEAGUE_AVG_WHIP
        else:
            team_era = LEAGUE_AVG_ERA
            team_whip = LEAGUE_AVG_WHIP
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
            # for Bayesian regression in pitcher_engine (quality_mult, 32% weight).
            # FanGraphs IP preferred when available; falls back to the MLB Stats
            # API season IP (same fallback as the live path in run_module.py) so
            # pitchers FanGraphs doesn't cover (rookies, call-ups, name-matching
            # misses) don't silently collapse quality_mult to exactly 1.0.
            "innings_pitched": fg.get("ip") if fg.get("ip") is not None else ps.get("innings_pitched", 0),
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
        # F6: Defensive efficiency — DER from season pitching stats (same source as live path).
        # OAA from Baseball Savant not available via free API; engine degrades to DER-only.
        "defense_home": (
            {"team_name": home_name, "der": home_pitch["der"], "bip": home_pitch["bip"], "oaa": None}
            if use_team_full_season_defense
            and home_pitch.get("der")
            and home_pitch.get("bip", 0) > 0
            else {}
        ),
        "defense_away": (
            {"team_name": away_name, "der": away_pitch["der"], "bip": away_pitch["bip"], "oaa": None}
            if use_team_full_season_defense
            and away_pitch.get("der")
            and away_pitch.get("bip", 0) > 0
            else {}
        ),
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

    # F7: inject historical weather data (Open-Meteo ERA5 reanalysis)
    if weather_fetcher is not None:
        game_data["weather"] = weather_fetcher.get(venue, game_date)

    return game_data, lh, la


_PIT_REQUIRED_FIELDS = (
    "siera",
    "xfip",
    "fip",
    "k_pct",
    "bb_pct",
    "innings_pitched",
    "est_woba",
    "brl_percent",
    "ev95percent",
)


def _prediction_cutoff_for_row(row: sqlite3.Row | Dict[str, Any]) -> str:
    """Return the PIT cutoff for a historical game row."""
    keys = row.keys() if hasattr(row, "keys") else row
    for field in ("prediction_cutoff_utc", "prediction_cutoff", "as_of_date"):
        if field in keys and row[field]:
            return str(row[field])
    return f"{str(row['game_date'])[:10]}T23:59:59Z"


def _experimental_pitcher_pit_cutoff_for_row(row: sqlite3.Row | Dict[str, Any]) -> str:
    """Return the experimental pitcher PIT cutoff aligned to previous-day caches."""
    keys = row.keys() if hasattr(row, "keys") else row
    for field in ("prediction_cutoff_utc", "prediction_cutoff", "as_of_date"):
        if field in keys and row[field]:
            return str(row[field])
    game_day = datetime.strptime(str(row["game_date"])[:10], "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    return (game_day - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _team_tte_pit_cutoff_for_row(row: sqlite3.Row | Dict[str, Any]) -> str:
    """Return the Team/TTE PIT cutoff aligned to previous-day team snapshots."""
    game_day = datetime.strptime(str(row["game_date"])[:10], "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    return (game_day - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _defense_pit_cutoff_for_row(row: sqlite3.Row | Dict[str, Any]) -> str:
    """Return the previous-day cutoff required by Defense PIT."""
    game_day = datetime.strptime(str(row["game_date"])[:10], "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    return (game_day - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bullpen_pit_cutoff_for_row(row: sqlite3.Row | Dict[str, Any]) -> str:
    """Return the previous-day cutoff required by Bullpen PIT."""
    game_day = datetime.strptime(str(row["game_date"])[:10], "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    return (game_day - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _merge_pit_pitcher(base: Dict[str, Any], adapted: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay PIT values onto the existing MLB Stats API safe fallback dict."""
    merged = dict(base)
    for key, value in adapted.items():
        if value is not None:
            merged[key] = value
    merged["pitcher_pit_snapshot"] = adapted
    return merged


def apply_experimental_pitcher_pit_mode(
    *,
    game_data: Dict[str, Any],
    home_pitcher_id: Optional[int],
    away_pitcher_id: Optional[int],
    season: int,
    requested_as_of_date: str,
    snapshot_builder: Any,
    adapter: Any,
) -> Dict[str, Any]:
    """Inject adapted PIT pitcher snapshots into game_data when available."""

    def _apply(role: str, pitcher_id: Optional[int]) -> Dict[str, Any]:
        meta = {
            "pitcher_id": pitcher_id,
            "pit_found": False,
            "fangraphs_found": False,
            "savant_found": False,
            "prior_baseline_found": False,
            "fangraphs_as_of_date": None,
            "savant_as_of_date": None,
            "prior_baseline_as_of_date": None,
            "provenance_source": "league_average_safe_fallback",
            "missing_fields": list(_PIT_REQUIRED_FIELDS),
            "source_fingerprints": {"fangraphs": None, "savant": None},
            "fallback_used": True,
        }
        if not pitcher_id:
            return meta

        snapshot = snapshot_builder.build_pitcher_snapshot(
            pitcher=pitcher_id,
            season=season,
            requested_as_of_date=requested_as_of_date,
        )
        adapted = adapter(snapshot)
        found = bool(adapted.get("found"))
        meta.update(
            {
                "pit_found": found,
                "fangraphs_found": bool(adapted.get("fangraphs_found")),
                "savant_found": bool(adapted.get("savant_found")),
                "prior_baseline_found": bool(adapted.get("prior_baseline_found")),
                "fangraphs_as_of_date": adapted.get("fangraphs_as_of_date"),
                "savant_as_of_date": adapted.get("savant_as_of_date"),
                "prior_baseline_as_of_date": adapted.get("prior_baseline_as_of_date"),
                "provenance_source": adapted.get(
                    "provenance_source", "league_average_safe_fallback"
                ),
                "missing_fields": [
                    field for field in _PIT_REQUIRED_FIELDS if adapted.get(field) is None
                ],
                "source_fingerprints": adapted.get("source_fingerprints", {}),
                "fallback_used": not found,
            }
        )
        if found:
            game_data[role] = _merge_pit_pitcher(game_data.get(role, {}), adapted)
        return meta

    metadata = {
        "requested_as_of_date": requested_as_of_date,
        "home": _apply("pitcher_home", home_pitcher_id),
        "away": _apply("pitcher_away", away_pitcher_id),
    }
    metadata.update(
        {
            "home_pitcher_pit_found": metadata["home"]["pit_found"],
            "away_pitcher_pit_found": metadata["away"]["pit_found"],
            "home_fangraphs_as_of_date": metadata["home"]["fangraphs_as_of_date"],
            "home_savant_as_of_date": metadata["home"]["savant_as_of_date"],
            "away_fangraphs_as_of_date": metadata["away"]["fangraphs_as_of_date"],
            "away_savant_as_of_date": metadata["away"]["savant_as_of_date"],
            "home_missing_fields": metadata["home"]["missing_fields"],
            "away_missing_fields": metadata["away"]["missing_fields"],
        }
    )
    game_data["experimental_pitcher_pit"] = metadata
    return metadata


def apply_experimental_team_tte_pit_mode(
    *,
    game_data: Dict[str, Any],
    season: int,
    game_date: str,
    snapshot_builder: Any,
    adapter: Any,
    requested_as_of_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Build Team/TTE PIT lambdas for a historical game without touching live TTE."""

    requested = requested_as_of_date or _team_tte_pit_cutoff_for_row(
        {"game_date": game_date}
    )

    def _apply(side: str, team_id: Optional[int], team_name: str) -> Dict[str, Any]:
        pit_entity_id = TEAM_TTE_PIT_ENTITY_IDS.get(team_name, str(team_id) if team_id else None)
        meta = {
            "team_id": team_id,
            "pit_entity_id": pit_entity_id,
            "team_name": team_name,
            "pit_found": False,
            "lambda_offense": None,
            "fallback_used": "missing_team_id" if not pit_entity_id else "snapshot_not_found",
            "requested_as_of_date": requested,
            "team_offense_as_of_date": None,
            "prior_baseline_as_of_date": None,
            "missing_inputs": [],
            "source_fingerprints": {},
        }
        if not pit_entity_id:
            return meta

        if requested_as_of_date:
            snapshot = snapshot_builder.build_team_snapshot(
                team_id=pit_entity_id,
                season=season,
                requested_as_of_date=requested_as_of_date,
                team_name=team_name,
            )
        else:
            snapshot = snapshot_builder.build_for_game(
                team_id=pit_entity_id,
                season=season,
                game_date=game_date,
                team_name=team_name,
            )
        adapted = adapter(snapshot)
        missing_inputs = adapted.get("provenance", {}).get("missing_inputs", [])
        meta.update(
            {
                "pit_found": adapted.get("lambda_offense") is not None,
                "lambda_offense": adapted.get("lambda_offense"),
                "fallback_used": adapted.get("fallback_used"),
                "requested_as_of_date": snapshot.get("requested_as_of_date"),
                "team_offense_as_of_date": snapshot.get("team_offense_as_of_date"),
                "prior_baseline_as_of_date": snapshot.get("prior_baseline_as_of_date"),
                "team_rolling_found": snapshot.get("team_rolling_found"),
                "prior_baseline_found": snapshot.get("prior_baseline_found"),
                "blend_current_weight": adapted.get("blend_current_weight"),
                "blend_prior_weight": adapted.get("blend_prior_weight"),
                "sample_size_status": adapted.get("sample_size_status"),
                "missing_inputs": missing_inputs,
                "source_fingerprints": snapshot.get("source_fingerprints", {}),
            }
        )
        if adapted.get("lambda_offense") is not None:
            game_data[f"{side}_team"]["tte_pit_lambda_offense"] = adapted["lambda_offense"]
            game_data[f"{side}_team"]["tte_pit_snapshot"] = adapted
        return meta

    home = _apply(
        "home",
        game_data.get("home_team_id"),
        game_data.get("home_team", {}).get("name", ""),
    )
    away = _apply(
        "away",
        game_data.get("away_team_id"),
        game_data.get("away_team", {}).get("name", ""),
    )
    metadata = {
        "requested_as_of_date": requested,
        "home": home,
        "away": away,
        "home_team_tte_pit_found": home["pit_found"],
        "away_team_tte_pit_found": away["pit_found"],
        "home_lambda_offense": home["lambda_offense"],
        "away_lambda_offense": away["lambda_offense"],
        "home_fallback_used": home["fallback_used"],
        "away_fallback_used": away["fallback_used"],
    }
    game_data["experimental_team_tte_pit"] = metadata
    return metadata


def apply_experimental_defense_pit_mode(
    *,
    game_data: Dict[str, Any],
    season: int,
    game_date: str,
    snapshot_builder: Any,
    adapter: Any,
    requested_as_of_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach Defense PIT provenance and block legacy DER/BIP adjustment."""
    requested = requested_as_of_date or _defense_pit_cutoff_for_row(
        {"game_date": game_date}
    )

    # Defense PIT is authoritative in this opt-in mode. Full-season DER/BIP is
    # removed even when build_game_data was called by an external test/caller.
    game_data["defense_home"] = {}
    game_data["defense_away"] = {}

    def _apply(side: str, team_name: str, team_id: Optional[int]) -> Dict[str, Any]:
        entity_id = TEAM_TTE_PIT_ENTITY_IDS.get(team_name)
        mapping_failure = entity_id is None
        if entity_id is None:
            snapshot = {
                "found": False,
                "team_id": team_id,
                "season": season,
                "requested_as_of_date": requested,
                "current_defense_found": False,
                "prior_baseline_found": False,
                "source_fingerprints": {},
            }
        else:
            snapshot = snapshot_builder.build_snapshot(
                team_id=entity_id,
                season=season,
                requested_as_of_date=requested,
            )
        adapted = adapter(snapshot)
        provenance = adapted.get("provenance", {})
        return {
            "side": side,
            "team_id": team_id,
            "pit_entity_id": entity_id,
            "team_name": team_name,
            "mapping_failure": mapping_failure,
            "requested_as_of_date": requested,
            "provenance_source": adapted.get(
                "provenance_source", "neutral_defense_adjustment"
            ),
            "neutral_fallback": (
                adapted.get("provenance_source") == "neutral_defense_adjustment"
            ),
            "fallback_used": adapted.get("fallback_used"),
            "current_defense_found": bool(adapted.get("current_defense_found")),
            "prior_baseline_found": bool(adapted.get("prior_baseline_found")),
            "current_as_of_date": provenance.get("current_as_of_date"),
            "prior_baseline_as_of_date": provenance.get("prior_baseline_as_of_date"),
            "source_window_start_date": adapted.get("source_window_start_date"),
            "source_window_end_date": adapted.get("source_window_end_date"),
            "prior_source_window_start_date": adapted.get(
                "prior_source_window_start_date"
            ),
            "prior_source_window_end_date": adapted.get("prior_source_window_end_date"),
            "bip_count": adapted.get("bip_count"),
            "xba_bip_count": adapted.get("xba_bip_count"),
            "sample_size_status": adapted.get("sample_size_status"),
            "contact_adjusted_defense_proxy": adapted.get(
                "contact_adjusted_defense_proxy", 0.0
            ),
            "applied_multiplier": adapted.get("defense_multiplier", 1.0),
            "current_weight": adapted.get("current_weight"),
            "prior_weight": adapted.get("prior_weight"),
            "source_fingerprints": provenance.get("source_fingerprints", {}),
        }

    home = _apply(
        "home",
        game_data.get("home_team", {}).get("name", ""),
        game_data.get("home_team_id"),
    )
    away = _apply(
        "away",
        game_data.get("away_team", {}).get("name", ""),
        game_data.get("away_team_id"),
    )
    metadata = {
        "requested_as_of_date": requested,
        "home": home,
        "away": away,
        "legacy_full_season_der_bip_blocked": True,
        "home_defense_applies_to": "away_lambda",
        "away_defense_applies_to": "home_lambda",
    }
    game_data["experimental_defense_pit"] = metadata
    return metadata


# ── Bullpen PIT: PIT-native league averages (Savant, reliever-only) ──────────
# _PIT_LG_K_BB and _PIT_LG_BARREL_PC re-centered 2026-07-11 against real
# tbf-weighted league means from savant.team_bullpen.rolling (2024-2025):
# K%-BB% empirical ~0.135-0.145 (was hardcoded 0.162, an uncited guess never
# validated against real PIT data); barrel/contact empirical ~0.074-0.081
# (was hardcoded 0.085). _PIT_LG_XWOBA_AG=0.312 verified still correct for
# relievers specifically (empirical 0.310-0.311) — left unchanged.
_PIT_LG_XWOBA_AG    = 0.312   # xwOBA against, pitcher side
_PIT_LG_K_BB        = 0.140   # K% − BB% for team relievers
_PIT_LG_BARREL_PC   = 0.074   # barrel per contact for relievers
_K_PIT_XWOBA        = 200     # Bayesian stabilisation constant (BF)
_K_PIT_K_BB         = 180
_PIT_INNINGS_WEIGHT = 0.40    # fixed (no starter avg_ips; ≈ 5.4 avg IPS)
_PIT_PITCHES_PER_IN = 15.0    # pitch-to-inning conversion for workload
# Re-centered 2026-07-11: real tbf-weighted mean pitches_last_3_days implies
# ~10.6 IP-equivalent bullpen usage, not 9.0 — the old value made every
# bullpen look chronically overworked (+2.4% mean workload_mult for no reason).
_PIT_NORMAL_IP_3D   = 10.6    # expected bullpen load over 3 days


def _regress_bp(observed: float, mean: float, n: float, k: float) -> float:
    """Bayesian regression toward mean. At n=0 → mean; at n=k → 50/50."""
    if n <= 0:
        return mean
    return (observed * n + mean * k) / (n + k)


def calculate_pit_bullpen_adjustment(
    adapted: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute a PIT-native Bullpen lambda multiplier from xwOBA, K-BB%, and barrel rate.

    Signals (no ERA available in PIT data):
      xwOBA against  0.50 — removes BABIP luck, most predictive
      K% − BB%       0.35 — command/whiff stability for relievers
      barrel/contact 0.15 — hard-contact indicator

    All metrics are Bayesian-regressed toward league averages using
    relief_batters_faced as the sample-size anchor.
    Workload converts pitches_last_3_days to approximate innings.
    """
    if adapted.get("neutral_fallback"):
        return {
            "applied_multiplier": 1.0,
            "adjustment_formula": "neutral_only_no_data",
            "mapping_status": "neutral",
            "provenance_source": adapted.get(
                "provenance_source", "neutral_bullpen_adjustment"
            ),
            "quality_metrics_used": adapted.get("quality_metrics", {}),
            "workload_facts_used": adapted.get("workload_facts", {}),
            "sample_size_status": adapted.get("sample_size_status"),
            "source_fingerprints": adapted.get("source_fingerprints", {}),
        }

    quality  = adapted.get("quality_metrics") or {}
    workload = adapted.get("workload_facts") or {}
    tbf      = float(quality.get("relief_batters_faced") or 0.0)

    def _to_f(v: Any) -> Optional[float]:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # xwOBA factor (higher xwOBA against → more runs → factor > 1)
    xwoba_raw = _to_f(quality.get("xwoba_against"))
    if xwoba_raw is not None:
        xwoba_reg    = _regress_bp(xwoba_raw, _PIT_LG_XWOBA_AG, tbf, _K_PIT_XWOBA)
        xwoba_factor = xwoba_reg / _PIT_LG_XWOBA_AG
    else:
        xwoba_factor = 1.0
        xwoba_reg    = _PIT_LG_XWOBA_AG

    # K%-BB% factor (higher K-BB → better command → fewer runs → factor < 1)
    k_bb_raw = _to_f(quality.get("k_minus_bb_pct"))
    if k_bb_raw is not None:
        k_bb_reg    = _regress_bp(k_bb_raw, _PIT_LG_K_BB, tbf, _K_PIT_K_BB)
        delta_k_bb  = k_bb_reg - _PIT_LG_K_BB
        k_bb_factor = max(0.85, min(1.15, 1.0 - delta_k_bb * 1.5))
    else:
        k_bb_factor = 1.0
        k_bb_reg    = _PIT_LG_K_BB

    # Barrel-per-contact factor (higher → more hard contact → factor > 1)
    barrel_raw = _to_f(quality.get("barrel_per_contact"))
    if barrel_raw is not None:
        barrel_reg    = _regress_bp(barrel_raw, _PIT_LG_BARREL_PC, tbf, _K_PIT_XWOBA)
        barrel_factor = barrel_reg / _PIT_LG_BARREL_PC
    else:
        barrel_factor = 1.0
        barrel_reg    = _PIT_LG_BARREL_PC

    # 3-signal composite (no ERA in PIT data)
    quality_raw  = (
        xwoba_factor  * 0.50
        + k_bb_factor   * 0.35
        + barrel_factor * 0.15
    )
    quality_mult = max(BULLPEN_CLAMP_LOW, min(BULLPEN_CLAMP_HIGH, quality_raw))

    # Workload: convert pitches → approx innings, same fatigue curve as legacy
    pitches_3d   = float(workload.get("pitches_last_3_days") or 0.0)
    ip_3d_approx = pitches_3d / _PIT_PITCHES_PER_IN if pitches_3d > 0 else _PIT_NORMAL_IP_3D
    delta_ip     = ip_3d_approx - _PIT_NORMAL_IP_3D
    if delta_ip > 0:
        workload_mult = min(1.0 + delta_ip * 0.012, 1.10)
    else:
        workload_mult = max(1.0 + delta_ip * 0.005, 0.97)

    # Consecutive-days fatigue bump removed 2026-07-11: mean consecutive_days
    # for a TEAM bullpen (not one reliever) was ~5.2, so the >2 trigger fired
    # on 73% of snapshots and saturated the 1.10 cap on 23% of them — a
    # near-constant +2.7% inflation, not a differentiating signal. No
    # equivalent term exists in the legacy (non-PIT) bullpen_engine.py.
    consecutive = float(workload.get("consecutive_days") or 0.0)

    raw_mult   = quality_mult * workload_mult

    # Fixed innings weighting (no starter avg_ips available in PIT mode)
    total_mult = 1.0 + _PIT_INNINGS_WEIGHT * (raw_mult - 1.0)
    total_mult = max(BULLPEN_CLAMP_LOW, min(BULLPEN_CLAMP_HIGH, total_mult))

    return {
        "applied_multiplier":   round(total_mult, 4),
        "adjustment_formula":   "pit_native_xwoba_kbb_barrel",
        "mapping_status":       "active",
        "xwoba_raw":            xwoba_raw,
        "xwoba_reg":            round(xwoba_reg, 4),
        "xwoba_factor":         round(xwoba_factor, 4),
        "k_bb_raw":             k_bb_raw,
        "k_bb_reg":             round(k_bb_reg, 4),
        "k_bb_factor":          round(k_bb_factor, 4),
        "barrel_raw":           barrel_raw,
        "barrel_reg":           round(barrel_reg, 4),
        "barrel_factor":        round(barrel_factor, 4),
        "quality_mult":         round(quality_mult, 4),
        "workload_mult":        round(workload_mult, 4),
        "ip_3d_approx":         round(ip_3d_approx, 2),
        "consecutive_days":     consecutive,
        "raw_mult":             round(raw_mult, 4),
        "innings_weight":       _PIT_INNINGS_WEIGHT,
        "tbf":                  tbf,
        "provenance_source":    adapted.get("provenance_source"),
        "quality_metrics_used": adapted.get("quality_metrics", {}),
        "workload_facts_used":  adapted.get("workload_facts", {}),
        "sample_size_status":   adapted.get("sample_size_status"),
        "source_fingerprints":  adapted.get("source_fingerprints", {}),
    }


def apply_experimental_bullpen_pit_mode(
    *,
    game_data: Dict[str, Any],
    season: int,
    game_date: str,
    snapshot_builder: Any,
    adapter: Any,
    requested_as_of_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach Bullpen PIT provenance and remove all legacy bullpen payloads."""
    requested = requested_as_of_date or _bullpen_pit_cutoff_for_row(
        {"game_date": game_date}
    )
    game_data["bullpen_home"] = {}
    game_data["bullpen_away"] = {}

    def _apply(side: str, team_name: str, team_id: Optional[int]) -> Dict[str, Any]:
        entity_id = TEAM_TTE_PIT_ENTITY_IDS.get(team_name)
        mapping_failure = entity_id is None
        if entity_id is None:
            snapshot = {
                "team_id": team_id,
                "season": season,
                "requested_as_of_date": requested,
                "current_bullpen_found": False,
                "prior_baseline_found": False,
                "current": {},
                "prior": {},
                "source_fingerprints": {},
            }
        else:
            snapshot = snapshot_builder.build_snapshot(
                team_id=entity_id,
                season=season,
                requested_as_of_date=requested,
            )
        adapted = adapter(snapshot)
        adjustment = calculate_pit_bullpen_adjustment(adapted)
        current = snapshot.get("current", {})
        prior = snapshot.get("prior", {})
        return {
            **adapted,
            **adjustment,
            "side": side,
            "team_id": team_id,
            "pit_entity_id": entity_id,
            "team_name": team_name,
            "mapping_failure": mapping_failure,
            "requested_as_of_date": requested,
            "current_as_of_date": snapshot.get("current_as_of_date"),
            "prior_baseline_as_of_date": snapshot.get(
                "prior_baseline_as_of_date"
            ),
            "current_starter_pitches_included": int(
                current.get("starter_pitches_included") or 0
            ),
            "prior_starter_pitches_included": int(
                prior.get("starter_pitches_included") or 0
            ),
            "legacy_bullpen_blocked": True,
        }

    home = _apply(
        "home",
        game_data.get("home_team", {}).get("name", ""),
        game_data.get("home_team_id"),
    )
    away = _apply(
        "away",
        game_data.get("away_team", {}).get("name", ""),
        game_data.get("away_team_id"),
    )
    metadata = {
        "requested_as_of_date": requested,
        "home": home,
        "away": away,
        "legacy_bullpen_blocked": True,
        "home_bullpen_applies_to": "away_lambda",
        "away_bullpen_applies_to": "home_lambda",
        "adjustment_policy": "pit_native_xwoba_kbb_barrel",
    }
    game_data["experimental_bullpen_pit"] = metadata
    return metadata


def team_tte_pit_skip_reason(metadata: Dict[str, Any]) -> Optional[str]:
    """Return strict Team/TTE PIT skip reason, or None when both sides are covered."""
    home_missing = metadata.get("home", {}).get("lambda_offense") is None
    away_missing = metadata.get("away", {}).get("lambda_offense") is None
    if home_missing and away_missing:
        return "missing_both_team_tte_pit"
    if home_missing:
        return "missing_home_team_tte_pit"
    if away_missing:
        return "missing_away_team_tte_pit"
    return None


def _update_team_tte_pit_usage(
    usage: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    skipped: bool,
    skip_reason: Optional[str],
    sample: Optional[Dict[str, Any]] = None,
) -> None:
    usage["games_attempted"] += 1
    home_found = bool(metadata.get("home", {}).get("pit_found"))
    away_found = bool(metadata.get("away", {}).get("pit_found"))
    if home_found:
        usage["home_team_tte_pit_found"] += 1
    else:
        usage["home_team_tte_pit_missing"] += 1
    if away_found:
        usage["away_team_tte_pit_found"] += 1
    else:
        usage["away_team_tte_pit_missing"] += 1

    if skipped:
        usage["games_skipped_missing_team_tte_pit"] += 1
        if skip_reason == "missing_both_team_tte_pit":
            usage["both_missing"] += 1
        elif skip_reason == "missing_home_team_tte_pit":
            usage["home_missing_only"] += 1
        elif skip_reason == "missing_away_team_tte_pit":
            usage["away_missing_only"] += 1
    else:
        usage["games_processed_with_both_team_tte_pit"] += 1

    if sample is not None and len(usage["samples"]) < 3:
        usage["samples"].append(sample)


def _team_tte_pit_usage_summary(
    usage: Dict[str, Any],
    *,
    games_processed: int,
    failures: int,
) -> Dict[str, Any]:
    return {
        "games_attempted": usage["games_attempted"],
        "games_processed": games_processed,
        "failures": failures,
        "games_processed_with_both_team_tte_pit": usage[
            "games_processed_with_both_team_tte_pit"
        ],
        "games_skipped_missing_team_tte_pit": usage[
            "games_skipped_missing_team_tte_pit"
        ],
        "home_missing": usage["home_team_tte_pit_missing"],
        "away_missing": usage["away_team_tte_pit_missing"],
        "both_missing": usage["both_missing"],
        "home_missing_only": usage["home_missing_only"],
        "away_missing_only": usage["away_missing_only"],
        "home_team_tte_pit_found": usage["home_team_tte_pit_found"],
        "away_team_tte_pit_found": usage["away_team_tte_pit_found"],
        "team_tte_pit_lambdas_used": (
            usage["home_team_tte_pit_found"] + usage["away_team_tte_pit_found"]
        ),
        "missing_team_tte_pit_lambdas": (
            usage["home_team_tte_pit_missing"] + usage["away_team_tte_pit_missing"]
        ),
        "sample_games": usage["samples"],
    }


def _update_defense_pit_usage(
    usage: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    game_date: str,
) -> None:
    usage["games_attempted"] += 1
    if metadata.get("legacy_full_season_der_bip_blocked"):
        usage["legacy_defense_calls_blocked"] += 1

    requested = _parse_utc(metadata.get("requested_as_of_date"))
    for side in ("home", "away"):
        item = metadata.get(side, {})
        source = item.get("provenance_source", "neutral_defense_adjustment")
        if source == "current_defense_pit":
            usage["current_defense_pit_used"] += 1
        elif source == "prior_season_defense_baseline":
            usage["prior_season_baseline_used"] += 1
        else:
            usage["neutral_defense_adjustment_used"] += 1

        if item.get("mapping_failure"):
            usage["mapping_failures"] += 1

        current_as_of = _parse_utc(item.get("current_as_of_date"))
        if current_as_of and requested and current_as_of > requested:
            usage["future_snapshot_violations"] += 1
        if current_as_of and current_as_of.date().isoformat() >= str(game_date)[:10]:
            usage["same_day_violations"] += 1

        fingerprints = item.get("source_fingerprints", {})
        if source == "current_defense_pit" and not fingerprints.get("current_defense_pit"):
            usage["missing_fingerprints"] += 1
        if (
            source == "prior_season_defense_baseline"
            and not fingerprints.get("prior_season_defense_baseline")
        ):
            usage["missing_fingerprints"] += 1
        if (
            source == "current_defense_pit"
            and float(item.get("prior_weight") or 0.0) > 0
            and item.get("prior_baseline_found")
            and not fingerprints.get("prior_season_defense_baseline")
        ):
            usage["missing_fingerprints"] += 1


def _append_defense_pit_sample(
    usage: Dict[str, Any],
    *,
    game_pk: int,
    game_date: str,
    home_team: str,
    away_team: str,
    metadata: Dict[str, Any],
) -> None:
    if len(usage["samples"]) >= 3:
        return
    usage["samples"].append(
        {
            "game_pk": game_pk,
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "requested_as_of_date": metadata.get("requested_as_of_date"),
            "home_defense_pit": metadata.get("home"),
            "away_defense_pit": metadata.get("away"),
            "legacy_full_season_der_bip_blocked": metadata.get(
                "legacy_full_season_der_bip_blocked"
            ),
        }
    )


def _defense_pit_usage_summary(
    usage: Dict[str, Any],
    *,
    games_processed: int,
    failures: int,
) -> Dict[str, Any]:
    return {
        "games_attempted": usage["games_attempted"],
        "games_processed": games_processed,
        "failures": failures,
        "current_defense_pit_used": usage["current_defense_pit_used"],
        "prior_season_baseline_used": usage["prior_season_baseline_used"],
        "neutral_defense_adjustment_used": usage["neutral_defense_adjustment_used"],
        "legacy_defense_calls_blocked": usage["legacy_defense_calls_blocked"],
        "future_snapshot_violations": usage["future_snapshot_violations"],
        "same_day_violations": usage["same_day_violations"],
        "missing_fingerprints": usage["missing_fingerprints"],
        "duplicate_pit_keys": usage["duplicate_pit_keys"],
        "mapping_failures": usage["mapping_failures"],
        "sample_games": usage["samples"],
    }


def _defense_pit_duplicate_key_count(cache_db: Path) -> int:
    with sqlite3.connect(cache_db) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT 1
                FROM pit_metric_cache
                GROUP BY namespace, entity_id, season, as_of_date, source
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()
    return int(row[0]) if row else 0


def _update_bullpen_pit_usage(
    usage: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    game_date: str,
) -> None:
    usage["games_attempted"] += 1
    if metadata.get("legacy_bullpen_blocked"):
        usage["legacy_bullpen_calls_blocked"] += 1
    requested = _parse_utc(metadata.get("requested_as_of_date"))
    for side in ("home", "away"):
        item = metadata.get(side, {})
        source = item.get("provenance_source", "neutral_bullpen_adjustment")
        if source == "current_bullpen_pit":
            usage["current_only_bullpen_pit_uses"] += 1
        elif source == "current_prior_bullpen_blend":
            usage["current_prior_blended_uses"] += 1
        elif source == "prior_season_bullpen_baseline":
            usage["prior_only_baseline_uses"] += 1
        else:
            usage["neutral_bullpen_uses"] += 1

        if not item.get("current_bullpen_found"):
            usage["missing_current_bullpen"] += 1
        if (
            item.get("current_bullpen_found")
            and int(item.get("current_bf") or 0) < 200
        ):
            usage["current_thin_bullpen"] += 1

        current_as_of = _parse_utc(item.get("current_as_of_date"))
        if current_as_of and requested and current_as_of > requested:
            usage["future_snapshot_violations"] += 1
        if current_as_of and current_as_of.date().isoformat() >= str(game_date)[:10]:
            usage["same_day_violations"] += 1

        fingerprints = item.get("source_fingerprints", {})
        required_fingerprints = {
            "current_bullpen_pit": ("current_bullpen_pit",),
            "current_prior_bullpen_blend": (
                "current_bullpen_pit",
                "prior_season_bullpen_baseline",
            ),
            "prior_season_bullpen_baseline": (
                "prior_season_bullpen_baseline",
            ),
        }.get(source, ())
        usage["missing_fingerprints"] += sum(
            not fingerprints.get(key) for key in required_fingerprints
        )
        if (
            int(item.get("current_starter_pitches_included") or 0) > 0
            or int(item.get("prior_starter_pitches_included") or 0) > 0
        ):
            usage["starter_contamination_violations"] += 1


def _append_bullpen_pit_sample(
    usage: Dict[str, Any],
    *,
    game_pk: int,
    game_date: str,
    home_team: str,
    away_team: str,
    metadata: Dict[str, Any],
) -> None:
    if len(usage["samples"]) >= 3:
        return
    usage["samples"].append(
        {
            "game_pk": game_pk,
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "requested_as_of_date": metadata.get("requested_as_of_date"),
            "home_bullpen_pit": metadata.get("home"),
            "away_bullpen_pit": metadata.get("away"),
            "legacy_bullpen_blocked": metadata.get("legacy_bullpen_blocked"),
        }
    )


def _bullpen_pit_usage_summary(
    usage: Dict[str, Any],
    *,
    games_processed: int,
    failures: int,
) -> Dict[str, Any]:
    return {
        "games_attempted": usage["games_attempted"],
        "games_processed": games_processed,
        "failures": failures,
        "current_only_bullpen_pit_uses": usage[
            "current_only_bullpen_pit_uses"
        ],
        "current_prior_blended_uses": usage["current_prior_blended_uses"],
        "prior_only_baseline_uses": usage["prior_only_baseline_uses"],
        "neutral_bullpen_uses": usage["neutral_bullpen_uses"],
        "missing_current_bullpen": usage["missing_current_bullpen"],
        "current_thin_bullpen": usage["current_thin_bullpen"],
        "legacy_bullpen_calls_blocked": usage[
            "legacy_bullpen_calls_blocked"
        ],
        "future_snapshot_violations": usage["future_snapshot_violations"],
        "same_day_violations": usage["same_day_violations"],
        "missing_fingerprints": usage["missing_fingerprints"],
        "duplicate_pit_keys": usage["duplicate_pit_keys"],
        "starter_contamination_violations": usage[
            "starter_contamination_violations"
        ],
        "adjustment_policy": "pit_native_xwoba_kbb_barrel",
        "sample_games": usage["samples"],
    }


def _bullpen_pit_duplicate_key_count(cache_db: Path) -> int:
    with sqlite3.connect(cache_db) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT 1
                FROM pit_metric_cache
                WHERE namespace IN (
                    'savant.bullpen.relief_appearance.daily',
                    'savant.team_bullpen.rolling',
                    'savant.team_bullpen.prior_baseline'
                )
                GROUP BY namespace, entity_id, season, as_of_date, source
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()
    return int(row[0]) if row else 0


def _parse_utc(value: Any) -> Optional[datetime]:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _apply_defense_stage(
    lh: float,
    la: float,
    game_data: Dict[str, Any],
    *,
    use_defense_pit: bool,
) -> Tuple[float, float, Dict[str, Any]]:
    """Apply either experimental Defense PIT or the unchanged legacy stage."""
    if use_defense_pit:
        metadata = game_data.get("experimental_defense_pit")
        if metadata is None:
            raise ValueError("Defense PIT metadata is required in experimental mode")
        home_multiplier = float(
            metadata.get("home", {}).get("applied_multiplier", 1.0)
        )
        away_multiplier = float(
            metadata.get("away", {}).get("applied_multiplier", 1.0)
        )
        # Away defense faces home batters; home defense faces away batters.
        return (
            lh * away_multiplier,
            la * home_multiplier,
            {
                "mode": "experimental_defense_pit",
                "legacy_defense_called": False,
                "home_multiplier_on_away_lambda": home_multiplier,
                "away_multiplier_on_home_lambda": away_multiplier,
            },
        )

    if game_data.get("defense_home") or game_data.get("defense_away"):
        lh_def, la_def, metadata = adjust_for_defense(lh, la, game_data)
        metadata["mode"] = "legacy_der_bip"
        metadata["legacy_defense_called"] = True
        return lh_def, la_def, metadata
    return lh, la, {"mode": "none", "legacy_defense_called": False}


def _apply_bullpen_stage(
    lh: float,
    la: float,
    game_data: Dict[str, Any],
    *,
    use_bullpen_pit: bool,
) -> Tuple[float, float, Dict[str, Any]]:
    """Apply either fetch-free Bullpen PIT metadata or the legacy engine."""
    if use_bullpen_pit:
        metadata = game_data.get("experimental_bullpen_pit")
        if metadata is None:
            raise ValueError("Bullpen PIT metadata is required in experimental mode")
        home_multiplier = float(
            metadata.get("home", {}).get("applied_multiplier", 1.0)
        )
        away_multiplier = float(
            metadata.get("away", {}).get("applied_multiplier", 1.0)
        )
        return (
            lh * away_multiplier,
            la * home_multiplier,
            {
                "mode": "experimental_bullpen_pit",
                "legacy_bullpen_called": False,
                "home_multiplier_on_away_lambda": home_multiplier,
                "away_multiplier_on_home_lambda": away_multiplier,
            },
        )

    if game_data.get("bullpen_home") or game_data.get("bullpen_away"):
        lh_bp, la_bp, metadata = adjust_for_bullpen(lh, la, game_data)
        metadata["mode"] = "legacy_bullpen"
        metadata["legacy_bullpen_called"] = True
        return lh_bp, la_bp, metadata
    return lh, la, {"mode": "none", "legacy_bullpen_called": False}


# ── pipeline runner ────────────────────────────────────────────────────────

def run_pipeline(
    game_data: Dict,
    lh: float,
    la: float,
    learning: LearningEngine,
    season: int,
    n_mc: int = N_MC,
    use_team_tte_pit: bool = False,
    team_tte_pit_snapshot_builder: Any = None,
    team_tte_pit_adapter: Any = None,
    use_defense_pit: bool = False,
    defense_pit_snapshot_builder: Any = None,
    defense_pit_adapter: Any = None,
    use_bullpen_pit: bool = False,
    bullpen_pit_snapshot_builder: Any = None,
    bullpen_pit_adapter: Any = None,
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
    team_tte_pit_meta = None
    defense_pit_meta = None
    bullpen_pit_meta = None
    if use_team_tte_pit:
        if team_tte_pit_snapshot_builder is None or team_tte_pit_adapter is None:
            raise ValueError("Team/TTE PIT mode requires snapshot builder and adapter")
        team_tte_pit_meta = game_data.get("experimental_team_tte_pit")
        if team_tte_pit_meta is None:
            team_tte_pit_meta = apply_experimental_team_tte_pit_mode(
                game_data=game_data,
                season=season,
                game_date=str(game_data.get("game_date", "")),
                snapshot_builder=team_tte_pit_snapshot_builder,
                adapter=team_tte_pit_adapter,
                requested_as_of_date=game_data.get("team_tte_pit_requested_as_of_date"),
            )
        skip_reason = team_tte_pit_skip_reason(team_tte_pit_meta)
        if skip_reason:
            raise ValueError(f"Team/TTE PIT strict coverage failed: {skip_reason}")
        if team_tte_pit_meta["home"]["lambda_offense"] is not None:
            lh = float(team_tte_pit_meta["home"]["lambda_offense"])
        if team_tte_pit_meta["away"]["lambda_offense"] is not None:
            la = float(team_tte_pit_meta["away"]["lambda_offense"])
        tte_active = True
    elif _TTE_AVAILABLE and htid and atid:
        try:
            lh, _ = _get_tte_lambda(htid, home_team, season)
            la, _ = _get_tte_lambda(atid, away_team, season)
            tte_active = True
        except Exception:
            pass  # TTE failed — lh/la remain from build_game_data

    if use_defense_pit:
        defense_pit_meta = game_data.get("experimental_defense_pit")
        if defense_pit_meta is None:
            if defense_pit_snapshot_builder is None or defense_pit_adapter is None:
                raise ValueError("Defense PIT mode requires snapshot builder and adapter")
            defense_pit_meta = apply_experimental_defense_pit_mode(
                game_data=game_data,
                season=season,
                game_date=str(game_data.get("game_date", "")),
                snapshot_builder=defense_pit_snapshot_builder,
                adapter=defense_pit_adapter,
                requested_as_of_date=game_data.get("defense_pit_requested_as_of_date"),
            )

    if use_bullpen_pit:
        bullpen_pit_meta = game_data.get("experimental_bullpen_pit")
        if bullpen_pit_meta is None:
            if bullpen_pit_snapshot_builder is None or bullpen_pit_adapter is None:
                raise ValueError("Bullpen PIT mode requires snapshot builder and adapter")
            bullpen_pit_meta = apply_experimental_bullpen_pit_mode(
                game_data=game_data,
                season=season,
                game_date=str(game_data.get("game_date", "")),
                snapshot_builder=bullpen_pit_snapshot_builder,
                adapter=bullpen_pit_adapter,
                requested_as_of_date=game_data.get("bullpen_pit_requested_as_of_date"),
            )

    # ── Kalman adjustment (walk-forward: only sees games prior to this one) ───
    # CHRON-002 (roadmap Step 2, Commit A): prediction_source='backtest' —
    # reads/writes the backtest state_source namespace of kalman_state,
    # never production's live Kalman state.
    lh = learning.get_kalman_lambda_adjustment(home_team, "offense_home", season, lh, prediction_source="backtest")
    la = learning.get_kalman_lambda_adjustment(away_team, "offense_away", season, la, prediction_source="backtest")

    # REVERTIDO: Kalman defense_home removed (see run_module.py comment + AUDIT_FINDINGS.md)

    # ── Learned pipeline weights ──────────────────────────────────────────────
    # CHRON-002: prediction_source='backtest'.
    _w = learning.get_pipeline_weights(season, prediction_source="backtest")

    # ── Team bias (LearningEngine, walk-forward: only sees games strictly ──────
    #    before this one — see compute_team_bias's docstring for the leak this
    #    before_date param closes, found+fixed 2026-07-08) ──────────────────────
    try:
        _game_month = int(str(game_data.get("game_date", ""))[5:7])
    except (ValueError, TypeError):
        _game_month = None
    _bias_before_date = str(game_data.get("game_date", "")) or None
    # Diagnostic only (2026-07-11, revised 2026-07-12) — see run_module.py's
    # matching comment: this is NOT the bias-learning denominator (an
    # earlier version of this fix used it as one and caused a measured
    # double-count with the downstream engines; see learning_engine.py's
    # compute_team_bias_kalman_adjusted docstring for the postmortem).
    _sf["l0_home_lambda"] = lh
    _sf["l0_away_lambda"] = la
    # CHRON-001 (audit_20260714/): prediction_source='backtest' — this walk-
    # forward query reads prior games' backtest_lambda_home/away (what THIS
    # backtest run itself already computed and wrote for those earlier
    # games, via update_game_outcomes() below), never a live game's real
    # prediction. Preserves the exact same self-consistent walk-forward
    # behavior as before this fix (each run rewrote the columns it reads
    # from as it went) — just via the backtest_* shadow columns instead of
    # clobbering the live ones.
    _home_bias = learning.compute_team_bias_kalman_adjusted(
        game_data.get("home_team", {}).get("name", ""), season, "offense_home",
        month=_game_month, before_date=_bias_before_date,
        prediction_source="backtest",
    )
    _away_bias = learning.compute_team_bias_kalman_adjusted(
        game_data.get("away_team", {}).get("name", ""), season, "offense_away",
        month=_game_month, before_date=_bias_before_date,
        prediction_source="backtest",
    )
    lh *= _home_bias
    la *= _away_bias
    _sf["bias_on_home_lambda"] = _home_bias
    _sf["bias_on_away_lambda"] = _away_bias

    # ── PASO 2: Park + Weather ────────────────────────────────────────────────
    _lh_pre, _la_pre = lh, la
    lh_park, la_park, _park_meta = adjust_for_park_and_weather(lh, la, game_data)
    _raw_h = lh_park / _lh_pre if _lh_pre else 1.0
    _raw_a = la_park / _la_pre if _la_pre else 1.0
    _w_park = _w.get("park", 1.0)
    lh = _lh_pre * (1.0 + _w_park * (_raw_h - 1.0))
    la = _la_pre * (1.0 + _w_park * (_raw_a - 1.0))
    # Key format: "{stage}_on_{role}_lambda" (renamed 2026-07-06 for a uniform,
    # unambiguous convention across all 6 stages — matches run_module.py exactly,
    # required since learning_engine.py's _gradient_step() reads whichever run
    # produced the game's stored stage_factors_json).
    _sf["park_on_home_lambda"] = _raw_h
    _sf["park_on_away_lambda"] = _raw_a
    # DEFERRED F7: weather_mult stage factor — reactivar post-Sprint 3
    # _sf["weather_mult"] = float(_park_meta.get("weather_mult", 1.0))

    # ── PASO 3: HFA (crowd + travel asymmetric) ───────────────────────────────
    _lh_pre, _la_pre = lh, la
    lh_hfa, la_hfa, _ = get_adjusted_lambdas(lh, la, game_data)
    _raw_h = lh_hfa / _lh_pre if _lh_pre else 1.0
    _raw_a = la_hfa / _la_pre if _la_pre else 1.0
    _w_hfa = _w.get("hfa", 1.0)
    lh = _lh_pre * (1.0 + _w_hfa * (_raw_h - 1.0))
    la = _la_pre * (1.0 + _w_hfa * (_raw_a - 1.0))
    _sf["hfa_on_home_lambda"] = _raw_h
    _sf["hfa_on_away_lambda"] = _raw_a

    # ── PASO 4: Defensive Efficiency (DEE) ───────────────────────────────────────
    # Stage factor captures DEE-only ratio (DER fielding signal, orthogonal to Pitcher Engine).
    _lh_pre, _la_pre = lh, la
    if use_defense_pit or game_data.get("defense_home") or game_data.get("defense_away"):
        lh_def, la_def, _defense_meta = _apply_defense_stage(
            lh,
            la,
            game_data,
            use_defense_pit=use_defense_pit,
        )
        _raw_h = lh_def / _lh_pre if _lh_pre else 1.0
        _raw_a = la_def / _la_pre if _la_pre else 1.0
        _w_def = _w.get("defense", 1.0)
        lh = _lh_pre * (1.0 + _w_def * (_raw_h - 1.0))
        la = _la_pre * (1.0 + _w_def * (_raw_a - 1.0))
        _sf["defense_on_home_lambda"] = _raw_h   # away defense multiplier on home runs
        _sf["defense_on_away_lambda"] = _raw_a   # home defense multiplier on away runs

    # ── PASO 5: Pitcher Engine ────────────────────────────────────────────────
    _lh_pre, _la_pre = lh, la
    lh_pit, la_pit, _ = adjust_for_pitchers(lh, la, game_data)
    _raw_h = lh_pit / _lh_pre if _lh_pre else 1.0
    _raw_a = la_pit / _la_pre if _la_pre else 1.0
    _w_pit = _w.get("pitcher", 1.0)
    lh = _lh_pre * (1.0 + _w_pit * (_raw_h - 1.0))
    la = _la_pre * (1.0 + _w_pit * (_raw_a - 1.0))
    _sf["pitcher_on_home_lambda"] = _raw_h
    _sf["pitcher_on_away_lambda"] = _raw_a

    # ── F5 lambda — snapshot post-pitcher, pre-bullpen ────────────────────────
    lh_f5 = round(lh * F5_SCALE, 3)
    la_f5 = round(la * F5_SCALE, 3)

    # ── PASO 6: Bullpen Engine ────────────────────────────────────────────────
    _lh_pre, _la_pre = lh, la
    if use_bullpen_pit or game_data.get("bullpen_home") or game_data.get("bullpen_away"):
        lh_bp, la_bp, _bullpen_meta = _apply_bullpen_stage(
            lh,
            la,
            game_data,
            use_bullpen_pit=use_bullpen_pit,
        )
        _raw_h = lh_bp / _lh_pre if _lh_pre else 1.0
        _raw_a = la_bp / _la_pre if _la_pre else 1.0
    else:
        _raw_h = _raw_a = 1.0
    _w_bp = _w.get("bullpen", 1.0)
    lh = _lh_pre * (1.0 + _w_bp * (_raw_h - 1.0))
    la = _la_pre * (1.0 + _w_bp * (_raw_a - 1.0))
    _sf["bullpen_on_home_lambda"] = _raw_h
    _sf["bullpen_on_away_lambda"] = _raw_a

    # ── PASO 7: Contextual Engine (rest / B2B / umpire) ──────────────────────
    _lh_pre, _la_pre = lh, la
    lh_ctx, la_ctx, _ = adjust_for_context(lh, la, game_data)
    _raw_h = lh_ctx / _lh_pre if _lh_pre else 1.0
    _raw_a = la_ctx / _la_pre if _la_pre else 1.0
    _w_ctx = _w.get("context", 1.0)
    lh = _lh_pre * (1.0 + _w_ctx * (_raw_h - 1.0))
    la = _la_pre * (1.0 + _w_ctx * (_raw_a - 1.0))
    _sf["context_on_home_lambda"] = _raw_h
    _sf["context_on_away_lambda"] = _raw_a

    # ── Sanity clamp ──────────────────────────────────────────────────────────
    lh = max(1.5, min(lh, 12.0))
    la = max(1.5, min(la, 12.0))

    # ── PASO 8: Monte Carlo ───────────────────────────────────────────────────
    # Deterministic per-game seed (added 2026-07-06): monte_carlo_advanced()
    # defaults to rng_seed=None (true system entropy), so two backtest runs on
    # identical code previously produced different Brier/accuracy numbers
    # purely from MC sampling noise — confirmed as the explanation for a
    # 0.66pp accuracy / 0.0003 Brier swing between two otherwise-identical
    # runs the same day. Seeding from game_pk keeps each game's own randomness
    # genuinely independent (no shared/correlated stream across games) while
    # making the whole backtest byte-reproducible run-to-run, so a Brier delta
    # after a real code change can be trusted instead of being noise-sized.
    _game_pk_for_seed = game_data.get("game_pk")
    _mc_seed = int(_game_pk_for_seed) % (2**32) if _game_pk_for_seed else None
    mc = monte_carlo_advanced(
        lh=lh, la=la, n_max=n_mc,
        block=min(10_000, n_mc),
        analyze_f5=False,
        lh_f5=lh_f5, la_f5=la_f5,
        rng_seed=_mc_seed,
    )

    # Capture raw MC probabilities BEFORE Platt — needed for clean Platt refitting.
    # Using post-Platt values as training targets creates a circular dependency.
    p_home_mc = mc["p_home"]
    p_away_mc = mc["p_away"]

    # ── Platt calibration (dynamic params from learning engine) ───────────────
    # Applied asymmetrically — same logic as run_module.py. Symmetric normalization
    # cancels the b intercept for neutral games, destroying the structural home
    # advantage that recalibrate_platt() learned.
    # CHRON-001 (audit_20260714/): prediction_source='backtest' — if this
    # triggers a refit (cache stale/absent), it must train on this run's own
    # backtest_p_home(_raw), never a live game's real prediction.
    _pa, _pb = learning.get_platt_params(season, prediction_source="backtest")
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
        "team_tte_pit": team_tte_pit_meta,
        "defense_pit": defense_pit_meta,
        "bullpen_pit": bullpen_pit_meta,
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
    """Persist this backtest run's recomputed λ/probabilities for game_pk.

    CHRON-001 fix (audit_20260714/08_chronology_audit.md, roadmap Step 1):
    this used to write directly to game_outcomes' live prediction columns
    (lambda_home, p_home, ...) via an unconditional UPDATE...WHERE game_pk=?
    with no guard — a routine backtest run touching a game that had a real
    live prediction on file (record_prediction()) silently destroyed it,
    with zero audit trail. Verified via direct DB read: 563 of 615
    season-2026 rows had already been overwritten this way by a single
    2026-06-28 batch run, none recoverable from any backup on disk (see
    audit_20260714/chron001_forensics_report.md).

    Now writes ONLY to the backtest_* shadow columns + backtest_run_at.
    The live prediction columns (lambda_home, lambda_away, p_home, p_away,
    p_home_raw, p_away_raw, stage_factors_json) and `source` are NEVER
    touched here, regardless of the row's provenance — that invariant is
    what tests/test_chron001_provenance.py locks in.
    """
    now = datetime.now(timezone.utc).isoformat()
    sf_json = json.dumps(stage_factors) if stage_factors else None
    conn.execute(
        """
        UPDATE game_outcomes
        SET backtest_lambda_home = ?, backtest_lambda_away = ?,
            backtest_p_home = ?, backtest_p_away = ?,
            backtest_p_home_raw = ?, backtest_p_away_raw = ?,
            backtest_stage_factors_json = ?,
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
    team_tte_pit_summary: Optional[Dict[str, Any]] = None,
    defense_pit_summary: Optional[Dict[str, Any]] = None,
    bullpen_pit_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Compute and print a full backtest summary; save JSON sidecar."""
    n = len(results)
    if n == 0:
        if (
            team_tte_pit_summary is not None
            or defense_pit_summary is not None
            or bullpen_pit_summary is not None
        ):
            report = {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "total_games": 0,
                "seasons": [],
            }
            if team_tte_pit_summary is not None:
                report["team_tte_pit"] = team_tte_pit_summary
            if defense_pit_summary is not None:
                report["defense_pit"] = defense_pit_summary
            if bullpen_pit_summary is not None:
                report["bullpen_pit"] = bullpen_pit_summary
            report_path = out_path / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            report_path.write_text(json.dumps(report, indent=2))
            print(f"\nNo scored results. Report saved → {report_path}")
            return
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

    team_tte_rows = [r for r in results if r.get("team_tte_pit")]

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
    if team_tte_pit_summary is not None:
        report["team_tte_pit"] = team_tte_pit_summary
    elif team_tte_rows:
        report["team_tte_pit"] = {
            "games_with_metadata": len(team_tte_rows),
            "home_found": sum(
                1 for r in team_tte_rows
                if r["team_tte_pit"].get("home", {}).get("pit_found")
            ),
            "away_found": sum(
                1 for r in team_tte_rows
                if r["team_tte_pit"].get("away", {}).get("pit_found")
            ),
            "home_missing": sum(
                1 for r in team_tte_rows
                if not r["team_tte_pit"].get("home", {}).get("pit_found")
            ),
            "away_missing": sum(
                1 for r in team_tte_rows
                if not r["team_tte_pit"].get("away", {}).get("pit_found")
            ),
            "sample_games": [
                {
                    "game_pk": r["game_pk"],
                    "season": r["season"],
                    "home": r["team_tte_pit"].get("home"),
                    "away": r["team_tte_pit"].get("away"),
                }
                for r in team_tte_rows[:3]
            ],
        }
    if defense_pit_summary is not None:
        report["defense_pit"] = defense_pit_summary
    if bullpen_pit_summary is not None:
        report["bullpen_pit"] = bullpen_pit_summary

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


# ── ml_state advisory lock (2026-07-11, audit finding) ──────────────────────
# A duplicate backtest process launched this session was killed within
# seconds but still executed the reset-Platt-to-identity block against the
# live predictions_history.db before dying — the real run that started
# moments later silently warm-started from the corrupted (identity) state,
# running an entire multi-thousand-game backtest with Platt calibration
# disabled and no error or warning anywhere. Nothing previously stopped a
# second process from doing the same thing at any time. This is a plain
# file-based advisory lock (not OS-level flock — good enough for "don't
# launch two of these by mistake", not meant to survive adversarial use).
_ML_STATE_LOCK_PATH = DATA_DIR / "backtest_ml_state.lock"
_ML_STATE_LOCK_STALE_HOURS = 4.0   # longest real run this session was ~15 min


def _acquire_ml_state_lock(lock_path: Path = _ML_STATE_LOCK_PATH) -> None:
    if lock_path.exists():
        try:
            info = json.loads(lock_path.read_text())
            age_hours = (time.time() - float(info.get("started_at_epoch", 0))) / 3600.0
            holder_pid = info.get("pid", "?")
        except (json.JSONDecodeError, OSError, ValueError):
            age_hours = 0.0
            holder_pid = "?"
        if age_hours < _ML_STATE_LOCK_STALE_HOURS:
            raise RuntimeError(
                f"Another backtest process appears to hold the ml_state lock "
                f"({lock_path}, PID {holder_pid}, started {age_hours:.2f}h ago). "
                f"Two processes resetting/writing shared Kalman/Platt/pipeline-"
                f"weight state at once is exactly how a prior session silently "
                f"corrupted a full backtest run's calibration. If that process "
                f"is confirmed dead (not just slow), delete the lock file "
                f"manually and re-run — do not just retry, which would launch "
                f"a second concurrent writer."
            )
        logging.getLogger(__name__).warning(
            "Stale ml_state lock at %s (%.2fh old, PID %s) — assuming that "
            "process died and proceeding. If it's actually still running, "
            "this run's results may be corrupted.",
            lock_path, age_hours, holder_pid,
        )
    lock_path.write_text(json.dumps({
        "pid": os.getpid(),
        "started_at_epoch": time.time(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }))
    atexit.register(_release_ml_state_lock, lock_path)


def _release_ml_state_lock(lock_path: Path = _ML_STATE_LOCK_PATH) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def _assert_platt_not_silently_identity(
    learning: "LearningEngine", seasons: List[int], min_games: int = 200,
) -> None:
    """End-of-run sanity check: a season with enough graded games should not
    have ended up at identity Platt (a=1.0, b=0.0) unless it genuinely has
    no usable prior-season warm-start (only expected for the *first* season
    of a run with no season-1 data in the DB, e.g. season 2024 today). Loudly
    warns rather than silently reporting a contaminated-calibration Brier as
    if it were a real number — this is the exact failure mode this session's
    Platt-corruption incident produced with zero errors or warnings."""
    log_ = logging.getLogger(__name__)
    for season in sorted(seasons):
        # CHRON-002 (roadmap Step 2, Commit A): this sanity check is
        # backtest-exclusive (only ever called from the end-of-run step
        # below) — reads the 'backtest' state_source namespace, i.e. the
        # params THIS run itself just fitted, not production's live cache.
        params = learning.load_state("platt_params", "calibration", season, prediction_source="backtest")
        if not params:
            continue
        is_identity = (
            abs(float(params.get("a", 1.0)) - 1.0) < 1e-6
            and abs(float(params.get("b", 0.0))) < 1e-6
        )
        if is_identity:
            log_.warning(
                "  ⚠️  Season %d ended this run at IDENTITY Platt (a=1.0, b=0.0). "
                "Expected only if this is genuinely the first season in the DB "
                "with no prior-season data to warm-start from. If a prior "
                "season's data exists, this likely means ml_state was reset "
                "by a concurrent process during this run — do not trust this "
                "run's Brier/accuracy for season %d without investigating.",
                season, season,
            )


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MLB pipeline backtest + retrain")
    parser.add_argument("--limit",       type=int,   default=0,
                        help="Process only the first N games (0 = all)")
    parser.add_argument("--season", "--seasons", type=str, default="",
                        help="Restrict to one or more seasons, comma-separated "
                             "(e.g. --season 2024,2025). Empty = all seasons.")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip pipeline; just re-generate report from current DB state")
    parser.add_argument("--prefetch",    action="store_true",
                        help="Pre-warm team and starter caches, then exit")
    parser.add_argument("--no-cache",    action="store_true",
                        help="Ignore cached starter lookups (re-fetch everything)")
    parser.add_argument("--workers",     type=int,   default=1,
                        help="Parallel starter-fetch workers (default 1 = sequential)")
    parser.add_argument("--db-path", type=Path, default=DB_PATH,
                        help="SQLite game_outcomes DB path (default: production history DB)")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                        help="Backtest cache directory (default: .cache/backtest)")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR,
                        help="Directory for generated backtest reports")
    parser.add_argument("--n-mc", type=int, default=N_MC,
                        help="Monte Carlo simulations per game")
    parser.add_argument("--experimental-pitcher-pit-mode", action="store_true",
                        help="Use isolated daily PIT pitcher snapshots when available")
    parser.add_argument("--pitcher-pit-cache-db", type=Path, default=None,
                        help="PIT cache DB for --experimental-pitcher-pit-mode")
    parser.add_argument("--use-team-tte-pit", action="store_true",
                        help="Use isolated Team/TTE PIT offensive lambdas when available")
    parser.add_argument("--team-tte-pit-cache-db", type=Path, default=None,
                        help="PIT cache DB for --use-team-tte-pit")
    parser.add_argument("--use-defense-pit", action="store_true",
                        help="Use isolated contact-adjusted Team Defense PIT snapshots")
    parser.add_argument("--defense-pit-cache-db", type=Path, default=None,
                        help="PIT cache DB for --use-defense-pit")
    parser.add_argument("--use-bullpen-pit", action="store_true",
                        help="Use isolated relief-only Team Bullpen PIT snapshots")
    parser.add_argument("--bullpen-pit-cache-db", type=Path, default=None,
                        help="PIT cache DB for --use-bullpen-pit")
    parser.add_argument("--technical-pit-coverage-only", action="store_true",
                        help="Trace PIT coverage/provenance without scoring predictions")
    parser.add_argument("--use-full-pit", action="store_true",
                        help="Convenience flag: enables all 4 PIT modes "
                             "(pitcher, TTE, defense, bullpen) at once, so a "
                             "clean backtest run can't accidentally leave one "
                             "flag off and silently reintroduce a look-ahead leak")
    args = parser.parse_args()

    if args.use_full_pit:
        args.experimental_pitcher_pit_mode = True
        args.use_team_tte_pit = True
        args.use_defense_pit = True
        args.use_bullpen_pit = True

    if args.experimental_pitcher_pit_mode and not args.pitcher_pit_cache_db:
        parser.error("--pitcher-pit-cache-db is required with --experimental-pitcher-pit-mode")
    if args.use_team_tte_pit and not args.team_tte_pit_cache_db:
        parser.error("--team-tte-pit-cache-db is required with --use-team-tte-pit")
    if args.use_defense_pit and not args.defense_pit_cache_db:
        parser.error("--defense-pit-cache-db is required with --use-defense-pit")
    if args.use_bullpen_pit and not args.bullpen_pit_cache_db:
        parser.error("--bullpen-pit-cache-db is required with --use-bullpen-pit")
    if args.technical_pit_coverage_only and not (
        args.experimental_pitcher_pit_mode
        or args.use_team_tte_pit
        or args.use_defense_pit
        or args.use_bullpen_pit
    ):
        parser.error("--technical-pit-coverage-only requires at least one PIT mode")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    # ── load game_outcomes rows ─────────────────────────────────────────────
    conn = get_conn(args.db_path)
    _add_backtest_col(conn)

    where = "WHERE actual_home_runs IS NOT NULL"
    seasons_filter: List[int] = []
    if args.season:
        seasons_filter = [int(s.strip()) for s in args.season.split(",") if s.strip()]
        where += f" AND season IN ({','.join(str(s) for s in seasons_filter)})"
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
    cache = DiskCache(args.cache_dir)
    learning = LearningEngine(db_path=args.db_path)

    pit_snapshot_builder = None
    pit_snapshot_adapter = None
    if args.experimental_pitcher_pit_mode:
        from modules.baseball_module.advanced_pit_enrichment import (
            AdvancedPitcherDailySnapshotBuilder,
            adapt_unified_pitcher_snapshot,
        )

        pit_snapshot_builder = AdvancedPitcherDailySnapshotBuilder(
            cache_db=args.pitcher_pit_cache_db
        )
        pit_snapshot_adapter = adapt_unified_pitcher_snapshot
        log.info(
            "Experimental pitcher PIT mode ENABLED | pit_cache_db=%s",
            args.pitcher_pit_cache_db,
        )

    team_tte_pit_snapshot_builder = None
    team_tte_pit_adapter = None
    if args.use_team_tte_pit:
        from modules.baseball_module.advanced_pit_enrichment import (
            TTEDailySnapshotBuilder,
            adapt_tte_pit_snapshot_to_lambda,
        )

        team_tte_pit_snapshot_builder = TTEDailySnapshotBuilder(
            cache_db=args.team_tte_pit_cache_db
        )
        team_tte_pit_adapter = adapt_tte_pit_snapshot_to_lambda
        log.info(
            "Team/TTE PIT mode ENABLED | pit_cache_db=%s",
            args.team_tte_pit_cache_db,
        )

    defense_pit_snapshot_builder = None
    defense_pit_adapter = None
    if args.use_defense_pit:
        from modules.baseball_module.advanced_pit_enrichment import (
            TeamDefenseDailySnapshotBuilder,
            adapt_defense_pit_snapshot,
        )

        defense_pit_snapshot_builder = TeamDefenseDailySnapshotBuilder(
            cache_db=args.defense_pit_cache_db
        )
        defense_pit_adapter = adapt_defense_pit_snapshot
        log.info(
            "Defense PIT mode ENABLED | pit_cache_db=%s",
            args.defense_pit_cache_db,
        )

    bullpen_pit_snapshot_builder = None
    bullpen_pit_adapter = None
    if args.use_bullpen_pit:
        from modules.baseball_module.advanced_pit_enrichment import (
            TeamBullpenDailySnapshotBuilder,
            adapt_bullpen_pit_snapshot,
        )

        bullpen_pit_snapshot_builder = TeamBullpenDailySnapshotBuilder(
            cache_db=args.bullpen_pit_cache_db
        )
        bullpen_pit_adapter = adapt_bullpen_pit_snapshot
        log.info(
            "Bullpen PIT mode ENABLED | pit_cache_db=%s",
            args.bullpen_pit_cache_db,
        )

    # ── load Savant + FanGraphs data per season ──────────────────────────────
    _enrich_cache_dir = ROOT / ".cache"
    savant_by_season: Dict[int, Dict[int, Dict]] = {}
    fg_by_season: Dict[int, Dict[int, Dict]] = {}
    if _ENRICHMENT_AVAILABLE and not args.experimental_pitcher_pit_mode:
        _sv_fetcher = SavantFetcher(cache_dir=_enrich_cache_dir)
        _fg_fetcher = FanGraphsFetcher(cache_dir=_enrich_cache_dir)
        for _yr in seasons:
            log.info("Loading Savant + FanGraphs stats for season %d …", _yr)
            savant_by_season[_yr] = _sv_fetcher.get_all_pitcher_stats(_yr)
            fg_by_season[_yr]     = _fg_fetcher.get_all_pitcher_stats(_yr)
            log.info("  Savant: %d pitchers | FG: %d pitchers",
                     len(savant_by_season[_yr]), len(fg_by_season[_yr]))
    elif args.experimental_pitcher_pit_mode:
        log.info("Skipping full-season Savant/FanGraphs leaderboards in experimental PIT mode.")

    if args.no_cache:
        cache = DiskCache(args.cache_dir, ttl=1)  # effectively bypasses old entries

    # DEFERRED F7: weather fetcher built but not active.
    # Reactivar post-Sprint 3 cuando gradient descent funcione y pueda aprender
    # un peso separado para weather vs park_factor. Sin ese peso aprendido,
    # weather añade ~+0.00013 Brier y -0.8pp ROI (validado 2026-05-26).
    # _weather_fetcher = HistoricalWeatherFetcher(ROOT / ".cache" / "historical_weather_cache.json")
    # _weather_fetcher.prefetch_all(seasons=seasons)
    _weather_fetcher = None

    # ── prefetch mode ───────────────────────────────────────────────────────
    if args.prefetch:
        prefetch_team_stats(
            api,
            seasons,
            include_legacy_bullpen=not args.use_bullpen_pit,
        )
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
        generate_report(results, args.report_dir)
        conn.close()
        return

    # ── pre-warm team stats ─────────────────────────────────────────────────
    prefetch_team_stats(
        api,
        seasons,
        include_legacy_bullpen=not args.use_bullpen_pit,
    )

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
    _acquire_ml_state_lock()
    log.info("Resetting walk-forward learning state for seasons %s …", list(seasons))

    # Capture prior-season Platt params BEFORE reset so warm-start can use them.
    # Seasons processed in ascending order so each N seeds from N-1's fitted params.
    # CHRON-002 (roadmap Step 2, Commit A): all reads/writes below are
    # explicitly prediction_source='backtest' — this whole reset/warm-start
    # dance is backtest-to-backtest continuity (this run's season N
    # warm-starting from a PREVIOUS backtest run's season N-1 fit), never
    # touching or touched by production's live state.
    _prior_platt: Dict[int, Optional[Dict]] = {}
    for season in sorted(seasons):
        _prior_platt[season] = learning.load_state(
            "platt_params", "calibration", season - 1, prediction_source="backtest",
        )

    n_kal_deleted = learning.reset_kalman_for_seasons(list(seasons), prediction_source="backtest")
    n_wt_reset    = learning.reset_pipeline_weights(list(seasons), prediction_source="backtest")
    n_plt_deleted = learning.reset_platt_params(list(seasons), prediction_source="backtest")
    log.info(
        "  Kalman: %d rows deleted | weights: %d seasons reset | Platt: %d rows deleted",
        n_kal_deleted, n_wt_reset, n_plt_deleted,
    )

    # Warm-start each season's Platt with the prior season's fitted params.
    # Minimum 50 samples required (same threshold as recalibrate_platt).
    #
    # 2026-07-11 fix (audit finding: Platt warm-start hysteresis): when the
    # prior season is ALSO part of this same multi-season run, do NOT seed
    # it here from `_prior_platt` — that value is whatever a *previous,
    # separate* process run last wrote (identity if never run, or a
    # different pipeline/code version's fit otherwise), not this run's own
    # walk-forward result for that season. Seeding from it meant every
    # multi-season run's later seasons were calibrated against a
    # one-run-stale fit, so identical code run twice in a row didn't
    # reproduce the same numbers until the params reached a fixed point,
    # and a code change to season N's pipeline wasn't reflected in season
    # N+1's warm-start until the NEXT run. Such seasons are instead
    # warm-started mid-loop (see the main loop below), from a fresh
    # `recalibrate_platt()` call on THIS run's own just-completed season —
    # chronologically correct and self-consistent within one run.
    _PLATT_MIN_SAMPLES = 50
    _seasons_set = set(seasons)
    for season in sorted(seasons):
        if (season - 1) in _seasons_set:
            log.info(
                "  Platt season %d: prior season %d is in this same run — "
                "will warm-start mid-loop from its own fresh fit, not a stale prior run",
                season, season - 1,
            )
            continue
        prior = _prior_platt.get(season)
        if prior and prior.get("n", 0) >= _PLATT_MIN_SAMPLES:
            learning.save_state(
                "platt_params", "calibration",
                {"a": prior["a"], "b": prior["b"], "n": 0},
                sample_count=0,
                season=season,
                prediction_source="backtest",
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
    pit_usage = {
        "home_pitcher_pit_found": 0,
        "away_pitcher_pit_found": 0,
        "home_pitcher_pit_missing": 0,
        "away_pitcher_pit_missing": 0,
        "current_pit_snapshots_used": 0,
        "prior_season_baselines_used": 0,
        "league_average_safe_fallbacks_used": 0,
        "full_season_pitcher_fetches_avoided": bool(args.experimental_pitcher_pit_mode),
        "samples": [],
    }
    team_tte_pit_usage = {
        "games_attempted": 0,
        "games_processed_with_both_team_tte_pit": 0,
        "games_skipped_missing_team_tte_pit": 0,
        "home_team_tte_pit_found": 0,
        "away_team_tte_pit_found": 0,
        "home_team_tte_pit_missing": 0,
        "away_team_tte_pit_missing": 0,
        "both_missing": 0,
        "home_missing_only": 0,
        "away_missing_only": 0,
        "samples": [],
    }
    defense_pit_usage = {
        "games_attempted": 0,
        "current_defense_pit_used": 0,
        "prior_season_baseline_used": 0,
        "neutral_defense_adjustment_used": 0,
        "legacy_defense_calls_blocked": 0,
        "future_snapshot_violations": 0,
        "same_day_violations": 0,
        "missing_fingerprints": 0,
        "duplicate_pit_keys": (
            _defense_pit_duplicate_key_count(args.defense_pit_cache_db)
            if args.use_defense_pit
            else 0
        ),
        "mapping_failures": 0,
        "samples": [],
    }
    bullpen_pit_usage = {
        "games_attempted": 0,
        "current_only_bullpen_pit_uses": 0,
        "current_prior_blended_uses": 0,
        "prior_only_baseline_uses": 0,
        "neutral_bullpen_uses": 0,
        "missing_current_bullpen": 0,
        "current_thin_bullpen": 0,
        "legacy_bullpen_calls_blocked": 0,
        "future_snapshot_violations": 0,
        "same_day_violations": 0,
        "missing_fingerprints": 0,
        "duplicate_pit_keys": (
            _bullpen_pit_duplicate_key_count(args.bullpen_pit_cache_db)
            if args.use_bullpen_pit
            else 0
        ),
        "starter_contamination_violations": 0,
        "samples": [],
    }
    t0 = time.time()

    # F3: track last game date AND venue per team to detect meaningful B2B.
    # "Meaningful B2B" = played yesterday AND changed cities (overnight travel),
    # as opposed to within-series consecutive games in the same venue (no extra
    # fatigue beyond the normal 3-4 game road-trip rhythm).
    # Key: team name → (last_date "YYYY-MM-DD", last_venue str)
    _team_last_game: Dict[str, Tuple[str, str]] = {}

    # 2026-07-11 fix (Platt warm-start hysteresis, see the warm-start block
    # above): track season transitions so the season that just finished gets
    # refit from THIS run's own data before the next season's first game
    # reads its warm-start.
    _loop_season: Optional[int] = None

    for idx, row in enumerate(rows, 1):
        game_pk   = row["game_pk"]
        game_date = row["game_date"]
        season    = row["season"]
        home_name = row["home_team"]
        away_name = row["away_team"]
        home_won  = row["home_won"]

        if _loop_season is not None and season != _loop_season and _loop_season in _seasons_set:
            # CHRON-001 (audit_20260714/): prediction_source='backtest' —
            # refit against this run's own backtest_p_home(_raw), never a
            # live game's real prediction.
            a_fit, b_fit = learning.recalibrate_platt(_loop_season, prediction_source="backtest")
            log.info(
                "  Mid-run Platt refit (season boundary %d → %d): season %d a=%.4f b=%.4f",
                _loop_season, season, _loop_season, a_fit, b_fit,
            )
            if season in _seasons_set and season - 1 == _loop_season:
                learning.save_state(
                    "platt_params", "calibration",
                    {"a": a_fit, "b": b_fit, "n": 0},
                    sample_count=0,
                    season=season,
                    prediction_source="backtest",
                )
                log.info(
                    "  Platt warm-start season %d ← season %d (in-run, fresh): a=%.4f b=%.4f",
                    season, _loop_season, a_fit, b_fit,
                )
        _loop_season = season

        # F3: detect meaningful B2B — played yesterday AND changed cities.
        # Within-series games (same venue, consecutive days) are normal MLB
        # rhythm and don't trigger extra fatigue.  The penalty applies when a
        # team flew overnight to reach the current game.
        _today_str = game_date[:10]                          # "YYYY-MM-DD"
        _yesterday = (
            datetime.strptime(_today_str, "%Y-%m-%d").date()
            - timedelta(days=1)
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
                savant_stats=None if args.experimental_pitcher_pit_mode else savant_by_season.get(season),
                fg_stats=None if args.experimental_pitcher_pit_mode else fg_by_season.get(season),
                weather_fetcher=_weather_fetcher,
                pitcher_game_log_as_of_date=(
                    _experimental_pitcher_pit_cutoff_for_row(row)
                    if args.experimental_pitcher_pit_mode
                    else None
                ),
                use_pitcher_full_season_fallback=not args.experimental_pitcher_pit_mode,
                use_team_full_season_offense_base=not args.use_team_tte_pit,
                use_team_full_season_defense=not args.use_defense_pit,
                use_legacy_full_season_bullpen=not args.use_bullpen_pit,
                # Same season-aggregate get_team_pitching_stats() call that
                # feeds der/bip (gated above) also feeds team_era/team_whip/
                # runs_allowed_per_game — tied to the same flag so the two
                # can't drift out of sync and silently reopen the leak.
                use_team_full_season_pitching_base=not args.use_defense_pit,
            )
            if args.experimental_pitcher_pit_mode:
                pit_meta = apply_experimental_pitcher_pit_mode(
                    game_data=game_data,
                    home_pitcher_id=starters["home_pitcher_id"],
                    away_pitcher_id=starters["away_pitcher_id"],
                    season=season,
                    requested_as_of_date=_experimental_pitcher_pit_cutoff_for_row(row),
                    snapshot_builder=pit_snapshot_builder,
                    adapter=pit_snapshot_adapter,
                )
                if pit_meta["home"]["pit_found"]:
                    pit_usage["home_pitcher_pit_found"] += 1
                else:
                    pit_usage["home_pitcher_pit_missing"] += 1
                if pit_meta["away"]["pit_found"]:
                    pit_usage["away_pitcher_pit_found"] += 1
                else:
                    pit_usage["away_pitcher_pit_missing"] += 1
                for side in ("home", "away"):
                    provenance_source = pit_meta[side].get(
                        "provenance_source", "league_average_safe_fallback"
                    )
                    if provenance_source == "current_pit":
                        pit_usage["current_pit_snapshots_used"] += 1
                    elif provenance_source == "prior_season_baseline":
                        pit_usage["prior_season_baselines_used"] += 1
                    else:
                        pit_usage["league_average_safe_fallbacks_used"] += 1
                if len(pit_usage["samples"]) < 3:
                    pit_usage["samples"].append(
                        {
                            "game_pk": game_pk,
                            "game_date": game_date,
                            "home_team": home_name,
                            "away_team": away_name,
                            "home_pitcher_pit": pit_meta["home"],
                            "away_pitcher_pit": pit_meta["away"],
                        }
                    )
            defense_meta = None
            if args.use_defense_pit:
                game_data["defense_pit_requested_as_of_date"] = (
                    _defense_pit_cutoff_for_row(row)
                )
                defense_meta = apply_experimental_defense_pit_mode(
                    game_data=game_data,
                    season=season,
                    game_date=game_date,
                    requested_as_of_date=game_data["defense_pit_requested_as_of_date"],
                    snapshot_builder=defense_pit_snapshot_builder,
                    adapter=defense_pit_adapter,
                )
                _update_defense_pit_usage(
                    defense_pit_usage,
                    defense_meta,
                    game_date=game_date,
                )
            bullpen_meta = None
            if args.use_bullpen_pit:
                game_data["bullpen_pit_requested_as_of_date"] = (
                    _bullpen_pit_cutoff_for_row(row)
                )
                bullpen_meta = apply_experimental_bullpen_pit_mode(
                    game_data=game_data,
                    season=season,
                    game_date=game_date,
                    requested_as_of_date=game_data[
                        "bullpen_pit_requested_as_of_date"
                    ],
                    snapshot_builder=bullpen_pit_snapshot_builder,
                    adapter=bullpen_pit_adapter,
                )
                _update_bullpen_pit_usage(
                    bullpen_pit_usage,
                    bullpen_meta,
                    game_date=game_date,
                )
            if args.use_team_tte_pit:
                game_data["team_tte_pit_requested_as_of_date"] = _team_tte_pit_cutoff_for_row(row)
                team_meta = apply_experimental_team_tte_pit_mode(
                    game_data=game_data,
                    season=season,
                    game_date=game_date,
                    requested_as_of_date=game_data["team_tte_pit_requested_as_of_date"],
                    snapshot_builder=team_tte_pit_snapshot_builder,
                    adapter=team_tte_pit_adapter,
                )
                skip_reason = team_tte_pit_skip_reason(team_meta)
                sample = {
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "home_team": home_name,
                    "away_team": away_name,
                    "team_tte_pit_skip": bool(skip_reason),
                    "skip_reason": skip_reason,
                    "home_team_tte_pit": team_meta.get("home"),
                    "away_team_tte_pit": team_meta.get("away"),
                }
                _update_team_tte_pit_usage(
                    team_tte_pit_usage,
                    team_meta,
                    skipped=bool(skip_reason),
                    skip_reason=skip_reason,
                    sample=sample,
                )
                if skip_reason:
                    log.info(
                        "game_pk=%d SKIPPED Team/TTE PIT coverage: %s",
                        game_pk,
                        skip_reason,
                    )
                    continue
            if args.technical_pit_coverage_only:
                if args.use_defense_pit:
                    _apply_defense_stage(
                        1.0,
                        1.0,
                        game_data,
                        use_defense_pit=True,
                    )
                    _append_defense_pit_sample(
                        defense_pit_usage,
                        game_pk=game_pk,
                        game_date=game_date,
                        home_team=home_name,
                        away_team=away_name,
                        metadata=defense_meta or {},
                    )
                if args.use_bullpen_pit:
                    _apply_bullpen_stage(
                        1.0,
                        1.0,
                        game_data,
                        use_bullpen_pit=True,
                    )
                    _append_bullpen_pit_sample(
                        bullpen_pit_usage,
                        game_pk=game_pk,
                        game_date=game_date,
                        home_team=home_name,
                        away_team=away_name,
                        metadata=bullpen_meta or {},
                    )
                n_ok += 1
                continue
            # F3: inject B2B flags computed from schedule context above
            game_data["back_to_back_away"] = _b2b_away
            game_data["back_to_back_home"] = _b2b_home

            pred = run_pipeline(
                game_data,
                lh,
                la,
                learning,
                season,
                args.n_mc,
                use_team_tte_pit=args.use_team_tte_pit,
                team_tte_pit_snapshot_builder=team_tte_pit_snapshot_builder,
                team_tte_pit_adapter=team_tte_pit_adapter,
                use_defense_pit=args.use_defense_pit,
                defense_pit_snapshot_builder=defense_pit_snapshot_builder,
                defense_pit_adapter=defense_pit_adapter,
                use_bullpen_pit=args.use_bullpen_pit,
                bullpen_pit_snapshot_builder=bullpen_pit_snapshot_builder,
                bullpen_pit_adapter=bullpen_pit_adapter,
            )
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
            # untruncate_home_runs() corrects for the 2026-07-11 audit finding
            # (walk-off-truncated home runs biasing the "offense_home"/
            # "defense_away" learning targets — see learning_engine.py).
            # CHRON-002 (roadmap Step 2, Commit A): prediction_source='backtest'.
            learning.update_kalman(home_name, "offense_home", season, untruncate_home_runs(float(row["actual_home_runs"])), prediction_source="backtest")
            learning.update_kalman(away_name, "offense_away", season, float(row["actual_away_runs"]), prediction_source="backtest")
            learning.update_kalman(home_name, "defense_home", season, float(row["actual_away_runs"]), prediction_source="backtest")
            learning.update_kalman(away_name, "defense_away", season, untruncate_home_runs(float(row["actual_home_runs"])), prediction_source="backtest")

            # Sprint 3 fix: invoke gradient descent that was structurally disconnected
            # from the backtest loop. Without this call, pipeline weights never update
            # from 1.000 regardless of how many games are processed.
            # Diagnostic (scripts/diagnose_gradient.py) confirmed:
            #   - Gradients are real and non-zero (pitcher avg=0.026, context avg=0.056)
            #   - DB persistence works correctly
            #   - Only the invocation was missing
            # stage_factors_json is written by update_game_outcomes() above; lambda
            # fields are also set — so this call has all data it needs.
            # CHRON-001 (audit_20260714/): prediction_source='backtest' —
            # update_game_outcomes() (just above) now writes this game's λ/
            # stage_factors to the backtest_* shadow columns, not the live
            # ones, so this read must target the same columns it just wrote.
            learning._gradient_step(
                game_pk,
                int(row["actual_home_runs"]),
                int(row["actual_away_runs"]),
                season,
                prediction_source="backtest",
            )

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
                "team_tte_pit":    pred.get("team_tte_pit"),
                "defense_pit":     pred.get("defense_pit"),
                "bullpen_pit":     pred.get("bullpen_pit"),
            })
            if args.use_defense_pit:
                _append_defense_pit_sample(
                    defense_pit_usage,
                    game_pk=game_pk,
                    game_date=game_date,
                    home_team=home_name,
                    away_team=away_name,
                    metadata=pred.get("defense_pit") or {},
                )
            if args.use_bullpen_pit:
                _append_bullpen_pit_sample(
                    bullpen_pit_usage,
                    game_pk=game_pk,
                    game_date=game_date,
                    home_team=home_name,
                    away_team=away_name,
                    metadata=pred.get("bullpen_pit") or {},
                )
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
    if args.experimental_pitcher_pit_mode:
        print("\nEXPERIMENTAL PITCHER PIT SUMMARY")
        print(json.dumps(
            {
                "games_processed": n_ok,
                "failures": n_err,
                "pit_pitcher_snapshots_used": (
                    pit_usage["home_pitcher_pit_found"]
                    + pit_usage["away_pitcher_pit_found"]
                ),
                "missing_pit_pitcher_snapshots": (
                    pit_usage["home_pitcher_pit_missing"]
                    + pit_usage["away_pitcher_pit_missing"]
                ),
                "home_pitcher_pit_found": pit_usage["home_pitcher_pit_found"],
                "away_pitcher_pit_found": pit_usage["away_pitcher_pit_found"],
                "current_pit_snapshots_used": pit_usage["current_pit_snapshots_used"],
                "prior_season_baselines_used": pit_usage["prior_season_baselines_used"],
                "league_average_safe_fallbacks_used": pit_usage[
                    "league_average_safe_fallbacks_used"
                ],
                "full_season_pitcher_fetches_avoided": pit_usage[
                    "full_season_pitcher_fetches_avoided"
                ],
                "sample_games": pit_usage["samples"],
            },
            indent=2,
            sort_keys=True,
        ))
    if args.use_team_tte_pit:
        team_tte_summary = _team_tte_pit_usage_summary(
            team_tte_pit_usage,
            games_processed=n_ok,
            failures=n_err,
        )
        print("\nTEAM/TTE PIT SUMMARY")
        print(json.dumps(team_tte_summary, indent=2, sort_keys=True))
    defense_pit_summary = None
    if args.use_defense_pit:
        defense_pit_summary = _defense_pit_usage_summary(
            defense_pit_usage,
            games_processed=n_ok,
            failures=n_err,
        )
        print("\nDEFENSE PIT SUMMARY")
        print(json.dumps(defense_pit_summary, indent=2, sort_keys=True))
    bullpen_pit_summary = None
    if args.use_bullpen_pit:
        bullpen_pit_summary = _bullpen_pit_usage_summary(
            bullpen_pit_usage,
            games_processed=n_ok,
            failures=n_err,
        )
        print("\nBULLPEN PIT SUMMARY")
        print(json.dumps(bullpen_pit_summary, indent=2, sort_keys=True))

    if args.technical_pit_coverage_only:
        coverage_report = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "technical_pit_coverage_only": True,
            "games_loaded": len(rows),
            "games_processed": n_ok,
            "failures": n_err,
            "pitcher_pit": (
                {
                    "current_pit_snapshots_used": pit_usage[
                        "current_pit_snapshots_used"
                    ],
                    "prior_season_baselines_used": pit_usage[
                        "prior_season_baselines_used"
                    ],
                    "league_average_safe_fallbacks_used": pit_usage[
                        "league_average_safe_fallbacks_used"
                    ],
                    "full_season_pitcher_fetches_avoided": pit_usage[
                        "full_season_pitcher_fetches_avoided"
                    ],
                }
                if args.experimental_pitcher_pit_mode
                else None
            ),
            "team_tte_pit": (
                _team_tte_pit_usage_summary(
                    team_tte_pit_usage,
                    games_processed=n_ok,
                    failures=n_err,
                )
                if args.use_team_tte_pit
                else None
            ),
            "defense_pit": defense_pit_summary,
            "bullpen_pit": bullpen_pit_summary,
        }
        report_path = args.report_dir / (
            f"pit_coverage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        report_path.write_text(json.dumps(coverage_report, indent=2, sort_keys=True) + "\n")
        print(f"\nTechnical PIT coverage report saved → {report_path}")
        conn.close()
        return

    # ── step 4: recalibrate learning engine bias ───────────────────────────
    log.info("Recalibrating learning engine bias …")
    teams = conn.execute(
        "SELECT DISTINCT home_team AS t FROM game_outcomes "
        "UNION SELECT DISTINCT away_team FROM game_outcomes"
    ).fetchall()

    refreshed = 0
    for t in teams:
        for season in seasons:
            # CHRON-001 (audit_20260714/): prediction_source='backtest' —
            # final post-run refresh reads this run's own backtest_lambda_*/
            # backtest_stage_factors_json, never a live game's prediction.
            # Note (not fixed here, out of this step's scope — see
            # audit_20260714/14_remediation_roadmap.md): the ml_state
            # "team_bias" cache key itself is not source-scoped, so this
            # write still lands in the same cache slot a live call with
            # before_date=None would read — identical to the pre-fix
            # behavior (which also populated that cache from whatever this
            # backtest run had just computed), not a regression introduced
            # by this fix.
            bias = learning.compute_team_bias(t["t"], season, prediction_source="backtest")
            if bias != 1.0:
                refreshed += 1

    log.info("  %d teams | %d non-neutral biases written to ml_state", len(teams), refreshed)

    # ── step 4b: Platt recalibration per season ─────────────────────────────
    # Kalman states are now fully populated by the walk-forward loop above.
    log.info("Running Platt recalibration per season …")
    for season in seasons:
        try:
            # CHRON-001 (audit_20260714/): prediction_source='backtest'.
            a, b = learning.recalibrate_platt(season, prediction_source="backtest")
            log.info("  Season %d: Platt a=%.4f b=%.4f", season, a, b)
        except Exception as exc:
            log.warning("  Platt failed for season %d: %s", season, exc)

    _assert_platt_not_silently_identity(learning, list(seasons))

    # ── step 5: report ─────────────────────────────────────────────────────
    generate_report(
        results,
        args.report_dir,
        team_tte_pit_summary=(
            _team_tte_pit_usage_summary(
                team_tte_pit_usage,
                games_processed=n_ok,
                failures=n_err,
            )
            if args.use_team_tte_pit
            else None
        ),
        defense_pit_summary=defense_pit_summary,
        bullpen_pit_summary=bullpen_pit_summary,
    )
    conn.close()


if __name__ == "__main__":
    main()
