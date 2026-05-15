"""
odds_fetcher.py — The Odds API: single source of truth.

Public API
----------
get_odds_data()
    Returns normalized event list for the UI dropdown.
    Uses on-disk cache (TTL = ODDS_FILE_CACHE_TTL seconds).

get_best_odds_for_teams(home, away, sport)
    Searches ALL bookmakers in the cached raw events for the best
    available ML prices + Pinnacle reference for a specific matchup.
    No additional API call — reuses the same cache.

Internal flow
-------------
  _fetch_all_sports()  →  raw API events (full bookmakers list intact)
  _save_cache()        →  write raw events to .cache/odds_last.json
  _load_cache()        →  read raw events; None if missing / expired
  _normalize_event()   →  flatten one raw event to a UI-friendly dict
                          (best odds across all bookmakers, not just first)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from config import ODDS_FILE_CACHE_TTL

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
BASE_URL     = "https://api.the-odds-api.com/v4"
REGION       = "us"
MARKETS      = ["h2h", "totals", "spreads"]

SPORTS_KEYS = [
    "baseball_mlb",
    "basketball_nba",
    "basketball_ncaab",
    "basketball_euroleague",
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_usa_mls",
    "mma_mixed_martial_arts",
    "americanfootball_nfl",
    "americanfootball_ncaaf",
    "icehockey_nhl",
]

CACHE_DIR  = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "odds_last.json"
CACHE_TTL  = ODDS_FILE_CACHE_TTL   # seconds

_PINNACLE_KEY = "pinnaclesports"


# ── HTTP helper ───────────────────────────────────────────────────────────


def _get(url: str, max_retries: int = 3, delay: int = 2) -> Optional[Any]:
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", delay * 2))
                logger.warning("Rate limited — waiting %ds", wait)
                time.sleep(wait)
                continue
            if resp.status_code == 401:
                logger.error("Invalid or expired API key")
                return None
            logger.warning("Attempt %d/%d — HTTP %d", attempt + 1, max_retries, resp.status_code)
        except requests.exceptions.Timeout:
            logger.warning("Timeout on attempt %d/%d", attempt + 1, max_retries)
        except requests.exceptions.ConnectionError:
            logger.warning("Connection error on attempt %d/%d", attempt + 1, max_retries)
        except Exception as exc:
            logger.error("Unexpected error: %s", exc)
        if attempt < max_retries - 1:
            time.sleep(delay * (attempt + 1))
    return None


# ── Cache (stores RAW events — bookmakers list intact) ────────────────────


def _save_cache(raw_events: List[Dict]) -> None:
    try:
        CACHE_FILE.write_text(
            json.dumps(
                {
                    "timestamp":          time.time(),
                    "timestamp_readable": datetime.now().isoformat(),
                    "total_events":       len(raw_events),
                    "data":               raw_events,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info("Cache saved: %d raw events", len(raw_events))
    except Exception as exc:
        logger.warning("Error saving cache: %s", exc)


def _load_cache() -> Optional[List[Dict]]:
    if not CACHE_FILE.exists():
        return None
    try:
        content = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(content, list):
            # Legacy format written by old app.py — treat as expired
            return None
        ts = content.get("timestamp", 0)
        if time.time() - ts < CACHE_TTL:
            age = (time.time() - ts) / 60
            logger.info("Cache hit (%.1f min old)", age)
            return content.get("data", [])
        logger.info("Cache expired (>%d min)", CACHE_TTL // 60)
        return None
    except Exception as exc:
        logger.warning("Error reading cache: %s", exc)
        return None


# ── Fetch ─────────────────────────────────────────────────────────────────


def _fetch_all_sports() -> List[Dict]:
    """Fetch raw events for all configured sports. Returns list of raw API dicts."""
    all_events: List[Dict] = []

    for sport_key in SPORTS_KEYS:
        url = (
            f"{BASE_URL}/sports/{sport_key}/odds/"
            f"?apiKey={ODDS_API_KEY}"
            f"&regions={REGION}"
            f"&markets={','.join(MARKETS)}"
            f"&oddsFormat=decimal"
        )
        data = _get(url)
        if not data:
            continue
        all_events.extend(data)
        logger.info("%s: %d events fetched", sport_key, len(data))
        time.sleep(0.3)   # gentle pacing

    return all_events


def _get_raw_events() -> List[Dict]:
    """Return cached raw events, fetching fresh if expired or missing."""
    if not ODDS_API_KEY:
        logger.warning("ODDS_API_KEY not set — trying stale cache")
        if CACHE_FILE.exists():
            try:
                content = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                data = content if isinstance(content, list) else content.get("data", [])
                return data
            except Exception:
                pass
        return []

    cached = _load_cache()
    if cached is not None:
        return cached

    logger.info("Fetching fresh odds from The Odds API...")
    events = _fetch_all_sports()
    if events:
        _save_cache(events)
    return events


# ── Normalization (raw → flat UI dict) ────────────────────────────────────


def _normalize_event(event: Dict) -> Dict:
    """Flatten one raw API event to a simple dict for the UI.

    Uses the BEST available price across all bookmakers for home/away ML.
    Falls back to first-bookmaker totals for over/under display.
    """
    result: Dict[str, Any] = {
        "sport_key":    event.get("sport_key", ""),
        "sport_title":  event.get("sport_title", ""),
        "home_team":    event.get("home_team", ""),
        "away_team":    event.get("away_team", ""),
        "commence_time": event.get("commence_time", ""),
        "home_odds":    None,
        "draw_odds":    None,
        "away_odds":    None,
        "total_line":   None,
        "over_odds":    None,
        "under_odds":   None,
        "bookmaker":    None,
        "last_update":  datetime.now().isoformat(),
    }

    home = result["home_team"]
    away = result["away_team"]
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return result

    best_home: float = 0.0
    best_away: float = 0.0
    best_draw: Optional[float] = None

    for bm in bookmakers:
        if result["bookmaker"] is None:
            result["bookmaker"] = bm.get("title", "Unknown")
        for market in bm.get("markets", []):
            mkey = market.get("key", "")
            outcomes = market.get("outcomes", [])
            if mkey == "h2h":
                for o in outcomes:
                    name  = o.get("name", "").strip()
                    price = o.get("price") or 0.0
                    if name == home:
                        best_home = max(best_home, price)
                    elif name == away:
                        best_away = max(best_away, price)
                    elif name.lower() in ("draw", "empate", "tie"):
                        best_draw = max(best_draw or 0.0, price) or None
            elif mkey == "totals" and result["total_line"] is None and outcomes:
                result["total_line"] = outcomes[0].get("point")
                for o in outcomes:
                    n = o.get("name", "").lower()
                    if n == "over":
                        result["over_odds"] = o.get("price")
                    elif n == "under":
                        result["under_odds"] = o.get("price")

    result["home_odds"] = best_home or None
    result["away_odds"] = best_away or None
    result["draw_odds"] = best_draw
    return result


# ── Public API ────────────────────────────────────────────────────────────


def get_odds_data() -> List[Dict]:
    """Return normalized events for the UI.  Cached; no duplicate API calls."""
    raw = _get_raw_events()
    normalized = [_normalize_event(e) for e in raw]
    return [n for n in normalized if n["home_odds"] and n["away_odds"]]


def get_best_odds_for_teams(
    home_team: str,
    away_team: str,
    sport: str = "baseball_mlb",
    region: str = "us",   # kept for API compatibility, unused (cache is US-only)
) -> Dict:
    """Find best ML prices + Pinnacle reference for a specific matchup.

    Searches the cached raw events — no extra API call.
    Returns {} if the matchup is not found.
    """
    raw = _get_raw_events()

    for event in raw:
        if event.get("sport_key") != sport:
            continue

        g_home = event.get("home_team", "")
        g_away = event.get("away_team", "")

        home_match = home_team.lower() in g_home.lower() or g_home.lower() in home_team.lower()
        away_match = away_team.lower() in g_away.lower() or g_away.lower() in away_team.lower()
        if not (home_match and away_match):
            continue

        best_home = 0.0
        best_away = 0.0
        pin_home: Optional[float] = None
        pin_away: Optional[float] = None
        total_line: Optional[float] = None
        total_over: Optional[float] = None
        total_under: Optional[float] = None
        best_rl_home = 0.0
        best_rl_away = 0.0

        for bm in event.get("bookmakers", []):
            is_pinnacle = (
                bm.get("key", "").lower() == _PINNACLE_KEY
                or "pinnacle" in bm.get("title", "").lower()
            )
            for market in bm.get("markets", []):
                mkey     = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if mkey == "h2h":
                    for outcome in outcomes:
                        name  = outcome.get("name", "")
                        price = outcome.get("price", 0.0) or 0.0
                        if name == g_home:
                            best_home = max(best_home, price)
                            if is_pinnacle:
                                pin_home = price
                        elif name == g_away:
                            best_away = max(best_away, price)
                            if is_pinnacle:
                                pin_away = price

                elif mkey == "totals":
                    for outcome in outcomes:
                        n     = outcome.get("name", "").lower()
                        price = outcome.get("price", 0.0) or 0.0
                        if n == "over":
                            if total_line is None:
                                total_line = outcome.get("point")
                            total_over = max(total_over or 0.0, price) or None
                        elif n == "under":
                            total_under = max(total_under or 0.0, price) or None

                elif mkey == "spreads":
                    for outcome in outcomes:
                        name  = outcome.get("name", "")
                        price = outcome.get("price", 0.0) or 0.0
                        if name == g_home:
                            best_rl_home = max(best_rl_home, price)
                        elif name == g_away:
                            best_rl_away = max(best_rl_away, price)

        return {
            "home_team":    g_home,
            "away_team":    g_away,
            "ml_home":      best_home if best_home > 0 else None,
            "ml_away":      best_away if best_away > 0 else None,
            "pin_home":     pin_home,
            "pin_away":     pin_away,
            "total_line":   total_line,
            "total_over":   total_over if (total_over or 0) > 0 else None,
            "total_under":  total_under if (total_under or 0) > 0 else None,
            "runline_home": best_rl_home if best_rl_home > 0 else None,
            "runline_away": best_rl_away if best_rl_away > 0 else None,
            "game_id":      event.get("id"),
        }

    return {}
