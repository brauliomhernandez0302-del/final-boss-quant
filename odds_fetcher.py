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

ROOT         = Path(__file__).parent
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
BASE_URL     = "https://api.the-odds-api.com/v4"
REGION       = "us,eu"  # eu required for Pinnacle — see _PINNACLE_KEY below
MARKETS      = ["h2h", "totals", "spreads"]
# NOTE (2026-07-12): h2h_h1/totals_h1/spreads_h1 (F5 markets) were added here in
# d170807 but the bulk /sports/{sport}/odds/ endpoint doesn't support period
# markets — The Odds API returns 422 INVALID_MARKET for the WHOLE request when
# any are included, breaking odds fetch for all 14 sports. F5 markets need the
# per-event endpoint (/sports/{sport}/events/{eventId}/odds) instead; the
# normalization branches for h2h_h1/totals_h1/spreads_h1 below are left in
# place for when that's wired up.

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

CACHE_DIR  = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "odds_last.json"
CACHE_TTL  = ODDS_FILE_CACHE_TTL   # seconds

# Was "pinnaclesports" — wrong key, never matched The Odds API's real
# bookmaker key (confirmed against fetch_historical_odds.py's working
# _PINNACLE_KEYS = {"pinnacle"}). Combined with REGION previously excluding
# "eu" (Pinnacle isn't US-licensed), Pinnacle data has been 100% absent
# from live production all season — which silently disabled Platt-2D
# calibration (core/value_detector.py gates it on having a Pinnacle fair
# line). Verified via data/predictions_history.db: game_outcomes.ml_home_pin
# was NULL for every live row since at least 2026-07-04.
_PINNACLE_KEY = "pinnacle"


# ── HTTP helper ───────────────────────────────────────────────────────────


_MAX_RETRY_AFTER = 30  # cap server-provided Retry-After so a bad header can't hang the caller


def _get(url: str, max_retries: int = 3, delay: int = 2) -> Optional[Any]:
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=(5, 30))
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                if attempt < max_retries - 1:
                    wait = min(int(resp.headers.get("Retry-After", delay * 2)), _MAX_RETRY_AFTER)
                    logger.warning("Rate limited — waiting %ds", wait)
                    time.sleep(wait)
                else:
                    logger.warning("Rate limited on final attempt — giving up")
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


def _fetch_all_sports() -> tuple[List[Dict], bool]:
    """Fetch raw events for all configured sports.

    Returns (events, all_ok) — all_ok is False if any sport's fetch failed,
    so the caller can avoid caching a partial result as if it were complete.
    """
    all_events: List[Dict] = []
    all_ok = True

    for sport_key in SPORTS_KEYS:
        url = (
            f"{BASE_URL}/sports/{sport_key}/odds/"
            f"?apiKey={ODDS_API_KEY}"
            f"&regions={REGION}"
            f"&markets={','.join(MARKETS)}"
            f"&oddsFormat=decimal"
        )
        data = _get(url)
        if data is None:
            # Real fetch failure (network/HTTP error) — distinct from a valid
            # 200 response with an empty list (e.g. an off-season sport with
            # no events right now, which is not a failure).
            all_ok = False
            logger.warning("%s: fetch failed — skipping this sport", sport_key)
            continue
        all_events.extend(data)
        logger.info("%s: %d events fetched", sport_key, len(data))
        time.sleep(0.3)   # gentle pacing

    return all_events, all_ok


_last_full_failure_ts: float = 0.0
_FULL_FAILURE_BACKOFF_SECONDS = 60  # avoid hammering the API on repeated calls during an outage


def _get_raw_events() -> List[Dict]:
    """Return cached raw events, fetching fresh if expired or missing."""
    global _last_full_failure_ts

    if not ODDS_API_KEY:
        if CACHE_FILE.exists():
            try:
                content = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                data = content if isinstance(content, list) else content.get("data", [])
                ts = content.get("timestamp") if isinstance(content, dict) else None
                if ts:
                    logger.warning(
                        "ODDS_API_KEY not set — serving cache %.0f min old (may be stale)",
                        (time.time() - ts) / 60,
                    )
                else:
                    logger.warning("ODDS_API_KEY not set — serving cache of unknown age")
                return data
            except Exception:
                pass
        logger.warning("ODDS_API_KEY not set and no cache available")
        return []

    cached = _load_cache()
    if cached is not None:
        return cached

    if time.time() - _last_full_failure_ts < _FULL_FAILURE_BACKOFF_SECONDS:
        logger.warning(
            "Skipping fetch — full failure %.0fs ago, backing off %ds",
            time.time() - _last_full_failure_ts, _FULL_FAILURE_BACKOFF_SECONDS,
        )
        return []

    logger.info("Fetching fresh odds from The Odds API...")
    events, all_ok = _fetch_all_sports()
    if events and all_ok:
        _save_cache(events)
    elif events:
        logger.warning("Partial fetch (some sports failed) — not caching, will retry next call")
    else:
        _last_full_failure_ts = time.time()
        logger.warning("Full fetch failure — backing off %ds before retry", _FULL_FAILURE_BACKOFF_SECONDS)
    return events


# ── Normalization (raw → flat UI dict) ────────────────────────────────────


def _accumulate_point_price(
    by_point: Dict[float, float], point_counts: Dict[float, int],
    point: Optional[float], price: float,
) -> None:
    """Record one bookmaker's (point, price) quote for a line-based market
    (totals/spreads) — grouped by point so the caller can later pick a
    single consensus line instead of blending prices across incompatible
    lines."""
    if point is None or price <= 0:
        return
    by_point[point] = max(by_point.get(point, 0.0), price)
    point_counts[point] = point_counts.get(point, 0) + 1


def _consensus_line_and_price(
    by_point: Dict[float, float],
    point_counts: Dict[float, int],
    pin_point: Optional[float],
) -> "tuple[Optional[float], Optional[float]]":
    """(consensus_point, best_price_at_that_point).

    Consensus prefers Pinnacle's own quoted point when available (Pinnacle
    is the sharpest line), else whichever point the most bookmakers agree
    on. This is what prevents shopping the max price across incompatible
    lines — e.g. pairing a probability computed for "Over 9.5" with a
    book's "Over 10.5" price (a harder bet that pays more), or a book that
    has the road team favored on the runline instead of home.
    """
    if pin_point is not None:
        target = pin_point
    elif point_counts:
        target = max(point_counts, key=point_counts.get)
    else:
        return None, None
    return target, (by_point.get(target) or None)


def _normalize_event(event: Dict) -> Dict:
    """Flatten one raw API event to a UI-friendly dict.

    Extracts across ALL bookmakers:
      - Best ML home/away (h2h)
      - Pinnacle ML home/away (sharp reference)
      - Best over/under odds + total line
      - Best runline (spreads) home/away
    """
    result: Dict[str, Any] = {
        "sport_key":     event.get("sport_key", ""),
        "sport_title":   event.get("sport_title", ""),
        "home_team":     event.get("home_team", ""),
        "away_team":     event.get("away_team", ""),
        "commence_time": event.get("commence_time", ""),
        "home_odds":     None,
        "draw_odds":     None,
        "away_odds":     None,
        "pin_home":      None,
        "pin_away":      None,
        "pin_total":     None,
        "total_line":    None,
        "over_odds":     None,
        "under_odds":    None,
        "runline_home":  None,
        "runline_away":  None,
        # F5 (first 5 innings) — NAMING NOTE (2026-07-06): this is a THIRD
        # F5 naming scheme, distinct from both get_best_odds_for_teams()'s
        # (f5_ml_home/f5_total_over, matching GameOdds' convention) and each
        # other. This one feeds ui/odds_loader.py::get_odds_data() only (the
        # UI dropdown), not run_module()/GameOdds directly, so it isn't the
        # bug that was just fixed — but if anyone later wires this dict's
        # data more directly into F5 analysis, it WILL hit the exact same
        # class of silent-key-mismatch bug a second time. Rename to match
        # GameOdds' convention before connecting it to anything F5-related.
        "f5_home_odds":  None,
        "f5_away_odds":  None,
        "f5_total_line": None,
        "f5_over_odds":  None,
        "f5_under_odds": None,
        "f5_rl_home":    None,
        "f5_rl_away":    None,
        "bookmaker":     None,
        "last_update":   datetime.now().isoformat(),
    }

    home = result["home_team"]
    away = result["away_team"]
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return result

    best_home: float = 0.0
    best_away: float = 0.0
    best_draw: Optional[float] = None
    best_f5_home: float = 0.0
    best_f5_away: float = 0.0

    # Line-based markets (totals/spreads, full-game and F5): grouped by
    # point rather than blended into a single max() — see
    # _consensus_line_and_price()'s docstring for why.
    over_by_point: Dict[float, float] = {}
    over_point_counts: Dict[float, int] = {}
    under_by_point: Dict[float, float] = {}
    pin_total_point: Optional[float] = None

    rl_home_by_point: Dict[float, float] = {}
    rl_home_point_counts: Dict[float, int] = {}
    rl_away_by_point: Dict[float, float] = {}
    rl_away_point_counts: Dict[float, int] = {}
    pin_rl_home_point: Optional[float] = None
    pin_rl_away_point: Optional[float] = None

    f5_over_by_point: Dict[float, float] = {}
    f5_over_point_counts: Dict[float, int] = {}
    f5_under_by_point: Dict[float, float] = {}

    f5_rl_home_by_point: Dict[float, float] = {}
    f5_rl_home_point_counts: Dict[float, int] = {}
    f5_rl_away_by_point: Dict[float, float] = {}
    f5_rl_away_point_counts: Dict[float, int] = {}

    # Prices below are shopped for the best price across ALL bookmakers, not
    # any single one — "bookmaker" records how many books were compared, not
    # the source of any individual price (it used to store just the first
    # book's title, which misleadingly implied a single-book quote).
    result["bookmaker"] = f"best of {len(bookmakers)} books"

    for bm in bookmakers:
        bm_key = bm.get("key", "").lower()
        is_pinnacle = bm_key == _PINNACLE_KEY or "pinnacle" in bm.get("title", "").lower()

        for market in bm.get("markets", []):
            mkey     = market.get("key", "")
            outcomes = market.get("outcomes", [])

            if mkey == "h2h":
                for o in outcomes:
                    name  = o.get("name", "").strip()
                    price = o.get("price") or 0.0
                    if name == home:
                        best_home = max(best_home, price)
                        if is_pinnacle:
                            result["pin_home"] = price
                    elif name == away:
                        best_away = max(best_away, price)
                        if is_pinnacle:
                            result["pin_away"] = price
                    elif name.lower() in ("draw", "empate", "tie"):
                        best_draw = max(best_draw or 0.0, price) or None

            elif mkey == "totals" and outcomes:
                for o in outcomes:
                    n     = o.get("name", "").lower()
                    price = o.get("price") or 0.0
                    point = o.get("point")
                    if n == "over":
                        _accumulate_point_price(over_by_point, over_point_counts, point, price)
                        if is_pinnacle and point is not None:
                            pin_total_point = point
                    elif n == "under":
                        _accumulate_point_price(under_by_point, {}, point, price)

            elif mkey == "spreads" and outcomes:
                for o in outcomes:
                    name  = o.get("name", "").strip()
                    price = o.get("price") or 0.0
                    point = o.get("point")
                    if name == home:
                        _accumulate_point_price(rl_home_by_point, rl_home_point_counts, point, price)
                        if is_pinnacle and point is not None:
                            pin_rl_home_point = point
                    elif name == away:
                        _accumulate_point_price(rl_away_by_point, rl_away_point_counts, point, price)
                        if is_pinnacle and point is not None:
                            pin_rl_away_point = point

            elif mkey == "h2h_h1" and outcomes:
                for o in outcomes:
                    name  = o.get("name", "").strip()
                    price = o.get("price") or 0.0
                    if name == home:
                        best_f5_home = max(best_f5_home, price)
                    elif name == away:
                        best_f5_away = max(best_f5_away, price)

            elif mkey == "totals_h1" and outcomes:
                for o in outcomes:
                    n     = o.get("name", "").lower()
                    price = o.get("price") or 0.0
                    point = o.get("point")
                    if n == "over":
                        _accumulate_point_price(f5_over_by_point, f5_over_point_counts, point, price)
                    elif n == "under":
                        _accumulate_point_price(f5_under_by_point, {}, point, price)

            elif mkey == "spreads_h1" and outcomes:
                for o in outcomes:
                    name  = o.get("name", "").strip()
                    price = o.get("price") or 0.0
                    point = o.get("point")
                    if name == home:
                        _accumulate_point_price(f5_rl_home_by_point, f5_rl_home_point_counts, point, price)
                    elif name == away:
                        _accumulate_point_price(f5_rl_away_by_point, f5_rl_away_point_counts, point, price)

    total_line, over_price = _consensus_line_and_price(over_by_point, over_point_counts, pin_total_point)
    _, under_price = _consensus_line_and_price(under_by_point, over_point_counts, pin_total_point)
    rl_home_point, rl_home_price = _consensus_line_and_price(rl_home_by_point, rl_home_point_counts, pin_rl_home_point)
    rl_away_point, rl_away_price = _consensus_line_and_price(rl_away_by_point, rl_away_point_counts, pin_rl_away_point)
    f5_total_line, f5_over_price = _consensus_line_and_price(f5_over_by_point, f5_over_point_counts, None)
    _, f5_under_price = _consensus_line_and_price(f5_under_by_point, f5_over_point_counts, None)
    _, f5_rl_home_price = _consensus_line_and_price(f5_rl_home_by_point, f5_rl_home_point_counts, None)
    _, f5_rl_away_price = _consensus_line_and_price(f5_rl_away_by_point, f5_rl_away_point_counts, None)

    result["home_odds"]    = best_home      or None
    result["away_odds"]    = best_away      or None
    result["draw_odds"]    = best_draw
    result["total_line"]   = total_line
    result["pin_total"]    = pin_total_point
    result["over_odds"]    = over_price
    result["under_odds"]   = under_price
    result["runline_home"] = rl_home_price
    result["runline_away"] = rl_away_price
    result["f5_home_odds"] = best_f5_home   or None
    result["f5_away_odds"] = best_f5_away   or None
    result["f5_total_line"] = f5_total_line
    result["f5_over_odds"]  = f5_over_price
    result["f5_under_odds"] = f5_under_price
    result["f5_rl_home"]    = f5_rl_home_price
    result["f5_rl_away"]    = f5_rl_away_price
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
    region: str = "us",   # kept for API compatibility, unused — see module-level REGION
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
        best_f5_home = 0.0
        best_f5_away = 0.0

        # Line-based markets: grouped by point, same reasoning as
        # _normalize_event() — see _consensus_line_and_price()'s docstring.
        over_by_point: Dict[float, float] = {}
        over_point_counts: Dict[float, int] = {}
        under_by_point: Dict[float, float] = {}
        pin_total_point: Optional[float] = None

        rl_home_by_point: Dict[float, float] = {}
        rl_home_point_counts: Dict[float, int] = {}
        rl_away_by_point: Dict[float, float] = {}
        rl_away_point_counts: Dict[float, int] = {}
        pin_rl_home_point: Optional[float] = None
        pin_rl_away_point: Optional[float] = None

        f5_over_by_point: Dict[float, float] = {}
        f5_over_point_counts: Dict[float, int] = {}
        f5_under_by_point: Dict[float, float] = {}

        f5_rl_home_by_point: Dict[float, float] = {}
        f5_rl_home_point_counts: Dict[float, int] = {}
        f5_rl_away_by_point: Dict[float, float] = {}
        f5_rl_away_point_counts: Dict[float, int] = {}

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
                        name  = outcome.get("name", "").strip()
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
                        point = outcome.get("point")
                        if n == "over":
                            _accumulate_point_price(over_by_point, over_point_counts, point, price)
                            if is_pinnacle and point is not None:
                                pin_total_point = point
                        elif n == "under":
                            _accumulate_point_price(under_by_point, {}, point, price)

                elif mkey == "spreads":
                    for outcome in outcomes:
                        name  = outcome.get("name", "").strip()
                        price = outcome.get("price", 0.0) or 0.0
                        point = outcome.get("point")
                        if name == g_home:
                            _accumulate_point_price(rl_home_by_point, rl_home_point_counts, point, price)
                            if is_pinnacle and point is not None:
                                pin_rl_home_point = point
                        elif name == g_away:
                            _accumulate_point_price(rl_away_by_point, rl_away_point_counts, point, price)
                            if is_pinnacle and point is not None:
                                pin_rl_away_point = point

                elif mkey == "h2h_h1":
                    for outcome in outcomes:
                        name  = outcome.get("name", "").strip()
                        price = outcome.get("price", 0.0) or 0.0
                        if name == g_home:
                            best_f5_home = max(best_f5_home, price)
                        elif name == g_away:
                            best_f5_away = max(best_f5_away, price)

                elif mkey == "totals_h1":
                    for outcome in outcomes:
                        n     = outcome.get("name", "").lower()
                        price = outcome.get("price", 0.0) or 0.0
                        point = outcome.get("point")
                        if n == "over":
                            _accumulate_point_price(f5_over_by_point, f5_over_point_counts, point, price)
                        elif n == "under":
                            _accumulate_point_price(f5_under_by_point, {}, point, price)

                elif mkey == "spreads_h1":
                    for outcome in outcomes:
                        name  = outcome.get("name", "").strip()
                        price = outcome.get("price", 0.0) or 0.0
                        point = outcome.get("point")
                        if name == g_home:
                            _accumulate_point_price(f5_rl_home_by_point, f5_rl_home_point_counts, point, price)
                        elif name == g_away:
                            _accumulate_point_price(f5_rl_away_by_point, f5_rl_away_point_counts, point, price)

        total_line, total_over = _consensus_line_and_price(over_by_point, over_point_counts, pin_total_point)
        _, total_under = _consensus_line_and_price(under_by_point, over_point_counts, pin_total_point)
        _, best_rl_home = _consensus_line_and_price(rl_home_by_point, rl_home_point_counts, pin_rl_home_point)
        _, best_rl_away = _consensus_line_and_price(rl_away_by_point, rl_away_point_counts, pin_rl_away_point)
        f5_total_line, best_f5_over = _consensus_line_and_price(f5_over_by_point, f5_over_point_counts, None)
        _, best_f5_under = _consensus_line_and_price(f5_under_by_point, f5_over_point_counts, None)
        _, best_f5_rl_home = _consensus_line_and_price(f5_rl_home_by_point, f5_rl_home_point_counts, None)
        _, best_f5_rl_away = _consensus_line_and_price(f5_rl_away_by_point, f5_rl_away_point_counts, None)
        pin_total = pin_total_point

        return {
            "home_team":     g_home,
            "away_team":     g_away,
            "ml_home":       best_home if best_home > 0 else None,
            "ml_away":       best_away if best_away > 0 else None,
            "pin_home":      pin_home,
            "pin_away":      pin_away,
            "pin_total":     pin_total,
            "total_line":    total_line,
            "total_over":    total_over,
            "total_under":   total_under,
            "runline_home":  best_rl_home,
            "runline_away":  best_rl_away,
            # f5_ml_home/f5_ml_away/f5_total_over/f5_total_under: named to
            # match GameOdds' established convention (core/value_detector.py)
            # and run_module.py's read side. Previously these were
            # "f5_home"/"f5_away"/"f5_over"/"f5_under" — a naming mismatch
            # that silently meant F5 markets could never activate anywhere
            # run_module() relies on this function for real odds (e.g.
            # track_record/publisher.py's publish_mlb_picks, which calls
            # run_mlb() without an explicit market_odds override) — only
            # f5_total_line happened to match by coincidence, letting
            # analyze_first5()'s outer gate pass while every inner F5
            # ML/totals check silently failed on None. Found + fixed
            # 2026-07-06, confirmed via GameOdds' own self-test data
            # (value_detector.py) using this exact naming.
            "f5_ml_home":    best_f5_home if best_f5_home > 0 else None,
            "f5_ml_away":    best_f5_away if best_f5_away > 0 else None,
            "f5_total_line": f5_total_line,
            "f5_total_over": best_f5_over,
            "f5_total_under": best_f5_under,
            "f5_rl_home":    best_f5_rl_home,
            "f5_rl_away":    best_f5_rl_away,
            "game_id":       event.get("id"),
        }

    return {}
