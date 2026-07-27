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
from datetime import datetime, timedelta, timezone
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
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_usa_mls",
    "mma_mixed_martial_arts",
]
# NFL/NCAAF/NHL/NCAAB/Euroleague removed 2026-07-14 — no module, analyzer, or
# UI tab anywhere in this repo consumes them (verified via repo-wide grep).
# Fetching them only burned The Odds API quota for sports nobody analyzes;
# see the ODDS_API_KEY quota-exhaustion history in project memory. Re-add the
# specific key if/when a real analyzer for that sport gets wired up.

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
        "runline_line":  None,
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
    # Pinnacle's own PRICE on each side, not just its line. Until 2026-07-26
    # only the point was kept here, so nothing downstream could ever devig
    # Pinnacle's two-sided close for a total or a runline — the exact input
    # docs/PROTOCOLO_CLV_V1.md's primary metric is defined on. The data was
    # already in every response (MARKETS includes totals/spreads); it was
    # parsed and dropped.
    pin_total_over: Optional[float] = None
    pin_total_under: Optional[float] = None

    rl_home_by_point: Dict[float, float] = {}
    rl_home_point_counts: Dict[float, int] = {}
    rl_away_by_point: Dict[float, float] = {}
    rl_away_point_counts: Dict[float, int] = {}
    pin_rl_home_point: Optional[float] = None
    pin_rl_away_point: Optional[float] = None
    pin_rl_home_price: Optional[float] = None
    pin_rl_away_price: Optional[float] = None

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
                            pin_total_over  = price or None
                    elif n == "under":
                        _accumulate_point_price(under_by_point, {}, point, price)
                        if is_pinnacle and point is not None:
                            pin_total_under = price or None

            elif mkey == "spreads" and outcomes:
                for o in outcomes:
                    name  = o.get("name", "").strip()
                    price = o.get("price") or 0.0
                    point = o.get("point")
                    if name == home:
                        _accumulate_point_price(rl_home_by_point, rl_home_point_counts, point, price)
                        if is_pinnacle and point is not None:
                            pin_rl_home_point = point
                            pin_rl_home_price = price or None
                    elif name == away:
                        _accumulate_point_price(rl_away_by_point, rl_away_point_counts, point, price)
                        if is_pinnacle and point is not None:
                            pin_rl_away_point = point
                            pin_rl_away_price = price or None

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
    # Pinnacle's two-sided close for the line markets, each with the point it
    # was quoted at. Purely additive — nothing in the live prediction path
    # reads these keys; they exist so a closing snapshot can be devigged the
    # same way the moneyline one already is.
    result["pin_total_over"]       = pin_total_over
    result["pin_total_under"]      = pin_total_under
    result["pin_total_point"]      = pin_total_point
    result["pin_runline_home"]     = pin_rl_home_price
    result["pin_runline_away"]     = pin_rl_away_price
    result["pin_runline_home_point"] = pin_rl_home_point
    result["pin_runline_away_point"] = pin_rl_away_point
    # Magnitude only (not signed) — the home/away sign convention
    # ("home is always -line") is a separate, pre-existing assumption
    # elsewhere in the pipeline (GameOdds/analyze_runline), not something
    # this fetcher decides. rl_home_point and rl_away_point aren't
    # guaranteed to agree in sign if books disagree on which side is
    # favored (see _consensus_line_and_price's docstring) — magnitude is
    # what actually matters for the cover-probability math.
    result["runline_line"] = (
        abs(rl_home_point) if rl_home_point is not None
        else (abs(rl_away_point) if rl_away_point is not None else None)
    )
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


_ODDS_MATCH_WINDOW = timedelta(hours=6)

# Margen por el que el mejor candidato tiene que ganarle al segundo para
# considerarse identificado. La regla anterior sólo se abstenía ante un empate
# EXACTO, que en la práctica no ocurre nunca: con dos juegos del mismo par de
# equipos en la ventana (un doubleheader), bastaba con estar un minuto más cerca
# para quedarse con el precio del otro partido.
#
# 90 minutos separa con holgura los casos reales de los dudosos: un doubleheader
# tradicional tiene sus dos juegos a ~3.5h y uno partido a ~5-7h, así que el
# candidato correcto siempre gana por horas; un margen menor a 90 min significa
# que los dos eventos son igual de plausibles y no hay con qué decidir.
# Verificado el 2026-07-26 sobre los 27 juegos de la ventana: 0 o 1 candidato por
# juego, nunca más — este umbral es inerte en operación normal y sólo actúa el
# día que hay doubleheader, que es para lo que existe.
_ODDS_MATCH_MIN_MARGIN = timedelta(minutes=90)


def _parse_commence(value: str) -> Optional[datetime]:
    """Parse an ISO-8601 commence_time (The Odds API or MLB schedule format,
    both use a 'Z'-suffixed UTC timestamp) into a tz-aware datetime. None on
    any parse failure — callers must treat that as "can't disambiguate"."""
    try:
        v = value[:-1] + "+00:00" if value.endswith("Z") else value
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError, AttributeError):
        return None


def get_best_odds_for_teams(
    home_team: str,
    away_team: str,
    commence_time: str,
    sport: str = "baseball_mlb",
    region: str = "us",   # kept for API compatibility, unused — see module-level REGION
) -> Dict:
    """Find best ML prices + Pinnacle reference for a specific matchup.

    `commence_time` (required — the target game's own real start time, ISO
    8601, e.g. from the MLB schedule's `gameDate`) disambiguates between
    multiple odds-API events for the same two teams — a multi-game series or
    a doubleheader. Team-name matching ALONE (the pre-2026-07-19 behavior)
    silently returned whichever matching event happened to be first in the
    API's event list, which could be a different day's game for the same two
    teams entirely — a confirmed real bug, see
    audit_20260714/verificacion_operativa/reporte.md (V2).

    Searches the cached raw events — no extra API call. Returns {} (never a
    guess) if no matching event falls within ±6h of `commence_time`, if two
    candidates are exactly tied in distance (ambiguous), or if
    `commence_time` itself doesn't parse.
    """
    target = _parse_commence(commence_time)
    if target is None:
        logger.warning(
            "get_best_odds_for_teams: unparseable commence_time=%r for %s @ %s — "
            "cannot disambiguate, returning no odds",
            commence_time, away_team, home_team,
        )
        return {}

    raw = _get_raw_events()

    candidates: List[tuple] = []
    for candidate_event in raw:
        if candidate_event.get("sport_key") != sport:
            continue
        g_home = candidate_event.get("home_team", "")
        g_away = candidate_event.get("away_team", "")
        home_match = home_team.lower() in g_home.lower() or g_home.lower() in home_team.lower()
        away_match = away_team.lower() in g_away.lower() or g_away.lower() in away_team.lower()
        if not (home_match and away_match):
            continue
        event_commence = _parse_commence(candidate_event.get("commence_time", ""))
        if event_commence is None:
            continue
        delta = abs(event_commence - target)
        if delta <= _ODDS_MATCH_WINDOW:
            candidates.append((delta, candidate_event))

    if not candidates:
        logger.warning(
            "get_best_odds_for_teams: no odds event for %s @ %s within %s of commence_time=%s",
            away_team, home_team, _ODDS_MATCH_WINDOW, commence_time,
        )
        return {}

    candidates.sort(key=lambda c: c[0])
    if len(candidates) > 1 and (candidates[1][0] - candidates[0][0]) < _ODDS_MATCH_MIN_MARGIN:
        logger.warning(
            "get_best_odds_for_teams: ambiguous match for %s @ %s — 2 odds events "
            "casi igual de cerca de commence_time=%s (%s y %s, margen < %s), "
            "refusing to guess",
            away_team, home_team, commence_time,
            candidates[0][0], candidates[1][0], _ODDS_MATCH_MIN_MARGIN,
        )
        return {}

    event = candidates[0][1]
    # The winning event's OWN team names — distinct from the search loop's
    # g_home/g_away above, which by this point hold whatever candidate_event
    # was LAST examined in `raw` (any MLB game, not necessarily this one).
    # Every block below that matches an outcome's `name` against the home/
    # away side, plus the return dict's home_team/away_team, must use these,
    # not g_home/g_away — using the stale loop variables here was a real bug
    # (found 2026-07-21 building the React matchup dashboard): h2h/spreads
    # outcome names never matched g_home/g_away's stale value whenever the
    # matched event wasn't also the last MLB event in the raw list, so
    # ml_home/ml_away/pin_home/pin_away silently stayed None and the
    # returned home_team/away_team labeled a different game entirely. Every
    # existing test fixture used the same two teams for every event in its
    # mocked raw list, so g_home/g_away's stale value always coincidentally
    # matched — masking this in the full suite until a real multi-game
    # schedule was fetched. See tests/test_odds_fetcher_wrong_event_team_names.py.
    home_name = event.get("home_team", "")
    away_name = event.get("away_team", "")
    best_home = 0.0
    best_away = 0.0
    best_home_book: Optional[str] = None
    best_away_book: Optional[str] = None
    pin_home: Optional[float] = None
    pin_away: Optional[float] = None
    best_f5_home = 0.0
    best_f5_away = 0.0
    # Every bookmaker with a valid two-sided h2h quote, kept so
    # docs/PROTOCOLO_CLV_V1.md's pre-registered fallback (median devigged
    # price across books when Pinnacle's close is unavailable) has real
    # per-book data to work with — not just the aggregate best price.
    all_books_h2h: List[Dict[str, Any]] = []

    # Line-based markets: grouped by point, same reasoning as
    # _normalize_event() — see _consensus_line_and_price()'s docstring.
    over_by_point: Dict[float, float] = {}
    over_point_counts: Dict[float, int] = {}
    under_by_point: Dict[float, float] = {}
    pin_total_point: Optional[float] = None
    # See _normalize_event()'s note: Pinnacle's PRICE per side on the line
    # markets, not just its point. This is the function
    # track_record/capture_closing_lines.py sweeps with, so this is the one
    # that decides whether a runline/total close is recoverable at all.
    pin_total_over: Optional[float] = None
    pin_total_under: Optional[float] = None

    rl_home_by_point: Dict[float, float] = {}
    rl_home_point_counts: Dict[float, int] = {}
    rl_away_by_point: Dict[float, float] = {}
    rl_away_point_counts: Dict[float, int] = {}
    pin_rl_home_point: Optional[float] = None
    pin_rl_away_point: Optional[float] = None
    pin_rl_home_price: Optional[float] = None
    pin_rl_away_price: Optional[float] = None

    f5_over_by_point: Dict[float, float] = {}
    f5_over_point_counts: Dict[float, int] = {}
    f5_under_by_point: Dict[float, float] = {}

    f5_rl_home_by_point: Dict[float, float] = {}
    f5_rl_home_point_counts: Dict[float, int] = {}
    f5_rl_away_by_point: Dict[float, float] = {}
    f5_rl_away_point_counts: Dict[float, int] = {}

    for bm in event.get("bookmakers", []):
        book_name = bm.get("title") or bm.get("key") or "unknown"
        is_pinnacle = (
            bm.get("key", "").lower() == _PINNACLE_KEY
            or "pinnacle" in bm.get("title", "").lower()
        )
        bm_h2h_home: Optional[float] = None
        bm_h2h_away: Optional[float] = None
        for market in bm.get("markets", []):
            mkey     = market.get("key", "")
            outcomes = market.get("outcomes", [])

            if mkey == "h2h":
                for outcome in outcomes:
                    name  = outcome.get("name", "").strip()
                    price = outcome.get("price", 0.0) or 0.0
                    if name == home_name:
                        bm_h2h_home = price
                        if price > best_home:
                            best_home = price
                            best_home_book = book_name
                        if is_pinnacle:
                            pin_home = price
                    elif name == away_name:
                        bm_h2h_away = price
                        if price > best_away:
                            best_away = price
                            best_away_book = book_name
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
                            pin_total_over  = price or None
                    elif n == "under":
                        _accumulate_point_price(under_by_point, {}, point, price)
                        if is_pinnacle and point is not None:
                            pin_total_under = price or None

            elif mkey == "spreads":
                for outcome in outcomes:
                    name  = outcome.get("name", "").strip()
                    price = outcome.get("price", 0.0) or 0.0
                    point = outcome.get("point")
                    if name == home_name:
                        _accumulate_point_price(rl_home_by_point, rl_home_point_counts, point, price)
                        if is_pinnacle and point is not None:
                            pin_rl_home_point = point
                            pin_rl_home_price = price or None
                    elif name == away_name:
                        _accumulate_point_price(rl_away_by_point, rl_away_point_counts, point, price)
                        if is_pinnacle and point is not None:
                            pin_rl_away_point = point
                            pin_rl_away_price = price or None

            elif mkey == "h2h_h1":
                for outcome in outcomes:
                    name  = outcome.get("name", "").strip()
                    price = outcome.get("price", 0.0) or 0.0
                    if name == home_name:
                        best_f5_home = max(best_f5_home, price)
                    elif name == away_name:
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
                    if name == home_name:
                        _accumulate_point_price(f5_rl_home_by_point, f5_rl_home_point_counts, point, price)
                    elif name == away_name:
                        _accumulate_point_price(f5_rl_away_by_point, f5_rl_away_point_counts, point, price)

        if bm_h2h_home is not None and bm_h2h_home > 0 and bm_h2h_away is not None and bm_h2h_away > 0:
            all_books_h2h.append({"book": book_name, "home": bm_h2h_home, "away": bm_h2h_away})

    total_line, total_over = _consensus_line_and_price(over_by_point, over_point_counts, pin_total_point)
    _, total_under = _consensus_line_and_price(under_by_point, over_point_counts, pin_total_point)
    rl_home_point, best_rl_home = _consensus_line_and_price(rl_home_by_point, rl_home_point_counts, pin_rl_home_point)
    rl_away_point, best_rl_away = _consensus_line_and_price(rl_away_by_point, rl_away_point_counts, pin_rl_away_point)
    f5_total_line, best_f5_over = _consensus_line_and_price(f5_over_by_point, f5_over_point_counts, None)
    _, best_f5_under = _consensus_line_and_price(f5_under_by_point, f5_over_point_counts, None)
    _, best_f5_rl_home = _consensus_line_and_price(f5_rl_home_by_point, f5_rl_home_point_counts, None)
    _, best_f5_rl_away = _consensus_line_and_price(f5_rl_away_by_point, f5_rl_away_point_counts, None)
    pin_total = pin_total_point

    return {
        "home_team":     home_name,
        "away_team":     away_name,
        "ml_home":       best_home if best_home > 0 else None,
        "ml_away":       best_away if best_away > 0 else None,
        "ml_home_book":  best_home_book,
        "ml_away_book":  best_away_book,
        "pin_home":      pin_home,
        "pin_away":      pin_away,
        "all_books_h2h": all_books_h2h,
        "pin_total":     pin_total,
        "total_line":    total_line,
        "total_over":    total_over,
        "total_under":   total_under,
        "runline_home":  best_rl_home,
        "runline_away":  best_rl_away,
        # Pinnacle's own two-sided close on the line markets, each with the
        # point it was quoted at (2026-07-26). Additive: no existing key
        # changes value, and nothing in the live prediction path reads these —
        # they exist so track_record can store a runline/total close that is
        # actually devig-able, instead of storing the MONEYLINE Pinnacle pair
        # next to a runline pick, which is what it did before.
        "pin_total_over":         pin_total_over,
        "pin_total_under":        pin_total_under,
        "pin_total_point":        pin_total_point,
        "pin_runline_home":       pin_rl_home_price,
        "pin_runline_away":       pin_rl_away_price,
        "pin_runline_home_point": pin_rl_home_point,
        "pin_runline_away_point": pin_rl_away_point,
        # Magnitude only — kept for the live EV path (core/value_detector.py
        # ::analyze_runline, frozen — see CLAUDE.md), which still assumes
        # "home is always the favorite" and doesn't consume a sign.
        "runline_line": (
            abs(rl_home_point) if rl_home_point is not None
            else (abs(rl_away_point) if rl_away_point is not None else None)
        ),
        # Signed points — home/away aren't guaranteed to agree in sign if
        # books disagree on who's favored, so both are exposed independently
        # rather than derived from one another. Used by track_record's
        # post-game grading (reconciler.py) to know which side was actually
        # favored for THIS pick, instead of assuming home always is.
        "runline_home_point": rl_home_point,
        "runline_away_point": rl_away_point,
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
