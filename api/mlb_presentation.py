"""api/mlb_presentation.py — read-only MLB Stats API lookups for the React
matchup dashboard, independent of run_module.py and every frozen engine file.

run_module()'s own output (results['game_info']) only carries team/pitcher
NAMES, not the ids/lineups/roster needed to render photos and a projected
lineup. Rather than touch run_module.py (part of the ML prediction path,
frozen per docs/PROTOCOLO_CLV_V1.md/CLAUDE.md) to thread those ids through,
this module re-fetches them directly from MLBStatsAPI/MLB Stats API — the
exact same schedule endpoint data_fetchers.py already calls, read a second
time for presentation purposes only. Zero coupling to bullpen_engine.py's or
pitcher_engine.py's internals: these are independent, duplicate-but-harmless
reads of public MLB data.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from data_fetchers import MLBStatsAPI, _current_mlb_season
from modules.baseball_module.context_engine.bullpen_engine import (
    CACHE_DIR as _BULLPEN_CACHE_DIR,
    _fetch_reliever_ids,
    _fetch_savant_pitcher_expected,
    _fetch_team_roster,
)
from modules.baseball_module.data_enrichment.fangraphs_fetcher import FanGraphsFetcher

_BASE_URL = "https://statsapi.mlb.com/api/v1"
_mlb_api = MLBStatsAPI()


def _today_and_tomorrow() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d"), (now + timedelta(days=1)).strftime("%Y-%m-%d")


# Games in these states aren't upcoming/live predictions — a "Final" game
# has already been decided (no market to evaluate against, hero falls back
# to Platt-1D-only, best_bets is empty — the whole dashboard degrades to
# explain a game that's over) and a "Postponed" entry is a rained-out
# placeholder, not the game that will actually be played (see the makeup
# game's own, different game_pk). Exact-match on MLB Stats API's
# `detailedState` — not a substring/fuzzy filter.
_PICKER_EXCLUDED_STATUSES = {"Final", "Postponed"}


def list_scheduled_games() -> List[Dict[str, Any]]:
    """Today + tomorrow's MLB schedule, same window publisher.py itself
    scans, trimmed to what a game picker needs. Excludes games that are
    already decided or rained out — see _PICKER_EXCLUDED_STATUSES."""
    today, tomorrow = _today_and_tomorrow()
    games = (_mlb_api.get_todays_games(date=today) or []) + \
            (_mlb_api.get_todays_games(date=tomorrow) or [])
    return [
        {
            "game_pk": g.get("game_pk"),
            "home_team": g.get("home_team"),
            "away_team": g.get("away_team"),
            "home_team_id": g.get("home_team_id"),
            "away_team_id": g.get("away_team_id"),
            "venue": g.get("venue"),
            "status": g.get("status"),
            "game_date": g.get("game_date"),
            "official_date": g.get("official_date"),
        }
        for g in games
        if g.get("game_pk") and g.get("status") not in _PICKER_EXCLUDED_STATUSES
    ]


def find_scheduled_game(game_pk: int) -> Optional[Dict[str, Any]]:
    """The full parsed schedule entry for one game_pk (team ids, pitcher ids,
    lineups, venue, status) — None if it's not in today/tomorrow's window.

    lineup_confirmed follows the same threshold run_module.py's own TTE
    lineup fetch uses (>=9 batters per side) — MLB populates `lineups` only
    once the real starting order is posted, empty before that. Never
    presented as confirmed when it isn't (project provenance rule).
    """
    today, tomorrow = _today_and_tomorrow()
    for date in (today, tomorrow):
        for g in _mlb_api.get_todays_games(date=date) or []:
            if g.get("game_pk") == game_pk:
                home_lineup = g.get("home_lineup") or []
                away_lineup = g.get("away_lineup") or []
                g["lineup_confirmed"] = len(home_lineup) >= 9 and len(away_lineup) >= 9
                return g
    return None


_SOURCES_TTL_SECONDS = 3600
_sources_cache: Dict[int, tuple] = {}


def _season_pitcher_sources(season: int) -> tuple:
    """The two season-wide maps `bullpen_engine.py` aggregates over, read from
    the engine's own fetchers (and therefore its own 24h disk caches, already
    warm on any day the pipeline ran). Held in-process for an hour so a page
    refresh doesn't re-parse a full season of Savant/FanGraphs rows."""
    cached = _sources_cache.get(season)
    now = datetime.now(timezone.utc).timestamp()
    if cached and now - cached[0] < _SOURCES_TTL_SECONDS:
        return cached[1], cached[2]

    savant_exp = _fetch_savant_pitcher_expected(season)
    fg_pitchers = FanGraphsFetcher(cache_dir=_BULLPEN_CACHE_DIR).get_all_pitcher_stats(season)
    _sources_cache[season] = (now, savant_exp, fg_pitchers)
    return savant_exp, fg_pitchers


def _fetch_roster_positions(team_id: int, season: int) -> Dict[int, str]:
    """Position abbreviation per player on the gameday roster.

    Same endpoint `_fetch_team_roster` uses; it keeps only {id: name}, and the
    position is what lets the UI say out loud that a listed "reliever" is
    actually a catcher who threw a mop-up inning (see `fetch_bullpen_usage`).
    Best-effort: an empty map just means no position labels."""
    try:
        r = requests.get(
            f"{_BASE_URL}/teams/{team_id}/roster",
            params={"rosterType": "gameday", "season": season},
            timeout=(5, 20),
        )
        r.raise_for_status()
        entries = r.json().get("roster", [])
    except Exception:
        return {}
    out: Dict[int, str] = {}
    for entry in entries:
        pid = (entry.get("person") or {}).get("id")
        abbr = (entry.get("position") or {}).get("abbreviation")
        if pid and abbr:
            out[int(pid)] = abbr
    return out


def fetch_bullpen_usage(team_id: int) -> Dict[str, Any]:
    """The relievers `bullpen_engine.py` ACTUALLY aggregated, with one count.

    Before this, the dashboard showed three numbers that are different by
    construction and were displayed as if they measured the same thing
    (audit_20260714/val_audit/reporte.md VAL-7.2): the roster list length, the
    engine's `n_pitchers` (Savant coverage) and its `n_siera_pitchers`
    (FanGraphs coverage) — e.g. 11 listed / 4 / 4 for one real bullpen.

    The list now IS the engine's contributor set, rebuilt from the engine's own
    inputs so it can't drift from what the model did:

      1. `_fetch_team_roster` + `_fetch_reliever_ids` — the same classification
         the engine restricts both aggregates to (a reliever is a roster player
         with zero starts and at least one pitching appearance this season).
      2. Each classified reliever is kept only if the engine had data to weight
         them with: Savant plate appearances (the xwOBA/barrel aggregate,
         PA-weighted) and/or FanGraphs SIERA-or-xFIP with innings (the SIERA
         aggregate, IP-weighted). A reliever with neither contributed exactly
         nothing to `total_mult` and is not listed.

    `n_used` — the single defined count — is the length of that list, and the
    per-pitcher `savant_pa`/`siera_ip` are the actual weights, so a name with a
    tiny number is visibly a tiny contributor rather than an equal-looking row.
    `n_savant`/`n_siera` are still returned because they must reproduce the
    engine's `n_pitchers`/`n_siera_pitchers` exactly — that equality is the
    check that this mirror is faithful, not three more numbers for the UI to
    show side by side.

    Note on position players: the engine's own classifier admits anyone with a
    pitching appearance and no starts, so a catcher who threw a blowout inning
    (real case: Carson Kelly, 1.0 IP, ERA 18.00) IS in the engine's SIERA
    aggregate, weighted by that single inning. Hiding him would make the UI
    disagree with the model; he is listed with his position and his 1.0 IP.

    Today's probable starter is NOT filtered out here: the engine doesn't
    filter him either, and he can only appear if it classified him as a pure
    reliever (an opener), in which case the engine did use him.
    """
    season = _current_mlb_season()
    empty: Dict[str, Any] = {
        "pitchers": [], "n_used": 0, "n_classified": 0,
        "n_savant": 0, "n_siera": 0, "degraded": True,
    }

    roster = _fetch_team_roster(team_id, season)
    if not roster:
        return empty

    reliever_ids = _fetch_reliever_ids(team_id, season, roster)
    if reliever_ids is None:
        # Role classification failed outright — same degraded state the engine
        # itself falls back to (whole-roster aggregate). Say so instead of
        # printing a list that would mean something different from the model's.
        return empty

    savant_exp, fg_pitchers = _season_pitcher_sources(season)
    positions = _fetch_roster_positions(team_id, season)

    pitchers: List[Dict[str, Any]] = []
    for pid, name in roster.items():
        if pid not in reliever_ids:
            continue

        exp = savant_exp.get(pid)
        savant_pa = float(exp["pa"]) if exp else None

        fg = fg_pitchers.get(pid) or fg_pitchers.get(str(pid))
        siera_ip = None
        if fg:
            primary = fg.get("siera")
            if primary is None:
                primary = fg.get("xfip")
            ip = fg.get("ip")
            if primary is not None and ip:
                siera_ip = float(ip)

        if savant_pa is None and siera_ip is None:
            continue  # classified as a reliever, but weighted zero by the engine

        pitchers.append({
            "id": pid,
            "name": name,
            "position": positions.get(pid),
            "savant_pa": savant_pa,
            "siera_ip": siera_ip,
        })

    pitchers.sort(key=lambda p: (p["savant_pa"] or 0.0, p["siera_ip"] or 0.0), reverse=True)

    return {
        "pitchers": pitchers,
        "n_used": len(pitchers),
        "n_classified": len(reliever_ids & roster.keys()),
        "n_savant": sum(1 for p in pitchers if p["savant_pa"] is not None),
        "n_siera": sum(1 for p in pitchers if p["siera_ip"] is not None),
        "degraded": False,
    }


def fetch_pitcher_bio(player_id: int) -> Dict[str, Any]:
    """Throwing hand for one pitcher — not carried anywhere in the existing
    pipeline's game_data (platoon splits use opposing-lineup handedness, not
    the pitcher's own), so this is a small dedicated lookup."""
    try:
        r = requests.get(f"{_BASE_URL}/people/{player_id}", timeout=(5, 20))
        r.raise_for_status()
        people = r.json().get("people") or []
        if not people:
            return {}
        person = people[0]
        return {"throws": (person.get("pitchHand") or {}).get("code")}
    except Exception:
        return {}
