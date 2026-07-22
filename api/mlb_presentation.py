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

from data_fetchers import MLBStatsAPI

_BASE_URL = "https://statsapi.mlb.com/api/v1"
_mlb_api = MLBStatsAPI()


def _today_and_tomorrow() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d"), (now + timedelta(days=1)).strftime("%Y-%m-%d")


def list_scheduled_games() -> List[Dict[str, Any]]:
    """Today + tomorrow's MLB schedule, same window publisher.py itself
    scans, trimmed to what a game picker needs."""
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
        if g.get("game_pk")
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


def fetch_bullpen_roster(
    team_id: int, exclude_pitcher_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Active-roster pitchers for `team_id`, minus the probable starter.

    NOT "today's confirmed available relievers" — MLB doesn't publish that
    pre-game, and this project's provenance rule forbids presenting a guess
    as a confirmed fact. Callers/UI must label this as roster info, not a
    bullpen-availability confirmation.
    """
    try:
        r = requests.get(
            f"{_BASE_URL}/teams/{team_id}/roster",
            params={"rosterType": "active"},
            timeout=(5, 20),
        )
        r.raise_for_status()
        roster = r.json().get("roster", [])
    except Exception:
        return []

    out = []
    for entry in roster:
        position = (entry.get("position") or {}).get("abbreviation")
        if position != "P":
            continue
        person = entry.get("person") or {}
        pid = person.get("id")
        if not pid or pid == exclude_pitcher_id:
            continue
        out.append({"id": pid, "name": person.get("fullName")})
    return out


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
