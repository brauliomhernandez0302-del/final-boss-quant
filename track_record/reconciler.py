"""
track_record/reconciler.py — post-game result resolution.

Fetches final scores from MLB Stats API, matches them to pending picks,
and resolves WIN / LOSS / PUSH for each market.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from track_record.db import TrackRecordDB

log = logging.getLogger("track_record.reconciler")


def _fetch_mlb_final(game_pk: int) -> Optional[Tuple[int, int]]:
    """Return (home_score, away_score) for a completed MLB game, or None."""
    try:
        import statsapi
        data = statsapi.get("game", {"gamePk": game_pk})
        linescore = data.get("liveData", {}).get("linescore", {})
        teams = linescore.get("teams", {})
        h = teams.get("home", {}).get("runs")
        a = teams.get("away", {}).get("runs")
        status = (
            data.get("gameData", {})
            .get("status", {})
            .get("abstractGameState", "")
        )
        if status != "Final" or h is None or a is None:
            return None
        return int(h), int(a)
    except Exception as e:
        log.debug(f"statsapi error for pk={game_pk}: {e}")
        return None


def _fetch_mlb_final_v2(game_pk: int) -> Optional[Tuple[int, int]]:
    """Fallback: use data_fetchers.MLBStatsAPI."""
    try:
        from data_fetchers import MLBStatsAPI
        api = MLBStatsAPI()
        import requests
        url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        teams = data.get("teams", {})
        h = teams.get("home", {}).get("runs")
        a = teams.get("away", {}).get("runs")
        if h is None or a is None:
            return None
        # verify the game is final
        sched_url = f"https://statsapi.mlb.com/api/v1/schedule?gamePk={game_pk}&hydrate=linescore"
        sresp = requests.get(sched_url, timeout=10)
        sresp.raise_for_status()
        sdata = sresp.json()
        dates = sdata.get("dates", [{}])
        games = dates[0].get("games", [{}]) if dates else [{}]
        state = games[0].get("status", {}).get("abstractGameState", "") if games else ""
        if state != "Final":
            return None
        return int(h), int(a)
    except Exception as e:
        log.debug(f"MLBStatsAPI fallback error for pk={game_pk}: {e}")
        return None


def _fetch_f5_score(game_pk: int) -> Optional[Tuple[int, int]]:
    """Return (home_f5_runs, away_f5_runs) from innings 1-5, or None if incomplete."""
    try:
        import requests
        url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/linescore"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        innings = resp.json().get("innings", [])
        completed = [inn for inn in innings if 1 <= inn.get("num", 0) <= 5]
        if len(completed) < 5:
            return None
        home_f5 = sum(int(inn.get("home", {}).get("runs", 0) or 0) for inn in completed)
        away_f5 = sum(int(inn.get("away", {}).get("runs", 0) or 0) for inn in completed)
        return home_f5, away_f5
    except Exception as e:
        log.debug(f"F5 linescore error for pk={game_pk}: {e}")
        return None


def _get_final_score(sport: str, game_pk: int) -> Optional[Tuple[int, int]]:
    if sport == "MLB":
        result = _fetch_mlb_final(game_pk)
        if result is None:
            result = _fetch_mlb_final_v2(game_pk)
        return result
    # NBA / UFC: not yet implemented
    return None


def _resolve_market(
    market: str,
    home_score: int,
    away_score: int,
    total_line: Optional[float] = None,
    runline: float = 1.5,
    f5_home: Optional[int] = None,
    f5_away: Optional[int] = None,
) -> str:
    """Determine WIN / LOSS / PUSH for a given market and final score."""
    diff = home_score - away_score  # positive = home wins
    total = home_score + away_score

    if market == "ML_HOME":
        if diff > 0:
            return "WIN"
        if diff < 0:
            return "LOSS"
        return "PUSH"

    if market == "ML_AWAY":
        if diff < 0:
            return "WIN"
        if diff > 0:
            return "LOSS"
        return "PUSH"

    if market == "RL_HOME":
        cover = diff - runline
        if cover > 0:
            return "WIN"
        if cover < 0:
            return "LOSS"
        return "PUSH"

    if market == "RL_AWAY":
        cover = -diff - runline
        if cover > 0:
            return "WIN"
        if cover < 0:
            return "LOSS"
        return "PUSH"

    if market in ("OVER", "F5_OVER"):
        if total_line is None:
            return "VOID"
        if total > total_line:
            return "WIN"
        if total < total_line:
            return "LOSS"
        return "PUSH"

    if market in ("UNDER", "F5_UNDER"):
        if total_line is None:
            return "VOID"
        if total < total_line:
            return "WIN"
        if total > total_line:
            return "LOSS"
        return "PUSH"

    if market == "F5_HOME":
        if f5_home is None or f5_away is None:
            return "VOID"
        diff_f5 = f5_home - f5_away
        if diff_f5 > 0: return "WIN"
        if diff_f5 < 0: return "LOSS"
        return "PUSH"

    if market == "F5_AWAY":
        if f5_home is None or f5_away is None:
            return "VOID"
        diff_f5 = f5_home - f5_away
        if diff_f5 < 0: return "WIN"
        if diff_f5 > 0: return "LOSS"
        return "PUSH"

    return "VOID"


def _calc_pnl(result: str, stake_units: float, odds_decimal: Optional[float]) -> float:
    if result == "WIN":
        dec = odds_decimal or 1.909
        return round(stake_units * (dec - 1), 4)
    if result == "LOSS":
        return round(-stake_units, 4)
    return 0.0  # PUSH / VOID


def reconcile_pending(
    db: Optional[TrackRecordDB] = None,
    lookback_days: int = 7,
    sport: str = "MLB",
) -> Dict[str, int]:
    """
    Find all unresolved picks whose game_date is in the past and try to
    resolve them. Returns counts of resolved/skipped/voided.
    """
    if db is None:
        db = TrackRecordDB()

    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")

    pending = db.get_pending(sport=sport)
    stats = {"resolved": 0, "skipped": 0, "voided": 0, "errors": 0}

    for pick in pending:
        gdate = pick["game_date"]
        if gdate >= today:
            # game hasn't been played yet
            continue
        if gdate < cutoff:
            log.debug(f"Pick {pick['pick_uid']} too old ({gdate}), skipping")
            stats["skipped"] += 1
            continue

        game_pk = pick["game_pk"]
        if not game_pk:
            log.debug(f"No game_pk for {pick['pick_uid']}, skipping")
            stats["skipped"] += 1
            continue

        score = _get_final_score(sport, int(game_pk))
        if score is None:
            log.debug(f"No final score yet for pk={game_pk}")
            stats["skipped"] += 1
            continue

        home_score, away_score = score
        market = pick["market"]
        total_line = pick["total_line"] if "total_line" in pick.keys() else None
        f5_score = None
        if market in ("F5_HOME", "F5_AWAY", "F5_OVER", "F5_UNDER"):
            f5_score = _fetch_f5_score(int(game_pk))
        result = _resolve_market(
            market, home_score, away_score,
            total_line=total_line,
            f5_home=f5_score[0] if f5_score else None,
            f5_away=f5_score[1] if f5_score else None,
        )
        stake = pick["stake_units"] or 1.0
        pnl = _calc_pnl(result, stake, pick["odds_decimal"])

        ok = db.resolve_pick(
            pick_uid=pick["pick_uid"],
            actual_home_score=home_score,
            actual_away_score=away_score,
            result=result,
            profit_loss_units=pnl,
        )
        if ok:
            log.info(
                f"Resolved {pick['pick_uid']}: {pick['away_team']}@{pick['home_team']} "
                f"{away_score}-{home_score}  [{result}]  PnL={pnl:+.4f}u"
            )
            # update daily snapshot for that game's date
            db.upsert_daily_snapshot(gdate)
            if result == "VOID":
                stats["voided"] += 1
            else:
                stats["resolved"] += 1
        else:
            log.debug(f"Already resolved or not found: {pick['pick_uid']}")
            stats["errors"] += 1

    return stats


def reconcile_all_sports(
    db: Optional[TrackRecordDB] = None,
    lookback_days: int = 7,
) -> Dict[str, Any]:
    if db is None:
        db = TrackRecordDB()
    results: Dict[str, Any] = {}
    for sport in ["MLB"]:  # expand as NBA/UFC modules mature
        results[sport] = reconcile_pending(db, lookback_days=lookback_days, sport=sport)
    return results
