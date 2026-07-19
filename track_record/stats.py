"""
track_record/stats.py — compute track-record statistics from the DB.

Returns structured dicts ready to hand straight to the Streamlit UI.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from track_record.db import TrackRecordDB


def _safe_roi(pnl: float, staked: float) -> float:
    return round(pnl / staked * 100, 2) if staked > 0 else 0.0


def _sharpe(returns: List[float]) -> float:
    """Per-bet Sharpe on unit returns (mean/std)."""
    if len(returns) < 2:
        return 0.0
    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(var)
    return round(mean / std, 3) if std > 0 else 0.0


def compute_stats(
    db: Optional[TrackRecordDB] = None,
    sport: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full statistics pack for the Streamlit track record page.
    Returns a dict with keys:
      headline, by_sport, by_market, by_tier, by_month,
      bankroll_curve, recent_picks
    """
    if db is None:
        db = TrackRecordDB()

    picks = db.get_picks(sport=sport, limit=5000)
    resolved = [p for p in picks if p["result"] is not None]
    pending  = [p for p in picks if p["result"] is None]

    # ── headline ─────────────────────────────────────────────────────────────
    wins   = sum(1 for p in resolved if p["result"] == "WIN")
    losses = sum(1 for p in resolved if p["result"] == "LOSS")
    pushes = sum(1 for p in resolved if p["result"] == "PUSH")
    total_res = wins + losses + pushes
    total_pnl = sum(p["profit_loss_units"] or 0 for p in resolved)
    total_staked = sum(p["stake_units"] or 0 for p in resolved)
    avg_ev = (
        sum(p["ev_pct"] or 0 for p in picks) / len(picks) if picks else 0.0
    )
    returns = [p["profit_loss_units"] or 0 for p in resolved if p["stake_units"]]
    headline = {
        "total_picks": len(picks),
        "resolved":    total_res,
        "pending":     len(pending),
        "wins":        wins,
        "losses":      losses,
        "pushes":      pushes,
        "win_rate":    round(wins / (wins + losses), 4) if (wins + losses) > 0 else 0.0,
        "total_units": round(total_pnl, 4),
        "total_staked": round(total_staked, 4),
        "roi_pct":     _safe_roi(total_pnl, total_staked),
        "avg_ev_pct":  round(avg_ev, 4),
        "sharpe":      _sharpe(returns),
    }

    # ── by sport ─────────────────────────────────────────────────────────────
    by_sport: Dict[str, Dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "pushes": 0, "pnl": 0.0, "staked": 0.0}
    )
    for p in resolved:
        sp = p["sport"]
        by_sport[sp]["wins"]   += int(p["result"] == "WIN")
        by_sport[sp]["losses"] += int(p["result"] == "LOSS")
        by_sport[sp]["pushes"] += int(p["result"] == "PUSH")
        by_sport[sp]["pnl"]    += p["profit_loss_units"] or 0
        by_sport[sp]["staked"] += p["stake_units"] or 0
    for sp, d in by_sport.items():
        wl = d["wins"] + d["losses"]
        d["win_rate"] = round(d["wins"] / wl, 4) if wl > 0 else 0.0
        d["roi_pct"]  = _safe_roi(d["pnl"], d["staked"])

    # ── by market ────────────────────────────────────────────────────────────
    by_market: Dict[str, Dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "pushes": 0, "pnl": 0.0, "staked": 0.0}
    )
    for p in resolved:
        m = p["market"]
        by_market[m]["wins"]   += int(p["result"] == "WIN")
        by_market[m]["losses"] += int(p["result"] == "LOSS")
        by_market[m]["pushes"] += int(p["result"] == "PUSH")
        by_market[m]["pnl"]    += p["profit_loss_units"] or 0
        by_market[m]["staked"] += p["stake_units"] or 0
    for m, d in by_market.items():
        wl = d["wins"] + d["losses"]
        d["win_rate"] = round(d["wins"] / wl, 4) if wl > 0 else 0.0
        d["roi_pct"]  = _safe_roi(d["pnl"], d["staked"])

    # ── by confidence tier ───────────────────────────────────────────────────
    by_tier: Dict[str, Dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "staked": 0.0}
    )
    for p in resolved:
        t = (p["confidence_tier"] or "UNKNOWN").upper()
        by_tier[t]["wins"]   += int(p["result"] == "WIN")
        by_tier[t]["losses"] += int(p["result"] == "LOSS")
        by_tier[t]["pnl"]    += p["profit_loss_units"] or 0
        by_tier[t]["staked"] += p["stake_units"] or 0
    for t, d in by_tier.items():
        wl = d["wins"] + d["losses"]
        d["win_rate"] = round(d["wins"] / wl, 4) if wl > 0 else 0.0
        d["roi_pct"]  = _safe_roi(d["pnl"], d["staked"])

    # ── by month ─────────────────────────────────────────────────────────────
    by_month: Dict[str, Dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "staked": 0.0}
    )
    for p in resolved:
        month = (p["game_date"] or "")[:7]  # YYYY-MM
        if not month:
            continue
        by_month[month]["wins"]   += int(p["result"] == "WIN")
        by_month[month]["losses"] += int(p["result"] == "LOSS")
        by_month[month]["pnl"]    += p["profit_loss_units"] or 0
        by_month[month]["staked"] += p["stake_units"] or 0
    for m, d in by_month.items():
        wl = d["wins"] + d["losses"]
        d["win_rate"] = round(d["wins"] / wl, 4) if wl > 0 else 0.0
        d["roi_pct"]  = _safe_roi(d["pnl"], d["staked"])

    # ── bankroll curve ───────────────────────────────────────────────────────
    bk_history = db.get_bankroll_history()
    bankroll_curve = [
        {
            "date":          row["game_date"],
            "sport":         row["sport"],
            "market":        row["market"],
            "running_total": row["running_total"],
            "units_pnl":     row["units_pnl"],
        }
        for row in bk_history
    ]

    # ── recent picks (last 20 resolved + pending) ────────────────────────────
    recent = picks[:50]
    recent_picks = [
        {
            "published_at":   row["published_at"],
            "game_date":      row["game_date"],
            "sport":          row["sport"],
            "modo":           row["publish_mode"] if "publish_mode" in row.keys() else "quarantine",
            "matchup":        f"{row['away_team']} @ {row['home_team']}",
            "market":         row["market"],
            "model_prob":     row["model_prob"],
            "ev_pct":         row["ev_pct"],
            "tier":           row["confidence_tier"],
            "odds":           row["odds_decimal"],
            "stake":          row["stake_units"],
            "result":         row["result"],
            "score":          (
                f"{row['actual_away_score']}-{row['actual_home_score']}"
                if row["actual_home_score"] is not None else "—"
            ),
            "pnl":            row["profit_loss_units"],
        }
        for row in recent
    ]

    return {
        "headline":       headline,
        "by_sport":       dict(by_sport),
        "by_market":      dict(by_market),
        "by_tier":        dict(by_tier),
        "by_month":       dict(sorted(by_month.items())),
        "bankroll_curve": bankroll_curve,
        "recent_picks":   recent_picks,
    }
