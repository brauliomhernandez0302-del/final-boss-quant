"""track_record/clv.py — shared p_close computation for docs/PROTOCOLO_CLV_V1.md.

Single source of truth for turning a pick's closing-line snapshot into
p_close. scripts/clv_report.py (and anything else that ever needs p_close)
imports this instead of reimplementing devigging — the protocol is explicit
that remove_vig_multiplicative is the only devig function allowed ("cero
reimplementaciones — la lección de ODDS-001"), so this module only calls
into core.value_detector, never reimplements the math itself.

Callers pass a plain dict per pick (e.g. dict(sqlite3.Row) — sqlite3.Row
itself has no .get()) with at least: market, closing_pin_home,
closing_pin_away, closing_all_books_json.
"""
from __future__ import annotations

import json
from statistics import median
from typing import Any, Dict, Optional

from core.value_detector import remove_vig_multiplicative

_SIDE_INDEX = {"ML_HOME": 0, "ML_AWAY": 1}


def compute_p_close(pick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return {"p_close", "source", "n_books"} for a pick's closing snapshot,
    or None if there is no usable close at all — the protocol's "atrición"
    case (docs/PROTOCOLO_CLV_V1.md's Muestra primaria y D0 section). Callers
    must count a None as attrition, never substitute a guessed value.

    Scoped to ML_HOME/ML_AWAY only, per the protocol.

    source == "pinnacle": PRIMARY-metric-eligible — both sides of Pinnacle's
    own closing quote were captured, devigged together.

    source == "fallback_median": the protocol's pre-registered SECONDARY
    fallback only — median of each available book's own devigged
    probability for the pick's side, used when no Pinnacle close exists.
    Callers must never feed this into the primary metric.
    """
    market = pick.get("market")
    side_idx = _SIDE_INDEX.get(market)
    if side_idx is None:
        return None

    pin_home = pick.get("closing_pin_home")
    pin_away = pick.get("closing_pin_away")
    if pin_home and pin_away and pin_home > 1.0 and pin_away > 1.0:
        fair = remove_vig_multiplicative([pin_home, pin_away])
        return {"p_close": fair[side_idx], "source": "pinnacle", "n_books": 1}

    raw = pick.get("closing_all_books_json")
    if raw:
        try:
            books = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            books = []
        fair_probs = []
        for b in books or []:
            h, a = b.get("home"), b.get("away")
            if h and a and h > 1.0 and a > 1.0:
                fair = remove_vig_multiplicative([h, a])
                fair_probs.append(fair[side_idx])
        if fair_probs:
            return {
                "p_close": median(fair_probs),
                "source": "fallback_median",
                "n_books": len(fair_probs),
            }

    return None


def compute_ev_vs_close(pick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """EV-vs-close = p_close * O_taken - 1 (docs/PROTOCOLO_CLV_V1.md's
    Definiciones). Returns None (attrition) if either O_taken or a usable
    close is missing. O_taken is the pick's own odds_decimal — the price
    actually taken at publish time, regardless of which book gave it."""
    o_taken = pick.get("odds_decimal")
    if not o_taken or o_taken <= 1.0:
        return None
    close = compute_p_close(pick)
    if close is None:
        return None
    ev_vs_close_pct = (close["p_close"] * o_taken - 1.0) * 100.0
    return {
        "ev_vs_close_pct": ev_vs_close_pct,
        "p_close": close["p_close"],
        "source": close["source"],
        "n_books": close["n_books"],
    }
