"""
track_record/publisher.py — pre-game pick publication.

Runs the full MLB / NBA / UFC analysis pipeline for all today's games,
then saves every bet that meets the EV/tier threshold to the track record
DB with a published_at timestamp that proves the pick was made before
the game started.

Only publishes if published_at < game_commence_time - min_lead_minutes.
"""

from __future__ import annotations

import functools
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from track_record.db import TrackRecordDB
from core.value_detector import kelly_criterion

log = logging.getLogger("track_record.publisher")


@functools.lru_cache(maxsize=1)
def _get_engine_commit() -> Optional[str]:
    """The prediction engine's git HEAD, stamped onto every published pick
    (docs/PROTOCOLO_CLV_V1.md's engine-freeze validity condition needs this
    to detect a mid-window engine change). Cached — one subprocess call per
    process, not one per pick."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        log.warning("Could not resolve engine_commit (git rev-parse HEAD failed)")
        return None

# Minimum minutes before first pitch that we must publish
MIN_LEAD_MINUTES = 30

# Only publish picks at or above this tier (SLIGHT|MEDIUM|HIGH|ULTRA) — NOT
# enforced while config.QUARANTINE_MODE is true (Fase 2A commit 4): every
# pick the pipeline generates publishes to the quarantine ledger, extreme
# EVs included, so they're observable with real CLV instead of being
# silently discarded before the public band (Fase 2C) exists to judge them.
MIN_TIER = "SLIGHT"

_TIER_RANK = {"SLIGHT": 1, "MEDIUM": 2, "HIGH": 3, "ULTRA": 4}


def _tier_ok(tier: Optional[str]) -> bool:
    if tier is None:
        return True  # include if tier unknown
    return _tier_rank(tier) >= _tier_rank(MIN_TIER)


def _tier_rank(tier: str) -> int:
    t = (tier or "").upper()
    for name, rank in _TIER_RANK.items():
        if name in t:
            return rank
    return 0


def _american_to_decimal(american: float) -> float:
    if american >= 100:
        return round(american / 100 + 1, 4)
    return round(100 / abs(american) + 1, 4)


def _market_label(bet: Dict[str, Any]) -> str:
    """Normalise the 'market' or 'bet_type' field from value_detector output.

    F5 markets MUST be checked before the generic OVER/UNDER/HOME/AWAY
    patterns: value_detector's real F5 market-name strings are "F5 ML HOME"/
    "F5 ML AWAY" (moneyline) and "F5 OVER {line}"/"F5 UNDER {line}" (totals)
    (see core/value_detector.py::analyze_first5). A plain substring-mapping
    checked in the wrong order let "F5 OVER 4.5" match the generic "OVER"
    key (since "OVER" IS a substring of it) before ever reaching an F5-aware
    check — colliding pick_uid with a real full-game Over on the same game
    (one silently dropped via INSERT OR IGNORE) and causing the reconciler
    to grade it against the full-game score/line instead of the first-5-
    innings score/line. "F5 ML HOME"/"F5 ML AWAY" fared even worse: the old
    mapping's "F5 HOME"/"F5 AWAY" keys aren't contiguous substrings of them
    (the "ML " token breaks the match), so they fell through entirely,
    were stored as the raw unmapped string, and reconciler.py's
    _resolve_market() — which only recognizes ML_HOME/ML_AWAY/RL_HOME/
    RL_AWAY/OVER+F5_OVER/UNDER+F5_UNDER/F5_HOME/F5_AWAY — resolved every
    single one as VOID. Found + fixed 2026-07-06.
    """
    raw = (
        bet.get("market")
        or bet.get("bet_type")
        or bet.get("type")
        or ""
    ).upper()

    if raw.startswith("F5"):
        if "OVER" in raw:
            return "F5_OVER"
        if "UNDER" in raw:
            return "F5_UNDER"
        if "AWAY" in raw:
            return "F5_AWAY"
        if "HOME" in raw:
            return "F5_HOME"
        return raw or "F5_HOME"

    # value_detector uses labels like 'ML_HOME', 'MONEYLINE HOME', 'OVER', etc.
    mapping = {
        "ML_HOME": "ML_HOME", "MONEYLINE HOME": "ML_HOME",
        "ML_AWAY": "ML_AWAY", "MONEYLINE AWAY": "ML_AWAY",
        "RL_HOME": "RL_HOME", "RUNLINE HOME": "RL_HOME",
        "RL_AWAY": "RL_AWAY", "RUNLINE AWAY": "RL_AWAY",
        "OVER": "OVER", "O/U OVER": "OVER", "TOTALS OVER": "OVER",
        "UNDER": "UNDER", "O/U UNDER": "UNDER", "TOTALS UNDER": "UNDER",
    }
    for k, v in mapping.items():
        if k in raw:
            return v
    return raw or "ML_HOME"


def publish_mlb_picks(
    db: TrackRecordDB,
    games: Optional[List[Dict]] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run the MLB pipeline for all today's games and publish qualifying picks.
    Returns list of published pick dicts.
    """
    try:
        from data_fetchers import MLBDataIntegrator
        from modules.baseball_module.core.run_module import run_module as run_mlb
        from odds_fetcher import get_best_odds_for_teams
    except ImportError as e:
        log.error(f"MLB import error: {e}")
        return []

    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    if games is None:
        try:
            integrator = MLBDataIntegrator()
            games = integrator.mlb_api.get_todays_games(date=today) or []
            games += integrator.mlb_api.get_todays_games(date=tomorrow) or []
        except Exception as e:
            log.error(f"Failed to fetch MLB schedule: {e}")
            return []

    published: List[Dict[str, Any]] = []
    now_utc = datetime.now(timezone.utc)

    for game in games:
        game_pk = game.get("game_pk")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        # Prefer official_date (MLB's own schedule-day field) over
        # game_date[:10] (a raw UTC start-time timestamp) — a late-night
        # start's UTC date can be one day ahead of the schedule day
        # get_todays_games()/get_todays_games(tomorrow) actually query by,
        # which caused pick_uid/game_date to disagree with the day
        # run_module()'s own internal game lookup resolved the game under.
        game_date = str(game.get("official_date") or game.get("game_date", today))[:10]
        # 2026-07-19 fix: this was reading commence_time/game_datetime, keys
        # data_fetchers.py's _parse_game() never populates for MLB (only
        # game_date, the full ISO start-time timestamp) — so commence_raw
        # was ALWAYS "" and the entire MIN_LEAD_MINUTES pre-game-lead check
        # below silently never fired for any real MLB game, undermining this
        # module's own core guarantee ("every pick has a pre-game
        # published_at timestamp", per its module docstring). game_date is
        # kept last in the fallback chain so a future/other-sport caller
        # that DOES populate commence_time/game_datetime is unaffected.
        commence_raw = game.get("commence_time") or game.get("game_datetime") or game.get("game_date") or ""

        # --- enforce pre-game lead ---
        if commence_raw:
            try:
                if commence_raw.endswith("Z"):
                    commence_raw = commence_raw[:-1] + "+00:00"
                commence_dt = datetime.fromisoformat(commence_raw)
                if commence_dt.tzinfo is None:
                    commence_dt = commence_dt.replace(tzinfo=timezone.utc)
                lead = (commence_dt - now_utc).total_seconds() / 60
                if lead < MIN_LEAD_MINUTES:
                    log.info(
                        f"Skipping {away_team}@{home_team} — only {lead:.0f}m before game"
                    )
                    continue
            except Exception:
                pass  # can't parse time → proceed

        log.info(f"Analyzing {away_team} @ {home_team} (pk={game_pk})")

        try:
            result = run_mlb(
                game_id=game_pk,
                use_hfa=True,
                use_pitcher=True,
                analyze_f5=True,
                # 2026-07-19 fix: this call never passed `persist` at all, so
                # it always used run_module()'s default persist=True — a
                # publisher-level --dry-run never actually stopped the
                # underlying pipeline call from writing to the live
                # game_outcomes ledger (source='live'). Confirmed live: it
                # contaminated the ledger again with a real dry-run batch
                # after commit 1 shipped persist=False, since nothing here
                # ever passed it through. dry_run=True now means what it
                # says at every layer, not just track_record.db's own write.
                persist=not dry_run,
            )
        except Exception as e:
            log.warning(f"  Pipeline error for game {game_pk}: {e}")
            continue

        if result.get("status") == "error":
            log.warning(f"  Pipeline returned error: {result.get('error')}")
            continue

        probs = result.get("probabilities", {})
        best_bets = result.get("best_bets") or []
        p_home = probs.get("p_home") or probs.get("home_win") or 0.5
        p_away = probs.get("p_away") or probs.get("away_win") or 0.5

        # Get current market odds for decimal/implied prob columns
        market_odds: Dict[str, Any] = {}
        try:
            market_odds = get_best_odds_for_teams(
                home_team=home_team, away_team=away_team,
                commence_time=commence_raw, sport="baseball_mlb"
            ) or {}
        except Exception:
            pass

        if not best_bets:
            # If value_detector produced nothing, synthesise a minimal ML pick.
            # ml_home_odds comes from get_best_odds_for_teams() (odds_fetcher.py
            # requests oddsFormat=decimal), so it's ALWAYS decimal already —
            # unlike odds_raw below (line ~244), which may come from a bet
            # dict of unknown provenance and genuinely needs the >=100 guard.
            # Running it through _american_to_decimal() unconditionally here
            # treated a real decimal price (e.g. 1.91) as if it were American,
            # producing dec=53.36 and a wildly inflated fake EV.
            ml_home_odds = market_odds.get("ml_home")
            if ml_home_odds and p_home > 0.5:
                dec = float(ml_home_odds)
                ev = (p_home * (dec - 1)) - (1 - p_home)
                if ev > 0.02:
                    best_bets = [{
                        "market": "ML_HOME",
                        "model_prob": p_home,
                        "ev_pct": ev,
                        # Same fraction/clip (MIN_KELLY/MAX_KELLY) as the main
                        # pipeline's best_bets, so a synthesised fallback pick
                        # never bypasses the risk cap.
                        "kelly_fraction": kelly_criterion(p_home, dec, fractional=0.25),
                        "confidence_tier": "SLIGHT",
                        "odds": ml_home_odds,
                    }]

        for bet in best_bets:
            market = _market_label(bet)
            tier = (bet.get("confidence_tier") or bet.get("tier") or "")
            if not config.QUARANTINE_MODE and not _tier_ok(tier):
                continue

            # pick_uid is deterministic so re-runs are idempotent
            pick_uid = f"MLB:{game_pk}:{market}:{game_date}"

            # 'probability' is the real key name from
            # analyze_market_generic() (core/value_detector.py); 'model_prob'/
            # 'prob' are legacy aliases kept in case a caller passes a
            # differently-shaped bet dict (e.g. the synthesised ML fallback
            # a few lines up, which does use 'model_prob'). Checking
            # 'probability' FIRST matters now that all_opportunities spreads
            # the real bet dict (2026-07-06) — without it, every real
            # pipeline pick silently fell through to the generic p_home/
            # p_away guess instead of its own per-market probability.
            model_prob = float(
                bet.get("probability")
                or bet.get("model_prob")
                or bet.get("prob")
                or (p_home if "HOME" in market else p_away)
            )
            ev_pct = float(bet.get("ev_pct") or bet.get("ev") or 0.0)
            odds_raw = bet.get("odds") or bet.get("odds_decimal")
            odds_dec = (
                _american_to_decimal(odds_raw)
                if odds_raw and abs(odds_raw) >= 100
                else (float(odds_raw) if odds_raw else None)
            )
            implied = round(1 / odds_dec, 4) if odds_dec else None
            # docs/PROTOCOLO_CLV_V1.md cut (c) needs to know which book gave
            # O_taken. Only trust an EXACT match against
            # get_best_odds_for_teams()'s own ml_home_book/ml_away_book —
            # odds_dec may come from a differently-sourced bet dict (e.g. the
            # synthesised fallback above, or a future odds source), and
            # guessing a book for a price we didn't verify came from it would
            # fabricate a plausible-looking value (see FALL-001/FALL-002).
            odds_book = None
            if market == "ML_HOME" and odds_dec is not None:
                ref = market_odds.get("ml_home")
                if ref is not None and abs(odds_dec - float(ref)) < 1e-9:
                    odds_book = market_odds.get("ml_home_book")
            elif market == "ML_AWAY" and odds_dec is not None:
                ref = market_odds.get("ml_away")
                if ref is not None and abs(odds_dec - float(ref)) < 1e-9:
                    odds_book = market_odds.get("ml_away_book")
            # Signed run-line point for THIS pick's own side (e.g. -1.5 if
            # this side is favored, +1.5 if it's the underdog) — needed by
            # reconciler.py to grade RL_HOME/RL_AWAY correctly regardless of
            # which team is actually favored. NULL for any other market.
            runline_point = None
            if market == "RL_HOME":
                runline_point = market_odds.get("runline_home_point")
            elif market == "RL_AWAY":
                runline_point = market_odds.get("runline_away_point")
            kelly = float(bet.get("kelly_fraction") or bet.get("kelly") or 0.0)
            stake = round(kelly * 100, 4)  # in units (100-unit bankroll)
            total_line = (
                float(bet.get("line") or bet.get("total_line") or 0) or
                (float(market_odds.get("total_line") or 0) or None)
            ) if market in ("OVER", "UNDER", "F5_OVER", "F5_UNDER") else None

            # probabilities carries raw Monte Carlo sample arrays
            # (home_samples/away_samples/total_samples, size n_sims each —
            # confirmed live 2026-07-06) which are neither JSON-serializable
            # nor useful in an audit snapshot; every publish with a real
            # bet would otherwise crash at json.dumps() below with
            # "TypeError: Object of type ndarray is not JSON serializable",
            # unguarded by any try/except in this loop, before the pick was
            # ever saved.
            mc_probs = {
                k: v for k, v in (result.get("probabilities") or {}).items()
                if not k.endswith("_samples")
            }
            pipeline_snap = {
                "lambdas": result.get("lambdas_history", {}),
                "mc_probs": mc_probs,
                "bet": bet,
            }

            if dry_run:
                log.info(
                    f"  [DRY RUN] {pick_uid}  EV={ev_pct:.2f}%  tier={tier}"
                )
            else:
                row_id = db.publish_pick(
                    pick_uid=pick_uid,
                    game_date=game_date,
                    sport="MLB",
                    game_pk=game_pk,
                    home_team=home_team,
                    away_team=away_team,
                    market=market,
                    model_prob=model_prob,
                    ev_pct=ev_pct,
                    implied_prob=implied,
                    kelly_fraction=kelly,
                    confidence_tier=tier or None,
                    odds_decimal=odds_dec,
                    stake_units=stake,
                    pipeline_json=json.dumps(pipeline_snap),
                    total_line=total_line,
                    commence_time=commence_raw or None,
                    publish_mode="quarantine" if config.QUARANTINE_MODE else "public",
                    engine_commit=_get_engine_commit(),
                    odds_book=odds_book,
                    runline_point=runline_point,
                )
                if row_id:
                    log.info(
                        f"  Published #{row_id}: {pick_uid}  EV={ev_pct:.2f}%  tier={tier}"
                    )
                else:
                    log.debug(f"  Already published: {pick_uid}")

            published.append({
                "pick_uid": pick_uid,
                "sport": "MLB",
                "game_pk": game_pk,
                "home_team": home_team,
                "away_team": away_team,
                "game_date": game_date,
                "market": market,
                "model_prob": model_prob,
                "ev_pct": ev_pct,
                "confidence_tier": tier,
                "odds_decimal": odds_dec,
                "stake_units": stake,
            })

    return published


def publish_daily_picks(
    db: Optional[TrackRecordDB] = None,
    sports: Optional[List[str]] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Top-level entry: publish picks for all enabled sports.
    Returns combined list of all published picks.
    """
    if db is None:
        db = TrackRecordDB()
    if sports is None:
        sports = ["MLB"]  # NBA / UFC can be added as modules mature

    all_picks: List[Dict[str, Any]] = []

    if "MLB" in sports:
        log.info("Publishing MLB picks...")
        all_picks += publish_mlb_picks(db, dry_run=dry_run)

    log.info(f"Total picks published this run: {len(all_picks)}")
    return all_picks
