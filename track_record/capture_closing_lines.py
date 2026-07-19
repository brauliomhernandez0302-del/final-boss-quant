"""track_record/capture_closing_lines.py — real closing-line (CLV) capture.

2026-07-12: this is the fast skill signal the project didn't have. Every
prior "CLV" reference in this codebase (backtest_and_retrain.py's clv_home/
clv_away) is edge-as-ratio against a single Pinnacle snapshot, not a real
bet-time-vs-closing-time price pair — see feedback_clv_misnomer memory. This
script closes that gap for live picks: it sweeps picks whose game hasn't
started yet, fetches the current market via the existing
get_best_odds_for_teams() (reused, not reimplemented), and records the
closing Pinnacle price on the same side the pick was made. CLV converges to
a skill/no-skill answer in roughly 100 picks — an order of magnitude faster
than live ROI, whose confidence interval stays several points wide for
hundreds of picks (see project memory for the full reasoning).

Intended to run as a scheduled sweep, repeatedly, throughout the day (cron,
or any periodic runner) — NOT just once. "Last pre-start capture wins"
(2026-07-19, Fase 2A commit 3): every call re-captures and overwrites every
pick whose game hasn't started yet, so the LATEST sweep before first pitch
is what ends up recorded as the closing line — there's no need to time a
single sweep precisely, and no harm in running several. Once a pick's game
starts, get_picks_needing_closing_capture() stops returning it (its last
pre-start capture is final) and capture_closing_line() independently rejects
any attempt to capture at or after that point, as a second line of defense.

Picks with no commence_time (a sport/publisher that doesn't populate it) are
skipped WITH A WARNING, never captured anyway — there is no honest "closing"
concept without knowing when the game starts.

Usage:
    python3 -m track_record.capture_closing_lines
    python3 -m track_record.capture_closing_lines --game-date 2026-07-12
    python3 -m track_record.capture_closing_lines --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from track_record.db import TrackRecordDB

log = logging.getLogger(__name__)


def capture_closing_lines(
    db: TrackRecordDB,
    *,
    sport: str = "MLB",
    game_date: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Sweep pending closing-line captures and record them. Returns a
    summary dict for logging/scripting."""
    from odds_fetcher import get_best_odds_for_teams

    _ODDS_FETCHER_SPORT = {"MLB": "baseball_mlb"}

    picks = db.get_picks_needing_closing_capture(sport=sport, game_date=game_date)
    summary = {
        "considered": len(picks), "captured": 0, "no_market_data": 0,
        "skipped_no_commence_time": 0, "rejected_post_start": 0, "errors": 0,
    }

    for pick in picks:
        if not pick["commence_time"]:
            log.warning(
                "Skipping pick_uid=%s (%s @ %s): no commence_time — cannot "
                "know when this game starts, so there is no honest 'closing' "
                "line to capture for it",
                pick["pick_uid"], pick["away_team"], pick["home_team"],
            )
            summary["skipped_no_commence_time"] += 1
            continue

        odds_sport = _ODDS_FETCHER_SPORT.get(pick["sport"])
        if odds_sport is None:
            log.debug("Skipping pick_uid=%s: no odds-fetcher sport mapping for %s",
                      pick["pick_uid"], pick["sport"])
            continue
        try:
            market_odds = get_best_odds_for_teams(
                home_team=pick["home_team"], away_team=pick["away_team"],
                commence_time=pick["commence_time"], sport=odds_sport,
            ) or {}
        except Exception as exc:
            log.warning("get_best_odds_for_teams failed for pick_uid=%s: %s",
                        pick["pick_uid"], exc)
            summary["errors"] += 1
            continue

        pin_home = market_odds.get("pin_home")
        pin_away = market_odds.get("pin_away")
        # Closing "odds_decimal" for this specific pick's own side/market —
        # best-market price (not necessarily Pinnacle) on the matching side,
        # so a non-ML market still gets SOME closing reference recorded even
        # though clv_pct itself is only computed for ML_HOME/ML_AWAY (v1).
        side_closing_price = None
        if pick["market"] == "ML_HOME":
            side_closing_price = market_odds.get("ml_home")
        elif pick["market"] == "ML_AWAY":
            side_closing_price = market_odds.get("ml_away")

        if pin_home is None and pin_away is None and side_closing_price is None:
            log.debug("No market data yet for pick_uid=%s (%s @ %s) — will retry next sweep",
                      pick["pick_uid"], pick["away_team"], pick["home_team"])
            summary["no_market_data"] += 1
            continue

        if dry_run:
            log.info("[dry-run] would capture pick_uid=%s market=%s "
                      "closing_odds=%s pin_home=%s pin_away=%s",
                      pick["pick_uid"], pick["market"], side_closing_price, pin_home, pin_away)
            summary["captured"] += 1
            continue

        ok = db.capture_closing_line(
            pick["pick_uid"],
            closing_odds_decimal=side_closing_price,
            closing_pin_home=pin_home,
            closing_pin_away=pin_away,
        )
        if ok:
            summary["captured"] += 1
            log.info("Captured closing line: pick_uid=%s market=%s closing_odds=%s",
                      pick["pick_uid"], pick["market"], side_closing_price)
        else:
            # get_picks_needing_closing_capture() already filters to
            # not-yet-started games, so this should be rare — but a slow
            # sweep could still cross commence_time between query and here.
            summary["rejected_post_start"] += 1
            log.warning(
                "Rejected capture for pick_uid=%s: game appears to have "
                "started between the query and this capture attempt",
                pick["pick_uid"],
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", default="MLB")
    parser.add_argument("--game-date", default=None,
                         help="YYYY-MM-DD; default = all pending picks with no closing snapshot")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db = TrackRecordDB()
    summary = capture_closing_lines(
        db, sport=args.sport, game_date=args.game_date, dry_run=args.dry_run,
    )
    log.info("Closing-line sweep done: %s", summary)


if __name__ == "__main__":
    main()
