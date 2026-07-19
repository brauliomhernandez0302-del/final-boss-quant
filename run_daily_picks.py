#!/usr/bin/env python3
"""
run_daily_picks.py — CLI runner for the daily pick publication + reconciliation cycle.

Run this once a day, at least 30 minutes before the first game.

Usage:
    python run_daily_picks.py              # publish today's picks + reconcile yesterday
    python run_daily_picks.py --dry-run    # preview without writing to DB
    python run_daily_picks.py --reconcile-only
    python run_daily_picks.py --publish-only
    python run_daily_picks.py --lookback 14

Cron example (2 PM ET = 18:00 UTC):
    0 18 * * * cd /home/raulio && source mi_entorno/bin/activate && python run_daily_picks.py >> logs/daily_picks.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("daily_picks")


def _banner(title: str) -> None:
    w = 60
    log.info("=" * w)
    log.info(f"  {title}")
    log.info("=" * w)


def run_publish(db, dry_run: bool, sports: list[str]) -> list[dict]:
    from track_record.publisher import publish_daily_picks
    _banner("PUBLISH DAILY PICKS")
    picks = publish_daily_picks(db=db, sports=sports, dry_run=dry_run)
    log.info(f"Picks published: {len(picks)}")
    for p in picks:
        log.info(
            f"  {p['pick_uid']}  EV={p['ev_pct']:.2f}%  tier={p.get('confidence_tier')}  "
            f"stake={p.get('stake_units', 0):.2f}u"
        )
    return picks


def run_reconcile(db, lookback: int) -> dict:
    from track_record.reconciler import reconcile_all_sports
    _banner("RECONCILE PENDING PICKS")
    results = reconcile_all_sports(db=db, lookback_days=lookback)
    for sport, stats in results.items():
        log.info(
            f"{sport}: resolved={stats['resolved']}  voided={stats['voided']}  "
            f"skipped={stats['skipped']}  errors={stats['errors']}"
        )
    return results


def print_summary(db) -> None:
    from track_record.stats import compute_stats
    stats = compute_stats(db=db)
    hl = stats["headline"]
    _banner("TRACK RECORD SUMMARY")
    log.info(f"Total picks:  {hl['total_picks']}")
    log.info(f"Resolved:     {hl['resolved']}  ({hl['pending']} pending)")
    log.info(f"Record:       {hl['wins']}W - {hl['losses']}L - {hl['pushes']}P")
    log.info(f"Win Rate:     {hl['win_rate']:.1%}")
    log.info(f"ROI:          {hl['roi_pct']:+.2f}%")
    log.info(f"Units P&L:    {hl['total_units']:+.4f}u")
    log.info(f"Sharpe:       {hl['sharpe']:.3f}")
    log.info(f"Avg EV:       {hl['avg_ev_pct']:+.2f}%")


def export_json(db, out_path: Path) -> None:
    """Export all resolved picks to a JSON file for public sharing."""
    from track_record.stats import compute_stats
    stats = compute_stats(db=db)
    stats["generated_at"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(json.dumps(stats, indent=2, default=str))
    log.info(f"Exported track record to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily picks publisher + reconciler")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview picks without writing to DB")
    parser.add_argument("--publish-only", action="store_true",
                        help="Only publish picks, skip reconciliation")
    parser.add_argument("--reconcile-only", action="store_true",
                        help="Only reconcile pending, skip publishing")
    parser.add_argument("--lookback", type=int, default=7,
                        help="Days to look back for pending games (default: 7)")
    parser.add_argument("--sports", nargs="+", default=["MLB"],
                        help="Sports to process (default: MLB)")
    parser.add_argument("--export", type=Path, default=None,
                        help="Export track record stats to JSON file")
    parser.add_argument("--db", type=Path, default=None,
                        help="Override DB path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from track_record.db import TrackRecordDB, DB_PATH
    db_path = args.db or DB_PATH
    db = TrackRecordDB(db_path)

    _banner(f"FINAL BOSS QUANT — Daily Picks  {datetime.now():%Y-%m-%d %H:%M}")
    log.info(f"DB:      {db_path}")
    log.info(f"Sports:  {args.sports}")
    log.info(f"Dry run: {args.dry_run}")

    if not args.reconcile_only:
        run_publish(db, dry_run=args.dry_run, sports=args.sports)

    if not args.publish_only and not args.dry_run:
        run_reconcile(db, lookback=args.lookback)

    print_summary(db)

    if args.export:
        export_json(db, args.export)

    _banner("DONE")


if __name__ == "__main__":
    main()
