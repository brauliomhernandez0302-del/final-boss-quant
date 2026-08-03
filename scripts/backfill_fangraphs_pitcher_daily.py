"""Resume/backfill fangraphs.pitcher.daily PIT cache rows for a season.

Unlike the Savant rolling builder, FanGraphsDailyPITPersistence.persist_cutoff()
is cheap per call (one HTTP request per cutoff, no local re-scan cost) — the
earlier full-season run wasn't slow, it just tripped a ConnectionError from
FanGraphs after ~139 rapid-fire requests. This script adds a delay between
calls and a one-shot retry so a full-season backfill survives that.

Usage:
  python3 scripts/backfill_fangraphs_pitcher_daily.py \
    --pit-cache-db data/pit_cache_pitcher.db \
    --season 2024 --season-start-date 2024-03-28 \
    --start-cutoff 2024-08-14 --end-cutoff 2024-09-29
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests

from modules.baseball_module.advanced_pit_enrichment.fangraphs_daily_pit_persistence import (
    FanGraphsDailyPITPersistence,
)


def _date_range(start: str, end: str):
    d = date.fromisoformat(start)
    e = date.fromisoformat(end)
    while d <= e:
        yield d.isoformat()
        d += timedelta(days=1)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pit-cache-db", required=True, type=Path)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--season-start-date", required=True)
    parser.add_argument("--start-cutoff", required=True)
    parser.add_argument("--end-cutoff", required=True)
    parser.add_argument("--delay-seconds", type=float, default=1.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fangraphs = FanGraphsDailyPITPersistence(pit_cache_db=args.pit_cache_db)

    cutoff_dates = list(_date_range(args.start_cutoff, args.end_cutoff))
    _log(f"FanGraphs daily backfill: {len(cutoff_dates)} cutoffs "
         f"{args.start_cutoff}..{args.end_cutoff}")

    for i, cutoff in enumerate(cutoff_dates, 1):
        rows = None
        for attempt in (1, 2):
            try:
                rows = fangraphs.persist_cutoff(
                    season=args.season,
                    season_start_date=args.season_start_date,
                    as_of_date=cutoff,
                )
                break
            except requests.exceptions.RequestException as exc:
                _log(f"  [{i}/{len(cutoff_dates)}] cutoff={cutoff} attempt={attempt} "
                     f"failed: {exc!r}")
                if attempt == 2:
                    raise
                time.sleep(5.0)

        _log(f"  [{i}/{len(cutoff_dates)}] cutoff={cutoff} pitchers={len(rows)}")
        time.sleep(args.delay_seconds)

    _log(f"Done: {len(cutoff_dates)} cutoffs persisted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
