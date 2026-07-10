"""Efficient incremental backfill for savant.pitcher.rolling PIT cache rows.

SavantRollingPITPersistence.persist_cutoff() / SavantRollingPITBuilder.build_for_as_of_date()
re-scan and re-aggregate every raw event from season_start_date to as_of_date
on EVERY call. Calling it once per day across a full season is O(days^2) in
total events scanned — the season_start-through-day-180 cutoff re-reads and
re-aggregates the same 180 days of events that day-179's cutoff already did.
On a full MLB season this stalls out (observed: >20 minutes stuck on a single
mid-season cutoff, memory climbing past 1.3GB).

This script produces byte-for-byte equivalent pit_metric_cache rows (same
namespace/source/payload shape as SavantRollingPITPersistence) but computes
them in O(days) total: aggregate each day's raw events exactly once via
SavantDailyAggregator, fold into a running per-pitcher accumulator, and
persist the cumulative snapshot after each day.

Usage:
  python3 scripts/build_pitcher_savant_rolling_incremental.py \
    --pit-cache-db data/pit_cache_pitcher.db \
    --raw-savant-db data/pit_raw/raw_savant_2024.db \
    --season 2024 --season-start-date 2024-03-28 --end-date 2024-09-30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.baseball_module.advanced_pit_enrichment.pit_cache import PITCache
from modules.baseball_module.advanced_pit_enrichment.raw_savant_events_cache import (
    RawSavantEventsCache,
)
from modules.baseball_module.advanced_pit_enrichment.savant_daily_aggregator import (
    SavantDailyAggregator,
)
from modules.baseball_module.advanced_pit_enrichment.savant_rolling_pit_builder import (
    _Accumulator,
)
from modules.baseball_module.advanced_pit_enrichment.savant_rolling_pit_persistence import (
    SavantRollingPITPersistence,
    _metrics_to_payload,
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
    parser.add_argument("--raw-savant-db", required=True, type=Path)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--season-start-date", required=True)
    parser.add_argument("--end-date", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    raw_cache = RawSavantEventsCache(args.raw_savant_db)
    daily_aggregator = SavantDailyAggregator(cache=raw_cache)
    pit_cache = PITCache(args.pit_cache_db)

    cutoff_dates = list(_date_range(args.season_start_date, args.end_date))
    _log(f"Incremental Savant pitcher rolling: {len(cutoff_dates)} days "
         f"{args.season_start_date}..{args.end_date}")

    accumulators: dict[int, _Accumulator] = {}
    t0 = time.perf_counter()

    for i, day in enumerate(cutoff_dates, 1):
        # Single-day window only — O(that day's events), not cumulative.
        daily_rows = daily_aggregator.aggregate_by_date_range(start_date=day, end_date=day)
        for row in daily_rows:
            accumulators.setdefault(row.pitcher, _Accumulator()).add(row)

        fetched_at = datetime.now(timezone.utc).isoformat()
        as_of_date = f"{day}T00:00:00Z"
        fingerprint = _cheap_fingerprint(
            season=args.season,
            season_start_date=args.season_start_date,
            as_of_date=day,
            pitcher_count=len(accumulators),
            daily_row_count=len(daily_rows),
        )

        for pitcher, accumulator in accumulators.items():
            metrics = accumulator.to_metrics(as_of_date=day, pitcher=pitcher)
            data = _metrics_to_payload(
                metrics,
                source_window_start_date=args.season_start_date,
                source_window_end_date=day,
                metric_version=SavantRollingPITPersistence.METRIC_VERSION,
            )
            pit_cache.save_record(
                namespace=SavantRollingPITPersistence.NAMESPACE,
                entity_id=pitcher,
                season=args.season,
                as_of_date=as_of_date,
                source=SavantRollingPITPersistence.SOURCE,
                source_fingerprint=fingerprint,
                data=data,
                fetched_at=fetched_at,
            )

        if i % 10 == 0 or i == len(cutoff_dates):
            elapsed = time.perf_counter() - t0
            _log(f"  [{i}/{len(cutoff_dates)}] date={day} "
                 f"pitchers_seen={len(accumulators)} daily_rows={len(daily_rows)} "
                 f"elapsed={elapsed:.1f}s")

    _log(f"Done in {time.perf_counter() - t0:.1f}s | {len(accumulators)} pitchers total")
    return 0


def _cheap_fingerprint(
    *, season: int, season_start_date: str, as_of_date: str, pitcher_count: int, daily_row_count: int,
) -> str:
    payload = {
        "metric_version": SavantRollingPITPersistence.METRIC_VERSION,
        "season": season,
        "season_start_date": season_start_date,
        "source_window_end_date": as_of_date,
        "pitcher_count": pitcher_count,
        "daily_row_count": daily_row_count,
        "builder": "incremental_v1",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"savant:raw:rolling:incremental:{season}:{as_of_date}:{digest}"


if __name__ == "__main__":
    raise SystemExit(main())
