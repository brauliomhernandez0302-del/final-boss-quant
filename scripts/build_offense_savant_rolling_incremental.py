"""Efficient incremental backfill for savant.team_offense.rolling AND
savant.batter.rolling PIT cache rows, built together in one pass.

SavantTeamOffenseRollingPITPersistence.persist_cutoff() / build_teams_for_as_of_date()
re-scan and re-aggregate every raw event from season_start_date to as_of_date
on EVERY call — O(days^2) in total events scanned across a full-season
backfill. This is why the team-offense rolling build in data/pit_cache_merged.db
stopped partway through both seasons (92/187 cutoff dates for 2024, 60/186 for
2025) — same class of stall already fixed for the pitcher domain via
scripts/build_pitcher_savant_rolling_incremental.py.

This script produces byte-for-byte equivalent pit_metric_cache rows (same
namespace/source/payload shape as SavantTeamOffenseRollingPITPersistence /
SavantBatterRollingPITPersistence) but computes them in O(days) total:
aggregate each day's raw events exactly once via SavantOffenseDailyAggregator
(team AND batter grouping, same day's events read once each), fold into
running per-team and per-batter accumulators, and persist both cumulative
snapshots after each day.

Usage:
  python3 scripts/build_offense_savant_rolling_incremental.py \
    --pit-cache-db data/pit_cache_merged.db \
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
from modules.baseball_module.advanced_pit_enrichment.savant_offense_daily_aggregator import (
    METRIC_VERSION as OFFENSE_METRIC_VERSION,
    SavantOffenseDailyAggregator,
    SavantBatterRollingPITPersistence,
    SavantTeamOffenseRollingPITPersistence,
    _Accumulator,
    _rolling_payload,
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
    parser.add_argument("--skip-batters", action="store_true",
                         help="Only build team_offense.rolling, skip batter.rolling")
    return parser.parse_args(argv)


def _cheap_fingerprint(*, entity: str, season: int, season_start_date: str,
                        as_of_date: str, entity_count: int, daily_row_count: int) -> str:
    payload = {
        "metric_version": OFFENSE_METRIC_VERSION,
        "entity": entity,
        "season": season,
        "season_start_date": season_start_date,
        "source_window_end_date": as_of_date,
        "entity_count": entity_count,
        "daily_row_count": daily_row_count,
        "builder": "incremental_v1",
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"savant:raw:offense:{entity}:incremental:{season}:{as_of_date}:{digest}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    raw_cache = RawSavantEventsCache(args.raw_savant_db)
    daily_aggregator = SavantOffenseDailyAggregator(cache=raw_cache)
    pit_cache = PITCache(args.pit_cache_db)

    cutoff_dates = list(_date_range(args.season_start_date, args.end_date))
    _log(f"Incremental Savant offense rolling (team{'   ' if args.skip_batters else ' + batter'}): "
         f"{len(cutoff_dates)} days {args.season_start_date}..{args.end_date}")

    team_accumulators: dict[str, _Accumulator] = {}
    batter_accumulators: dict[int, _Accumulator] = {}
    t0 = time.perf_counter()

    for i, day in enumerate(cutoff_dates, 1):
        # Single-day window only — O(that day's events), not cumulative.
        team_daily = daily_aggregator.aggregate_teams_by_date_range(start_date=day, end_date=day)
        for row in team_daily.rows:
            team_accumulators.setdefault(row.batting_team, _Accumulator()).add(row)

        if not args.skip_batters:
            batter_daily = daily_aggregator.aggregate_batters_by_date_range(start_date=day, end_date=day)
            for row in batter_daily:
                batter_accumulators.setdefault(row.batter, _Accumulator()).add(row)

        fetched_at = datetime.now(timezone.utc).isoformat()
        as_of_date = f"{day}T00:00:00Z"

        team_fp = _cheap_fingerprint(
            entity="team", season=args.season, season_start_date=args.season_start_date,
            as_of_date=day, entity_count=len(team_accumulators), daily_row_count=len(team_daily.rows),
        )
        for team, accumulator in team_accumulators.items():
            metrics = accumulator.to_team_metrics(as_of_date=day, batting_team=team)
            data = _rolling_payload(
                metrics,
                source_window_start_date=args.season_start_date,
                source_window_end_date=day,
                metric_version=OFFENSE_METRIC_VERSION,
            )
            data["missing_batting_team_rows"] = team_daily.missing_team_rows
            pit_cache.save_record(
                namespace=SavantTeamOffenseRollingPITPersistence.NAMESPACE,
                entity_id=team,
                season=args.season,
                as_of_date=as_of_date,
                source=SavantTeamOffenseRollingPITPersistence.SOURCE,
                source_fingerprint=team_fp,
                data=data,
                fetched_at=fetched_at,
            )

        if not args.skip_batters:
            batter_fp = _cheap_fingerprint(
                entity="batter", season=args.season, season_start_date=args.season_start_date,
                as_of_date=day, entity_count=len(batter_accumulators), daily_row_count=len(batter_daily),
            )
            for batter, accumulator in batter_accumulators.items():
                metrics = accumulator.to_batter_metrics(as_of_date=day, batter=batter)
                data = _rolling_payload(
                    metrics,
                    source_window_start_date=args.season_start_date,
                    source_window_end_date=day,
                    metric_version=OFFENSE_METRIC_VERSION,
                )
                pit_cache.save_record(
                    namespace=SavantBatterRollingPITPersistence.NAMESPACE,
                    entity_id=batter,
                    season=args.season,
                    as_of_date=as_of_date,
                    source=SavantBatterRollingPITPersistence.SOURCE,
                    source_fingerprint=batter_fp,
                    data=data,
                    fetched_at=fetched_at,
                )

        if i % 10 == 0 or i == len(cutoff_dates):
            elapsed = time.perf_counter() - t0
            _log(f"  [{i}/{len(cutoff_dates)}] date={day} "
                 f"teams={len(team_accumulators)} batters={len(batter_accumulators)} "
                 f"elapsed={elapsed:.1f}s")

    _log(f"Done in {time.perf_counter() - t0:.1f}s | "
         f"{len(team_accumulators)} teams, {len(batter_accumulators)} batters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
