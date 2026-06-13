"""Build canonical experimental pitcher PIT cache records.

This script intentionally writes only the namespaces consumed by
AdvancedPitcherDailySnapshotBuilder:

- fangraphs.pitcher.daily / fangraphs_daily_cutoff
- savant.pitcher.rolling / baseball_savant_rolling
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.baseball_module.advanced_pit_enrichment.fangraphs_daily_pit_persistence import (
    FanGraphsDailyPITPersistence,
)
from modules.baseball_module.advanced_pit_enrichment.savant_raw_ingestor import (
    SavantRawIngestor,
)
from modules.baseball_module.advanced_pit_enrichment.savant_rolling_pit_persistence import (
    SavantRollingPITPersistence,
)


@dataclass(frozen=True)
class CutoffPlan:
    game_date: str
    cutoff_date: str


@dataclass(frozen=True)
class BuildSummary:
    cutoff_date: str
    fangraphs_pitchers: int
    savant_pitchers: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical experimental MLB pitcher PIT cache records."
    )
    parser.add_argument("--pit-cache-db", required=True, type=Path)
    parser.add_argument("--raw-savant-db", required=True, type=Path)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--season-start-date", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--cutoff-policy",
        choices=("previous_day",),
        default="previous_day",
        help="Map game dates to PIT cutoffs. previous_day means D uses D-1.",
    )
    return parser.parse_args(argv)


def build_cutoff_plan(
    *,
    start_date: str,
    end_date: str,
    cutoff_policy: str,
) -> list[CutoffPlan]:
    if cutoff_policy != "previous_day":
        raise ValueError(f"unsupported cutoff policy: {cutoff_policy}")

    plans: list[CutoffPlan] = []
    for game_day in _date_range(start_date, end_date):
        cutoff_day = game_day - timedelta(days=1)
        plans.append(CutoffPlan(game_date=game_day.isoformat(), cutoff_date=cutoff_day.isoformat()))
    return plans


def build_experimental_pitcher_pit_cache(args: argparse.Namespace) -> list[BuildSummary]:
    plans = build_cutoff_plan(
        start_date=args.start_date,
        end_date=args.end_date,
        cutoff_policy=args.cutoff_policy,
    )
    cutoff_dates = sorted({plan.cutoff_date for plan in plans})
    if not cutoff_dates:
        return []

    raw_ingestor = SavantRawIngestor(args.raw_savant_db)
    raw_ingestor.ingest_date_range(start_date=args.season_start_date, end_date=cutoff_dates[-1])

    fangraphs = FanGraphsDailyPITPersistence(pit_cache_db=args.pit_cache_db)
    savant = SavantRollingPITPersistence(
        raw_cache_db=args.raw_savant_db,
        pit_cache_db=args.pit_cache_db,
    )

    summaries: list[BuildSummary] = []
    for cutoff_date in cutoff_dates:
        fg_rows = fangraphs.persist_cutoff(
            season=args.season,
            season_start_date=args.season_start_date,
            as_of_date=cutoff_date,
        )
        sv_rows = savant.persist_cutoff(
            season=args.season,
            season_start_date=args.season_start_date,
            as_of_date=cutoff_date,
        )
        summary = BuildSummary(
            cutoff_date=cutoff_date,
            fangraphs_pitchers=len(fg_rows),
            savant_pitchers=len(sv_rows),
        )
        summaries.append(summary)
        print(
            f"cutoff={summary.cutoff_date} "
            f"fangraphs_pitchers={summary.fangraphs_pitchers} "
            f"savant_pitchers={summary.savant_pitchers}"
        )

    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_experimental_pitcher_pit_cache(args)
    return 0


def _date_range(start_date: str, end_date: str) -> Iterable[date]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


if __name__ == "__main__":
    raise SystemExit(main())
