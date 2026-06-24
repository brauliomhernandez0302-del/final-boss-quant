"""Build a durable, resumable historical raw Savant cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.baseball_module.advanced_pit_enrichment.savant_raw_ingestor import (  # noqa: E402
    SavantRawIngestor,
    historical_summary_to_dict,
)

DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "pit_raw"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a durable, resumable historical raw Savant event cache."
    )
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--cache-db", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cache_db = args.cache_db or DEFAULT_CACHE_ROOT / f"raw_savant_{args.season}.db"
    manifest = (
        args.manifest
        or DEFAULT_CACHE_ROOT / f"raw_savant_{args.season}.manifest.json"
    )
    summary = SavantRawIngestor(cache_db).ingest_historical_date_range(
        season=args.season,
        start_date=args.start_date,
        end_date=args.end_date,
        manifest_path=manifest,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    data = historical_summary_to_dict(summary)
    print(
        f"provider=savant_raw season={args.season} "
        f"window={args.start_date}..{args.end_date} "
        f"fetched_dates={len(data['fetched_dates'])} "
        f"skipped_dates={len(data['skipped_dates'])} "
        f"failed_dates={len(data['failed_dates'])} "
        f"total_events_saved={data['total_events_saved']} "
        f"distinct_game_dates={data['distinct_game_dates']} "
        f"database={data['database_path']} manifest={data['manifest_path']}"
    )
    return 1 if data["failed_dates"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
