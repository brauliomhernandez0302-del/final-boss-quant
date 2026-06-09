"""Populate isolated raw Savant event storage."""

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
    summary_to_dict,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate isolated raw Savant event cache.")
    parser.add_argument("--cache-db", required=True, type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = SavantRawIngestor(args.cache_db).ingest_date_range(
        start_date=args.start_date,
        end_date=args.end_date,
    )
    data = summary_to_dict(summary)
    print(
        f"provider=savant_raw window={args.start_date}..{args.end_date} "
        f"events_saved={data['events_saved']} "
        f"distinct_dates={data['distinct_dates']} "
        f"distinct_pitchers={data['distinct_pitchers']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
