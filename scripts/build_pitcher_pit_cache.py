"""Populate isolated point-in-time pitcher cache records.

This script intentionally stays disconnected from live and backtest pipelines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.baseball_module.advanced_pit_enrichment.fangraphs_pit_fetcher import (
    FanGraphsPITFetcher,
)
from modules.baseball_module.advanced_pit_enrichment.savant_pit_fetcher import (
    SavantPITFetcher,
)

Provider = str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate the isolated MLB PIT pitcher cache for a date window."
    )
    parser.add_argument("--cache-db", required=True, type=Path)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument(
        "--provider",
        choices=("all", "fangraphs", "savant"),
        default="all",
        help="PIT provider to populate. Defaults to all.",
    )
    return parser.parse_args(argv)


def build_pitcher_pit_cache(args: argparse.Namespace) -> list[tuple[Provider, int]]:
    providers = _selected_providers(args.provider)
    summaries: list[tuple[Provider, int]] = []

    for provider in providers:
        fetcher = _build_fetcher(provider, args.cache_db)
        rows = fetcher.fetch_pitcher_metrics_by_date_range(
            season=args.season,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        count = len(rows)
        summaries.append((provider, count))
        print(
            f"provider={provider} "
            f"window={args.start_date}..{args.end_date} "
            f"pitcher_count={count}"
        )

    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_pitcher_pit_cache(args)
    return 0


def _selected_providers(provider: Provider) -> tuple[Provider, ...]:
    if provider == "all":
        return ("fangraphs", "savant")
    return (provider,)


def _build_fetcher(provider: Provider, cache_db: Path):
    if provider == "fangraphs":
        return FanGraphsPITFetcher(cache_db)
    if provider == "savant":
        return SavantPITFetcher(cache_db)
    raise ValueError(f"unsupported provider: {provider}")


if __name__ == "__main__":
    raise SystemExit(main())
