"""Build a leakage-safe prior-season pitcher baseline in the PIT cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.baseball_module.advanced_pit_enrichment import (  # noqa: E402
    PitcherPriorBaselinePersistence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-savant-db", required=True, type=Path)
    parser.add_argument("--pit-cache-db", required=True, type=Path)
    parser.add_argument("--target-season", required=True, type=int)
    parser.add_argument("--prior-season", required=True, type=int)
    parser.add_argument("--prior-season-start-date", required=True)
    parser.add_argument("--prior-season-end-date", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = PitcherPriorBaselinePersistence(
        raw_cache_db=args.raw_savant_db,
        pit_cache_db=args.pit_cache_db,
    ).persist_prior_baseline(
        target_season=args.target_season,
        prior_season=args.prior_season,
        prior_season_start_date=args.prior_season_start_date,
        prior_season_end_date=args.prior_season_end_date,
    )
    print(
        f"pitcher_prior_baseline target_season={args.target_season} "
        f"prior_season={args.prior_season} rows={len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
