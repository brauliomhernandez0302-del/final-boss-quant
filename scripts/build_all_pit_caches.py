"""Build all PIT caches (TTE + Defense + Bullpen) for a target season.

Usage:
  python3 scripts/build_all_pit_caches.py --target-season 2024
  python3 scripts/build_all_pit_caches.py --target-season 2025

Requires:
  data/pit_raw/raw_savant_{prior_season}.db   -- complete prior season
  data/pit_raw/raw_savant_{target_season}.db  -- complete current season

Outputs:
  data/pit_cache_{target_season}.db           -- shared PIT cache
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

from modules.baseball_module.advanced_pit_enrichment import (
    PITCache,
    SavantOffenseRollingBuilder,
    TTEPITNamespaces,
    TTEPITSources,
    TTEPriorBaselineBuilder,
)
from modules.baseball_module.advanced_pit_enrichment.raw_savant_events_cache import (
    RawSavantEventsCache,
)
from modules.baseball_module.advanced_pit_enrichment.savant_offense_daily_aggregator import (
    METRIC_VERSION as TTE_METRIC_VERSION,
    _inputs_fingerprint as _tte_inputs_fingerprint,
    _rolling_payload as _tte_rolling_payload,
)
from modules.baseball_module.advanced_pit_enrichment.team_defense_pit_builder import (
    TeamDefensePITBuilder,
)
from modules.baseball_module.advanced_pit_enrichment.team_defense_prior_baseline import (
    TeamDefensePriorBaseline,
)
from modules.baseball_module.advanced_pit_enrichment.bullpen_pit_builder import (
    TeamBullpenPITBuilder,
)
from modules.baseball_module.advanced_pit_enrichment.bullpen_prior_baseline import (
    BullpenPriorBaselineBuilder,
)

SEASON_WINDOWS = {
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-30"),
    2025: ("2025-03-27", "2025-09-28"),
}

NON_MLB_TEAM_IDS = frozenset({"AL", "NL"})


def _date_range(start: str, end: str):
    d = date.fromisoformat(start)
    e = date.fromisoformat(end)
    while d <= e:
        yield d.isoformat()
        d += timedelta(days=1)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_tte_prior(
    *,
    raw_2023: Path,
    pit_cache: PITCache,
    target_season: int,
    prior_season: int,
) -> None:
    prior_start, prior_end = SEASON_WINDOWS[prior_season]
    _log(f"TTE prior baseline: season={target_season} prior={prior_season} window={prior_start}..{prior_end}")
    result = TTEPriorBaselineBuilder(
        raw_cache_db=raw_2023,
        pit_cache=pit_cache,
    ).persist_prior_baseline(
        season=target_season,
        prior_season=prior_season,
        prior_season_start_date=prior_start,
        prior_season_end_date=prior_end,
    )
    _log(f"  TTE prior done: {len(result.rows)} teams")


def build_tte_rolling(
    *,
    raw_current: Path,
    pit_cache: PITCache,
    target_season: int,
    season_start_date: str,
    season_end_date: str,
) -> None:
    raw_cache = RawSavantEventsCache(raw_current)
    builder = SavantOffenseRollingBuilder(cache=raw_cache)
    from datetime import datetime, timezone
    cutoff_dates = list(_date_range(season_start_date, season_end_date))
    _log(f"TTE rolling: {len(cutoff_dates)} cutoff dates from {season_start_date} to {season_end_date}")

    for i, cutoff_date in enumerate(cutoff_dates, 1):
        result = builder.build_teams_for_as_of_date(
            season_start_date=season_start_date,
            as_of_date=cutoff_date,
        )
        fingerprint = _tte_inputs_fingerprint(
            raw_cache=raw_cache,
            season=target_season,
            season_start_date=season_start_date,
            as_of_date=cutoff_date,
            metric_version=TTE_METRIC_VERSION,
            entity="team",
        )
        fetched_at = datetime.now(timezone.utc).isoformat()
        saved = 0
        for team_id, metrics in result.rows.items():
            if str(team_id) in NON_MLB_TEAM_IDS:
                continue
            data = _tte_rolling_payload(
                metrics,
                source_window_start_date=season_start_date,
                source_window_end_date=cutoff_date,
                metric_version=TTE_METRIC_VERSION,
            )
            data["missing_batting_team_rows"] = result.missing_team_rows
            pit_cache.save_record(
                namespace=TTEPITNamespaces.TEAM_OFFENSE_ROLLING,
                entity_id=team_id,
                season=target_season,
                as_of_date=f"{cutoff_date}T00:00:00Z",
                source=TTEPITSources.TEAM_OFFENSE_ROLLING,
                source_fingerprint=fingerprint,
                data=data,
                fetched_at=fetched_at,
            )
            saved += 1

        if i % 30 == 0 or i == len(cutoff_dates):
            _log(f"  TTE rolling [{i}/{len(cutoff_dates)}] cutoff={cutoff_date} teams={saved}")


def build_defense_prior(
    *,
    raw_2023: Path,
    pit_cache: PITCache,
    target_season: int,
    prior_season: int,
) -> None:
    prior_start, prior_end = SEASON_WINDOWS[prior_season]
    _log(f"Defense prior baseline: season={target_season} prior={prior_season}")
    result = TeamDefensePriorBaseline(
        raw_cache_db=raw_2023,
        pit_cache=pit_cache,
    ).persist_prior_baseline(
        season=target_season,
        prior_season=prior_season,
        prior_season_start_date=prior_start,
        prior_season_end_date=prior_end,
    )
    _log(f"  Defense prior done: {len(result.rows)} teams")


def build_defense_rolling(
    *,
    raw_current: Path,
    pit_cache: PITCache,
    target_season: int,
    season_start_date: str,
    season_end_date: str,
    window_days: int = 60,
) -> None:
    defense_builder = TeamDefensePITBuilder(
        raw_cache_db=raw_current,
        pit_cache=pit_cache,
    )
    cutoff_dates = list(_date_range(season_start_date, season_end_date))
    _log(f"Defense rolling: {len(cutoff_dates)} cutoff dates, window={window_days}d")

    for i, cutoff_date in enumerate(cutoff_dates, 1):
        defense_builder.persist_cutoff(
            season=target_season,
            season_start_date=season_start_date,
            as_of_date=cutoff_date,
            window_days=window_days,
        )
        if i % 30 == 0 or i == len(cutoff_dates):
            _log(f"  Defense rolling [{i}/{len(cutoff_dates)}] cutoff={cutoff_date}")


def build_bullpen_prior(
    *,
    raw_2023: Path,
    pit_cache: PITCache,
    target_season: int,
    prior_season: int,
) -> None:
    prior_start, prior_end = SEASON_WINDOWS[prior_season]
    _log(f"Bullpen prior baseline: season={target_season} prior={prior_season}")
    result = BullpenPriorBaselineBuilder(
        raw_cache_db=raw_2023,
        pit_cache=pit_cache,
    ).persist_prior_baseline(
        season=target_season,
        prior_season=prior_season,
        prior_season_start_date=prior_start,
        prior_season_end_date=prior_end,
    )
    _log(f"  Bullpen prior done: {len(result.rows)} teams")


def build_bullpen_rolling(
    *,
    raw_current: Path,
    pit_cache: PITCache,
    target_season: int,
    season_start_date: str,
    season_end_date: str,
) -> None:
    bullpen_builder = TeamBullpenPITBuilder(
        raw_cache_db=raw_current,
        pit_cache=pit_cache,
    )
    cutoff_dates = list(_date_range(season_start_date, season_end_date))
    _log(f"Bullpen rolling: {len(cutoff_dates)} cutoff dates")

    for i, cutoff_date in enumerate(cutoff_dates, 1):
        bullpen_builder.persist_cutoff(
            season=target_season,
            season_start_date=season_start_date,
            as_of_date=cutoff_date,
        )
        if i % 30 == 0 or i == len(cutoff_dates):
            _log(f"  Bullpen rolling [{i}/{len(cutoff_dates)}] cutoff={cutoff_date}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all PIT caches for a target season.")
    parser.add_argument("--target-season", required=True, type=int)
    parser.add_argument(
        "--pit-cache-db",
        type=Path,
        help="Output PIT cache DB (default: data/pit_cache_{season}.db)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "pit_raw",
        help="Directory containing raw_savant_{season}.db files",
    )
    parser.add_argument(
        "--skip-tte",    action="store_true", help="Skip TTE PIT build"
    )
    parser.add_argument(
        "--skip-defense", action="store_true", help="Skip Defense PIT build"
    )
    parser.add_argument(
        "--skip-bullpen", action="store_true", help="Skip Bullpen PIT build"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target = args.target_season
    prior = target - 1

    if target not in SEASON_WINDOWS:
        print(f"ERROR: no season window configured for {target}. Add it to SEASON_WINDOWS.")
        return 1
    if prior not in SEASON_WINDOWS:
        print(f"ERROR: no season window configured for prior {prior}.")
        return 1

    raw_prior = args.raw_dir / f"raw_savant_{prior}.db"
    raw_current = args.raw_dir / f"raw_savant_{target}.db"

    for path, label in [(raw_prior, f"prior raw ({prior})"), (raw_current, f"current raw ({target})")]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            return 1

    pit_cache_db = args.pit_cache_db or (REPO_ROOT / "data" / f"pit_cache_{target}.db")
    pit_cache = PITCache(pit_cache_db)

    season_start, season_end = SEASON_WINDOWS[target]

    _log(f"=== Building PIT caches for season {target} ===")
    _log(f"  prior raw DB  : {raw_prior}")
    _log(f"  current raw DB: {raw_current}")
    _log(f"  output PIT DB : {pit_cache_db}")
    _log(f"  season window : {season_start} .. {season_end}")

    t0 = time.perf_counter()

    if not args.skip_tte:
        _log("--- TTE ---")
        build_tte_prior(raw_2023=raw_prior, pit_cache=pit_cache, target_season=target, prior_season=prior)
        build_tte_rolling(
            raw_current=raw_current, pit_cache=pit_cache,
            target_season=target, season_start_date=season_start, season_end_date=season_end,
        )

    if not args.skip_defense:
        _log("--- Defense ---")
        build_defense_prior(raw_2023=raw_prior, pit_cache=pit_cache, target_season=target, prior_season=prior)
        build_defense_rolling(
            raw_current=raw_current, pit_cache=pit_cache,
            target_season=target, season_start_date=season_start, season_end_date=season_end,
        )

    if not args.skip_bullpen:
        _log("--- Bullpen ---")
        build_bullpen_prior(raw_2023=raw_prior, pit_cache=pit_cache, target_season=target, prior_season=prior)
        build_bullpen_rolling(
            raw_current=raw_current, pit_cache=pit_cache,
            target_season=target, season_start_date=season_start, season_end_date=season_end,
        )

    elapsed = time.perf_counter() - t0
    _log(f"=== Done in {elapsed:.1f}s | output: {pit_cache_db} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
