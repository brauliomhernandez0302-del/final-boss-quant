"""Prior-season contact-adjusted Team Defense PIT baseline persistence."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pit_cache import PITCache, PITCacheRecord
from .raw_savant_events_cache import RawSavantEventsCache
from .team_defense_pit_builder import (
    METRIC_VERSION,
    TeamDefensePITBuilder,
    TeamDefensePITNamespaces,
    TeamDefensePITSources,
    _metrics_payload,
    _source_fingerprint,
)


BASELINE_VERSION = "contact_adjusted_defense_prior_baseline_v1"


@dataclass(frozen=True)
class TeamDefensePriorBaselineBuildResult:
    rows: dict[str, dict[str, Any]]
    season: int
    prior_season: int
    as_of_date: str
    build_report: dict[str, Any]


class TeamDefensePriorBaseline:
    """Build one-pass projected prior baselines without raw JSON decoding."""

    NAMESPACE = TeamDefensePITNamespaces.TEAM_DEFENSE_PRIOR_BASELINE
    SOURCE = TeamDefensePITSources.TEAM_DEFENSE_PRIOR_BASELINE
    BASELINE_VERSION = BASELINE_VERSION

    def __init__(
        self,
        *,
        raw_cache_db: Path | str,
        pit_cache_db: Path | str | None = None,
        pit_cache: PITCache | None = None,
        raw_cache: RawSavantEventsCache | None = None,
        defense_builder: TeamDefensePITBuilder | None = None,
    ):
        if pit_cache_db is None and pit_cache is None:
            raise ValueError("pit_cache_db is required unless pit_cache is provided")
        self.raw_cache = raw_cache or RawSavantEventsCache(raw_cache_db)
        self.pit_cache = pit_cache or PITCache(pit_cache_db)  # type: ignore[arg-type]
        self.defense_builder = defense_builder or TeamDefensePITBuilder(
            raw_cache_db=raw_cache_db,
            pit_cache=self.pit_cache,
            raw_cache=self.raw_cache,
        )

    def persist_prior_baseline(
        self,
        *,
        season: int,
        prior_season: int,
        prior_season_start_date: str,
        prior_season_end_date: str,
        fetched_at: str | None = None,
    ) -> TeamDefensePriorBaselineBuildResult:
        if prior_season != season - 1:
            raise ValueError("prior_season must be exactly season - 1")
        if not prior_season_start_date.startswith(f"{prior_season}-"):
            raise ValueError("prior_season_start_date must stay inside prior_season")
        if not prior_season_end_date.startswith(f"{prior_season}-"):
            raise ValueError("prior_season_end_date must stay inside prior_season")

        started = time.perf_counter()
        result = self.defense_builder.aggregate_window(
            start_date=prior_season_start_date,
            end_date=prior_season_end_date,
        )
        fingerprint = _source_fingerprint(
            source=self.SOURCE,
            season=season,
            start_date=prior_season_start_date,
            end_date=prior_season_end_date,
            rows_processed=result.rows_processed,
            teams=len(result.rows),
            metric_version=self.BASELINE_VERSION,
            input_fingerprints=self.raw_cache.source_fingerprints_by_date_range(
                start_date=prior_season_start_date,
                end_date=prior_season_end_date,
            ),
        )
        cutoff = f"{prior_season_end_date}T23:59:59Z"
        fetched = fetched_at or datetime.now(timezone.utc).isoformat()
        persisted: dict[str, dict[str, Any]] = {}
        for team_id, metrics in result.rows.items():
            payload = _metrics_payload(
                metrics,
                source_window_start_date=prior_season_start_date,
                source_window_end_date=prior_season_end_date,
                source_fingerprint=fingerprint,
            )
            payload.update(
                {
                    "season": int(season),
                    "prior_season": int(prior_season),
                    "baseline_version": self.BASELINE_VERSION,
                    "metric_version": METRIC_VERSION,
                    "duplicate_pa_rows": result.duplicate_pa_rows,
                    "invalid_fielding_team_rows": result.invalid_fielding_team_rows,
                    "excluded_home_runs": result.excluded_home_runs,
                    "excluded_non_bip_events": result.excluded_non_bip_events,
                }
            )
            self.pit_cache.save_record(
                namespace=self.NAMESPACE,
                entity_id=team_id,
                season=season,
                as_of_date=cutoff,
                source=self.SOURCE,
                source_fingerprint=fingerprint,
                data=payload,
                fetched_at=fetched,
            )
            persisted[team_id] = payload

        build_report = {
            "rows_processed": result.rows_processed,
            "baseline_rows_produced": len(persisted),
            "duplicate_pa_rows": result.duplicate_pa_rows,
            "invalid_fielding_team_rows": result.invalid_fielding_team_rows,
            "excluded_home_runs": result.excluded_home_runs,
            "excluded_non_bip_events": result.excluded_non_bip_events,
            "excluded_non_mlb_teams": list(result.excluded_non_mlb_teams),
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "source_fingerprint": fingerprint,
        }
        return TeamDefensePriorBaselineBuildResult(
            rows=persisted,
            season=int(season),
            prior_season=int(prior_season),
            as_of_date=cutoff,
            build_report=build_report,
        )

    def get_latest_prior_baseline(
        self,
        *,
        team_id: int | str,
        season: int,
        requested_as_of_date: str,
    ) -> PITCacheRecord | None:
        return self.pit_cache.get_latest(
            namespace=self.NAMESPACE,
            entity_id=team_id,
            season=season,
            as_of_date=requested_as_of_date,
            source=self.SOURCE,
        )
