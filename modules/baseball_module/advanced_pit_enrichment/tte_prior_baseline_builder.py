"""Prior-season Team/TTE baseline persistence for experimental PIT snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pit_cache import PITCache, PITCacheRecord
from .raw_savant_events_cache import RawSavantEventsCache
from .savant_offense_daily_aggregator import SavantOffenseRollingBuilder
from .tte_daily_snapshot_builder import TTEPITNamespaces, TTEPITSources


BASELINE_VERSION = "tte_prior_baseline_v1"


@dataclass(frozen=True)
class TTEPriorBaselineBuildResult:
    rows: dict[str, dict[str, Any]]
    season: int
    prior_season: int
    as_of_date: str


class TTEPriorBaselineBuilder:
    """Build and persist prior-season team offense baselines from raw Savant events."""

    NAMESPACE = TTEPITNamespaces.TEAM_OFFENSE_PRIOR_BASELINE
    SOURCE = TTEPITSources.TEAM_OFFENSE_PRIOR_BASELINE
    BASELINE_VERSION = BASELINE_VERSION

    def __init__(
        self,
        *,
        raw_cache_db: Path | str,
        pit_cache_db: Path | str | None = None,
        pit_cache: PITCache | None = None,
        rolling_builder: SavantOffenseRollingBuilder | None = None,
    ):
        if pit_cache_db is None and pit_cache is None:
            raise ValueError("pit_cache_db is required unless pit_cache is provided")
        self.raw_cache = RawSavantEventsCache(raw_cache_db)
        self.pit_cache = pit_cache or PITCache(pit_cache_db)  # type: ignore[arg-type]
        self.rolling_builder = rolling_builder or SavantOffenseRollingBuilder(cache=self.raw_cache)

    def persist_prior_baseline(
        self,
        *,
        season: int,
        prior_season_start_date: str,
        prior_season_end_date: str,
        prior_season: int | None = None,
        team_names: dict[str, str] | None = None,
        fetched_at: str | None = None,
    ) -> TTEPriorBaselineBuildResult:
        """Persist one prior-season baseline per team for the requested current season."""
        prior = prior_season or int(season) - 1
        result = self.rolling_builder.build_teams_for_as_of_date(
            season_start_date=prior_season_start_date,
            as_of_date=prior_season_end_date,
        )
        fingerprint = _baseline_fingerprint(
            raw_cache=self.raw_cache,
            season=season,
            prior_season=prior,
            prior_season_start_date=prior_season_start_date,
            prior_season_end_date=prior_season_end_date,
            baseline_version=self.BASELINE_VERSION,
        )
        fetched = fetched_at or datetime.now(timezone.utc).isoformat()
        cutoff = f"{prior_season_end_date}T23:59:59Z"

        persisted: dict[str, dict[str, Any]] = {}
        for team_id, metrics in result.rows.items():
            payload = {
                "team_id": _coerce_int(team_id),
                "team_name": (team_names or {}).get(str(team_id)),
                "season": int(season),
                "prior_season": int(prior),
                "team_est_woba_prior": metrics.est_woba,
                "team_woba_prior": metrics.woba,
                "bb_pct_prior": metrics.bb_pct,
                "k_pct_prior": metrics.k_pct,
                "barrel_pa_prior": metrics.barrel_pa,
                "brl_percent_prior": metrics.brl_percent,
                "ev95percent_prior": metrics.ev95percent,
                "pa_prior": metrics.plate_appearances,
                "bip_prior": metrics.batted_ball_count,
                "source_fingerprint": fingerprint,
                "baseline_version": self.BASELINE_VERSION,
                "source_window_start_date": prior_season_start_date,
                "source_window_end_date": prior_season_end_date,
                "missing_batting_team_rows": result.missing_team_rows,
            }
            if payload["team_name"] is None:
                payload.pop("team_name")
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
            persisted[str(team_id)] = payload

        return TTEPriorBaselineBuildResult(
            rows=persisted,
            season=int(season),
            prior_season=int(prior),
            as_of_date=cutoff,
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


def _baseline_fingerprint(
    *,
    raw_cache: RawSavantEventsCache,
    season: int,
    prior_season: int,
    prior_season_start_date: str,
    prior_season_end_date: str,
    baseline_version: str,
) -> str:
    events = raw_cache.get_events_by_date_range(
        start_date=prior_season_start_date,
        end_date=prior_season_end_date,
    )
    raw_event_count = len(events)
    distinct_dates = len({event.game_date for event in events})
    payload = {
        "baseline_version": baseline_version,
        "distinct_dates": distinct_dates,
        "prior_season": prior_season,
        "prior_season_end_date": prior_season_end_date,
        "prior_season_start_date": prior_season_start_date,
        "raw_event_count": raw_event_count,
        "season": season,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return (
        f"savant:raw:team_offense_prior:{baseline_version}:{season}:"
        f"{prior_season}:{prior_season_start_date}:{prior_season_end_date}:"
        f"{raw_event_count}:{distinct_dates}:{digest}"
    )


def _coerce_int(value: int | str) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value
