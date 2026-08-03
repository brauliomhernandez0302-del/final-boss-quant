"""Prior-season Team/TTE baseline persistence for experimental PIT snapshots."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pit_cache import PITCache, PITCacheRecord
from .raw_savant_events_cache import RawSavantEventsCache
from .savant_offense_daily_aggregator import SavantOffenseRollingBuilder
from .tte_daily_snapshot_builder import TTEPITNamespaces, TTEPITSources


BASELINE_VERSION = "tte_prior_baseline_v1"
NON_MLB_TEAM_IDS = frozenset({"AL", "NL"})


@dataclass(frozen=True)
class TTEPriorBaselineBuildResult:
    rows: dict[str, dict[str, Any]]
    season: int
    prior_season: int
    as_of_date: str
    build_report: dict[str, Any] | None = None


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
        total_started = time.perf_counter()
        prior = prior_season or int(season) - 1
        # Anti-leak guard (2026-07-09): the other 3 prior-baseline builders
        # (pitcher, team defense, bullpen) already validate that the window
        # dates fall inside prior_season; this one didn't. Real callers today
        # source dates from a trusted table (SEASON_WINDOWS in
        # scripts/build_all_pit_caches.py) so this was never exploited, but a
        # future caller passing a wrong end date would silently build a
        # "prior baseline" from current-season data with no defense at all.
        if int(prior) != int(season) - 1:
            raise ValueError("prior_season must be exactly season - 1")
        if not prior_season_start_date.startswith(f"{prior}-"):
            raise ValueError("prior_season_start_date must stay inside prior_season")
        if not prior_season_end_date.startswith(f"{prior}-"):
            raise ValueError("prior_season_end_date must stay inside prior_season")
        print(
            "[TTEPriorBaselineBuilder] start "
            f"season={season} prior_season={prior} "
            f"window={prior_season_start_date}..{prior_season_end_date}",
            flush=True,
        )

        stage_started = time.perf_counter()
        raw_event_count = self.raw_cache.count_events_by_date_range(
            start_date=prior_season_start_date,
            end_date=prior_season_end_date,
        )
        distinct_dates = self.raw_cache.count_distinct_dates_by_date_range(
            start_date=prior_season_start_date,
            end_date=prior_season_end_date,
        )
        metadata_elapsed = time.perf_counter() - stage_started
        print(
            "[TTEPriorBaselineBuilder] raw metadata "
            f"raw_event_count={raw_event_count} distinct_game_dates={distinct_dates} "
            f"elapsed_sec={metadata_elapsed:.3f}",
            flush=True,
        )

        stage_started = time.perf_counter()
        result = self.rolling_builder.build_teams_for_as_of_date(
            season_start_date=prior_season_start_date,
            as_of_date=prior_season_end_date,
        )
        aggregation_elapsed = time.perf_counter() - stage_started
        excluded_non_mlb_teams = sorted(
            str(team_id) for team_id in result.rows if str(team_id) in NON_MLB_TEAM_IDS
        )
        team_rows = {
            team_id: metrics
            for team_id, metrics in result.rows.items()
            if str(team_id) not in NON_MLB_TEAM_IDS
        }
        distinct_teams = len(team_rows)
        print(
            "[TTEPriorBaselineBuilder] aggregation "
            f"rows_processed={result.rows_processed} distinct_teams={distinct_teams} "
            f"excluded_non_mlb_teams={excluded_non_mlb_teams} "
            f"missing_batting_team_rows={result.missing_team_rows} "
            f"elapsed_sec={aggregation_elapsed:.3f}",
            flush=True,
        )

        stage_started = time.perf_counter()
        fingerprint = _baseline_fingerprint(
            season=season,
            prior_season=prior,
            prior_season_start_date=prior_season_start_date,
            prior_season_end_date=prior_season_end_date,
            baseline_version=self.BASELINE_VERSION,
            raw_event_count=raw_event_count,
            distinct_dates=distinct_dates,
        )
        fingerprint_elapsed = time.perf_counter() - stage_started
        print(
            "[TTEPriorBaselineBuilder] fingerprint "
            f"source_fingerprint={fingerprint} elapsed_sec={fingerprint_elapsed:.3f}",
            flush=True,
        )
        fetched = fetched_at or datetime.now(timezone.utc).isoformat()
        cutoff = f"{prior_season_end_date}T23:59:59Z"

        stage_started = time.perf_counter()
        persisted: dict[str, dict[str, Any]] = {}
        for team_id, metrics in team_rows.items():
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
        persistence_elapsed = time.perf_counter() - stage_started
        total_elapsed = time.perf_counter() - total_started
        build_report = {
            "raw_event_count": raw_event_count,
            "distinct_game_dates": distinct_dates,
            "distinct_teams": distinct_teams,
            "excluded_non_mlb_teams": excluded_non_mlb_teams,
            "rows_processed": result.rows_processed,
            "missing_batting_team_rows": result.missing_team_rows,
            "baseline_rows_produced": len(persisted),
            "elapsed_sec": {
                "metadata": metadata_elapsed,
                "aggregation": aggregation_elapsed,
                "fingerprint": fingerprint_elapsed,
                "persistence": persistence_elapsed,
                "total": total_elapsed,
            },
        }
        print(
            "[TTEPriorBaselineBuilder] complete "
            f"baseline_rows_produced={len(persisted)} total_elapsed_sec={total_elapsed:.3f}",
            flush=True,
        )

        return TTEPriorBaselineBuildResult(
            rows=persisted,
            season=int(season),
            prior_season=int(prior),
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


def _baseline_fingerprint(
    *,
    season: int,
    prior_season: int,
    prior_season_start_date: str,
    prior_season_end_date: str,
    baseline_version: str,
    raw_event_count: int,
    distinct_dates: int,
) -> str:
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
