"""Prior-season team bullpen baseline from relief-only Raw Savant facts."""

from __future__ import annotations

import resource
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bullpen_pit_builder import SAMPLE_BF_THRESHOLD, aggregate_team_bullpen_metrics
from .bullpen_relief_appearance_builder import (
    FACT_BUILDER_VERSION,
    ROLE_RULE_VERSION,
    BullpenPITNamespaces,
    BullpenPITSources,
    BullpenReliefAppearanceBuilder,
    ReliefAppearanceFact,
    _fingerprint,
)
from .pit_cache import PITCache
from .raw_savant_events_cache import RawSavantEventsCache


BASELINE_VERSION = "team_bullpen_prior_baseline_v1"


@dataclass(frozen=True)
class BullpenPriorBaselineBuildResult:
    rows: dict[str, dict[str, Any]]
    season: int
    prior_season: int
    as_of_date: str
    build_report: dict[str, Any]


class BullpenPriorBaselineBuilder:
    """Persist target-season baselines built only from the prior raw season."""

    NAMESPACE = BullpenPITNamespaces.TEAM_BULLPEN_PRIOR_BASELINE
    SOURCE = BullpenPITSources.TEAM_BULLPEN_PRIOR_BASELINE
    BASELINE_VERSION = BASELINE_VERSION

    def __init__(
        self,
        *,
        raw_cache_db: Path | str,
        pit_cache_db: Path | str | None = None,
        pit_cache: PITCache | None = None,
        raw_cache: RawSavantEventsCache | None = None,
        appearance_builder: BullpenReliefAppearanceBuilder | None = None,
    ):
        if pit_cache_db is None and pit_cache is None:
            raise ValueError("pit_cache_db is required unless pit_cache is provided")
        self.pit_cache = pit_cache or PITCache(pit_cache_db)  # type: ignore[arg-type]
        self.raw_cache = raw_cache or RawSavantEventsCache(raw_cache_db)
        self.appearance_builder = appearance_builder or BullpenReliefAppearanceBuilder(
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
    ) -> BullpenPriorBaselineBuildResult:
        if prior_season != season - 1:
            raise ValueError("prior_season must be exactly season - 1")
        if not prior_season_start_date.startswith(f"{prior_season}-"):
            raise ValueError("prior_season_start_date must stay inside prior_season")
        if not prior_season_end_date.startswith(f"{prior_season}-"):
            raise ValueError("prior_season_end_date must stay inside prior_season")

        started = time.perf_counter()
        result = self.appearance_builder.build(
            start_date=prior_season_start_date,
            end_date=prior_season_end_date,
        )
        grouped: dict[str, list[ReliefAppearanceFact]] = {}
        for fact in result.facts:
            grouped.setdefault(fact.team_id, []).append(fact)

        cutoff = f"{prior_season_end_date}T23:59:59Z"
        fetched = fetched_at or datetime.now(timezone.utc).isoformat()
        input_fingerprints = self.raw_cache.source_fingerprints_by_date_range(
            start_date=prior_season_start_date,
            end_date=prior_season_end_date,
        )
        persisted: dict[str, dict[str, Any]] = {}
        available_rows = 0
        for team_id, facts in sorted(grouped.items()):
            metrics = aggregate_team_bullpen_metrics(team_id, facts)
            available = metrics.relief_batters_faced >= SAMPLE_BF_THRESHOLD
            available_rows += int(available)
            team_rejections = [
                item for item in result.rejected_team_games if item.team_id == team_id
            ]
            fingerprint = _fingerprint(
                {
                    "source": self.SOURCE,
                    "baseline_version": self.BASELINE_VERSION,
                    "role_rule_version": ROLE_RULE_VERSION,
                    "team_id": team_id,
                    "target_season": season,
                    "prior_season": prior_season,
                    "source_window": [
                        prior_season_start_date,
                        prior_season_end_date,
                    ],
                    "fact_fingerprints": sorted(
                        fact.source_fingerprint for fact in facts
                    ),
                    "input_fingerprints": input_fingerprints,
                },
                prefix="savant:raw:bullpen:prior-baseline",
            )
            payload = {
                **asdict(metrics),
                "season": int(season),
                "prior_season": int(prior_season),
                "baseline_available": available,
                "minimum_bf_required": SAMPLE_BF_THRESHOLD,
                "source_window_start_date": prior_season_start_date,
                "source_window_end_date": prior_season_end_date,
                "quality_window_start_date": prior_season_start_date,
                "quality_window_end_date": prior_season_end_date,
                "baseline_version": self.BASELINE_VERSION,
                "fact_builder_version": FACT_BUILDER_VERSION,
                "role_rule_version": ROLE_RULE_VERSION,
                "source_fingerprint": fingerprint,
                "accepted_relief_team_games": metrics.relief_team_games,
                "rejected_relief_team_games": len(team_rejections),
                "rejected_reason_counts": dict(
                    sorted(Counter(item.reason for item in team_rejections).items())
                ),
                "starter_pitches_excluded": result.starter_pitches_excluded,
                "starter_pitches_included": result.starter_pitches_included,
                "duplicate_pitch_contributions_prevented": (
                    result.duplicate_pitch_contributions_prevented
                ),
                "duplicate_pa_contributions_prevented": (
                    result.duplicate_pa_contributions_prevented
                ),
                "input_source_fingerprints": list(input_fingerprints),
                "roster_fallback_used": False,
                "legacy_bullpen_fallback_used": False,
            }
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

        elapsed = time.perf_counter() - started
        max_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        build_report = {
            "rows_processed": result.rows_processed,
            "relief_facts": len(result.facts),
            "baseline_rows_produced": len(persisted),
            "baseline_rows_available": available_rows,
            "accepted_relief_team_games": result.accepted_team_games,
            "rejected_relief_team_games": len(result.rejected_team_games),
            "rejected_reason_counts": result.rejected_reason_counts,
            "starter_pitches_excluded": result.starter_pitches_excluded,
            "starter_pitches_included": result.starter_pitches_included,
            "duplicate_pitch_contributions_prevented": (
                result.duplicate_pitch_contributions_prevented
            ),
            "duplicate_pa_contributions_prevented": (
                result.duplicate_pa_contributions_prevented
            ),
            "excluded_non_mlb_teams": list(result.excluded_non_mlb_teams),
            "elapsed_sec": round(elapsed, 3),
            "max_rss_mb": round(max_rss_kib / 1024.0, 3),
            "source_window_start_date": prior_season_start_date,
            "source_window_end_date": prior_season_end_date,
            "input_source_fingerprints": list(input_fingerprints),
        }
        return BullpenPriorBaselineBuildResult(
            rows=persisted,
            season=int(season),
            prior_season=int(prior_season),
            as_of_date=cutoff,
            build_report=build_report,
        )
