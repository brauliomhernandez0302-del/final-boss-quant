"""Persist rolling Savant PIT metrics into the isolated PIT cache."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pit_cache import PITCache, PITCacheRecord
from .raw_savant_events_cache import RawSavantEventsCache
from .savant_rolling_pit_builder import SavantRollingPITBuilder, SavantRollingPitcherMetrics


class SavantRollingPITPersistence:
    """Source-isolated persistence for rolling Savant pitcher snapshots."""

    NAMESPACE = "savant.pitcher.rolling"
    SOURCE = "baseball_savant_rolling"
    METRIC_VERSION = "savant_rolling_v1"

    def __init__(
        self,
        *,
        raw_cache_db: Path | str,
        pit_cache_db: Path | str | None = None,
        pit_cache: PITCache | None = None,
        builder: SavantRollingPITBuilder | None = None,
    ):
        if pit_cache_db is None and pit_cache is None:
            raise ValueError("pit_cache_db is required unless pit_cache is provided")

        self.raw_cache = RawSavantEventsCache(raw_cache_db)
        self.pit_cache = pit_cache or PITCache(pit_cache_db)  # type: ignore[arg-type]
        self.builder = builder or SavantRollingPITBuilder(cache=self.raw_cache)

    def persist_cutoff(
        self,
        *,
        season: int,
        season_start_date: str,
        as_of_date: str,
        fetched_at: str | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Build and upsert one rolling snapshot row per pitcher for a cutoff."""
        metrics = self.builder.build_for_as_of_date(
            season_start_date=season_start_date,
            as_of_date=as_of_date,
        )
        fingerprint = self.inputs_fingerprint(
            season_start_date=season_start_date,
            as_of_date=as_of_date,
        )
        fetched = fetched_at or datetime.now(timezone.utc).isoformat()

        persisted: dict[int, dict[str, Any]] = {}
        for pitcher, row in metrics.items():
            data = _metrics_to_payload(
                row,
                source_window_start_date=season_start_date,
                source_window_end_date=as_of_date,
                metric_version=self.METRIC_VERSION,
            )
            self.pit_cache.save_record(
                namespace=self.NAMESPACE,
                entity_id=pitcher,
                season=season,
                as_of_date=as_of_date,
                source=self.SOURCE,
                source_fingerprint=fingerprint,
                data=data,
                fetched_at=fetched,
            )
            persisted[pitcher] = data

        return persisted

    def get_latest_pitcher_snapshot(
        self,
        *,
        pitcher: int | str,
        season: int,
        requested_as_of_date: str,
    ) -> PITCacheRecord | None:
        """Return the latest rolling Savant row at or before requested cutoff."""
        return self.pit_cache.get_latest(
            namespace=self.NAMESPACE,
            entity_id=pitcher,
            season=season,
            as_of_date=requested_as_of_date,
            source=self.SOURCE,
        )

    def inputs_fingerprint(self, *, season_start_date: str, as_of_date: str) -> str:
        events = self.raw_cache.get_events_by_date_range(
            start_date=season_start_date,
            end_date=as_of_date,
        )
        raw_event_count = len(events)
        distinct_dates = len({event.game_date for event in events})
        payload = {
            "metric_version": self.METRIC_VERSION,
            "raw_event_count": raw_event_count,
            "season_start_date": season_start_date,
            "source": "savant raw",
            "source_window_end_date": as_of_date,
            "distinct_dates": distinct_dates,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        return (
            "savant:raw:rolling:"
            f"{self.METRIC_VERSION}:{season_start_date}:{as_of_date}:"
            f"{raw_event_count}:{distinct_dates}:{digest}"
        )


def _metrics_to_payload(
    metrics: SavantRollingPitcherMetrics,
    *,
    source_window_start_date: str,
    source_window_end_date: str,
    metric_version: str,
) -> dict[str, Any]:
    return {
        "est_woba": metrics.est_woba,
        "woba": metrics.woba,
        "woba_numerator": metrics.woba_numerator,
        "woba_denominator": metrics.woba_denominator,
        "brl_percent": metrics.brl_percent,
        "barrel_count": metrics.barrel_count,
        "batted_ball_count": metrics.batted_ball_count,
        "pa": metrics.pa,
        "bip": metrics.bip,
        "avg_hit_speed": metrics.avg_hit_speed,
        "ev95plus": metrics.ev95plus,
        "ev95percent": metrics.ev95percent,
        "sweet_spot_pct": metrics.sweet_spot_pct,
        "source_window_start_date": source_window_start_date,
        "source_window_end_date": source_window_end_date,
        "metric_version": metric_version,
    }
