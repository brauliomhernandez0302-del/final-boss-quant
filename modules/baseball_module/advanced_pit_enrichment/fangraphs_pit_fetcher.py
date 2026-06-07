"""Placeholder for FanGraphs point-in-time fetching.

This module is deliberately not connected to the live FanGraphs fetcher or any
pipeline entrypoint yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pit_cache import PITCache


class FanGraphsPITFetcher:
    SOURCE = "fangraphs"
    PITCHER_NAMESPACE = "fangraphs.pitcher"

    def __init__(self, cache_db: Path | str):
        self.cache = PITCache(cache_db)

    def get_cached_pitcher_metrics(
        self,
        *,
        mlbam_id: int | str,
        season: int,
        as_of_date: str,
    ) -> dict[str, Any]:
        record = self.cache.get_latest(
            namespace=self.PITCHER_NAMESPACE,
            entity_id=mlbam_id,
            season=season,
            as_of_date=as_of_date,
            source=self.SOURCE,
        )
        return record.data if record else {}

    def fetch_pitcher_metrics_by_date_range(
        self,
        *,
        season: int,
        start_date: str,
        end_date: str,
    ) -> dict[int, dict[str, Any]]:
        raise NotImplementedError("FanGraphs PIT network fetching is not implemented yet")
