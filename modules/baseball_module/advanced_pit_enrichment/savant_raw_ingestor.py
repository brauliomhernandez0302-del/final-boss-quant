"""Isolated raw daily Baseball Savant/Statcast ingestion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .raw_savant_events_cache import RawSavantEventsCache
from .savant_pit_fetcher import (
    _MAX_ROWS_PER_DAY,
    _STATCAST_URL,
    _TIMEOUT,
    _date_range,
    _parse_csv,
    _validate_daily_rows,
)


@dataclass(frozen=True)
class SavantRawIngestionSummary:
    events_saved: int
    distinct_dates: int
    distinct_pitchers: int
    source_fingerprint: str


class SavantRawIngestor:
    """Fetch raw daily Statcast CSV rows and persist them without aggregation."""

    SOURCE = "baseball_savant_raw"

    def __init__(
        self,
        cache_db: Path | str,
        *,
        session: requests.Session | None = None,
    ):
        self.cache = RawSavantEventsCache(cache_db)
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; FinalBossQuant/1.0)"})

    def ingest_date_range(self, *, start_date: str, end_date: str) -> SavantRawIngestionSummary:
        rows_by_day: list[tuple[str, list[dict[str, str]]]] = []
        raw_texts: list[str] = []

        for day in _date_range(start_date, end_date):
            day_text = self._fetch_statcast_day(day.isoformat())
            raw_texts.append(day_text)
            rows = _parse_csv(day_text)
            _validate_daily_rows(rows=rows, requested_date=day.isoformat())
            if len(rows) >= _MAX_ROWS_PER_DAY:
                raise RuntimeError(f"Savant daily CSV may be truncated for {day}: {len(rows)} rows")
            rows_by_day.append((day.isoformat(), rows))

        rows = [row for _, day_rows in rows_by_day for row in day_rows]
        fingerprint = _fingerprint(start_date=start_date, end_date=end_date, raw_texts=raw_texts)
        self.cache.save_events(
            rows,
            source_fingerprint=fingerprint,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )

        return SavantRawIngestionSummary(
            events_saved=len(rows),
            distinct_dates=len({row.get("game_date") for row in rows if row.get("game_date")}),
            distinct_pitchers=len({row.get("pitcher") for row in rows if row.get("pitcher")}),
            source_fingerprint=fingerprint,
        )

    def _fetch_statcast_day(self, day: str) -> str:
        response = self._session.get(
            _STATCAST_URL,
            params={
                "all": "true",
                "type": "details",
                "player_type": "pitcher",
                "game_date_gt": day,
                "game_date_lt": day,
            },
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        return response.text


def _fingerprint(*, start_date: str, end_date: str, raw_texts: list[str]) -> str:
    serialized = json.dumps(raw_texts, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return f"savant:statcast-raw:v1:{start_date}:{end_date}:{digest}"


def summary_to_dict(summary: SavantRawIngestionSummary) -> dict[str, Any]:
    return {
        "events_saved": summary.events_saved,
        "distinct_dates": summary.distinct_dates,
        "distinct_pitchers": summary.distinct_pitchers,
        "source_fingerprint": summary.source_fingerprint,
    }
