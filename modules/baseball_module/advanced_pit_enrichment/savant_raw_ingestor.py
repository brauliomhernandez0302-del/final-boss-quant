"""Isolated raw daily Baseball Savant/Statcast ingestion."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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


@dataclass(frozen=True)
class SavantHistoricalIngestionSummary:
    season: int
    requested_start_date: str
    requested_end_date: str
    completed_dates: tuple[str, ...]
    failed_dates: dict[str, str]
    skipped_dates: tuple[str, ...]
    fetched_dates: tuple[str, ...]
    total_events_saved: int
    distinct_game_dates: int
    distinct_pitchers: int
    database_path: str
    manifest_path: str
    source_fingerprint: str


class SavantRawIngestor:
    """Fetch raw daily Statcast CSV rows and persist them without aggregation."""

    SOURCE = "baseball_savant_raw"
    PROVIDER = "Baseball Savant Statcast Search CSV"
    MANIFEST_VERSION = "raw_savant_manifest_v1"

    def __init__(
        self,
        cache_db: Path | str,
        *,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cache = RawSavantEventsCache(cache_db)
        self._session = session or requests.Session()
        self._sleep = sleep
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; FinalBossQuant/1.0)"})

    def ingest_date_range(self, *, start_date: str, end_date: str) -> SavantRawIngestionSummary:
        raw_texts: list[str] = []
        events_saved = 0
        dates_with_events: set[str] = set()
        pitchers: set[str] = set()

        for day in _date_range(start_date, end_date):
            day_text = self._fetch_statcast_day(day.isoformat())
            raw_texts.append(day_text)
            rows = _parse_csv(day_text)
            _validate_daily_rows(rows=rows, requested_date=day.isoformat())
            if len(rows) >= _MAX_ROWS_PER_DAY:
                raise RuntimeError(f"Savant daily CSV may be truncated for {day}: {len(rows)} rows")
            day_fingerprint = _fingerprint(
                start_date=day.isoformat(),
                end_date=day.isoformat(),
                raw_texts=[day_text],
            )
            self.cache.save_events(
                rows,
                source_fingerprint=day_fingerprint,
                fetched_at=datetime.now(timezone.utc).isoformat(),
            )
            events_saved += len(rows)
            dates_with_events.update(
                row.get("game_date") for row in rows if row.get("game_date")
            )
            pitchers.update(row.get("pitcher") for row in rows if row.get("pitcher"))

        return SavantRawIngestionSummary(
            events_saved=events_saved,
            distinct_dates=len(dates_with_events),
            distinct_pitchers=len(pitchers),
            source_fingerprint=_fingerprint(
                start_date=start_date,
                end_date=end_date,
                raw_texts=raw_texts,
            ),
        )

    def ingest_historical_date_range(
        self,
        *,
        season: int,
        start_date: str,
        end_date: str,
        manifest_path: Path | str,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
        progress: Callable[[str], None] = print,
    ) -> SavantHistoricalIngestionSummary:
        """Ingest one date at a time with durable manifest-based resume."""
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        manifest_file = Path(manifest_path)
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest = _load_manifest(
            manifest_file,
            season=season,
            database_path=self.cache.db_path,
        )
        manifest["requested_start_date"] = start_date
        manifest["requested_end_date"] = end_date

        completed = set(manifest.get("completed_dates", []))
        failed = dict(manifest.get("failed_dates", {}))
        requested_days = [day.isoformat() for day in _date_range(start_date, end_date)]
        skipped_dates: list[str] = []
        fetched_dates: list[str] = []

        for index, day in enumerate(requested_days, start=1):
            if day in completed:
                skipped_dates.append(day)
                progress(f"[{index}/{len(requested_days)}] date={day} status=skipped")
                continue

            last_error: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    progress(
                        f"[{index}/{len(requested_days)}] date={day} "
                        f"status=fetching attempt={attempt}/{max_retries}"
                    )
                    day_text = self._fetch_statcast_day(day)
                    rows = _parse_csv(day_text)
                    _validate_daily_rows(rows=rows, requested_date=day)
                    if len(rows) >= _MAX_ROWS_PER_DAY:
                        raise RuntimeError(
                            f"Savant daily CSV may be truncated for {day}: {len(rows)} rows"
                        )
                    fingerprint = _fingerprint(
                        start_date=day,
                        end_date=day,
                        raw_texts=[day_text],
                    )
                    self.cache.save_events(
                        rows,
                        source_fingerprint=fingerprint,
                        fetched_at=datetime.now(timezone.utc).isoformat(),
                    )
                    completed.add(day)
                    failed.pop(day, None)
                    fetched_dates.append(day)
                    _refresh_manifest(
                        manifest,
                        cache=self.cache,
                        completed_dates=completed,
                        failed_dates=failed,
                    )
                    _write_manifest(manifest_file, manifest)
                    progress(
                        f"[{index}/{len(requested_days)}] date={day} "
                        f"status=completed rows={len(rows)}"
                    )
                    last_error = None
                    break
                except (requests.RequestException, RuntimeError, ValueError) as exc:
                    last_error = exc
                    if attempt < max_retries:
                        progress(
                            f"[{index}/{len(requested_days)}] date={day} "
                            f"status=retry error={type(exc).__name__}"
                        )
                        self._sleep(retry_delay_seconds * attempt)

            if last_error is not None:
                failed[day] = f"{type(last_error).__name__}: {last_error}"
                _refresh_manifest(
                    manifest,
                    cache=self.cache,
                    completed_dates=completed,
                    failed_dates=failed,
                )
                _write_manifest(manifest_file, manifest)
                progress(
                    f"[{index}/{len(requested_days)}] date={day} "
                    f"status=failed error={failed[day]}"
                )

        _refresh_manifest(
            manifest,
            cache=self.cache,
            completed_dates=completed,
            failed_dates=failed,
        )
        _write_manifest(manifest_file, manifest)
        return SavantHistoricalIngestionSummary(
            season=int(season),
            requested_start_date=start_date,
            requested_end_date=end_date,
            completed_dates=tuple(sorted(completed)),
            failed_dates=dict(sorted(failed.items())),
            skipped_dates=tuple(skipped_dates),
            fetched_dates=tuple(fetched_dates),
            total_events_saved=int(manifest["total_events_saved"]),
            distinct_game_dates=int(manifest["distinct_game_dates"]),
            distinct_pitchers=int(manifest["distinct_pitchers"]),
            database_path=str(self.cache.db_path.resolve()),
            manifest_path=str(manifest_file.resolve()),
            source_fingerprint=str(manifest["source_fingerprint"]),
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


def historical_summary_to_dict(summary: SavantHistoricalIngestionSummary) -> dict[str, Any]:
    return {
        "season": summary.season,
        "requested_start_date": summary.requested_start_date,
        "requested_end_date": summary.requested_end_date,
        "completed_dates": list(summary.completed_dates),
        "failed_dates": summary.failed_dates,
        "skipped_dates": list(summary.skipped_dates),
        "fetched_dates": list(summary.fetched_dates),
        "total_events_saved": summary.total_events_saved,
        "distinct_game_dates": summary.distinct_game_dates,
        "distinct_pitchers": summary.distinct_pitchers,
        "database_path": summary.database_path,
        "manifest_path": summary.manifest_path,
        "source_fingerprint": summary.source_fingerprint,
    }


def _load_manifest(
    path: Path,
    *,
    season: int,
    database_path: Path,
) -> dict[str, Any]:
    if path.exists():
        data = json.loads(path.read_text())
        if int(data.get("season")) != int(season):
            raise ValueError(
                f"manifest season {data.get('season')} does not match requested season {season}"
            )
        existing_db = Path(str(data.get("database_path", ""))).resolve()
        if existing_db != database_path.resolve():
            raise ValueError(
                f"manifest database_path {existing_db} does not match {database_path.resolve()}"
            )
        return data
    return {
        "manifest_version": SavantRawIngestor.MANIFEST_VERSION,
        "schema_version": RawSavantEventsCache.SCHEMA_VERSION,
        "season": int(season),
        "requested_start_date": None,
        "requested_end_date": None,
        "completed_dates": [],
        "failed_dates": {},
        "total_events_saved": 0,
        "distinct_game_dates": 0,
        "distinct_pitchers": 0,
        "database_path": str(database_path.resolve()),
        "source": SavantRawIngestor.SOURCE,
        "provider": SavantRawIngestor.PROVIDER,
        "source_fingerprint": None,
        "last_updated_utc": None,
    }


def _refresh_manifest(
    manifest: dict[str, Any],
    *,
    cache: RawSavantEventsCache,
    completed_dates: set[str],
    failed_dates: dict[str, str],
) -> None:
    manifest["completed_dates"] = sorted(completed_dates)
    manifest["failed_dates"] = dict(sorted(failed_dates.items()))
    manifest["total_events_saved"] = cache.count_events()
    manifest["distinct_game_dates"] = cache.count_distinct_dates()
    manifest["distinct_pitchers"] = cache.count_distinct_pitchers()
    manifest["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["source_fingerprint"] = _manifest_fingerprint(manifest)


def _manifest_fingerprint(manifest: dict[str, Any]) -> str:
    payload = {
        "manifest_version": manifest["manifest_version"],
        "schema_version": manifest["schema_version"],
        "season": manifest["season"],
        "completed_dates": manifest["completed_dates"],
        "failed_dates": manifest["failed_dates"],
        "total_events_saved": manifest["total_events_saved"],
        "distinct_game_dates": manifest["distinct_game_dates"],
        "distinct_pitchers": manifest["distinct_pitchers"],
        "source": manifest["source"],
        "provider": manifest["provider"],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"savant:historical-cache:{SavantRawIngestor.MANIFEST_VERSION}:{digest}"


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
