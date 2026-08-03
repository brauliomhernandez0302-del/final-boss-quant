"""SQLite cache for point-in-time MLB enrichment records."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PITCacheRecord:
    namespace: str
    entity_id: str
    season: int
    as_of_date: str
    source: str
    source_fingerprint: str
    data: dict[str, Any]
    fetched_at: str


class PITCache:
    """Persistent point-in-time metric cache.

    Records are source-specific and keyed by namespace, entity, season, cutoff,
    and source. `get_latest` only returns rows whose `as_of_date` is at or
    before the supplied prediction cutoff.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def save_record(
        self,
        *,
        namespace: str,
        entity_id: int | str,
        season: int,
        as_of_date: str,
        source: str,
        source_fingerprint: str,
        data: dict[str, Any],
        fetched_at: str | None = None,
    ) -> None:
        """Insert or update one deterministic PIT record identity."""
        self._validate_identity(namespace=namespace, entity_id=entity_id, source=source)
        cutoff = self._normalize_datetime(as_of_date)
        fetched = self._normalize_datetime(fetched_at or datetime.now(timezone.utc).isoformat())
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pit_metric_cache (
                    namespace,
                    entity_id,
                    season,
                    as_of_date,
                    source,
                    source_fingerprint,
                    data_json,
                    fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, entity_id, season, as_of_date, source)
                DO UPDATE SET
                    source_fingerprint = excluded.source_fingerprint,
                    data_json = excluded.data_json,
                    fetched_at = excluded.fetched_at
                """,
                (
                    namespace,
                    str(entity_id),
                    int(season),
                    cutoff,
                    source,
                    source_fingerprint,
                    payload,
                    fetched,
                ),
            )

    def save_many(self, records: Iterable[dict[str, Any]]) -> None:
        for record in records:
            self.save_record(**record)

    def get_record(
        self,
        *,
        namespace: str,
        entity_id: int | str,
        season: int,
        as_of_date: str,
        source: str,
    ) -> PITCacheRecord | None:
        """Return an exact PIT record identity."""
        cutoff = self._normalize_datetime(as_of_date)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM pit_metric_cache
                WHERE namespace = ?
                  AND entity_id = ?
                  AND season = ?
                  AND as_of_date = ?
                  AND source = ?
                """,
                (namespace, str(entity_id), int(season), cutoff, source),
            ).fetchone()

        return self._row_to_record(row)

    def get_latest(
        self,
        *,
        namespace: str,
        entity_id: int | str,
        season: int,
        as_of_date: str,
        source: str,
    ) -> PITCacheRecord | None:
        """Return the newest source-isolated record at or before cutoff."""
        cutoff = self._normalize_datetime(as_of_date)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM pit_metric_cache
                WHERE namespace = ?
                  AND entity_id = ?
                  AND season = ?
                  AND as_of_date <= ?
                  AND source = ?
                ORDER BY as_of_date DESC, fetched_at DESC
                LIMIT 1
                """,
                (namespace, str(entity_id), int(season), cutoff, source),
            ).fetchone()

        return self._row_to_record(row)

    def list_cutoffs(self, *, namespace: str, season: int, source: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT as_of_date
                FROM pit_metric_cache
                WHERE namespace = ?
                  AND season = ?
                  AND source = ?
                ORDER BY as_of_date ASC
                """,
                (namespace, int(season), source),
            ).fetchall()

        return [row["as_of_date"] for row in rows]

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS pit_metric_cache (
                    namespace TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    as_of_date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY(namespace, entity_id, season, as_of_date, source)
                );

                CREATE INDEX IF NOT EXISTS idx_pit_metric_latest
                    ON pit_metric_cache(
                        namespace,
                        entity_id,
                        season,
                        source,
                        as_of_date
                    );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _validate_identity(*, namespace: str, entity_id: int | str, source: str) -> None:
        if not namespace:
            raise ValueError("namespace is required")
        if entity_id is None or str(entity_id) == "":
            raise ValueError("entity_id is required")
        if not source:
            raise ValueError("source is required")

    @staticmethod
    def _normalize_datetime(value: str) -> str:
        if not value:
            raise ValueError("datetime value is required")

        raw = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"invalid ISO datetime: {value}") from exc

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _row_to_record(row: sqlite3.Row | None) -> PITCacheRecord | None:
        if row is None:
            return None

        return PITCacheRecord(
            namespace=row["namespace"],
            entity_id=row["entity_id"],
            season=int(row["season"]),
            as_of_date=row["as_of_date"],
            source=row["source"],
            source_fingerprint=row["source_fingerprint"],
            data=json.loads(row["data_json"]),
            fetched_at=row["fetched_at"],
        )
