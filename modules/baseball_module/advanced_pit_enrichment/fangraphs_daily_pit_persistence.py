"""Persist daily/cutoff FanGraphs pitcher metrics into the isolated PIT cache."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .fangraphs_pit_fetcher import (
    _FG_URL,
    _TIMEOUT,
    _f,
    _i,
    _leaderboard_params,
    _strip_html,
    _validate_date_range,
)
from .pit_cache import PITCache, PITCacheRecord


class FanGraphsDailyPITPersistence:
    """Source-isolated daily/cutoff persistence for FanGraphs pitcher metrics."""

    NAMESPACE = "fangraphs.pitcher.daily"
    SOURCE = "fangraphs_daily_cutoff"
    METRIC_VERSION = "fangraphs_daily_cutoff_v1"

    def __init__(
        self,
        *,
        pit_cache_db: Path | str | None = None,
        pit_cache: PITCache | None = None,
        session: requests.Session | None = None,
    ):
        if pit_cache_db is None and pit_cache is None:
            raise ValueError("pit_cache_db is required unless pit_cache is provided")

        self.pit_cache = pit_cache or PITCache(pit_cache_db)  # type: ignore[arg-type]
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; FinalBossQuant/1.0)"})

    def persist_cutoff(
        self,
        *,
        season: int,
        season_start_date: str,
        as_of_date: str,
        fetched_at: str | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Fetch, parse, and upsert one row per pitcher for a FanGraphs cutoff."""
        payload = self._fetch_payload(
            season=season,
            season_start_date=season_start_date,
            as_of_date=as_of_date,
        )
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        parsed = _parse_rows(rows)
        fingerprint = self.inputs_fingerprint(
            season=season,
            season_start_date=season_start_date,
            as_of_date=as_of_date,
            row_count=len(parsed),
        )
        fetched = fetched_at or datetime.now(timezone.utc).isoformat()

        persisted: dict[int, dict[str, Any]] = {}
        for mlbam_id, row in parsed.items():
            data = _metrics_to_payload(
                row,
                mlbam_id=mlbam_id,
                source_window_start_date=season_start_date,
                source_window_end_date=as_of_date,
                metric_version=self.METRIC_VERSION,
            )
            self.pit_cache.save_record(
                namespace=self.NAMESPACE,
                entity_id=mlbam_id,
                season=season,
                as_of_date=as_of_date,
                source=self.SOURCE,
                source_fingerprint=fingerprint,
                data=data,
                fetched_at=fetched,
            )
            persisted[mlbam_id] = data

        return persisted

    def get_latest_pitcher_snapshot(
        self,
        *,
        pitcher: int | str,
        season: int,
        requested_as_of_date: str,
    ) -> PITCacheRecord | None:
        """Return the latest FanGraphs daily row at or before requested cutoff."""
        return self.pit_cache.get_latest(
            namespace=self.NAMESPACE,
            entity_id=pitcher,
            season=season,
            as_of_date=requested_as_of_date,
            source=self.SOURCE,
        )

    def inputs_fingerprint(
        self,
        *,
        season: int,
        season_start_date: str,
        as_of_date: str,
        row_count: int,
    ) -> str:
        data = {
            "as_of_date": as_of_date,
            "metric_version": self.METRIC_VERSION,
            "provider": "FanGraphs",
            "row_count": row_count,
            "season": int(season),
            "season_start_date": season_start_date,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        return (
            "fangraphs:daily-cutoff:"
            f"{self.METRIC_VERSION}:{season}:{season_start_date}:{as_of_date}:"
            f"{row_count}:{digest}"
        )

    def _fetch_payload(self, *, season: int, season_start_date: str, as_of_date: str) -> Any:
        response = self._session.get(
            _FG_URL,
            params=_leaderboard_params(
                season=season,
                start_date=season_start_date,
                end_date=as_of_date,
            ),
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        _validate_date_range(payload=payload, start_date=season_start_date, end_date=as_of_date)
        return payload


def _parse_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        mlbam_id = _i(row.get("xMLBAMID"))
        if mlbam_id is None:
            continue

        out[mlbam_id] = {
            "siera": _f(row.get("SIERA")),
            "xfip": _f(row.get("xFIP")),
            "xera": _f(row.get("xERA")),
            "fip": _f(row.get("FIP")),
            "k_pct": _f(row.get("K%")),
            "bb_pct": _f(row.get("BB%")),
            "ip": _f(row.get("IP")),
            "player_name": _strip_html(row.get("Name", "")),
            "fg_playerid": row.get("playerid") or None,
        }

    return out


def _metrics_to_payload(
    metrics: dict[str, Any],
    *,
    mlbam_id: int,
    source_window_start_date: str,
    source_window_end_date: str,
    metric_version: str,
) -> dict[str, Any]:
    return {
        "siera": metrics.get("siera"),
        "xfip": metrics.get("xfip"),
        "xera": metrics.get("xera"),
        "fip": metrics.get("fip"),
        "k_pct": metrics.get("k_pct"),
        "bb_pct": metrics.get("bb_pct"),
        "ip": metrics.get("ip"),
        "player_name": metrics.get("player_name"),
        "fg_playerid": metrics.get("fg_playerid"),
        "mlbam_id": mlbam_id,
        "source_window_start_date": source_window_start_date,
        "source_window_end_date": source_window_end_date,
        "metric_version": metric_version,
    }
