"""Isolated FanGraphs point-in-time fetching.

This module is deliberately not connected to the live FanGraphs fetcher or any
pipeline entrypoint yet.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .pit_cache import PITCache

_FG_URL = "https://www.fangraphs.com/api/leaders/major-league/data"
_TIMEOUT = (5, 30)
_TAG_RE = re.compile(r"<[^>]+>")


class FanGraphsPITFetcher:
    SOURCE = "fangraphs"
    PITCHER_NAMESPACE = "fangraphs.pitcher"

    def __init__(self, cache_db: Path | str, session: requests.Session | None = None):
        self.cache = PITCache(cache_db)
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; FinalBossQuant/1.0)"})

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
        """Fetch and cache pitcher metrics for a historical date window.

        The persisted `as_of_date` is `end_date`, so later callers can query by
        prediction cutoff without seeing rows from future windows.
        """
        as_of_date = _as_utc_cutoff(end_date)
        response = self._session.get(
            _FG_URL,
            params=_leaderboard_params(season=season, start_date=start_date, end_date=end_date),
            timeout=_TIMEOUT,
        )
        response.raise_for_status()

        payload = response.json()
        _validate_date_range(payload=payload, start_date=start_date, end_date=end_date)
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        parsed = self._parse_rows(rows)
        fingerprint = _fingerprint(
            season=season,
            start_date=start_date,
            end_date=end_date,
            payload=payload,
        )
        fetched_at = datetime.now(timezone.utc).isoformat()

        for mlbam_id, metrics in parsed.items():
            self.cache.save_record(
                namespace=self.PITCHER_NAMESPACE,
                entity_id=mlbam_id,
                season=season,
                as_of_date=as_of_date,
                source=self.SOURCE,
                source_fingerprint=fingerprint,
                data=metrics,
                fetched_at=fetched_at,
            )

        return parsed

    def _parse_rows(self, rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for row in rows:
            mlbam_id = _i(row.get("xMLBAMID"))
            if mlbam_id is None:
                continue

            out[mlbam_id] = {
                "xfip": _f(row.get("xFIP")),
                "siera": _f(row.get("SIERA")),
                "fip": _f(row.get("FIP")),
                "xera": _f(row.get("xERA")),
                "era": _f(row.get("ERA")),
                "war": _f(row.get("WAR")),
                "k_pct": _f(row.get("K%")),
                "bb_pct": _f(row.get("BB%")),
                "k_bb_pct": _f(row.get("K-BB%")),
                "swstr_pct": _f(row.get("SwStr%")),
                "babip": _f(row.get("BABIP")),
                "lob_pct": _f(row.get("LOB%")),
                "hr_fb": _f(row.get("HR/FB")),
                "gb_pct": _f(row.get("GB%")),
                "fb_pct": _f(row.get("FB%")),
                "ld_pct": _f(row.get("LD%")),
                "ip": _f(row.get("IP")),
                "whip": _f(row.get("WHIP")),
                "avg_bat_speed": _f(row.get("AvgBatSpeed")),
                "hard_contact_pct": _f(row.get("Hard%")),
                "fg_playerid": row.get("playerid"),
                "fg_name": _strip_html(row.get("Name", "")),
                "fg_team": _strip_html(row.get("Team", "")),
                "fg_year": _i(row.get("Season")),
            }

        return out


def _leaderboard_params(*, season: int, start_date: str, end_date: str) -> dict[str, str]:
    return {
        "pos": "all",
        "stats": "pit",
        "lg": "all",
        "qual": "0",
        "season": str(season),
        "season1": str(season),
        "ind": "0",
        "month": "1000",
        "team": "0",
        "pageitems": "600",
        "pagenum": "1",
        "type": "8",
        "startdate": start_date,
        "enddate": end_date,
    }


def _validate_date_range(*, payload: Any, start_date: str, end_date: str) -> None:
    if not isinstance(payload, dict):
        return

    actual = payload.get("dateRange")
    expected = f"{start_date} and {end_date}"
    if actual and actual != expected:
        raise RuntimeError(f"FanGraphs returned dateRange {actual!r}, expected {expected!r}")


def _as_utc_cutoff(value: str) -> str:
    raw = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _fingerprint(*, season: int, start_date: str, end_date: str, payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return f"fangraphs:major-league-data:v1:{season}:{start_date}:{end_date}:{digest}"


def _strip_html(value: str) -> str:
    return _TAG_RE.sub("", value or "").strip()


def _f(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "", "null", "NULL") else None
    except (TypeError, ValueError):
        return None


def _i(value: Any) -> int | None:
    try:
        return int(float(value)) if value not in (None, "", "null", "NULL") else None
    except (TypeError, ValueError):
        return None
