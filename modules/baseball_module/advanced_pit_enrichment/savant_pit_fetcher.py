"""Isolated Baseball Savant point-in-time fetching.

This module is deliberately not connected to the live Savant fetcher or any
pipeline entrypoint yet.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from .pit_cache import PITCache

_STATCAST_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
_TIMEOUT = (5, 60)
_MAX_ROWS_PER_DAY = 24_999


class SavantPITFetcher:
    SOURCE = "baseball_savant"
    PITCHER_NAMESPACE = "savant.pitcher"

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
        """Fetch daily Statcast CSVs, aggregate by pitcher, and cache the window."""
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

        parsed = _aggregate_pitcher_rows([row for _, rows in rows_by_day for row in rows])
        fingerprint = _fingerprint(
            season=season,
            start_date=start_date,
            end_date=end_date,
            raw_texts=raw_texts,
        )
        as_of_date = _as_utc_cutoff(end_date)
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


def _parse_csv(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    return list(reader)


def _validate_daily_rows(*, rows: list[dict[str, str]], requested_date: str) -> None:
    for row in rows:
        actual = row.get("game_date")
        if actual and actual != requested_date:
            raise RuntimeError(f"Savant returned game_date {actual!r}, expected {requested_date!r}")


def _aggregate_pitcher_rows(rows: list[dict[str, str]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pitcher_id = _i(row.get("pitcher"))
        if pitcher_id is None:
            continue
        grouped[pitcher_id].append(row)

    return {pitcher_id: _aggregate_one_pitcher(pitcher_rows) for pitcher_id, pitcher_rows in grouped.items()}


def _aggregate_one_pitcher(rows: list[dict[str, str]]) -> dict[str, Any]:
    batted_ball_rows = [row for row in rows if _f(row.get("launch_speed")) is not None]
    launch_speeds = [_f(row.get("launch_speed")) for row in batted_ball_rows]
    launch_speeds = [value for value in launch_speeds if value is not None]
    launch_angles = [_f(row.get("launch_angle")) for row in batted_ball_rows]
    launch_angles = [value for value in launch_angles if value is not None]
    xwoba_values = [_f(row.get("estimated_woba_using_speedangle")) for row in rows]
    xwoba_values = [value for value in xwoba_values if value is not None]
    woba_values = [_f(row.get("woba_value")) for row in rows]
    woba_denoms = [_f(row.get("woba_denom")) for row in rows]

    woba_numerator = sum(value for value in woba_values if value is not None)
    woba_denominator = sum(value for value in woba_denoms if value is not None)
    pa = int(woba_denominator)
    hard_hit_count = sum(1 for value in launch_speeds if value >= 95.0)
    barrel_count = sum(1 for row in rows if _i(row.get("launch_speed_angle")) == 6)
    sweet_spot_count = sum(1 for value in launch_angles if 8.0 <= value <= 32.0)
    dates = sorted({row.get("game_date") for row in rows if row.get("game_date")})

    return {
        "player_name": rows[0].get("player_name"),
        "pitches": len(rows),
        "pa": pa,
        "bip": len(batted_ball_rows),
        "attempts": len(batted_ball_rows),
        "est_woba": _mean(xwoba_values),
        "woba": _safe_div(woba_numerator, woba_denominator),
        "avg_hit_speed": _mean(launch_speeds),
        "max_hit_speed": max(launch_speeds) if launch_speeds else None,
        "avg_hit_angle": _mean(launch_angles),
        "ev95plus": hard_hit_count,
        "ev95percent": _pct(hard_hit_count, len(launch_speeds)),
        "brl_count": barrel_count,
        "brl_percent": _pct(barrel_count, len(batted_ball_rows)),
        "brl_pa": _pct(barrel_count, pa),
        "sweet_spot_pct": _pct(sweet_spot_count, len(launch_angles)),
        "first_game_date": dates[0] if dates else None,
        "last_game_date": dates[-1] if dates else None,
    }


def _date_range(start_date: str, end_date: str) -> list[date]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _as_utc_cutoff(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _fingerprint(*, season: int, start_date: str, end_date: str, raw_texts: list[str]) -> str:
    serialized = json.dumps(raw_texts, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return f"savant:statcast-search:v1:{season}:{start_date}:{end_date}:{digest}"


def _mean(values: list[float]) -> float | None:
    return _safe_div(sum(values), len(values))


def _safe_div(numerator: float, denominator: float) -> float | None:
    if not denominator:
        return None
    return numerator / denominator


def _pct(numerator: int, denominator: int) -> float | None:
    value = _safe_div(float(numerator), float(denominator))
    return value * 100.0 if value is not None else None


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
