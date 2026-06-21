"""SQLite storage for raw daily Baseball Savant/Statcast events.

This module is intentionally isolated from live and backtest pipelines.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RawSavantEvent:
    game_date: str
    game_pk: int
    at_bat_number: int
    pitch_number: int
    pitcher: int | None
    batter: int | None
    events: str | None
    launch_speed: float | None
    launch_angle: float | None
    estimated_woba_using_speedangle: float | None
    woba_value: float | None
    woba_denom: float | None
    launch_speed_angle: int | None
    raw_json: dict[str, Any]
    source_fingerprint: str
    fetched_at: str


@dataclass(frozen=True)
class RawSavantTeamOffenseEvent:
    game_date: str
    game_pk: int
    at_bat_number: int
    pitch_number: int
    events: str | None
    launch_speed: float | None
    launch_angle: float | None
    estimated_woba_using_speedangle: float | None
    woba_value: float | None
    woba_denom: float | None
    launch_speed_angle: int | None
    batting_team: str | None
    bat_team: str | None
    batter_team: str | None
    team_batting: str | None
    home_team: str | None
    away_team: str | None
    inning_topbot: str | None
    inning_half: str | None


class RawSavantEventsCache:
    """Persistent raw event cache keyed by Statcast pitch identity."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def save_events(
        self,
        events: Iterable[dict[str, Any]],
        *,
        source_fingerprint: str,
        fetched_at: str | None = None,
    ) -> None:
        if not source_fingerprint:
            raise ValueError("source_fingerprint is required")

        fetched = _normalize_datetime(fetched_at or datetime.now(timezone.utc).isoformat())
        rows = [
            _event_to_row(event, source_fingerprint=source_fingerprint, fetched_at=fetched)
            for event in events
        ]
        if not rows:
            return

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO raw_savant_events (
                    game_date,
                    game_pk,
                    at_bat_number,
                    pitch_number,
                    pitcher,
                    batter,
                    events,
                    launch_speed,
                    launch_angle,
                    estimated_woba_using_speedangle,
                    woba_value,
                    woba_denom,
                    launch_speed_angle,
                    raw_json,
                    source_fingerprint,
                    fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_pk, at_bat_number, pitch_number)
                DO UPDATE SET
                    game_date = excluded.game_date,
                    pitcher = excluded.pitcher,
                    batter = excluded.batter,
                    events = excluded.events,
                    launch_speed = excluded.launch_speed,
                    launch_angle = excluded.launch_angle,
                    estimated_woba_using_speedangle = excluded.estimated_woba_using_speedangle,
                    woba_value = excluded.woba_value,
                    woba_denom = excluded.woba_denom,
                    launch_speed_angle = excluded.launch_speed_angle,
                    raw_json = excluded.raw_json,
                    source_fingerprint = excluded.source_fingerprint,
                    fetched_at = excluded.fetched_at
                """,
                rows,
            )

    def get_events_by_date_range(self, *, start_date: str, end_date: str) -> list[RawSavantEvent]:
        _validate_date(start_date, field_name="start_date")
        _validate_date(end_date, field_name="end_date")
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM raw_savant_events
                WHERE game_date >= ?
                  AND game_date <= ?
                ORDER BY game_date, game_pk, at_bat_number, pitch_number
                """,
                (start_date, end_date),
            ).fetchall()

        return [_row_to_event(row) for row in rows]

    def iter_team_offense_events_by_date_range(
        self,
        *,
        start_date: str,
        end_date: str,
    ) -> Iterable[RawSavantTeamOffenseEvent]:
        """Yield only columns needed for team offense aggregation.

        Team attribution fields are stored in raw_json by Savant, so this uses
        SQLite JSON extraction to avoid decoding the full JSON payload in Python
        for every raw event.
        """
        _validate_date(start_date, field_name="start_date")
        _validate_date(end_date, field_name="end_date")
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    game_date,
                    game_pk,
                    at_bat_number,
                    pitch_number,
                    events,
                    launch_speed,
                    launch_angle,
                    estimated_woba_using_speedangle,
                    woba_value,
                    woba_denom,
                    launch_speed_angle,
                    json_extract(raw_json, '$.batting_team') AS batting_team,
                    json_extract(raw_json, '$.bat_team') AS bat_team,
                    json_extract(raw_json, '$.batter_team') AS batter_team,
                    json_extract(raw_json, '$.team_batting') AS team_batting,
                    json_extract(raw_json, '$.home_team') AS home_team,
                    json_extract(raw_json, '$.away_team') AS away_team,
                    json_extract(raw_json, '$.inning_topbot') AS inning_topbot,
                    json_extract(raw_json, '$.inning_half') AS inning_half
                FROM raw_savant_events
                WHERE game_date >= ?
                  AND game_date <= ?
                ORDER BY game_date, game_pk, at_bat_number, pitch_number
                """,
                (start_date, end_date),
            )
            for row in rows:
                yield _row_to_team_offense_event(row)

    def count_events_by_date_range(self, *, start_date: str, end_date: str) -> int:
        _validate_date(start_date, field_name="start_date")
        _validate_date(end_date, field_name="end_date")
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")

        with self._connect() as conn:
            return int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM raw_savant_events
                    WHERE game_date >= ?
                      AND game_date <= ?
                    """,
                    (start_date, end_date),
                ).fetchone()[0]
            )

    def count_distinct_dates_by_date_range(self, *, start_date: str, end_date: str) -> int:
        _validate_date(start_date, field_name="start_date")
        _validate_date(end_date, field_name="end_date")
        if end_date < start_date:
            raise ValueError("end_date must be on or after start_date")

        with self._connect() as conn:
            return int(
                conn.execute(
                    """
                    SELECT COUNT(DISTINCT game_date)
                    FROM raw_savant_events
                    WHERE game_date >= ?
                      AND game_date <= ?
                    """,
                    (start_date, end_date),
                ).fetchone()[0]
            )

    def count_events(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM raw_savant_events").fetchone()[0])

    def count_distinct_dates(self) -> int:
        with self._connect() as conn:
            return int(
                conn.execute("SELECT COUNT(DISTINCT game_date) FROM raw_savant_events").fetchone()[0]
            )

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_savant_events (
                    game_date TEXT NOT NULL,
                    game_pk INTEGER NOT NULL,
                    at_bat_number INTEGER NOT NULL,
                    pitch_number INTEGER NOT NULL,
                    pitcher INTEGER,
                    batter INTEGER,
                    events TEXT,
                    launch_speed REAL,
                    launch_angle REAL,
                    estimated_woba_using_speedangle REAL,
                    woba_value REAL,
                    woba_denom REAL,
                    launch_speed_angle INTEGER,
                    raw_json TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY(game_pk, at_bat_number, pitch_number)
                );

                CREATE INDEX IF NOT EXISTS idx_raw_savant_events_date
                    ON raw_savant_events(game_date);

                CREATE INDEX IF NOT EXISTS idx_raw_savant_events_pitcher_date
                    ON raw_savant_events(pitcher, game_date);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn


def _event_to_row(
    event: dict[str, Any],
    *,
    source_fingerprint: str,
    fetched_at: str,
) -> tuple[Any, ...]:
    game_date = str(_required(event, "game_date"))
    _validate_date(game_date, field_name="game_date")

    raw_json = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)
    return (
        game_date,
        _required_int(event, "game_pk"),
        _required_int(event, "at_bat_number"),
        _required_int(event, "pitch_number"),
        _optional_int(event.get("pitcher")),
        _optional_int(event.get("batter")),
        _optional_str(event.get("events")),
        _optional_float(event.get("launch_speed")),
        _optional_float(event.get("launch_angle")),
        _optional_float(event.get("estimated_woba_using_speedangle")),
        _optional_float(event.get("woba_value")),
        _optional_float(event.get("woba_denom")),
        _optional_int(event.get("launch_speed_angle")),
        raw_json,
        source_fingerprint,
        fetched_at,
    )


def _row_to_event(row: sqlite3.Row) -> RawSavantEvent:
    return RawSavantEvent(
        game_date=row["game_date"],
        game_pk=int(row["game_pk"]),
        at_bat_number=int(row["at_bat_number"]),
        pitch_number=int(row["pitch_number"]),
        pitcher=_nullable_int(row["pitcher"]),
        batter=_nullable_int(row["batter"]),
        events=row["events"],
        launch_speed=row["launch_speed"],
        launch_angle=row["launch_angle"],
        estimated_woba_using_speedangle=row["estimated_woba_using_speedangle"],
        woba_value=row["woba_value"],
        woba_denom=row["woba_denom"],
        launch_speed_angle=_nullable_int(row["launch_speed_angle"]),
        raw_json=json.loads(row["raw_json"]),
        source_fingerprint=row["source_fingerprint"],
        fetched_at=row["fetched_at"],
    )


def _row_to_team_offense_event(row: sqlite3.Row) -> RawSavantTeamOffenseEvent:
    return RawSavantTeamOffenseEvent(
        game_date=row["game_date"],
        game_pk=int(row["game_pk"]),
        at_bat_number=int(row["at_bat_number"]),
        pitch_number=int(row["pitch_number"]),
        events=row["events"],
        launch_speed=row["launch_speed"],
        launch_angle=row["launch_angle"],
        estimated_woba_using_speedangle=row["estimated_woba_using_speedangle"],
        woba_value=row["woba_value"],
        woba_denom=row["woba_denom"],
        launch_speed_angle=_nullable_int(row["launch_speed_angle"]),
        batting_team=_optional_str(row["batting_team"]),
        bat_team=_optional_str(row["bat_team"]),
        batter_team=_optional_str(row["batter_team"]),
        team_batting=_optional_str(row["team_batting"]),
        home_team=_optional_str(row["home_team"]),
        away_team=_optional_str(row["away_team"]),
        inning_topbot=_optional_str(row["inning_topbot"]),
        inning_half=_optional_str(row["inning_half"]),
    )


def _required(event: dict[str, Any], field_name: str) -> Any:
    value = event.get(field_name)
    if value in (None, ""):
        raise ValueError(f"{field_name} is required")
    return value


def _required_int(event: dict[str, Any], field_name: str) -> int:
    return int(float(_required(event, field_name)))


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "null", "NULL"):
        return None
    return int(float(value))


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "null", "NULL"):
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _nullable_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _validate_date(value: str, *, field_name: str) -> None:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD") from exc


def _normalize_datetime(value: str) -> str:
    raw = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid ISO datetime: {value}") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc).isoformat()
