"""Daily and rolling offensive Savant PIT aggregations from raw events.

This module is intentionally isolated from live and backtest paths. It builds
the source rows that future Team/TTE PIT snapshots can consume.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pit_cache import PITCache, PITCacheRecord
from .raw_savant_events_cache import RawSavantEvent, RawSavantEventsCache
from .tte_daily_snapshot_builder import TTEPITNamespaces, TTEPITSources


METRIC_VERSION = "savant_offense_rolling_v1"


@dataclass(frozen=True)
class SavantDailyBatterOffenseMetrics:
    game_date: str
    batter: int
    plate_appearances: int
    batted_ball_count: int
    est_woba: float | None
    est_woba_count: int
    woba: float | None
    woba_numerator: float | None
    woba_denominator: float | None
    barrel_count: int
    brl_percent: float | None
    barrel_pa: float | None
    bb_count: int
    k_count: int
    bb_pct: float | None
    k_pct: float | None
    avg_hit_speed: float | None
    ev95plus: int
    ev95percent: float | None
    sweet_spot_count: int
    sweet_spot_denominator: int
    sweet_spot_pct: float | None


@dataclass(frozen=True)
class SavantDailyTeamOffenseMetrics:
    game_date: str
    batting_team: str
    plate_appearances: int
    batted_ball_count: int
    est_woba: float | None
    est_woba_count: int
    woba: float | None
    woba_numerator: float | None
    woba_denominator: float | None
    barrel_count: int
    brl_percent: float | None
    barrel_pa: float | None
    bb_count: int
    k_count: int
    bb_pct: float | None
    k_pct: float | None
    avg_hit_speed: float | None
    ev95plus: int
    ev95percent: float | None
    sweet_spot_count: int
    sweet_spot_denominator: int
    sweet_spot_pct: float | None


@dataclass(frozen=True)
class SavantTeamOffenseDailyResult:
    rows: list[SavantDailyTeamOffenseMetrics]
    missing_team_rows: int


@dataclass(frozen=True)
class SavantRollingBatterOffenseMetrics:
    as_of_date: str
    batter: int
    plate_appearances: int
    batted_ball_count: int
    est_woba: float | None
    woba: float | None
    woba_numerator: float | None
    woba_denominator: float | None
    barrel_count: int
    brl_percent: float | None
    barrel_pa: float | None
    bb_count: int
    k_count: int
    bb_pct: float | None
    k_pct: float | None
    avg_hit_speed: float | None
    ev95plus: int
    ev95percent: float | None
    sweet_spot_pct: float | None


@dataclass(frozen=True)
class SavantRollingTeamOffenseMetrics:
    as_of_date: str
    batting_team: str
    plate_appearances: int
    batted_ball_count: int
    est_woba: float | None
    woba: float | None
    woba_numerator: float | None
    woba_denominator: float | None
    barrel_count: int
    brl_percent: float | None
    barrel_pa: float | None
    bb_count: int
    k_count: int
    bb_pct: float | None
    k_pct: float | None
    avg_hit_speed: float | None
    ev95plus: int
    ev95percent: float | None
    sweet_spot_pct: float | None


@dataclass(frozen=True)
class SavantRollingTeamOffenseResult:
    rows: dict[str, SavantRollingTeamOffenseMetrics]
    missing_team_rows: int
    rows_processed: int = 0


class SavantOffenseDailyAggregator:
    """Build daily batter and batting-team offensive metrics from raw events."""

    def __init__(self, cache_db: Path | str | None = None, *, cache: RawSavantEventsCache | None = None):
        if cache_db is None and cache is None:
            raise ValueError("cache_db is required unless cache is provided")
        self.cache = cache or RawSavantEventsCache(cache_db)  # type: ignore[arg-type]

    def aggregate_batters_by_date_range(
        self,
        *,
        start_date: str,
        end_date: str,
    ) -> list[SavantDailyBatterOffenseMetrics]:
        events = self.cache.get_events_by_date_range(start_date=start_date, end_date=end_date)
        grouped: dict[tuple[str, int], list[RawSavantEvent]] = defaultdict(list)

        for event in events:
            if event.batter is None:
                continue
            grouped[(event.game_date, event.batter)].append(event)

        return [
            _aggregate_batter(game_date=game_date, batter=batter, events=batter_events)
            for (game_date, batter), batter_events in sorted(grouped.items())
        ]

    def aggregate_teams_by_date_range(
        self,
        *,
        start_date: str,
        end_date: str,
    ) -> SavantTeamOffenseDailyResult:
        events = self.cache.iter_team_offense_events_by_date_range(
            start_date=start_date,
            end_date=end_date,
        )
        grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
        missing_team_rows = 0

        for event in events:
            batting_team = _batting_team(event)
            if batting_team is None:
                missing_team_rows += 1
                continue
            grouped[(event.game_date, batting_team)].append(event)

        rows = [
            _aggregate_team(game_date=game_date, batting_team=team, events=team_events)
            for (game_date, team), team_events in sorted(grouped.items())
        ]
        return SavantTeamOffenseDailyResult(rows=rows, missing_team_rows=missing_team_rows)

    def aggregate_teams_rolling_by_date_range(
        self,
        *,
        start_date: str,
        end_date: str,
        as_of_date: str,
    ) -> SavantRollingTeamOffenseResult:
        accumulators: dict[str, _RawTeamAccumulator] = {}
        missing_team_rows = 0
        rows_processed = 0

        for event in self.cache.iter_team_offense_events_by_date_range(
            start_date=start_date,
            end_date=end_date,
        ):
            rows_processed += 1
            if event.game_date > as_of_date:
                continue
            batting_team = _batting_team(event)
            if batting_team is None:
                missing_team_rows += 1
                continue
            accumulator = accumulators.setdefault(batting_team, _RawTeamAccumulator())
            accumulator.add_event(event)

        rows = {
            team: accumulator.to_team_metrics(as_of_date=as_of_date, batting_team=team)
            for team, accumulator in sorted(accumulators.items())
        }
        return SavantRollingTeamOffenseResult(
            rows=rows,
            missing_team_rows=missing_team_rows,
            rows_processed=rows_processed,
        )


class SavantOffenseRollingBuilder:
    """Build cumulative batter/team offensive metrics at a PIT cutoff date."""

    def __init__(
        self,
        cache_db: Path | str | None = None,
        *,
        cache: RawSavantEventsCache | None = None,
        daily_aggregator: SavantOffenseDailyAggregator | None = None,
    ):
        if daily_aggregator is not None:
            self.daily_aggregator = daily_aggregator
        else:
            if cache_db is None and cache is None:
                raise ValueError("cache_db is required unless cache or daily_aggregator is provided")
            self.daily_aggregator = SavantOffenseDailyAggregator(cache_db, cache=cache)

    def build_batters_for_as_of_date(
        self,
        *,
        season_start_date: str,
        as_of_date: str,
    ) -> dict[int, SavantRollingBatterOffenseMetrics]:
        daily_rows = self.daily_aggregator.aggregate_batters_by_date_range(
            start_date=season_start_date,
            end_date=as_of_date,
        )
        accumulators: dict[int, _Accumulator] = {}

        for row in daily_rows:
            if row.game_date > as_of_date:
                continue
            accumulator = accumulators.setdefault(row.batter, _Accumulator())
            accumulator.add(row)

        return {
            batter: accumulator.to_batter_metrics(as_of_date=as_of_date, batter=batter)
            for batter, accumulator in sorted(accumulators.items())
        }

    def build_teams_for_as_of_date(
        self,
        *,
        season_start_date: str,
        as_of_date: str,
    ) -> SavantRollingTeamOffenseResult:
        return self.daily_aggregator.aggregate_teams_rolling_by_date_range(
            start_date=season_start_date,
            end_date=as_of_date,
            as_of_date=as_of_date,
        )


class SavantBatterRollingPITPersistence:
    """Persist rolling batter offense snapshots to the canonical PIT cache."""

    NAMESPACE = TTEPITNamespaces.BATTER_ROLLING
    SOURCE = TTEPITSources.BATTER_ROLLING
    METRIC_VERSION = METRIC_VERSION

    def __init__(
        self,
        *,
        raw_cache_db: Path | str,
        pit_cache_db: Path | str | None = None,
        pit_cache: PITCache | None = None,
        builder: SavantOffenseRollingBuilder | None = None,
    ):
        if pit_cache_db is None and pit_cache is None:
            raise ValueError("pit_cache_db is required unless pit_cache is provided")
        self.raw_cache = RawSavantEventsCache(raw_cache_db)
        self.pit_cache = pit_cache or PITCache(pit_cache_db)  # type: ignore[arg-type]
        self.builder = builder or SavantOffenseRollingBuilder(cache=self.raw_cache)

    def persist_cutoff(
        self,
        *,
        season: int,
        season_start_date: str,
        as_of_date: str,
        fetched_at: str | None = None,
    ) -> dict[int, dict[str, Any]]:
        metrics = self.builder.build_batters_for_as_of_date(
            season_start_date=season_start_date,
            as_of_date=as_of_date,
        )
        fingerprint = _inputs_fingerprint(
            raw_cache=self.raw_cache,
            season=season,
            season_start_date=season_start_date,
            as_of_date=as_of_date,
            metric_version=self.METRIC_VERSION,
            entity="batter",
        )
        fetched = fetched_at or datetime.now(timezone.utc).isoformat()

        persisted: dict[int, dict[str, Any]] = {}
        for batter, row in metrics.items():
            data = _rolling_payload(
                row,
                source_window_start_date=season_start_date,
                source_window_end_date=as_of_date,
                metric_version=self.METRIC_VERSION,
            )
            self.pit_cache.save_record(
                namespace=self.NAMESPACE,
                entity_id=batter,
                season=season,
                as_of_date=as_of_date,
                source=self.SOURCE,
                source_fingerprint=fingerprint,
                data=data,
                fetched_at=fetched,
            )
            persisted[batter] = data
        return persisted

    def get_latest_batter_snapshot(
        self,
        *,
        batter: int | str,
        season: int,
        requested_as_of_date: str,
    ) -> PITCacheRecord | None:
        return self.pit_cache.get_latest(
            namespace=self.NAMESPACE,
            entity_id=batter,
            season=season,
            as_of_date=requested_as_of_date,
            source=self.SOURCE,
        )


class SavantTeamOffenseRollingPITPersistence:
    """Persist rolling team offense snapshots to the canonical PIT cache."""

    NAMESPACE = TTEPITNamespaces.TEAM_OFFENSE_ROLLING
    SOURCE = TTEPITSources.TEAM_OFFENSE_ROLLING
    METRIC_VERSION = METRIC_VERSION

    def __init__(
        self,
        *,
        raw_cache_db: Path | str,
        pit_cache_db: Path | str | None = None,
        pit_cache: PITCache | None = None,
        builder: SavantOffenseRollingBuilder | None = None,
    ):
        if pit_cache_db is None and pit_cache is None:
            raise ValueError("pit_cache_db is required unless pit_cache is provided")
        self.raw_cache = RawSavantEventsCache(raw_cache_db)
        self.pit_cache = pit_cache or PITCache(pit_cache_db)  # type: ignore[arg-type]
        self.builder = builder or SavantOffenseRollingBuilder(cache=self.raw_cache)

    def persist_cutoff(
        self,
        *,
        season: int,
        season_start_date: str,
        as_of_date: str,
        fetched_at: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        result = self.builder.build_teams_for_as_of_date(
            season_start_date=season_start_date,
            as_of_date=as_of_date,
        )
        fingerprint = _inputs_fingerprint(
            raw_cache=self.raw_cache,
            season=season,
            season_start_date=season_start_date,
            as_of_date=as_of_date,
            metric_version=self.METRIC_VERSION,
            entity="team",
        )
        fetched = fetched_at or datetime.now(timezone.utc).isoformat()

        persisted: dict[str, dict[str, Any]] = {}
        for team, row in result.rows.items():
            data = _rolling_payload(
                row,
                source_window_start_date=season_start_date,
                source_window_end_date=as_of_date,
                metric_version=self.METRIC_VERSION,
            )
            data["missing_batting_team_rows"] = result.missing_team_rows
            self.pit_cache.save_record(
                namespace=self.NAMESPACE,
                entity_id=team,
                season=season,
                as_of_date=as_of_date,
                source=self.SOURCE,
                source_fingerprint=fingerprint,
                data=data,
                fetched_at=fetched,
            )
            persisted[team] = data
        return persisted

    def get_latest_team_snapshot(
        self,
        *,
        team: int | str,
        season: int,
        requested_as_of_date: str,
    ) -> PITCacheRecord | None:
        return self.pit_cache.get_latest(
            namespace=self.NAMESPACE,
            entity_id=team,
            season=season,
            as_of_date=requested_as_of_date,
            source=self.SOURCE,
        )


class _Accumulator:
    def __init__(self):
        self.est_woba_sum = 0.0
        self.est_woba_count = 0
        self.woba_numerator = 0.0
        self.woba_denominator = 0.0
        self.barrel_count = 0
        self.bb_count = 0
        self.k_count = 0
        self.batted_ball_count = 0
        self.plate_appearances = 0
        self.hit_speed_sum = 0.0
        self.hit_speed_count = 0
        self.ev95plus = 0
        self.sweet_spot_count = 0
        self.sweet_spot_denominator = 0

    def add(self, row: Any) -> None:
        if row.est_woba is not None:
            self.est_woba_sum += row.est_woba * row.est_woba_count
            self.est_woba_count += row.est_woba_count
        if row.woba_numerator is not None:
            self.woba_numerator += row.woba_numerator
        if row.woba_denominator is not None:
            self.woba_denominator += row.woba_denominator
        if row.avg_hit_speed is not None:
            self.hit_speed_sum += row.avg_hit_speed * row.batted_ball_count
            self.hit_speed_count += row.batted_ball_count
        if row.sweet_spot_pct is not None:
            self.sweet_spot_count += row.sweet_spot_count
            self.sweet_spot_denominator += row.sweet_spot_denominator

        self.barrel_count += row.barrel_count
        self.bb_count += row.bb_count
        self.k_count += row.k_count
        self.batted_ball_count += row.batted_ball_count
        self.plate_appearances += row.plate_appearances
        self.ev95plus += row.ev95plus

    def to_batter_metrics(self, *, as_of_date: str, batter: int) -> SavantRollingBatterOffenseMetrics:
        return SavantRollingBatterOffenseMetrics(
            as_of_date=as_of_date,
            batter=batter,
            **self._common_metrics(),
        )

    def to_team_metrics(
        self,
        *,
        as_of_date: str,
        batting_team: str,
    ) -> SavantRollingTeamOffenseMetrics:
        return SavantRollingTeamOffenseMetrics(
            as_of_date=as_of_date,
            batting_team=batting_team,
            **self._common_metrics(),
        )

    def _common_metrics(self) -> dict[str, Any]:
        return {
            "plate_appearances": self.plate_appearances,
            "batted_ball_count": self.batted_ball_count,
            "est_woba": _safe_div(self.est_woba_sum, self.est_woba_count),
            "woba": _safe_div(self.woba_numerator, self.woba_denominator),
            "woba_numerator": self.woba_numerator if self.woba_denominator else None,
            "woba_denominator": self.woba_denominator if self.woba_denominator else None,
            "barrel_count": self.barrel_count,
            "brl_percent": _pct(self.barrel_count, self.batted_ball_count),
            "barrel_pa": _rate(self.barrel_count, self.plate_appearances),
            "bb_count": self.bb_count,
            "k_count": self.k_count,
            "bb_pct": _rate(self.bb_count, self.plate_appearances),
            "k_pct": _rate(self.k_count, self.plate_appearances),
            "avg_hit_speed": _safe_div(self.hit_speed_sum, self.hit_speed_count),
            "ev95plus": self.ev95plus,
            "ev95percent": _pct(self.ev95plus, self.batted_ball_count),
            "sweet_spot_pct": _pct(self.sweet_spot_count, self.sweet_spot_denominator),
        }


class _RawTeamAccumulator:
    def __init__(self):
        self.est_woba_sum = 0.0
        self.est_woba_count = 0
        self.woba_numerator = 0.0
        self.woba_denominator = 0.0
        self.has_woba_value = False
        self.has_woba_denom = False
        self.barrel_count = 0
        self.batted_ball_count = 0
        self.hit_speed_sum = 0.0
        self.hit_speed_count = 0
        self.ev95plus = 0
        self.sweet_spot_count = 0
        self.sweet_spot_denominator = 0
        self.pa_events: dict[tuple[int, int], Any] = {}

    def add_event(self, event: Any) -> None:
        if event.estimated_woba_using_speedangle is not None:
            self.est_woba_sum += event.estimated_woba_using_speedangle
            self.est_woba_count += 1
        if event.woba_value is not None:
            self.woba_numerator += event.woba_value
            self.has_woba_value = True
        if event.woba_denom is not None:
            self.woba_denominator += event.woba_denom
            self.has_woba_denom = True

        if event.launch_speed_angle == 6:
            self.barrel_count += 1

        # MATH-002 fix (2026-07-18, audit_20260714/math002_diagnostico/reporte.md):
        # this is a separate reimplementation of _aggregate_common()'s counting
        # logic (streaming accumulator vs. list comprehension) and had the same
        # foul-inflation bug independently — scoped to PA-terminal events here
        # too, matching _aggregate_common()'s fix. is_pa_event is a per-event
        # boolean (Statcast sets woba_denom/events only on the terminal pitch
        # of an at-bat), so evaluating it inline while streaming is safe and
        # doesn't need to see later pitches of the same at-bat.
        is_pa_event = _is_plate_appearance_event(event)
        if is_pa_event and event.launch_speed is not None:
            self.batted_ball_count += 1
            self.hit_speed_sum += event.launch_speed
            self.hit_speed_count += 1
            if event.launch_speed >= 95.0:
                self.ev95plus += 1
            if event.launch_angle is not None:
                self.sweet_spot_denominator += 1
                if 8.0 <= event.launch_angle <= 32.0:
                    self.sweet_spot_count += 1

        if is_pa_event:
            key = (event.game_pk, event.at_bat_number)
            existing = self.pa_events.get(key)
            if existing is None or event.pitch_number >= existing.pitch_number:
                self.pa_events[key] = event

    def to_team_metrics(
        self,
        *,
        as_of_date: str,
        batting_team: str,
    ) -> SavantRollingTeamOffenseMetrics:
        plate_appearance_events = list(self.pa_events.values())
        plate_appearances = len(plate_appearance_events)
        bb_count = sum(1 for event in plate_appearance_events if _event_label(event) == "walk")
        k_count = sum(
            1
            for event in plate_appearance_events
            if _event_label(event) in {"strikeout", "strikeout_double_play"}
        )

        woba_numerator = self.woba_numerator if self.has_woba_value else None
        woba_denominator = self.woba_denominator if self.has_woba_denom else None

        return SavantRollingTeamOffenseMetrics(
            as_of_date=as_of_date,
            batting_team=batting_team,
            plate_appearances=plate_appearances,
            batted_ball_count=self.batted_ball_count,
            est_woba=_safe_div(self.est_woba_sum, self.est_woba_count),
            woba=_safe_div(woba_numerator, woba_denominator),
            woba_numerator=woba_numerator if woba_denominator else None,
            woba_denominator=woba_denominator if woba_denominator else None,
            barrel_count=self.barrel_count,
            brl_percent=_pct(self.barrel_count, self.batted_ball_count),
            barrel_pa=_rate(self.barrel_count, plate_appearances),
            bb_count=bb_count,
            k_count=k_count,
            bb_pct=_rate(bb_count, plate_appearances),
            k_pct=_rate(k_count, plate_appearances),
            avg_hit_speed=_safe_div(self.hit_speed_sum, self.hit_speed_count),
            ev95plus=self.ev95plus,
            ev95percent=_pct(self.ev95plus, self.batted_ball_count),
            sweet_spot_pct=_pct(self.sweet_spot_count, self.sweet_spot_denominator),
        )


def _aggregate_batter(
    *,
    game_date: str,
    batter: int,
    events: list[RawSavantEvent],
) -> SavantDailyBatterOffenseMetrics:
    return SavantDailyBatterOffenseMetrics(
        game_date=game_date,
        batter=batter,
        **_aggregate_common(events),
    )


def _aggregate_team(
    *,
    game_date: str,
    batting_team: str,
    events: list[Any],
) -> SavantDailyTeamOffenseMetrics:
    return SavantDailyTeamOffenseMetrics(
        game_date=game_date,
        batting_team=batting_team,
        **_aggregate_common(events),
    )


def _aggregate_common(events: list[Any]) -> dict[str, Any]:
    pa_events = _plate_appearance_events(events)
    # MATH-002 fix (2026-07-18, audit_20260714/math002_diagnostico/reporte.md):
    # must be scoped to pa_events, not the raw `events` list — Statcast tracks
    # launch_speed on fouls too (ball physically contacted, PA continues), and
    # counting those inflated batted_ball_count ~1.9x vs. the true count of
    # PA-terminal batted-ball events (confirmed empirically, 16/16 samples).
    batted_ball_events = [event for event in pa_events if event.launch_speed is not None]
    launch_speeds = [event.launch_speed for event in batted_ball_events if event.launch_speed is not None]
    launch_angles = [event.launch_angle for event in batted_ball_events if event.launch_angle is not None]
    est_woba_values = [
        event.estimated_woba_using_speedangle
        for event in events
        if event.estimated_woba_using_speedangle is not None
    ]
    woba_values = [event.woba_value for event in events if event.woba_value is not None]
    woba_denoms = [event.woba_denom for event in events if event.woba_denom is not None]
    woba_numerator = sum(woba_values) if woba_values else None
    woba_denominator = sum(woba_denoms) if woba_denoms else None
    barrel_count = sum(1 for event in events if event.launch_speed_angle == 6)
    plate_appearances = len(pa_events)
    bb_count = sum(1 for event in pa_events if _event_label(event) == "walk")
    k_count = sum(
        1
        for event in pa_events
        if _event_label(event) in {"strikeout", "strikeout_double_play"}
    )
    ev95plus = sum(1 for value in launch_speeds if value >= 95.0)
    sweet_spot_count = sum(1 for value in launch_angles if 8.0 <= value <= 32.0)

    return {
        "plate_appearances": plate_appearances,
        "batted_ball_count": len(batted_ball_events),
        "est_woba": _mean(est_woba_values),
        "est_woba_count": len(est_woba_values),
        "woba": _safe_div(woba_numerator, woba_denominator),
        "woba_numerator": woba_numerator if woba_denominator else None,
        "woba_denominator": woba_denominator if woba_denominator else None,
        "barrel_count": barrel_count,
        "brl_percent": _pct(barrel_count, len(batted_ball_events)),
        "barrel_pa": _rate(barrel_count, plate_appearances),
        "bb_count": bb_count,
        "k_count": k_count,
        "bb_pct": _rate(bb_count, plate_appearances),
        "k_pct": _rate(k_count, plate_appearances),
        "avg_hit_speed": _mean(launch_speeds),
        "ev95plus": ev95plus,
        "ev95percent": _pct(ev95plus, len(launch_speeds)),
        "sweet_spot_count": sweet_spot_count,
        "sweet_spot_denominator": len(launch_angles),
        "sweet_spot_pct": _pct(sweet_spot_count, len(launch_angles)),
    }


def _batting_team(event: Any) -> str | None:
    raw = getattr(event, "raw_json", None)
    for key in ("batting_team", "bat_team", "batter_team", "team_batting"):
        value = _clean_team(getattr(event, key, None))
        if value:
            return value
        value = _clean_team(raw.get(key)) if raw else None
        if value:
            return value

    inning_topbot = str(
        getattr(event, "inning_topbot", None)
        or getattr(event, "inning_half", None)
        or (raw.get("inning_topbot") if raw else None)
        or (raw.get("inning_half") if raw else None)
        or ""
    ).lower()
    home_team = _clean_team(getattr(event, "home_team", None))
    away_team = _clean_team(getattr(event, "away_team", None))
    if raw:
        home_team = home_team or _clean_team(raw.get("home_team"))
        away_team = away_team or _clean_team(raw.get("away_team"))
    if home_team and away_team:
        if inning_topbot.startswith("top"):
            return away_team
        if inning_topbot.startswith("bot"):
            return home_team
    return None


def _clean_team(value: Any) -> str | None:
    if value in (None, "", "null", "NULL"):
        return None
    return str(value)


def _rolling_payload(
    metrics: SavantRollingBatterOffenseMetrics | SavantRollingTeamOffenseMetrics,
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
        "barrel_pa": metrics.barrel_pa,
        "barrel_count": metrics.barrel_count,
        "bb_count": metrics.bb_count,
        "k_count": metrics.k_count,
        "bb_pct": metrics.bb_pct,
        "k_pct": metrics.k_pct,
        "batted_ball_count": metrics.batted_ball_count,
        "pa": metrics.plate_appearances,
        "plate_appearances": metrics.plate_appearances,
        "bip": metrics.batted_ball_count,
        "avg_hit_speed": metrics.avg_hit_speed,
        "ev95plus": metrics.ev95plus,
        "ev95percent": metrics.ev95percent,
        "sweet_spot_pct": metrics.sweet_spot_pct,
        "source_window_start_date": source_window_start_date,
        "source_window_end_date": source_window_end_date,
        "metric_version": metric_version,
    }


def _inputs_fingerprint(
    *,
    raw_cache: RawSavantEventsCache,
    season: int,
    season_start_date: str,
    as_of_date: str,
    metric_version: str,
    entity: str,
) -> str:
    events = raw_cache.get_events_by_date_range(
        start_date=season_start_date,
        end_date=as_of_date,
    )
    raw_event_count = len(events)
    distinct_dates = len({event.game_date for event in events})
    payload = {
        "distinct_dates": distinct_dates,
        "entity": entity,
        "metric_version": metric_version,
        "raw_event_count": raw_event_count,
        "season": season,
        "season_start_date": season_start_date,
        "source_window_end_date": as_of_date,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return (
        f"savant:raw:offense:{entity}:{metric_version}:{season}:"
        f"{season_start_date}:{as_of_date}:{raw_event_count}:{distinct_dates}:{digest}"
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_div(numerator: float | None, denominator: float | int | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _pct(numerator: int, denominator: int) -> float | None:
    value = _safe_div(float(numerator), denominator)
    return value * 100.0 if value is not None else None


def _rate(numerator: int, denominator: int) -> float | None:
    return _safe_div(float(numerator), denominator)


def _plate_appearance_events(events: list[Any]) -> list[Any]:
    by_pa: dict[tuple[int, int], Any] = {}
    for event in events:
        if not _is_plate_appearance_event(event):
            continue
        key = (event.game_pk, event.at_bat_number)
        existing = by_pa.get(key)
        if existing is None or event.pitch_number >= existing.pitch_number:
            by_pa[key] = event
    return list(by_pa.values())


def _is_plate_appearance_event(event: Any) -> bool:
    label = _event_label(event)
    if not label:
        return False
    if event.woba_denom is not None:
        return True
    return label in _PA_EVENT_LABELS


def _event_label(event: Any) -> str:
    raw = getattr(event, "raw_json", None)
    return str(event.events or (raw.get("events") if raw else None) or "").strip().lower()


_PA_EVENT_LABELS = {
    "single",
    "double",
    "triple",
    "home_run",
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "fielders_choice",
    "fielders_choice_out",
    "double_play",
    "strikeout",
    "strikeout_double_play",
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "sac_fly",
    "sac_fly_double_play",
    "sac_bunt",
    "sac_bunt_double_play",
    "catcher_interf",
    "field_error",
}
