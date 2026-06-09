"""Daily pitcher aggregation from raw Savant/Statcast events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .raw_savant_events_cache import RawSavantEvent, RawSavantEventsCache


@dataclass(frozen=True)
class SavantDailyPitcherMetrics:
    game_date: str
    pitcher: int
    pitches: int
    pa: int
    bip: int
    batted_ball_count: int
    est_woba: float | None
    est_woba_count: int
    woba: float | None
    woba_numerator: float | None
    woba_denominator: float | None
    barrel_count: int
    brl_percent: float | None
    avg_hit_speed: float | None
    ev95plus: int
    ev95percent: float | None
    sweet_spot_count: int
    sweet_spot_denominator: int
    sweet_spot_pct: float | None


class SavantDailyAggregator:
    """Build per-pitcher daily metrics from raw Savant event storage."""

    def __init__(self, cache_db: Path | str | None = None, *, cache: RawSavantEventsCache | None = None):
        if cache_db is None and cache is None:
            raise ValueError("cache_db is required unless cache is provided")
        self.cache = cache or RawSavantEventsCache(cache_db)  # type: ignore[arg-type]

    def aggregate_by_date_range(
        self,
        *,
        start_date: str,
        end_date: str,
    ) -> list[SavantDailyPitcherMetrics]:
        events = self.cache.get_events_by_date_range(start_date=start_date, end_date=end_date)
        grouped: dict[tuple[str, int], list[RawSavantEvent]] = defaultdict(list)

        for event in events:
            if event.pitcher is None:
                continue
            grouped[(event.game_date, event.pitcher)].append(event)

        return [
            _aggregate_one(game_date=game_date, pitcher=pitcher, events=pitcher_events)
            for (game_date, pitcher), pitcher_events in sorted(grouped.items())
        ]


def _aggregate_one(
    *,
    game_date: str,
    pitcher: int,
    events: list[RawSavantEvent],
) -> SavantDailyPitcherMetrics:
    batted_ball_events = [event for event in events if event.launch_speed is not None]
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
    ev95plus = sum(1 for value in launch_speeds if value >= 95.0)
    sweet_spot_count = sum(1 for value in launch_angles if 8.0 <= value <= 32.0)

    return SavantDailyPitcherMetrics(
        game_date=game_date,
        pitcher=pitcher,
        pitches=len(events),
        pa=int(woba_denominator) if woba_denominator is not None else 0,
        bip=len(batted_ball_events),
        batted_ball_count=len(batted_ball_events),
        est_woba=_mean(est_woba_values),
        est_woba_count=len(est_woba_values),
        woba=_safe_div(woba_numerator, woba_denominator),
        woba_numerator=woba_numerator,
        woba_denominator=woba_denominator,
        barrel_count=barrel_count,
        brl_percent=_pct(barrel_count, len(batted_ball_events)),
        avg_hit_speed=_mean(launch_speeds),
        ev95plus=ev95plus,
        ev95percent=_pct(ev95plus, len(launch_speeds)),
        sweet_spot_count=sweet_spot_count,
        sweet_spot_denominator=len(launch_angles),
        sweet_spot_pct=_pct(sweet_spot_count, len(launch_angles)),
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _pct(numerator: int, denominator: int) -> float | None:
    value = _safe_div(float(numerator), float(denominator))
    return value * 100.0 if value is not None else None
