"""Rolling Savant PIT pitcher metrics from raw daily events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .raw_savant_events_cache import RawSavantEventsCache
from .savant_daily_aggregator import SavantDailyAggregator


@dataclass(frozen=True)
class SavantRollingPitcherMetrics:
    as_of_date: str
    pitcher: int
    est_woba: float | None
    woba: float | None
    brl_percent: float | None
    barrel_count: int
    batted_ball_count: int
    pa: int
    bip: int
    avg_hit_speed: float | None
    ev95percent: float | None
    sweet_spot_pct: float | None


class SavantRollingPITBuilder:
    """Build cumulative per-pitcher Savant metrics at a PIT cutoff date."""

    def __init__(
        self,
        cache_db: Path | str | None = None,
        *,
        cache: RawSavantEventsCache | None = None,
        daily_aggregator: SavantDailyAggregator | None = None,
    ):
        if daily_aggregator is not None:
            self.daily_aggregator = daily_aggregator
        else:
            if cache_db is None and cache is None:
                raise ValueError("cache_db is required unless cache or daily_aggregator is provided")
            self.daily_aggregator = SavantDailyAggregator(cache_db, cache=cache)

    def build_for_as_of_date(
        self,
        *,
        season_start_date: str,
        as_of_date: str,
    ) -> dict[int, SavantRollingPitcherMetrics]:
        daily_rows = self.daily_aggregator.aggregate_by_date_range(
            start_date=season_start_date,
            end_date=as_of_date,
        )
        accumulators: dict[int, _Accumulator] = {}

        for row in daily_rows:
            if row.game_date > as_of_date:
                continue
            accumulator = accumulators.setdefault(row.pitcher, _Accumulator())
            accumulator.add(row)

        return {
            pitcher: accumulator.to_metrics(as_of_date=as_of_date, pitcher=pitcher)
            for pitcher, accumulator in sorted(accumulators.items())
        }

    def build_pitcher_for_as_of_date(
        self,
        *,
        pitcher: int,
        season_start_date: str,
        as_of_date: str,
    ) -> SavantRollingPitcherMetrics | None:
        return self.build_for_as_of_date(
            season_start_date=season_start_date,
            as_of_date=as_of_date,
        ).get(pitcher)


class _Accumulator:
    def __init__(self):
        self.est_woba_sum = 0.0
        self.est_woba_count = 0
        self.woba_numerator = 0.0
        self.woba_denominator = 0.0
        self.barrel_count = 0
        self.batted_ball_count = 0
        self.pa = 0
        self.bip = 0
        self.hit_speed_sum = 0.0
        self.hit_speed_count = 0
        self.ev95plus = 0
        self.sweet_spot_count = 0
        self.sweet_spot_denominator = 0

    def add(self, row) -> None:
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
        self.batted_ball_count += row.batted_ball_count
        self.pa += row.pa
        self.bip += row.bip
        self.ev95plus += row.ev95plus

    def to_metrics(self, *, as_of_date: str, pitcher: int) -> SavantRollingPitcherMetrics:
        return SavantRollingPitcherMetrics(
            as_of_date=as_of_date,
            pitcher=pitcher,
            est_woba=_safe_div(self.est_woba_sum, self.est_woba_count),
            woba=_safe_div(self.woba_numerator, self.woba_denominator),
            brl_percent=_pct(self.barrel_count, self.batted_ball_count),
            barrel_count=self.barrel_count,
            batted_ball_count=self.batted_ball_count,
            pa=self.pa,
            bip=self.bip,
            avg_hit_speed=_safe_div(self.hit_speed_sum, self.hit_speed_count),
            ev95percent=_pct(self.ev95plus, self.batted_ball_count),
            sweet_spot_pct=_pct(self.sweet_spot_count, self.sweet_spot_denominator),
        )


def _safe_div(numerator: float, denominator: float | int) -> float | None:
    if not denominator:
        return None
    return numerator / denominator


def _pct(numerator: int, denominator: int) -> float | None:
    value = _safe_div(float(numerator), denominator)
    return value * 100.0 if value is not None else None
