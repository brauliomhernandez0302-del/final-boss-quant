"""Leakage-safe prior-season pitcher baseline persistence."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pit_cache import PITCache
from .raw_savant_events_cache import RawSavantEventsCache, RawSavantPitcherMetricEvent
from .savant_rolling_pit_builder import SavantRollingPITBuilder, SavantRollingPitcherMetrics


class PitcherPriorBaselinePersistence:
    """Persist prior-season Savant pitcher profiles for a target season."""

    NAMESPACE = "pitcher.prior_season.baseline"
    SOURCE = "baseball_savant_pitcher_prior_baseline"
    METRIC_VERSION = "pitcher_prior_baseline_v1"

    def __init__(
        self,
        *,
        raw_cache_db: Path | str | None = None,
        pit_cache_db: Path | str | None = None,
        raw_cache: RawSavantEventsCache | None = None,
        pit_cache: PITCache | None = None,
        builder: SavantRollingPITBuilder | None = None,
    ):
        if raw_cache_db is None and raw_cache is None and builder is None:
            raise ValueError("raw_cache_db, raw_cache, or builder is required")
        if pit_cache_db is None and pit_cache is None:
            raise ValueError("pit_cache_db or pit_cache is required")

        self.raw_cache = raw_cache or (
            RawSavantEventsCache(raw_cache_db) if raw_cache_db is not None else None
        )
        self.pit_cache = pit_cache or PITCache(pit_cache_db)  # type: ignore[arg-type]
        self.builder = builder or SavantRollingPITBuilder(cache=self.raw_cache)
        self.last_build_report: dict[str, Any] | None = None

    def persist_prior_baseline(
        self,
        *,
        target_season: int,
        prior_season: int,
        prior_season_start_date: str,
        prior_season_end_date: str,
        fetched_at: str | None = None,
    ) -> dict[int, dict[str, Any]]:
        self._validate_window(
            target_season=target_season,
            prior_season=prior_season,
            start_date=prior_season_start_date,
            end_date=prior_season_end_date,
        )
        started = time.perf_counter()
        if self.raw_cache is not None:
            metrics, aggregation_report = self._build_one_pass(
                start_date=prior_season_start_date,
                end_date=prior_season_end_date,
            )
        else:
            # Retained for injected builders in unit tests and callers. Normal
            # raw-cache persistence always takes the bounded one-pass path.
            metrics = self.builder.build_for_as_of_date(
                season_start_date=prior_season_start_date,
                as_of_date=prior_season_end_date,
            )
            aggregation_report = {
                "events_scanned": None,
                "duplicate_plate_appearances": None,
            }
        fingerprint = self.inputs_fingerprint(
            target_season=target_season,
            prior_season=prior_season,
            start_date=prior_season_start_date,
            end_date=prior_season_end_date,
            row_count=len(metrics),
        )
        baseline_as_of = f"{prior_season_end_date}T23:59:59Z"
        fetched = fetched_at or datetime.now(timezone.utc).isoformat()
        persisted: dict[int, dict[str, Any]] = {}

        for pitcher, row in metrics.items():
            data = _payload(
                row,
                prior_season=prior_season,
                start_date=prior_season_start_date,
                end_date=prior_season_end_date,
            )
            self.pit_cache.save_record(
                namespace=self.NAMESPACE,
                entity_id=pitcher,
                season=target_season,
                as_of_date=baseline_as_of,
                source=self.SOURCE,
                source_fingerprint=fingerprint,
                data=data,
                fetched_at=fetched,
            )
            persisted[pitcher] = data
        self.last_build_report = {
            **aggregation_report,
            "pitchers_persisted": len(persisted),
            "metric_null_count": sum(
                value is None
                for data in persisted.values()
                for key, value in data.items()
                if key
                in {
                    "est_woba",
                    "woba",
                    "brl_percent",
                    "avg_hit_speed",
                    "ev95percent",
                    "sweet_spot_pct",
                }
            ),
            "source_window_start_date": prior_season_start_date,
            "source_window_end_date": prior_season_end_date,
            "elapsed_sec": time.perf_counter() - started,
        }
        return persisted

    def _build_one_pass(
        self,
        *,
        start_date: str,
        end_date: str,
    ) -> tuple[dict[int, SavantRollingPitcherMetrics], dict[str, int]]:
        if self.raw_cache is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("raw_cache is required for one-pass aggregation")

        accumulators: dict[int, _PriorAccumulator] = {}
        events_scanned = 0
        for event in self.raw_cache.iter_pitcher_metric_events_by_date_range(
            start_date=start_date,
            end_date=end_date,
        ):
            events_scanned += 1
            accumulators.setdefault(event.pitcher, _PriorAccumulator()).add(event)

        metrics = {
            pitcher: accumulator.to_metrics(as_of_date=end_date, pitcher=pitcher)
            for pitcher, accumulator in sorted(accumulators.items())
        }
        return metrics, {
            "events_scanned": events_scanned,
            "duplicate_plate_appearances": sum(
                accumulator.duplicate_plate_appearances
                for accumulator in accumulators.values()
            ),
        }

    def inputs_fingerprint(
        self,
        *,
        target_season: int,
        prior_season: int,
        start_date: str,
        end_date: str,
        row_count: int,
    ) -> str:
        payload = {
            "metric_version": self.METRIC_VERSION,
            "target_season": target_season,
            "prior_season": prior_season,
            "source_window_start_date": start_date,
            "source_window_end_date": end_date,
            "row_count": row_count,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return (
            "savant:raw:pitcher_prior:"
            f"{self.METRIC_VERSION}:{target_season}:{prior_season}:"
            f"{start_date}:{end_date}:{row_count}:{digest}"
        )

    @staticmethod
    def _validate_window(
        *,
        target_season: int,
        prior_season: int,
        start_date: str,
        end_date: str,
    ) -> None:
        if prior_season != target_season - 1:
            raise ValueError("prior_season must equal target_season - 1")
        try:
            parsed_start = datetime.strptime(start_date, "%Y-%m-%d").date()
            parsed_end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("prior baseline dates must be YYYY-MM-DD") from exc
        if parsed_start.year != prior_season:
            raise ValueError("prior baseline start date must be inside prior_season")
        if parsed_end.year != prior_season:
            raise ValueError("prior baseline end date must be inside prior_season")
        if parsed_end < parsed_start:
            raise ValueError("prior baseline end date must be on or after start date")


def _payload(
    metrics: SavantRollingPitcherMetrics,
    *,
    prior_season: int,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    return {
        "prior_season": prior_season,
        "est_woba": metrics.est_woba,
        "woba": metrics.woba,
        "brl_percent": metrics.brl_percent,
        "barrel_count": metrics.barrel_count,
        "batted_ball_count": metrics.batted_ball_count,
        "pa": metrics.pa,
        "bip": metrics.bip,
        "avg_hit_speed": metrics.avg_hit_speed,
        "ev95percent": metrics.ev95percent,
        "sweet_spot_pct": metrics.sweet_spot_pct,
        "source_window_start_date": start_date,
        "source_window_end_date": end_date,
        "metric_version": PitcherPriorBaselinePersistence.METRIC_VERSION,
    }


class _PriorAccumulator:
    """One-pass equivalent of daily aggregation followed by rolling aggregation."""

    def __init__(self) -> None:
        self.est_woba_sum = 0.0
        self.est_woba_count = 0
        self.woba_numerator = 0.0
        self.woba_denominator = 0.0
        self.barrel_count = 0
        self.batted_ball_count = 0
        self.hit_speed_sum = 0.0
        self.hit_speed_count = 0
        self.ev95plus = 0
        self.sweet_spot_count = 0
        self.sweet_spot_denominator = 0
        self.pa_contributions: dict[tuple[int, int], tuple[int, float | None, float]] = {}
        self.duplicate_plate_appearances = 0

    def add(self, event: RawSavantPitcherMetricEvent) -> None:
        if event.estimated_woba_using_speedangle is not None:
            self.est_woba_sum += event.estimated_woba_using_speedangle
            self.est_woba_count += 1

        if event.woba_value is not None or event.woba_denom is not None:
            self._add_plate_appearance(event)

        if event.launch_speed_angle == 6:
            self.barrel_count += 1
        if event.launch_speed is not None:
            self.batted_ball_count += 1
            self.hit_speed_sum += event.launch_speed
            self.hit_speed_count += 1
            if event.launch_speed >= 95.0:
                self.ev95plus += 1
            if event.launch_angle is not None:
                self.sweet_spot_denominator += 1
                if 8.0 <= event.launch_angle <= 32.0:
                    self.sweet_spot_count += 1

    def _add_plate_appearance(self, event: RawSavantPitcherMetricEvent) -> None:
        key = (event.game_pk, event.at_bat_number)
        existing = self.pa_contributions.get(key)
        if existing is not None:
            self.duplicate_plate_appearances += 1
            existing_pitch, existing_value, existing_denom = existing
            if event.pitch_number < existing_pitch:
                return
            self.woba_denominator -= existing_denom
            if existing_value is not None:
                self.woba_numerator -= existing_value

        denom = float(event.woba_denom or 0.0)
        self.pa_contributions[key] = (event.pitch_number, event.woba_value, denom)
        self.woba_denominator += denom
        if event.woba_value is not None:
            self.woba_numerator += event.woba_value

    def to_metrics(self, *, as_of_date: str, pitcher: int) -> SavantRollingPitcherMetrics:
        pa = int(self.woba_denominator)
        return SavantRollingPitcherMetrics(
            as_of_date=as_of_date,
            pitcher=pitcher,
            est_woba=_safe_div(self.est_woba_sum, self.est_woba_count),
            woba=_safe_div(self.woba_numerator, self.woba_denominator),
            woba_numerator=self.woba_numerator if self.woba_denominator else None,
            woba_denominator=self.woba_denominator if self.woba_denominator else None,
            brl_percent=_pct(self.barrel_count, self.batted_ball_count),
            barrel_count=self.barrel_count,
            batted_ball_count=self.batted_ball_count,
            pa=pa,
            bip=self.batted_ball_count,
            avg_hit_speed=_safe_div(self.hit_speed_sum, self.hit_speed_count),
            ev95plus=self.ev95plus,
            ev95percent=_pct(self.ev95plus, self.batted_ball_count),
            sweet_spot_pct=_pct(self.sweet_spot_count, self.sweet_spot_denominator),
        )


def _safe_div(numerator: float, denominator: float | int) -> float | None:
    return numerator / denominator if denominator else None


def _pct(numerator: int, denominator: int) -> float | None:
    value = _safe_div(float(numerator), denominator)
    return value * 100.0 if value is not None else None
