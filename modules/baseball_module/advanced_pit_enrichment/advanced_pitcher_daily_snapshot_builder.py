"""Unified daily point-in-time pitcher snapshot construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .fangraphs_daily_pit_persistence import FanGraphsDailyPITPersistence
from .pit_cache import PITCache, PITCacheRecord
from .pitcher_prior_baseline import PitcherPriorBaselinePersistence
from .savant_rolling_pit_persistence import SavantRollingPITPersistence


class AdvancedPitcherDailySnapshotBuilder:
    """Merge isolated FanGraphs daily and Savant rolling PIT rows."""

    SNAPSHOT_VERSION = "advanced_pitcher_daily_snapshot_v1"

    def __init__(self, pit_cache: PITCache | None = None, *, cache_db: Path | str | None = None):
        if pit_cache is None and cache_db is None:
            raise ValueError("pit_cache or cache_db is required")
        self.cache = pit_cache or PITCache(cache_db)  # type: ignore[arg-type]

    def build_pitcher_snapshot(
        self,
        *,
        pitcher: int | str,
        season: int,
        requested_as_of_date: str,
    ) -> dict[str, Any]:
        fangraphs_record = self.cache.get_latest(
            namespace=FanGraphsDailyPITPersistence.NAMESPACE,
            entity_id=pitcher,
            season=season,
            as_of_date=requested_as_of_date,
            source=FanGraphsDailyPITPersistence.SOURCE,
        )
        savant_record = self.cache.get_latest(
            namespace=SavantRollingPITPersistence.NAMESPACE,
            entity_id=pitcher,
            season=season,
            as_of_date=requested_as_of_date,
            source=SavantRollingPITPersistence.SOURCE,
        )
        prior_record = None
        if fangraphs_record is None and savant_record is None:
            prior_record = self.cache.get_latest(
                namespace=PitcherPriorBaselinePersistence.NAMESPACE,
                entity_id=pitcher,
                season=season,
                as_of_date=requested_as_of_date,
                source=PitcherPriorBaselinePersistence.SOURCE,
            )

        snapshot: dict[str, Any] = {
            "found": bool(fangraphs_record or savant_record or prior_record),
            "mlbam_id": _coerce_int(pitcher),
            "requested_as_of_date": requested_as_of_date,
            "fangraphs_found": fangraphs_record is not None,
            "savant_found": savant_record is not None,
            "prior_baseline_found": prior_record is not None,
            "fangraphs_as_of_date": fangraphs_record.as_of_date if fangraphs_record else None,
            "savant_as_of_date": savant_record.as_of_date if savant_record else None,
            "prior_baseline_as_of_date": prior_record.as_of_date if prior_record else None,
            "provenance_source": (
                "current_pit"
                if fangraphs_record or savant_record
                else "prior_season_baseline"
                if prior_record
                else "league_average_safe_fallback"
            ),
            "source_fingerprints": {
                "fangraphs": fangraphs_record.source_fingerprint if fangraphs_record else None,
                "savant": savant_record.source_fingerprint if savant_record else None,
                "prior_season_baseline": (
                    prior_record.source_fingerprint if prior_record else None
                ),
            },
            "snapshot_version": self.SNAPSHOT_VERSION,
        }

        if fangraphs_record:
            _merge_fangraphs(snapshot, fangraphs_record.data)
        if savant_record:
            _merge_savant(snapshot, savant_record.data)
        if prior_record:
            _merge_savant(snapshot, prior_record.data)
            snapshot["prior_season"] = prior_record.data.get("prior_season")

        return snapshot


def _merge_fangraphs(snapshot: dict[str, Any], data: dict[str, Any]) -> None:
    snapshot.update(
        {
            "siera": data.get("siera"),
            "xfip": data.get("xfip"),
            "xera": data.get("xera"),
            "fip": data.get("fip"),
            "k_pct": _as_rate(data.get("k_pct")),
            "bb_pct": _as_rate(data.get("bb_pct")),
            "innings_pitched": data.get("ip"),
            "ip": data.get("ip"),
            "player_name": data.get("player_name"),
            "fg_playerid": data.get("fg_playerid"),
            "mlbam_id": data.get("mlbam_id", snapshot.get("mlbam_id")),
        }
    )


def _merge_savant(snapshot: dict[str, Any], data: dict[str, Any]) -> None:
    pa = data.get("pa")
    bip = data.get("bip")
    snapshot.update(
        {
            "est_woba": data.get("est_woba"),
            "woba": data.get("woba"),
            "brl_percent": data.get("brl_percent"),
            "barrel_count": data.get("barrel_count"),
            "batted_ball_count": data.get("batted_ball_count"),
            "ev95percent": data.get("ev95percent"),
            "sweet_spot_pct": data.get("sweet_spot_pct"),
            "pa": pa,
            "bip": bip,
            "savant_pa": pa,
            "savant_bip": bip,
        }
    )


def _as_rate(value: Any) -> float | None:
    if value is None:
        return None
    rate = float(value)
    return rate / 100.0 if rate > 1.0 else rate


def _coerce_int(value: int | str) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _record_metadata(record: PITCacheRecord | None) -> dict[str, Any]:
    if record is None:
        return {"found": False}
    return {
        "found": True,
        "as_of_date": record.as_of_date,
        "source_fingerprint": record.source_fingerprint,
    }
