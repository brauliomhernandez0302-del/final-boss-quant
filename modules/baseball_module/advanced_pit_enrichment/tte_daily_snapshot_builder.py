"""Daily point-in-time Team/TTE snapshot construction.

This module defines the experimental Team/TTE PIT contract and populates the
team offense portion from canonical Savant rolling snapshots. It does not
calculate True Talent offense and is intentionally disconnected from live and
backtest paths.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .pit_cache import PITCache


class TTEPITNamespaces:
    """Canonical PIT namespaces reserved for experimental Team/TTE inputs."""

    BATTER_ROLLING = "savant.batter.rolling"
    TEAM_OFFENSE_ROLLING = "savant.team_offense.rolling"
    TTE_TEAM_DAILY = "tte.team.daily"


class TTEPITSources:
    """Canonical source labels for experimental Team/TTE PIT inputs."""

    BATTER_ROLLING = "baseball_savant_batter_rolling"
    TEAM_OFFENSE_ROLLING = "baseball_savant_team_offense_rolling"
    TTE_TEAM_DAILY = "tte_team_daily"


class TTEDailySnapshotBuilder:
    """Read the experimental Team/TTE PIT snapshot contract from PITCache."""

    SNAPSHOT_VERSION = "tte_daily_snapshot_v1"

    def __init__(self, pit_cache: PITCache | None = None, *, cache_db: Path | str | None = None):
        if pit_cache is None and cache_db is None:
            raise ValueError("pit_cache or cache_db is required")
        self.cache = pit_cache or PITCache(cache_db)  # type: ignore[arg-type]

    def build_for_game(
        self,
        *,
        team_id: int | str,
        season: int,
        game_date: str,
        team_name: str | None = None,
    ) -> dict[str, Any]:
        """Build using the previous calendar day as the requested PIT cutoff."""
        return self.build_team_snapshot(
            team_id=team_id,
            season=season,
            requested_as_of_date=previous_day_cutoff_for_game_date(game_date),
            team_name=team_name,
        )

    def build_team_snapshot(
        self,
        *,
        team_id: int | str,
        season: int,
        requested_as_of_date: str,
        team_name: str | None = None,
    ) -> dict[str, Any]:
        """Return the latest Team/TTE PIT snapshot at or before cutoff."""
        tte_record = self.cache.get_latest(
            namespace=TTEPITNamespaces.TTE_TEAM_DAILY,
            entity_id=team_id,
            season=season,
            as_of_date=requested_as_of_date,
            source=TTEPITSources.TTE_TEAM_DAILY,
        )
        team_record = self.cache.get_latest(
            namespace=TTEPITNamespaces.TEAM_OFFENSE_ROLLING,
            entity_id=team_id,
            season=season,
            as_of_date=requested_as_of_date,
            source=TTEPITSources.TEAM_OFFENSE_ROLLING,
        )
        batter_record = self.cache.get_latest(
            namespace=TTEPITNamespaces.BATTER_ROLLING,
            entity_id=team_id,
            season=season,
            as_of_date=requested_as_of_date,
            source=TTEPITSources.BATTER_ROLLING,
        )

        data = tte_record.data if tte_record else {}
        team_data = team_record.data if team_record else {}
        snapshot = {
            "found": team_record is not None,
            "team_id": _coerce_int(team_id),
            "team_name": data.get("team_name", team_data.get("team_name", team_name)),
            "season": int(season),
            "requested_as_of_date": requested_as_of_date,
            "team_offense_as_of_date": team_record.as_of_date if team_record else None,
            "batter_rolling_found": batter_record is not None,
            "team_rolling_found": team_record is not None,
            "lambda_offense": None,
            "runs_per_game": _first_present(team_data, "runs_per_game"),
            "team_est_woba": _first_present(team_data, "team_est_woba", "est_woba"),
            "team_woba": _first_present(team_data, "team_woba", "woba"),
            "team_brl_percent": _first_present(team_data, "team_brl_percent", "brl_percent"),
            "team_ev95percent": _first_present(team_data, "team_ev95percent", "ev95percent"),
            "barrel_pa": _first_present(team_data, "barrel_pa", "team_barrel_pa"),
            "bb_pct": _first_present(team_data, "bb_pct", "team_bb_pct"),
            "k_pct": _first_present(team_data, "k_pct", "team_k_pct"),
            "pa": _first_present(team_data, "pa", "plate_appearances"),
            "bip": _first_present(team_data, "bip", "batted_ball_count"),
            "source_fingerprints": {
                "tte_team_daily": tte_record.source_fingerprint if tte_record else None,
                "savant_team_offense_rolling": (
                    team_record.source_fingerprint if team_record else None
                ),
                "savant_batter_rolling": batter_record.source_fingerprint if batter_record else None,
            },
            "snapshot_version": self.SNAPSHOT_VERSION,
        }
        return snapshot


def previous_day_cutoff_for_game_date(game_date: str) -> str:
    """Map game date D to previous calendar day at 23:59:59 UTC."""
    game_day = datetime.strptime(str(game_date)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return (game_day - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_int(value: int | str) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _first_present(primary: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in primary:
            return primary[key]
    return None
