"""Advanced pitcher point-in-time snapshot construction.

This builder composes already-cached PIT sources into the dict shape consumed by
the existing Pitcher Engine. It is intentionally not connected to live or
backtest entrypoints yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .fangraphs_pit_fetcher import FanGraphsPITFetcher
from .pit_cache import PITCacheRecord
from .savant_pit_fetcher import SavantPITFetcher


class AdvancedPitcherSnapshotBuilder:
    """Build Pitcher Engine-compatible snapshots from isolated PIT caches."""

    def __init__(
        self,
        cache_db: Path | str | None = None,
        *,
        fangraphs_fetcher: FanGraphsPITFetcher | None = None,
        savant_fetcher: SavantPITFetcher | None = None,
    ):
        if cache_db is None and (fangraphs_fetcher is None or savant_fetcher is None):
            raise ValueError("cache_db is required unless both fetchers are provided")

        self.fangraphs = fangraphs_fetcher or FanGraphsPITFetcher(cache_db)  # type: ignore[arg-type]
        self.savant = savant_fetcher or SavantPITFetcher(cache_db)  # type: ignore[arg-type]

    def build_pitcher_snapshot(
        self,
        *,
        mlbam_id: int | str,
        season: int,
        as_of_date: str,
        name: str | None = None,
        base_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a Pitcher Engine-compatible snapshot for one pitcher.

        `base_stats` is where historical MLB Stats API game-log fields belong:
        ERA, FIP, WHIP, days rest, last pitch count, recent form, matchup, and
        platoon splits. FanGraphs/Savant PIT values then add advanced metrics
        without overwriting existing valid base values with nulls.
        """
        fg_record = self.fangraphs.cache.get_latest(
            namespace=self.fangraphs.PITCHER_NAMESPACE,
            entity_id=mlbam_id,
            season=season,
            as_of_date=as_of_date,
            source=self.fangraphs.SOURCE,
        )
        savant_record = self.savant.cache.get_latest(
            namespace=self.savant.PITCHER_NAMESPACE,
            entity_id=mlbam_id,
            season=season,
            as_of_date=as_of_date,
            source=self.savant.SOURCE,
        )

        snapshot = dict(base_stats or {})
        snapshot.setdefault("name", name or snapshot.get("name") or _record_name(fg_record, savant_record))
        snapshot["mlbam_id"] = str(mlbam_id)

        if fg_record:
            _merge_present(
                snapshot,
                {
                    "xfip": fg_record.data.get("xfip"),
                    "siera": fg_record.data.get("siera"),
                    "war": fg_record.data.get("war"),
                    "k_pct": _as_rate(fg_record.data.get("k_pct")),
                    "bb_pct": _as_rate(fg_record.data.get("bb_pct")),
                    "k_bb_pct": _as_rate(fg_record.data.get("k_bb_pct")),
                    "swstr_pct": _as_rate(fg_record.data.get("swstr_pct")),
                    "babip": fg_record.data.get("babip"),
                    "lob_pct": fg_record.data.get("lob_pct"),
                    "hr_fb_pct": fg_record.data.get("hr_fb"),
                    "innings_pitched": fg_record.data.get("ip"),
                },
            )
            _set_if_missing(snapshot, "fip", fg_record.data.get("fip"))
            _set_if_missing(snapshot, "era", fg_record.data.get("era"))
            _set_if_missing(snapshot, "xera", fg_record.data.get("xera"))

        if savant_record:
            _merge_present(
                snapshot,
                {
                    "est_woba": savant_record.data.get("est_woba"),
                    "brl_percent": savant_record.data.get("brl_percent"),
                    "brl_pa": savant_record.data.get("brl_pa"),
                    "avg_hit_speed": savant_record.data.get("avg_hit_speed"),
                    "max_hit_speed": savant_record.data.get("max_hit_speed"),
                    "avg_hit_angle": savant_record.data.get("avg_hit_angle"),
                    "ev95percent": savant_record.data.get("ev95percent"),
                    "ev95plus": savant_record.data.get("ev95plus"),
                    "sweet_spot_pct": savant_record.data.get("sweet_spot_pct"),
                    "savant_pa": savant_record.data.get("pa"),
                    "savant_bip": savant_record.data.get("bip"),
                    "savant_pitches": savant_record.data.get("pitches"),
                },
            )

        snapshot["pit_metadata"] = {
            "as_of_date": as_of_date,
            "fangraphs": _record_metadata(fg_record),
            "savant": _record_metadata(savant_record),
        }
        return snapshot

    def build_for_game(
        self,
        *,
        season: int,
        as_of_date: str,
        home_pitcher_id: int | str,
        away_pitcher_id: int | str,
        home_pitcher_name: str | None = None,
        away_pitcher_name: str | None = None,
        home_base_stats: dict[str, Any] | None = None,
        away_base_stats: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Build both home and away starter snapshots without pipeline wiring."""
        return {
            "pitcher_home": self.build_pitcher_snapshot(
                mlbam_id=home_pitcher_id,
                season=season,
                as_of_date=as_of_date,
                name=home_pitcher_name,
                base_stats=home_base_stats,
            ),
            "pitcher_away": self.build_pitcher_snapshot(
                mlbam_id=away_pitcher_id,
                season=season,
                as_of_date=as_of_date,
                name=away_pitcher_name,
                base_stats=away_base_stats,
            ),
        }


def _merge_present(target: dict[str, Any], values: dict[str, Any]) -> None:
    target.update({key: value for key, value in values.items() if value is not None})


def _set_if_missing(target: dict[str, Any], key: str, value: Any) -> None:
    if target.get(key) is None and value is not None:
        target[key] = value


def _as_rate(value: Any) -> float | None:
    if value is None:
        return None
    rate = float(value)
    return rate / 100.0 if rate > 1.0 else rate


def _record_name(*records: PITCacheRecord | None) -> str:
    for record in records:
        if not record:
            continue
        name = record.data.get("fg_name") or record.data.get("player_name")
        if name:
            return str(name)
    return "Unknown"


def _record_metadata(record: PITCacheRecord | None) -> dict[str, Any]:
    if not record:
        return {"found": False}
    return {
        "found": True,
        "as_of_date": record.as_of_date,
        "source_fingerprint": record.source_fingerprint,
        "fetched_at": record.fetched_at,
    }
