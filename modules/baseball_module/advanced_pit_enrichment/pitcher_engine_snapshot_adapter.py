"""Adapt unified PIT pitcher snapshots to the Pitcher Engine input shape."""

from __future__ import annotations

from typing import Any


PITCHER_ENGINE_FIELDS = (
    "name",
    "mlbam_id",
    "siera",
    "xfip",
    "xera",
    "fip",
    "era",
    "innings_pitched",
    "est_woba",
    "brl_percent",
    "k_pct",
    "bb_pct",
    "era_last_5",
    "era_trend",
    "days_rest",
    "last_pitch_count",
    "quality_start_pct",
    "avg_innings_per_start",
    "platoon_splits",
    "era_vs_opp",
    "ip_vs_opp",
    "whip",
)


OPTIONAL_ENGINE_FIELDS = (
    "era",
    "era_last_5",
    "era_trend",
    "days_rest",
    "last_pitch_count",
    "quality_start_pct",
    "avg_innings_per_start",
    "platoon_splits",
    "era_vs_opp",
    "ip_vs_opp",
    "whip",
)


def adapt_unified_pitcher_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return a Pitcher Engine-ready dict without fabricating metric values."""
    adapted = dict(snapshot)

    if adapted.get("name") is None and adapted.get("player_name") is not None:
        adapted["name"] = adapted.get("player_name")

    if adapted.get("innings_pitched") is None and "ip" in adapted:
        adapted["innings_pitched"] = adapted.get("ip")
    elif adapted.get("ip") is None and "innings_pitched" in adapted:
        adapted["ip"] = adapted.get("innings_pitched")

    for field in OPTIONAL_ENGINE_FIELDS:
        adapted.setdefault(field, None)

    return adapted
