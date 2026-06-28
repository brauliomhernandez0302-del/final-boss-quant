"""Adapter for Bullpen PIT facts; intentionally disconnected from lambda."""

from __future__ import annotations

from typing import Any


ADAPTER_VERSION = "bullpen_pit_adapter_v1"
QUALITY_FIELDS = (
    "relief_appearances",
    "relief_team_games",
    "relief_pitch_count",
    "relief_batters_faced",
    "strikeouts",
    "walks",
    "k_pct",
    "bb_pct",
    "k_minus_bb_pct",
    "xwoba_against",
    "xwoba_count",
    "woba_against",
    "barrel_count",
    "contact_count",
    "barrel_per_contact",
)
WORKLOAD_FIELDS = (
    "pitches_last_1_day",
    "appearances_last_1_day",
    "pitches_last_3_days",
    "appearances_last_3_days",
    "pitches_last_7_days",
    "appearances_last_7_days",
    "consecutive_days",
    "last_used_date",
)


def adapt_bullpen_pit_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Apply current → prior → neutral while keeping multiplier exactly 1.0."""
    current = snapshot.get("current") or {}
    prior = snapshot.get("prior") or {}
    current_available = bool(
        snapshot.get("current_bullpen_found")
        and int(current.get("relief_batters_faced") or 0) > 0
    )
    prior_available = bool(
        snapshot.get("prior_baseline_found")
        and prior.get("baseline_available")
        and int(prior.get("relief_batters_faced") or 0) >= int(
            prior.get("minimum_bf_required") or 200
        )
    )

    if current_available:
        selected = current
        provenance_source = "current_bullpen_pit"
        fallback_used = None
        neutral_fallback = False
    elif prior_available:
        selected = prior
        provenance_source = "prior_season_bullpen_baseline"
        fallback_used = "current_bullpen_pit_unavailable"
        neutral_fallback = False
    else:
        selected = {}
        provenance_source = "neutral_bullpen_adjustment"
        fallback_used = "current_and_prior_bullpen_unavailable"
        neutral_fallback = True

    quality = {field: selected.get(field) for field in QUALITY_FIELDS}
    workload = {
        field: (current.get(field) if provenance_source == "current_bullpen_pit" else None)
        for field in WORKLOAD_FIELDS
    }
    return {
        "found": not neutral_fallback,
        "team_id": snapshot.get("team_id"),
        "season": snapshot.get("season"),
        "provenance_source": provenance_source,
        "fallback_used": fallback_used,
        "neutral_fallback": neutral_fallback,
        "applied_multiplier": 1.0,
        "quality_metrics": quality,
        "workload_facts": workload,
        "sample_size_status": (
            selected.get("sample_size_status") if selected else "neutral"
        ),
        "current_bullpen_found": bool(snapshot.get("current_bullpen_found")),
        "prior_baseline_found": bool(snapshot.get("prior_baseline_found")),
        "prior_baseline_available": prior_available,
        "source_window_start_date": selected.get("source_window_start_date"),
        "source_window_end_date": selected.get("source_window_end_date"),
        "requested_as_of_date": snapshot.get("requested_as_of_date"),
        "selected_as_of_date": (
            snapshot.get("current_as_of_date")
            if provenance_source == "current_bullpen_pit"
            else snapshot.get("prior_baseline_as_of_date")
            if provenance_source == "prior_season_bullpen_baseline"
            else None
        ),
        "source_fingerprints": snapshot.get("source_fingerprints", {}),
        "provenance": {
            "adapter_version": ADAPTER_VERSION,
            "snapshot_version": snapshot.get("snapshot_version"),
            "requested_as_of_date": snapshot.get("requested_as_of_date"),
            "current_as_of_date": snapshot.get("current_as_of_date"),
            "prior_baseline_as_of_date": snapshot.get(
                "prior_baseline_as_of_date"
            ),
            "source_fingerprints": snapshot.get("source_fingerprints", {}),
            "starter_statistics_used": False,
            "roster_fallback_used": False,
            "legacy_bullpen_fallback_used": False,
            "full_season_leaderboard_used": False,
            "lambda_integration_enabled": False,
            "neutral_multiplier": 1.0,
        },
    }
