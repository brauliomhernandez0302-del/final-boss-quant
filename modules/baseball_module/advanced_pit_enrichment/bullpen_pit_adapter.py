"""Adapter for Bullpen PIT facts; intentionally disconnected from lambda."""

from __future__ import annotations

from typing import Any


ADAPTER_VERSION = "bullpen_pit_adapter_v2"
SUFFICIENT_BF = 200
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
    current_bf = int(current.get("relief_batters_faced") or 0)
    prior_bf = int(prior.get("relief_batters_faced") or 0)
    current_available = bool(
        snapshot.get("current_bullpen_found")
        and current_bf > 0
    )
    prior_available = bool(
        snapshot.get("prior_baseline_found")
        and prior.get("baseline_available")
        and prior_bf >= int(
            prior.get("minimum_bf_required") or SUFFICIENT_BF
        )
    )

    current_weight = 0.0
    prior_weight = 0.0
    blend_formula = None
    if current_available and current_bf >= SUFFICIENT_BF:
        provenance_source = "current_bullpen_pit"
        fallback_used = None
        neutral_fallback = False
        current_weight = 1.0
        quality = _quality_values(current)
    elif current_available and prior_available:
        provenance_source = "current_prior_bullpen_blend"
        fallback_used = "current_bullpen_pit_thin_blended_with_prior"
        neutral_fallback = False
        current_weight = current_bf / SUFFICIENT_BF
        prior_weight = 1.0 - current_weight
        blend_formula = (
            "current_weight=current_bf/200; prior_weight=1-current_weight; "
            "rate=current_weight*current_rate+prior_weight*prior_rate"
        )
        quality = _blend_quality(current, prior, current_weight, prior_weight)
    elif prior_available:
        provenance_source = "prior_season_bullpen_baseline"
        fallback_used = "current_bullpen_pit_unavailable"
        neutral_fallback = False
        prior_weight = 1.0
        quality = _quality_values(prior)
    else:
        provenance_source = "neutral_bullpen_adjustment"
        fallback_used = (
            "thin_current_without_available_prior"
            if current_available
            else "current_and_prior_bullpen_unavailable"
        )
        neutral_fallback = True
        quality = {field: None for field in QUALITY_FIELDS}

    workload = {
        field: (
            current.get(field)
            if provenance_source
            in {"current_bullpen_pit", "current_prior_bullpen_blend"}
            else None
        )
        for field in WORKLOAD_FIELDS
    }
    selected = (
        current
        if provenance_source in {"current_bullpen_pit", "current_prior_bullpen_blend"}
        else prior
        if provenance_source == "prior_season_bullpen_baseline"
        else {}
    )
    return {
        "found": not neutral_fallback,
        "team_id": snapshot.get("team_id"),
        "season": snapshot.get("season"),
        "provenance_source": provenance_source,
        "fallback_used": fallback_used,
        "neutral_fallback": neutral_fallback,
        "applied_multiplier": 1.0,
        "quality_metrics": quality,
        "current_quality_metrics": _quality_values(current),
        "prior_quality_metrics": _quality_values(prior),
        "workload_facts": workload,
        "sample_size_status": (
            selected.get("sample_size_status") if selected else "neutral"
        ),
        "current_bullpen_found": bool(snapshot.get("current_bullpen_found")),
        "prior_baseline_found": bool(snapshot.get("prior_baseline_found")),
        "prior_baseline_available": prior_available,
        "current_bf": current_bf,
        "prior_bf": prior_bf,
        "current_weight": round(current_weight, 6),
        "prior_weight": round(prior_weight, 6),
        "blend_formula": blend_formula,
        "current_sample_size_status": current.get("sample_size_status"),
        "prior_sample_size_status": prior.get("sample_size_status"),
        "source_window_start_date": selected.get("source_window_start_date"),
        "source_window_end_date": selected.get("source_window_end_date"),
        "current_source_window_start_date": current.get(
            "source_window_start_date"
        ),
        "current_source_window_end_date": current.get("source_window_end_date"),
        "prior_source_window_start_date": prior.get("source_window_start_date"),
        "prior_source_window_end_date": prior.get("source_window_end_date"),
        "requested_as_of_date": snapshot.get("requested_as_of_date"),
        "selected_as_of_date": (
            snapshot.get("current_as_of_date")
            if provenance_source
            in {"current_bullpen_pit", "current_prior_bullpen_blend"}
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


def _quality_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {field: payload.get(field) for field in QUALITY_FIELDS}


def _blend_quality(
    current: dict[str, Any],
    prior: dict[str, Any],
    current_weight: float,
    prior_weight: float,
) -> dict[str, Any]:
    rate_fields = {
        "k_pct",
        "bb_pct",
        "k_minus_bb_pct",
        "xwoba_against",
        "woba_against",
        "barrel_per_contact",
    }
    output: dict[str, Any] = {}
    for field in QUALITY_FIELDS:
        if field not in rate_fields:
            output[field] = current.get(field)
            continue
        current_value = _number_or_none(current.get(field))
        prior_value = _number_or_none(prior.get(field))
        if current_value is None and prior_value is None:
            output[field] = None
        elif current_value is None:
            output[field] = prior_value
        elif prior_value is None:
            output[field] = current_value
        else:
            output[field] = round(
                current_weight * current_value + prior_weight * prior_value,
                6,
            )
    return output


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
