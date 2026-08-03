"""Adapter for the isolated contact-adjusted Team Defense PIT contract."""

from __future__ import annotations

from typing import Any


ADAPTER_VERSION = "defense_pit_adapter_v1"
CURRENT_BIP_SHRINKAGE = 500
MULTIPLIER_MIN = 0.92
MULTIPLIER_MAX = 1.08


def adapt_defense_pit_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Apply current → prior → neutral hierarchy without external fallbacks.

    A positive proxy means more batter-outs than expected, so it produces a
    multiplier below 1.0 for the opponent. The 1:1 mapping from proxy points to
    multiplier points is intentionally simple and remains disconnected from
    model weights until separately validated.
    """
    current_proxy = _number_or_none(snapshot.get("contact_adjusted_defense_proxy"))
    current_n = _positive_int_or_zero(snapshot.get("xba_bip_count"))
    prior_proxy = _number_or_none(
        snapshot.get("prior_contact_adjusted_defense_proxy")
    )

    current_weight = 0.0
    prior_weight = 0.0
    if snapshot.get("current_defense_found") and current_proxy is not None and current_n > 0:
        current_weight = current_n / (current_n + CURRENT_BIP_SHRINKAGE)
        prior_weight = 1.0 - current_weight
        baseline_proxy = prior_proxy if prior_proxy is not None else 0.0
        proxy = current_weight * current_proxy + prior_weight * baseline_proxy
        provenance_source = "current_defense_pit"
        fallback_used = None
    elif snapshot.get("prior_baseline_found") and prior_proxy is not None:
        proxy = prior_proxy
        prior_weight = 1.0
        provenance_source = "prior_season_defense_baseline"
        fallback_used = "current_defense_pit_unavailable"
    else:
        proxy = 0.0
        provenance_source = "neutral_defense_adjustment"
        fallback_used = "current_and_prior_defense_unavailable"

    multiplier = _clamp(1.0 - proxy, MULTIPLIER_MIN, MULTIPLIER_MAX)
    if provenance_source == "prior_season_defense_baseline":
        bip_count = snapshot.get("prior_bip_count")
        xba_bip_count = snapshot.get("prior_xba_bip_count")
        sample_size_status = "prior_baseline"
    elif provenance_source == "current_defense_pit":
        bip_count = snapshot.get("bip_count")
        xba_bip_count = snapshot.get("xba_bip_count")
        sample_size_status = snapshot.get("sample_size_status", "missing")
    else:
        bip_count = None
        xba_bip_count = None
        sample_size_status = "neutral"
    return {
        "found": provenance_source != "neutral_defense_adjustment",
        "team_id": snapshot.get("team_id"),
        "season": snapshot.get("season"),
        "contact_adjusted_defense_proxy": round(proxy, 6),
        "defense_multiplier": round(multiplier, 6),
        "provenance_source": provenance_source,
        "fallback_used": fallback_used,
        "current_defense_found": bool(snapshot.get("current_defense_found")),
        "prior_baseline_found": bool(snapshot.get("prior_baseline_found")),
        "bip_count": bip_count,
        "xba_bip_count": xba_bip_count,
        "sample_size_status": sample_size_status,
        "current_weight": round(current_weight, 6),
        "prior_weight": round(prior_weight, 6),
        "source_window_start_date": snapshot.get("source_window_start_date"),
        "source_window_end_date": snapshot.get("source_window_end_date"),
        "prior_source_window_start_date": snapshot.get(
            "prior_source_window_start_date"
        ),
        "prior_source_window_end_date": snapshot.get("prior_source_window_end_date"),
        "provenance": {
            "adapter_version": ADAPTER_VERSION,
            "snapshot_version": snapshot.get("snapshot_version"),
            "requested_as_of_date": snapshot.get("requested_as_of_date"),
            "current_as_of_date": snapshot.get("current_as_of_date"),
            "prior_baseline_as_of_date": snapshot.get("prior_baseline_as_of_date"),
            "source_fingerprints": snapshot.get("source_fingerprints", {}),
            "full_season_der_fallback": False,
            "neutral_proxy": 0.0,
            "neutral_multiplier": 1.0,
        },
    }


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_int_or_zero(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
