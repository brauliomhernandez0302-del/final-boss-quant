"""Adapter from experimental TTE PIT snapshots to offensive lambda.

This module is intentionally isolated from live and backtest paths. It mirrors
the current True Talent offense formula only when all required PIT-safe inputs
are explicitly present.
"""

from __future__ import annotations

from typing import Any

from config import LEAGUE_AVG_RUNS, LEAGUE_AVG_WOBA, LEAGUE_AVG_XWOBA


FORMULA_VERSION = "tte_pit_adapter_v1"

LG_RPG = LEAGUE_AVG_RUNS
LG_XWOBA = LEAGUE_AVG_XWOBA  # single source of truth: config.py
LG_WOBA = LEAGUE_AVG_WOBA
LG_BARREL_PA = 0.088
LG_BB_PCT = 0.086
LG_K_PCT = 0.224

K_XWOBA = 150
K_BARREL = 120
K_BB = 120
K_K = 60
PRIOR_PA_EQUIVALENT = 1000
MIN_OK_PA = K_XWOBA

LAMBDA_MIN = 3.0
LAMBDA_MAX = 7.0
PLATE_FACTOR_MIN = 0.85
PLATE_FACTOR_MAX = 1.15


def adapt_tte_pit_snapshot_to_lambda(
    snapshot: dict[str, Any],
    *,
    league_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a TTE PIT snapshot to a TTE-compatible lambda contract.

    The adapter computes a lambda only when legacy-formula inputs are complete.
    Prior-season baseline inputs are expected on the snapshot and are blended
    using the same inverse-proportional PA rule as the legacy TTE.
    """
    baseline = league_baseline or {}
    output = _base_output(snapshot)
    missing_inputs = _missing_formula_inputs(snapshot, baseline)

    sample_size_status = _sample_size_status(output["pa"])
    output["sample_size_status"] = sample_size_status

    if not snapshot.get("found"):
        output["fallback_used"] = "snapshot_not_found"
        output["provenance"]["missing_inputs"] = missing_inputs
        return output

    if sample_size_status == "missing":
        output["fallback_used"] = "missing_sample_size"
        output["provenance"]["missing_inputs"] = missing_inputs
        return output

    if missing_inputs:
        output["fallback_used"] = "missing_inputs:" + ",".join(missing_inputs)
        output["provenance"]["missing_inputs"] = missing_inputs
        return output

    lambda_offense, formula_provenance = _compute_lambda(snapshot, baseline)
    output["lambda_offense"] = lambda_offense
    output["fallback_used"] = None
    output["blend_current_weight"] = formula_provenance["current_weight"]
    output["blend_prior_weight"] = formula_provenance["prior_weight"]
    output["provenance"].update(formula_provenance)
    output["provenance"]["missing_inputs"] = []
    return output


def _base_output(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "found": bool(snapshot.get("found")),
        "lambda_offense": None,
        "runs_per_game": snapshot.get("runs_per_game"),
        "team_est_woba": snapshot.get("team_est_woba"),
        "team_woba": snapshot.get("team_woba"),
        "team_brl_percent": snapshot.get("team_brl_percent"),
        "team_ev95percent": snapshot.get("team_ev95percent"),
        "pa": snapshot.get("pa"),
        "bip": snapshot.get("bip"),
        "sample_size_status": "missing",
        "fallback_used": None,
        "blend_current_weight": None,
        "blend_prior_weight": None,
        "prior_baseline_found": bool(snapshot.get("prior_baseline_found")),
        "formula_version": FORMULA_VERSION,
        "provenance": {
            "snapshot_version": snapshot.get("snapshot_version"),
            "requested_as_of_date": snapshot.get("requested_as_of_date"),
            "team_offense_as_of_date": snapshot.get("team_offense_as_of_date"),
            "prior_baseline_as_of_date": snapshot.get("prior_baseline_as_of_date"),
            "source_fingerprints": snapshot.get("source_fingerprints", {}),
        },
    }


def _missing_formula_inputs(snapshot: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in ("team_est_woba", "pa"):
        if _is_missing(snapshot.get(key)):
            missing.append(key)

    if _first_present(snapshot, "team_barrel_pa", "barrel_pa") is None:
        missing.append("barrel_pa")
    if _first_present(snapshot, "bb_pct", "team_bb_pct") is None:
        missing.append("bb_pct")
    if _first_present(snapshot, "k_pct", "team_k_pct") is None:
        missing.append("k_pct")
    if not snapshot.get("prior_baseline_found"):
        missing.append("prior_baseline")
    for key in ("team_est_woba_prior", "barrel_pa_prior", "bb_pct_prior", "k_pct_prior"):
        if _is_missing(snapshot.get(key)):
            missing.append(key)
    return missing


def _compute_lambda(snapshot: dict[str, Any], baseline: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    pa = float(snapshot["pa"])
    lg_rpg = float(baseline.get("league_avg_runs", LG_RPG))
    lg_xwoba = float(baseline.get("league_avg_xwoba", LG_XWOBA))
    lg_barrel_pa = float(baseline.get("league_barrel_pa", LG_BARREL_PA))
    lg_bb_pct = float(baseline.get("league_bb_pct", LG_BB_PCT))
    lg_k_pct = float(baseline.get("league_k_pct", LG_K_PCT))

    xwoba_cur = float(snapshot["team_est_woba"])
    barrel_cur = float(_first_present(snapshot, "team_barrel_pa", "barrel_pa"))
    bb_cur = float(_first_present(snapshot, "bb_pct", "team_bb_pct"))
    k_cur = float(_first_present(snapshot, "k_pct", "team_k_pct"))
    xwoba_prior = float(snapshot["team_est_woba_prior"])
    barrel_prior = float(snapshot["barrel_pa_prior"])
    bb_prior = float(snapshot["bb_pct_prior"])
    k_prior = float(snapshot["k_pct_prior"])

    xwoba_reg = _regress(xwoba_cur, lg_xwoba, pa, K_XWOBA)
    barrel_reg = _regress(barrel_cur, lg_barrel_pa, pa, K_BARREL)
    bb_reg = _regress(bb_cur, lg_bb_pct, pa, K_BB)
    k_reg = _regress(k_cur, lg_k_pct, pa, K_K)

    disc_cur = (bb_reg - k_reg) - (lg_bb_pct - lg_k_pct)
    plate_factor = _clamp(1.0 + disc_cur * 3.5, PLATE_FACTOR_MIN, PLATE_FACTOR_MAX)

    f_xwoba = xwoba_reg / lg_xwoba
    f_barrel = barrel_reg / lg_barrel_pa
    f_plate = plate_factor
    composite = f_xwoba * 0.50 + f_barrel * 0.30 + f_plate * 0.20
    lambda_cur = composite * lg_rpg

    disc_prior = (bb_prior - k_prior) - (lg_bb_pct - lg_k_pct)
    plate_prior = _clamp(1.0 + disc_prior * 3.5, PLATE_FACTOR_MIN, PLATE_FACTOR_MAX)
    composite_prior = (
        (xwoba_prior / lg_xwoba) * 0.50
        + (barrel_prior / lg_barrel_pa) * 0.30
        + plate_prior * 0.20
    )
    lambda_prior = composite_prior * lg_rpg

    prior_w = PRIOR_PA_EQUIVALENT / (PRIOR_PA_EQUIVALENT + pa)
    current_w = pa / (PRIOR_PA_EQUIVALENT + pa)
    lambda_offense = round(current_w * lambda_cur + prior_w * lambda_prior, 4)
    lambda_offense = _clamp(lambda_offense, LAMBDA_MIN, LAMBDA_MAX)

    provenance = {
        "formula_inputs": {
            "team_est_woba": xwoba_cur,
            "barrel_pa": barrel_cur,
            "bb_pct": bb_cur,
            "k_pct": k_cur,
            "pa": int(pa),
            "prior_lambda_offense": lambda_prior,
            "team_est_woba_prior": xwoba_prior,
            "barrel_pa_prior": barrel_prior,
            "bb_pct_prior": bb_prior,
            "k_pct_prior": k_prior,
        },
        "legacy_constants": {
            "k_xwoba": K_XWOBA,
            "k_barrel": K_BARREL,
            "k_bb": K_BB,
            "k_k": K_K,
            "prior_pa_equivalent": PRIOR_PA_EQUIVALENT,
            "lambda_min": LAMBDA_MIN,
            "lambda_max": LAMBDA_MAX,
            "plate_factor_min": PLATE_FACTOR_MIN,
            "plate_factor_max": PLATE_FACTOR_MAX,
        },
        "factors": {
            "f_xwoba": round(f_xwoba, 4),
            "f_barrel": round(f_barrel, 4),
            "f_plate": round(f_plate, 4),
        },
        "lambda_cur": round(lambda_cur, 4),
        "lambda_prior": round(lambda_prior, 4),
        "prior_weight": round(prior_w, 3),
        "current_weight": round(current_w, 3),
    }
    return lambda_offense, provenance


def _sample_size_status(pa: Any) -> str:
    if _is_missing(pa):
        return "missing"
    try:
        pa_value = float(pa)
    except (TypeError, ValueError):
        return "missing"
    if pa_value <= 0:
        return "missing"
    if pa_value < MIN_OK_PA:
        return "thin"
    return "ok"


def _first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _is_missing(value: Any) -> bool:
    return value is None


def _regress(observed: float, mean: float, n: float, k: float) -> float:
    if n <= 0:
        return mean
    return (observed * n + mean * k) / (n + k)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
