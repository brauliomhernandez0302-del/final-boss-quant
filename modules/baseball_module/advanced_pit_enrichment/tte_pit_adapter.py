"""Adapter from experimental TTE PIT snapshots to offensive lambda.

This module is intentionally isolated from live/backtest ORCHESTRATION (no
import of the live pipeline entrypoint or the backtest driver script). It
DOES share the core
composite-score/regression/blend MATH with the live engine, via
`offense/tte_formula.py` (unified 2026-07-12 — the two paths had drifted
into independently-duplicated copies of the same formula; see that module's
docstring for the postmortem). It mirrors the current True Talent offense
formula only when all required PIT-safe inputs are explicitly present.
"""

from __future__ import annotations

from typing import Any

from config import LEAGUE_AVG_RUNS, LEAGUE_AVG_WOBA, LEAGUE_AVG_XWOBA
from ..offense.tte_formula import (
    regress as _shared_regress,
    season_lambda,
    blend_current_prior,
    PRIOR_PA_EQUIVALENT as _SHARED_PRIOR_PA_EQUIVALENT,
    LAMBDA_MIN as _SHARED_LAMBDA_MIN,
    LAMBDA_MAX as _SHARED_LAMBDA_MAX,
    PLATE_FACTOR_MIN,
    PLATE_FACTOR_MAX,
)


FORMULA_VERSION = "tte_pit_adapter_v1"

LG_RPG = LEAGUE_AVG_RUNS
LG_XWOBA = LEAGUE_AVG_XWOBA  # single source of truth: config.py
LG_WOBA = LEAGUE_AVG_WOBA
# MATH-002 fix (2026-07-18, audit_20260714/math002_diagnostico/paso5c1_alineacion_metrica.md):
# this adapter now matches the live offense engine's barrel term exactly
# instead of being its own internally-consistent-but-different per-PA
# metric. The 2026-07-11 re-centering (see git history) had switched the
# PRIOR to a per-PA value (0.054, walk-forward from prior_baseline) while the
# numerator stayed per-PA too (barrel_count/plate_appearances) —
# self-consistent, but not the same thing the live engine measures. That fix
# was a stopgap: at the time, `batted_ball_count`/`bip` (the PIT cache's own
# attempts count) was itself broken by foul-ball inflation (MATH-002, fixed
# in commit 5b of this roadmap step), so there was no trustworthy
# per-attempt `n` to shrink toward a per-attempt prior with. Now that
# batted_ball_count is fixed, this reverts to the live offense engine's
# exact definition (the sibling module next to tte_formula.py, line 70):
# LG_BARREL_PA = 0.088 ("per PA" in that constant's own comment is a
# codebase-wide legacy misnomer, confirmed by the team-aggregation helper
# that produces it, which computes barrels_sum/attempts_sum — i.e.
# per-batted-ball-event, not per-PA).
# Hardcoded here rather than imported: this module's own isolation test
# (tests/test_tte_pit_adapter.py) explicitly forbids importing the live
# offense engine module into this module (intentional orchestration
# isolation) — same accepted duplicated-constant risk tte_formula.py's
# docstring already documents (that refactor unified the MATH, not the
# league constants).
LG_BARREL_PA = 0.088
LG_BB_PCT = 0.086
LG_K_PCT = 0.224

K_XWOBA = 150
K_BARREL = 120
K_BB = 120
K_K = 60
MIN_OK_PA = K_XWOBA

# 2026-07-12: PRIOR_PA_EQUIVALENT/LAMBDA_MIN/LAMBDA_MAX moved to the shared
# tte_formula module (single source of truth vs. the live engine) — kept as
# local aliases so this file's other references and its provenance dict
# don't need to change.
PRIOR_PA_EQUIVALENT = _SHARED_PRIOR_PA_EQUIVALENT
LAMBDA_MIN = _SHARED_LAMBDA_MIN
LAMBDA_MAX = _SHARED_LAMBDA_MAX


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

    # MATH-002 (2026-07-18): barrel's current-season input is now
    # team_brl_percent/brl_percent (per-attempt, matching live) instead of
    # barrel_pa/team_barrel_pa (per-PA) — see paso5c1_alineacion_metrica.md.
    # The "barrel_pa"/"barrel_pa_prior" labels are kept as-is; they identify
    # the barrel INPUT slot in missing_inputs/fallback_used, not the field name.
    if _first_present(snapshot, "team_brl_percent", "brl_percent") is None:
        missing.append("barrel_pa")
    if _is_missing(_first_present(snapshot, "bip", "batted_ball_count")):
        missing.append("bip")
    if _first_present(snapshot, "bb_pct", "team_bb_pct") is None:
        missing.append("bb_pct")
    if _first_present(snapshot, "k_pct", "team_k_pct") is None:
        missing.append("k_pct")
    if not snapshot.get("prior_baseline_found"):
        missing.append("prior_baseline")
    if snapshot.get("brl_percent_prior") is None:
        missing.append("barrel_pa_prior")
    for key in ("team_est_woba_prior", "bb_pct_prior", "k_pct_prior"):
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
    # MATH-002 fix (2026-07-18, paso5c1_alineacion_metrica.md): barrel_cur is
    # now a per-attempt rate (team_brl_percent/100 == barrel_count/attempts),
    # matching the live offense engine's definition exactly, instead of the
    # per-PA barrel_pa field.
    barrel_cur = float(_first_present(snapshot, "team_brl_percent", "brl_percent")) / 100.0
    bip_cur = float(_first_present(snapshot, "bip", "batted_ball_count"))
    bb_cur = float(_first_present(snapshot, "bb_pct", "team_bb_pct"))
    k_cur = float(_first_present(snapshot, "k_pct", "team_k_pct"))
    xwoba_prior = float(snapshot["team_est_woba_prior"])
    barrel_prior = float(snapshot["brl_percent_prior"]) / 100.0
    bb_prior = float(snapshot["bb_pct_prior"])
    k_prior = float(snapshot["k_pct_prior"])

    # Barrel is regressed toward its league mean using attempts (batted-ball
    # events) as the Bayesian sample size, not pa — it's fundamentally a
    # per-BBE rate, same as the live offense engine. This
    # was previously left unfixed pending the PIT cache's own
    # batted_ball_count foul-inflation bug (MATH-002, fixed in commit 5b of
    # this roadmap step) so the switch wouldn't introduce a corrupted count
    # into a now-load-bearing computation.
    xwoba_reg = _shared_regress(xwoba_cur, lg_xwoba, pa, K_XWOBA)
    barrel_reg = _shared_regress(barrel_cur, lg_barrel_pa, bip_cur, K_BARREL)
    bb_reg = _shared_regress(bb_cur, lg_bb_pct, pa, K_BB)
    k_reg = _shared_regress(k_cur, lg_k_pct, pa, K_K)

    disc_cur = (bb_reg - k_reg) - (lg_bb_pct - lg_k_pct)  # kept for provenance only
    lambda_cur, cur_factors = season_lambda(
        xwoba_reg=xwoba_reg, barrel_reg=barrel_reg, bb_reg=bb_reg, k_reg=k_reg,
        lg_xwoba=lg_xwoba, lg_barrel_rate=lg_barrel_pa,
        lg_bb_pct=lg_bb_pct, lg_k_pct=lg_k_pct, lg_rpg=lg_rpg,
    )
    f_xwoba, f_barrel, f_plate = cur_factors["f_xwoba"], cur_factors["f_barrel"], cur_factors["f_plate"]

    lambda_prior, _pri_factors = season_lambda(
        xwoba_reg=xwoba_prior, barrel_reg=barrel_prior, bb_reg=bb_prior, k_reg=k_prior,
        lg_xwoba=lg_xwoba, lg_barrel_rate=lg_barrel_pa,
        lg_bb_pct=lg_bb_pct, lg_k_pct=lg_k_pct, lg_rpg=lg_rpg,
    )

    lambda_offense, current_w, prior_w = blend_current_prior(
        lambda_cur, lambda_prior, pa, PRIOR_PA_EQUIVALENT,
    )

    provenance = {
        "formula_inputs": {
            "team_est_woba": xwoba_cur,
            "barrel_pa": barrel_cur,
            "bip": int(bip_cur),
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
