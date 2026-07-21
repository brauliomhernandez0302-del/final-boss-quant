"""scripts/clv_report.py — docs/PROTOCOLO_CLV_V1.md's evaluation instrument.

Computes exactly what the protocol pre-registers: the primary metric (mean
EV-vs-close over the primary sample, bootstrap 95% CI, 10,000 resamples,
fixed seed), the three secondaries, the four pre-registered cuts, attrition,
and the staleness distribution of the closes actually used. D0 is read from
the protocol document's own "Registro" section — never hardcoded here — so
a run always reflects whatever boundary is actually logged there, not a
guess baked into this script.

This script does NOT emit a verdict. It reports the numbers and prints the
protocol's own pre-registered thresholds alongside them for reference; a
human reads the verdict against docs/PROTOCOLO_CLV_V1.md. Anything else
would let the instrument grade its own exam.

p_close is always computed via track_record.clv.compute_ev_vs_close(), which
itself only ever calls core.value_detector.remove_vig_multiplicative — no
devig logic is reimplemented here (see PROTOCOLO_CLV_V1.md's Definiciones,
"cero reimplementaciones — la lección de ODDS-001").

Usage:
    python3 scripts/clv_report.py [--db data/track_record.db] [--sport MLB]
    python3 scripts/clv_report.py --no-write   # print only, skip report file
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from track_record.clv import compute_ev_vs_close

PROTOCOL_PATH = REPO_ROOT / "docs" / "PROTOCOLO_CLV_V1.md"
DEFAULT_DB = REPO_ROOT / "data" / "track_record.db"
REPORT_DIR = REPO_ROOT / "reports" / "clv"

# Protocol's cut (a): buckets of |edge| at publish time.
_EDGE_BUCKETS: List[tuple] = [
    ("<5%", lambda e: e < 5.0),
    ("5-10%", lambda e: 5.0 <= e <= 10.0),
    (">10%", lambda e: e > 10.0),
]

# Bootstrap reproducibility: fixed seed, per the protocol's own anti-
# p-hacking design (Métrica primaria section).
_BOOTSTRAP_SEED = 20260720
_BOOTSTRAP_N_RESAMPLES = 10_000

_THRESHOLDS_FOR_REFERENCE_ONLY = {
    "hay_senal_de_edge": "IC95% inferior > 0 Y estimador puntual >= +1.0%, n>=300",
    "no_hay_edge": "IC95% superior < +1.0%",
    "kill_switch_no_edge_anticipado": "mirada quincenal, n>=150, IC95% superior < -1.0%",
    "no_concluyente": "el IC cruza ambos umbrales — extender en bloques de 2 semanas",
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_d0(protocol_path: Path = PROTOCOL_PATH) -> Optional[str]:
    """D0 lives in the protocol's own Registro section. Returns an ISO date
    string, or None if D0 hasn't been reached yet ("pendiente" or any other
    non-date placeholder) — never guessed or defaulted here."""
    text = protocol_path.read_text()
    m = re.search(r"^-\s*D0:\s*(\S+)", text, re.MULTILINE)
    if not m:
        return None
    val = m.group(1).strip()
    return val if re.match(r"^\d{4}-\d{2}-\d{2}$", val) else None


def load_picks(db_path: Path, sport: str = "MLB") -> List[Dict[str, Any]]:
    """Every moneyline pick ever published for `sport` — the report itself
    splits shakedown (pre-D0) from primary (D0 onward), so nothing is
    filtered by date here."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM picks WHERE sport = ? AND market IN ('ML_HOME','ML_AWAY') "
        "ORDER BY published_at",
        (sport,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def bootstrap_mean_ci(
    values: List[float],
    n_resamples: int = _BOOTSTRAP_N_RESAMPLES,
    seed: int = _BOOTSTRAP_SEED,
    ci: float = 0.95,
) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = arr[idx].mean(axis=1)
    alpha = 1.0 - ci
    lo = float(np.percentile(resampled_means, 100 * alpha / 2))
    hi = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return {
        "mean": float(arr.mean()), "ci_lo": lo, "ci_hi": hi,
        "n": n, "n_resamples": n_resamples, "seed": seed,
    }


def _edge_bucket(ev_pct: float) -> str:
    ev_abs = abs(ev_pct)
    for label, pred in _EDGE_BUCKETS:
        if pred(ev_abs):
            return label
    return "unknown"


def _staleness_bucket(minutes_before_start: Optional[float]) -> str:
    if minutes_before_start is None:
        return "unknown"
    return "<60min" if minutes_before_start < 60 else ">=60min"


def _brier(prob: float, outcome: int) -> float:
    return (prob - outcome) ** 2


def _cut_by(records: List[Dict[str, Any]], keyfn: Callable[[Dict[str, Any]], str]) -> Dict[str, Any]:
    groups: Dict[str, List[float]] = {}
    for r in records:
        groups.setdefault(keyfn(r), []).append(r["ev_vs_close_pct"])
    return {
        k: {"n": len(v), "mean_ev_vs_close_pct": round(sum(v) / len(v), 4)}
        for k, v in groups.items()
    }


def evaluate_sample(pool: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Descriptive report for one sample window (shakedown or primary). Does
    NOT decide anything — see module docstring."""
    ev_records: List[Dict[str, Any]] = []
    n_attrition = 0
    for p in pool:
        r = compute_ev_vs_close(p)
        if r is None:
            n_attrition += 1
            continue
        ev_records.append({**p, **r})

    # Primary metric only ever uses Pinnacle-devigged closes — the median-
    # across-books fallback is real, usable data but explicitly SECONDARY/
    # descriptive per the protocol, never fed into the decisive number.
    primary_eligible = [r for r in ev_records if r["source"] == "pinnacle"]
    fallback_only = [r for r in ev_records if r["source"] == "fallback_median"]
    ev_values = [r["ev_vs_close_pct"] for r in primary_eligible]

    n_total = len(pool)
    result: Dict[str, Any] = {
        "label": label,
        "n_picks": n_total,
        "n_attrition": n_attrition,
        "attrition_pct": round(100.0 * n_attrition / n_total, 2) if n_total else None,
        "n_fallback_median_only": len(fallback_only),
        "n_primary_eligible": len(primary_eligible),
        "primary_metric_mean_ev_vs_close_pct": bootstrap_mean_ci(ev_values),
        "secondary": {},
        "cuts": {},
        "staleness_distribution": None,
    }

    if ev_values:
        result["secondary"]["beat_close_rate_pct"] = round(
            100.0 * sum(1 for v in ev_values if v > 0) / len(ev_values), 2
        )

    # ROI/units — independent of close availability, explicitly non-decisive
    # (protocol: "a esta n el ROI es ruido... no rescata un CLV malo ni
    # viceversa"). Computed over every resolved pick in the pool, not just
    # the primary-eligible-for-CLV subset.
    resolved_pool = [p for p in pool if p.get("result") in ("WIN", "LOSS", "PUSH")]
    if resolved_pool:
        result["secondary"]["roi_units_total"] = round(
            sum(p.get("profit_loss_units") or 0.0 for p in resolved_pool), 4
        )
        result["secondary"]["roi_units_n_resolved"] = len(resolved_pool)

    # Model Brier vs p_close Brier — same resolved games, primary-eligible
    # closes only (so the comparison uses the same-quality close as the
    # primary metric).
    resolved_primary = [r for r in primary_eligible if r.get("result") in ("WIN", "LOSS")]
    if resolved_primary:
        model_briers, close_briers = [], []
        for r in resolved_primary:
            outcome = 1 if r["result"] == "WIN" else 0
            # decision_prob is the exact probability the pick's own EV/Kelly
            # was computed from (Platt-2D-corrected when fair_source ==
            # "pinnacle" — see track_record/db.py's schema comment). Picks
            # published before that column existed have it NULL; fall back
            # to model_prob for those rather than dropping them from the
            # comparison.
            model_prob = r.get("decision_prob")
            if model_prob is None:
                model_prob = r["model_prob"]
            model_briers.append(_brier(model_prob, outcome))
            close_briers.append(_brier(r["p_close"], outcome))
        result["secondary"]["model_brier"] = round(sum(model_briers) / len(model_briers), 5)
        result["secondary"]["p_close_brier"] = round(sum(close_briers) / len(close_briers), 5)
        result["secondary"]["n_brier_resolved"] = len(resolved_primary)

    # The 4 pre-registered cuts (max — none added post hoc), primary-eligible
    # pool only, purely descriptive.
    result["cuts"]["a_edge_bucket"] = _cut_by(primary_eligible, lambda r: _edge_bucket(r["ev_pct"]))
    result["cuts"]["b_staleness_bucket"] = _cut_by(
        primary_eligible, lambda r: _staleness_bucket(r.get("minutes_before_start"))
    )
    result["cuts"]["c_book"] = _cut_by(primary_eligible, lambda r: r.get("odds_book") or "unknown")
    markets_present = {r["market"] for r in primary_eligible}
    if len(markets_present) > 1:
        result["cuts"]["d_market"] = _cut_by(primary_eligible, lambda r: r["market"])

    stale_vals = [
        r["minutes_before_start"] for r in primary_eligible
        if r.get("minutes_before_start") is not None
    ]
    if stale_vals:
        arr = np.array(stale_vals, dtype=float)
        result["staleness_distribution"] = {
            "n": len(stale_vals),
            "mean_minutes": round(float(arr.mean()), 2),
            "median_minutes": round(float(np.median(arr)), 2),
            "min_minutes": round(float(arr.min()), 2),
            "max_minutes": round(float(arr.max()), 2),
        }

    return result


def build_report(picks: List[Dict[str, Any]], d0: Optional[str]) -> Dict[str, Any]:
    shakedown_pool = [p for p in picks if d0 is None or p["game_date"] < d0]
    primary_pool = [p for p in picks if d0 is not None and p["game_date"] >= d0]

    report = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "docs/PROTOCOLO_CLV_V1.md",
        "d0": d0,
        "shakedown": evaluate_sample(
            shakedown_pool, "shakedown (pre-D0, descriptive only — NOT the primary sample)"
        ),
        "primary": evaluate_sample(primary_pool, "primary (D0 onward)") if d0 is not None else None,
        "thresholds_for_reference_only_not_a_verdict": _THRESHOLDS_FOR_REFERENCE_ONLY,
    }
    return report


def _print_summary(report: Dict[str, Any]) -> None:
    _log(f"D0: {report['d0'] or 'pendiente — todo lo de abajo es shakedown, no hay muestra primaria aún'}")
    for key in ("shakedown", "primary"):
        block = report.get(key)
        if block is None:
            continue
        _log(f"--- {block['label']} ---")
        _log(f"  n_picks={block['n_picks']}  n_attrition={block['n_attrition']} "
             f"({block['attrition_pct']}%)  n_fallback_median_only={block['n_fallback_median_only']}")
        pm = block["primary_metric_mean_ev_vs_close_pct"]
        if pm:
            _log(f"  mean EV-vs-close = {pm['mean']:.4f}%  "
                 f"95% CI [{pm['ci_lo']:.4f}%, {pm['ci_hi']:.4f}%]  (n={pm['n']}, bootstrap={pm['n_resamples']})")
        else:
            _log("  mean EV-vs-close: no primary-eligible picks (all attrition/fallback)")
        if block["secondary"]:
            _log(f"  secondary: {block['secondary']}")
        if block["staleness_distribution"]:
            _log(f"  staleness (minutes before start) of closes used: {block['staleness_distribution']}")
        if block["attrition_pct"] is not None and block["attrition_pct"] > 15.0:
            _log("  WARNING: attrition > 15% — protocol treats this as an instrument "
                 "problem, not a modeling result (docs/PROTOCOLO_CLV_V1.md).")
    _log("Thresholds shown for reference only — this script does not render a verdict. "
         "Compare manually against docs/PROTOCOLO_CLV_V1.md.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--sport", default="MLB")
    parser.add_argument("--no-write", action="store_true", help="print only, skip writing a report file")
    args = parser.parse_args()

    d0 = read_d0()
    picks = load_picks(Path(args.db), sport=args.sport)
    report = build_report(picks, d0)
    _print_summary(report)

    if not args.no_write:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORT_DIR / f"clv_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(report, indent=2, default=str))
        _log(f"Report written: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
