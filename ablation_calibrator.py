#!/usr/bin/env python3
"""
ablation_calibrator.py
======================
Ablation test: WITH auto_calibrator vs WITHOUT (Kalman + team_bias only).

Scenario A — WITH:
  lh_base → Kalman(35% blend) → LambdaCalibrator(5 stat factors
            + Kalman-dampened team_bias + ±15% combined cap)
            → HFA → pitcher → regression → clip → MC → Platt

Scenario B — WITHOUT:
  lh_base → Kalman(35% blend) → dampened_team_bias only
            → HFA → pitcher → regression → clip → MC → Platt

Both scenarios are identical from HFA onward, isolating the contribution
of the stat-factor calibration stage. Uses only data already cached in
.cache/backtest/ and .cache/ — zero new network calls.

Usage:
  python3 ablation_calibrator.py            # all 5422 games
  python3 ablation_calibrator.py --limit 200  # quick smoke-test
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Reuse backtest infrastructure (cached API, game-data builder, Platt fn)
from backtest_and_retrain import (
    DiskCache, CACHE_DIR, DB_PATH, REPORT_DIR,
    _platt, _devig,
    build_game_data, fetch_starters,
    prefetch_team_stats,
)
from config import DATA_DIR, LEAGUE_AVG_ERA, LEAGUE_AVG_RUNS, LEAGUE_AVG_WHIP
from data_fetchers import MLBDataIntegrator, MLBStatsAPI, ParkFactors
from modules.baseball_module.calibration.auto_calibrator import LambdaCalibrator
from modules.baseball_module.calibration.learning_engine import LearningEngine
from modules.baseball_module.context_engine.pitcher_engine import adjust_for_pitchers
from modules.baseball_module.context_engine.pitchers_regression import (
    calculate_pitcher_regression,
)
from modules.baseball_module.core.run_module import _compute_f5_lambda
from modules.baseball_module.hfa.hfa_engine import get_adjusted_lambdas
from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced

try:
    from modules.baseball_module.data_enrichment.savant_fetcher import SavantFetcher
    from modules.baseball_module.data_enrichment.fangraphs_fetcher import FanGraphsFetcher
    _ENRICHMENT_AVAILABLE = True
except ImportError:
    _ENRICHMENT_AVAILABLE = False

import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("ablation")
log.setLevel(logging.INFO)

# 10 000 MC sims gives SE ≈ 0.005 @ p=0.5 — sufficient to rank two
# scenarios against each other; halves runtime vs the backtest's 50 000.
_N_MC = 10_000


# ── Shared tail: HFA → pitcher → regression → clip → MC → Platt ──────────────

def _run_tail(
    lh: float,
    la: float,
    game_data: Dict,
) -> Dict[str, float]:
    """Pipeline from HFA onward — identical for both scenarios."""
    lh, la, _ = get_adjusted_lambdas(lh, la, game_data)
    lh, la, _ = adjust_for_pitchers(lh, la, game_data)

    fa, _ = calculate_pitcher_regression(
        pitcher_stats=game_data.get("pitcher_away", {}),
        opponent_stats=game_data.get("home_team", {}),
    )
    fh, _ = calculate_pitcher_regression(
        pitcher_stats=game_data.get("pitcher_home", {}),
        opponent_stats=game_data.get("away_team", {}),
    )
    lh *= fa
    la *= fh

    lh = max(3.0, min(lh, 7.0))
    la = max(3.0, min(la, 7.0))

    mc = monte_carlo_advanced(
        lh=lh, la=la, n_max=_N_MC,
        block=min(10_000, _N_MC),
        analyze_f5=False,
        lh_f5=None, la_f5=None,
    )
    p_h_raw = _platt(mc["p_home"])
    p_a_raw = _platt(mc["p_away"])
    total = p_h_raw + p_a_raw
    return {
        "lh": round(lh, 4),
        "la": round(la, 4),
        "p_home": round(p_h_raw / total, 5),
        "p_away": round(p_a_raw / total, 5),
    }


# ── Scenario A: WITH auto_calibrator ─────────────────────────────────────────

def run_with(
    lh_base: float,
    la_base: float,
    game_data: Dict,
    learning: LearningEngine,
    season: int,
) -> Dict[str, float]:
    home = game_data["home_team"]["name"]
    away = game_data["away_team"]["name"]

    # Kalman blend (same as run_module.py line 381-382)
    lh = learning.get_kalman_lambda_adjustment(home, "offense_home", season, lh_base)
    la = learning.get_kalman_lambda_adjustment(away, "offense_away", season, la_base)

    # Full stat-factor calibrator (includes Kalman-dampened team bias + ±15% cap)
    cal = LambdaCalibrator(learning_engine=learning)
    lh, la = cal.calibrate(lh, la, game_data)

    return _run_tail(lh, la, game_data)


# ── Scenario B: WITHOUT auto_calibrator ──────────────────────────────────────

def run_without(
    lh_base: float,
    la_base: float,
    game_data: Dict,
    learning: LearningEngine,
    season: int,
) -> Dict[str, float]:
    home = game_data["home_team"]["name"]
    away = game_data["away_team"]["name"]

    # Kalman blend
    lh = learning.get_kalman_lambda_adjustment(home, "offense_home", season, lh_base)
    la = learning.get_kalman_lambda_adjustment(away, "offense_away", season, la_base)

    # Dampened team bias only — no stat factors, no ±15% cap
    bias_h = learning.compute_team_bias_kalman_adjusted(home, season, "offense_home")
    bias_a = learning.compute_team_bias_kalman_adjusted(away, season, "offense_away")
    lh *= bias_h
    la *= bias_a

    return _run_tail(lh, la, game_data)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _brier(results: List[Dict], key: str) -> float:
    return sum((r[key] - r["home_won"]) ** 2 for r in results) / len(results)


def _accuracy(results: List[Dict], key: str) -> float:
    return sum((r[key] > 0.5) == bool(r["home_won"]) for r in results) / len(results)


def _mean_spread(results: List[Dict], lh_key: str, la_key: str) -> float:
    spreads = [r[lh_key] - r[la_key] for r in results]
    return sum(spreads) / len(spreads)


def _pct_positive_spread(results: List[Dict], lh_key: str, la_key: str) -> float:
    return sum(r[lh_key] > r[la_key] for r in results) / len(results)


def _roi_at_threshold(
    results: List[Dict],
    ph_key: str,
    pa_key: str,
    threshold: float = 0.10,
) -> Dict:
    _EXTREME = 4.0
    bettable = [
        r for r in results
        if r.get("pin_fh") and r.get("pin_fa")
        and (r.get("ml_home_pin") or 0) <= _EXTREME
        and (r.get("ml_away_pin") or 0) <= _EXTREME
    ]
    bets = staked = profit = 0
    for r in bettable:
        edge_h = r[ph_key] - r["pin_fh"]
        edge_a = r[pa_key] - r["pin_fa"]
        if edge_h >= edge_a and edge_h >= threshold:
            bets += 1; staked += 1.0
            profit += (r["ml_home_pin"] - 1) if r["home_won"] else -1.0
        elif edge_a > edge_h and edge_a >= threshold:
            bets += 1; staked += 1.0
            profit += (r["ml_away_pin"] - 1) if not r["home_won"] else -1.0
    roi = profit / staked * 100 if staked > 0 else 0.0
    return {"bets": bets, "profit": round(profit, 2), "roi_pct": round(roi, 2)}


def _calibration_buckets(results: List[Dict], ph_key: str) -> Dict:
    edges = [
        ("<40%",    0.00, 0.40),
        ("40-45%",  0.40, 0.45),
        ("45-50%",  0.45, 0.50),
        ("50-55%",  0.50, 0.55),
        ("55-60%",  0.55, 0.60),
        ("60-70%",  0.60, 0.70),
        (">70%",    0.70, 1.01),
    ]
    out = {}
    for label, lo, hi in edges:
        bucket = [r for r in results if lo <= r[ph_key] < hi]
        n = len(bucket)
        wins = sum(r["home_won"] for r in bucket)
        mean_p = sum(r[ph_key] for r in bucket) / n if n else 0
        out[label] = {
            "n": n,
            "pred_pct": round(mean_p * 100, 1),
            "actual_pct": round(wins / n * 100, 1) if n else None,
        }
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--season", type=int, default=0)
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    where = "WHERE actual_home_runs IS NOT NULL"
    if args.season:
        where += f" AND season = {args.season}"
    order = "ORDER BY game_date ASC"
    limit = f"LIMIT {args.limit}" if args.limit else ""
    rows = conn.execute(
        f"SELECT * FROM game_outcomes {where} {order} {limit}"
    ).fetchall()
    conn.close()

    seasons = sorted({r["season"] for r in rows})
    n_total = len(rows)
    log.info("Games: %d  |  seasons: %s", n_total, seasons)

    # ── shared objects ────────────────────────────────────────────────────────
    import requests as _req
    session = _req.Session()
    api       = MLBStatsAPI()
    integrator = MLBDataIntegrator()
    park_factors = ParkFactors()
    cache     = DiskCache(CACHE_DIR)
    learning  = LearningEngine(db_path=DB_PATH)

    # Patch out F5 by-inning stats: the ablation doesn't use F5 lambdas
    # (both _run_tail() calls pass lh_f5=None) and the endpoint returns
    # 400 for ~60% of historical pitchers, adding ~0.3s per game in failed
    # network round-trips that account for >80% of total runtime.
    api.get_pitcher_f5_stats = lambda *a, **kw: None

    # Enrich caches per season (Savant + FanGraphs — optional)
    savant_by_season: Dict[int, Dict] = {}
    fg_by_season:     Dict[int, Dict] = {}
    if _ENRICHMENT_AVAILABLE:
        sv = SavantFetcher(cache_dir=ROOT / ".cache")
        fg = FanGraphsFetcher(cache_dir=ROOT / ".cache")
        for yr in seasons:
            savant_by_season[yr] = sv.get_all_pitcher_stats(yr)
            fg_by_season[yr]     = fg.get_all_pitcher_stats(yr)
            log.info("  enrichment loaded: season=%d  Savant=%d  FG=%d",
                     yr, len(savant_by_season[yr]), len(fg_by_season[yr]))

    # ── main loop ─────────────────────────────────────────────────────────────
    results: List[Dict] = []
    n_ok = n_err = 0
    t0 = time.time()

    for idx, row in enumerate(rows, 1):
        game_pk   = row["game_pk"]
        game_date = row["game_date"]
        season    = row["season"]
        home_name = row["home_team"]
        away_name = row["away_team"]
        home_won  = int(row["home_won"])

        starters = fetch_starters(game_pk, session, cache)

        try:
            game_data, lh_base, la_base = build_game_data(
                game_pk, game_date, home_name, away_name, season,
                starters["home_pitcher_id"],
                starters["away_pitcher_id"],
                api, integrator, park_factors,
                savant_stats=savant_by_season.get(season),
                fg_stats=fg_by_season.get(season),
            )

            pred_with    = run_with(lh_base, la_base, game_data, learning, season)
            pred_without = run_without(lh_base, la_base, game_data, learning, season)

            pin_fh = pin_fa = None
            if row["ml_home_pin"] and row["ml_away_pin"]:
                pin_fh, pin_fa = _devig(row["ml_home_pin"], row["ml_away_pin"])

            results.append({
                "game_pk":   game_pk,
                "season":    season,
                "home_won":  home_won,
                "pin_fh":    pin_fh,
                "pin_fa":    pin_fa,
                "ml_home_pin": row["ml_home_pin"],
                "ml_away_pin": row["ml_away_pin"],
                # WITH calibrator
                "lh_w":  pred_with["lh"],
                "la_w":  pred_with["la"],
                "ph_w":  pred_with["p_home"],
                "pa_w":  pred_with["p_away"],
                # WITHOUT calibrator
                "lh_wo": pred_without["lh"],
                "la_wo": pred_without["la"],
                "ph_wo": pred_without["p_home"],
                "pa_wo": pred_without["p_away"],
            })
            n_ok += 1

        except Exception as exc:
            log.debug("game_pk=%d FAILED: %s", game_pk, exc)
            n_err += 1

        if idx % 500 == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            eta = (n_total - idx) / rate
            log.info("  [%d/%d] ok=%d err=%d  %.1f g/s  ETA %.0fs",
                     idx, n_total, n_ok, n_err, rate, eta)
            # Checkpoint: flush partial results so a killed run is recoverable
            _ckpt = REPORT_DIR / "ablation_calibrator_checkpoint.json"
            _ckpt.write_text(json.dumps({
                "games_processed": len(results),
                "n_errors": n_err,
                "partial": True,
                "results_sample": results[-5:],  # last 5 for debugging
            }))

    elapsed_total = time.time() - t0
    log.info("Done: %d ok / %d err  (%.0fs, %.1f g/s)",
             n_ok, n_err, elapsed_total, n_ok / elapsed_total)

    if not results:
        log.error("No results to report.")
        return

    # ── compute metrics ───────────────────────────────────────────────────────
    n = len(results)
    brier_w  = _brier(results, "ph_w")
    brier_wo = _brier(results, "ph_wo")
    acc_w    = _accuracy(results, "ph_w")
    acc_wo   = _accuracy(results, "ph_wo")

    spread_w  = _mean_spread(results, "lh_w",  "la_w")
    spread_wo = _mean_spread(results, "lh_wo", "la_wo")
    pos_w     = _pct_positive_spread(results, "lh_w",  "la_w")
    pos_wo    = _pct_positive_spread(results, "lh_wo", "la_wo")

    roi_w  = _roi_at_threshold(results, "ph_w",  "pa_w",  0.10)
    roi_wo = _roi_at_threshold(results, "ph_wo", "pa_wo", 0.10)
    roi5_w  = _roi_at_threshold(results, "ph_w",  "pa_w",  0.05)
    roi5_wo = _roi_at_threshold(results, "ph_wo", "pa_wo", 0.05)

    # per-season breakdown
    by_season_w:  Dict[int, Dict] = defaultdict(lambda: {"n": 0, "correct": 0, "brier": 0.0, "spread": 0.0})
    by_season_wo: Dict[int, Dict] = defaultdict(lambda: {"n": 0, "correct": 0, "brier": 0.0, "spread": 0.0})
    for r in results:
        s = r["season"]
        by_season_w[s]["n"]       += 1
        by_season_w[s]["correct"] += int((r["ph_w"] > 0.5) == bool(r["home_won"]))
        by_season_w[s]["brier"]   += (r["ph_w"] - r["home_won"]) ** 2
        by_season_w[s]["spread"]  += r["lh_w"] - r["la_w"]
        by_season_wo[s]["n"]       += 1
        by_season_wo[s]["correct"] += int((r["ph_wo"] > 0.5) == bool(r["home_won"]))
        by_season_wo[s]["brier"]   += (r["ph_wo"] - r["home_won"]) ** 2
        by_season_wo[s]["spread"]  += r["lh_wo"] - r["la_wo"]

    cal_w  = _calibration_buckets(results, "ph_w")
    cal_wo = _calibration_buckets(results, "ph_wo")

    # ── build report dict ─────────────────────────────────────────────────────
    report = {
        "run_at":      datetime.now(timezone.utc).isoformat(),
        "n_games":     n,
        "n_errors":    n_err,
        "seasons":     seasons,
        "n_mc_sims":   _N_MC,
        "WITH_calibrator": {
            "brier":          round(brier_w, 5),
            "accuracy_pct":   round(acc_w * 100, 2),
            "mean_lambda_spread": round(spread_w, 4),
            "pct_home_favored": round(pos_w * 100, 1),
            "roi_edge10": roi_w,
            "roi_edge5":  roi5_w,
        },
        "WITHOUT_calibrator": {
            "brier":          round(brier_wo, 5),
            "accuracy_pct":   round(acc_wo * 100, 2),
            "mean_lambda_spread": round(spread_wo, 4),
            "pct_home_favored": round(pos_wo * 100, 1),
            "roi_edge10": roi_wo,
            "roi_edge5":  roi5_wo,
        },
        "delta": {
            "brier":           round(brier_wo - brier_w, 5),
            "accuracy_pp":     round((acc_w - acc_wo) * 100, 2),
            "lambda_spread":   round(spread_w - spread_wo, 4),
        },
        "by_season": {},
        "calibration": {"WITH": cal_w, "WITHOUT": cal_wo},
    }
    for s in sorted(set(list(by_season_w) + list(by_season_wo))):
        sw  = by_season_w[s]
        swo = by_season_wo[s]
        report["by_season"][str(s)] = {
            "n": sw["n"],
            "WITH":    {"acc": round(sw["correct"]/sw["n"]*100,2),
                        "brier": round(sw["brier"]/sw["n"],5),
                        "spread": round(sw["spread"]/sw["n"],4)},
            "WITHOUT": {"acc": round(swo["correct"]/swo["n"]*100,2),
                        "brier": round(swo["brier"]/swo["n"],5),
                        "spread": round(swo["spread"]/swo["n"],4)},
        }

    # ── save JSON ─────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = REPORT_DIR / f"ablation_calibrator_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2))

    # ── print summary ─────────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  ABLATION: WITH vs WITHOUT auto_calibrator  —  {n} games")
    print(f"  Seasons: {seasons}  |  MC sims/game: {_N_MC:,}  |  Errors: {n_err}")
    print(sep)

    print(f"\n  {'Metric':<30}  {'WITH cal':>12}  {'WITHOUT cal':>12}  {'Δ (W-WO)':>10}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*10}")
    print(f"  {'Brier score':<30}  {brier_w:>12.5f}  {brier_wo:>12.5f}  "
          f"  {brier_w-brier_wo:>+9.5f}")
    print(f"  {'Accuracy':<30}  {acc_w*100:>11.2f}%  {acc_wo*100:>11.2f}%  "
          f"  {(acc_w-acc_wo)*100:>+9.2f}pp")
    print(f"  {'Mean λ spread (H−A)':<30}  {spread_w:>+12.4f}  {spread_wo:>+12.4f}  "
          f"  {spread_w-spread_wo:>+9.4f}")
    print(f"  {'% games home favored':<30}  {pos_w*100:>11.1f}%  {pos_wo*100:>11.1f}%")

    print(f"\n  ROI SIMULATION (flat 1-unit @ Pinnacle, best-edge side)")
    print(f"  {'Threshold':<12}  {'Bets':>6}  {'Profit':>9}  {'ROI':>8}"
          f"  |  {'Bets':>6}  {'Profit':>9}  {'ROI':>8}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*9}  {'-'*8}"
          f"  |  {'-'*6}  {'-'*9}  {'-'*8}")
    for (thr, rw, rwo) in [("edge>=10%", roi_w, roi_wo), ("edge>=5%", roi5_w, roi5_wo)]:
        print(f"  {thr:<12}  {rw['bets']:>6}  {rw['profit']:>+9.2f}  {rw['roi_pct']:>+7.2f}%"
              f"  |  {rwo['bets']:>6}  {rwo['profit']:>+9.2f}  {rwo['roi_pct']:>+7.2f}%")

    print(f"\n  CALIBRATION — predicted vs actual win %  (WITH | WITHOUT)")
    print(f"  {'Bucket':<10}  {'N':>5}  {'Pred%':>7}  {'Act% W':>8}  {'Δ':>6}"
          f"  |  {'Act% WO':>8}  {'Δ':>6}")
    for label in cal_w:
        cw  = cal_w[label]
        cwo = cal_wo[label]
        n_b = cw["n"]
        if n_b == 0:
            continue
        dw  = (cw["actual_pct"]  or 0) - cw["pred_pct"]
        dwo = (cwo["actual_pct"] or 0) - cwo["pred_pct"]
        print(f"  {label:<10}  {n_b:>5}  {cw['pred_pct']:>7.1f}%  "
              f"{cw['actual_pct'] or 0:>7.1f}%  {dw:>+5.1f}"
              f"  |  {cwo['actual_pct'] or 0:>7.1f}%  {dwo:>+5.1f}")

    print(f"\n  BY SEASON")
    print(f"  {'Season':>6}  {'N':>5}  {'Acc W':>7}  {'Brier W':>8}  "
          f"{'Spread W':>9}  |  {'Acc WO':>7}  {'Brier WO':>8}  {'Spread WO':>10}")
    for s in sorted(report["by_season"]):
        sv = report["by_season"][s]
        w_ = sv["WITH"]; wo_ = sv["WITHOUT"]
        print(f"  {s:>6}  {sv['n']:>5}  {w_['acc']:>7.2f}%  {w_['brier']:>8.5f}  "
              f"{w_['spread']:>+9.4f}  |  {wo_['acc']:>7.2f}%  {wo_['brier']:>8.5f}  "
              f"{wo_['spread']:>+10.4f}")

    print(f"\n  Report saved → {out_path}")
    print(sep)


if __name__ == "__main__":
    main()
