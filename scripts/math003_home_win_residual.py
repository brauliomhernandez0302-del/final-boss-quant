"""MATH-003 — repeatable home-win-probability residual analysis.

Re-derives, on demand, the same diagnostic that originally justified
hfa_engine.py's `_UNIFORM_HOME_MULT` (added 2026-07-11): does the model's raw
(pre-Platt) home win probability match the real home win rate, per season?

Uses `backtest_p_home_raw` (the model's own probability, before Platt
market-blending) against `home_won`, not the Platt-calibrated
`backtest_p_home` — matching the original fix's own reasoning
(hfa_engine.py's module docstring): Platt calibration differs structurally
between seasons (2024 has no prior-season data in this DB, so it runs at
identity; 2025 doesn't), so a downstream-only read would conflate the two.
The bias this constant corrects lives in lambda-space, upstream of Platt.

"Without the uniform term" uses Skellam only as a *sensitivity* (marginal
delta), never as an absolute probability — matching the original fix's own
language ("sized via Skellam sensitivity"). Sanity-checked empirically before
writing this: comparing Skellam's absolute P(home_win) against the real
model's stored backtest_p_home_raw at the SAME (lambda_home, lambda_away)
pair shows a large systematic gap (~-5.8pp mean, 2024 sample) — the real
engine samples Negative-Binomial, not independent Poisson, so Skellam is NOT
a usable absolute stand-in here. A *difference* largely cancels that
systematic bias: the correction is a simple multiplicative factor on
lambda_home only (lh_new = lh * (1 + _UNIFORM_HOME_MULT)), applied at the
HFA stage with nothing downstream re-multiplying it before Monte Carlo, so
dividing the stored final backtest_lambda_home by (1 + _UNIFORM_HOME_MULT)
recovers the exact pre-correction lambda. This script computes
delta = Skellam_p_home(lh, la) - Skellam_p_home(lh/mult, la) per game (the
marginal contribution the uniform term is estimated to have added) and
subtracts that delta from the real, stored backtest_p_home_raw to
reconstruct "without" — not a full pipeline re-run, but a same-methodology,
sensitivity-only use of Skellam consistent with how the constant was
originally sized.

Usage: python3 scripts/math003_home_win_residual.py [--db data/predictions_history.db]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.stats import skellam

from modules.baseball_module.hfa.hfa_engine import _UNIFORM_HOME_MULT


def skellam_p_home(lh: float, la: float) -> float:
    """P(home_runs - away_runs > 0) for independent Poisson(lh), Poisson(la)."""
    return float(1.0 - skellam.cdf(0, lh, la))


def analyze(db_path: Path) -> dict:
    con = sqlite3.connect(db_path)
    rows = con.execute(
        """
        SELECT season, backtest_lambda_home, backtest_lambda_away,
               backtest_p_home_raw, home_won
        FROM game_outcomes
        WHERE season IN (2024, 2025)
          AND source = 'backtest'
          AND backtest_lambda_home IS NOT NULL
          AND backtest_lambda_away IS NOT NULL
          AND backtest_p_home_raw IS NOT NULL
          AND home_won IS NOT NULL
        """
    ).fetchall()

    by_season: dict[int, list[tuple]] = {2024: [], 2025: []}
    for season, lh, la, p_raw, home_won in rows:
        by_season[int(season)].append((lh, la, p_raw, int(home_won)))

    hfa_mult = 1.0 + _UNIFORM_HOME_MULT
    results = {}
    for season, game_rows in by_season.items():
        n = len(game_rows)
        if n == 0:
            continue
        actual_home_rate = sum(hw for *_, hw in game_rows) / n
        mean_p_raw_with = sum(p for _, _, p, _ in game_rows) / n

        p_without = []
        for lh, la, p_raw, _hw in game_rows:
            lh_no_uniform = lh / hfa_mult
            delta = skellam_p_home(lh, la) - skellam_p_home(lh_no_uniform, la)
            p_without.append(p_raw - delta)
        mean_p_raw_without = sum(p_without) / n

        results[season] = {
            "n_games": n,
            "actual_home_win_rate": round(actual_home_rate, 4),
            "model_p_home_raw_with_uniform_term": round(mean_p_raw_with, 4),
            "residual_with_uniform_term": round(actual_home_rate - mean_p_raw_with, 4),
            "model_p_home_raw_without_uniform_term_skellam_reestimate": round(mean_p_raw_without, 4),
            "residual_without_uniform_term": round(actual_home_rate - mean_p_raw_without, 4),
        }

    return {
        "current_uniform_home_mult": _UNIFORM_HOME_MULT,
        "by_season": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "predictions_history.db")
    args = parser.parse_args()

    report = analyze(args.db)
    print(f"current _UNIFORM_HOME_MULT = {report['current_uniform_home_mult']}")
    print()
    for season, r in sorted(report["by_season"].items()):
        print(f"season {season} (n={r['n_games']}):")
        print(f"  actual home win rate                              = {r['actual_home_win_rate']:.4f}")
        print(f"  model p_home_raw WITH  uniform term (current)     = {r['model_p_home_raw_with_uniform_term']:.4f}  (residual {r['residual_with_uniform_term']:+.4f})")
        print(f"  model p_home_raw WITHOUT uniform term (Skellam)   = {r['model_p_home_raw_without_uniform_term_skellam_reestimate']:.4f}  (residual {r['residual_without_uniform_term']:+.4f})")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
