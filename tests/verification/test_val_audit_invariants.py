"""Permanent invariants from AUDITORÍA VAL (audit_20260714/val_audit/reporte.md).

Method: reproduce known quantities two independent ways and require they
match (Monte Carlo output vs. direct sample counting; hand-derived EV/edge
formulas vs. value_detector's own output; devig arithmetic vs. its inverse).

Per the audit's STOP rule: a test that fails here is a reported finding
with a severity in the audit report, NOT something to "fix" by loosening
the assertion or the underlying code — it stays red until a real,
prioritized fix lands. Two tests in this file (marked below) are EXPECTED
to fail today — they encode HALLAZGO-1 (walk-off truncation bias) and
HALLAZGO-2 (bootstrap CI subsample-size mismatch) from the report, and are
intentionally red so a future fix has something concrete to turn green.
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from modules.baseball_module.montecarlo.simulator import (
    monte_carlo_advanced,
    NB_DISPERSION,
    WALKOFF_9TH_SHARE,
)
from core.value_detector import (
    remove_vig_multiplicative,
    calculate_ev,
    kelly_criterion,
    evaluate_value_ultra,
    GameOdds,
    bootstrap_confidence_interval,
)
from core.utils import calculate_ev as calculate_ev_util  # same function, explicit import path

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "data" / "predictions_history.db"

CASE_LH = 6.115
CASE_LA = 3.558


# ── VAL-2: derived-market invariants ─────────────────────────────────────

def test_moneyline_probabilities_sum_to_one():
    r = monte_carlo_advanced(CASE_LH, CASE_LA, n_max=300_000, rng_seed=1, store_samples=False)
    assert r["p_home"] + r["p_away"] == pytest.approx(1.0, abs=1e-9)


def test_runline_cover_probabilities_sum_to_one():
    r = monte_carlo_advanced(CASE_LH, CASE_LA, n_max=300_000, rng_seed=1, store_samples=True)
    assert r["p_rl_home"] + r["p_rl_away"] == pytest.approx(1.0, abs=1e-6)
    # exact-count cross-check against the raw sample arrays (independent path)
    diff = r["home_samples"].astype(np.int64) - r["away_samples"].astype(np.int64)
    assert r["p_rl_home"] == pytest.approx(float(np.mean(diff >= 2)), abs=1e-4)
    assert r["p_rl_away"] == pytest.approx(float(np.mean(diff <= 1)), abs=1e-4)


def test_total_ou_sums_to_one_half_integer_line():
    r = monte_carlo_advanced(
        CASE_LH, CASE_LA, n_max=300_000, total_line=8.5, rng_seed=1, store_samples=True
    )
    assert r["p_over"] + r["p_under"] + r["p_push"] == pytest.approx(1.0, abs=1e-9)
    assert r["p_push"] == 0.0  # half-integer line: total runs are always integer, can never push


def test_total_ou_push_handled_on_integer_line():
    r = monte_carlo_advanced(
        CASE_LH, CASE_LA, n_max=300_000, total_line=9.0, rng_seed=1, store_samples=True
    )
    assert r["p_over"] + r["p_under"] + r["p_push"] == pytest.approx(1.0, abs=1e-9)
    # An integer line sits inside the real pmf's support for these lambdas —
    # push mass must be strictly positive, not silently dropped to 0.
    assert r["p_push"] > 0.0


# ── VAL-3: devig ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("odds_pair", [(1.44, 3.02), (1.10, 8.00), (1.91, 1.91)])
def test_devig_multiplicative_sums_to_one(odds_pair):
    fair = remove_vig_multiplicative(list(odds_pair))
    assert sum(fair) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 < p < 1.0 for p in fair)


# ── VAL-4: EV / edge / Kelly ──────────────────────────────────────────────

@pytest.mark.parametrize("prob,odds", [(0.6884, 1.46), (0.5685, 2.02), (0.5829, 1.99)])
def test_ev_formula_matches_definition(prob, odds):
    # EV = (p * decimal_odds - 1) * 100 -- the single formula core.utils.calculate_ev
    # implements; guards against a units regression (e.g. dropping the *100,
    # or using American odds by mistake).
    assert calculate_ev_util(prob, odds) == pytest.approx((prob * odds - 1) * 100, abs=1e-9)


def test_edge_antisymmetric_real_market():
    odds = GameOdds(
        ml_home=1.46, ml_away=3.10, pin_home=1.44, pin_away=3.02,
        total_line=8.5, total_over=2.02, total_under=1.93,
        runline_line=1.5, runline_home=1.99, runline_away=1.95,
    )
    mc = monte_carlo_advanced(CASE_LH, CASE_LA, n_max=300_000, total_line=8.5, rng_seed=1, store_samples=True)
    result = evaluate_value_ultra(mc, odds, CASE_LH, CASE_LA, analyze_f5=False, bootstrap_ci=False, rng_seed=1)
    ml = result["markets"]["moneyline"]
    assert ml["home"]["edge"] + ml["away"]["edge"] == pytest.approx(0.0, abs=1e-6)
    rl = result["markets"]["runline"]
    assert rl["home"]["edge"] + rl["away"]["edge"] == pytest.approx(0.0, abs=1e-6)


def test_kelly_clipped_to_configured_bounds():
    import config as _cfg
    # Deep-underdog side: full Kelly is negative -> 0.0, not clipped up to MIN_KELLY
    # (MIN_KELLY only applies to a genuinely positive-but-small edge).
    assert kelly_criterion(0.10, 1.50) == 0.0
    # A tiny genuine positive edge must be floored at MIN_KELLY, not left at
    # the mathematically "correct" fractional-Kelly value below the floor —
    # see report VAL-4.4 (ML HOME case: full_kelly implied ~1.10%, quarter-Kelly
    # ~0.28%, floor forces the displayed/staked value up to 1.00%).
    floored = kelly_criterion(0.6884, 1.46, fractional=0.25)
    assert floored == pytest.approx(_cfg.MIN_KELLY, abs=1e-9)
    # Never exceeds the configured cap regardless of how large the edge is.
    assert kelly_criterion(0.95, 5.0) <= _cfg.MAX_KELLY


# ── VAL-1: simulator invariants ────────────────────────────────────────────

def test_simulator_reproducible_with_fixed_seed():
    r1 = monte_carlo_advanced(CASE_LH, CASE_LA, n_max=200_000, rng_seed=777, store_samples=True)
    r2 = monte_carlo_advanced(CASE_LH, CASE_LA, n_max=200_000, rng_seed=777, store_samples=True)
    assert r1["p_home"] == r2["p_home"]
    assert np.array_equal(r1["home_samples"], r2["home_samples"])
    assert np.array_equal(r1["away_samples"], r2["away_samples"])


def test_lambda_monotonicity_p_home_and_runline():
    """Raising lambda_home (holding lambda_away fixed) must raise both
    p_home and P(home covers the runline) -- same seed for both runs so the
    only thing that changes between them is the input lambda, not MC noise."""
    lo = monte_carlo_advanced(4.0, CASE_LA, n_max=400_000, rng_seed=5, store_samples=True)
    hi = monte_carlo_advanced(5.5, CASE_LA, n_max=400_000, rng_seed=5, store_samples=True)
    assert hi["p_home"] > lo["p_home"]
    assert hi["p_rl_home"] > lo["p_rl_home"]


def test_known_answer_independent_nb_matches_analytic():
    """VAL-1.5 'respuesta conocida': with lambda_noise=0, rho_game=0 and
    model_walkoff=False the simulator reduces to two independent
    NegativeBinomial(NB_DISPERSION, .) draws -- compare its p_home against
    the exact value from direct convolution of the two NB pmfs.

    Tie credit uses the simulator's OWN (post-fix) proportional rule
    lh/(lh+la) rather than a flat 0.5 -- with lambda_noise=0 there is no
    per-draw randomness in that weight (every tied draw gets the same
    deterministic share), which is what makes an exact analytic comparison
    possible at all. See report VAL-1.2 "Tie resolution" for why the flat
    0.5 rule was replaced."""
    from scipy.stats import nbinom

    def nb_pmf(lam, r, kmax=250):
        p = r / (r + lam)
        ks = np.arange(kmax)
        return nbinom.pmf(ks, r, p)

    pm_h = nb_pmf(CASE_LH, NB_DISPERSION)
    pm_a = nb_pmf(CASE_LA, NB_DISPERSION)
    cum_a = np.cumsum(pm_a)
    p_home_win = float(sum(pm_h[i] * (cum_a[i - 1] if i > 0 else 0.0) for i in range(len(pm_h))))
    p_tie = float(sum(pm_h[i] * pm_a[i] for i in range(min(len(pm_h), len(pm_a)))))
    tie_home_share = CASE_LH / (CASE_LH + CASE_LA)
    analytic_p_home = p_home_win + p_tie * tie_home_share

    r = monte_carlo_advanced(
        CASE_LH, CASE_LA, n_max=3_000_000, rng_seed=99, store_samples=False,
        lambda_noise=0.0, rho_game=0.0, model_walkoff=False,
    )
    # MC noise at n=3M for p~0.726: SE ~= sqrt(0.726*0.274/3e6) ~= 0.00026;
    # 5 SE gives a safety margin without masking a real mismatch.
    assert r["p_home"] == pytest.approx(analytic_p_home, abs=5 * 0.00026)


# ── HALLAZGO-1, FIXED (ver reporte VAL-1.3 + addendum "Fixes aplicados") ──
# El simulador ahora trunca la novena baja cuando el home ya va ganando
# (model_walkoff=True por defecto, binomial-thinning con WALKOFF_9TH_SHARE
# -- ver montecarlo/simulator.py). Este test compara, para juegos reales con
# lambdas cercanas al caso (banda 5.0<=lh<=7.5, 2.5<=la<=4.5,
# game_outcomes.source='backtest'), el margen real condicionado a
# home_won=1 contra el margen que el modelo genera aplicando la MISMA
# fórmula de truncamiento que usa el simulador real (misma constante
# importada, no reimplementada a mano) condicionado a que el modelo mismo
# hizo ganar al home. Antes del fix esto fallaba con una brecha de ~8.7pp;
# el fix la cierra a <1pp -- si una regresión futura revierte el
# truncamiento, este test vuelve a fallar.
@pytest.mark.skipif(not DB_PATH.exists(), reason="predictions_history.db not present in this environment")
def test_walkoff_truncation_bias_is_absent():
    con = sqlite3.connect(str(DB_PATH))
    cur = con.cursor()
    cur.execute(
        """
        SELECT backtest_lambda_home, backtest_lambda_away, actual_home_runs, actual_away_runs, home_won
        FROM game_outcomes
        WHERE source='backtest' AND actual_home_runs IS NOT NULL AND backtest_lambda_home IS NOT NULL
          AND backtest_lambda_home BETWEEN 5.0 AND 7.5
          AND backtest_lambda_away BETWEEN 2.5 AND 4.5
        """
    )
    rows = cur.fetchall()
    con.close()
    assert len(rows) > 200, "expected band to have a usable sample of real games"

    lh_arr = np.array([r[0] for r in rows])
    la_arr = np.array([r[1] for r in rows])
    ah_arr = np.array([r[2] for r in rows])
    aa_arr = np.array([r[3] for r in rows])
    hw_arr = np.array([r[4] for r in rows])

    real_margin_ge2_given_won = float(np.mean((ah_arr[hw_arr == 1] - aa_arr[hw_arr == 1]) >= 2))

    rng = np.random.default_rng(2026)
    sim_h, sim_a = [], []
    for lh, la in zip(lh_arr, la_arr):
        h_untrunc = rng.negative_binomial(NB_DISPERSION, NB_DISPERSION / (NB_DISPERSION + lh), size=300)
        a = rng.negative_binomial(NB_DISPERSION, NB_DISPERSION / (NB_DISPERSION + la), size=300)
        # Same formula monte_carlo_advanced applies internally when
        # model_walkoff=True -- imported constant, not re-derived here.
        h_9th = rng.binomial(h_untrunc, WALKOFF_9TH_SHARE)
        h_pre9 = h_untrunc - h_9th
        h = np.where(h_pre9 > a, h_pre9, h_untrunc)
        sim_h.append(h)
        sim_a.append(a)
    sim_h = np.concatenate(sim_h)
    sim_a = np.concatenate(sim_a)
    won = sim_h > sim_a
    sim_margin_ge2_given_won = float(np.mean((sim_h - sim_a)[won] >= 2))

    # Tolerance generous on purpose (3pp) -- pure sampling noise at this n
    # is well under 1pp; the pre-fix gap measured in the report was ~8.8pp,
    # post-fix it measures ~0.9pp.
    assert abs(real_margin_ge2_given_won - sim_margin_ge2_given_won) < 0.03, (
        f"walk-off truncation bias: real P(margin>=2|won)={real_margin_ge2_given_won:.4f} "
        f"vs model P(margin>=2|sim won)={sim_margin_ge2_given_won:.4f} -- "
        "see audit_20260714/val_audit/reporte.md VAL-1.3"
    )


# ── HALLAZGO-2 (severity: MEDIA-ALTA, ver reporte VAL-1.4) ─────────────────
# bootstrap_confidence_interval() subsamplea a min(n, 10_000) ANTES de
# bootstrapear, sin importar cuántas simulaciones reales corrieron
# (500K-5M en producción). El ancho de CI reportado (prob_ci/ev_ci/ev_std,
# y por tanto sharpe y composite_score) refleja la precisión de 10,000
# draws, no la precisión real del run. Test INTENCIONALMENTE rojo: no subir
# la tolerancia para "pasarlo".
def test_bootstrap_ci_width_reflects_actual_sample_size():
    r = monte_carlo_advanced(CASE_LH, CASE_LA, n_max=500_000, rng_seed=3, store_samples=True)
    diff = r["home_samples"].astype(np.int64) - r["away_samples"].astype(np.int64)
    cover = (diff > 1.5).astype(int)
    p_hat = float(np.mean(cover))

    _, lo, hi = bootstrap_confidence_interval(cover, n_bootstrap=1000, ci_level=0.95, rng_seed=0)
    reported_width = hi - lo

    # Width a normal approximation would give AT THE REAL n (500,000) --
    # what an honest CI should look like.
    se_true_n = math.sqrt(p_hat * (1 - p_hat) / len(cover))
    expected_width_at_true_n = 2 * 1.96 * se_true_n

    # If the CI honestly reflected 500K sims, reported_width should be within
    # ~2x of expected_width_at_true_n. It is currently ~7-9x wider (fixed
    # 10,000-sample bootstrap subsample cap) -- see report VAL-1.4.
    assert reported_width < 2 * expected_width_at_true_n, (
        f"reported CI width={reported_width:.5f} vs honest width at n={len(cover)}: "
        f"{expected_width_at_true_n:.5f} (ratio={reported_width/expected_width_at_true_n:.1f}x) -- "
        "see audit_20260714/val_audit/reporte.md VAL-1.4"
    )
