
# ==========================================================
# MONTE CARLO ENGINE G9 PRO ULTRA v2.0
# ==========================================================

import numpy as np
import math
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Negative Binomial dispersion parameter r.
# Empirical MLB 2024-2026 (5,422 games): var/mean = 2.263, mean = 4.427.
# Theoretical estimates from this data: marginal r ≈ 3.51, conditional-on-λ
# OLS r ≈ 3.80, decomposed (−cross-game λ variance) r ≈ 3.67.
#
# r=3.0 was tried first (best raw distribution-tail match: P(0 runs) 6.63%
# model vs 6.69% real, P(8+ runs) 16.13% vs 16.06%) but was REJECTED after
# backtesting: it over-compressed probabilities (Platt a=1.34), collapsing
# the >70% confidence bucket from 190→17 games and costing −5.70pp on
# ROI edge≥10%. r=6.0 has a *worse* raw tail-probability fit (var/mean=1.75,
# P(0 runs)=3.67%, both further from the 2.263/6.69% empirical targets than
# r=3.0) but was ACCEPTED because it produces better betting-relevant
# metrics: ROI edge≥10% +7.39% (new high at the time) and improved
# calibration in the 40-45%/60-70% buckets. See docs/AUDIT_FINDINGS.md
# "PASO 4 — NEGATIVE BINOMIAL" for the full r=3.0 vs r=6.0 backtest comparison.
NB_DISPERSION: float = 6.0

# Var(-ln U) for U~Uniform(0,1) = pi^2/6 (known result: -ln U ~ Exponential(1),
# whose variance is 1). Cov(-ln U, -ln(1-U)) = 1 - pi^2/6 (1-U is also
# Uniform(0,1), but negatively coupled to U through the shared draw). Both
# used only to convert a target rho_game into a sharing probability for the
# Gamma-mixing correlation trick below (_p_share_for_rho) — see
# monte_carlo_advanced's docstring.
_VAR_NEG_LOG_U = math.pi ** 2 / 6.0
_COV_NEG_LOG_U_ANTITHETIC = 1.0 - _VAR_NEG_LOG_U

@dataclass
class MonteCarloLimits:
    MIN_LAMBDA: float = 0.1
    MAX_LAMBDA: float = 20.0
    MIN_SIMS: int = 10_000
    MAX_SIMS: int = 10_000_000
    DEFAULT_BLOCK: int = 200_000
    MIN_BLOCK: int = 10_000
    MAX_NOISE: float = 0.30
    MIN_SE: float = 0.0001
    MAX_SE: float = 0.01

LIMITS = MonteCarloLimits()

# Fraction of expected full-game runs that score in the first 5 innings.
# MLB empirical range: 55–58%; 57.5% is the calibrated midpoint.
F5_SCALE = 0.575

# Assumed share of a team's 9-inning runs that occur in the 9th inning
# specifically — used ONLY by the walk-off truncation below. Uniform
# per-inning assumption (1/9), NOT a fitted empirical constant like
# F5_SCALE above (no per-9th-inning split exists anywhere in this
# codebase yet). Documented explicitly as an assumption so a future
# calibration pass has a clear, named place to plug a real number in.
WALKOFF_9TH_SHARE = 1.0 / 9.0

def validate_inputs(lh, la, n_max, block, lambda_noise, early_stop_se, total_line):
    if not (LIMITS.MIN_LAMBDA <= lh <= LIMITS.MAX_LAMBDA):
        raise ValueError(f"λ_home={lh:.2f} fuera de rango")
    if not (LIMITS.MIN_LAMBDA <= la <= LIMITS.MAX_LAMBDA):
        raise ValueError(f"λ_away={la:.2f} fuera de rango")
    if not (LIMITS.MIN_SIMS <= n_max <= LIMITS.MAX_SIMS):
        raise ValueError(f"n_max={n_max:,} fuera de rango")
    if not (LIMITS.MIN_BLOCK <= block <= n_max):
        raise ValueError(f"block={block:,} inválido")
    if not (0 <= lambda_noise <= LIMITS.MAX_NOISE):
        raise ValueError(f"lambda_noise={lambda_noise} inválido")
    if not (LIMITS.MIN_SE <= early_stop_se <= LIMITS.MAX_SE):
        raise ValueError(f"early_stop_se={early_stop_se} inválido")
    if total_line is not None and not (0 < total_line < 50):
        raise ValueError(f"total_line={total_line} fuera de rango")
    # rho_game validated separately in monte_carlo_advanced after validate_inputs

def monte_carlo_advanced(
    lh: float,
    la: float,
    n_max: int = 5_000_000,
    block: int = 200_000,
    total_line: Optional[float] = None,
    rng_seed: Optional[int] = None,
    lambda_noise: float = 0.05,
    early_stop_se: float = 0.0005,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    store_samples: bool = True,
    analyze_f5: bool = False,
    lh_f5: Optional[float] = None,
    la_f5: Optional[float] = None,
    rho_game: float = -0.008,
    model_walkoff: bool = True,
) -> Dict[str, Any]:
    """
    Vectorized block-based Monte Carlo simulation for MLB run scoring.

    Each simulation draws an independent epistemic λ perturbation per side
    (λ_noise = λ + lambda_noise·λ·z, z ~ N(0,1) independent per team — no
    reason two teams' *parameter uncertainty* should be correlated), then
    scores ~ NegativeBinomial(NB_DISPERSION, p(λ_noise)) for each side via
    the exact Poisson-Gamma mixture representation (NB(r,λ) ≡
    Poisson(Gamma(r, λ/r))). rho_game is injected by having home and away
    SHARE one unit of the underlying Gamma-mixing randomness with a
    probability derived in closed form from rho_game/λh/λa (see
    `_p_share_for_rho`) — antithetic uniforms (U, 1-U) for negative
    correlation, the same U for positive — so the FINAL run totals (not
    just the λ inputs) carry the target correlation. Each side's own NB
    marginal is preserved EXACTLY regardless of whether a given simulation
    shares or not (both branches sum to exactly NB_DISPERSION independent-
    or-shared unit-Gamma components). NB replaces Poisson to match empirical
    MLB run variance: var/mean ≈ 2.26 vs Poisson's 1.0. The marginal
    distributions are preserved exactly — only the joint structure (total
    variance) changes.

    (History, audit_20260714/val_audit/reporte.md VAL-1 bonus finding:
    versions before this one applied rho_game only to the λ-noise inputs,
    which are ~5% of λ in magnitude — the correlation was almost entirely
    swamped by the NB draw's own much larger conditional variance and barely
    survived into the observed home/away run correlation (measured: input
    ρ=-0.008 → observed ρ≈-0.0007, ~11x attenuation; even ρ=-0.5 input only
    gave observed ρ≈-0.0024). An exact Gaussian-copula-on-the-NB-draw
    version was tried next and DOES hit the target correlation precisely,
    but relies on scipy's `nbinom.ppf`/`gamma.ppf`, which benchmarked ~22x
    slower than direct sampling — a 5M-sim worst-case run went from ~6s to
    ~34s. The Gamma-mixing-share trick here keeps the "only the joint
    structure changes" property AND stays on fast native samplers
    (rng.standard_gamma / rng.poisson, no ppf): empirically within ~10-15%
    of the target rho_game across a range of λ combinations, versus the
    original mechanism's >10x undershoot.)

    rho_game < 0: negative correlation → compresses total run variance (pitcher
    duels keep both teams down; one team scoring big makes the other slightly less
    likely to also exceed their mean). Default -0.008 measured empirically from
    5,422 backtest games (actual rho(home_runs, away_runs) = -0.0078).

    Noise adds only ~1% variance on top of pure Poisson and leaves the mean exact.

    model_walkoff (default True): real MLB games never play the bottom of
    the 9th if the home team is already strictly ahead after 8 innings (and
    a bottom 9th that IS played ends the instant the go-ahead run scores —
    that finer mid-inning truncation is NOT modeled, only the "whole half-
    inning skipped" case). Implemented as binomial thinning: each simulated
    home_runs draw is split into a "through 8 innings" / "9th inning" part
    at WALKOFF_9TH_SHARE, and the 9th-inning part is discarded whenever the
    through-8 score alone already exceeds away's final score. away_runs is
    never truncated (away always completes their at-bat in every inning
    they play). Found and quantified in audit_20260714/val_audit/reporte.md
    VAL-1.3: without this, P(home margin>=2 | home won) and P(total>line)
    were both systematically overestimated (+8.7pp / +5.1pp measured against
    881 real games with similar λ) — exactly the runline/totals markets
    where this system's biggest EVs concentrate. Set False to reproduce the
    old (pre-fix) unbiased-NB-only behavior, e.g. for analytic known-answer
    comparisons against a pure NB distribution.

    Tie resolution: MLB has no draws — a simulation tied after regulation
    represents a real game that would go to extra innings. Rather than a
    fixed 50/50 split (which ignores which team was actually favored), each
    tied simulation's win credit is split proportionally to that draw's own
    λ_noise share (lh_noise/(lh_noise+la_noise)) — the same λ already
    driving that simulation's NB draw, so a game where the stronger team
    happens to tie still (correctly) leans toward the stronger team winning
    the "extra frame," rather than coin-flipping regardless of the pregame
    favorite. p_home + p_away still sums to exactly 1.0 by construction
    (tie credit is a partition, not an independent extra draw).

    Early stopping fires after every block once sims_done >= 500_000 and
    SE(p_home) < early_stop_se.  Default threshold 0.0005 ≈ 1 M sims for
    typical win probabilities.

    Returns
    -------
    dict with keys: n, p_home, p_away, mean_home, mean_away, mean_total,
    std_total, bivariate_rho, converged_early, and (when store_samples=True)
    percentiles, home_samples, away_samples, total_samples, and optional
    p_over/p_under/p_push/total_line/f5_home/f5_away/f5_draw.
    """

    # Clamp block to n_max before validation: the default block (200,000)
    # is larger than MIN_SIMS (10,000), so any caller passing a smaller
    # n_max without also overriding block would otherwise hit
    # "block inválido" — a crash for a perfectly reasonable request
    # (run_module.py doesn't override block; only backtest_and_retrain.py did).
    block = min(block, n_max)
    validate_inputs(lh, la, n_max, block, lambda_noise, early_stop_se, total_line)
    if not (-1.0 < rho_game < 1.0):
        raise ValueError(f"rho_game={rho_game} must be in (-1, 1)")

    # rho_game is applied at the Gamma-mixing-share step below, not to the λ
    # noise (see docstring) — the λ-noise Cholesky decomposition for the F5
    # sub-simulation further down still uses this shared sqrt term.
    _rho_sqrt_comp = math.sqrt(max(0.0, 1.0 - rho_game ** 2))
    _sigma_h = lambda_noise * max(lh, 0.5)
    _sigma_a = lambda_noise * max(la, 0.5)

    # Probability of sharing one unit of Gamma-mixing randomness between
    # home/away, derived once (at the input lh/la, not the per-draw noisy
    # lh_noise/la_noise — a second-order refinement not worth the extra
    # per-block cost) so the resulting home/away run correlation lands near
    # rho_game. Var(NB) = λ(1+λ/r) (Poisson-Gamma mixture variance);
    # Cov per shared unit = ±(π²/6 or 1-π²/6)·θ_h·θ_a, θ=λ/r — see the
    # docstring and NB_DISPERSION-adjacent constants above for the derivation.
    def _p_share_for_rho(_lh: float, _la: float, _rho: float) -> float:
        if _rho == 0.0 or _lh <= 0 or _la <= 0:
            return 0.0
        var_h = _lh * (1.0 + _lh / NB_DISPERSION)
        var_a = _la * (1.0 + _la / NB_DISPERSION)
        theta_h = _lh / NB_DISPERSION
        theta_a = _la / NB_DISPERSION
        cov_unit = _COV_NEG_LOG_U_ANTITHETIC if _rho < 0 else _VAR_NEG_LOG_U
        p = (_rho * math.sqrt(var_h * var_a)) / (cov_unit * theta_h * theta_a)
        return float(np.clip(p, 0.0, 1.0))

    _p_share = _p_share_for_rho(lh, la, rho_game)

    rng = np.random.default_rng(rng_seed)
    sims_done = 0

    logger.info(
        f"🎲 Monte Carlo: λ_h={lh:.2f}, λ_a={la:.2f}, max={n_max:,}, ρ={rho_game:+.2f}"
    )

    # Win/loss counters — the authoritative source for p_home/p_away
    wins_home_total = 0
    wins_away_total = 0
    ties_total = 0
    # Proportional (not flat 0.5) tie-break credit accumulator — see
    # docstring "Tie resolution".
    tie_home_share_total = 0.0

    f5_wins_home_total = 0
    f5_wins_away_total = 0
    f5_ties_total = 0

    # Running sums for mean/variance — computed incrementally so they are
    # always consistent with the same draws used for the win counters above.
    # (Previously, store_samples=False regenerated fresh samples from a new
    # RNG state, giving mean/std from a different draw than the counters.)
    sum_home = 0.0
    sum_away = 0.0
    sum_total = 0.0
    sum_sq_total = 0.0  # for variance: E[X²] - E[X]²

    if store_samples:
        all_home: list = []
        all_away: list = []
        all_total: list = []

    while sims_done < n_max:
        b = min(block, n_max - sims_done)

        # Independent epistemic λ perturbation per side — no team-to-team
        # correlation injected here; rho_game is applied below, directly on
        # the NB draw's Gamma-mixing component (see docstring).
        zh = rng.standard_normal(size=b)
        za = rng.standard_normal(size=b)
        lh_noise = np.clip(lh + _sigma_h * zh, LIMITS.MIN_LAMBDA, LIMITS.MAX_LAMBDA)
        la_noise = np.clip(la + _sigma_a * za, LIMITS.MIN_LAMBDA, LIMITS.MAX_LAMBDA)

        # NB(r, λ) ≡ Poisson(Gamma(r, λ/r)) — sample the Gamma-mixing
        # variable, then Poisson conditional on it. With probability
        # _p_share, home/away share one unit of that Gamma randomness
        # (antithetic for rho_game<0, same draw for rho_game>0); the
        # remaining NB_DISPERSION-1 units are always independent. Either
        # way each side's total is exactly Gamma(NB_DISPERSION, λ/r) in
        # distribution, so the NB marginal is untouched — only the
        # home/away covariance changes. See docstring + _p_share_for_rho.
        theta_h = lh_noise / NB_DISPERSION
        theta_a = la_noise / NB_DISPERSION
        if _p_share > 0.0:
            share_mask = rng.random(size=b) < _p_share
            u_share = rng.random(size=b)
            shared_h = -np.log(u_share)
            shared_a = -np.log(1.0 - u_share) if rho_game < 0 else shared_h
            own_h = rng.standard_gamma(1.0, size=b)
            own_a = rng.standard_gamma(1.0, size=b)
            extra_h = np.where(share_mask, shared_h, own_h)
            extra_a = np.where(share_mask, shared_a, own_a)
            g_h = rng.standard_gamma(NB_DISPERSION - 1.0, size=b) + extra_h
            g_a = rng.standard_gamma(NB_DISPERSION - 1.0, size=b) + extra_a
        else:
            g_h = rng.standard_gamma(NB_DISPERSION, size=b)
            g_a = rng.standard_gamma(NB_DISPERSION, size=b)

        home_runs_untrunc = rng.poisson(theta_h * g_h)
        away_runs = rng.poisson(theta_a * g_a)

        if model_walkoff:
            # Real MLB rule: home does not bat the bottom of the 9th if
            # already strictly ahead after 8 innings. Binomial-thin each
            # home_runs draw into "innings 1-8" vs "inning 9" at
            # WALKOFF_9TH_SHARE, then discard the 9th-inning part whenever
            # the through-8 score alone was already a strict lead. away_runs
            # is never truncated (away always completes the innings they play).
            home_9th = rng.binomial(home_runs_untrunc, WALKOFF_9TH_SHARE)
            home_pre9 = home_runs_untrunc - home_9th
            already_ahead = home_pre9 > away_runs
            home_runs = np.where(already_ahead, home_pre9, home_runs_untrunc)
        else:
            home_runs = home_runs_untrunc

        total_runs = home_runs + away_runs

        tie_mask = home_runs == away_runs
        wins_home_total += int(np.sum(home_runs > away_runs))
        wins_away_total += int(np.sum(away_runs > home_runs))
        ties_total      += int(np.sum(tie_mask))
        if np.any(tie_mask):
            # Proportional tie-break credit (see docstring "Tie resolution")
            # instead of a flat 0.5 -- uses this draw's own λ_noise, the
            # same parameter that drove its NB sample.
            tie_home_share_total += float(np.sum(
                lh_noise[tie_mask] / (lh_noise[tie_mask] + la_noise[tie_mask])
            ))

        sum_home     += float(np.sum(home_runs))
        sum_away     += float(np.sum(away_runs))
        sum_total    += float(np.sum(total_runs))
        sum_sq_total += float(np.dot(total_runs.astype(np.float64),
                                     total_runs.astype(np.float64)))

        if analyze_f5:
            if lh_f5 is not None:
                zf1 = rng.standard_normal(size=b)
                zf2 = rng.standard_normal(size=b)
                _s_f5h = lambda_noise * max(lh_f5, 0.5)
                lh_f5_noise = np.clip(
                    lh_f5 + _s_f5h * zf1,
                    LIMITS.MIN_LAMBDA, LIMITS.MAX_LAMBDA,
                )
                if la_f5 is not None:
                    _s_f5a = lambda_noise * max(la_f5, 0.5)
                    la_f5_noise = np.clip(
                        la_f5 + _s_f5a * (rho_game * zf1 + _rho_sqrt_comp * zf2),
                        LIMITS.MIN_LAMBDA, LIMITS.MAX_LAMBDA,
                    )
                else:
                    la_f5_noise = np.clip(la_noise * F5_SCALE, LIMITS.MIN_LAMBDA, LIMITS.MAX_LAMBDA)
            else:
                lh_f5_noise = np.clip(lh_noise * F5_SCALE, LIMITS.MIN_LAMBDA, LIMITS.MAX_LAMBDA)
                if la_f5 is not None:
                    _s_f5a = lambda_noise * max(la_f5, 0.5)
                    zf1 = rng.standard_normal(size=b)
                    zf2 = rng.standard_normal(size=b)
                    la_f5_noise = np.clip(
                        la_f5 + _s_f5a * (rho_game * zf1 + _rho_sqrt_comp * zf2),
                        LIMITS.MIN_LAMBDA, LIMITS.MAX_LAMBDA,
                    )
                else:
                    la_f5_noise = np.clip(la_noise * F5_SCALE, LIMITS.MIN_LAMBDA, LIMITS.MAX_LAMBDA)

            home_f5 = rng.negative_binomial(NB_DISPERSION, NB_DISPERSION / (NB_DISPERSION + lh_f5_noise))
            away_f5 = rng.negative_binomial(NB_DISPERSION, NB_DISPERSION / (NB_DISPERSION + la_f5_noise))
            f5_wins_home_total += int(np.sum(home_f5 > away_f5))
            f5_wins_away_total += int(np.sum(away_f5 > home_f5))
            f5_ties_total      += int(np.sum(home_f5 == away_f5))

        if store_samples:
            all_home.append(home_runs)
            all_away.append(away_runs)
            all_total.append(total_runs)

        sims_done += b

        if progress_callback:
            try:
                progress_callback(sims_done, n_max)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

        # Early stop: check after every block once past the warm-up minimum.
        # The old code checked only when sims_done % 100_000 == 0, which
        # silently skipped early stopping for non-100K-divisible block sizes
        # (e.g. block=333K never satisfied the modulo condition).
        if sims_done >= 500_000:
            p_est = (wins_home_total + tie_home_share_total) / sims_done
            se = math.sqrt(max(p_est * (1.0 - p_est), 1e-9) / sims_done)
            if se < early_stop_se:
                logger.info(
                    f"✅ Convergencia en {sims_done:,} sims (SE={se:.6f} < {early_stop_se})"
                )
                break

    # ── Statistics from running accumulators (always consistent with wins) ──
    mean_home  = sum_home  / sims_done
    mean_away  = sum_away  / sims_done
    mean_total = sum_total / sims_done
    var_total  = max(sum_sq_total / sims_done - mean_total ** 2, 0.0)
    std_total  = math.sqrt(var_total)

    # Ties are split proportionally to each tied draw's own λ_noise share
    # (see docstring "Tie resolution"), not a flat 0.5 — result still sums
    # to exactly 1.0: tie_home_share_total + tie_away_share_total = ties_total
    # by construction (each tied draw contributes shares summing to 1).
    tie_away_share_total = ties_total - tie_home_share_total
    p_home = (wins_home_total + tie_home_share_total) / sims_done
    p_away = (wins_away_total + tie_away_share_total) / sims_done

    results: Dict[str, Any] = {
        "n":               sims_done,
        "p_home":          float(p_home),
        "p_away":          float(p_away),
        "mean_home":       mean_home,
        "mean_away":       mean_away,
        "mean_total":      mean_total,
        "std_total":       std_total,
        "bivariate_rho":   rho_game,
        "converged_early": sims_done < n_max,
    }

    if store_samples:
        final_home  = np.concatenate(all_home)
        final_away  = np.concatenate(all_away)
        final_total = np.concatenate(all_total)

        results["percentiles"] = {
            f"p{p}": float(np.percentile(final_total, p))
            for p in [10, 25, 50, 75, 90]
        }
        # Expose arrays so value_detector can compute bootstrap CIs and
        # sample-based O/U without re-running the simulation.
        results["home_samples"]  = final_home
        results["away_samples"]  = final_away
        results["total_samples"] = final_total

        # Run-line probabilities from stored samples (zero extra sims).
        # home -1.5 covers when home wins by 2+; away +1.5 covers when diff <= 1.
        _diff = final_home.astype(np.int32) - final_away.astype(np.int32)
        results["p_rl_home"] = round(float(np.mean(_diff >= 2)), 4)
        results["p_rl_away"] = round(float(np.mean(_diff <= 1)), 4)

    # ── O/U probabilities ──────────────────────────────────────────────────
    # Use `is not None` (not truthiness) so line=0 would not be skipped.
    # When total_line is absent and analyze_f5=True, derive a "fair" full-game
    # line from the simulation mean so F5 callers get O/U context without
    # needing to pass it explicitly. Otherwise None → skip O/U block.
    line_to_use = (
        total_line if total_line is not None
        else (round(mean_total * 2) / 2 if analyze_f5 else None)
    )
    if line_to_use is not None and not store_samples:
        logger.warning(
            "total_line=%.1f provided but store_samples=False — O/U probabilities not computed",
            line_to_use,
        )
    if line_to_use is not None and store_samples:
        n_arr = len(final_total)
        over  = int(np.sum(final_total > line_to_use))
        under = int(np.sum(final_total < line_to_use))
        push  = n_arr - over - under
        results.update({
            "p_over":      float(over  / n_arr),
            "p_under":     float(under / n_arr),
            "p_push":      float(push  / n_arr),
            "total_line":  float(line_to_use),
        })

    if analyze_f5:
        results.update({
            "f5_home": round(f5_wins_home_total / sims_done, 4),
            "f5_away": round(f5_wins_away_total / sims_done, 4),
            "f5_draw": round(f5_ties_total      / sims_done, 4),
        })

    logger.info(f"✅ Completado: {sims_done:,} sims | P(Home)={p_home:.3f}")
    return results


def monte_carlo_simple(lh: float, la: float, total_line: float = 8.5):
    return monte_carlo_advanced(
        lh, la,
        n_max=200_000,
        block=50_000,
        total_line=total_line,
        rng_seed=42,
        store_samples=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("🧪 TEST: Monte Carlo")
    result = monte_carlo_advanced(
        lh=4.3, la=3.9,
        total_line=8.0,
        n_max=1_000_000,
        store_samples=True,
    )

    print(f"\n📊 RESULTADOS:")
    print(f"  Sims:             {result['n']:,}")
    print(f"  P(Home):          {result['p_home']:.4f}")
    print(f"  P(Away):          {result['p_away']:.4f}")
    print(f"  Sum:              {result['p_home'] + result['p_away']:.6f}")
    print(f"  P(Over):          {result.get('p_over', 'N/A')}")
    print(f"  Media Total:      {result['mean_total']:.3f} ± {result['std_total']:.3f}")
    print(f"  Percentiles:      {result.get('percentiles', 'N/A')}")
    print(f"  home_samples:     {len(result.get('home_samples', []))} samples")
    print(f"  Convergió temprano: {result['converged_early']}")
