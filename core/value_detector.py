# ==========================================================
# VALUE DETECTOR G10 QUANTUM ULTRA - ALL MARKETS EDITION
# ==========================================================
# Autor: Braulio & Claude
# Descripción:
#   Sistema completo de detección de valor para TODOS los mercados:
#   - Moneyline (Full Game)
#   - Totales O/U (dinámico)
#   - Run Line ±1.5
#   - First 5 Innings (ML + Totals)
#   
#   Compatible con The Odds API
# ==========================================================

import numpy as np
import math
import logging
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from scipy.stats import norm, skellam, poisson
import config as _cfg
from core.utils import calculate_ev

logger = logging.getLogger(__name__)

# ==========================================================
# CONFIGURACIÓN AVANZADA
# ==========================================================

class ValueTier(Enum):
    # EV thresholds reset 2026-07-05, RE-refit 2026-07-06 against the first
    # batched backtest re-run that folds in all of that day's engine fixes
    # (LG_DER, LEAGUE_AVG_XWOBA, TTE barrel%, pitcher/bullpen/defense/
    # park-weather fixes) plus a forced Platt-2D refit (a=0.0170, b=0.3847,
    # c=0.7416 -- model weight b went up, market weight c down slightly vs.
    # the prior fit, consistent with the engine fixes making the model more
    # trustworthy). Percentiles of the corrected-EV distribution from this
    # fresh backtest (4,627 games, moneyline, post-Platt-2D) shifted up
    # 5-15% from the 07-05 fit, confirming this re-refit was worth doing,
    # not just measurement noise. New cutoffs ≈ top 5% / 20% / 50% /
    # any-positive of the 2,739 positive-EV sides observed:
    # p95=7.25%, p80=4.26%, p50=2.04%.
    ULTRA = ("🔥 ULTRA VALUE", 7.25, "S")
    HIGH = ("🟢 HIGH VALUE", 4.25, "A")
    MEDIUM = ("🟡 MEDIUM VALUE", 2.0, "B")
    SLIGHT = ("⚪ SLIGHT VALUE", 0.5, "C")
    NEUTRAL = ("⚫ NEUTRAL", 0.0, "D")
    NEGATIVE = ("🔴 NEGATIVE EV", -999, "F")

@dataclass
class ValueConfig:
    """Configuración del sistema de valor."""
    MIN_KELLY: float = _cfg.MIN_KELLY
    MAX_KELLY: float = _cfg.MAX_KELLY
    FRACTIONAL_KELLY: float = _cfg.KELLY_FRACTION
    MIN_CONFIDENCE: float = _cfg.MIN_CONFIDENCE
    MIN_EDGE: float = _cfg.MIN_EDGE
    VIG_METHODS: List[str] = None
    BOOTSTRAP_SAMPLES: int = _cfg.BOOTSTRAP_SAMPLES
    CI_LEVEL: float = _cfg.CI_LEVEL

    def __post_init__(self):
        if self.VIG_METHODS is None:
            self.VIG_METHODS = list(_cfg.VIG_METHODS)

CONFIG = ValueConfig()

# ==========================================================
# ESTRUCTURA DE ODDS (Compatible con The Odds API)
# ==========================================================

@dataclass
class GameOdds:
    """Estructura completa de cuotas desde The Odds API."""
    # Full Game Moneyline
    ml_home: Optional[float] = None
    ml_away: Optional[float] = None

    # Pinnacle moneyline — used as the fair line reference for EV / edge
    pin_home: Optional[float] = None
    pin_away: Optional[float] = None

    # Full Game Totals
    total_line: Optional[float] = None
    total_over: Optional[float] = None
    total_under: Optional[float] = None

    # Run Line (±1.5)
    runline_line: float = 1.5  # Estándar MLB
    runline_home: Optional[float] = None  # Home -1.5
    runline_away: Optional[float] = None  # Away +1.5

    # First 5 Innings Moneyline
    f5_ml_home: Optional[float] = None
    f5_ml_away: Optional[float] = None

    # First 5 Innings Totals
    f5_total_line: Optional[float] = None
    f5_total_over: Optional[float] = None
    f5_total_under: Optional[float] = None

# ==========================================================
# FUNCIONES DE AJUSTE DE VIG
# ==========================================================

def remove_vig_multiplicative(odds_list: List[float]) -> List[float]:
    """Método multiplicativo (más común)."""
    implied = [1/o for o in odds_list]
    total = sum(implied)
    return [imp / total for imp in implied]

def remove_vig_power(odds_list: List[float], k: float = 1.2) -> List[float]:
    """Método power (Joseph Buchdahl)."""
    implied = [1/o for o in odds_list]
    adjusted = [imp ** k for imp in implied]
    total = sum(adjusted)
    return [adj / total for adj in adjusted]

def remove_vig_shin(odds_list: List[float]) -> List[float]:
    """Método Shin (asume información privilegiada)."""
    implied = [1/o for o in odds_list]
    total = sum(implied)
    margin = total - 1
    z = margin / 2
    adjusted = [(imp - z * (1 - imp)) / (1 - z) for imp in implied]
    return adjusted

def _pinnacle_fair_probs(pin_home: float, pin_away: float) -> Tuple[float, float]:
    """
    Devig a Pinnacle two-way moneyline with the multiplicative method.
    Returns (fair_home, fair_away) — probabilities that sum to 1.0.
    Pinnacle carries ~2-3% vig; removing it gives the sharpest available
    fair-line reference for EV and edge calculations.
    """
    fair = remove_vig_multiplicative([pin_home, pin_away])
    return fair[0], fair[1]


def adjust_for_vig(odds_dict: Dict[str, float], method: str = 'multiplicative') -> Dict[str, float]:
    """Ajusta cuotas por overround."""
    if method == 'none':
        return {k: 1/v for k, v in odds_dict.items()}
    
    odds_list = list(odds_dict.values())
    keys = list(odds_dict.keys())
    
    if method == 'multiplicative':
        adjusted = remove_vig_multiplicative(odds_list)
    elif method == 'power':
        adjusted = remove_vig_power(odds_list)
    elif method == 'shin':
        adjusted = remove_vig_shin(odds_list)
    else:
        raise ValueError(f"Unknown vig method: {method}")
    
    return dict(zip(keys, adjusted))

# ==========================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ==========================================================

def bootstrap_confidence_interval(
    samples: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng_seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Vectorized bootstrap CI.  Subsample to 10K max — sufficient for stable
    CI estimates and 50–100× faster than a Python loop on 500K+ arrays.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(samples)
    n_sub = min(n, 10_000)
    sub = (
        rng.choice(samples, size=n_sub, replace=False).astype(np.float64)
        if n_sub < n else samples.astype(np.float64)
    )
    # One vectorized matrix op: n_bootstrap × n_sub → means
    idx = rng.integers(0, n_sub, size=(n_bootstrap, n_sub))
    bootstrap_means = sub[idx].mean(axis=1)
    alpha = 1 - ci_level
    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    return float(np.mean(samples)), lower, upper

# ==========================================================
# MÉTRICAS AVANZADAS
# ==========================================================

def calculate_ev_stats(
    model_prob: float,
    odds: float,
    prob_ci: Optional[Tuple[float, float]] = None,
    push_prob: float = 0.0,
) -> Dict[str, float]:
    """Calcula EV con intervalos de confianza.

    push_prob — probability mass that neither wins nor loses (stake
    refunded), excluded from model_prob/prob_ci. Only real for integer
    total lines (e.g. total_line=9.0 can push at exactly 9 runs; a
    half-integer line like 9.5 never pushes). calculate_ev()'s
    (prob*odds - 1) formula implicitly treats "not win" as "full loss" —
    without this correction, a push silently gets scored as a loss instead
    of a stake return, understating EV. Zero (default) for every other
    market, where it's a no-op.
    """
    ev = calculate_ev(model_prob, odds) + push_prob * 100

    if prob_ci:
        ev_lower = calculate_ev(prob_ci[0], odds) + push_prob * 100
        ev_upper = calculate_ev(prob_ci[1], odds) + push_prob * 100
    else:
        ev_lower = ev_upper = ev

    return {
        'ev': round(ev, 3),
        'ev_lower': round(ev_lower, 3),
        'ev_upper': round(ev_upper, 3),
        'ev_std': round((ev_upper - ev_lower) / 4, 3)
    }

def kelly_criterion(
    model_prob: float,
    odds: float,
    fractional: float = _cfg.KELLY_FRACTION
) -> float:
    """Calcula fracción Kelly óptima."""
    if odds <= 1.0 or model_prob <= 0:
        return 0.0

    full_kelly = (model_prob * odds - 1) / (odds - 1)
    if full_kelly <= 0:
        return 0.0

    kelly = np.clip(full_kelly * fractional, CONFIG.MIN_KELLY, CONFIG.MAX_KELLY)
    return round(kelly, 4)

def sharpe_ratio(ev: float, ev_std: float) -> float:
    """Calcula Sharpe ratio."""
    if ev_std == 0:
        return 0.0
    return round(ev / ev_std, 3)

# Internal 0-100 scaling anchors for calculate_composite_score's components,
# recalibrated 2026-07-05 against the real post-Platt-2D corrected EV
# distribution (same 2,643 positive-EV moneyline sides as the ValueTier
# reset above). The old constants (ev*3, kelly*200, sharpe*20) were never
# revisited after Platt-2D shrank real edges toward the market — three
# independent, uncited scaling bugs found via the same empirical check
# (Fable review 2026-07-05):
#   ev_score: ev*3 needs ev=33.3 to reach 100, but real p99 ev is only
#     10.96 (p99 backed by ~26 obs, too thin/volatile to anchor on) — even
#     the best real bets used ~1/3 of the component's range. Rescaled to
#     the more stable p97 (~79 backing obs): ev=7.92 -> ev_score=100.
#     RE-refit 2026-07-06 against the first batched engine-fix backtest
#     (2,739 positive-EV sides, forced Platt-2D refit) -- p97 moved
#     7.92->8.86, a real ~12% shift (not noise), confirming this anchor
#     needs to move again each time the batched backtest is re-run, same
#     as the ValueTier EV thresholds above.
_EV_SCORE_ANCHOR = 8.86
#   kelly_score: kelly*200 assumes kelly can approach 0.5, but
#     kelly_criterion() hard-clips to CONFIG.MAX_KELLY=0.15 (quarter-Kelly,
#     15%-of-bankroll cap) — kelly_score's real achievable ceiling was 30,
#     not 100, permanently shorting kelly_component by 7 of its 10 nominal
#     points and making ValueTier.ULTRA's gate (composite>=67) mathematically
#     unreachable regardless of confidence, since the true ceiling of the
#     whole formula was ~65.1 (verified: calculate_composite_score(ev=100,
#     confidence=1.0, edge=100, kelly=MAX_KELLY, sharpe=100,
#     market_efficiency=1.0) == 65.1). Rescaled against the real hard cap.
#   sharpe_score: sharpe*20 needs sharpe=5 to reach 100, but real p80
#     sharpe is already 6.42 -- the opposite failure mode (saturating too
#     early, not too late): roughly the top 20% of real bets all received
#     the identical maxed-out sharpe_component, collapsing differentiation
#     exactly at the ULTRA/HIGH boundary. Note: sharpe is derived from EV
#     divided by a roughly-constant Monte Carlo sampling-precision target
#     (early-stopping SE<0.003), so sharpe correlates strongly with ev
#     itself (r=0.82 on the same population, measured) -- sharpe_score is
#     substantially redundant with ev_score. At only 0.10 composite weight
#     this is bounded double-counting, not a load-bearing distortion (unlike
#     e.g. Pitcher Engine's kbb_mult/SIERA collinearity), so it's rescaled
#     for consistency rather than redesigned. Anchored more conservatively
#     at p90 (not p97 like ev_score) because its input, ev_std, isn't a
#     directly observed backtest quantity like ev/edge -- it's a physics-
#     derived proxy (ev_std ~= 0.98*SE*odds*100, since d(ev)/dp=odds and MC
#     early-stopping targets SE(p)<0.003), stacking estimation uncertainty
#     on top of the usual thin-tail percentile noise.
#     RE-refit 2026-07-06 alongside _EV_SCORE_ANCHOR (8.83->9.47, ~7% shift,
#     same batched-backtest population); ev/sharpe correlation confirmed
#     stable (r=0.818, was 0.816) -- the redundancy finding still holds.
#
#     RE-refit 2026-07-12: the "early-stopping SE<0.003" assumption above
#     was the backtest's precision (n_max=50_000, no early-stop override --
#     backtest_and_retrain.py always hits that cap before early_stop_se's
#     default 0.0005 target is reachable, so its *effective* SE is whatever
#     50_000 sims gives: sqrt(0.25/50_000)=0.00224 at the worst-case p=0.5,
#     close to the comment's rounded 0.003). Live runs (run_module.py,
#     n_max=5_000_000) DO reach the 0.0005 early-stop target for virtually
#     every game -- live SE is ~4.47x tighter than backtest SE. Since
#     ev_std is directly proportional to SE (same physics-derived proxy
#     noted above), live sharpe values run ~4.47x hotter than the backtest
#     population this anchor was fit against for the same EV -- confirmed
#     as the likely cause of sharpe_score saturating at 100 for nearly
#     every live positive-EV bet (contributing to today's 6/7 moneyline
#     picks landing ULTRA tier). Rescaled by that same ratio:
#     9.47 * (sqrt(0.25/50_000) / 0.0005) = 9.47 * 4.472 = 42.35. This is a
#     tier/composite-score constant only -- it does not touch lambda,
#     probability, or the moneyline Brier/accuracy backtest metric at all,
#     so validate by checking the live tier distribution, not by re-running
#     backtest_and_retrain.py (which would correctly show zero change).
_SHARPE_SCORE_ANCHOR = 42.35

def calculate_composite_score(
    ev: float,
    confidence: float,
    edge: float,
    kelly: float,
    sharpe: float,
    market_efficiency: float = 0.5
) -> Dict[str, float]:
    """Score compuesto multi-dimensional."""
    ev_score = np.clip(ev / _EV_SCORE_ANCHOR * 100, 0, 100)
    conf_score = confidence * 100
    edge_score = np.clip(edge * 10, 0, 100)
    kelly_score = np.clip(kelly / CONFIG.MAX_KELLY * 100, 0, 100)
    sharpe_score = np.clip(sharpe / _SHARPE_SCORE_ANCHOR * 100, 0, 100)

    market_penalty = 1 - (market_efficiency * 0.3)
    
    composite = (
        ev_score * 0.40 +
        conf_score * 0.25 +
        edge_score * 0.15 +
        kelly_score * 0.10 +
        sharpe_score * 0.10
    ) * market_penalty
    
    return {
        'composite_score': round(composite, 2),
        'ev_component': round(ev_score * 0.40, 2),
        'conf_component': round(conf_score * 0.25, 2),
        'edge_component': round(edge_score * 0.15, 2),
        'kelly_component': round(kelly_score * 0.10, 2),
        'sharpe_component': round(sharpe_score * 0.10, 2),
        'market_penalty': round(market_penalty, 3)
    }

    # composite_score gates reset 2026-07-05, same population/methodology as
    # the ValueTier EV thresholds above (top 5%/20%/50%/any-positive of the
    # real post-Platt-2D positive-EV sides) -- computed AFTER the three
    # scaling fixes above (kelly/ev/sharpe), since the pre-fix distribution
    # was contaminated by all three bugs simultaneously, not just at the
    # tiers where it happened to produce visibly-impossible numbers (old
    # ULTRA gate=67 was unreachable, ceiling ~65.1). The old gates' apparent
    # non-degenerate pass-rates at MEDIUM/SLIGHT weren't a deliberately
    # chosen selectivity target either -- same contaminated distribution,
    # they just landed somewhere nonzero by coincidence of gate placement.
    # RE-refit 2026-07-06 against the first batched engine-fix backtest
    # (2,739 positive-EV sides, forced Platt-2D refit) -- gates barely moved
    # (p95=49.60, p80=37.73, p50=26.95, p5=16.51 vs. the 07-05 fit's
    # 50.33/37.52/26.33/15.59) because ev_score's anchor moved proportionally
    # with the EV distribution itself (a ratio component self-stabilizes),
    # unlike the ValueTier EV thresholds above which moved a real 5-15%.
    # Gates: ULTRA=p95=50, HIGH=p80=38, MEDIUM=p50=27, SLIGHT=p5=17 (a low
    # floor, not a selectivity target, matching SLIGHT's EV floor being "any
    # real positive edge" rather than a percentile cut). Verified the
    # realized joint population under this function's actual if/elif
    # AND-logic (EV floor x composite floor together, not each marginal
    # alone): of the 2,739-bet population, ULTRA=3.87%, HIGH=12.71%,
    # MEDIUM=27.24%, SLIGHT=32.24%, NEUTRAL=12.38% (positive EV, below
    # SLIGHT's composite floor), NEGATIVE=11.57% (confidence/edge floor
    # rejects) -- close to the intended top-5%/15%/50% shape despite
    # EV-rank and composite-rank not being perfectly correlated; not a
    # second unreachable-tier surprise. Provisional again: re-run this whole
    # refit (EV thresholds + scaling anchors + composite gates, same single
    # pass) the next time the batched backtest re-runs.
def classify_value_tier(
    ev: float,
    confidence: float,
    edge: float,
    composite_score: float
) -> ValueTier:
    """Clasifica tier de valor."""
    if ev < 0 or confidence < CONFIG.MIN_CONFIDENCE or edge < CONFIG.MIN_EDGE:
        return ValueTier.NEGATIVE

    if composite_score >= 50 and ev >= ValueTier.ULTRA.value[1]:
        return ValueTier.ULTRA
    elif composite_score >= 38 and ev >= ValueTier.HIGH.value[1]:
        return ValueTier.HIGH
    elif composite_score >= 27 and ev >= ValueTier.MEDIUM.value[1]:
        return ValueTier.MEDIUM
    elif composite_score >= 17 and ev >= ValueTier.SLIGHT.value[1]:
        return ValueTier.SLIGHT
    else:
        return ValueTier.NEUTRAL

# ==========================================================
# ANÁLISIS DE MERCADOS ESPECÍFICOS
# ==========================================================

def compute_data_quality_confidence(game_meta: Optional[Dict[str, Any]]) -> float:
    """
    Real epistemic-confidence score: how much genuine current-season data
    backs this game's inputs, vs. the old MC-sampling-based "confidence"
    that collapsed to ~0.99 for nearly every bet regardless of real
    uncertainty (docs/AUDIT_FINDINGS.md — Value Detector BUG #1).

    Combines three signals (v1 scope, per Fable review 2026-07-04), each
    taking the WORSE of the two teams/pitchers — a game's real input
    quality is bounded by its weakest leg, not the average:

      qB_SP  (40%) — starting-pitcher innings pitched this season / 30.
        quality_mult (32% of Pitcher Engine's weight, the single largest
        sub-factor pipeline-wide) is still mostly shrunk toward league
        average below this, i.e. genuinely less differentiated.
      qA_TTE (35%) — 1 − prior_weight. Whether the offensive λ is real
        current-season signal or mostly a prior-season fallback blend.
      qC_Kalman (25%) — Kalman observations / 10. Whether the learned
        run-rate correction is actually active for this team/context.

    Missing individual fields default to the WORST value for that signal
    (prior_weight=1.0, ip=0, n_obs=0) rather than silently assuming good
    data — absence of information must never produce high confidence.

    Scope note: this answers "does the model have real information for
    this game," not the separate, still-open edge-overconfidence/
    calibration problem (audit's high-edge bucket: predicted 60.3% vs
    actual 51.7%). Do not treat this as having fixed that; it hasn't.
    """
    gm = game_meta or {}
    prior_weight = max(
        float(gm.get("home_prior_weight", 1.0) or 1.0),
        float(gm.get("away_prior_weight", 1.0) or 1.0),
    )
    sp_ip = min(
        float(gm.get("home_sp_ip", 0) or 0),
        float(gm.get("away_sp_ip", 0) or 0),
    )
    kalman_n = min(
        float(gm.get("home_kalman_n_obs", 0) or 0),
        float(gm.get("away_kalman_n_obs", 0) or 0),
    )

    q_tte    = 1.0 - min(1.0, max(0.0, prior_weight))
    q_sp     = min(1.0, sp_ip / 30.0)
    q_kalman = min(1.0, kalman_n / 10.0)

    score = 0.40 * q_sp + 0.35 * q_tte + 0.25 * q_kalman
    return round(max(0.0, min(1.0, score)), 4)


def analyze_market_generic(
    model_prob: float,
    odds: float,
    prob_ci: Tuple[float, float],
    overround: float,
    true_implied: float,
    fractional_kelly: float,
    market_name: str,
    confidence: float,
    push_prob: float = 0.0,
) -> Dict[str, Any]:
    """Análisis genérico para cualquier mercado.

    confidence — real epistemic-confidence score for this GAME (not this
    market), from compute_data_quality_confidence(). Same value across
    every market of the same game; computed once by the caller.

    push_prob — see calculate_ev_stats()'s docstring. Zero for every
    market except integer-line totals.
    """

    ev_stats = calculate_ev_stats(model_prob, odds, prob_ci, push_prob)
    edge = (model_prob - true_implied) * 100
    kelly = kelly_criterion(model_prob, odds, fractional_kelly)
    sharpe = sharpe_ratio(ev_stats['ev'], ev_stats['ev_std'])

    score = calculate_composite_score(
        ev_stats['ev'], confidence, edge, kelly, sharpe,
        market_efficiency=(1 - overround / 100)
    )
    
    tier = classify_value_tier(ev_stats['ev'], confidence, edge, score['composite_score'])
    
    return {
        'market': market_name,
        'probability': round(model_prob, 4),
        'prob_ci': [round(prob_ci[0], 4), round(prob_ci[1], 4)],
        'odds': odds,
        'implied_prob_raw': round(1/odds, 4),
        'implied_prob_true': round(true_implied, 4),
        'ev': ev_stats['ev'],
        'ev_ci': [ev_stats['ev_lower'], ev_stats['ev_upper']],
        'ev_std': ev_stats['ev_std'],
        'edge': round(edge, 3),
        'kelly': kelly,
        'sharpe': sharpe,
        'confidence': round(confidence, 3),
        'composite_score': score['composite_score'],
        'score_breakdown': score,
        'tier': tier.value[0],
        'tier_grade': tier.value[2],
        'tier_enum': tier,
    }

def analyze_runline(
    mc_result: Dict[str, Any],
    runline_home: float,
    runline_away: float,
    runline_line: float,
    home_samples: Optional[np.ndarray],
    away_samples: Optional[np.ndarray],
    n_sims: int,
    fractional_kelly: float,
    vig_method: str,
    bootstrap_ci: bool,
    confidence: float,
) -> Dict[str, Any]:
    """
    Analiza Run Line (±runline_line — 1.5 es el estándar MLB, pero se
    respeta el valor real recibido: la cobertura se calcula contra
    runline_line, no contra un umbral 1.5 fijo).

    Home -runline_line = Home gana por más que la línea
    Away +runline_line = Away cubre si el diferencial no llega a la línea
    """
    logger.info(f"📊 Analizando Run Line ±{runline_line}")
    
    if home_samples is not None and away_samples is not None:
        # Diferencial de runs
        diff = home_samples - away_samples

        # Home -runline_line: gana por más que la línea (diff es entero,
        # runline_line es un medio-entero típico de MLB, ej. 1.5 -> diff>=2)
        p_home_cover = np.mean(diff > runline_line)

        # Away +runline_line: cubre si el diferencial no llega a la línea
        p_away_cover = np.mean(diff < runline_line)

        if bootstrap_ci:
            _, p_home_lower, p_home_upper = bootstrap_confidence_interval(
                (diff > runline_line).astype(int), CONFIG.BOOTSTRAP_SAMPLES, CONFIG.CI_LEVEL
            )
            _, p_away_lower, p_away_upper = bootstrap_confidence_interval(
                (diff < runline_line).astype(int), CONFIG.BOOTSTRAP_SAMPLES, CONFIG.CI_LEVEL
            )
            home_ci = (p_home_lower, p_home_upper)
            away_ci = (p_away_lower, p_away_upper)
        else:
            se_home = math.sqrt(p_home_cover * (1 - p_home_cover) / n_sims)
            se_away = math.sqrt(p_away_cover * (1 - p_away_cover) / n_sims)
            margin_home = 1.96 * se_home
            margin_away = 1.96 * se_away
            home_ci = (max(0, p_home_cover - margin_home), min(1, p_home_cover + margin_home))
            away_ci = (max(0, p_away_cover - margin_away), min(1, p_away_cover + margin_away))
    else:
        # Fallback sin samples: use Skellam distribution (difference of two independent
        # Poisson variables) which is exact for integer run differentials.
        # P(home-away > runline_line) = 1 - P(D <= floor(runline_line))
        # where D ~ Skellam(lh, la). floor() is exact for the conventional
        # half-integer MLB runline (1.5 -> 1, 2.5 -> 2, ...).
        logger.warning("⚠️ Sin samples para Run Line, usando Skellam CDF")
        lh = mc_result.get('mean_home', 4.5)
        la = mc_result.get('mean_away', 4.5)
        lh = max(lh, 0.01)
        la = max(la, 0.01)
        _rl_floor = math.floor(runline_line)

        p_home_cover = float(1 - skellam.cdf(_rl_floor, lh, la))
        p_away_cover = float(skellam.cdf(_rl_floor, lh, la))

        # Bernoulli SE for the probability estimate (same formula as the samples path).
        # Note: Var(D) = lh + la is the run-differential variance, NOT the SE of P(D>runline_line).
        se_home = math.sqrt(p_home_cover * (1 - p_home_cover) / n_sims)
        se_away = math.sqrt(p_away_cover * (1 - p_away_cover) / n_sims)
        home_ci = (max(0.0, p_home_cover - 1.96 * se_home), min(1.0, p_home_cover + 1.96 * se_home))
        away_ci = (max(0.0, p_away_cover - 1.96 * se_away), min(1.0, p_away_cover + 1.96 * se_away))
    
    # Ajuste vig
    odds_dict = {'home': runline_home, 'away': runline_away}
    true_implied = adjust_for_vig(odds_dict, method=vig_method)
    overround = (1/runline_home + 1/runline_away - 1) * 100
    
    # Análisis
    home_analysis = analyze_market_generic(
        p_home_cover, runline_home, home_ci, overround,
        true_implied['home'], fractional_kelly, f"RUNLINE HOME -{runline_line}", confidence
    )

    away_analysis = analyze_market_generic(
        p_away_cover, runline_away, away_ci, overround,
        true_implied['away'], fractional_kelly, f"RUNLINE AWAY +{runline_line}", confidence
    )
    
    # Mejor pick
    weighted_home = home_analysis['ev'] * home_analysis['confidence'] * (home_analysis['kelly'] * 100)
    weighted_away = away_analysis['ev'] * away_analysis['confidence'] * (away_analysis['kelly'] * 100)
    
    home_valid = home_analysis['tier_enum'] != ValueTier.NEGATIVE
    away_valid = away_analysis['tier_enum'] != ValueTier.NEGATIVE

    # A valid (non-NEGATIVE) side must never lose to "NO BET" just because
    # the other side's weighted score happened to be higher while that
    # other side was itself NEGATIVE-tier (disqualified) — only compare
    # weighted scores between two sides that are both actually bettable.
    if home_valid and (not away_valid or weighted_home > weighted_away):
        best_side = f"HOME -{runline_line}"
        best = home_analysis
    elif away_valid:
        best_side = f"AWAY +{runline_line}"
        best = away_analysis
    else:
        best_side = "NO BET"
        best = None
    
    return {
        'home': home_analysis,
        'away': away_analysis,
        'recommendation': {
            'best_side': best_side,
            'best_tier': best['tier'] if best else ValueTier.NEGATIVE.value[0],
            'best_grade': best['tier_grade'] if best else 'F',
            'best_ev': best['ev'] if best else 0,
            'best_score': best['composite_score'] if best else 0,
            'suggested_kelly': best['kelly'] if best else 0,
        },
        'market_info': {
            'runline': runline_line,
            'overround_pct': round(overround, 2),
        }
    }

def analyze_first5(
    lh: float,
    la: float,
    f5_odds: GameOdds,
    n_max: int,
    fractional_kelly: float,
    vig_method: str,
    confidence: float,
    rng_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analiza First 5 Innings.

    Ejecuta un Monte Carlo específico para F5 con lambdas ajustadas.
    """
    logger.info("📊 Analizando First 5 Innings")

    # Use the same F5 scale constant as the simulator (empirical 57.5% calibrated midpoint).
    from modules.baseball_module.montecarlo.simulator import F5_SCALE as _F5_SCALE
    f5_factor = _F5_SCALE
    lh_f5 = lh * f5_factor
    la_f5 = la * f5_factor
    
    # Mini Monte Carlo para F5
    from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced
    
    mc_f5 = monte_carlo_advanced(
        lh=lh_f5,
        la=la_f5,
        n_max=min(n_max, 2_000_000),  # Menos sims para F5
        total_line=f5_odds.f5_total_line,
        rng_seed=rng_seed,
        store_samples=True
    )
    
    n_sims = mc_f5['n']
    result = {'metadata': {'n_simulations': n_sims, 'f5_factor': f5_factor}}
    
    # MONEYLINE F5
    if f5_odds.f5_ml_home and f5_odds.f5_ml_away:
        p_home = mc_f5['p_home']
        p_away = mc_f5['p_away']
        
        se_home = math.sqrt(p_home * (1 - p_home) / n_sims)
        se_away = math.sqrt(p_away * (1 - p_away) / n_sims)
        margin_home = 1.96 * se_home
        margin_away = 1.96 * se_away
        
        home_ci = (max(0, p_home - margin_home), min(1, p_home + margin_home))
        away_ci = (max(0, p_away - margin_away), min(1, p_away + margin_away))
        
        odds_dict = {'home': f5_odds.f5_ml_home, 'away': f5_odds.f5_ml_away}
        true_implied = adjust_for_vig(odds_dict, method=vig_method)
        overround = (1/f5_odds.f5_ml_home + 1/f5_odds.f5_ml_away - 1) * 100
        
        home_ml = analyze_market_generic(
            p_home, f5_odds.f5_ml_home, home_ci, overround,
            true_implied['home'], fractional_kelly, "F5 ML HOME", confidence
        )

        away_ml = analyze_market_generic(
            p_away, f5_odds.f5_ml_away, away_ci, overround,
            true_implied['away'], fractional_kelly, "F5 ML AWAY", confidence
        )
        
        result['moneyline'] = {
            'home': home_ml,
            'away': away_ml,
        }
    
    # TOTALS F5
    if (f5_odds.f5_total_line and f5_odds.f5_total_over and f5_odds.f5_total_under):
        mean_total = mc_f5['mean_total']
        std_total = mc_f5['std_total']
        
        # Poisson CDF: total runs ~ Poisson(lh_f5 + la_f5).
        # For a half-point line (e.g. 4.5): no push, over and under are complementary.
        # For an integer line (e.g. 4.0): push at total==4; over/under don't sum to 1.
        lam_total = max(mean_total, 0.01)
        line_val = f5_odds.f5_total_line
        line_floor = int(math.floor(line_val))
        is_integer_line = (line_val == line_floor)

        p_over = float(1 - poisson.cdf(line_floor, lam_total))
        p_under = float(
            poisson.cdf(line_floor - 1, lam_total) if is_integer_line
            else poisson.cdf(line_floor, lam_total)
        )
        # Same integer-line push handling as evaluate_value_ultra()'s main
        # totals section — see calculate_ev_stats()'s push_prob docstring.
        p_push = float(poisson.pmf(line_floor, lam_total)) if is_integer_line else 0.0

        se = math.sqrt(p_over * (1 - p_over) / max(n_sims, 1))
        margin = 1.96 * se
        over_ci = (max(0.0, p_over - margin), min(1.0, p_over + margin))
        under_ci = (max(0.0, p_under - margin), min(1.0, p_under + margin))

        odds_dict = {'over': f5_odds.f5_total_over, 'under': f5_odds.f5_total_under}
        true_implied = adjust_for_vig(odds_dict, method=vig_method)
        overround = (1/f5_odds.f5_total_over + 1/f5_odds.f5_total_under - 1) * 100

        over_total = analyze_market_generic(
            p_over, f5_odds.f5_total_over, over_ci, overround,
            true_implied['over'], fractional_kelly, f"F5 OVER {f5_odds.f5_total_line}", confidence,
            push_prob=p_push,
        )

        under_total = analyze_market_generic(
            p_under, f5_odds.f5_total_under, under_ci, overround,
            true_implied['under'], fractional_kelly, f"F5 UNDER {f5_odds.f5_total_line}", confidence,
            push_prob=p_push,
        )
        
        result['total'] = {
            'over': over_total,
            'under': under_total,
            'line': f5_odds.f5_total_line,
        }
    
    return result

# ==========================================================
# EVALUADOR ULTRA (TODOS LOS MERCADOS)
# ==========================================================

def evaluate_value_ultra(
    mc_result: Dict[str, Any],
    odds: GameOdds,
    lh: float,
    la: float,
    vig_method: str = 'multiplicative',
    fractional_kelly: float = 0.25,
    bootstrap_ci: bool = True,
    home_samples: Optional[np.ndarray] = None,
    away_samples: Optional[np.ndarray] = None,
    total_samples: Optional[np.ndarray] = None,
    analyze_f5: bool = True,
    rng_seed: Optional[int] = None,
    game_meta: Optional[Dict[str, Any]] = None,
    p_home_corrector: Optional[Callable[[float, float], float]] = None,
) -> Dict[str, Any]:
    """
    EVALUADOR ULTRA - TODOS LOS MERCADOS.

    Analiza:
    - Moneyline Full Game
    - Totals O/U (dinámico)
    - Run Line ±1.5
    - First 5 Innings (ML + Totals)

    game_meta — real epistemic-confidence inputs for this game (starting
    pitcher IP, TTE prior_weight, Kalman n_obs per team). See
    compute_data_quality_confidence() for the formula. Missing/None means
    every field defaults to its worst-case value — confidence will be low,
    not silently high.

    p_home_corrector — optional Platt-2D correction, signature
    (raw_p_home, market_prob_home) -> corrected_p_home. Kept as a plain
    callable (not a LearningEngine import) so this sport-agnostic module
    doesn't depend on MLB-specific calibration internals; the MLB caller
    binds its own fitted (a,b,c) and season via a closure/partial. Applied
    ONLY to moneyline (and F5 moneyline) against a Pinnacle fair line —
    it was fit on that specific relationship and has no evidence it
    generalizes to runline/totals or non-Pinnacle devigged lines.
    """

    confidence = compute_data_quality_confidence(game_meta)

    logger.info("=" * 80)
    logger.info("🚀 VALUE DETECTOR G10 QUANTUM ULTRA - ALL MARKETS")
    logger.info("=" * 80)
    
    n_sims = mc_result['n']
    converged = mc_result.get('converged_early', False)
    
    all_markets = {}
    all_bets = []
    
    # ==========================================================
    # 1. MONEYLINE FULL GAME
    # ==========================================================
    
    if odds.ml_home and odds.ml_away:
        logger.info("📊 Analizando Moneyline Full Game")

        p_home = mc_result['p_home']
        p_away = mc_result['p_away']

        # Overround from the bet odds we're actually taking
        overround = (1/odds.ml_home + 1/odds.ml_away - 1) * 100

        # Fair-line reference: Pinnacle devigged > consensus devigged
        # Pinnacle carries ~2-3% vig vs 5-8% for softer books, making its
        # devigged line the most accurate publicly available fair-probability.
        if odds.pin_home and odds.pin_away:
            fair_home, fair_away = _pinnacle_fair_probs(odds.pin_home, odds.pin_away)
            pin_vig_pct = round((1/odds.pin_home + 1/odds.pin_away - 1) * 100, 2)
            fair_source = "pinnacle"
            logger.info(
                f"   📌 Pinnacle fair line: {odds.pin_home}/{odds.pin_away} "
                f"→ {fair_home:.4f}/{fair_away:.4f} (vig={pin_vig_pct:.1f}%)"
            )
        else:
            devigged = adjust_for_vig({'home': odds.ml_home, 'away': odds.ml_away}, method=vig_method)
            fair_home, fair_away = devigged['home'], devigged['away']
            pin_vig_pct = None
            fair_source = vig_method

        # Platt-2D: shrink model_prob toward the market's fair line before
        # computing edge/EV/Kelly. Only valid against a Pinnacle fair line —
        # it was fit on that specific (p_home, pinnacle_fair, outcome) triple.
        if p_home_corrector is not None and fair_source == "pinnacle":
            p_home_raw_local = p_home
            p_home = p_home_corrector(p_home_raw_local, fair_home)
            p_away = 1.0 - p_home
            logger.info(
                f"   🎯 Platt-2D: p_home {p_home_raw_local:.4f} → {p_home:.4f} "
                f"(market fair={fair_home:.4f})"
            )

        se_home = math.sqrt(p_home * (1 - p_home) / n_sims)
        se_away = math.sqrt(p_away * (1 - p_away) / n_sims)
        margin_home = 1.96 * se_home
        margin_away = 1.96 * se_away

        home_ci = (max(0, p_home - margin_home), min(1, p_home + margin_home))
        away_ci = (max(0, p_away - margin_away), min(1, p_away + margin_away))

        home_ml = analyze_market_generic(
            p_home, odds.ml_home, home_ci, overround,
            fair_home, fractional_kelly, "MONEYLINE HOME", confidence
        )

        away_ml = analyze_market_generic(
            p_away, odds.ml_away, away_ci, overround,
            fair_away, fractional_kelly, "MONEYLINE AWAY", confidence
        )

        all_markets['moneyline'] = {
            'home': home_ml,
            'away': away_ml,
            'fair_source': fair_source,
            'pin_home': odds.pin_home,
            'pin_away': odds.pin_away,
            'pin_fair_home': round(fair_home, 4) if fair_source == 'pinnacle' else None,
            'pin_fair_away': round(fair_away, 4) if fair_source == 'pinnacle' else None,
            'pin_vig_pct': pin_vig_pct,
        }
        
        if home_ml['tier_enum'] != ValueTier.NEGATIVE:
            all_bets.append({**home_ml, 'side': 'HOME', 'weighted': home_ml['ev'] * home_ml['confidence'] * home_ml['kelly'] * 100})
        if away_ml['tier_enum'] != ValueTier.NEGATIVE:
            all_bets.append({**away_ml, 'side': 'AWAY', 'weighted': away_ml['ev'] * away_ml['confidence'] * away_ml['kelly'] * 100})
    
    # ==========================================================
    # 2. TOTALS O/U (DINÁMICO)
    # ==========================================================
    
    if odds.total_line and odds.total_over and odds.total_under:
        logger.info(f"📊 Analizando Totals O/U (Línea: {odds.total_line})")
        
        mean_total = mc_result['mean_total']
        std_total = mc_result['std_total']
        
        if total_samples is not None:
            p_over = np.mean(total_samples > odds.total_line)
            p_under = np.mean(total_samples < odds.total_line)
            # Only nonzero for an integer line (total runs are always
            # integer-valued, so a half-integer line like 9.5 can never
            # push) — see calculate_ev_stats()'s push_prob docstring.
            p_push = float(np.mean(total_samples == odds.total_line))

            if bootstrap_ci:
                _, p_over_lower, p_over_upper = bootstrap_confidence_interval(
                    (total_samples > odds.total_line).astype(int), CONFIG.BOOTSTRAP_SAMPLES, CONFIG.CI_LEVEL
                )
                _, p_under_lower, p_under_upper = bootstrap_confidence_interval(
                    (total_samples < odds.total_line).astype(int), CONFIG.BOOTSTRAP_SAMPLES, CONFIG.CI_LEVEL
                )
                over_ci = (p_over_lower, p_over_upper)
                under_ci = (p_under_lower, p_under_upper)
            else:
                se_over = math.sqrt(p_over * (1 - p_over) / n_sims)
                se_under = math.sqrt(p_under * (1 - p_under) / n_sims)
                over_ci = (max(0, p_over - 1.96*se_over), min(1, p_over + 1.96*se_over))
                under_ci = (max(0, p_under - 1.96*se_under), min(1, p_under + 1.96*se_under))
        else:
            # No samples path: use Poisson CDF (total runs ~ Poisson(lh + la))
            lam_total = max(mean_total, 0.01)
            line_val = odds.total_line
            line_floor = int(math.floor(line_val))
            is_integer_line = (line_val == line_floor)

            p_over = float(1 - poisson.cdf(line_floor, lam_total))
            p_under = float(
                poisson.cdf(line_floor - 1, lam_total) if is_integer_line
                else poisson.cdf(line_floor, lam_total)
            )
            p_push = float(poisson.pmf(line_floor, lam_total)) if is_integer_line else 0.0

            se = math.sqrt(p_over * (1 - p_over) / max(n_sims, 1))
            margin = 1.96 * se
            over_ci = (max(0.0, p_over - margin), min(1.0, p_over + margin))
            under_ci = (max(0.0, p_under - margin), min(1.0, p_under + margin))
        
        odds_dict = {'over': odds.total_over, 'under': odds.total_under}
        true_implied = adjust_for_vig(odds_dict, method=vig_method)
        overround = (1/odds.total_over + 1/odds.total_under - 1) * 100
        
        over_total = analyze_market_generic(
            p_over, odds.total_over, over_ci, overround,
            true_implied['over'], fractional_kelly, f"OVER {odds.total_line}", confidence,
            push_prob=p_push,
        )

        under_total = analyze_market_generic(
            p_under, odds.total_under, under_ci, overround,
            true_implied['under'], fractional_kelly, f"UNDER {odds.total_line}", confidence,
            push_prob=p_push,
        )
        
        all_markets['total'] = {'over': over_total, 'under': under_total, 'line': odds.total_line}
        
        if over_total['tier_enum'] != ValueTier.NEGATIVE:
            all_bets.append({**over_total, 'side': f"OVER {odds.total_line}", 'line': odds.total_line, 'weighted': over_total['ev'] * over_total['confidence'] * over_total['kelly'] * 100})
        if under_total['tier_enum'] != ValueTier.NEGATIVE:
            all_bets.append({**under_total, 'side': f"UNDER {odds.total_line}", 'line': odds.total_line, 'weighted': under_total['ev'] * under_total['confidence'] * under_total['kelly'] * 100})
    
    # ==========================================================
    # 3. RUN LINE ±1.5
    # ==========================================================
    
    if odds.runline_home and odds.runline_away:
        runline_result = analyze_runline(
            mc_result, odds.runline_home, odds.runline_away, odds.runline_line,
            home_samples, away_samples, n_sims, fractional_kelly, vig_method, bootstrap_ci,
            confidence,
        )
        
        all_markets['runline'] = runline_result
        
        if runline_result['home']['tier_enum'] != ValueTier.NEGATIVE:
            all_bets.append({**runline_result['home'], 'side': f"HOME -{odds.runline_line}", 
                           'weighted': runline_result['home']['ev'] * runline_result['home']['confidence'] * runline_result['home']['kelly'] * 100})
        if runline_result['away']['tier_enum'] != ValueTier.NEGATIVE:
            all_bets.append({**runline_result['away'], 'side': f"AWAY +{odds.runline_line}",
                           'weighted': runline_result['away']['ev'] * runline_result['away']['confidence'] * runline_result['away']['kelly'] * 100})
    
    # ==========================================================
    # 4. FIRST 5 INNINGS
    # ==========================================================
    
    if analyze_f5 and (odds.f5_ml_home or odds.f5_total_line):
        f5_result = analyze_first5(
            lh, la, odds, mc_result['n'], fractional_kelly, vig_method, confidence, rng_seed
        )
        
        all_markets['first5'] = f5_result
        
        # Agregar F5 ML bets
        if 'moneyline' in f5_result:
            if f5_result['moneyline']['home']['tier_enum'] != ValueTier.NEGATIVE:
                all_bets.append({**f5_result['moneyline']['home'], 'side': 'F5 HOME',
                               'weighted': f5_result['moneyline']['home']['ev'] * f5_result['moneyline']['home']['confidence'] * f5_result['moneyline']['home']['kelly'] * 100})
            if f5_result['moneyline']['away']['tier_enum'] != ValueTier.NEGATIVE:
                all_bets.append({**f5_result['moneyline']['away'], 'side': 'F5 AWAY',
                               'weighted': f5_result['moneyline']['away']['ev'] * f5_result['moneyline']['away']['confidence'] * f5_result['moneyline']['away']['kelly'] * 100})
        
        # Agregar F5 Totals bets
        if 'total' in f5_result:
            if f5_result['total']['over']['tier_enum'] != ValueTier.NEGATIVE:
                all_bets.append({**f5_result['total']['over'], 'side': f"F5 OVER {f5_result['total']['line']}",
                               'line': f5_result['total']['line'],
                               'weighted': f5_result['total']['over']['ev'] * f5_result['total']['over']['confidence'] * f5_result['total']['over']['kelly'] * 100})
            if f5_result['total']['under']['tier_enum'] != ValueTier.NEGATIVE:
                all_bets.append({**f5_result['total']['under'], 'side': f"F5 UNDER {f5_result['total']['line']}",
                               'line': f5_result['total']['line'],
                               'weighted': f5_result['total']['under']['ev'] * f5_result['total']['under']['confidence'] * f5_result['total']['under']['kelly'] * 100})
    
    # ==========================================================
    # RANKING GLOBAL
    # ==========================================================
    
    all_bets.sort(key=lambda x: x['weighted'], reverse=True)
    
    if all_bets:
        best = all_bets[0]
        global_recommendation = {
            'best_market': best['market'],
            'best_side': best['side'],
            'best_tier': best['tier'],
            'best_grade': best['tier_grade'],
            'best_ev': best['ev'],
            'best_score': best['composite_score'],
            'suggested_kelly': best['kelly'],
            'all_opportunities': [
                # Spread the full underlying bet dict rather than hand-picking
                # fields: this was a minimal top-10 display projection, but
                # two independent downstream consumers (track_record/
                # publisher.py needed odds/model_prob/line; ui/mlb.py's
                # save_value_picks needed confidence) both turned out to
                # silently fall back to wrong defaults because this dict
                # didn't carry a field they assumed was there — found
                # 2026-07-06, twice in one session. Rather than keep
                # patching field-by-field, expose everything; 'tier_enum'
                # is the one deliberate exclusion (a raw ValueTier Enum,
                # not JSON-serializable — 'tier'/'tier_grade' already carry
                # its string/grade for external consumption).
                {
                    **{k: v for k, v in bet.items() if k != 'tier_enum'},
                    'rank': i+1,
                    'score': bet['composite_score'],
                    'weighted_score': bet['weighted'],
                }
                for i, bet in enumerate(all_bets[:10])  # Top 10
            ]
        }
    else:
        global_recommendation = {
            'best_market': 'NONE',
            'best_side': 'NO BET',
            'best_tier': ValueTier.NEGATIVE.value[0],
            'best_grade': 'F',
            'best_ev': 0,
            'best_score': 0,
            'suggested_kelly': 0,
            'all_opportunities': []
        }
    
    # ==========================================================
    # OUTPUT FINAL
    # ==========================================================
    
    # Determine the fair-source that was used (moneyline may not have run)
    _ml = all_markets.get('moneyline', {})
    _fair_source_used = _ml.get('fair_source', vig_method)

    result = {
        'metadata': {
            'n_simulations': n_sims,
            'converged_early': converged,
            'vig_method': vig_method,
            'fractional_kelly': fractional_kelly,
            'fair_source': _fair_source_used,
            'pin_home': odds.pin_home,
            'pin_away': odds.pin_away,
        },
        'markets': all_markets,
        'global_recommendation': global_recommendation
    }
    
    logger.info("=" * 80)
    logger.info(f"🏆 MEJOR OPORTUNIDAD GLOBAL:")
    logger.info(f"   Market: {global_recommendation['best_market']}")
    logger.info(f"   Side: {global_recommendation['best_side']}")
    logger.info(f"   Tier: {global_recommendation['best_tier']} [{global_recommendation['best_grade']}]")
    logger.info(f"   EV: {global_recommendation['best_ev']:.2f}%")
    logger.info(f"   Kelly: {global_recommendation['suggested_kelly']*100:.1f}% bankroll")
    logger.info("=" * 80)
    
    return result

# ==========================================================
# WRAPPER COMPLETO
# ==========================================================

def full_game_analysis(
    lh: float,
    la: float,
    odds: GameOdds,
    n_max: int = 5_000_000,
    fractional_kelly: float = 0.25,
    analyze_f5: bool = True
) -> Dict[str, Any]:
    """Análisis completo del juego."""
    from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced
    
    logger.info("🚀 INICIANDO ANÁLISIS COMPLETO")
    
    mc_result = monte_carlo_advanced(
        lh=lh,
        la=la,
        n_max=n_max,
        total_line=odds.total_line,
        store_samples=True
    )
    
    value_result = evaluate_value_ultra(
        mc_result=mc_result,
        odds=odds,
        lh=lh,
        la=la,
        fractional_kelly=fractional_kelly,
        home_samples=mc_result.get('home_samples'),
        away_samples=mc_result.get('away_samples'),
        total_samples=mc_result.get('total_samples'),
        analyze_f5=analyze_f5
    )
    
    return {
        'monte_carlo': mc_result,
        'value_analysis': value_result
    }

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # Ejemplo con todas las cuotas
    odds = GameOdds(
        ml_home=1.85,
        ml_away=2.05,
        total_line=8.5,
        total_over=1.95,
        total_under=1.90,
        runline_home=2.20,  # Home -1.5
        runline_away=1.67,  # Away +1.5
        f5_ml_home=1.90,
        f5_ml_away=2.00,
        f5_total_line=4.5,
        f5_total_over=1.92,
        f5_total_under=1.93,
    )
    
    result = full_game_analysis(
        lh=4.5,
        la=3.8,
        odds=odds,
        n_max=2_000_000,
        analyze_f5=True
    )
    
    print("\n" + "="*80)
    print("🏆 TOP OPPORTUNITIES:")
    print("="*80)
    for opp in result['value_analysis']['global_recommendation']['all_opportunities'][:5]:
        print(f"#{opp['rank']:2} | {opp['market']:15} | {opp['side']:20} | "
              f"EV: {opp['ev']:6.2f}% | Score: {opp['score']:5.1f} | "
              f"{opp['tier']:20} | Kelly: {opp['kelly']*100:4.1f}%")
