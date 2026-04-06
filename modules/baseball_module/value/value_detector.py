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
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ==========================================================
# CONFIGURACIÓN AVANZADA
# ==========================================================

class ValueTier(Enum):
    ULTRA = ("🔥 ULTRA VALUE", 15.0, "S")
    HIGH = ("🟢 HIGH VALUE", 8.0, "A")
    MEDIUM = ("🟡 MEDIUM VALUE", 4.0, "B")
    SLIGHT = ("⚪ SLIGHT VALUE", 1.0, "C")
    NEUTRAL = ("⚫ NEUTRAL", 0.0, "D")
    NEGATIVE = ("🔴 NEGATIVE EV", -999, "F")

@dataclass
class ValueConfig:
    """Configuración del sistema de valor."""
    MIN_KELLY: float = 0.01
    MAX_KELLY: float = 0.15
    FRACTIONAL_KELLY: float = 0.25
    MIN_CONFIDENCE: float = 0.65
    MIN_EDGE: float = 0.5
    VIG_METHODS: List[str] = None
    BOOTSTRAP_SAMPLES: int = 1000
    CI_LEVEL: float = 0.95
    
    def __post_init__(self):
        if self.VIG_METHODS is None:
            self.VIG_METHODS = ['multiplicative', 'power', 'shin']

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
    """Calcula IC usando bootstrap."""
    rng = np.random.default_rng(rng_seed)
    n = len(samples)
    
    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(samples, size=n, replace=True)
        bootstrap_means[i] = np.mean(resample)
    
    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    mean = np.mean(samples)
    
    return mean, lower, upper

# ==========================================================
# MÉTRICAS AVANZADAS
# ==========================================================

def calculate_ev_stats(
    model_prob: float,
    odds: float,
    prob_ci: Optional[Tuple[float, float]] = None
) -> Dict[str, float]:
    """Calcula EV con intervalos de confianza."""
    ev = (model_prob * odds - 1) * 100
    
    if prob_ci:
        ev_lower = (prob_ci[0] * odds - 1) * 100
        ev_upper = (prob_ci[1] * odds - 1) * 100
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
    fractional: float = 0.25
) -> float:
    """Calcula fracción Kelly óptima."""
    if odds <= 1.0 or model_prob <= 0:
        return 0.0
    
    full_kelly = (model_prob * odds - 1) / (odds - 1)
    kelly = np.clip(full_kelly * fractional, CONFIG.MIN_KELLY, CONFIG.MAX_KELLY)
    
    return round(kelly, 4)

def sharpe_ratio(ev: float, ev_std: float) -> float:
    """Calcula Sharpe ratio."""
    if ev_std == 0:
        return 0.0
    return round(ev / ev_std, 3)

def calculate_composite_score(
    ev: float,
    confidence: float,
    edge: float,
    kelly: float,
    sharpe: float,
    market_efficiency: float = 0.5
) -> Dict[str, float]:
    """Score compuesto multi-dimensional."""
    ev_score = np.clip(ev * 3, 0, 100)
    conf_score = confidence * 100
    edge_score = np.clip(edge * 10, 0, 100)
    kelly_score = np.clip(kelly * 200, 0, 100)
    sharpe_score = np.clip(sharpe * 20, 0, 100)
    
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

def classify_value_tier(
    ev: float,
    confidence: float,
    edge: float,
    composite_score: float
) -> ValueTier:
    """Clasifica tier de valor."""
    if ev < 0 or confidence < CONFIG.MIN_CONFIDENCE or edge < CONFIG.MIN_EDGE:
        return ValueTier.NEGATIVE
    
    if composite_score >= 75 and ev >= ValueTier.ULTRA.value[1]:
        return ValueTier.ULTRA
    elif composite_score >= 60 and ev >= ValueTier.HIGH.value[1]:
        return ValueTier.HIGH
    elif composite_score >= 45 and ev >= ValueTier.MEDIUM.value[1]:
        return ValueTier.MEDIUM
    elif composite_score >= 30 and ev >= ValueTier.SLIGHT.value[1]:
        return ValueTier.SLIGHT
    else:
        return ValueTier.NEUTRAL

# ==========================================================
# ANÁLISIS DE MERCADOS ESPECÍFICOS
# ==========================================================

def analyze_market_generic(
    model_prob: float,
    odds: float,
    prob_ci: Tuple[float, float],
    overround: float,
    true_implied: float,
    fractional_kelly: float,
    market_name: str
) -> Dict[str, Any]:
    """Análisis genérico para cualquier mercado."""
    
    ev_stats = calculate_ev_stats(model_prob, odds, prob_ci)
    edge = (model_prob - true_implied) * 100
    kelly = kelly_criterion(model_prob, odds, fractional_kelly)
    sharpe = sharpe_ratio(ev_stats['ev'], ev_stats['ev_std'])
    
    confidence = 1 / (1 + abs(ev_stats['ev_std'] / max(abs(ev_stats['ev']), 0.1)))
    confidence = np.clip(confidence, 0, 1)
    
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
    bootstrap_ci: bool
) -> Dict[str, Any]:
    """
    Analiza Run Line (±1.5).
    
    Home -1.5 = Home gana por 2+ runs
    Away +1.5 = Away gana o pierde por 1 run
    """
    logger.info(f"📊 Analizando Run Line ±{runline_line}")
    
    if home_samples is not None and away_samples is not None:
        # Diferencial de runs
        diff = home_samples - away_samples
        
        # Home -1.5: necesita ganar por 2+
        p_home_cover = np.mean(diff >= 2)
        
        # Away +1.5: cubre si gana o pierde por 1
        p_away_cover = np.mean(diff <= 1)
        
        if bootstrap_ci:
            _, p_home_lower, p_home_upper = bootstrap_confidence_interval(
                (diff >= 2).astype(int), CONFIG.BOOTSTRAP_SAMPLES, CONFIG.CI_LEVEL
            )
            _, p_away_lower, p_away_upper = bootstrap_confidence_interval(
                (diff <= 1).astype(int), CONFIG.BOOTSTRAP_SAMPLES, CONFIG.CI_LEVEL
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
        # Fallback sin samples (aproximación usando distribución)
        logger.warning("⚠️ Sin samples para Run Line, usando aproximación")
        mean_diff = mc_result['mean_home'] - mc_result['mean_away']
        std_diff = math.sqrt(mc_result['mean_home'] + mc_result['mean_away'])  # Poisson variance
        
        p_home_cover = 1 - norm.cdf(1.5, loc=mean_diff, scale=std_diff)
        p_away_cover = norm.cdf(1.5, loc=mean_diff, scale=std_diff)
        
        se = std_diff / math.sqrt(n_sims)
        margin = 1.96 * se
        home_ci = (max(0, p_home_cover - margin), min(1, p_home_cover + margin))
        away_ci = (max(0, p_away_cover - margin), min(1, p_away_cover + margin))
    
    # Ajuste vig
    odds_dict = {'home': runline_home, 'away': runline_away}
    true_implied = adjust_for_vig(odds_dict, method=vig_method)
    overround = (1/runline_home + 1/runline_away - 1) * 100
    
    # Análisis
    home_analysis = analyze_market_generic(
        p_home_cover, runline_home, home_ci, overround,
        true_implied['home'], fractional_kelly, f"RUNLINE HOME -{runline_line}"
    )
    
    away_analysis = analyze_market_generic(
        p_away_cover, runline_away, away_ci, overround,
        true_implied['away'], fractional_kelly, f"RUNLINE AWAY +{runline_line}"
    )
    
    # Mejor pick
    weighted_home = home_analysis['ev'] * home_analysis['confidence'] * (home_analysis['kelly'] * 100)
    weighted_away = away_analysis['ev'] * away_analysis['confidence'] * (away_analysis['kelly'] * 100)
    
    if weighted_home > weighted_away and home_analysis['tier_enum'] != ValueTier.NEGATIVE:
        best_side = f"HOME -{runline_line}"
        best = home_analysis
    elif away_analysis['tier_enum'] != ValueTier.NEGATIVE:
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
    rng_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analiza First 5 Innings.
    
    Ejecuta un Monte Carlo específico para F5 con lambdas ajustadas.
    """
    logger.info("📊 Analizando First 5 Innings")
    
    # Ajustar lambdas para F5 (aproximadamente 55-60% del juego completo)
    f5_factor = 0.58
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
            true_implied['home'], fractional_kelly, "F5 ML HOME"
        )
        
        away_ml = analyze_market_generic(
            p_away, f5_odds.f5_ml_away, away_ci, overround,
            true_implied['away'], fractional_kelly, "F5 ML AWAY"
        )
        
        result['moneyline'] = {
            'home': home_ml,
            'away': away_ml,
        }
    
    # TOTALS F5
    if (f5_odds.f5_total_line and f5_odds.f5_total_over and f5_odds.f5_total_under):
        mean_total = mc_f5['mean_total']
        std_total = mc_f5['std_total']
        
        p_over = 1 - norm.cdf(f5_odds.f5_total_line, loc=mean_total, scale=std_total)
        p_under = norm.cdf(f5_odds.f5_total_line, loc=mean_total, scale=std_total)
        
        # IC aproximado
        se = (norm.pdf((f5_odds.f5_total_line - mean_total) / std_total) * 
              math.sqrt(std_total**2 / n_sims))
        margin = 1.96 * se
        
        over_ci = (max(0, p_over - margin), min(1, p_over + margin))
        under_ci = (max(0, p_under - margin), min(1, p_under + margin))
        
        odds_dict = {'over': f5_odds.f5_total_over, 'under': f5_odds.f5_total_under}
        true_implied = adjust_for_vig(odds_dict, method=vig_method)
        overround = (1/f5_odds.f5_total_over + 1/f5_odds.f5_total_under - 1) * 100
        
        over_total = analyze_market_generic(
            p_over, f5_odds.f5_total_over, over_ci, overround,
            true_implied['over'], fractional_kelly, f"F5 OVER {f5_odds.f5_total_line}"
        )
        
        under_total = analyze_market_generic(
            p_under, f5_odds.f5_total_under, under_ci, overround,
            true_implied['under'], fractional_kelly, f"F5 UNDER {f5_odds.f5_total_line}"
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
    rng_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    EVALUADOR ULTRA - TODOS LOS MERCADOS.
    
    Analiza:
    - Moneyline Full Game
    - Totals O/U (dinámico)
    - Run Line ±1.5
    - First 5 Innings (ML + Totals)
    """
    
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
        
        se_home = math.sqrt(p_home * (1 - p_home) / n_sims)
        se_away = math.sqrt(p_away * (1 - p_away) / n_sims)
        margin_home = 1.96 * se_home
        margin_away = 1.96 * se_away
        
        home_ci = (max(0, p_home - margin_home), min(1, p_home + margin_home))
        away_ci = (max(0, p_away - margin_away), min(1, p_away + margin_away))
        
        odds_dict = {'home': odds.ml_home, 'away': odds.ml_away}
        true_implied = adjust_for_vig(odds_dict, method=vig_method)
        overround = (1/odds.ml_home + 1/odds.ml_away - 1) * 100
        
        home_ml = analyze_market_generic(
            p_home, odds.ml_home, home_ci, overround,
            true_implied['home'], fractional_kelly, "MONEYLINE HOME"
        )
        
        away_ml = analyze_market_generic(
            p_away, odds.ml_away, away_ci, overround,
            true_implied['away'], fractional_kelly, "MONEYLINE AWAY"
        )
        
        all_markets['moneyline'] = {'home': home_ml, 'away': away_ml}
        
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
            p_over = 1 - norm.cdf(odds.total_line, loc=mean_total, scale=std_total)
            p_under = norm.cdf(odds.total_line, loc=mean_total, scale=std_total)
            
            se = norm.pdf((odds.total_line - mean_total) / std_total) * math.sqrt(std_total**2 / n_sims)
            margin = 1.96 * se
            over_ci = (max(0, p_over - margin), min(1, p_over + margin))
            under_ci = (max(0, p_under - margin), min(1, p_under + margin))
        
        odds_dict = {'over': odds.total_over, 'under': odds.total_under}
        true_implied = adjust_for_vig(odds_dict, method=vig_method)
        overround = (1/odds.total_over + 1/odds.total_under - 1) * 100
        
        over_total = analyze_market_generic(
            p_over, odds.total_over, over_ci, overround,
            true_implied['over'], fractional_kelly, f"OVER {odds.total_line}"
        )
        
        under_total = analyze_market_generic(
            p_under, odds.total_under, under_ci, overround,
            true_implied['under'], fractional_kelly, f"UNDER {odds.total_line}"
        )
        
        all_markets['total'] = {'over': over_total, 'under': under_total, 'line': odds.total_line}
        
        if over_total['tier_enum'] != ValueTier.NEGATIVE:
            all_bets.append({**over_total, 'side': f"OVER {odds.total_line}", 'weighted': over_total['ev'] * over_total['confidence'] * over_total['kelly'] * 100})
        if under_total['tier_enum'] != ValueTier.NEGATIVE:
            all_bets.append({**under_total, 'side': f"UNDER {odds.total_line}", 'weighted': under_total['ev'] * under_total['confidence'] * under_total['kelly'] * 100})
    
    # ==========================================================
    # 3. RUN LINE ±1.5
    # ==========================================================
    
    if odds.runline_home and odds.runline_away:
        runline_result = analyze_runline(
            mc_result, odds.runline_home, odds.runline_away, odds.runline_line,
            home_samples, away_samples, n_sims, fractional_kelly, vig_method, bootstrap_ci
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
            lh, la, odds, mc_result['n'], fractional_kelly, vig_method, rng_seed
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
                               'weighted': f5_result['total']['over']['ev'] * f5_result['total']['over']['confidence'] * f5_result['total']['over']['kelly'] * 100})
            if f5_result['total']['under']['tier_enum'] != ValueTier.NEGATIVE:
                all_bets.append({**f5_result['total']['under'], 'side': f"F5 UNDER {f5_result['total']['line']}",
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
                {
                    'rank': i+1,
                    'market': bet['market'],
                    'side': bet['side'],
                    'ev': bet['ev'],
                    'score': bet['composite_score'],
                    'tier': bet['tier'],
                    'kelly': bet['kelly'],
                    'weighted_score': bet['weighted']
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
    
    result = {
        'metadata': {
            'n_simulations': n_sims,
            'converged_early': converged,
            'vig_method': vig_method,
            'fractional_kelly': fractional_kelly,
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
    from monte_carlo_engine import monte_carlo_advanced
    
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
