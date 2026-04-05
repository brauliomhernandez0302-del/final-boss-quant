"""
NBA G10+ ULTRA PRO V2 - ADVANCED GAME PREDICTION SYSTEM
=========================================================

Sistema profesional de predicción NBA con 12 engines/dictámenes mejorados:
Context, Injuries, Pace, Monte Carlo, Shooting Luck, Sharp Money,
Blowout, Matchups, HFA, Risk, Stability, Trends.

MEJORAS V2:
- Monte Carlo con correlación y distribución realista
- EV real calculado contra odds del mercado
- Injury Engine con interacciones no lineales
- Matchups expandido con 10+ factores
- Altitude adjustment (Denver factor)
- Sharp Money mejorado con steam moves
- Kelly Criterion optimizado
- Validación de datos mejorada

Autor: Braulio
Versión: G10+ Ultra Pro V2
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTES Y CONFIGURACIÓN V2
# ============================================================

# Promedios NBA 2024-25 season
NBA_AVG_POINTS = 114.5
NBA_AVG_PACE = 99.5
NBA_AVG_ORtg = 114.5
NBA_AVG_DRtg = 114.5
NBA_AVG_eFG = 0.555
NBA_AVG_PAINT_PTS = 48.0
NBA_AVG_FASTBREAK = 14.0
NBA_AVG_TURNOVERS = 13.5
NBA_AVG_FT_RATE = 0.25

# Pesos de los 12 Dictámenes/Engines (normalizados)
ENGINE_WEIGHTS: Dict[str, float] = {
    "context": 0.18,
    "injuries": 0.20,
    "pace": 0.14,
    "monte_carlo": 0.16,
    "shooting_luck": 0.08,
    "sharp_money": 0.12,
    "matchups": 0.10,
    "hfa": 0.04,
    "blowout": 0.03,
    "risk": 0.05,
    "stability": 0.05,
    "trends": 0.05,
}

# Context Engine - Pesos de factores mejorados
CONTEXT_FACTORS: Dict[str, float] = {
    "b2b": -3.0,
    "3in4": -2.0,
    "4in6": -1.5,
    "5in7": -1.0,
    "timezone_west_to_east": -1.2,
    "timezone_east_to_west": -0.6,
    "cross_country_flight": -0.8,
    "rest_advantage_3plus": 2.0,
    "rest_advantage_2": 1.0,
    "win_streak_bonus": 0.35,
    "loss_streak_penalty": -0.25,
    "rivalry_boost": 0.6,
    "desperation_boost": 1.0,
    "elimination_game": 1.5,
    "national_tv": 0.3,
    "altitude_denver": -1.8,
    "altitude_utah": -0.8,
}

# Injury Impact - Niveles mejorados con más granularidad
INJURY_LEVELS: Dict[str, Dict[str, float]] = {
    "mvp": {"offensive": -6.5, "defensive": -4.0, "pace": -2.5, "chemistry": -2.0},
    "superstar": {"offensive": -5.0, "defensive": -3.0, "pace": -2.0, "chemistry": -1.5},
    "allstar": {"offensive": -4.0, "defensive": -2.5, "pace": -1.5, "chemistry": -1.0},
    "star": {"offensive": -3.5, "defensive": -2.0, "pace": -1.2, "chemistry": -0.8},
    "starter": {"offensive": -2.0, "defensive": -1.2, "pace": -0.5, "chemistry": -0.5},
    "rotation": {"offensive": -0.8, "defensive": -0.5, "pace": -0.2, "chemistry": -0.2},
    "bench": {"offensive": -0.3, "defensive": -0.2, "pace": 0.0, "chemistry": 0.0},
}

# Matchup Factors expandidos
MATCHUP_FACTORS: Dict[str, float] = {
    "paint_scoring": 0.12,
    "fastbreak": 0.10,
    "turnover_diff": 0.15,
    "free_throw_rate": 0.08,
    "second_chance": 0.10,
    "bench_scoring": 0.08,
    "three_point_volume": 0.12,
    "defensive_rebounding": 0.10,
    "assist_ratio": 0.08,
    "clutch_performance": 0.07,
}

# Blowout & Pace constants
BLOWOUT_THRESHOLD = 18
BLOWOUT_SEVERE_THRESHOLD = 25
BLOWOUT_PACE_REDUCTION = 0.12
BLOWOUT_SEVERE_REDUCTION = 0.18

# Shooting luck regression
REGRESSION_FACTOR = 0.40
SAMPLE_SIZE_THRESHOLD = 5

# Sharp money thresholds
SHARP_MONEY_THRESHOLD = 10.0
RLM_THRESHOLD = 1.5
STEAM_MOVE_THRESHOLD = 2.0

# Altitude arenas
ALTITUDE_ARENAS: Dict[str, float] = {
    "Denver Nuggets": 1.8,
    "Utah Jazz": 0.8,
    "Phoenix Suns": 0.3,
}

# Correlación base entre scores (pace compartido)
SCORE_CORRELATION = 0.18


# ============================================================
# DATA CLASSES PARA ESTRUCTURA
# ============================================================

@dataclass
class TeamData:
    """Estructura de datos de equipo."""
    name: str
    offensive_rating: float = NBA_AVG_ORtg
    defensive_rating: float = NBA_AVG_DRtg
    pace: float = NBA_AVG_PACE
    pace_last_5: float = NBA_AVG_PACE
    pace_last_10: float = NBA_AVG_PACE
    pace_home: float = NBA_AVG_PACE
    pace_away: float = NBA_AVG_PACE
    pace_vs_similar: float = NBA_AVG_PACE
    efg_last_5: float = NBA_AVG_eFG
    efg_season: float = NBA_AVG_eFG
    scoring_variance: float = 8.0
    consistency_score: float = 70.0
    points_last_3: float = 0.0
    points_last_5: float = 0.0
    points_last_10: float = 0.0
    points_per_game: float = NBA_AVG_POINTS
    hfa_multiplier: float = 1.0
    # Matchup stats
    points_in_paint: float = NBA_AVG_PAINT_PTS
    opp_points_in_paint: float = NBA_AVG_PAINT_PTS
    fastbreak_points: float = NBA_AVG_FASTBREAK
    opp_fastbreak_points: float = NBA_AVG_FASTBREAK
    turnovers: float = NBA_AVG_TURNOVERS
    steals: float = 7.5
    free_throw_rate: float = NBA_AVG_FT_RATE
    offensive_rebounds: float = 10.0
    defensive_rebounds: float = 34.0
    assists: float = 25.0
    three_pt_offense: float = 0.36
    three_pt_defense: float = 0.36
    three_pt_attempts: float = 35.0
    bench_points: float = 35.0
    clutch_net_rating: float = 0.0
    pnr_offense_rating: float = 100.0
    pnr_defense_rating: float = 100.0
    rebounding_rate: float = 50.0


@dataclass
class GameContext:
    """Estructura de contexto del partido."""
    game_date: str = ""
    home_b2b: bool = False
    away_b2b: bool = False
    home_3in4: bool = False
    away_3in4: bool = False
    home_4in6: bool = False
    away_4in6: bool = False
    home_5in7: bool = False
    away_5in7: bool = False
    home_days_rest: int = 1
    away_days_rest: int = 1
    away_travel_miles: int = 0
    timezone_direction: str = "none"
    home_win_streak: int = 0
    away_win_streak: int = 0
    home_loss_streak: int = 0
    away_loss_streak: int = 0
    rivalry_game: bool = False
    home_desperation: bool = False
    away_desperation: bool = False
    elimination_game: bool = False
    national_tv: bool = False
    playoff_game: bool = False
    home_injuries: List[Dict[str, Any]] = field(default_factory=list)
    away_injuries: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MarketData:
    """Estructura de datos de mercado."""
    opening_spread: float = 0.0
    current_spread: float = 0.0
    opening_total: float = 225.0
    current_total: float = 225.0
    tickets_pct_home: float = 50.0
    money_pct_home: float = 50.0
    tickets_pct_over: float = 50.0
    money_pct_over: float = 50.0
    home_ml_open: int = 0
    home_ml_current: int = 0
    away_ml_open: int = 0
    away_ml_current: int = 0
    steam_moves: List[Dict[str, Any]] = field(default_factory=list)
    sharp_reports: List[str] = field(default_factory=list)


@dataclass
class BetRecommendation:
    """Estructura de recomendación de apuesta."""
    market: str
    side: str
    probability: float
    implied_prob: float
    edge: float
    ev_per_unit: float
    kelly_full: float
    kelly_quarter: float
    confidence: float
    rating: str
    reliability: float
    bet_worthy: bool


# ============================================================
# CLASE PRINCIPAL: NBA ANALYZER G10+ V2
# ============================================================

class NBAAnalyzerG10PlusV2:
    """
    Analizador avanzado NBA V2 con 12 engines/dictámenes mejorados.
    """

    def __init__(self) -> None:
        self.engine_weights = ENGINE_WEIGHTS.copy()
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Valida que los pesos sumen aproximadamente 1.0."""
        total = sum(self.engine_weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Engine weights sum to {total:.3f}, normalizing...")
            for key in self.engine_weights:
                self.engine_weights[key] /= total

    # --------------------------------------------------------
    # MÉTODO PRINCIPAL
    # --------------------------------------------------------
    def analyze_game(
        self,
        home_team: Dict[str, Any],
        away_team: Dict[str, Any],
        game_context: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
        n_simulations: int = 100_000,
    ) -> Dict[str, Any]:
        """
        Análisis completo con 12 dictámenes mejorados.
        """

        logger.info("\n" + "=" * 70)
        logger.info("NBA G10+ ULTRA PRO V2 - COMPREHENSIVE ANALYSIS")
        logger.info(f"{away_team.get('name', 'Away')} @ {home_team.get('name', 'Home')}")
        logger.info("=" * 70 + "\n")

        try:
            if game_context is None:
                game_context = {}
            if market_data is None:
                market_data = {}

            # ========================================
            # DICTAMEN 1: CONTEXT ENGINE (18%)
            # ========================================
            logger.info("🔥 DICTAMEN 1/12: CONTEXT ENGINE V2 (18%)")
            context_analysis = self._analyze_context_v2(
                home_team, away_team, game_context
            )

            home_context_adj = context_analysis["home_adjustment"]
            away_context_adj = context_analysis["away_adjustment"]

            logger.info(f"  Home context: {home_context_adj:+.2f} pts")
            logger.info(f"  Away context: {away_context_adj:+.2f} pts")
            logger.info(f"  Altitude factor: {context_analysis.get('altitude_factor', 0):+.2f}")
            logger.info(f"  Context confidence: {context_analysis['confidence']:.1%}\n")

            # ========================================
            # DICTAMEN 2: INJURY & LINEUP IMPACT V2 (20%)
            # ========================================
            logger.info("⚕️ DICTAMEN 2/12: INJURY & LINEUP IMPACT V2 (20%)")
            injury_analysis = self._analyze_injuries_v2(
                home_team, away_team, game_context
            )

            home_injury_adj = injury_analysis["home_adjustment"]
            away_injury_adj = injury_analysis["away_adjustment"]

            logger.info(f"  Home injury impact: {home_injury_adj:+.2f} pts")
            logger.info(f"  Away injury impact: {away_injury_adj:+.2f} pts")
            logger.info(f"  Chemistry disruption: {injury_analysis.get('chemistry_impact', 0):+.2f}")
            logger.info(f"  Injury confidence: {injury_analysis['confidence']:.1%}\n")

            # ========================================
            # DICTAMEN 3: DYNAMIC PACE MODEL V2 (14%)
            # ========================================
            logger.info("📊 DICTAMEN 3/12: DYNAMIC PACE MODEL V2 (14%)")
            pace_analysis = self._analyze_pace_v2(
                home_team, away_team, game_context
            )

            game_pace = pace_analysis["predicted_pace"]

            logger.info(f"  Predicted pace: {game_pace:.1f}")
            logger.info(
                f"  Q1-Q3 pace: {pace_analysis['early_pace']:.1f} | "
                f"Q4 pace: {pace_analysis['fourth_quarter_pace']:.1f}"
            )
            logger.info(f"  Pace volatility: {pace_analysis['pace_volatility']:.2f}")
            logger.info(f"  Pace confidence: {pace_analysis['confidence']:.1%}\n")

            # ========================================
            # DICTAMEN 4: MONTE CARLO ENGINE V2 (16%)
            # ========================================
            logger.info("🎲 DICTAMEN 4/12: MONTE CARLO SIMULATION V2 (16%)")

            # Calcular ratings base
            home_ortg = home_team.get("offensive_rating", NBA_AVG_ORtg)
            home_drtg = home_team.get("defensive_rating", NBA_AVG_DRtg)
            away_ortg = away_team.get("offensive_rating", NBA_AVG_ORtg)
            away_drtg = away_team.get("defensive_rating", NBA_AVG_DRtg)

            # Aplicar ajustes contexto/lesiones
            total_home_adj = home_context_adj + home_injury_adj
            total_away_adj = away_context_adj + away_injury_adj

            # HFA (dictamen 9 integrado)
            hfa_analysis = self._analyze_hfa_v2(home_team, away_team, game_context)
            hfa_points = hfa_analysis["hfa_points"]

            # Predicción de puntos base
            home_points_base = (
                home_ortg * away_drtg / NBA_AVG_ORtg
            ) * (game_pace / 100.0)
            away_points_base = (
                away_ortg * home_drtg / NBA_AVG_ORtg
            ) * (game_pace / 100.0)

            home_points_expected = home_points_base + hfa_points + total_home_adj
            away_points_expected = away_points_base + total_away_adj

            # Monte Carlo V2 con correlación
            home_var = home_team.get("scoring_variance", 8.0)
            away_var = away_team.get("scoring_variance", 8.0)

            sim_results = self._run_monte_carlo_v2(
                home_points_expected,
                away_points_expected,
                home_var,
                away_var,
                pace_analysis,
                n_simulations,
            )

            logger.info(f"  Simulations: {n_simulations:,}")
            logger.info(f"  Correlation used: {sim_results['correlation_used']:.2f}")
            logger.info(f"  Home expected: {home_points_expected:.1f} (σ={home_var:.1f})")
            logger.info(f"  Away expected: {away_points_expected:.1f} (σ={away_var:.1f})")
            logger.info(
                f"  Total expected: {home_points_expected + away_points_expected:.1f}\n"
            )

            # ========================================
            # DICTAMEN 5: SHOOTING LUCK & EFFICIENCY V2 (8%)
            # ========================================
            logger.info("🎯 DICTAMEN 5/12: SHOOTING LUCK & EFFICIENCY V2 (8%)")
            shooting_analysis = self._analyze_shooting_luck_v2(
                home_team, away_team
            )

            home_luck_adj = shooting_analysis["home_regression_adj"]
            away_luck_adj = shooting_analysis["away_regression_adj"]

            logger.info(f"  Home luck adjustment: {home_luck_adj:+.2f} pts")
            logger.info(f"  Away luck adjustment: {away_luck_adj:+.2f} pts")
            logger.info(f"  Home shooting trend: {shooting_analysis['home_trend']}")
            logger.info(f"  Away shooting trend: {shooting_analysis['away_trend']}")
            logger.info(
                f"  Shooting confidence: {shooting_analysis['confidence']:.1%}\n"
            )

            # Aplicar ajuste de luck
            home_points_final = home_points_expected + home_luck_adj
            away_points_final = away_points_expected + away_luck_adj
            total_final = home_points_final + away_points_final
            spread_final = home_points_final - away_points_final

            # ========================================
            # DICTAMEN 6: SHARP MONEY MODEL V2 (12%)
            # ========================================
            logger.info("💵 DICTAMEN 6/12: SHARP MONEY ANALYSIS V2 (12%)")
            sharp_analysis = self._analyze_sharp_money_v2(
                market_data,
                spread_final,
                total_final,
            )

            logger.info(f"  Market edge detected: {sharp_analysis['edge_detected']}")
            logger.info(f"  Reverse line movement: {sharp_analysis['rlm_detected']}")
            logger.info(f"  Steam moves: {sharp_analysis['steam_move_count']}")
            logger.info(f"  Sharp side: {sharp_analysis['sharp_side'] or 'None'}")
            logger.info(
                f"  Sharp money confidence: {sharp_analysis['confidence']:.1%}\n"
            )

            # ========================================
            # DICTAMEN 7: BLOWOUT & DAMPENER V2 (3%)
            # ========================================
            logger.info("🌡️ DICTAMEN 7/12: BLOWOUT & SCORE DAMPENER V2 (3%)")
            blowout_analysis = self._analyze_blowout_risk_v2(
                home_points_final,
                away_points_final,
                sim_results,
            )

            logger.info(
                f"  Blowout probability: {blowout_analysis['blowout_prob']:.1%}"
            )
            logger.info(
                f"  Severe blowout prob: {blowout_analysis['severe_blowout_prob']:.1%}"
            )
            logger.info(
                f"  Pace dampening factor: {blowout_analysis['pace_dampening']:.1%}\n"
            )

            # ========================================
            # DICTAMEN 8: MATCHUPS ANALYSIS V2 (10%)
            # ========================================
            logger.info("🧮 DICTAMEN 8/12: TACTICAL MATCHUPS V2 (10%)")
            matchup_analysis = self._analyze_matchups_v2(
                home_team, away_team
            )

            home_matchup_adj = matchup_analysis["home_advantage"]
            away_matchup_adj = matchup_analysis["away_advantage"]

            logger.info(f"  Home matchup edge: {home_matchup_adj:+.2f} pts")
            logger.info(f"  Away matchup edge: {away_matchup_adj:+.2f} pts")
            logger.info(f"  Key factors: {len(matchup_analysis['factors_identified'])}")
            logger.info(
                f"  Matchup confidence: {matchup_analysis['confidence']:.1%}\n"
            )

            # ========================================
            # DICTAMEN 9: HFA ENGINE V2 (4%) - ya integrado
            # ========================================
            logger.info("🏠 DICTAMEN 9/12: HOME COURT ADVANTAGE V2 (4%)")
            logger.info(f"  Base HFA: {hfa_analysis['base_hfa']:.1f}")
            logger.info(f"  Adjusted HFA: {hfa_points:+.1f}")
            logger.info(f"  Crowd factor: {hfa_analysis['crowd_factor']:.2f}")
            logger.info(f"  HFA confidence: {hfa_analysis['confidence']:.1%}\n")

            # ========================================
            # DICTAMEN 10: RISK & VARIANCE ENGINE V2 (5%)
            # ========================================
            logger.info("🧯 DICTAMEN 10/12: RISK & VARIANCE ENGINE V2 (5%)")
            risk_analysis = self._analyze_risk_v2(
                sim_results,
                home_team,
                away_team,
                pace_analysis,
            )

            logger.info(f"  Game volatility: {risk_analysis['volatility']:.2f}")
            logger.info(f"  Score std dev: {risk_analysis['score_std']:.2f}")
            logger.info(f"  Confidence level: {risk_analysis['confidence']:.1%}")
            logger.info(f"  Base Kelly: {risk_analysis['kelly_fraction']:.3f}")
            logger.info(f"  Risk tier: {risk_analysis['risk_tier']}\n")

            # ========================================
            # DICTAMEN 11: STABILITY MODEL V2 (5%)
            # ========================================
            logger.info("🧩 DICTAMEN 11/12: CONSISTENCY & STABILITY V2 (5%)")
            stability_analysis = self._analyze_stability_v2(
                home_team, away_team, sim_results
            )

            logger.info(
                f"  Home consistency: {stability_analysis['home_consistency']:.1%}"
            )
            logger.info(
                f"  Away consistency: {stability_analysis['away_consistency']:.1%}"
            )
            logger.info(
                f"  Pick reliability: {stability_analysis['reliability']:.1%}"
            )
            logger.info(
                f"  Variance ratio: {stability_analysis['variance_ratio']:.2f}\n"
            )

            # ========================================
            # DICTAMEN 12: TREND-SLIDING MODEL V2 (5%)
            # ========================================
            logger.info("🧠 DICTAMEN 12/12: TREND-SLIDING ANALYSIS V2 (5%)")
            trend_analysis = self._analyze_trends_v2(
                home_team, away_team
            )

            home_trend_adj = trend_analysis["home_trend_adj"]
            away_trend_adj = trend_analysis["away_trend_adj"]

            logger.info(f"  Home trend: {trend_analysis['home_trend']} ({home_trend_adj:+.2f})")
            logger.info(f"  Away trend: {trend_analysis['away_trend']} ({away_trend_adj:+.2f})")
            logger.info(f"  Home momentum: {trend_analysis['home_momentum']:.2f}")
            logger.info(f"  Away momentum: {trend_analysis['away_momentum']:.2f}\n")

            # ========================================
            # FINAL ADJUSTMENTS & CALCULATIONS
            # ========================================
            logger.info("=" * 70)
            logger.info("FINAL CALCULATIONS V2")
            logger.info("=" * 70 + "\n")

            home_points_ultra = (
                home_points_final + home_matchup_adj + home_trend_adj
            )
            away_points_ultra = (
                away_points_final + away_matchup_adj + away_trend_adj
            )

            total_ultra = home_points_ultra + away_points_ultra
            spread_ultra = home_points_ultra - away_points_ultra

            # Probabilidades de victoria
            p_home_win = float(
                np.mean(sim_results["home_scores"] > sim_results["away_scores"])
            )
            p_away_win = 1.0 - p_home_win

            # Market analysis con EV real
            market_betting = self._analyze_betting_markets_v2(
                sim_results,
                total_ultra,
                spread_ultra,
                market_data,
            )

            # Composite EV
            composite_ev = self._calculate_composite_ev_v2(
                context_analysis,
                injury_analysis,
                pace_analysis,
                shooting_analysis,
                sharp_analysis,
                matchup_analysis,
                hfa_analysis,
                risk_analysis,
                stability_analysis,
                trend_analysis,
                blowout_analysis,
                market_betting,
            )

            # Mejores apuestas con EV real
            best_bets = self._identify_best_bets_v2(
                market_betting,
                composite_ev,
                risk_analysis,
                stability_analysis,
                home_team.get("name", "Home"),
                away_team.get("name", "Away"),
                market_data,
            )

            logger.info("✅ Analysis complete!\n")

            # ========================================
            # RESULT PACKAGE
            # ========================================
            result: Dict[str, Any] = {
                "status": "success",
                "game_info": {
                    "home_team": home_team.get("name", "Home"),
                    "away_team": away_team.get("name", "Away"),
                    "game_date": game_context.get(
                        "game_date", datetime.now().strftime("%Y-%m-%d")
                    ),
                },
                "predictions": {
                    "home_points": round(float(home_points_ultra), 1),
                    "away_points": round(float(away_points_ultra), 1),
                    "total": round(float(total_ultra), 1),
                    "spread": round(float(spread_ultra), 1),
                },
                "probabilities": {
                    "home_win": round(float(p_home_win), 4),
                    "away_win": round(float(p_away_win), 4),
                },
                "simulation_stats": {
                    "home_mean": round(float(np.mean(sim_results["home_scores"])), 2),
                    "away_mean": round(float(np.mean(sim_results["away_scores"])), 2),
                    "home_std": round(float(np.std(sim_results["home_scores"])), 2),
                    "away_std": round(float(np.std(sim_results["away_scores"])), 2),
                    "total_mean": round(float(np.mean(sim_results["totals"])), 2),
                    "total_std": round(float(np.std(sim_results["totals"])), 2),
                    "spread_mean": round(float(np.mean(sim_results["spreads"])), 2),
                    "spread_std": round(float(np.std(sim_results["spreads"])), 2),
                },
                "engines": {
                    "1_context": context_analysis,
                    "2_injuries": injury_analysis,
                    "3_pace": pace_analysis,
                    "4_monte_carlo": {
                        "n_simulations": n_simulations,
                        "correlation": sim_results["correlation_used"],
                        "home_avg": round(float(np.mean(sim_results["home_scores"])), 2),
                        "away_avg": round(float(np.mean(sim_results["away_scores"])), 2),
                    },
                    "5_shooting_luck": shooting_analysis,
                    "6_sharp_money": sharp_analysis,
                    "7_blowout": blowout_analysis,
                    "8_matchups": matchup_analysis,
                    "9_hfa": hfa_analysis,
                    "10_risk": risk_analysis,
                    "11_stability": stability_analysis,
                    "12_trends": trend_analysis,
                },
                "composite_ev": composite_ev,
                "market_analysis": market_betting,
                "best_bets": best_bets,
                "metadata": {
                    "model_version": "NBA G10+ Ultra Pro V2",
                    "n_simulations": n_simulations,
                    "timestamp": datetime.now().isoformat(),
                    "engine_weights": self.engine_weights,
                },
            }

            return result

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    # ========================================================
    # DICTAMEN 1: CONTEXT ENGINE V2
    # ========================================================
    def _analyze_context_v2(
        self,
        home_team: Dict[str, Any],
        away_team: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analiza contexto completo del partido con altitude adjustment.
        """
        home_adj = 0.0
        away_adj = 0.0
        factors_applied: List[str] = []

        # Fatigue analysis - Back to back
        if context.get("home_b2b", False):
            home_adj += CONTEXT_FACTORS["b2b"]
            factors_applied.append("Home B2B")

        if context.get("away_b2b", False):
            away_adj += CONTEXT_FACTORS["b2b"]
            factors_applied.append("Away B2B")

        # 3 in 4 days
        if context.get("home_3in4", False):
            home_adj += CONTEXT_FACTORS["3in4"]
            factors_applied.append("Home 3in4")

        if context.get("away_3in4", False):
            away_adj += CONTEXT_FACTORS["3in4"]
            factors_applied.append("Away 3in4")

        # 4 in 6 days
        if context.get("home_4in6", False):
            home_adj += CONTEXT_FACTORS["4in6"]
            factors_applied.append("Home 4in6")

        if context.get("away_4in6", False):
            away_adj += CONTEXT_FACTORS["4in6"]
            factors_applied.append("Away 4in6")

        # 5 in 7 days
        if context.get("home_5in7", False):
            home_adj += CONTEXT_FACTORS["5in7"]
            factors_applied.append("Home 5in7")

        if context.get("away_5in7", False):
            away_adj += CONTEXT_FACTORS["5in7"]
            factors_applied.append("Away 5in7")

        # Travel & timezone
        away_travel = context.get("away_travel_miles", 0)
        timezone_direction = context.get("timezone_direction", "none")

        if away_travel > 1500:
            if timezone_direction == "west_to_east":
                away_adj += CONTEXT_FACTORS["timezone_west_to_east"]
                factors_applied.append("Away West→East travel")
            elif timezone_direction == "east_to_west":
                away_adj += CONTEXT_FACTORS["timezone_east_to_west"]
                factors_applied.append("Away East→West travel")

        if away_travel > 2000:
            away_adj += CONTEXT_FACTORS["cross_country_flight"]
            factors_applied.append("Cross-country flight")

        # Rest advantage
        home_rest = context.get("home_days_rest", 1)
        away_rest = context.get("away_days_rest", 1)

        if home_rest >= 3 and away_rest <= 1:
            home_adj += CONTEXT_FACTORS["rest_advantage_3plus"]
            factors_applied.append("Home 3+ days rest advantage")
        elif home_rest >= 2 and away_rest <= 0:
            home_adj += CONTEXT_FACTORS["rest_advantage_2"]
            factors_applied.append("Home 2 days rest advantage")

        if away_rest >= 3 and home_rest <= 1:
            away_adj += CONTEXT_FACTORS["rest_advantage_3plus"]
            factors_applied.append("Away 3+ days rest advantage")
        elif away_rest >= 2 and home_rest <= 0:
            away_adj += CONTEXT_FACTORS["rest_advantage_2"]
            factors_applied.append("Away 2 days rest advantage")

        # Momentum - Win/Loss streaks
        home_streak = context.get("home_win_streak", 0)
        away_streak = context.get("away_win_streak", 0)

        if home_streak >= 3:
            streak_bonus = CONTEXT_FACTORS["win_streak_bonus"] * min(home_streak, 10)
            home_adj += streak_bonus
            factors_applied.append(f"Home {home_streak}W streak (+{streak_bonus:.1f})")

        if away_streak >= 3:
            streak_bonus = CONTEXT_FACTORS["win_streak_bonus"] * min(away_streak, 10)
            away_adj += streak_bonus
            factors_applied.append(f"Away {away_streak}W streak (+{streak_bonus:.1f})")

        home_loss_streak = context.get("home_loss_streak", 0)
        away_loss_streak = context.get("away_loss_streak", 0)

        if home_loss_streak >= 3:
            loss_penalty = CONTEXT_FACTORS["loss_streak_penalty"] * min(home_loss_streak, 10)
            home_adj += loss_penalty
            factors_applied.append(f"Home {home_loss_streak}L streak ({loss_penalty:.1f})")

        if away_loss_streak >= 3:
            loss_penalty = CONTEXT_FACTORS["loss_streak_penalty"] * min(away_loss_streak, 10)
            away_adj += loss_penalty
            factors_applied.append(f"Away {away_loss_streak}L streak ({loss_penalty:.1f})")

        # Motivation factors
        if context.get("rivalry_game", False):
            both_boost = CONTEXT_FACTORS["rivalry_boost"]
            home_adj += both_boost
            away_adj += both_boost
            factors_applied.append("Rivalry game")

        if context.get("home_desperation", False):
            home_adj += CONTEXT_FACTORS["desperation_boost"]
            factors_applied.append("Home playoff desperation")

        if context.get("away_desperation", False):
            away_adj += CONTEXT_FACTORS["desperation_boost"]
            factors_applied.append("Away playoff desperation")

        if context.get("elimination_game", False):
            home_adj += CONTEXT_FACTORS["elimination_game"]
            away_adj += CONTEXT_FACTORS["elimination_game"]
            factors_applied.append("Elimination game intensity")

        if context.get("national_tv", False):
            home_adj += CONTEXT_FACTORS["national_tv"]
            away_adj += CONTEXT_FACTORS["national_tv"]
            factors_applied.append("National TV spotlight")

        # ALTITUDE ADJUSTMENT (nuevo en V2)
        home_name = home_team.get("name", "")
        altitude_factor = 0.0

        if home_name in ALTITUDE_ARENAS:
            altitude_penalty = ALTITUDE_ARENAS[home_name]
            away_adj -= altitude_penalty
            altitude_factor = -altitude_penalty
            factors_applied.append(f"Altitude penalty for away team (-{altitude_penalty:.1f})")

        # Calculate confidence
        base_confidence = 0.60
        factor_bonus = len(factors_applied) * 0.03
        confidence = min(0.95, base_confidence + factor_bonus)

        return {
            "home_adjustment": round(float(home_adj), 2),
            "away_adjustment": round(float(away_adj), 2),
            "altitude_factor": round(float(altitude_factor), 2),
            "factors_applied": factors_applied,
            "confidence": round(float(confidence), 3),
            "weight": self.engine_weights["context"],
        }

    # ========================================================
    # DICTAMEN 2: INJURY & LINEUP IMPACT V2
    # ========================================================
    def _analyze_injuries_v2(
        self,
        home_team: Dict[str, Any],
        away_team: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analiza impacto de lesiones con efectos no lineales y química.
        """

        def calculate_team_impact(injuries: List[Dict], team_type: str) -> Tuple[float, float, List[Dict]]:
            """Calcula impacto con interacciones no lineales."""
            if not injuries:
                return 0.0, 0.0, []

            impacts: List[float] = []
            chemistry_impacts: List[float] = []
            injuries_detail: List[Dict] = []

            for inj in injuries:
                player_name = inj.get("player", "Unknown")
                level = inj.get("level", "rotation")

                if level in INJURY_LEVELS:
                    impact_data = INJURY_LEVELS[level]
                    base_impact = impact_data["offensive"]
                    chem_impact = impact_data.get("chemistry", 0)

                    impacts.append(abs(base_impact))
                    chemistry_impacts.append(chem_impact)

                    injuries_detail.append({
                        "team": team_type,
                        "player": player_name,
                        "level": level,
                        "base_impact": base_impact,
                        "chemistry_impact": chem_impact,
                    })

            if len(impacts) == 0:
                return 0.0, 0.0, injuries_detail

            # Ordenar de mayor a menor impacto
            impacts.sort(reverse=True)

            # Modelo no lineal:
            # Primer injured: impacto completo (100%)
            # Segundo: 80% (equipo ya ajustó rotación)
            # Tercero: 65%
            # Cuarto+: 50% cada uno
            multipliers = [1.0, 0.80, 0.65, 0.50, 0.50, 0.50, 0.50]

            total = 0.0
            for i, imp in enumerate(impacts):
                mult = multipliers[i] if i < len(multipliers) else 0.50
                total += imp * mult

            # Contar starters/stars out
            high_impact_out = sum(
                1 for inj in injuries
                if inj.get("level") in ["mvp", "superstar", "allstar", "star", "starter"]
            )

            # Multiplicador de caos si faltan 3+ jugadores importantes
            if high_impact_out >= 3:
                chaos_mult = 1.0 + (high_impact_out - 2) * 0.12
                total *= chaos_mult

            # Chemistry total
            chemistry_total = sum(chemistry_impacts)

            return -total, chemistry_total, injuries_detail

        home_adj, home_chem, home_detail = calculate_team_impact(
            context.get("home_injuries", []), "home"
        )
        away_adj, away_chem, away_detail = calculate_team_impact(
            context.get("away_injuries", []), "away"
        )

        total_injuries = len(home_detail) + len(away_detail)

        # Confidence basada en cantidad de lesiones
        if total_injuries == 0:
            confidence = 0.92
        elif total_injuries <= 2:
            confidence = 0.80
        elif total_injuries <= 4:
            confidence = 0.70
        else:
            confidence = 0.60

        return {
            "home_adjustment": round(float(home_adj), 2),
            "away_adjustment": round(float(away_adj), 2),
            "chemistry_impact": round(float(home_chem + away_chem), 2),
            "home_chemistry": round(float(home_chem), 2),
            "away_chemistry": round(float(away_chem), 2),
            "injuries": home_detail + away_detail,
            "total_injuries": total_injuries,
            "confidence": round(float(confidence), 3),
            "weight": self.engine_weights["injuries"],
        }

    # ========================================================
    # DICTAMEN 3: DYNAMIC PACE MODEL V2
    # ========================================================
    def _analyze_pace_v2(
        self,
        home_team: Dict[str, Any],
        away_team: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calcula pace dinámico con múltiples factores y volatilidad.
        """

        # Base pace
        home_pace_base = home_team.get("pace", NBA_AVG_PACE)
        away_pace_base = away_team.get("pace", NBA_AVG_PACE)

        # Rolling pace (últimos 5 y 10 juegos)
        home_pace_l5 = home_team.get("pace_last_5", home_pace_base)
        home_pace_l10 = home_team.get("pace_last_10", home_pace_base)
        away_pace_l5 = away_team.get("pace_last_5", away_pace_base)
        away_pace_l10 = away_team.get("pace_last_10", away_pace_base)

        # Pace vs similar opponents
        home_pace_vs = home_team.get("pace_vs_similar", home_pace_base)
        away_pace_vs = away_team.get("pace_vs_similar", away_pace_base)

        # Home/Away splits
        home_pace_home = home_team.get("pace_home", home_pace_base)
        away_pace_away = away_team.get("pace_away", away_pace_base)

        # Weighted average mejorado
        home_pace_weighted = (
            home_pace_base * 0.15
            + home_pace_l5 * 0.30
            + home_pace_l10 * 0.20
            + home_pace_vs * 0.15
            + home_pace_home * 0.20
        )

        away_pace_weighted = (
            away_pace_base * 0.15
            + away_pace_l5 * 0.30
            + away_pace_l10 * 0.20
            + away_pace_vs * 0.15
            + away_pace_away * 0.20
        )

        # Game pace = promedio ponderado hacia el más lento
        # (el equipo más lento controla más el pace)
        if home_pace_weighted < away_pace_weighted:
            game_pace = home_pace_weighted * 0.55 + away_pace_weighted * 0.45
        else:
            game_pace = home_pace_weighted * 0.45 + away_pace_weighted * 0.55

        # Ajustes por fatiga
        if context.get("home_b2b", False) or context.get("away_b2b", False):
            game_pace *= 0.98  # Equipos cansados juegan más lento

        # Pace por cuartos
        early_pace = game_pace * 1.02  # Q1-Q3 ligeramente más rápido
        fourth_quarter_pace = game_pace * 0.96  # Q4 más lento (half-court)

        # Volatilidad del pace
        pace_volatility = abs(home_pace_l5 - home_pace_base) + abs(away_pace_l5 - away_pace_base)
        pace_volatility /= 2.0

        confidence = max(0.65, min(0.88, 0.80 - pace_volatility * 0.02))

        return {
            "predicted_pace": round(float(game_pace), 2),
            "early_pace": round(float(early_pace), 2),
            "fourth_quarter_pace": round(float(fourth_quarter_pace), 2),
            "home_pace_weighted": round(float(home_pace_weighted), 2),
            "away_pace_weighted": round(float(away_pace_weighted), 2),
            "pace_volatility": round(float(pace_volatility), 2),
            "confidence": round(float(confidence), 3),
            "weight": self.engine_weights["pace"],
        }

    # ========================================================
    # DICTAMEN 4: MONTE CARLO SIMULATION V2
    # ========================================================
    def _run_monte_carlo_v2(
        self,
        home_expected: float,
        away_expected: float,
        home_var: float,
        away_var: float,
        pace_analysis: Dict[str, Any],
        n_sim: int,
    ) -> Dict[str, Any]:
        """
        Monte Carlo V2 con:
        - Correlación entre scores (pace compartido)
        - Blowout integrado en simulación
        - Floor realista
        """

        # Ajustar correlación basada en volatilidad del pace
        pace_volatility = pace_analysis.get("pace_volatility", 0)
        correlation = SCORE_CORRELATION + (pace_volatility * 0.01)
        correlation = min(0.35, max(0.10, correlation))

        # Matriz de covarianza
        cov_matrix = np.array([
            [home_var ** 2, correlation * home_var * away_var],
            [correlation * home_var * away_var, away_var ** 2]
        ])

        # Simulación multivariada
        try:
            scores = np.random.multivariate_normal(
                [home_expected, away_expected],
                cov_matrix,
                n_sim
            )
        except np.linalg.LinAlgError:
            # Fallback si la matriz no es semidefinida positiva
            home_scores = np.random.normal(home_expected, home_var, n_sim)
            away_scores = np.random.normal(away_expected, away_var, n_sim)
            scores = np.column_stack([home_scores, away_scores])

        home_scores = scores[:, 0]
        away_scores = scores[:, 1]

        # Blowout dampening DURANTE simulación
        spreads = home_scores - away_scores
        abs_spreads = np.abs(spreads)

        # Dampening gradual basado en spread
        # Spread > 18: empieza dampening
        # Spread > 25: dampening severo
        blowout_mask = abs_spreads > BLOWOUT_THRESHOLD
        severe_mask = abs_spreads > BLOWOUT_SEVERE_THRESHOLD

        # Calcular factor de dampening
        dampening = np.ones(n_sim)
        dampening[blowout_mask] = 1 - (abs_spreads[blowout_mask] - BLOWOUT_THRESHOLD) * 0.006
        dampening[severe_mask] = 1 - BLOWOUT_SEVERE_REDUCTION
        dampening = np.clip(dampening, 0.82, 1.0)

        # Aplicar al equipo perdedor
        home_losing = spreads < -BLOWOUT_THRESHOLD
        away_losing = spreads > BLOWOUT_THRESHOLD

        home_scores[home_losing] *= dampening[home_losing]
        away_scores[away_losing] *= dampening[away_losing]

        # Floor realista (ningún equipo anota menos de 85)
        home_scores = np.maximum(home_scores, 85.0)
        away_scores = np.maximum(away_scores, 85.0)

        # Ceiling realista (raro pasar de 145)
        home_scores = np.minimum(home_scores, 150.0)
        away_scores = np.minimum(away_scores, 150.0)

        totals = home_scores + away_scores
        spreads = home_scores - away_scores

        return {
            "home_scores": home_scores,
            "away_scores": away_scores,
            "totals": totals,
            "spreads": spreads,
            "correlation_used": round(float(correlation), 3),
        }

    # ========================================================
    # DICTAMEN 5: SHOOTING LUCK & EFFICIENCY V2
    # ========================================================
    def _analyze_shooting_luck_v2(
        self,
        home_team: Dict[str, Any],
        away_team: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Detecta shooting luck con regresión mejorada y tendencias.
        """

        home_efg_l5 = home_team.get("efg_last_5", NBA_AVG_eFG)
        home_efg_season = home_team.get("efg_season", NBA_AVG_eFG)

        away_efg_l5 = away_team.get("efg_last_5", NBA_AVG_eFG)
        away_efg_season = away_team.get("efg_season", NBA_AVG_eFG)

        # Diferencia entre recent y season
        home_shooting_diff = home_efg_l5 - home_efg_season
        away_shooting_diff = away_efg_l5 - away_efg_season

        # Regresión más sofisticada
        # Factor de regresión aumenta si la diferencia es extrema
        home_regression_factor = REGRESSION_FACTOR
        away_regression_factor = REGRESSION_FACTOR

        if abs(home_shooting_diff) > 0.04:
            home_regression_factor *= 1.2
        if abs(away_shooting_diff) > 0.04:
            away_regression_factor *= 1.2

        # Calcular ajuste (convertido a puntos)
        # ~200 field goal attempts por juego, cada 1% eFG ≈ 2 puntos
        home_regression = -home_shooting_diff * home_regression_factor * 180.0
        away_regression = -away_shooting_diff * away_regression_factor * 180.0

        # Determinar tendencia
        def get_trend(diff: float) -> str:
            if diff > 0.025:
                return "HOT 🔥"
            elif diff > 0.01:
                return "WARM"
            elif diff < -0.025:
                return "COLD ❄️"
            elif diff < -0.01:
                return "COOL"
            else:
                return "NEUTRAL"

        home_trend = get_trend(home_shooting_diff)
        away_trend = get_trend(away_shooting_diff)

        # Confidence basada en tamaño de muestra y extremidad
        base_confidence = 0.72
        if abs(home_shooting_diff) > 0.03 or abs(away_shooting_diff) > 0.03:
            base_confidence -= 0.05
        confidence = max(0.55, min(0.85, base_confidence))

        return {
            "home_regression_adj": round(float(home_regression), 2),
            "away_regression_adj": round(float(away_regression), 2),
            "home_efg_diff": round(float(home_shooting_diff), 4),
            "away_efg_diff": round(float(away_shooting_diff), 4),
            "home_trend": home_trend,
            "away_trend": away_trend,
            "home_hot_shooting": bool(home_shooting_diff > 0.02),
            "away_hot_shooting": bool(away_shooting_diff > 0.02),
            "home_cold_shooting": bool(home_shooting_diff < -0.02),
            "away_cold_shooting": bool(away_shooting_diff < -0.02),
            "confidence": round(float(confidence), 3),
            "weight": self.engine_weights["shooting_luck"],
        }

    # ========================================================
    # DICTAMEN 6: SHARP MONEY MODEL V2
    # ========================================================
    def _analyze_sharp_money_v2(
        self,
        market_data: Dict[str, Any],
        predicted_spread: float,
        predicted_total: float,
    ) -> Dict[str, Any]:
        """
        Analiza sharp money, RLM, steam moves mejorado.
        """

        opening_spread = market_data.get("opening_spread", predicted_spread)
        current_spread = market_data.get("current_spread", predicted_spread)

        opening_total = market_data.get("opening_total", predicted_total)
        current_total = market_data.get("current_total", predicted_total)

        tickets_pct_home = market_data.get("tickets_pct_home", 50.0)
        money_pct_home = market_data.get("money_pct_home", 50.0)

        tickets_pct_over = market_data.get("tickets_pct_over", 50.0)
        money_pct_over = market_data.get("money_pct_over", 50.0)

        steam_moves = market_data.get("steam_moves", [])
        sharp_reports = market_data.get("sharp_reports", [])

        # Line movement
        spread_move = current_spread - opening_spread
        total_move = current_total - opening_total

        # RLM Detection mejorado
        rlm_spread = False
        rlm_total = False

        # RLM en spread: línea se mueve contra el público
        if tickets_pct_home < 45 and spread_move < -RLM_THRESHOLD:
            rlm_spread = True  # Público en away, línea baja (more home)
        if tickets_pct_home > 55 and spread_move > RLM_THRESHOLD:
            rlm_spread = True  # Público en home, línea sube (less home)

        # RLM en total
        if tickets_pct_over < 45 and total_move > RLM_THRESHOLD:
            rlm_total = True
        if tickets_pct_over > 55 and total_move < -RLM_THRESHOLD:
            rlm_total = True

        rlm_detected = rlm_spread or rlm_total

        # Sharp side detection (diferencia money vs tickets)
        sharp_side: Optional[str] = None
        sharp_total_side: Optional[str] = None

        money_ticket_diff_spread = money_pct_home - tickets_pct_home
        money_ticket_diff_total = money_pct_over - tickets_pct_over

        if money_ticket_diff_spread > SHARP_MONEY_THRESHOLD:
            sharp_side = "home"
        elif money_ticket_diff_spread < -SHARP_MONEY_THRESHOLD:
            sharp_side = "away"

        if money_ticket_diff_total > SHARP_MONEY_THRESHOLD:
            sharp_total_side = "over"
        elif money_ticket_diff_total < -SHARP_MONEY_THRESHOLD:
            sharp_total_side = "under"

        # Steam moves count
        steam_move_count = len(steam_moves)

        # Edge detected
        edge_detected = (
            rlm_detected or
            sharp_side is not None or
            sharp_total_side is not None or
            steam_move_count >= 2
        )

        # Confidence
        confidence = 0.50
        if rlm_detected:
            confidence += 0.12
        if sharp_side is not None:
            confidence += 0.10
        if steam_move_count >= 2:
            confidence += 0.08
        if len(sharp_reports) > 0:
            confidence += 0.05

        confidence = min(0.88, confidence)

        return {
            "rlm_detected": bool(rlm_detected),
            "rlm_spread": bool(rlm_spread),
            "rlm_total": bool(rlm_total),
            "sharp_side": sharp_side,
            "sharp_total_side": sharp_total_side,
            "edge_detected": bool(edge_detected),
            "tickets_pct_home": round(float(tickets_pct_home), 1),
            "money_pct_home": round(float(money_pct_home), 1),
            "tickets_pct_over": round(float(tickets_pct_over), 1),
            "money_pct_over": round(float(money_pct_over), 1),
            "line_movement_spread": round(float(spread_move), 2),
            "line_movement_total": round(float(total_move), 2),
            "steam_move_count": steam_move_count,
            "steam_moves": steam_moves,
            "sharp_reports": sharp_reports,
            "confidence": round(float(confidence), 3),
            "weight": self.engine_weights["sharp_money"],
        }

    # ========================================================
    # DICTAMEN 7: BLOWOUT & DAMPENER V2
    # ========================================================
    def _analyze_blowout_risk_v2(
        self,
        home_expected: float,
        away_expected: float,
        sim_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evalúa riesgo de blowout con múltiples niveles.
        """

        spreads = sim_results["spreads"]
        abs_spreads = np.abs(spreads)

        blowout_prob = float(np.mean(abs_spreads > BLOWOUT_THRESHOLD))
        severe_blowout_prob = float(np.mean(abs_spreads > BLOWOUT_SEVERE_THRESHOLD))

        # Determinar dampening
        if blowout_prob > 0.30:
            pace_dampening = BLOWOUT_PACE_REDUCTION
        elif blowout_prob > 0.20:
            pace_dampening = BLOWOUT_PACE_REDUCTION * 0.6
        else:
            pace_dampening = 0.0

        # Dirección del blowout probable
        home_blowout_prob = float(np.mean(spreads > BLOWOUT_THRESHOLD))
        away_blowout_prob = float(np.mean(spreads < -BLOWOUT_THRESHOLD))

        blowout_direction = None
        if home_blowout_prob > away_blowout_prob + 0.05:
            blowout_direction = "home"
        elif away_blowout_prob > home_blowout_prob + 0.05:
            blowout_direction = "away"

        return {
            "blowout_prob": round(float(blowout_prob), 4),
            "severe_blowout_prob": round(float(severe_blowout_prob), 4),
            "home_blowout_prob": round(float(home_blowout_prob), 4),
            "away_blowout_prob": round(float(away_blowout_prob), 4),
            "blowout_direction": blowout_direction,
            "pace_dampening": round(float(pace_dampening), 3),
            "high_blowout_risk": bool(blowout_prob > 0.25),
            "weight": self.engine_weights["blowout"],
        }

    # ========================================================
    # DICTAMEN 8: MATCHUPS ANALYSIS V2
    # ========================================================
    def _analyze_matchups_v2(
        self,
        home_team: Dict[str, Any],
        away_team: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Análisis táctico de matchups expandido con 10+ factores.
        """
        home_adv = 0.0
        away_adv = 0.0
        factors_identified: List[str] = []

        # 1. Paint scoring vs interior defense
        home_paint = home_team.get("points_in_paint", NBA_AVG_PAINT_PTS)
        away_paint_def = away_team.get("opp_points_in_paint", NBA_AVG_PAINT_PTS)
        away_paint = away_team.get("points_in_paint", NBA_AVG_PAINT_PTS)
        home_paint_def = home_team.get("opp_points_in_paint", NBA_AVG_PAINT_PTS)

        if home_paint > away_paint_def + 4:
            home_adv += 1.8
            factors_identified.append("Home paint advantage")
        if away_paint > home_paint_def + 4:
            away_adv += 1.8
            factors_identified.append("Away paint advantage")

        # 2. Fastbreak points
        home_fb = home_team.get("fastbreak_points", NBA_AVG_FASTBREAK)
        away_fb_def = away_team.get("opp_fastbreak_points", NBA_AVG_FASTBREAK)
        away_fb = away_team.get("fastbreak_points", NBA_AVG_FASTBREAK)
        home_fb_def = home_team.get("opp_fastbreak_points", NBA_AVG_FASTBREAK)

        if home_fb > away_fb_def + 3:
            home_adv += 1.2
            factors_identified.append("Home transition edge")
        if away_fb > home_fb_def + 3:
            away_adv += 1.2
            factors_identified.append("Away transition edge")

        # 3. Turnover battle
        home_tov = home_team.get("turnovers", NBA_AVG_TURNOVERS)
        away_steals = away_team.get("steals", 7.5)
        away_tov = away_team.get("turnovers", NBA_AVG_TURNOVERS)
        home_steals = home_team.get("steals", 7.5)

        tov_risk_home = (away_steals - 7.5) - (home_tov - NBA_AVG_TURNOVERS) * 0.3
        tov_risk_away = (home_steals - 7.5) - (away_tov - NBA_AVG_TURNOVERS) * 0.3

        if tov_risk_home > 1.0:
            away_adv += tov_risk_home * 0.4
            factors_identified.append("Away forcing turnovers")
        if tov_risk_away > 1.0:
            home_adv += tov_risk_away * 0.4
            factors_identified.append("Home forcing turnovers")

        # 4. Free throw rate (agresividad)
        home_ftr = home_team.get("free_throw_rate", NBA_AVG_FT_RATE)
        away_ftr = away_team.get("free_throw_rate", NBA_AVG_FT_RATE)

        if home_ftr > away_ftr + 0.05:
            home_adv += 1.0
            factors_identified.append("Home FT advantage")
        if away_ftr > home_ftr + 0.05:
            away_adv += 1.0
            factors_identified.append("Away FT advantage")

        # 5. Offensive rebounding (second chance)
        home_oreb = home_team.get("offensive_rebounds", 10.0)
        away_dreb = away_team.get("defensive_rebounds", 34.0)
        away_oreb = away_team.get("offensive_rebounds", 10.0)
        home_dreb = home_team.get("defensive_rebounds", 34.0)

        if home_oreb > 11.5 and away_dreb < 33.0:
            home_adv += 1.0
            factors_identified.append("Home second chance edge")
        if away_oreb > 11.5 and home_dreb < 33.0:
            away_adv += 1.0
            factors_identified.append("Away second chance edge")

        # 6. Bench scoring
        home_bench = home_team.get("bench_points", 35.0)
        away_bench = away_team.get("bench_points", 35.0)

        if home_bench > away_bench + 6:
            home_adv += 0.8
            factors_identified.append("Home bench advantage")
        if away_bench > home_bench + 6:
            away_adv += 0.8
            factors_identified.append("Away bench advantage")

        # 7. Three point volume & efficiency
        home_3pt_off = home_team.get("three_pt_offense", 0.36)
        home_3pt_vol = home_team.get("three_pt_attempts", 35.0)
        away_3pt_def = away_team.get("three_pt_defense", 0.36)

        away_3pt_off = away_team.get("three_pt_offense", 0.36)
        away_3pt_vol = away_team.get("three_pt_attempts", 35.0)
        home_3pt_def = home_team.get("three_pt_defense", 0.36)

        if home_3pt_off > away_3pt_def + 0.03 and home_3pt_vol > 36:
            home_adv += 1.5
            factors_identified.append("Home 3PT advantage")
        if away_3pt_off > home_3pt_def + 0.03 and away_3pt_vol > 36:
            away_adv += 1.5
            factors_identified.append("Away 3PT advantage")

        # 8. Assist ratio (ball movement)
        home_ast = home_team.get("assists", 25.0)
        away_ast = away_team.get("assists", 25.0)

        if home_ast > away_ast + 3:
            home_adv += 0.6
            factors_identified.append("Home ball movement edge")
        if away_ast > home_ast + 3:
            away_adv += 0.6
            factors_identified.append("Away ball movement edge")

        # 9. Clutch performance
        home_clutch = home_team.get("clutch_net_rating", 0.0)
        away_clutch = away_team.get("clutch_net_rating", 0.0)

        if home_clutch > away_clutch + 5:
            home_adv += 0.5
            factors_identified.append("Home clutch advantage")
        if away_clutch > home_clutch + 5:
            away_adv += 0.5
            factors_identified.append("Away clutch advantage")

        # 10. Pick and roll
        home_pnr_off = home_team.get("pnr_offense_rating", 100.0)
        away_pnr_def = away_team.get("pnr_defense_rating", 100.0)
        away_pnr_off = away_team.get("pnr_offense_rating", 100.0)
        home_pnr_def = home_team.get("pnr_defense_rating", 100.0)

        if home_pnr_off > away_pnr_def + 5:
            home_adv += 1.2
            factors_identified.append("Home PnR advantage")
        if away_pnr_off > home_pnr_def + 5:
            away_adv += 1.2
            factors_identified.append("Away PnR advantage")

        # 11. Rebounding rate general
        home_reb = home_team.get("rebounding_rate", 50.0)
        away_reb = away_team.get("rebounding_rate", 50.0)
        reb_diff = home_reb - away_reb
        home_adv += reb_diff * 0.06

        if abs(reb_diff) > 2:
            factors_identified.append(f"{'Home' if reb_diff > 0 else 'Away'} rebounding edge")

        # Confidence basada en factores encontrados
        confidence = min(0.88, 0.58 + len(factors_identified) * 0.04)

        return {
            "home_advantage": round(float(home_adv), 2),
            "away_advantage": round(float(away_adv), 2),
            "net_advantage": round(float(home_adv - away_adv), 2),
            "factors_identified": factors_identified,
            "factors_count": len(factors_identified),
            "confidence": round(float(confidence), 3),
            "weight": self.engine_weights["matchups"],
        }

    # ========================================================
    # DICTAMEN 9: HFA ENGINE V2
    # ========================================================
    def _analyze_hfa_v2(
        self,
        home_team: Dict[str, Any],
        away_team: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Home court advantage dinámico V2.
        """

        base_hfa = 2.8  # Promedio NBA moderno

        home_hfa_multiplier = home_team.get("hfa_multiplier", 1.0)

        # Modificadores
        crowd_factor = 1.0

        # Si away team viene descansado, HFA se reduce
        if context.get("away_days_rest", 1) >= 3:
            home_hfa_multiplier *= 0.85
            crowd_factor *= 0.95

        # Si away viene de B2B, HFA aumenta
        if context.get("away_b2b", False):
            home_hfa_multiplier *= 1.15
            crowd_factor *= 1.05

        # Si home viene de B2B, HFA se reduce
        if context.get("home_b2b", False):
            home_hfa_multiplier *= 0.90

        # Playoffs aumenta HFA
        if context.get("playoff_game", False):
            home_hfa_multiplier *= 1.25
            crowd_factor *= 1.15

        # Elimination game
        if context.get("elimination_game", False):
            home_hfa_multiplier *= 1.15

        # National TV puede reducir ligeramente (equipos visitantes más motivados)
        if context.get("national_tv", False):
            home_hfa_multiplier *= 0.95

        hfa_points = base_hfa * home_hfa_multiplier
        confidence = 0.82

        return {
            "base_hfa": round(float(base_hfa), 2),
            "hfa_points": round(float(hfa_points), 2),
            "multiplier": round(float(home_hfa_multiplier), 3),
            "crowd_factor": round(float(crowd_factor), 3),
            "confidence": round(float(confidence), 3),
            "weight": self.engine_weights["hfa"],
        }

    # ========================================================
    # DICTAMEN 10: RISK & VARIANCE V2
    # ========================================================
    def _analyze_risk_v2(
        self,
        sim_results: Dict[str, Any],
        home_team: Dict[str, Any],
        away_team: Dict[str, Any],
        pace_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calcula volatilidad y riesgo mejorado.
        """

        totals = sim_results["totals"]
        spreads = sim_results["spreads"]

        total_std = float(np.std(totals))
        spread_std = float(np.std(spreads))
        score_std = (total_std + spread_std) / 2

        # Base volatility
        volatility = total_std

        # Ajustar por pace
        pace = pace_analysis["predicted_pace"]
        pace_volatility = pace_analysis["pace_volatility"]

        if pace > 102:
            volatility *= 1.08
        elif pace < 96:
            volatility *= 0.92

        # Ajustar por pace volatility
        volatility *= (1 + pace_volatility * 0.02)

        # Risk tier
        if volatility < 10:
            risk_tier = "LOW"
        elif volatility < 13:
            risk_tier = "MEDIUM"
        elif volatility < 16:
            risk_tier = "HIGH"
        else:
            risk_tier = "VERY HIGH"

        # Confidence inversamente proporcional a volatilidad
        confidence = max(0.45, min(0.92, 1.0 - (volatility / 25.0)))

        # Kelly fraction basada en riesgo
        if risk_tier == "LOW":
            kelly_fraction = 0.04
        elif risk_tier == "MEDIUM":
            kelly_fraction = 0.03
        elif risk_tier == "HIGH":
            kelly_fraction = 0.02
        else:
            kelly_fraction = 0.01

        return {
            "volatility": round(float(volatility), 2),
            "score_std": round(float(score_std), 2),
            "total_std": round(float(total_std), 2),
            "spread_std": round(float(spread_std), 2),
            "risk_tier": risk_tier,
            "confidence": round(float(confidence), 3),
            "kelly_fraction": round(float(kelly_fraction), 4),
            "weight": self.engine_weights["risk"],
        }

    # ========================================================
    # DICTAMEN 11: STABILITY MODEL V2
    # ========================================================
    def _analyze_stability_v2(
        self,
        home_team: Dict[str, Any],
        away_team: Dict[str, Any],
        sim_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evalúa consistencia de equipos V2.
        """

        home_consistency = home_team.get("consistency_score", 70.0)
        away_consistency = away_team.get("consistency_score", 70.0)

        avg_consistency = (home_consistency + away_consistency) / 2.0

        # Variance ratio de simulaciones
        home_scores = sim_results["home_scores"]
        away_scores = sim_results["away_scores"]

        home_var = float(np.var(home_scores))
        away_var = float(np.var(away_scores))

        variance_ratio = max(home_var, away_var) / (min(home_var, away_var) + 0.01)

        # Si un equipo es mucho más volátil que otro, la fiabilidad baja
        consistency_penalty = 0.0
        if variance_ratio > 1.5:
            consistency_penalty = (variance_ratio - 1.5) * 5

        adjusted_consistency = avg_consistency - consistency_penalty
        reliability = max(0.40, min(0.95, adjusted_consistency / 100.0))

        return {
            "home_consistency": round(float(home_consistency), 1),
            "away_consistency": round(float(away_consistency), 1),
            "avg_consistency": round(float(avg_consistency), 1),
            "variance_ratio": round(float(variance_ratio), 2),
            "reliability": round(float(reliability), 3),
            "weight": self.engine_weights["stability"],
        }

    # ========================================================
    # DICTAMEN 12: TREND-SLIDING V2
    # ========================================================
    def _analyze_trends_v2(
        self,
        home_team: Dict[str, Any],
        away_team: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analiza tendencias recientes con sliding window y momentum.
        """

        # Home trends
        home_l3 = home_team.get("points_last_3", 0.0)
        home_l5 = home_team.get("points_last_5", 0.0)
        home_l10 = home_team.get("points_last_10", 0.0)
        home_season = home_team.get("points_per_game", NBA_AVG_POINTS)

        # Away trends
        away_l3 = away_team.get("points_last_3", 0.0)
        away_l5 = away_team.get("points_last_5", 0.0)
        away_l10 = away_team.get("points_last_10", 0.0)
        away_season = away_team.get("points_per_game", NBA_AVG_POINTS)

        # Weighted trend (más peso a lo reciente)
        home_trend_weighted = 0.0
        if home_l3 > 0:
            home_trend_weighted = (
                home_l3 * 0.50 + home_l5 * 0.30 + home_l10 * 0.20
            )
        else:
            home_trend_weighted = home_season

        away_trend_weighted = 0.0
        if away_l3 > 0:
            away_trend_weighted = (
                away_l3 * 0.50 + away_l5 * 0.30 + away_l10 * 0.20
            )
        else:
            away_trend_weighted = away_season

        # Trend adjustment
        home_trend_adj = (home_trend_weighted - home_season) * 0.18
        away_trend_adj = (away_trend_weighted - away_season) * 0.18

        # Cap adjustments
        home_trend_adj = max(-3.0, min(3.0, home_trend_adj))
        away_trend_adj = max(-3.0, min(3.0, away_trend_adj))

        # Momentum calculation (aceleración del trend)
        home_momentum = 0.0
        if home_l3 > 0 and home_l5 > 0:
            home_momentum = (home_l3 - home_l5) / home_l5 * 100

        away_momentum = 0.0
        if away_l3 > 0 and away_l5 > 0:
            away_momentum = (away_l3 - away_l5) / away_l5 * 100

        # Trend direction
        def get_trend_direction(adj: float, momentum: float) -> str:
            if adj > 1.0 and momentum > 1.0:
                return "STRONG UP ⬆️"
            elif adj > 0.5:
                return "UP ↗️"
            elif adj < -1.0 and momentum < -1.0:
                return "STRONG DOWN ⬇️"
            elif adj < -0.5:
                return "DOWN ↘️"
            else:
                return "STABLE ➡️"

        home_trend = get_trend_direction(home_trend_adj, home_momentum)
        away_trend = get_trend_direction(away_trend_adj, away_momentum)

        return {
            "home_trend": home_trend,
            "away_trend": away_trend,
            "home_trend_adj": round(float(home_trend_adj), 2),
            "away_trend_adj": round(float(away_trend_adj), 2),
            "home_momentum": round(float(home_momentum), 2),
            "away_momentum": round(float(away_momentum), 2),
            "home_trend_weighted": round(float(home_trend_weighted), 1),
            "away_trend_weighted": round(float(away_trend_weighted), 1),
            "weight": self.engine_weights["trends"],
        }

    # ========================================================
    # BETTING MARKETS ANALYSIS V2
    # ========================================================
    def _analyze_betting_markets_v2(
        self,
        sim_results: Dict[str, Any],
        predicted_total: float,
        predicted_spread: float,
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analiza mercados de apuestas con EV real.
        """

        totals = sim_results["totals"]
        spreads = sim_results["spreads"]

        # Lines comunes
        common_totals = [210, 212.5, 215, 217.5, 220, 222.5, 225, 227.5, 230, 232.5, 235, 237.5, 240]
        common_spreads = [-12, -10, -8.5, -7, -5.5, -4, -3, -2, -1, 1, 2, 3, 4, 5.5, 7, 8.5, 10, 12]

        # Market lines
        market_total = market_data.get("current_total", predicted_total)
        market_spread = market_data.get("current_spread", predicted_spread)

        # Ensure market lines are in our analysis
        if market_total not in common_totals:
            common_totals.append(market_total)
            common_totals.sort()

        if market_spread not in common_spreads:
            common_spreads.append(market_spread)
            common_spreads.sort()

        total_lines: Dict[str, Any] = {}
        for line in common_totals:
            p_over = float(np.mean(totals > line))
            edge = abs(p_over - 0.5)

            total_lines[f"total_{line}"] = {
                "line": float(line),
                "p_over": round(p_over, 4),
                "p_under": round(1.0 - p_over, 4),
                "edge": round(edge, 4),
                "is_market_line": line == market_total,
            }

        spread_lines: Dict[str, Any] = {}
        for line in common_spreads:
            p_home_cover = float(np.mean(spreads > line))
            edge = abs(p_home_cover - 0.5)

            spread_lines[f"spread_{line:+.1f}"] = {
                "line": float(line),
                "p_home_cover": round(p_home_cover, 4),
                "p_away_cover": round(1.0 - p_home_cover, 4),
                "edge": round(edge, 4),
                "is_market_line": line == market_spread,
            }

        # Moneyline probabilities
        p_home_win = float(np.mean(sim_results["home_scores"] > sim_results["away_scores"]))
        p_away_win = 1.0 - p_home_win

        return {
            "total_lines": total_lines,
            "spread_lines": spread_lines,
            "market_total": market_total,
            "market_spread": market_spread,
            "predicted_total": round(predicted_total, 1),
            "predicted_spread": round(predicted_spread, 1),
            "moneyline": {
                "p_home_win": round(p_home_win, 4),
                "p_away_win": round(p_away_win, 4),
            }
        }

    # ========================================================
    # COMPOSITE EV CALCULATION V2
    # ========================================================
    def _calculate_composite_ev_v2(
        self,
        context: Dict[str, Any],
        injuries: Dict[str, Any],
        pace: Dict[str, Any],
        shooting: Dict[str, Any],
        sharp: Dict[str, Any],
        matchups: Dict[str, Any],
        hfa: Dict[str, Any],
        risk: Dict[str, Any],
        stability: Dict[str, Any],
        trends: Dict[str, Any],
        blowout: Dict[str, Any],
        market: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calcula EV compuesto de todos los engines V2.
        """

        # Weighted confidence
        weighted_confidence = (
            context["confidence"] * self.engine_weights["context"]
            + injuries["confidence"] * self.engine_weights["injuries"]
            + pace["confidence"] * self.engine_weights["pace"]
            + shooting["confidence"] * self.engine_weights["shooting_luck"]
            + sharp["confidence"] * self.engine_weights["sharp_money"]
            + matchups["confidence"] * self.engine_weights["matchups"]
            + hfa["confidence"] * self.engine_weights["hfa"]
            + risk["confidence"] * self.engine_weights["risk"]
        )

        # Reliability factor
        reliability = stability["reliability"]

        # Sharp money bonus
        sharp_bonus = 0.0
        if sharp["edge_detected"]:
            sharp_bonus = 0.05
        if sharp["rlm_detected"]:
            sharp_bonus += 0.03

        # Blowout penalty (reduce confidence si alto riesgo de blowout)
        blowout_penalty = 0.0
        if blowout["blowout_prob"] > 0.30:
            blowout_penalty = 0.05

        # Final EV multiplier
        ev_multiplier = (
            weighted_confidence *
            reliability *
            (1 + sharp_bonus) *
            (1 - blowout_penalty)
        )

        # Composite score (0-100)
        composite_score = weighted_confidence * 100.0

        # Betting grade
        if composite_score >= 80 and ev_multiplier > 0.65:
            grade = "A+"
        elif composite_score >= 75 and ev_multiplier > 0.60:
            grade = "A"
        elif composite_score >= 70 and ev_multiplier > 0.55:
            grade = "B+"
        elif composite_score >= 65 and ev_multiplier > 0.50:
            grade = "B"
        elif composite_score >= 60:
            grade = "C+"
        else:
            grade = "C"

        return {
            "weighted_confidence": round(float(weighted_confidence), 4),
            "reliability": round(float(reliability), 4),
            "sharp_bonus": round(float(sharp_bonus), 4),
            "blowout_penalty": round(float(blowout_penalty), 4),
            "ev_multiplier": round(float(ev_multiplier), 4),
            "composite_score": round(float(composite_score), 2),
            "grade": grade,
        }

    # ========================================================
    # TRUE EV CALCULATION (NEW IN V2)
    # ========================================================
    def _calculate_true_ev(
        self,
        probability: float,
        american_odds: int,
    ) -> Dict[str, float]:
        """
        Calcula EV real contra odds del mercado.
        """

        # Convertir odds americanos a decimales
        if american_odds > 0:
            decimal_odds = 1 + (american_odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(american_odds))

        # Probabilidad implícita del mercado (con vig)
        implied_prob = 1 / decimal_odds

        # Edge real
        edge = probability - implied_prob

        # EV por unidad apostada
        ev_per_unit = (probability * (decimal_odds - 1)) - (1 - probability)

        # Kelly Criterion
        if edge > 0 and decimal_odds > 1:
            kelly_full = edge / (decimal_odds - 1)
        else:
            kelly_full = 0.0

        kelly_quarter = kelly_full * 0.25
        kelly_half = kelly_full * 0.50

        # Criterio de apuesta worthy
        bet_worthy = edge > 0.03 and ev_per_unit > 0.02

        return {
            "probability": round(float(probability), 4),
            "implied_prob": round(float(implied_prob), 4),
            "decimal_odds": round(float(decimal_odds), 3),
            "edge": round(float(edge), 4),
            "ev_per_unit": round(float(ev_per_unit), 4),
            "kelly_full": round(float(kelly_full), 4),
            "kelly_quarter": round(float(kelly_quarter), 4),
            "kelly_half": round(float(kelly_half), 4),
            "bet_worthy": bool(bet_worthy),
        }

    # ========================================================
    # BEST BETS IDENTIFICATION V2
    # ========================================================
    def _identify_best_bets_v2(
        self,
        market: Dict[str, Any],
        composite_ev: Dict[str, Any],
        risk: Dict[str, Any],
        stability: Dict[str, Any],
        home_name: str,
        away_name: str,
        market_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Identifica mejores apuestas con EV real V2.
        """
        bets: List[Dict[str, Any]] = []

        min_edge = 0.04
        min_confidence = 0.62

        # Standard odds assumption (-110 = 52.4% implied)
        standard_american_odds = -110

        # Totals
        for key, data in market["total_lines"].items():
            edge = data["edge"]
            p_over = data["p_over"]
            p_under = data["p_under"]

            if edge > min_edge and composite_ev["weighted_confidence"] > min_confidence:
                # Calculate true EV
                if p_over > 0.54:
                    ev_calc = self._calculate_true_ev(p_over, standard_american_odds)
                    if ev_calc["bet_worthy"]:
                        rating = self._get_bet_rating(edge, ev_calc["ev_per_unit"], composite_ev["weighted_confidence"])
                        bets.append({
                            "market": "Total",
                            "side": f"OVER {data['line']}",
                            "probability": p_over,
                            "implied_prob": ev_calc["implied_prob"],
                            "edge": edge,
                            "ev_per_unit": ev_calc["ev_per_unit"],
                            "kelly_full": ev_calc["kelly_full"],
                            "kelly_quarter": ev_calc["kelly_quarter"],
                            "confidence": composite_ev["weighted_confidence"],
                            "rating": rating,
                            "reliability": stability["reliability"],
                            "risk_tier": risk["risk_tier"],
                            "bet_worthy": ev_calc["bet_worthy"],
                            "is_market_line": data.get("is_market_line", False),
                        })

                elif p_under > 0.54:
                    ev_calc = self._calculate_true_ev(p_under, standard_american_odds)
                    if ev_calc["bet_worthy"]:
                        rating = self._get_bet_rating(edge, ev_calc["ev_per_unit"], composite_ev["weighted_confidence"])
                        bets.append({
                            "market": "Total",
                            "side": f"UNDER {data['line']}",
                            "probability": p_under,
                            "implied_prob": ev_calc["implied_prob"],
                            "edge": edge,
                            "ev_per_unit": ev_calc["ev_per_unit"],
                            "kelly_full": ev_calc["kelly_full"],
                            "kelly_quarter": ev_calc["kelly_quarter"],
                            "confidence": composite_ev["weighted_confidence"],
                            "rating": rating,
                            "reliability": stability["reliability"],
                            "risk_tier": risk["risk_tier"],
                            "bet_worthy": ev_calc["bet_worthy"],
                            "is_market_line": data.get("is_market_line", False),
                        })

        # Spreads
        for key, data in market["spread_lines"].items():
            edge = data["edge"]
            p_home = data["p_home_cover"]
            p_away = data["p_away_cover"]

            if edge > min_edge and composite_ev["weighted_confidence"] > min_confidence:
                if p_home > 0.54:
                    ev_calc = self._calculate_true_ev(p_home, standard_american_odds)
                    if ev_calc["bet_worthy"]:
                        rating = self._get_bet_rating(edge, ev_calc["ev_per_unit"], composite_ev["weighted_confidence"])
                        bets.append({
                            "market": "Spread",
                            "side": f"{home_name} {data['line']:+.1f}",
                            "probability": p_home,
                            "implied_prob": ev_calc["implied_prob"],
                            "edge": edge,
                            "ev_per_unit": ev_calc["ev_per_unit"],
                            "kelly_full": ev_calc["kelly_full"],
                            "kelly_quarter": ev_calc["kelly_quarter"],
                            "confidence": composite_ev["weighted_confidence"],
                            "rating": rating,
                            "reliability": stability["reliability"],
                            "risk_tier": risk["risk_tier"],
                            "bet_worthy": ev_calc["bet_worthy"],
                            "is_market_line": data.get("is_market_line", False),
                        })

                elif p_away > 0.54:
                    ev_calc = self._calculate_true_ev(p_away, standard_american_odds)
                    if ev_calc["bet_worthy"]:
                        rating = self._get_bet_rating(edge, ev_calc["ev_per_unit"], composite_ev["weighted_confidence"])
                        bets.append({
                            "market": "Spread",
                            "side": f"{away_name} {-data['line']:+.1f}",
                            "probability": p_away,
                            "implied_prob": ev_calc["implied_prob"],
                            "edge": edge,
                            "ev_per_unit": ev_calc["ev_per_unit"],
                            "kelly_full": ev_calc["kelly_full"],
                            "kelly_quarter": ev_calc["kelly_quarter"],
                            "confidence": composite_ev["weighted_confidence"],
                            "rating": rating,
                            "reliability": stability["reliability"],
                            "risk_tier": risk["risk_tier"],
                            "bet_worthy": ev_calc["bet_worthy"],
                            "is_market_line": data.get("is_market_line", False),
                        })

        # Moneyline
        ml_data = market.get("moneyline", {})
        p_home_win = ml_data.get("p_home_win", 0.5)
        p_away_win = ml_data.get("p_away_win", 0.5)

        home_ml = market_data.get("home_ml_current", 0)
        away_ml = market_data.get("away_ml_current", 0)

        if home_ml != 0 and p_home_win > 0.52:
            ev_calc = self._calculate_true_ev(p_home_win, home_ml)
            if ev_calc["bet_worthy"]:
                rating = self._get_bet_rating(ev_calc["edge"], ev_calc["ev_per_unit"], composite_ev["weighted_confidence"])
                bets.append({
                    "market": "Moneyline",
                    "side": f"{home_name} ML",
                    "probability": p_home_win,
                    "implied_prob": ev_calc["implied_prob"],
                    "edge": ev_calc["edge"],
                    "ev_per_unit": ev_calc["ev_per_unit"],
                    "kelly_full": ev_calc["kelly_full"],
                    "kelly_quarter": ev_calc["kelly_quarter"],
                    "confidence": composite_ev["weighted_confidence"],
                    "rating": rating,
                    "reliability": stability["reliability"],
                    "risk_tier": risk["risk_tier"],
                    "bet_worthy": ev_calc["bet_worthy"],
                    "odds": home_ml,
                })

        if away_ml != 0 and p_away_win > 0.52:
            ev_calc = self._calculate_true_ev(p_away_win, away_ml)
            if ev_calc["bet_worthy"]:
                rating = self._get_bet_rating(ev_calc["edge"], ev_calc["ev_per_unit"], composite_ev["weighted_confidence"])
                bets.append({
                    "market": "Moneyline",
                    "side": f"{away_name} ML",
                    "probability": p_away_win,
                    "implied_prob": ev_calc["implied_prob"],
                    "edge": ev_calc["edge"],
                    "ev_per_unit": ev_calc["ev_per_unit"],
                    "kelly_full": ev_calc["kelly_full"],
                    "kelly_quarter": ev_calc["kelly_quarter"],
                    "confidence": composite_ev["weighted_confidence"],
                    "rating": rating,
                    "reliability": stability["reliability"],
                    "risk_tier": risk["risk_tier"],
                    "bet_worthy": ev_calc["bet_worthy"],
                    "odds": away_ml,
                })

        # Sort by composite score (edge * confidence * ev)
        bets.sort(
            key=lambda x: x["edge"] * x["confidence"] * max(0, x["ev_per_unit"] + 0.1),
            reverse=True
        )

        # Prioritize market lines
        market_line_bets = [b for b in bets if b.get("is_market_line", False)]
        other_bets = [b for b in bets if not b.get("is_market_line", False)]

        final_bets = market_line_bets[:2] + other_bets
        return final_bets[:8]

    def _get_bet_rating(self, edge: float, ev: float, confidence: float) -> str:
        """Determina el rating de la apuesta."""
        score = edge * 100 + ev * 50 + confidence * 20

        if score > 25 and ev > 0.05:
            return "A+ ⭐"
        elif score > 20 and ev > 0.04:
            return "A"
        elif score > 15 and ev > 0.03:
            return "B+"
        elif score > 10 and ev > 0.02:
            return "B"
        elif score > 5:
            return "C+"
        else:
            return "C"


# ============================================================
# FUNCIÓN PRINCIPAL (HOOK PARA TU APP)
# ============================================================

def run_module(
    data: Optional[Dict[str, Any]] = None,
    home_team: Optional[Dict[str, Any]] = None,
    away_team: Optional[Dict[str, Any]] = None,
    game_context: Optional[Dict[str, Any]] = None,
    market_data: Optional[Dict[str, Any]] = None,
    n_simulations: int = 100_000,
) -> Dict[str, Any]:
    """
    Función principal para ejecutar NBA G10+ Ultra Pro V2.
    Si no pasas datos, usa un demo Lakers vs Nuggets (con altitude test).
    """

    if data is not None:
        home_team = data.get("home_team", home_team)
        away_team = data.get("away_team", away_team)
        game_context = data.get("game_context", game_context)
        market_data = data.get("market_data", market_data)
        n_simulations = int(data.get("n_simulations", n_simulations))

    # Demo data mejorado (Lakers @ Nuggets para probar altitude)
    if home_team is None or away_team is None:
        home_team = {
            "name": "Denver Nuggets",
            "offensive_rating": 119.2,
            "defensive_rating": 112.5,
            "pace": 98.5,
            "pace_last_5": 99.2,
            "pace_last_10": 98.8,
            "pace_home": 99.0,
            "pace_vs_similar": 98.0,
            "efg_last_5": 0.570,
            "efg_season": 0.562,
            "scoring_variance": 7.5,
            "consistency_score": 78.0,
            "points_last_3": 118.0,
            "points_last_5": 117.0,
            "points_last_10": 116.5,
            "points_per_game": 116.8,
            "hfa_multiplier": 1.15,  # Denver tiene fuerte HFA
            "points_in_paint": 52.0,
            "opp_points_in_paint": 46.0,
            "fastbreak_points": 15.5,
            "opp_fastbreak_points": 12.0,
            "turnovers": 12.5,
            "steals": 8.2,
            "free_throw_rate": 0.28,
            "offensive_rebounds": 11.0,
            "defensive_rebounds": 35.0,
            "assists": 28.5,
            "three_pt_offense": 0.38,
            "three_pt_defense": 0.35,
            "three_pt_attempts": 34.0,
            "bench_points": 38.0,
            "clutch_net_rating": 8.5,
            "pnr_offense_rating": 108.0,
            "pnr_defense_rating": 95.0,
            "rebounding_rate": 52.0,
        }

        away_team = {
            "name": "Los Angeles Lakers",
            "offensive_rating": 117.8,
            "defensive_rating": 113.2,
            "pace": 100.5,
            "pace_last_5": 101.2,
            "pace_last_10": 100.8,
            "pace_away": 99.8,
            "pace_vs_similar": 99.5,
            "efg_last_5": 0.558,
            "efg_season": 0.552,
            "scoring_variance": 8.5,
            "consistency_score": 72.0,
            "points_last_3": 116.0,
            "points_last_5": 115.0,
            "points_last_10": 114.5,
            "points_per_game": 115.2,
            "hfa_multiplier": 1.0,
            "points_in_paint": 54.0,
            "opp_points_in_paint": 48.0,
            "fastbreak_points": 16.0,
            "opp_fastbreak_points": 14.5,
            "turnovers": 13.5,
            "steals": 7.8,
            "free_throw_rate": 0.30,
            "offensive_rebounds": 10.5,
            "defensive_rebounds": 33.5,
            "assists": 26.0,
            "three_pt_offense": 0.36,
            "three_pt_defense": 0.37,
            "three_pt_attempts": 32.0,
            "bench_points": 32.0,
            "clutch_net_rating": 3.5,
            "pnr_offense_rating": 105.0,
            "pnr_defense_rating": 98.0,
            "rebounding_rate": 49.5,
        }

        game_context = {
            "game_date": "2024-12-05",
            "home_b2b": False,
            "away_b2b": True,
            "away_3in4": True,
            "away_travel_miles": 850,
            "timezone_direction": "west_to_east",
            "home_days_rest": 2,
            "away_days_rest": 0,
            "home_win_streak": 4,
            "away_loss_streak": 2,
            "national_tv": True,
            "away_injuries": [
                {"player": "Anthony Davis", "level": "star"},
            ],
            "home_injuries": [],
        }

        market_data = {
            "opening_spread": -5.5,
            "current_spread": -6.5,
            "opening_total": 226.5,
            "current_total": 225.0,
            "tickets_pct_home": 48.0,
            "money_pct_home": 62.0,
            "tickets_pct_over": 55.0,
            "money_pct_over": 45.0,
            "home_ml_current": -240,
            "away_ml_current": 200,
            "steam_moves": [
                {"time": "10:30", "side": "home", "move": -1.0},
            ],
            "sharp_reports": ["Sharp action on Denver -6"],
        }

    analyzer = NBAAnalyzerG10PlusV2()
    return analyzer.analyze_game(
        home_team=home_team,
        away_team=away_team,
        game_context=game_context,
        market_data=market_data,
        n_simulations=int(n_simulations),
    )


# ============================================================
# MAIN - TEST RUN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NBA G10+ ULTRA PRO V2 - TEST RUN")
    print("=" * 70 + "\n")

    res = run_module()

    if res["status"] == "success":
        print("\n" + "=" * 70)
        print("📊 FINAL PREDICTIONS")
        print("=" * 70)
        print(
            f"  {res['game_info']['away_team']} @ {res['game_info']['home_team']}"
        )
        print(f"  Date: {res['game_info']['game_date']}")
        print()
        print(
            f"  {res['game_info']['home_team']}: "
            f"{res['predictions']['home_points']:.1f} pts"
        )
        print(
            f"  {res['game_info']['away_team']}: "
            f"{res['predictions']['away_points']:.1f} pts"
        )
        print(f"  Total: {res['predictions']['total']:.1f}")
        print(f"  Spread: {res['predictions']['spread']:+.1f}")

        print("\n" + "-" * 40)
        print("🎯 WIN PROBABILITIES")
        print("-" * 40)
        print(
            f"  {res['game_info']['home_team']}: "
            f"{res['probabilities']['home_win']:.1%}"
        )
        print(
            f"  {res['game_info']['away_team']}: "
            f"{res['probabilities']['away_win']:.1%}"
        )

        print("\n" + "-" * 40)
        print("📈 SIMULATION STATS")
        print("-" * 40)
        stats = res["simulation_stats"]
        print(f"  Total Mean: {stats['total_mean']:.1f} (σ={stats['total_std']:.1f})")
        print(f"  Spread Mean: {stats['spread_mean']:+.1f} (σ={stats['spread_std']:.1f})")

        print("\n" + "-" * 40)
        print("🏆 COMPOSITE ANALYSIS")
        print("-" * 40)
        ev = res["composite_ev"]
        print(f"  Grade: {ev['grade']}")
        print(f"  Confidence: {ev['weighted_confidence']:.1%}")
        print(f"  Reliability: {ev['reliability']:.1%}")
        print(f"  EV Multiplier: {ev['ev_multiplier']:.3f}")
        print(f"  Composite Score: {ev['composite_score']:.1f}/100")

        print("\n" + "=" * 70)
        print("💰 BEST BETS (with True EV)")
        print("=" * 70)

        for i, bet in enumerate(res["best_bets"][:5], 1):
            market_indicator = "📍" if bet.get("is_market_line", False) else "  "
            print(f"\n{market_indicator}{i}. [{bet['rating']}] {bet['market']}: {bet['side']}")
            print(f"      Prob: {bet['probability']:.1%} vs Implied: {bet['implied_prob']:.1%}")
            print(f"      Edge: {bet['edge']:.1%} | EV/Unit: {bet['ev_per_unit']:+.2%}")
            print(f"      Kelly Full: {bet['kelly_full']:.2%} | Quarter: {bet['kelly_quarter']:.2%}")
            print(f"      Risk: {bet['risk_tier']} | Reliability: {bet['reliability']:.1%}")

        print("\n" + "=" * 70)
        print("✅ Analysis Complete!")
        print("=" * 70 + "\n")

    else:
        print(f"\n❌ ERROR: {res.get('error')}")
        if "traceback" in res:
            print(f"\nTraceback:\n{res['traceback']}")
