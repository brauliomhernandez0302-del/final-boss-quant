"""
UFC MODULE G8+ ULTRA - FIGHT PREDICTION SYSTEM

Sistema complejo de predicción para peleas UFC/MMA
Incluye: Análisis de estilos, estadísticas, simulación de rounds
Autor: Braulio & Claude
Versión: G8+ Ultra
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================

# Ventajas por estilo de pelea (matchup matrix)
STYLE_ADVANTAGE: Dict[str, float] = {
    "striker_vs_striker": 0.0,       # Neutro
    "striker_vs_grappler": 0.15,     # Striker tiene ventaja
    "striker_vs_wrestler": 0.10,     # Striker leve ventaja
    "grappler_vs_striker": -0.15,    # Grappler desventaja
    "grappler_vs_grappler": 0.0,     # Neutro
    "grappler_vs_wrestler": -0.05,   # Grappler leve desventaja
    "wrestler_vs_striker": -0.10,    # Wrestler leve desventaja
    "wrestler_vs_grappler": 0.05,    # Wrestler leve ventaja
    "wrestler_vs_wrestler": 0.0,     # Neutro
}

# Pesos por división (en libras)
WEIGHT_CLASSES: Dict[str, int] = {
    "Flyweight": 125,
    "Bantamweight": 135,
    "Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
    "Women Strawweight": 115,
    "Women Flyweight": 125,
    "Women Bantamweight": 135,
    "Women Featherweight": 145,
}

# ============================================================
# CLASE PRINCIPAL: UFC ANALYZER
# ============================================================

class UFCAnalyzer:
    """
    Analizador complejo para peleas UFC.
    """

    def __init__(self) -> None:
        self.calibration_factor = 1.0
        self.home_advantage = 0.0  # UFC no tiene home advantage real

    def analyze_fight(
        self,
        fighter1: Dict[str, Any],
        fighter2: Dict[str, Any],
        weight_class: str = "Welterweight",
        is_title_fight: bool = False,
        n_simulations: int = 100_000,
    ) -> Dict[str, Any]:
        """
        Análisis completo de una pelea UFC.
        """

        logger.info("\n" + "=" * 60)
        logger.info("UFC G8+ ULTRA - FIGHT ANALYSIS")
        logger.info(f"{fighter1['name']} vs {fighter2['name']}")
        logger.info("=" * 60 + "\n")

        try:
            # Paso 1: Calcular skill ratings base
            logger.info("PASO 1: Calculando skill ratings...")
            skill1 = self._calculate_skill_rating(fighter1)
            skill2 = self._calculate_skill_rating(fighter2)

            logger.info(f"  {fighter1['name']}: {skill1:.3f}")
            logger.info(f"  {fighter2['name']}: {skill2:.3f}")

            # Paso 2: Ajuste por estilos de pelea (matchup)
            logger.info("\nPASO 2: Analizando matchup de estilos...")
            style_adj1, style_adj2 = self._calculate_style_matchup(
                fighter1.get("style", "balanced"),
                fighter2.get("style", "balanced"),
            )

            skill1_adjusted = skill1 * (1 + style_adj1)
            skill2_adjusted = skill2 * (1 + style_adj2)

            logger.info(f"  Style advantage F1: {style_adj1:+.1%}")
            logger.info(f"  Style advantage F2: {style_adj2:+.1%}")

            # Paso 3: Factores contextuales
            logger.info("\nPASO 3: Aplicando factores contextuales...")

            # Age factor
            age_adj1 = self._calculate_age_factor(fighter1.get("age", 30))
            age_adj2 = self._calculate_age_factor(fighter2.get("age", 30))

            # Momentum factor (racha)
            momentum1 = self._calculate_momentum(
                fighter1.get("recent_results", [])
            )
            momentum2 = self._calculate_momentum(
                fighter2.get("recent_results", [])
            )

            # Reach advantage
            reach_adj1, reach_adj2 = self._calculate_reach_advantage(
                fighter1.get("reach", 70),
                fighter2.get("reach", 70),
            )

            # Ajustes finales
            skill1_final = (
                skill1_adjusted
                * age_adj1
                * (1 + momentum1)
                * (1 + reach_adj1)
            )
            skill2_final = (
                skill2_adjusted
                * age_adj2
                * (1 + momentum2)
                * (1 + reach_adj2)
            )

            logger.info(f"  Age adjustment F1: {age_adj1:.3f}")
            logger.info(f"  Age adjustment F2: {age_adj2:.3f}")
            logger.info(f"  Momentum F1: {momentum1:+.1%}")
            logger.info(f"  Momentum F2: {momentum2:+.1%}")
            logger.info(f"  Reach advantage F1: {reach_adj1:+.1%}")
            logger.info(f"  Reach advantage F2: {reach_adj2:+.1%}")

            # Paso 4: Calcular probabilidades base
            logger.info("\nPASO 4: Calculando probabilidades base...")

            total_skill = skill1_final + skill2_final
            p_fighter1_win = skill1_final / total_skill
            p_fighter2_win = skill2_final / total_skill

            logger.info(f"  P({fighter1['name']} wins): {p_fighter1_win:.1%}")
            logger.info(f"  P({fighter2['name']} wins): {p_fighter2_win:.1%}")

            # Paso 5: Simular métodos de victoria
            logger.info(
                f"\nPASO 5: Simulando métodos de victoria ({n_simulations:,} sim)..."
            )

            finish_rates1 = fighter1.get(
                "finish_rate",
                {"ko_tko": 0.40, "submission": 0.20, "decision": 0.40},
            )
            finish_rates2 = fighter2.get(
                "finish_rate",
                {"ko_tko": 0.40, "submission": 0.20, "decision": 0.40},
            )

            simulation_results = self._simulate_fight_outcomes(
                p_fighter1_win,
                p_fighter2_win,
                finish_rates1,
                finish_rates2,
                is_title_fight=is_title_fight,
                n_sim=n_simulations,
            )

            # Paso 6: Analizar rounds
            if is_title_fight:
                logger.info(
                    "\nPASO 6: Análisis de rounds (Title Fight - 5 rounds)..."
                )
                round_analysis = self._analyze_round_probabilities(
                    skill1_final,
                    skill2_final,
                    fighter1.get("cardio", 0.8),
                    fighter2.get("cardio", 0.8),
                    n_rounds=5,
                )
            else:
                logger.info("\nPASO 6: Análisis de rounds (3 rounds)...")
                round_analysis = self._analyze_round_probabilities(
                    skill1_final,
                    skill2_final,
                    fighter1.get("cardio", 0.8),
                    fighter2.get("cardio", 0.8),
                    n_rounds=3,
                )

            logger.info("\n" + "=" * 60)
            logger.info("RESULTADOS FINALES")
            logger.info("=" * 60)

            result: Dict[str, Any] = {
                "status": "success",
                "fight_info": {
                    "fighter1": fighter1["name"],
                    "fighter2": fighter2["name"],
                    "weight_class": weight_class,
                    "is_title_fight": is_title_fight,
                },
                "skill_ratings": {
                    "fighter1_base": float(skill1),
                    "fighter2_base": float(skill2),
                    "fighter1_adjusted": float(skill1_adjusted),
                    "fighter2_adjusted": float(skill2_adjusted),
                    "fighter1_final": float(skill1_final),
                    "fighter2_final": float(skill2_final),
                },
                "adjustments": {
                    "style_matchup_f1": float(style_adj1),
                    "style_matchup_f2": float(style_adj2),
                    "age_factor_f1": float(age_adj1),
                    "age_factor_f2": float(age_adj2),
                    "momentum_f1": float(momentum1),
                    "momentum_f2": float(momentum2),
                    "reach_advantage_f1": float(reach_adj1),
                    "reach_advantage_f2": float(reach_adj2),
                },
                "probabilities": {
                    "fighter1_win": float(p_fighter1_win),
                    "fighter2_win": float(p_fighter2_win),
                    "fighter1_ko_tko": float(
                        simulation_results["f1_ko_tko"]
                    ),
                    "fighter1_submission": float(
                        simulation_results["f1_submission"]
                    ),
                    "fighter1_decision": float(
                        simulation_results["f1_decision"]
                    ),
                    "fighter2_ko_tko": float(
                        simulation_results["f2_ko_tko"]
                    ),
                    "fighter2_submission": float(
                        simulation_results["f2_submission"]
                    ),
                    "fighter2_decision": float(
                        simulation_results["f2_decision"]
                    ),
                    "goes_distance": float(
                        simulation_results["goes_distance"]
                    ),
                },
                "round_analysis": round_analysis,
                "best_bets": self._identify_best_bets(
                    p_fighter1_win,
                    p_fighter2_win,
                    simulation_results,
                    fighter1["name"],
                    fighter2["name"],
                ),
                "metadata": {
                    "model_version": "UFC G8+ Ultra",
                    "n_simulations": n_simulations,
                    "timestamp": datetime.now().isoformat(),
                },
            }

            logger.info("\n✅ Análisis completado exitosamente\n")
            return result

        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ Error en análisis: {e}")
            return {"status": "error", "error": str(e)}

    # ========================================================
    # MÉTODOS INTERNOS
    # ========================================================

    def _calculate_skill_rating(self, fighter: Dict[str, Any]) -> float:
        """
        Calcula skill rating base del peleador.
        """

        record = fighter.get("record", {"wins": 10, "losses": 5})
        wins = record.get("wins", 10)
        losses = record.get("losses", 5)

        total_fights = wins + losses
        if total_fights == 0:
            win_rate = 0.5
        else:
            # Laplace smoothing
            win_rate = (wins + 2) / (total_fights + 4)

        striking_acc = fighter.get("striking_accuracy", 0.45)
        takedown_acc = fighter.get("takedown_accuracy", 0.40)
        takedown_def = fighter.get("takedown_defense", 0.65)
        striking_def = fighter.get("striking_defense", 0.55)
        ko_rate = fighter.get("ko_rate", 0.30)
        sub_rate = fighter.get("submission_rate", 0.20)

        skill = (
            win_rate * 3.0
            + striking_acc * 2.0
            + takedown_acc * 1.5
            + takedown_def * 1.5
            + striking_def * 1.0
            + (ko_rate + sub_rate) * 1.0
        )

        total_fights = max(total_fights, 0)
        experience_bonus = min(total_fights / 30.0, 0.2)
        skill = skill * (1 + experience_bonus)

        skill = float(np.clip(skill, 0.0, 10.0))
        return skill

    def _calculate_style_matchup(
        self,
        style1: str,
        style2: str,
    ) -> Tuple[float, float]:
        """
        Calcula ventajas/desventajas por matchup de estilos.
        """

        matchup_key = f"{style1}_vs_{style2}"

        if matchup_key in STYLE_ADVANTAGE:
            adj1 = STYLE_ADVANTAGE[matchup_key]
            adj2 = -adj1
        else:
            adj1 = 0.0
            adj2 = 0.0

        return adj1, adj2

    def _calculate_age_factor(self, age: int) -> float:
        """
        Ajuste por edad del peleador (peak 27–32).
        """

        if 27 <= age <= 32:
            return 1.0
        if age < 27:
            return 0.95 + (age - 20) * 0.007
        decline = (age - 32) * 0.015
        return max(0.80, 1.0 - decline)

    def _calculate_momentum(self, recent_results: List[str]) -> float:
        """
        Calcula momentum basado en resultados recientes.
        """

        if not recent_results:
            return 0.0

        weights = [0.35, 0.25, 0.20, 0.12, 0.08]

        momentum = 0.0
        for i, result in enumerate(recent_results[:5]):
            if i >= len(weights):
                break
            r = result.upper()
            if r == "W":
                momentum += weights[i]
            elif r == "L":
                momentum -= weights[i]

        if len(recent_results) >= 3 and all(
            r.upper() == "W" for r in recent_results[:3]
        ):
            momentum += 0.10

        return float(np.clip(momentum, -0.20, 0.30))

    def _calculate_reach_advantage(
        self,
        reach1: float,
        reach2: float,
    ) -> Tuple[float, float]:
        """
        Calcula ventaja por alcance (reach).
        """

        reach_diff = reach1 - reach2
        reach_factor = float(np.clip(reach_diff * 0.005, -0.05, 0.05))
        return reach_factor, -reach_factor

    def _simulate_fight_outcomes(
        self,
        p_f1_win: float,
        p_f2_win: float,
        finish_rates1: Dict[str, float],
        finish_rates2: Dict[str, float],
        is_title_fight: bool = False,  # noqa: ARG002
        n_sim: int = 100_000,
    ) -> Dict[str, float]:
        """
        Simulación Monte Carlo de métodos de victoria.
        """

        winners = np.random.random(n_sim) < p_f1_win

        f1_ko = f1_sub = f1_dec = 0
        f2_ko = f2_sub = f2_dec = 0

        for winner_is_f1 in winners:
            if winner_is_f1:
                finish_type = np.random.random()
                if finish_type < finish_rates1["ko_tko"]:
                    f1_ko += 1
                elif finish_type < finish_rates1["ko_tko"] + finish_rates1["submission"]:
                    f1_sub += 1
                else:
                    f1_dec += 1
            else:
                finish_type = np.random.random()
                if finish_type < finish_rates2["ko_tko"]:
                    f2_ko += 1
                elif finish_type < finish_rates2["ko_tko"] + finish_rates2["submission"]:
                    f2_sub += 1
                else:
                    f2_dec += 1

        total = float(n_sim)
        return {
            "f1_ko_tko": f1_ko / total,
            "f1_submission": f1_sub / total,
            "f1_decision": f1_dec / total,
            "f2_ko_tko": f2_ko / total,
            "f2_submission": f2_sub / total,
            "f2_decision": f2_dec / total,
            "goes_distance": (f1_dec + f2_dec) / total,
        }

    def _analyze_round_probabilities(
        self,
        skill1: float,
        skill2: float,
        cardio1: float,
        cardio2: float,
        n_rounds: int = 3,
    ) -> Dict[str, Any]:
        """
        Analiza probabilidades round-by-round.
        """

        round_probs: Dict[str, Any] = {}

        for round_num in range(1, n_rounds + 1):
            fatigue_f1 = 1.0 - ((round_num - 1) * (1 - cardio1) * 0.15)
            fatigue_f2 = 1.0 - ((round_num - 1) * (1 - cardio2) * 0.15)

            skill1_round = skill1 * fatigue_f1
            skill2_round = skill2 * fatigue_f2

            total_skill = skill1_round + skill2_round
            p_f1_round = skill1_round / total_skill
            p_f2_round = skill2_round / total_skill

            round_probs[f"round_{round_num}"] = {
                "fighter1_win_prob": float(p_f1_round),
                "fighter2_win_prob": float(p_f2_round),
                "fatigue_f1": float(fatigue_f1),
                "fatigue_f2": float(fatigue_f2),
            }

        return round_probs

    def _identify_best_bets(
        self,
        p_f1: float,
        p_f2: float,
        sim_results: Dict[str, float],
        name1: str,
        name2: str,
    ) -> List[Dict[str, Any]]:
        """
        Identifica mejores apuestas potenciales.
        """

        bets: List[Dict[str, Any]] = []

        if p_f1 > 0.60:
            bets.append(
                {
                    "market": f"{name1} Moneyline",
                    "probability": p_f1,
                    "confidence": "HIGH" if p_f1 > 0.70 else "MEDIUM",
                    "rating": "A" if p_f1 > 0.70 else "B",
                }
            )

        if p_f2 > 0.60:
            bets.append(
                {
                    "market": f"{name2} Moneyline",
                    "probability": p_f2,
                    "confidence": "HIGH" if p_f2 > 0.70 else "MEDIUM",
                    "rating": "A" if p_f2 > 0.70 else "B",
                }
            )

        if sim_results["f1_ko_tko"] > 0.30:
            bets.append(
                {
                    "market": f"{name1} by KO/TKO",
                    "probability": sim_results["f1_ko_tko"],
                    "confidence": "MEDIUM",
                    "rating": "B+",
                }
            )

        if sim_results["f2_ko_tko"] > 0.30:
            bets.append(
                {
                    "market": f"{name2} by KO/TKO",
                    "probability": sim_results["f2_ko_tko"],
                    "confidence": "MEDIUM",
                    "rating": "B+",
                }
            )

        if sim_results["goes_distance"] > 0.50:
            bets.append(
                {
                    "market": "Fight goes the distance",
                    "probability": sim_results["goes_distance"],
                    "confidence": "MEDIUM",
                    "rating": "B",
                }
            )

        return bets

# ============================================================
# FUNCIÓN PRINCIPAL PARA INTEGRACIÓN CON APP
# ============================================================

def run_module(
    data: Optional[Dict[str, Any]] = None,
    fighter1_data: Optional[Dict[str, Any]] = None,
    fighter2_data: Optional[Dict[str, Any]] = None,
    weight_class: str = "Welterweight",
    is_title_fight: bool = False,
    n_simulations: int = 100_000,
) -> Dict[str, Any]:
    """
    Función principal para ejecutar el módulo UFC desde app.py.

    La app suele llamar: run_module(data=...) por eso el primer
    parámetro se llama exactamente 'data'.
    """

    # Si viene un diccionario genérico desde app.py, lo usamos
    if data is not None:
        fighter1_data = data.get("fighter1", fighter1_data)
        fighter2_data = data.get("fighter2", fighter2_data)
        weight_class = data.get("weight_class", weight_class)
        is_title_fight = data.get("is_title_fight", is_title_fight)
        n_simulations = data.get("n_simulations", n_simulations)

    # Si aún no hay datos, usamos un ejemplo de demo
    if fighter1_data is None or fighter2_data is None:
        fighter1_data = {
            "name": "Fighter A",
            "age": 29,
            "reach": 74,
            "style": "striker",
            "record": {"wins": 18, "losses": 3},
            "striking_accuracy": 0.52,
            "takedown_accuracy": 0.35,
            "takedown_defense": 0.75,
            "striking_defense": 0.60,
            "ko_rate": 0.45,
            "submission_rate": 0.15,
            "cardio": 0.85,
            "recent_results": ["W", "W", "W", "L", "W"],
            "finish_rate": {
                "ko_tko": 0.50,
                "submission": 0.15,
                "decision": 0.35,
            },
        }

        fighter2_data = {
            "name": "Fighter B",
            "age": 32,
            "reach": 72,
            "style": "grappler",
            "record": {"wins": 15, "losses": 5},
            "striking_accuracy": 0.42,
            "takedown_accuracy": 0.55,
            "takedown_defense": 0.60,
            "striking_defense": 0.52,
            "ko_rate": 0.25,
            "submission_rate": 0.40,
            "cardio": 0.80,
            "recent_results": ["W", "W", "L", "W", "L"],
            "finish_rate": {
                "ko_tko": 0.25,
                "submission": 0.45,
                "decision": 0.30,
            },
        }

    analyzer = UFCAnalyzer()
    return analyzer.analyze_fight(
        fighter1=fighter1_data,
        fighter2=fighter2_data,
        weight_class=weight_class,
        is_title_fight=is_title_fight,
        n_simulations=int(n_simulations),
    )

if __name__ == "__main__":
    # Pequeña prueba rápida en modo script
    res = run_module()
    print(res["probabilities"])
    print("\nBEST BETS:")
    for bet in res["best_bets"]:
        print(f"  [{bet['rating']}] {bet['market']}: {bet['probability']:.1%}")
