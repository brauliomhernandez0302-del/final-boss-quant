# ============================================================
# SCORE_PREDICTOR.PY — Versión G5 ULTRA (Poisson + Calibración)
# Autor: Braulio & GPT-5
# ============================================================

import numpy as np
import pandas as pd
from math import exp, factorial
from itertools import product


class ScorePredictor:
    """
    Modelo basado en Poisson para estimar probabilidades de marcadores exactos
    y derivar las probabilidades 1X2 (local, empate, visitante).
    """

    def __init__(self, base_avg_home=1.45, base_avg_away=1.25):
        self.base_avg_home = base_avg_home
        self.base_avg_away = base_avg_away

    # ------------------------------------------------------------
    # Probabilidad Poisson
    # ------------------------------------------------------------
    @staticmethod
    def poisson_prob(lmbda, k):
        """Devuelve P(X=k) para distribución de Poisson"""
        return (lmbda ** k) * exp(-lmbda) / factorial(k)

    # ------------------------------------------------------------
    # Predicción de marcador exacto y 1X2
    # ------------------------------------------------------------
    def predict_match(
        self,
        home_team,
        away_team,
        attack_home=1.0,
        attack_away=1.0,
        defense_home=1.0,
        defense_away=1.0,
    ):
        """
        Calcula la probabilidad de cada resultado y las 1X2
        Parámetros:
        home_team / away_team (str)
        attack_home / attack_away / defense_home / defense_away (floats)
        """
        # Parámetros ajustados
        lambda_home = self.base_avg_home * attack_home * defense_away
        lambda_away = self.base_avg_away * attack_away * defense_home

        max_goals = 6
        outcomes = []

        # Matriz de marcadores
        for home_goals, away_goals in product(range(max_goals + 1), repeat=2):
            p_home = self.poisson_prob(lambda_home, home_goals)
            p_away = self.poisson_prob(lambda_away, away_goals)
            p_total = p_home * p_away
            outcomes.append((home_goals, away_goals, p_total))

        df = pd.DataFrame(outcomes, columns=["home_goals", "away_goals", "prob"])
        df["result"] = df.apply(
            lambda x: "Home"
            if x.home_goals > x.away_goals
            else "Away"
            if x.home_goals < x.away_goals
            else "Draw",
            axis=1,
        )

        # Probabilidades totales
        prob_home = df.loc[df["result"] == "Home", "prob"].sum()
        prob_draw = df.loc[df["result"] == "Draw", "prob"].sum()
        prob_away = df.loc[df["result"] == "Away", "prob"].sum()

        # Normalización (por seguridad)
        total = prob_home + prob_draw + prob_away
        if total > 0:
            prob_home /= total
            prob_draw /= total
            prob_away /= total

        result = {
    "home_team": home_team,
    "away_team": away_team,
    "lambda_home": float(round(float(locals().get("lambda_home", 0) or 0), 3)),
    "lambda_away": float(round(float(locals().get("lambda_away", 0) or 0), 3)),
    "prob_home": float(round(float(locals().get("prob_home", 0) or 0) * 100, 2)),
    "prob_draw": float(round(float(locals().get("prob_draw", 0) or 0) * 100, 2)),
    "prob_away": float(round(float(locals().get("prob_away", 0) or 0) * 100, 2)),
    "home_goals": locals().get("home_goals", 0),
    "away_goals": locals().get("away_goals", 0),
    "matrix": locals().get("df"),
        }
        return result 
       

    # ------------------------------------------------------------
    # Mostrar tabla compacta de probabilidades
    # ------------------------------------------------------------
    def show_summary(self, result_dict):
        """Imprime resumen de probabilidades 1X2"""
        print("\n=== PREDICCIÓN POISSON ===")
        print(f"{result_dict['home_team']} vs {result_dict['away_team']}")
        print(
            f"λ Local: {result_dict['lambda_home']}, λ Visitante: {result_dict['lambda_away']}"
        )
        print(
            f"Probabilidades → Local: {result_dict['prob_home']}% | "
            f"Empate: {result_dict['prob_draw']}% | "
            f"Visitante: {result_dict['prob_away']}%"
        )

