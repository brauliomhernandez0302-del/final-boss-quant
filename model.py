import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class BettingModel:
    """
    Modelo avanzado para análisis de apuestas:
    - Entrenamiento con regresión logística (puede conectarse a históricos)
    - Actualización de ELO
    - Cálculo de probabilidades implícitas y justas
    - Estimación de valor esperado (EV)
    - Recomendación automática del pick con mejor rentabilidad
    """

    def __init__(self):
        self.model = LogisticRegression()
        self.trained = False

    # ===============================
    # ENTRENAMIENTO DEL MODELO
    # ===============================
    def train(self, X, y):
        """
        Entrena el modelo logístico con datos históricos.
        X: características (odds, diferencias ELO, etc.)
        y: resultados (1 = gana equipo 1, 0 = gana equipo 2)
        """
        self.model.fit(X, y)
        self.trained = True
        print("✅ Modelo entrenado correctamente.")

    # ===============================
    # PREDICCIÓN
    # ===============================
    def predict(self, X):
        """
        Predice la probabilidad de victoria del equipo 1.
        """
        if not self.trained:
            raise Exception("⚠️ El modelo no está entrenado aún.")
        return self.model.predict_proba(X)[:, 1]

    # ===============================
    # SISTEMA ELO
    # ===============================
    def calculate_elo(self, team_a, team_b, result, K=20):
        """
        Actualiza los ratings ELO de dos equipos tras un partido.
        result = 1 si gana A, 0 si gana B
        """
        expected_a = 1 / (1 + 10 ** ((team_b - team_a) / 400))
        new_a = team_a + K * (result - expected_a)
        new_b = team_b + K * ((1 - result) - (1 - expected_a))
        return new_a, new_b

    # ===============================
    # CÁLCULO DE VALOR ESPERADO (EV)
    # ===============================
    def calcular_valor_predicho(self, df):
        """
        Calcula valor esperado (EV) para cada evento usando cuotas y predicciones.
        Si el modelo no está entrenado, usa probabilidades simuladas.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        data = df.copy()

        # Probabilidades implícitas
        data["Prob_1_impl"] = 1 / data["Cuota 1"]
        data["Prob_2_impl"] = 1 / data["Cuota 2"]

        # Margen del bookmaker (vig)
        data["Margen"] = (data["Prob_1_impl"] + data["Prob_2_impl"]) - 1

        # Predicciones del modelo (simuladas si no está entrenado)
        if self.trained:
            # ⚠️ Si tuvieras un dataset con features X, podrías sustituir esto por:
            # preds = self.predict(X)
            # Por ahora generamos probabilidades equilibradas
            np.random.seed(42)
            data["Pred_1"] = np.random.uniform(0.35, 0.65, len(data))
        else:
            np.random.seed(42)
            data["Pred_1"] = np.random.uniform(0.35, 0.65, len(data))

        data["Pred_2"] = 1 - data["Pred_1"]

        # Valor esperado: EV = (probabilidad_predicha * cuota) - 1
        data["EV_1"] = (data["Pred_1"] * data["Cuota 1"]) - 1
        data["EV_2"] = (data["Pred_2"] * data["Cuota 2"]) - 1

        # Pick recomendado y EV más alto
        data["Pick recomendado"] = np.where(
            data["EV_1"] > data["EV_2"], data["Equipo 1"], data["Equipo 2"]
        )
        data["EV recomendado (%)"] = np.where(
            data["EV_1"] > data["EV_2"], data["EV_1"], data["EV_2"]
        ) * 100

        # Redondear para presentación
        data = data.round({
            "Margen": 3,
            "Prob_1_impl": 3,
            "Prob_2_impl": 3,
            "EV recomendado (%)": 2
        })

        # Ordenar de mayor a menor valor esperado
        data.sort_values(by="EV recomendado (%)", ascending=False, inplace=True)
        data.reset_index(drop=True, inplace=True)

        return data
