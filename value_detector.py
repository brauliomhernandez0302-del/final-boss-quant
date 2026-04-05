# =======================================================
# VALUE DETECTOR - Versión Profesional con Visualización
# Autor: Braulio & GPT-5
# Descripción:
# Analiza valor esperado (EV) con presentación visual tipo trader.
# =======================================================

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt


class ValueDetector:
    def __init__(self, threshold: float = 0.05):
        """
        :param threshold: margen mínimo de ventaja (5% por defecto)
        """
        self.threshold = threshold

    @staticmethod
    def calculate_ev(prob_real: float, odd: float) -> float:
        """Calcula el valor esperado (EV) de una apuesta."""
        return (prob_real * odd) - 1

    def analyze(self, match_name: str, probs: dict, odds: dict) -> pd.DataFrame:
        """
        Analiza si existe valor en las cuotas del mercado.
        :param match_name: Nombre del partido (str)
        :param probs: Probabilidades del modelo (dict con keys home_win_prob, draw_prob, away_win_prob)
        :param odds: Cuotas ofrecidas (dict con keys home_win, draw, away_win)
        :return: DataFrame con resultados y EV calculado
        """
        data = []

        for result_type in ["home_win", "draw", "away_win"]:
            prob = probs[result_type + "_prob"] / 100  # convertir a decimal
            odd = odds.get(result_type)

            if odd and odd > 1.01:
                ev = self.calculate_ev(prob, odd)
                status = "✅ Valor Positivo" if ev >= self.threshold else "⚠️ Sin valor"
                color = "#00C853" if ev >= self.threshold else "#FF5252"

                data.append({
                    "Partido": match_name,
                    "Resultado": result_type.replace("_", " ").title(),
                    "Probabilidad Modelo (%)": round(prob * 100, 2),
                    "Cuota Casa": odd,
                    "Valor Esperado (EV)": round(ev * 100, 2),
                    "Evaluación": status,
                    "Color": color
                })

        df = pd.DataFrame(data)
        return df.sort_values(by="Valor Esperado (EV)", ascending=False)

    def show_table(self, df: pd.DataFrame):
        """Muestra tabla y gráfico visual en Streamlit."""
        if df.empty:
            st.warning("No hay datos disponibles para mostrar.")
            return

        # Título
        st.markdown("### 💹 Análisis de Valor Esperado (EV)")

        # Mostrar tabla (sin columna de color)
        st.dataframe(df.drop(columns=["Color"]), use_container_width=True)

        # Gráfico de barras Altair
        chart = (
            alt.Chart(df)
            .mark_bar(size=50)
            .encode(
                x=alt.X("Resultado:N", title="Tipo de Resultado"),
                y=alt.Y("Valor Esperado (EV):Q", title="Valor (%)"),
                color=alt.Color(
                    "Evaluación:N",
                    scale=alt.Scale(
                        domain=["✅ Valor Positivo", "⚠️ Sin valor"],
                        range=["#00C853", "#FF5252"]
                    )
                ),
                tooltip=[
                    "Resultado",
                    "Probabilidad Modelo (%)",
                    "Cuota Casa",
                    "Valor Esperado (EV)",
                    "Evaluación"
                ]
            )
            .properties(width=600, height=350)
        )

        st.altair_chart(chart, use_container_width=True)

