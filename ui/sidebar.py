"""
ui/sidebar.py — Sidebar config panel and prediction history view.

Both functions receive `db` explicitly — no module-level globals.
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import streamlit as st

import config as _cfg
from db.predictions_db import PredictionsDB
from ui.components import CONFIG, ThemeColors


def render_sidebar(db: PredictionsDB, sport_name: str = "MLB") -> Dict[str, float]:
    """Render the sidebar and return the user's settings dict."""
    with st.sidebar:
        st.header("📊 Estadísticas Generales")
        stats = db.get_stats(sport_name or "MLB")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Picks",  stats["total_picks"])
        with c2:
            st.metric("EV+ Rate",     f"{stats['positive_ev_rate']:.0%}")

        st.metric("EV Promedio",     f"{stats['avg_ev'] * 100:.2f}%")
        st.metric("Rating Promedio", f"{stats['avg_rating']:.1f}/10")

        st.markdown("---")
        st.header("⚙️ Configuración")

        kelly_factor = st.slider(
            "Factor Kelly",
            min_value=0.1, max_value=0.5,
            value=CONFIG.DEFAULT_KELLY_FACTOR, step=0.05,
            help="Fracción del Kelly Criterion (0.25 = Quarter Kelly)",
        )
        min_ev = st.slider(
            "EV Mínimo (%)",
            min_value=0.0, max_value=10.0,
            value=CONFIG.DEFAULT_MIN_EV, step=0.5,
            help="Expected Value mínimo para considerar apuesta",
        )
        min_rating = st.slider(
            "Rating Mínimo",
            min_value=0.0, max_value=10.0,
            value=CONFIG.DEFAULT_MIN_RATING, step=0.5,
            help="Rating mínimo para mostrar en resultados",
        )

        st.markdown("---")

        if st.button("📜 Ver Historial", use_container_width=True):
            st.session_state["show_history"] = True

        if st.button("🗑️ Limpiar Caché", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Caché limpiado")

        return {
            "kelly_factor": kelly_factor,
            "min_ev":       min_ev,
            "min_rating":   min_rating,
        }


def render_history(
    db: PredictionsDB,
    sport: str,
    settings: Dict[str, Any],
) -> None:
    """Render the prediction history table + EV chart."""
    st.subheader("📜 Historial de Predicciones")

    history_df = db.read(sport=sport, limit=CONFIG.MAX_HISTORY_RECORDS)

    if history_df.empty:
        st.info("No hay historial para este deporte")
        if st.button("❌ Cerrar Historial"):
            st.session_state["show_history"] = False
            st.rerun()
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Picks", len(history_df))
    with c2:
        avg_ev = history_df["ev"].mean() * 100 if "ev" in history_df.columns else 0.0
        st.metric("EV Promedio", f"{avg_ev:.2f}%")
    with c3:
        avg_rating = history_df["rating"].mean() if "rating" in history_df.columns else 0.0
        st.metric("Rating Promedio", f"{avg_rating:.1f}/10")
    with c4:
        if st.button("❌ Cerrar"):
            st.session_state["show_history"] = False
            st.rerun()

    display_cols = [c for c in
                    ["timestamp", "sport", "home_team", "away_team", "pick_type", "ev", "kelly", "rating"]
                    if c in history_df.columns]
    st.dataframe(history_df[display_cols].head(CONFIG.MAX_DISPLAY_RECORDS), use_container_width=True)

    if "ev" in history_df.columns and len(history_df) > 10:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history_df.index, history_df["ev"] * 100,
                linewidth=2, color=ThemeColors.PRIMARY.value)
        ax.axhline(y=0, linestyle="--", alpha=0.5, color="white")
        ax.set_xlabel("Pick #", color="white")
        ax.set_ylabel("EV (%)", color="white")
        ax.set_facecolor(ThemeColors.BG_DARK.value)
        fig.patch.set_facecolor(ThemeColors.BG_DARK.value)
        ax.tick_params(colors="white")
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("white")
        st.pyplot(fig)
        plt.close(fig)
