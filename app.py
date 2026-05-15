"""
app.py — Streamlit entry point.

Sole responsibilities:
  - Page config + global db instance
  - SPORT_CONFIGS dict
  - NBA / UFC stubs (pending full integration)
  - _render_analysis_tab() — glue between odds loading and sport analyzers
  - main()
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

import config as _cfg
from db.predictions_db import AnalysisResult, GameData, PredictionsDB
from ui.components import CONFIG, SportConfig, ThemeColors, UIComponents
from ui.mlb import MLBAnalyzer, render_mlb_results
from ui.odds_loader import build_game_selector, filter_odds_by_sport, load_odds_data
from ui.sidebar import render_history, render_sidebar

# Add modules/ to sys.path for basketball_module / ufc_module dynamic imports
sys.path.insert(0, str(_cfg.MODULES_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global DB instance ────────────────────────────────────────────────────

_cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
_cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)

db = PredictionsDB(_cfg.DATA_DIR / "predictions_history.db")


# ── Sport configuration ───────────────────────────────────────────────────

SPORT_CONFIGS: Dict[str, SportConfig] = {
    "MLB": SportConfig(
        name="MLB", display_name="⚾ Baseball (MLB)",
        module_name="mlb_predictor",
        keywords=["baseball", "mlb"],
        icon="⚾", simulations=CONFIG.MLB_SIMULATIONS,
    ),
    "NBA": SportConfig(
        name="NBA", display_name="🏀 Basketball (NBA)",
        module_name="basketball_module",
        keywords=["basketball", "nba"],
        icon="🏀", simulations=CONFIG.NBA_SIMULATIONS,
    ),
    "UFC": SportConfig(
        name="UFC", display_name="🥊 UFC / MMA",
        module_name="ufc_module",
        keywords=["mma", "ufc"],
        icon="🥊", simulations=CONFIG.UFC_SIMULATIONS,
    ),
    "SOCCER": SportConfig(
        name="SOCCER", display_name="⚽ Soccer",
        module_name="football_module",
        keywords=["soccer", "football"],
        icon="⚽", enabled=False,
    ),
}


# ── NBA / UFC stubs (not yet connected to live data) ─────────────────────

class _NBAAnalyzer:
    def __init__(self, config: SportConfig, settings: Dict, db: PredictionsDB) -> None:
        self.config   = config
        self.settings = settings
        self.db       = db

    def analyze(self, game_data: GameData) -> AnalysisResult:
        try:
            import basketball_module
            payload = {
                "home_team":    {"name": game_data["home"]},
                "away_team":    {"name": game_data["away"]},
                "game_context": {
                    "game_date":       game_data.get("commence_time", ""),
                    "home_b2b":        False, "away_b2b": False,
                    "away_travel_miles": 0,
                    "home_days_rest":  1,     "away_days_rest": 1,
                    "home_injuries":   [],    "away_injuries": [],
                },
                "n_simulations": self.config.simulations,
            }
            with st.spinner("⚡ Ejecutando NBA MODULE G8+..."):
                try:
                    return basketball_module.run_module(data=payload)
                except TypeError:
                    return basketball_module.run_module(
                        home_team    = payload["home_team"],
                        away_team    = payload["away_team"],
                        game_context = payload["game_context"],
                        n_simulations = payload["n_simulations"],
                    )
        except ImportError as exc:
            return AnalysisResult(status="error", error=f"Módulo no disponible: {exc}")
        except Exception as exc:
            logger.exception("Error en análisis NBA")
            return AnalysisResult(status="error", error=str(exc))


class _UFCAnalyzer:
    def __init__(self, config: SportConfig, settings: Dict, db: PredictionsDB) -> None:
        self.config   = config
        self.settings = settings
        self.db       = db

    def analyze(self, game_data: GameData) -> AnalysisResult:
        try:
            import ufc_module
            with st.spinner("⚡ Ejecutando UFC MODULE G8+ Ultra..."):
                return ufc_module.run_module(
                    fighter1_data  = {"name": game_data["home"]},
                    fighter2_data  = {"name": game_data["away"]},
                    weight_class   = "TBD",
                    is_title_fight = False,
                    n_simulations  = self.config.simulations,
                )
        except ImportError as exc:
            return AnalysisResult(status="error", error=f"Módulo no disponible: {exc}")
        except Exception as exc:
            logger.exception("Error en análisis UFC")
            return AnalysisResult(status="error", error=str(exc))


def _render_nba_results(result: AnalysisResult, game_data: GameData, settings: Dict) -> None:
    if result.get("status") != "success":
        st.error(f"❌ Error en módulo NBA: {result.get('error', 'Unknown')}")
        return
    st.success("✅ Análisis NBA completado")
    info  = result["game_info"]
    preds = result["predictions"]
    probs = result["probabilities"]
    home, away = info["home_team"], info["away_team"]
    st.markdown(f"### 🏀 {away} @ {home} ({info.get('game_date', '')})")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric(f"{home} pts", f"{preds['home_points']:.1f}")
    with c2: st.metric(f"{away} pts", f"{preds['away_points']:.1f}")
    with c3: st.metric("Total",       f"{preds['total']:.1f}")
    with st.expander("🔍 Detalles técnicos (NBA)"):
        st.json(result.get("metadata", {}))


def _render_ufc_results(result: AnalysisResult, game_data: GameData, settings: Dict) -> None:
    if result.get("status") != "success":
        st.error(f"❌ Error en módulo UFC: {result.get('error', 'Unknown')}")
        return
    st.success("✅ Análisis UFC completado")
    fight_info = result.get("fight_info", {})
    probs      = result.get("probabilities", {})
    f1 = fight_info.get("fighter1", game_data["home"])
    f2 = fight_info.get("fighter2", game_data["away"])
    st.markdown(f"### 🥊 {f1} vs {f2}")
    c1, c2 = st.columns(2)
    with c1: st.metric(f"{f1} Win", f"{probs.get('fighter1_win', 0.0):.1%}")
    with c2: st.metric(f"{f2} Win", f"{probs.get('fighter2_win', 0.0):.1%}")
    with st.expander("🔍 Detalles técnicos UFC"):
        st.json(result.get("metadata", {}))


# ── Analysis tab ──────────────────────────────────────────────────────────


def _render_analysis_tab(settings: Dict[str, Any]) -> None:
    odds_df = load_odds_data()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("📦 Eventos Totales", len(odds_df))
    with c2:
        n_sports = (odds_df.get("sport_key", __import__("pandas").Series(dtype=str)).nunique()
                    if not odds_df.empty else 0)
        st.metric("🎯 Deportes Disponibles", int(n_sports))
    with c3:
        if st.button("🔄 Recargar Datos"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    st.subheader("🎯 Selecciona el Deporte")

    sport_options  = {cfg.display_name: name for name, cfg in SPORT_CONFIGS.items()}
    selected_label = st.selectbox("Deporte:", list(sport_options.keys()), index=0)
    sport_name     = sport_options[selected_label]
    sport_config   = SPORT_CONFIGS[sport_name]

    sport_df = filter_odds_by_sport(odds_df, sport_config.keywords)
    st.info(f"🔍 **{sport_name}**: {len(sport_df)} eventos encontrados")

    if not sport_df.empty:
        with st.expander("📊 Ver datos crudos"):
            st.dataframe(sport_df.head(20), use_container_width=True)

    st.markdown("---")
    st.subheader(f"{sport_config.icon} Análisis de {sport_name}")

    if not sport_config.enabled:
        st.warning(f"⚠️ Módulo de {sport_name} en desarrollo")
        return

    if sport_df.empty:
        st.warning(f"⚠️ No hay eventos {sport_name} disponibles en Odds API")
        return

    st.success(f"✅ {len(sport_df)} eventos {sport_name} encontrados")
    game_options, game_mapping = build_game_selector(sport_df)

    if not game_options:
        return

    selected_game = st.selectbox(
        f"🎯 Selecciona el evento {sport_name} a analizar:",
        options=game_options, index=0,
    )
    game_data = game_mapping[selected_game]

    st.markdown("### 📊 Evento Seleccionado:")
    st.info(f"**{selected_game}**")

    c1, c2 = st.columns(2)
    with c1: st.metric(f"💵 {game_data['away']} Odds", f"{game_data['away_odds']:.2f}")
    with c2: st.metric(f"💵 {game_data['home']} Odds", f"{game_data['home_odds']:.2f}")

    st.markdown("---")

    if st.button(f"{sport_config.icon} Analizar Evento ({sport_name})", use_container_width=True):
        try:
            if sport_name == "MLB":
                analyzer = MLBAnalyzer(sport_config, settings, db)
                result   = analyzer.analyze(game_data)
                render_mlb_results(result, game_data, settings, db, sport_config)

            elif sport_name == "NBA":
                analyzer = _NBAAnalyzer(sport_config, settings, db)
                result   = analyzer.analyze(game_data)
                _render_nba_results(result, game_data, settings)

            elif sport_name == "UFC":
                analyzer = _UFCAnalyzer(sport_config, settings, db)
                result   = analyzer.analyze(game_data)
                _render_ufc_results(result, game_data, settings)

            else:
                st.warning(f"Analizador para {sport_name} no implementado")

        except Exception as exc:
            st.error(f"❌ Error ejecutando análisis: {exc}")
            logger.exception("Error en análisis %s", sport_name)
            st.exception(exc)

    st.markdown("---")
    if st.session_state.get("show_history", False):
        render_history(db, sport_name, settings)


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title=CONFIG.APP_NAME,
        page_icon=CONFIG.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    UIComponents.apply_theme()
    UIComponents.render_header()
    st.markdown("---")
    UIComponents.render_status_bar()

    # Sidebar needs the sport selected in the analysis tab — pass a sensible default
    settings = render_sidebar(db, "MLB")

    tab_analysis, tab_track = st.tabs(["Análisis", "Track Record"])

    with tab_analysis:
        _render_analysis_tab(settings)

    with tab_track:
        try:
            from track_record.ui import render_track_record
            render_track_record()
        except Exception as exc:
            st.error(f"Track Record no disponible: {exc}")
            logger.exception("Error cargando track record")

    UIComponents.render_footer()


if __name__ == "__main__":
    main()
