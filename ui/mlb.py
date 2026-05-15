"""
ui/mlb.py — MLB analyzer and result rendering.

MLBAnalyzer:
  - Looks up the MLB game_pk from the odds API game data
  - Calls modules/baseball_module/core/run_module.py
  - Saves positive-EV picks to PredictionsDB

render_mlb_results:
  - Displays λ progression, win probabilities, value cards, best_bets
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

import config as _cfg
from db.predictions_db import AnalysisResult, GameData, PredictionData, PredictionsDB
from ui.components import (
    CONFIG,
    SportConfig,
    ThemeColors,
    UIComponents,
    calculate_ev,
    calculate_kelly,
    calculate_rating,
)

logger = logging.getLogger(__name__)


# ── Abstract base (shared by MLB; NBA/UFC stubs live in app.py) ───────────


class BaseAnalyzer:
    def __init__(
        self,
        sport_config: SportConfig,
        settings: Dict[str, float],
        db: PredictionsDB,
    ) -> None:
        self.config   = sport_config
        self.settings = settings
        self.db       = db

    def analyze(self, game_data: GameData) -> AnalysisResult:
        raise NotImplementedError

    def save_value_picks(
        self,
        game_info: Dict[str, str],
        probabilities: Dict[str, float],
        odds: Dict[str, float],
        notes: str = "",
    ) -> int:
        """Evaluate ML home/away and save any positive-EV picks. Returns count saved."""
        min_ev       = self.settings.get("min_ev", CONFIG.DEFAULT_MIN_EV) / 100
        kelly_factor = self.settings.get("kelly_factor", CONFIG.DEFAULT_KELLY_FACTOR)

        home  = game_info["home"]
        away  = game_info["away"]
        p_home = probabilities["home_win"]
        p_away = probabilities["away_win"]

        saved = 0
        for side, team, p, o in [
            ("home", home, p_home, odds["home"]),
            ("away", away, p_away, odds["away"]),
        ]:
            ev = calculate_ev(o, p)
            if not np.isnan(ev) and ev > min_ev:
                self.db.save(PredictionData(
                    timestamp     = datetime.now().isoformat(),
                    sport         = self.config.name,
                    home_team     = home,
                    away_team     = away,
                    p_home        = p_home,
                    p_away        = p_away,
                    pick_type     = f"{team} ML",
                    pick_value    = str(o),
                    ev            = ev,
                    kelly         = calculate_kelly(o, p, kelly_factor),
                    confidence    = p,
                    rating        = calculate_rating(ev),
                    model_version = f"{self.config.name} {CONFIG.APP_VERSION}",
                    notes         = notes,
                ))
                saved += 1

        return saved


# ── MLB Analyzer ──────────────────────────────────────────────────────────


class MLBAnalyzer(BaseAnalyzer):

    @staticmethod
    def _fuzzy_match(a: str, b: str) -> bool:
        a, b = a.lower(), b.lower()
        return a in b or b in a or any(w in b for w in a.split())

    def find_game_id(self, game_data: GameData) -> Optional[int]:
        """Search MLB Stats API for the game_pk matching the odds API game."""
        try:
            from data_fetchers import MLBDataIntegrator
            integrator = MLBDataIntegrator()
            today    = datetime.now().strftime("%Y-%m-%d")
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            all_games = (integrator.mlb_api.get_todays_games(date=today) or []) + \
                        (integrator.mlb_api.get_todays_games(date=tomorrow) or [])
            for g in all_games:
                if self._fuzzy_match(game_data["home"], g.get("home_team", "")) and \
                   self._fuzzy_match(game_data["away"], g.get("away_team", "")):
                    return int(g.get("game_pk", 0))
            return None
        except Exception as exc:
            logger.error("Error finding game ID: %s", exc)
            return None

    def analyze(self, game_data: GameData) -> AnalysisResult:
        try:
            from modules.baseball_module.core.run_module import run_module

            with st.spinner("🔍 Buscando Game ID en MLB Stats API..."):
                game_id = self.find_game_id(game_data)
                if game_id:
                    st.success(f"✅ Game ID encontrado: {game_id}")
                else:
                    st.warning("⚠️ No se encontró Game ID — usando modo simulado")
                    game_id = CONFIG.MLB_FALLBACK_GAME_ID

            with st.spinner("⚡ Ejecutando análisis MLB G10 Ultra Pro..."):
                result = run_module(
                    game_id        = game_id,
                    use_calibration = True,
                    use_hfa        = True,
                    use_pitcher    = True,
                    analyze_f5     = True,
                    n_max          = self.config.simulations,
                )

            return result

        except ImportError as exc:
            return AnalysisResult(status="error", error=f"Módulo no disponible: {exc}")
        except Exception as exc:
            logger.exception("Error en análisis MLB")
            return AnalysisResult(status="error", error=str(exc))


# ── Result rendering ──────────────────────────────────────────────────────


def render_value_analysis(
    home: str,
    away: str,
    p_home: float,
    p_away: float,
    home_odds: float,
    away_odds: float,
    settings: Dict[str, float],
) -> None:
    min_ev       = settings.get("min_ev", CONFIG.DEFAULT_MIN_EV)
    kelly_factor = settings.get("kelly_factor", CONFIG.DEFAULT_KELLY_FACTOR)

    ev_home = calculate_ev(home_odds, p_home)
    ev_away = calculate_ev(away_odds, p_away)

    col1, col2 = st.columns(2)
    with col1:
        UIComponents.render_value_card(
            title=f"🏠 {home}",
            ev=ev_home,
            kelly=calculate_kelly(home_odds, p_home, kelly_factor),
            odds=home_odds,
            prob=p_home,
            min_ev=min_ev,
        )
    with col2:
        UIComponents.render_value_card(
            title=f"✈️ {away}",
            ev=ev_away,
            kelly=calculate_kelly(away_odds, p_away, kelly_factor),
            odds=away_odds,
            prob=p_away,
            min_ev=min_ev,
        )


def render_mlb_results(
    result: AnalysisResult,
    game_data: GameData,
    settings: Dict[str, float],
    db: PredictionsDB,
    sport_config: SportConfig,
) -> None:
    if result.get("status") != "success":
        if result.get("status") == "no_games":
            st.warning("⚠️ No hay juegos disponibles")
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        return

    st.success("✅ Análisis completado con éxito!")

    info = result.get("game_info", {})
    home = info.get("home_team", game_data["home"])
    away = info.get("away_team", game_data["away"])

    st.markdown(f"### 🏟️ {away} @ {home}")

    # Final λ — last pipeline stage present (umpire → pitcher → hfa → ...)
    _hist   = result.get("lambdas_history", {})
    lambdas = (_hist.get("umpire") or _hist.get("pitcher") or
               _hist.get("hfa")   or _hist.get("calibration") or
               _hist.get("base")  or {})
    lh = float(lambdas.get("lh", 0.0))
    la = float(lambdas.get("la", 0.0))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("λ Home Final", f"{lh:.3f}")
    with col2:
        st.metric("λ Away Final", f"{la:.3f}")

    st.markdown("---")

    probs  = result.get("probabilities", {})
    p_home = float(probs.get("p_home", probs.get("home_win", 0.0)))
    p_away = float(probs.get("p_away", probs.get("away_win", 0.0)))
    total  = float(probs.get("mean_total", probs.get("total_expected", lh + la)))

    st.markdown("### 📊 Probabilidades del Modelo")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"🏠 {home} Win", f"{p_home:.1%}", delta=f"{(p_home - 0.5)*100:+.1f}%")
    with c2:
        st.metric(f"✈️ {away} Win", f"{p_away:.1%}", delta=f"{(p_away - 0.5)*100:+.1f}%")
    with c3:
        st.metric("Total Runs", f"{total:.1f}")

    st.markdown("---")

    render_value_analysis(
        home=home, away=away,
        p_home=p_home, p_away=p_away,
        home_odds=game_data["home_odds"],
        away_odds=game_data["away_odds"],
        settings=settings,
    )

    # Save positive-EV picks
    analyzer    = MLBAnalyzer(sport_config, settings, db)
    picks_saved = analyzer.save_value_picks(
        game_info    = {"home": home, "away": away},
        probabilities = {"home_win": p_home, "away_win": p_away},
        odds         = {"home": game_data["home_odds"], "away": game_data["away_odds"]},
        notes        = f"λh={lh:.3f}, λa={la:.3f}",
    )

    if picks_saved:
        st.success(f"✅ {picks_saved} picks guardadas en base de datos")
    else:
        st.info("ℹ️ No se encontraron value bets que cumplan los criterios mínimos")

    # Best bets from model's value_detector
    best_bets = result.get("best_bets", [])
    if best_bets:
        st.markdown("---")
        st.markdown("### 🎯 Value Bets del Modelo")
        for i, bet in enumerate(best_bets[:5], 1):
            market = bet.get("market", "Unknown")
            ev     = bet.get("ev", 0.0)
            rating = bet.get("rating", "C")
            color  = (ThemeColors.SUCCESS.value
                      if rating in ("A+", "A") else ThemeColors.PRIMARY.value)
            st.markdown(
                f"""
                <div class='value-card' style='border-left:4px solid {color};'>
                    <b>#{i} {market}</b> [{rating}] — EV:
                    <b style='color:{color}'>{ev:+.1f}%</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("🔍 Detalles técnicos"):
        st.json(result.get("metadata", {}))
