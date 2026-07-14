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
        best_bets: list,
        probabilities: Dict[str, float],
        notes: str = "",
    ) -> int:
        """Save positive-EV picks from the pipeline's value detector.

        Uses result['best_bets'] (all markets) rather than recomputing EV
        with only ML odds. Returns the number of rows saved.
        """
        min_ev_pct   = self.settings.get("min_ev", CONFIG.DEFAULT_MIN_EV)
        kelly_factor = self.settings.get("kelly_factor", CONFIG.DEFAULT_KELLY_FACTOR)

        home   = game_info["home"]
        away   = game_info["away"]
        p_home = probabilities.get("home_win", 0.5)
        p_away = probabilities.get("away_win", 0.5)

        saved = 0
        for bet in best_bets:
            ev_pct = float(bet.get("ev", 0.0))
            if ev_pct < min_ev_pct:
                continue

            market   = bet.get("market", "ML")
            side     = bet.get("side", "")
            kelly    = float(bet.get("kelly", 0.0))
            score    = float(bet.get("score", bet.get("composite_score", 0.0)))
            tier_grade = bet.get("tier_grade", "C")

            # Rescale composite score (0-100) to rating (0-10) for the history table
            rating = min(score / 10.0, 10.0)

            self.db.save(PredictionData(
                timestamp     = datetime.now().isoformat(),
                sport         = self.config.name,
                home_team     = home,
                away_team     = away,
                p_home        = p_home,
                p_away        = p_away,
                pick_type     = f"{market} {side}".strip(),
                pick_value    = tier_grade,
                ev            = ev_pct / 100.0,
                kelly         = kelly * (kelly_factor / CONFIG.DEFAULT_KELLY_FACTOR),
                confidence    = float(bet.get("confidence", 0.5)),
                rating        = rating,
                model_version = f"{self.config.name} {CONFIG.APP_VERSION}",
                notes         = notes,
                odds          = bet.get("odds"),
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

            # Build market_odds from the selector's GameData so run_module()
            # doesn't need a second API call (and won't miss on fuzzy-match).
            market_odds = None
            if game_data.get("home_odds") and game_data.get("away_odds"):
                market_odds = {
                    "ml_home":      game_data.get("home_odds"),
                    "ml_away":      game_data.get("away_odds"),
                    "pin_home":     game_data.get("pin_home"),
                    "pin_away":     game_data.get("pin_away"),
                    "total_line":   game_data.get("total_line"),
                    "total_over":   game_data.get("total_over"),
                    "total_under":  game_data.get("total_under"),
                    "runline_home": game_data.get("runline_home"),
                    "runline_away": game_data.get("runline_away"),
                    "runline_line": game_data.get("runline_line"),
                    # F5 — added 2026-07-06 alongside GameData; previously
                    # missing entirely, so the UI-selector-driven analysis
                    # path could never surface F5 markets regardless of the
                    # separate get_best_odds_for_teams() naming fix.
                    "f5_ml_home":    game_data.get("f5_ml_home"),
                    "f5_ml_away":    game_data.get("f5_ml_away"),
                    "f5_total_line": game_data.get("f5_total_line"),
                    "f5_total_over": game_data.get("f5_total_over"),
                    "f5_total_under": game_data.get("f5_total_under"),
                }

            with st.spinner("⚡ Ejecutando análisis MLB G10 Ultra Pro..."):
                result = run_module(
                    game_id         = game_id,
                    use_hfa         = True,
                    use_pitcher     = True,
                    analyze_f5      = True,
                    n_max           = self.config.simulations,
                    market_odds     = market_odds,
                )

            return result

        except ImportError as exc:
            logger.exception("Módulo MLB no disponible")
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
    save_picks: bool = True,
) -> None:
    """Render an MLB analysis result.

    save_picks — write best_bets to PredictionsDB. Must be False when
    re-rendering a result already stored in st.session_state (e.g. on a
    Streamlit rerun triggered by an unrelated widget) — otherwise the same
    picks get re-inserted on every rerun. Only the caller that just computed
    a fresh `result` should pass True.
    """
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

    # Final λ — run_module.py now stamps the exact (lh, la) fed to Monte
    # Carlo under 'final'. Fallback chain (real reverse-pipeline order:
    # HFA PASO7 -> Defense PASO6 -> Park+Weather PASO5 -> Bullpen PASO4 ->
    # Contextual PASO3 -> Pitcher PASO2 -> base) only matters for results
    # computed before 'final' existed, or a run with some stages disabled.
    # Previously checked 'contextual' (PASO 3) FIRST, so it always won the
    # or-chain and every past "λ Final" display was really the PASO-3
    # value, ~1-3% off from the true final — found + fixed 2026-07-06.
    _hist   = result.get("lambdas_history", {})
    lambdas = (
        _hist.get("final")        or _hist.get("hfa") or
        _hist.get("defense")      or _hist.get("park_weather") or
        _hist.get("bullpen")      or _hist.get("contextual") or
        _hist.get("pitcher")      or _hist.get("kalman_offense") or {}
    )
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

    # Show ML EV cards using the odds from the selector (quick visual reference).
    # Skip entirely rather than fabricate even-money odds — a made-up 2.0
    # price would render an EV/Kelly card indistinguishable from a real one.
    if game_data.get("home_odds") and game_data.get("away_odds"):
        render_value_analysis(
            home=home, away=away,
            p_home=p_home, p_away=p_away,
            home_odds=game_data["home_odds"],
            away_odds=game_data["away_odds"],
            settings=settings,
        )
    else:
        st.caption("💵 Odds de mercado no disponibles para la tarjeta rápida de EV.")

    # Best bets from the pipeline's full value detector (all markets)
    best_bets = result.get("best_bets", [])

    # Save positive-EV picks driven by the pipeline's value detector — only
    # on the run that actually computed this result, never on a re-render.
    if save_picks:
        analyzer    = MLBAnalyzer(sport_config, settings, db)
        picks_saved = analyzer.save_value_picks(
            game_info     = {"home": home, "away": away},
            best_bets     = best_bets,
            probabilities = {"home_win": p_home, "away_win": p_away},
            notes         = f"λh={lh:.3f}, λa={la:.3f}",
        )

        if picks_saved:
            st.success(f"✅ {picks_saved} picks guardadas en base de datos")
        else:
            st.info("ℹ️ No se encontraron value bets que cumplan los criterios mínimos")

    if best_bets:
        st.markdown("---")
        st.markdown("### 🎯 Value Bets del Modelo")

        # Show Pinnacle reference if available
        _mkt_odds = result.get("metadata", {}).get("market_odds", {})
        if _mkt_odds.get("pin_home") and _mkt_odds.get("pin_away"):
            st.caption(
                f"📌 Referencia Pinnacle: {home} {_mkt_odds['pin_home']} / "
                f"{away} {_mkt_odds['pin_away']}"
            )

        for i, bet in enumerate(best_bets[:5], 1):
            market = bet.get("market", "Unknown")
            side   = bet.get("side", "")
            ev     = bet.get("ev", 0.0)
            grade  = bet.get("tier_grade", "C")
            kelly  = bet.get("kelly", 0.0)
            # S/A = high value (green), B/C = moderate (blue)
            color  = (ThemeColors.SUCCESS.value if grade in ("S", "A")
                      else ThemeColors.PRIMARY.value)
            st.markdown(
                f"""
                <div class='value-card' style='border-left:4px solid {color};'>
                    <b>#{i} {market} — {side}</b> [{grade}] — EV:
                    <b style='color:{color}'>{ev:+.1f}%</b>
                    &nbsp;·&nbsp; Kelly: {kelly*100:.1f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("🔍 Detalles técnicos"):
        st.json(result.get("metadata", {}))
