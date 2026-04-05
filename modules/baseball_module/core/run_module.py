"""
RUN MODULE - BASEBALL MLB ANALYSIS G10 ULTRA PRO
=================================================

Pipeline completo de análisis MLB con arquitectura modular.

Autor: Braulio & Claude
Versión: G10 Ultra Pro + Selector de Partido + Juegos Hoy y Mañana
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# Data fetching
from data_fetchers import MLBStatsAPI

# Calibration - ABSOLUTO
from modules.baseball_module.calibration.auto_calibrator import AutoCalibrator

# HFA - ABSOLUTO
from modules.baseball_module.hfa.hfa_engine import get_adjusted_lambdas

# Pitcher - ABSOLUTO
from modules.baseball_module.context_engine.pitcher_engine import adjust_for_pitchers

# Pitcher Regression - ABSOLUTO
from modules.baseball_module.context_engine.pitchers_regression import calculate_pitcher_regression

# Monte Carlo - ABSOLUTO
from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced

# Value Detection - ABSOLUTO
from modules.baseball_module.value.value_detector import evaluate_value_ultra

# Odds API
try:
    from odds_api import get_odds
except ImportError:
    get_odds = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_module(
    game_id: Optional[int] = None,
    lh_base: float = 4.5,
    la_base: float = 4.2,
    use_calibration: bool = True,
    use_hfa: bool = True,
    use_pitcher: bool = True,
    use_regression: bool = True,
    analyze_f5: bool = True,
    n_max: int = 5_000_000
) -> Dict[str, Any]:
    """
    Ejecuta el análisis completo de un juego MLB.
    """

    logger.info("=" * 70)
    logger.info("🎯 INICIANDO ANÁLISIS MLB - SISTEMA G10 ULTRA PRO")
    logger.info("=" * 70)

    results = {
        'game_id': game_id,
        'lambdas_history': {},
        'metadata': {},
        'probabilities': {},
        'best_bets': [],
        'status': 'success'
    }

    try:
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 0: OBTENER DATOS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        logger.info("\n📊 PASO 0: Obteniendo datos del juego...")

        api = MLBStatsAPI()

        # ====================================================
        # NUEVA LÓGICA → BUSCA HOY Y MAÑANA + SELECTOR
        # ====================================================
        if game_id is None:
            logger.info("📅 Buscando juegos disponibles (hoy o mañana)...")

            today = datetime.now().date()
            tomorrow = today + timedelta(days=1)

            try:
                # Buscar juegos de hoy y de mañana
                games_today = api.get_todays_games() or []
                games_tomorrow = api.get_games_by_date(tomorrow.strftime("%Y-%m-%d")) or []
                games = games_today + games_tomorrow

                if not games:
                    logger.error("❌ No hay juegos disponibles hoy ni mañana")
                    results["status"] = "no_games"
                    return results

                logger.info(f"✅ {len(games)} juegos encontrados (hoy + mañana combinados)")

                # ===== Selector Streamlit =====
                try:
                    import streamlit as st

                    options = []
                    mapping = {}

                    for g in games:
                        home = g['teams']['home']['team']['name']
                        away = g['teams']['away']['team']['name']
                        date_str = datetime.fromisoformat(g['gameDate']).strftime("%Y-%m-%d %H:%M")
                        label = f"{away} @ {home} — {date_str}"
                        options.append(label)
                        mapping[label] = g['gamePk']

                    selected_label = st.selectbox(
                        "🎯 Selecciona el partido a analizar:",
                        options=options,
                        index=0
                    )
                    game_id = mapping[selected_label]
                    st.info(f"📊 Analizando: {selected_label}")
                    logger.info(f"   Juego seleccionado: {game_id}")

                except Exception:
                    # Si no hay Streamlit (modo terminal)
                    game_id = games[0]['gamePk']
                    logger.info(f"   Streamlit no disponible, usando primer juego: {game_id}")

            except Exception as e:
                logger.error(f"❌ Error obteniendo juegos: {e}")
                results['status'] = 'no_games'
                return results
        # ====================================================

        game_data = api.get_complete_game_data(game_id)

        if not game_data:
            logger.error(f"❌ No se pudo obtener datos del juego {game_id}")
            results['status'] = 'error_data'
            return results

        home_team = game_data.get('home_team', {}).get('name', 'Unknown')
        away_team = game_data.get('away_team', {}).get('name', 'Unknown')
        pitcher_home = game_data.get('pitcher_home', {}).get('name', 'Unknown')
        pitcher_away = game_data.get('pitcher_away', {}).get('name', 'Unknown')

        logger.info(f"   🏟️  {away_team} @ {home_team}")
        logger.info(f"   ⚾ Pitchers: {pitcher_away} vs {pitcher_home}")

        results['game_info'] = {
            'home_team': home_team,
            'away_team': away_team,
            'pitcher_home': pitcher_home,
            'pitcher_away': pitcher_away
        }

        lh = lh_base
        la = la_base

        results['lambdas_history']['base'] = {
            'lh': lh,
            'la': la
        }

        logger.info(f"   Lambda base: λ_h={lh:.3f}, λ_a={la:.3f}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 1: CALIBRATION ENGINE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if use_calibration:
            logger.info("\n🎯 PASO 1: Calibration Engine...")

            calibrator = AutoCalibrator()
            lh, la = calibrator.calibrate(lh, la, game_data)

            results['lambdas_history']['calibration'] = {
                'lh': lh,
                'la': la
            }

            logger.info(f"   ✅ Calibrated: λ_h={lh:.3f}, λ_a={la:.3f}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 2: HFA ENGINE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if use_hfa:
            logger.info("\n🏟️  PASO 2: HFA Engine (solo equipo)...")

            lh, la, hfa_meta = get_adjusted_lambdas(lh, la, game_data)

            results['lambdas_history']['hfa'] = {
                'lh': lh,
                'la': la
            }
            results['metadata']['hfa'] = hfa_meta

            logger.info(f"   ✅ HFA adjusted: λ_h={lh:.3f}, λ_a={la:.3f}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 3: PITCHER ENGINE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if use_pitcher:
            logger.info("\n⚾ PASO 3: Pitcher Engine (solo pitchers)...")

            lh, la, pitcher_meta = adjust_for_pitchers(lh, la, game_data)

            results['lambdas_history']['pitcher'] = {
                'lh': lh,
                'la': la
            }
            results['metadata']['pitcher'] = pitcher_meta

            logger.info(f"   ✅ Pitcher adjusted: λ_h={lh:.3f}, λ_a={la:.3f}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 4: PITCHER REGRESSION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if use_regression:
            logger.info("\n📈 PASO 4: Pitcher Regression...")

            factor_away, conf_away = calculate_pitcher_regression(
                pitcher_stats=game_data.get('pitcher_away', {}),
                opponent_stats=game_data.get('home_team', {})
            )

            factor_home, conf_home = calculate_pitcher_regression(
                pitcher_stats=game_data.get('pitcher_home', {}),
                opponent_stats=game_data.get('away_team', {})
            )

            lh = lh * factor_away
            la = la * factor_home

            results['lambdas_history']['regression'] = {
                'lh': lh,
                'la': la
            }
            results['metadata']['regression'] = {
                'factor_away': factor_away,
                'confidence_away': conf_away,
                'factor_home': factor_home,
                'confidence_home': conf_home
            }

            logger.info(f"   Pitcher Away regression: {factor_away:.3f} (conf: {conf_away:.2f})")
            logger.info(f"   Pitcher Home regression: {factor_home:.3f} (conf: {conf_home:.2f})")
            logger.info(f"   ✅ Final: λ_h={lh:.3f}, λ_a={la:.3f}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 5: MONTE CARLO SIMULATION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        logger.info(f"\n🎲 PASO 5: Monte Carlo ({n_max:,} simulaciones)...")

        mc_results = monte_carlo_advanced(
            lh=lh,
            la=la,
            n_max=n_max,
            analyze_f5=analyze_f5
        )

        results['probabilities'] = mc_results

        logger.info(f"   ✅ Simulaciones completadas")
        logger.info(f"   Home Win: {mc_results.get('home_win', 0):.1%}")
        logger.info(f"   Away Win: {mc_results.get('away_win', 0):.1%}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 6: VALUE DETECTION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        logger.info("\n💰 PASO 6: Value Detection...")

        market_odds = None
        if get_odds:
            try:
                market_odds = get_odds(game_id)
            except Exception as e:
                logger.warning(f"   ⚠️  No se pudieron obtener odds: {e}")

        if market_odds:
            value_results = evaluate_value_ultra(mc_results, market_odds)
            results['best_bets'] = value_results.get('best_bets', [])
            results['metadata']['value'] = value_results

            logger.info(f"   ✅ {len(results['best_bets'])} value bets encontradas")

            for i, bet in enumerate(results['best_bets'][:3], 1):
                logger.info(f"   {i}. {bet['market']}: EV={bet['ev']:+.1%}, Kelly={bet['kelly_pct']:.1%}")
        else:
            logger.warning("   ⚠️  Sin odds de mercado, saltando value detection")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # RESUMEN FINAL
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        logger.info("\n" + "=" * 70)
        logger.info("✅ ANÁLISIS COMPLETADO")
        logger.info("=" * 70)
        logger.info(f"Lambdas finales: λ_h={lh:.3f}, λ_a={la:.3f}")
        logger.info(f"Home Win: {mc_results.get('home_win', 0):.1%}")
        logger.info(f"Value Bets: {len(results['best_bets'])}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results['status'] = 'error'
        results['error'] = str(e)

    return results
