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
from config import MLB_SIMULATIONS, LEAGUE_AVG_RUNS, LEAGUE_AVG_WHIP

# Calibration - ABSOLUTO
from modules.baseball_module.calibration.auto_calibrator import LambdaCalibrator
from modules.baseball_module.calibration.learning_engine import LearningEngine

# HFA - ABSOLUTO
from modules.baseball_module.hfa.hfa_engine import get_adjusted_lambdas

# Pitcher - ABSOLUTO
from modules.baseball_module.context_engine.pitcher_engine import adjust_for_pitchers

# Pitcher Regression - ABSOLUTO
from modules.baseball_module.context_engine.pitchers_regression import calculate_pitcher_regression

# Monte Carlo - ABSOLUTO
from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced

# Value Detection - ABSOLUTO
from core.value_detector import evaluate_value_ultra

# Odds API
try:
    from odds_api import get_best_odds_for_teams
except ImportError:
    get_odds = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _compute_f5_lambda(pitcher_stats: Dict, bullpen_era: float = 4.20) -> Optional[float]:
    """
    Compute expected runs against this pitcher in the first 5 innings.

    Uses the pitcher's real byInning F5 ERA plus average IPS to determine
    how many of those 5 innings the starter vs the bullpen will cover.
    Returns None when avg_innings_per_start is unavailable (caller falls
    back to the F5_SCALE constant inside the simulator).
    """
    avg_ips = pitcher_stats.get('avg_innings_per_start')
    if avg_ips is None:
        return None
    f5_era = pitcher_stats.get('f5_era') or pitcher_stats.get('era', 4.20)
    starter_f5_ip = min(float(avg_ips), 5.0)
    bullpen_f5_ip = 5.0 - starter_f5_ip
    return round(starter_f5_ip * (f5_era / 9.0) + bullpen_f5_ip * (bullpen_era / 9.0), 3)


def run_module(
    game_id: Optional[int] = None,
    lh_base: float = LEAGUE_AVG_RUNS,
    la_base: float = 4.2,
    use_calibration: bool = True,
    use_hfa: bool = True,
    use_pitcher: bool = True,
    use_regression: bool = True,
    analyze_f5: bool = True,
    n_max: int = MLB_SIMULATIONS
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

    # Learning engine — initialised once per run; fetches pending outcomes first
    from config import DATA_DIR
    _learning = LearningEngine(db_path=DATA_DIR / "predictions_history.db")
    try:
        _learning.fetch_pending_outcomes()
    except Exception:
        pass  # never block analysis on learning failures

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
                        home = g['home_team']
                        away = g['away_team']
                        date_str = datetime.fromisoformat(g['game_date'].replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M")
                        label = f"{away} @ {home} — {date_str}"
                        options.append(label)
                        mapping[label] = g['game_pk']

                    selected_label = st.selectbox(
                        "🎯 Selecciona el partido a analizar:",
                        options=options,
                        index=0
                    )
                    game_id = mapping[selected_label]
                    st.info(f"📊 Analizando: {selected_label}")
                    logger.info(f"   Juego seleccionado: {game_id}")

                except ImportError:
                    # Terminal mode — no Streamlit
                    game_id = games[0]['game_pk']
                    logger.info(f"   Streamlit no disponible, usando primer juego: {game_id}")

            except Exception as e:
                logger.error(f"❌ Error obteniendo juegos: {e}")
                results['status'] = 'no_games'
                return results
        # ====================================================

        from data_fetchers import MLBDataIntegrator
        _integrator = MLBDataIntegrator()
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        all_games = (_integrator.get_complete_game_data(date=today) or []) + \
                    (_integrator.get_complete_game_data(date=tomorrow) or [])
        game_data = next((g for g in all_games if g.get('game_pk') == game_id), None)
        if not game_data:
            logger.error(f"❌ No se pudo obtener datos del juego {game_id}")
            results['status'] = 'error_data'
            return results

        home_team = game_data.get('home_team', 'Unknown')
        away_team = game_data.get('away_team', 'Unknown')
        pitcher_home = game_data.get('home_pitcher', 'Unknown')
        pitcher_away = game_data.get('away_pitcher', 'Unknown')

        logger.info(f"   🏟️  {away_team} @ {home_team}")
        logger.info(f"   ⚾ Pitchers: {pitcher_away} vs {pitcher_home}")

        results['game_info'] = {
            'home_team': home_team,
            'away_team': away_team,
            'pitcher_home': pitcher_home,
            'pitcher_away': pitcher_away
        }
# Normalizar game_data para los engines
        _home_off   = game_data.get('home_offensive_stats') or {}
        _away_off   = game_data.get('away_offensive_stats') or {}
        _home_pitch = game_data.get('home_pitching_stats') or {}
        _away_pitch = game_data.get('away_pitching_stats') or {}
        game_data['home_team'] = {
            'name': home_team,
            'runs_per_game': game_data.get('home_team_runs', {}).get('runs_scored_avg', LEAGUE_AVG_RUNS) if isinstance(game_data.get('home_team_runs'), dict) else LEAGUE_AVG_RUNS,
            'wins': game_data.get('home_team_form', {}).get('wins', 5) if isinstance(game_data.get('home_team_form'), dict) else 5,
            'losses': game_data.get('home_team_form', {}).get('losses', 5) if isinstance(game_data.get('home_team_form'), dict) else 5,
            'last_10': game_data.get('home_team_form', {}).get('last_10', '5-5') if isinstance(game_data.get('home_team_form'), dict) else '5-5',
            'streak': game_data.get('home_team_form', {}).get('streak', '') if isinstance(game_data.get('home_team_form'), dict) else '',
            # Offense
            'woba': _home_off.get('woba', 0.320),
            'ops': _home_off.get('ops', 0.735),
            'wrc_plus': _home_off.get('wrc_plus', 100.0),
            # Defense (used by calibrator's _calculate_defense_multiplier on the opponent)
            'runs_allowed_per_game': _home_pitch.get('runs_allowed_per_game', LEAGUE_AVG_RUNS),
            'team_era': _home_pitch.get('team_era', 4.15),
            'team_whip': _home_pitch.get('team_whip', 1.30),
            # Rest
            'rest_days': game_data.get('home_days_rest', 1),
        }
        game_data['away_team'] = {
            'name': away_team,
            'runs_per_game': game_data.get('away_team_runs', {}).get('runs_scored_avg', LEAGUE_AVG_RUNS) if isinstance(game_data.get('away_team_runs'), dict) else LEAGUE_AVG_RUNS,
            'wins': game_data.get('away_team_form', {}).get('wins', 5) if isinstance(game_data.get('away_team_form'), dict) else 5,
            'losses': game_data.get('away_team_form', {}).get('losses', 5) if isinstance(game_data.get('away_team_form'), dict) else 5,
            'last_10': game_data.get('away_team_form', {}).get('last_10', '5-5') if isinstance(game_data.get('away_team_form'), dict) else '5-5',
            'streak': game_data.get('away_team_form', {}).get('streak', '') if isinstance(game_data.get('away_team_form'), dict) else '',
            # Offense
            'woba': _away_off.get('woba', 0.320),
            'ops': _away_off.get('ops', 0.735),
            'wrc_plus': _away_off.get('wrc_plus', 100.0),
            # Defense
            'runs_allowed_per_game': _away_pitch.get('runs_allowed_per_game', LEAGUE_AVG_RUNS),
            'team_era': _away_pitch.get('team_era', 4.15),
            'team_whip': _away_pitch.get('team_whip', 1.30),
            # Rest + travel
            'rest_days': game_data.get('away_days_rest', 1),
            'miles_traveled': game_data.get('miles_traveled_away', 0),
            'time_zones_crossed': game_data.get('time_zones_crossed_away', 0),
        }
        # Top-level travel keys used by HFA engine and pitcher engine
        game_data.setdefault('miles_traveled_away', 0)
        game_data.setdefault('time_zones_crossed_away', 0)
        game_data.setdefault('back_to_back_away', False)
        home_ps = game_data.get('home_pitcher_stats', {}) if isinstance(game_data.get('home_pitcher_stats'), dict) else {}
        away_ps = game_data.get('away_pitcher_stats', {}) if isinstance(game_data.get('away_pitcher_stats'), dict) else {}

        game_data['pitcher_home'] = {
            'name': pitcher_home,
            'era': home_ps.get('era', 4.38),
            'fip': home_ps.get('fip', home_ps.get('era', 4.38)),
            'whip': home_ps.get('whip', LEAGUE_AVG_WHIP),
            'k_per_9': home_ps.get('k_per_9', 8.5),
            'era_last_5': home_ps.get('era_last_5', home_ps.get('era', 4.38)),
            'days_rest': home_ps.get('days_rest', 4),
            'last_pitch_count': home_ps.get('last_pitch_count', 90),
            'avg_innings_per_start': home_ps.get('avg_innings_per_start'),
            'f5_era': home_ps.get('f5_era'),
        }
        game_data['pitcher_away'] = {
            'name': pitcher_away,
            'era': away_ps.get('era', 4.38),
            'fip': away_ps.get('fip', away_ps.get('era', 4.38)),
            'whip': away_ps.get('whip', LEAGUE_AVG_WHIP),
            'k_per_9': away_ps.get('k_per_9', 8.5),
            'era_last_5': away_ps.get('era_last_5', away_ps.get('era', 4.38)),
            'days_rest': away_ps.get('days_rest', 4),
            'last_pitch_count': away_ps.get('last_pitch_count', 90),
            'avg_innings_per_start': away_ps.get('avg_innings_per_start'),
            'f5_era': away_ps.get('f5_era'),
        }
        game_data['park'] = {'name': game_data.get('venue', 'Unknown')}
        # Lambdas base con media ponderada por equipo
        from data_fetchers import MLBDataIntegrator as _Int
        _int = _Int()
        home_recent = game_data.get('home_team_runs', {})
        away_recent = game_data.get('away_team_runs', {})
        home_rpg = float(home_recent.get('runs_scored_avg', 0)) if isinstance(home_recent, dict) else 0
        away_rpg = float(away_recent.get('runs_scored_avg', 0)) if isinstance(away_recent, dict) else 0
        home_team_id = game_data.get('home_team_id')
        away_team_id = game_data.get('away_team_id')
        lh = _int.get_team_lambda(home_team, home_rpg, team_id=home_team_id)
        la = _int.get_team_lambda(away_team, away_rpg, team_id=away_team_id)
        results['lambdas_history']['base'] = {'lh': lh, 'la': la}
        logger.info(f"   Lambda base: λ_h={lh:.3f} ({home_team}), λ_a={la:.3f} ({away_team})")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 1: CALIBRATION ENGINE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if use_calibration:
            logger.info("\n🎯 PASO 1: Calibration Engine...")

            calibrator = LambdaCalibrator(learning_engine=_learning)
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

        # Real F5 lambdas: home runs depend on away pitcher; away runs on home pitcher.
        # Bullpen ERA fills innings not covered by the starter (avg_ips < 5).
        _home_bp_era = game_data.get('bullpen_home', {}).get('era', 4.20)
        _away_bp_era = game_data.get('bullpen_away', {}).get('era', 4.20)
        lh_f5 = _compute_f5_lambda(game_data.get('pitcher_away', {}), _away_bp_era)
        la_f5 = _compute_f5_lambda(game_data.get('pitcher_home', {}), _home_bp_era)
        if lh_f5 is not None:
            logger.info(f"   F5 λ_h={lh_f5:.3f} (away pitcher avg_IPS={game_data['pitcher_away'].get('avg_innings_per_start','?')} f5_ERA={game_data['pitcher_away'].get('f5_era','?')})")
        if la_f5 is not None:
            logger.info(f"   F5 λ_a={la_f5:.3f} (home pitcher avg_IPS={game_data['pitcher_home'].get('avg_innings_per_start','?')} f5_ERA={game_data['pitcher_home'].get('f5_era','?')})")

        mc_results = monte_carlo_advanced(
            lh=lh,
            la=la,
            n_max=n_max,
            analyze_f5=analyze_f5,
            lh_f5=lh_f5,
            la_f5=la_f5,
        )

        results['probabilities'] = mc_results

        logger.info(f"   ✅ Simulaciones completadas")
        logger.info(f"   Home Win: {mc_results.get('p_home', 0):.1%}")
        logger.info(f"   Away Win: {mc_results.get('p_away', 0):.1%}")

        # Persist prediction for future learning
        _gk = game_data.get('game_pk') or game_id
        _gd = str(game_data.get('game_date', datetime.now().strftime("%Y-%m-%d")))[:10]
        if _gk:
            try:
                _learning.record_prediction(
                    game_pk=int(_gk),
                    game_date=_gd,
                    season=datetime.now().year,
                    home_team=home_team if isinstance(home_team, str) else home_team.get('name', ''),
                    away_team=away_team if isinstance(away_team, str) else away_team.get('name', ''),
                    lambda_home=lh,
                    lambda_away=la,
                    p_home=mc_results.get('p_home', 0.5),
                    p_away=mc_results.get('p_away', 0.5),
                )
            except Exception:
                pass

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 6: VALUE DETECTION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        logger.info("\n💰 PASO 6: Value Detection...")

        market_odds = None
        try:
            market_odds = get_best_odds_for_teams(
                home_team=home_team,
                away_team=away_team,
                sport="baseball_mlb"
            )
        except Exception as e:
            logger.warning(f"   ⚠️  No se pudieron obtener odds: {e}")

        if market_odds and market_odds.get('ml_home') and market_odds.get('ml_away'):
            from core.value_detector import GameOdds
            game_odds = GameOdds(
                ml_home=market_odds['ml_home'],
                ml_away=market_odds['ml_away'],
                pin_home=market_odds.get('pin_home'),
                pin_away=market_odds.get('pin_away'),
            )
            value_results = evaluate_value_ultra(
                mc_result=mc_results,
                odds=game_odds,
                lh=lh,
                la=la,
                home_samples=mc_results.get('home_samples'),
                away_samples=mc_results.get('away_samples'),
                total_samples=mc_results.get('total_samples'),
                analyze_f5=False,
            )
            results['best_bets'] = value_results.get('global_recommendation', {}).get('all_opportunities', [])
            results['metadata']['value'] = value_results
            logger.info(f"   ✅ Value detection completado")
        else:
            logger.warning("   ⚠️  Sin odds de mercado disponibles")
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # RESUMEN FINAL
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        logger.info("\n" + "=" * 70)
        logger.info("✅ ANÁLISIS COMPLETADO")
        logger.info("=" * 70)
        logger.info(f"Lambdas finales: λ_h={lh:.3f}, λ_a={la:.3f}")
        logger.info(f"Home Win: {mc_results.get('p_home', 0):.1%}")
        logger.info(f"Value Bets: {len(results['best_bets'])}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results['status'] = 'error'
        results['error'] = str(e)

    return results
