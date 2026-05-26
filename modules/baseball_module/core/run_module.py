"""
RUN MODULE - BASEBALL MLB ANALYSIS G10 ULTRA PRO
=================================================

Pipeline completo de análisis MLB con arquitectura modular.

Autor: Braulio & Claude
Versión: G10 Ultra Pro + Selector de Partido + Juegos Hoy y Mañana
"""

import logging
from math import log as _log, exp as _exp
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Data fetching
from data_fetchers import MLBStatsAPI, MLBDataIntegrator, _current_mlb_season
from config import MLB_SIMULATIONS, LEAGUE_AVG_RUNS, LEAGUE_AVG_WHIP, LEAGUE_AVG_ERA

from modules.baseball_module.calibration.learning_engine import LearningEngine

# Park + Weather (symmetric) - ABSOLUTO
from modules.baseball_module.hfa.park_weather_engine import adjust_for_park_and_weather

# HFA (asymmetric: crowd + travel) - ABSOLUTO
from modules.baseball_module.hfa.hfa_engine import get_adjusted_lambdas

# Pitcher - ABSOLUTO
from modules.baseball_module.context_engine.pitcher_engine import adjust_for_pitchers

# Bullpen - ABSOLUTO
from modules.baseball_module.context_engine.bullpen_engine import adjust_for_bullpen

# Contextual (rest/B2B + umpire) - ABSOLUTO
from modules.baseball_module.context_engine.contextual_engine import adjust_for_context

# Defensive Efficiency - ABSOLUTO
from modules.baseball_module.context_engine.defensive_efficiency_engine import adjust_for_defense

# Monte Carlo - ABSOLUTO
from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced, F5_SCALE

# Value Detection - ABSOLUTO
from core.value_detector import evaluate_value_ultra

# External enrichment (Savant + FanGraphs) — optional; pipeline continues without them
try:
    from modules.baseball_module.data_enrichment.savant_fetcher import SavantFetcher as _SavantFetcher
    from modules.baseball_module.data_enrichment.fangraphs_fetcher import FanGraphsFetcher as _FGFetcher
    _ENRICHMENT_AVAILABLE = True
except ImportError:
    _ENRICHMENT_AVAILABLE = False

# True Talent Offense Engine — replaces get_team_lambda()
try:
    from modules.baseball_module.offense.true_talent_engine import get_true_talent_lambda as _get_tte_lambda
    _TTE_AVAILABLE = True
except ImportError:
    _TTE_AVAILABLE = False

# Odds — best prices lookup lives in odds_fetcher (same cache, no extra API call)
try:
    from odds_fetcher import get_best_odds_for_teams
except Exception:
    get_best_odds_for_teams = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PLATT_A_FALLBACK = 0.547
_PLATT_B_FALLBACK = 0.098


def _platt(p: float, a: float = _PLATT_A_FALLBACK, b: float = _PLATT_B_FALLBACK) -> float:
    """Shrink an over-confident probability using Platt logistic scaling."""
    p = max(0.01, min(p, 0.99))
    logit = _log(p / (1.0 - p))
    return 1.0 / (1.0 + _exp(-(a * logit + b)))


def _compute_lambda_noise(tte_active: bool, enrichment_available: bool, has_real_pitcher: bool) -> float:
    """
    Epistemic uncertainty for the MC noise parameter.
    More data sources → smaller noise (λ is better estimated).
    """
    if tte_active and enrichment_available and has_real_pitcher:
        return 0.04  # full stack: TTE + Savant/FG + real pitcher stats
    if tte_active or enrichment_available:
        return 0.06  # partial data
    return 0.08      # legacy fallback only



def run_module(
    game_id: Optional[int] = None,
    lh_base: float = LEAGUE_AVG_RUNS,  # unused — lambda comes from get_team_lambda()
    la_base: float = LEAGUE_AVG_RUNS,  # unused — lambda comes from get_team_lambda()
    use_hfa: bool = True,
    use_pitcher: bool = True,
    analyze_f5: bool = True,
    n_max: int = MLB_SIMULATIONS,
    market_odds: Optional[Dict] = None,  # pre-fetched odds from UI — skips second API call
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

        _integrator = MLBDataIntegrator()
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        all_games = (_integrator.get_complete_game_data(date=today, game_pk=game_id) or []) + \
                    (_integrator.get_complete_game_data(date=tomorrow, game_pk=game_id) or [])
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
            # Defense — used by DefensiveEfficiencyEngine (PASO 4)
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

        # Use team staff ERA as fallback so we never silently inject 4.38.
        _home_team_era  = _home_pitch.get('team_era',  LEAGUE_AVG_ERA)  if _home_pitch else LEAGUE_AVG_ERA
        _home_team_whip = _home_pitch.get('team_whip', LEAGUE_AVG_WHIP) if _home_pitch else LEAGUE_AVG_WHIP
        _away_team_era  = _away_pitch.get('team_era',  LEAGUE_AVG_ERA)  if _away_pitch else LEAGUE_AVG_ERA
        _away_team_whip = _away_pitch.get('team_whip', LEAGUE_AVG_WHIP) if _away_pitch else LEAGUE_AVG_WHIP

        _home_era = home_ps.get('era', _home_team_era)
        _away_era = away_ps.get('era', _away_team_era)

        game_data['pitcher_home'] = {
            'name':                  pitcher_home,
            'era':                   _home_era,
            'fip':                   home_ps.get('fip', _home_era),
            'whip':                  home_ps.get('whip', _home_team_whip),
            'k_per_9':               home_ps.get('k_per_9', 8.5),
            'era_last_5':            home_ps.get('era_last_5', _home_era),
            'era_trend':             home_ps.get('era_trend'),
            'k9_trend':              home_ps.get('k9_trend'),
            'quality_start_pct':     home_ps.get('quality_start_pct'),
            'days_rest':             home_ps.get('days_rest', 4),
            'last_pitch_count':      home_ps.get('last_pitch_count', 90),
            'avg_innings_per_start': home_ps.get('avg_innings_per_start'),
            'f5_era':                home_ps.get('f5_era'),
            'era_vs_opp':            home_ps.get('era_vs_opp'),
            'whip_vs_opp':           home_ps.get('whip_vs_opp'),
            'ip_vs_opp':             home_ps.get('ip_vs_opp'),
            # Platoon splits + opposing lineup handedness
            'platoon_splits':        home_ps.get('platoon_splits'),
        }
        game_data['pitcher_away'] = {
            'name':                  pitcher_away,
            'era':                   _away_era,
            'fip':                   away_ps.get('fip', _away_era),
            'whip':                  away_ps.get('whip', _away_team_whip),
            'k_per_9':               away_ps.get('k_per_9', 8.5),
            'era_last_5':            away_ps.get('era_last_5', _away_era),
            'era_trend':             away_ps.get('era_trend'),
            'k9_trend':              away_ps.get('k9_trend'),
            'quality_start_pct':     away_ps.get('quality_start_pct'),
            'days_rest':             away_ps.get('days_rest', 4),
            'last_pitch_count':      away_ps.get('last_pitch_count', 90),
            'avg_innings_per_start': away_ps.get('avg_innings_per_start'),
            'f5_era':                away_ps.get('f5_era'),
            'era_vs_opp':            away_ps.get('era_vs_opp'),
            'whip_vs_opp':           away_ps.get('whip_vs_opp'),
            'ip_vs_opp':             away_ps.get('ip_vs_opp'),
            # Platoon splits + opposing lineup handedness
            'platoon_splits':        away_ps.get('platoon_splits'),
        }
        # Propagate lineup LHB% to top-level game_data for pitcher engine
        game_data.setdefault('home_lineup_lhb_pct', game_data.get('home_lineup_lhb_pct', 0.45))
        game_data.setdefault('away_lineup_lhb_pct', game_data.get('away_lineup_lhb_pct', 0.45))

        # Resolve MLB season once — used for enrichment, Kalman, Platt, and record_prediction
        _season = _current_mlb_season()
        game_data['season'] = _season

        # ── Enrich pitchers with Savant + FanGraphs real data ────────────
        if _ENRICHMENT_AVAILABLE:
            try:
                from config import DATA_DIR as _DATA_DIR
                _cache = _DATA_DIR / ".cache"
                _savant = _SavantFetcher(cache_dir=_cache)
                _fg     = _FGFetcher(cache_dir=_cache)

                _sv_all = _savant.get_all_pitcher_stats(_season)
                _fg_all = _fg.get_all_pitcher_stats(_season)

                for _role, _pid_key in [("pitcher_home", "home_pitcher_id"),
                                        ("pitcher_away", "away_pitcher_id")]:
                    _mlbam = game_data.get(_pid_key)
                    if not _mlbam:
                        continue
                    _mlbam = int(_mlbam)
                    _sv = _sv_all.get(_mlbam, {})
                    _fg_d = _fg_all.get(_mlbam, {})
                    _enriched = {
                        # FanGraphs: real ERA estimators
                        "xfip":            _fg_d.get("xfip"),
                        "siera":           _fg_d.get("siera"),
                        "war":             _fg_d.get("war"),
                        "k_pct":           _fg_d.get("k_pct"),
                        "bb_pct":          _fg_d.get("bb_pct"),
                        "swstr_pct":       _fg_d.get("swstr_pct"),
                        # Luck indicators for pitchers_regression
                        "babip":           _fg_d.get("babip"),
                        "lob_pct":         _fg_d.get("lob_pct"),
                        "hr_fb_pct":       _fg_d.get("hr_fb"),
                        "innings_pitched": _fg_d.get("ip"),
                        # Baseball Savant: contact quality
                        "est_woba":        _sv.get("est_woba"),
                        "xera":            _sv.get("xera") or _fg_d.get("xera"),
                        "brl_percent":     _sv.get("brl_percent"),
                        "avg_hit_speed":   _sv.get("avg_hit_speed"),
                        "ev95percent":     _sv.get("ev95percent"),
                    }
                    # Only overwrite existing values when we have real data — never
                    # replace valid MLB Stats API values with None.
                    game_data[_role].update({k: v for k, v in _enriched.items() if v is not None})
                    logger.debug(
                        f"   [enrichment] {_role}: "
                        f"xFIP={game_data[_role].get('xfip')}, "
                        f"SIERA={game_data[_role].get('siera')}, "
                        f"xwOBA={game_data[_role].get('est_woba')}, "
                        f"Brl%={game_data[_role].get('brl_percent')}"
                    )
            except Exception as _e:
                logger.warning(f"   [enrichment] Savant/FG enrichment failed: {_e}")

        game_data['park'] = {'name': game_data.get('venue', 'Unknown')}
        # Lambdas base con media ponderada por equipo
        home_recent = game_data.get('home_team_runs', {})
        away_recent = game_data.get('away_team_runs', {})
        home_rpg = float(home_recent.get('runs_scored_avg', 0)) if isinstance(home_recent, dict) else 0
        away_rpg = float(away_recent.get('runs_scored_avg', 0)) if isinstance(away_recent, dict) else 0
        home_team_id = game_data.get('home_team_id')
        away_team_id = game_data.get('away_team_id')
        # ── Day-of lineup (fetched before TTE so confirmed batters are used) ──
        _lineup_home: list = []
        _lineup_away: list = []
        if _TTE_AVAILABLE and game_id:
            try:
                from modules.baseball_module.offense.true_talent_engine import _fetch_game_lineup
                _lineups = _fetch_game_lineup(game_id)
                _lineup_home = _lineups.get("home", [])
                _lineup_away = _lineups.get("away", [])
                if len(_lineup_home) >= 9:
                    logger.info(f"   Lineup confirmed: {len(_lineup_home)} home / {len(_lineup_away)} away batters")
                else:
                    logger.info("   Lineup not yet posted — TTE will use gameday roster")
            except Exception as _le:
                logger.debug(f"   Lineup fetch skipped: {_le}")

        # ── True Talent Offense Engine — park-neutral λ_base ─────────────
        _tte_active = False
        if _TTE_AVAILABLE and home_team_id and away_team_id:
            try:
                lh, _tte_home_meta = _get_tte_lambda(
                    home_team_id, home_team, _season,
                    lineup_ids=_lineup_home if len(_lineup_home) >= 9 else None,
                )
                la, _tte_away_meta = _get_tte_lambda(
                    away_team_id, away_team, _season,
                    lineup_ids=_lineup_away if len(_lineup_away) >= 9 else None,
                )
                results['metadata']['tte_home'] = _tte_home_meta
                results['metadata']['tte_away'] = _tte_away_meta
                _tte_active = True
                _lineup_tag = "lineup" if _tte_home_meta.get("lineup_confirmed") else "roster"
                logger.info(
                    f"   TTE λ_base [{_lineup_tag}]: {home_team}={lh:.3f} "
                    f"(xwOBA={_tte_home_meta['metrics']['xwoba_regressed']:.3f} "
                    f"wRC+={_tte_home_meta['metrics']['wrc_plus_approx']:.0f}) | "
                    f"{away_team}={la:.3f} "
                    f"(xwOBA={_tte_away_meta['metrics']['xwoba_regressed']:.3f} "
                    f"wRC+={_tte_away_meta['metrics']['wrc_plus_approx']:.0f})"
                )
            except Exception as _tte_err:
                logger.warning(f"   TTE failed ({_tte_err}), falling back to legacy λ")

        if not _tte_active:
            # Fallback: TTE unavailable, missing team IDs, or runtime error
            lh = _integrator.get_team_lambda(home_team, home_rpg, team_id=home_team_id)
            la = _integrator.get_team_lambda(away_team, away_rpg, team_id=away_team_id)
            logger.info(f"   λ_base (legacy): λ_h={lh:.3f}  λ_a={la:.3f}")

        # ── Kalman-adjusted base lambdas ──────────────────────────────────
        # Offensive Kalman: pulls each team's λ toward its observed run-scoring rate.
        lh = _learning.get_kalman_lambda_adjustment(home_team, "offense_home", _season, lh)
        la = _learning.get_kalman_lambda_adjustment(away_team, "offense_away", _season, la)

        # NOTE(Fase 2.1): solo defense_home se aplica. defense_away se intentó pero
        # se removió por causar regresión en bucket <40% y Brier global.
        # Diagnóstico empírico: defense_away tiene overlap parcial con Pitcher Engine
        # y Bullpen Engine cuando el equipo visitante es fuerte (correlación Pearson
        # r=0.25 con away_pitcher_mult, asimétricamente más alta en matchups
        # desiguales). Las 5,422 muestras del backtest mostraron Brier +0.00014 y
        # regresión en bucket <40% de -1.1pp → -3.0pp.
        #
        # DEFERRED: Milwaukee/Cardinals/Rays con bias persistente: probable necesidad
        # de "road performance modifier" o componente no-Poisson. NO atacar ahora.
        # Pendiente: audit interno de cada engine después de Fase 3.
        #
        # defense_home: pulls away team's λ toward the home team's observed runs-allowed
        # rate at their own park. Empirically orthogonal to Pitcher/Bullpen engines
        # (Pearson r=0.23 and r=-0.04 respectively). Corrects Cleveland from -11pp to
        # +2.7pp and improves 60-70% calibration bucket from -2.4pp to +0.3pp.
        la = _learning.get_kalman_lambda_adjustment(home_team, "defense_home", _season, la)
        # defense_away intentionally omitted — see NOTE(Fase 2.1) above.

        results['lambdas_history']['base'] = {'lh': lh, 'la': la}
        logger.info(f"   Lambda base (Kalman off+def): λ_h={lh:.3f} ({home_team}), λ_a={la:.3f} ({away_team})")

        # ── Team bias (LearningEngine) — corrects systematic model error per team ──
        # Applied after Kalman, before any engine modifies λ. The bias is
        # Kalman-dampened to avoid double-counting the 35% Kalman share.
        _home_bias = _learning.compute_team_bias_kalman_adjusted(home_team, _season, "offense_home")
        _away_bias = _learning.compute_team_bias_kalman_adjusted(away_team, _season, "offense_away")
        lh *= _home_bias
        la *= _away_bias
        if _home_bias != 1.0 or _away_bias != 1.0:
            logger.info(
                f"   Bias correction: λ_h×{_home_bias:.4f}={lh:.3f}  λ_a×{_away_bias:.4f}={la:.3f}"
            )

        # Stage factors tracker — populated per pipeline step for gradient descent.
        # Stores RAW engine ratios (weight=1.0 equivalent) so _gradient_step can
        # reconstruct the relationship between stage adjustment and prediction error.
        _stage_factors: Dict[str, float] = {}

        # Learned pipeline weights — scale each stage's adjustment.
        # Formula: λ_out = λ_in × (1 + w × (raw_ratio − 1))
        # w=1.0 → full engine adjustment; w=0.5 → half; w=1.5 → amplified.
        # Defaults to 1.0 for all stages until gradient descent has enough data.
        _weights = _learning.get_pipeline_weights(_season)
        logger.info(
            f"   Pipeline weights — "
            f"park:{_weights.get('park',1):.3f} "
            f"hfa:{_weights.get('hfa',1):.3f} "
            f"def:{_weights.get('defense',1):.3f} "
            f"pit:{_weights.get('pitcher',1):.3f} "
            f"bp:{_weights.get('bullpen',1):.3f} "
            f"ctx:{_weights.get('context',1):.3f}"
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 2: PITCHER ENGINE — sobre λ park-neutral del TTE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Aplicar FIP/xFIP/SIERA antes del parque separa calidad de
        # pitcheo pura del entorno; el parque escala el resultado en PASO 5.
        if use_pitcher:
            logger.info("\n⚾ PASO 2: Pitcher Engine (FIP/xFIP/SIERA sobre λ park-neutral)...")
            _lh_pre = lh; _la_pre = la
            lh_pit, la_pit, pitcher_meta = adjust_for_pitchers(lh, la, game_data)
            _raw_h_pit = lh_pit / _lh_pre if _lh_pre else 1.0
            _raw_a_pit = la_pit / _la_pre if _la_pre else 1.0
            _w_pit = _weights.get("pitcher", 1.0)
            lh = _lh_pre * (1.0 + _w_pit * (_raw_h_pit - 1.0))
            la = _la_pre * (1.0 + _w_pit * (_raw_a_pit - 1.0))
            # TODO(stage_factors_naming): the keys below are semantically inverted.
            # "home_pitcher" stores the AWAY pitcher's multiplier on λ_home (ratio
            # lh_after/lh_before, driven by adj_away["total_multiplier"]).
            # "away_pitcher" stores the HOME pitcher's multiplier on λ_away.
            # Renaming requires updating backtest_and_retrain.py, residual analysis
            # scripts, and learning_engine gradient-descent stage keys simultaneously.
            _stage_factors["home_pitcher"] = _raw_h_pit
            _stage_factors["away_pitcher"] = _raw_a_pit
            results['lambdas_history']['pitcher'] = {'lh': lh, 'la': la}
            results['metadata']['pitcher'] = pitcher_meta
            logger.info(f"   ✅ Pitcher adjusted: λ_h={lh:.3f}, λ_a={la:.3f} (w={_w_pit:.3f})")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 3: CONTEXTUAL ENGINE (B2B + rest + umpire)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Posición intencional: ANTES del Bullpen Engine.
        # El F5 snapshot se toma aquí: incluye pitcher + rest/umpire
        # (ambos aplican al F5) y excluye bullpen (starters lanzan F5).
        logger.info("\n🎯 PASO 3: Contextual Engine (rest + umpire)...")
        _lh_pre = lh; _la_pre = la
        lh_ctx, la_ctx, ctx_meta = adjust_for_context(lh, la, game_data)
        _raw_h_ctx = lh_ctx / _lh_pre if _lh_pre else 1.0
        _raw_a_ctx = la_ctx / _la_pre if _la_pre else 1.0
        _w_ctx = _weights.get("context", 1.0)
        lh = _lh_pre * (1.0 + _w_ctx * (_raw_h_ctx - 1.0))
        la = _la_pre * (1.0 + _w_ctx * (_raw_a_ctx - 1.0))
        results['lambdas_history']['contextual'] = {'lh': lh, 'la': la}
        results['metadata']['contextual'] = ctx_meta
        _stage_factors['home_context'] = _raw_h_ctx
        _stage_factors['away_context'] = _raw_a_ctx
        _ump = ctx_meta['umpire']
        logger.info(
            f"   ✅ home_rest={ctx_meta['home_rest_reason']}(×{ctx_meta['home_rest_mult']:.3f})"
            f"  away_rest={ctx_meta['away_rest_reason']}(×{ctx_meta['away_rest_mult']:.3f})"
            f"  ump={_ump['name']}(×{_ump['factor']:.3f})"
            f"  w={_w_ctx:.3f} → λ_h={lh:.3f}  λ_a={la:.3f}"
        )

        # ── F5 snapshot: post-pitcher + post-context, pre-bullpen ─────────────
        # Starters lanzan el F5 → pitcher quality + rest/umpire aplican.
        # Bullpen correctamente excluido (entra en PASO 4).
        #
        # F5 scale is dynamic per pitcher: a starter who averages 6+ IP always
        # covers all 5 innings → the bullpen (better run suppressors) only matters
        # in innings 6-9 → F5 is a larger fraction of total.  A short-inning starter
        # (avg 4 IP) may exit before the 5th → some F5 innings face a reliever.
        # Formula: base 0.575 ± 0.008 per IP from 5.0 (clamped to [0.52, 0.63]).
        def _f5_scale(avg_ip: float) -> float:
            return round(max(0.52, min(0.63, 0.575 + (min(avg_ip, 7.0) - 5.0) * 0.008)), 4)

        # lh_f5 ← affected by AWAY starter (how long he covers F5 innings for home team)
        # la_f5 ← affected by HOME starter
        _avg_ip_away = float(game_data.get('pitcher_away', {}).get('avg_innings_per_start') or 5.0)
        _avg_ip_home = float(game_data.get('pitcher_home', {}).get('avg_innings_per_start') or 5.0)
        _f5s_home = _f5_scale(_avg_ip_away)  # scale for lh_f5 (away pitcher)
        _f5s_away = _f5_scale(_avg_ip_home)  # scale for la_f5 (home pitcher)

        _post_ctx = results['lambdas_history'].get('contextual', {})
        if _post_ctx:
            lh_f5 = round(_post_ctx['lh'] * _f5s_home, 3)
            la_f5 = round(_post_ctx['la'] * _f5s_away, 3)
            logger.info(
                f"   F5 snapshot: λ_h={lh_f5:.3f} (×{_f5s_home}, away_avg={_avg_ip_away:.1f}IP)  "
                f"λ_a={la_f5:.3f} (×{_f5s_away}, home_avg={_avg_ip_home:.1f}IP)"
            )
        else:
            lh_f5 = None
            la_f5 = None

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 4: BULLPEN ENGINE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        _bp_home = game_data.get('bullpen_home', {})
        _bp_away = game_data.get('bullpen_away', {})
        _lh_pre = lh; _la_pre = la
        _raw_h_bp = _raw_a_bp = 1.0
        if _bp_home or _bp_away:
            logger.info("\n🔥 PASO 4: Bullpen Engine...")
            lh_bp, la_bp, bullpen_meta = adjust_for_bullpen(lh, la, game_data)
            _raw_h_bp = lh_bp / _lh_pre if _lh_pre else 1.0
            _raw_a_bp = la_bp / _la_pre if _la_pre else 1.0
            results['metadata']['bullpen'] = bullpen_meta
            logger.info(f"   ✅ Bullpen adjusted: λ_h={lh_bp:.3f}, λ_a={la_bp:.3f}")
        _w_bp = _weights.get("bullpen", 1.0)
        lh = _lh_pre * (1.0 + _w_bp * (_raw_h_bp - 1.0))
        la = _la_pre * (1.0 + _w_bp * (_raw_a_bp - 1.0))
        results['lambdas_history']['bullpen'] = {'lh': lh, 'la': la}
        _stage_factors['home_bullpen'] = _raw_h_bp
        _stage_factors['away_bullpen'] = _raw_a_bp

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 5: PARK + WEATHER ENGINE (simétrico: ambos equipos)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Escala el matchup completo (pitcheo + bullpen) al entorno del
        # parque. Aplicar después del pitcheo evita inflar métricas
        # park-neutrales (xFIP, SIERA) antes de aplicar el ajuste del pitcher.
        logger.info("\n🏟️  PASO 5: Park + Weather Engine...")
        _lh_pre = lh; _la_pre = la
        lh_park, la_park, park_meta = adjust_for_park_and_weather(lh, la, game_data)
        _raw_h_park = lh_park / _lh_pre if _lh_pre else 1.0
        _raw_a_park = la_park / _la_pre if _la_pre else 1.0
        _w_park = _weights.get("park", 1.0)
        lh = _lh_pre * (1.0 + _w_park * (_raw_h_park - 1.0))
        la = _la_pre * (1.0 + _w_park * (_raw_a_park - 1.0))
        results['lambdas_history']['park_weather'] = {'lh': lh, 'la': la}
        results['metadata']['park_weather'] = park_meta
        _stage_factors['home_park'] = _raw_h_park
        _stage_factors['away_park'] = _raw_a_park
        logger.info(
            f"   ✅ Park+Weather: park={park_meta['park_factor']:.3f}  "
            f"weather={park_meta['weather_mult']:.3f}  "
            f"λ_h={lh:.3f}  λ_a={la:.3f}  (w={_w_park:.3f})"
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 6: DEFENSIVE EFFICIENCY ENGINE (fielding puro)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        _def_home = game_data.get('defense_home') or {}
        _def_away = game_data.get('defense_away') or {}
        _lh_pre = lh; _la_pre = la
        _raw_h_def = _raw_a_def = 1.0
        if _def_home or _def_away:
            logger.info("\n🛡️  PASO 6: Defensive Efficiency Engine...")
            lh_def, la_def, def_meta = adjust_for_defense(lh, la, game_data)
            _raw_h_def = lh_def / _lh_pre if _lh_pre else 1.0
            _raw_a_def = la_def / _la_pre if _la_pre else 1.0
            results['metadata']['defense'] = def_meta
            logger.info(
                f"   ✅ Defense adjusted: λ_h={lh_def:.3f}  λ_a={la_def:.3f}  "
                f"(home_def×λ_a={def_meta['home_mult_on_away']:.4f}  "
                f"away_def×λ_h={def_meta['away_mult_on_home']:.4f})"
            )
        _w_def = _weights.get("defense", 1.0)
        lh = _lh_pre * (1.0 + _w_def * (_raw_h_def - 1.0))
        la = _la_pre * (1.0 + _w_def * (_raw_a_def - 1.0))
        results['lambdas_history']['defense'] = {'lh': lh, 'la': la}
        _stage_factors['home_defense'] = _raw_h_def
        _stage_factors['away_defense'] = _raw_a_def

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 7: HFA ENGINE (asimétrico: crowd home + travel away)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if use_hfa:
            logger.info("\n🏠 PASO 7: HFA Engine (crowd + travel)...")
            _lh_pre = lh; _la_pre = la
            lh_hfa, la_hfa, hfa_meta = get_adjusted_lambdas(lh, la, game_data)
            _raw_h_hfa = lh_hfa / _lh_pre if _lh_pre else 1.0
            _raw_a_hfa = la_hfa / _la_pre if _la_pre else 1.0
            _w_hfa = _weights.get("hfa", 1.0)
            lh = _lh_pre * (1.0 + _w_hfa * (_raw_h_hfa - 1.0))
            la = _la_pre * (1.0 + _w_hfa * (_raw_a_hfa - 1.0))
            _stage_factors["home_hfa"] = _raw_h_hfa
            _stage_factors["away_hfa"] = _raw_a_hfa
            results['lambdas_history']['hfa'] = {'lh': lh, 'la': la}
            results['metadata']['hfa'] = hfa_meta
            logger.info(f"   ✅ HFA adjusted: λ_h={lh:.3f}, λ_a={la:.3f} (w={_w_hfa:.3f})")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 8: MARKET ODDS (usa las pre-cargadas del selector o las fetcha)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Priority: odds passed by caller (from UI selector, already validated) >
        # fresh fetch by team-name fuzzy match (may miss if names differ).
        _fetched_odds: Optional[Dict] = None
        if market_odds and market_odds.get('ml_home') and market_odds.get('ml_away'):
            _fetched_odds = market_odds
            logger.info(
                f"   ✅ Odds del selector: ML {_fetched_odds['ml_home']}/{_fetched_odds['ml_away']}"
                + (f"  Pinnacle: {_fetched_odds.get('pin_home')}/{_fetched_odds.get('pin_away')}" if _fetched_odds.get('pin_home') else "")
                + (f"  Total: {_fetched_odds.get('total_line')}" if _fetched_odds.get('total_line') else "")
            )
        elif get_best_odds_for_teams is not None:
            try:
                _fetched_odds = get_best_odds_for_teams(
                    home_team=home_team,
                    away_team=away_team,
                    sport="baseball_mlb"
                ) or None
                if _fetched_odds:
                    logger.info(f"   ✅ Odds fetched: ML {_fetched_odds.get('ml_home')}/{_fetched_odds.get('ml_away')}")
                else:
                    logger.warning(f"   ⚠️  No se encontraron odds para {home_team} vs {away_team}")
            except Exception as e:
                logger.warning(f"   ⚠️  Error obteniendo odds: {e}")
        _market_total_line = (_fetched_odds or {}).get('total_line')

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 8: MONTE CARLO SIMULATION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        logger.info(f"\n🎲 PASO 8: Monte Carlo ({n_max:,} simulaciones)...")

        # F5 λ computado en PASO 3 (post-pitcher + post-context, pre-bullpen).
        # lh_f5 / la_f5 ya asignados — solo logear.
        if lh_f5 is not None:
            logger.info(f"   F5: λ_h={lh_f5:.3f}  λ_a={la_f5:.3f}  (post-pitcher+context × {F5_SCALE})")
        else:
            logger.info("   F5: sin snapshot disponible, el MC usará F5_SCALE internamente")

        # Wide clip — engines are calibrated to stay in range; clipping at [3, 7]
        # was overriding legitimate extreme outputs (ace + pitcher's park → λ≈2.5,
        # Coors + weak bullpen → λ≈9). Use [1.5, 12] as a sanity guard only.
        lh = max(1.5, min(lh, 12.0))
        la = max(1.5, min(la, 12.0))

        _has_real_pitcher = bool(
            game_data.get('pitcher_home', {}).get('fip') or
            game_data.get('pitcher_away', {}).get('fip')
        )
        _lambda_noise = _compute_lambda_noise(_TTE_AVAILABLE, _ENRICHMENT_AVAILABLE, _has_real_pitcher)
        logger.info(
            f"   λ finales pre-MC: λ_h={lh:.3f}  λ_a={la:.3f}  "
            f"total={lh+la:.3f}  noise={_lambda_noise:.2f}"
        )

        mc_results = monte_carlo_advanced(
            lh=lh,
            la=la,
            n_max=n_max,
            total_line=_market_total_line,
            lambda_noise=_lambda_noise,
            analyze_f5=analyze_f5,
            lh_f5=lh_f5,
            la_f5=la_f5,
        )

        # Capture raw MC probabilities BEFORE Platt overwrites them.
        # These are stored as p_home_raw so recalibrate_platt() fits on its
        # own input signal rather than its own output (circular dependency).
        _p_home_mc = mc_results['p_home']
        _p_away_mc = mc_results['p_away']

        # Dynamic Platt calibration — refitted weekly from live outcomes.
        # Applied asymmetrically: Platt transforms p_home only; p_away = 1 - p_home.
        # This preserves the b (intercept) parameter, which encodes the structural
        # home advantage (batting last, crowd on umpire calls) that Poisson run-scoring
        # cannot capture. Symmetric normalization cancelled b for neutral games,
        # destroying the structural signal recalibrate_platt() had learned.
        _pa, _pb = _learning.get_platt_params(_season)
        _p_home_cal = _platt(mc_results['p_home'], _pa, _pb)
        mc_results['p_home'] = round(_p_home_cal, 5)
        mc_results['p_away'] = round(1.0 - _p_home_cal, 5)

        results['probabilities'] = mc_results

        logger.info(f"   ✅ MC completado (noise={_lambda_noise:.2f}, Platt a={_pa:.3f} b={_pb:.3f})")
        logger.info(f"   Home Win: {mc_results.get('p_home', 0):.1%}  |  Away Win: {mc_results.get('p_away', 0):.1%}")
        logger.info(
            f"   Run Line: home-1.5={mc_results.get('p_rl_home', 'N/A'):.3f}  "
            f"away+1.5={mc_results.get('p_rl_away', 'N/A'):.3f}"
            if mc_results.get('p_rl_home') is not None else "   Run Line: N/A (no samples)"
        )
        if _market_total_line:
            logger.info(
                f"   Total O/U: line={_market_total_line}  "
                f"over={mc_results.get('p_over', 'N/A'):.3f}  "
                f"under={mc_results.get('p_under', 'N/A'):.3f}"
                if mc_results.get('p_over') is not None else f"   Total line={_market_total_line}"
            )

        # Persist prediction for future learning
        _gk = game_data.get('game_pk') or game_id
        _gd = str(game_data.get('game_date', datetime.now().strftime("%Y-%m-%d")))[:10]
        if _gk:
            try:
                _learning.record_prediction(
                    game_pk=int(_gk),
                    game_date=_gd,
                    season=_season,
                    home_team=home_team if isinstance(home_team, str) else home_team.get('name', ''),
                    away_team=away_team if isinstance(away_team, str) else away_team.get('name', ''),
                    lambda_home=lh,
                    lambda_away=la,
                    p_home=mc_results.get('p_home', 0.5),
                    p_away=mc_results.get('p_away', 0.5),
                    p_home_raw=float(_p_home_mc),
                    p_away_raw=float(_p_away_mc),
                    venue=game_data.get('venue'),
                    stage_factors=_stage_factors,
                )
            except Exception as _e:
                logger.debug(f"[learning] record_prediction failed: {_e}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 9: VALUE DETECTION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        logger.info("\n💰 PASO 9: Value Detection...")

        if _fetched_odds and _fetched_odds.get('ml_home') and _fetched_odds.get('ml_away'):
            from core.value_detector import GameOdds
            game_odds = GameOdds(
                ml_home=_fetched_odds['ml_home'],
                ml_away=_fetched_odds['ml_away'],
                pin_home=_fetched_odds.get('pin_home'),
                pin_away=_fetched_odds.get('pin_away'),
                total_line=_fetched_odds.get('total_line'),
                total_over=_fetched_odds.get('total_over'),
                total_under=_fetched_odds.get('total_under'),
                runline_home=_fetched_odds.get('runline_home'),
                runline_away=_fetched_odds.get('runline_away'),
                f5_ml_home=_fetched_odds.get('f5_ml_home'),
                f5_ml_away=_fetched_odds.get('f5_ml_away'),
                f5_total_line=_fetched_odds.get('f5_total_line'),
                f5_total_over=_fetched_odds.get('f5_total_over'),
                f5_total_under=_fetched_odds.get('f5_total_under'),
            )
            value_results = evaluate_value_ultra(
                mc_result=mc_results,
                odds=game_odds,
                lh=lh,
                la=la,
                home_samples=mc_results.get('home_samples'),
                away_samples=mc_results.get('away_samples'),
                total_samples=mc_results.get('total_samples'),
                analyze_f5=lh_f5 is not None,
            )
            results['best_bets'] = value_results.get('global_recommendation', {}).get('all_opportunities', [])
            results['metadata']['value'] = value_results
            results['metadata']['market_odds'] = _fetched_odds
            logger.info(f"   ✅ Value detection completado — {len(results['best_bets'])} oportunidades")
        else:
            logger.warning("   ⚠️  Sin odds de mercado — PASO 9 omitido")
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
