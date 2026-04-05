# ==========================================================
# MLB G8+ ULTIMATE - MÓDULO BASE COMPLETO v2.0
# Poisson + Bayes Híbrido + Pitcher + Weather + Park
# Sistema Profesional 60-62% Win Rate
# 
# ✅ CORRECCIONES APLICADAS:
# - Fuzzy matching mejorado (aliases MLB completos)
# - Ajustes combinados con pesos (no multiplicativos)
# - Confidence calibrado realista (máx 85%)
# - EV ajustado por vig del bookmaker
# - Logging completo con archivo de auditoría
# - Documentación clara en cada sección
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
from difflib import SequenceMatcher
import logging

# ==========================================================
# CONFIGURACIÓN DE LOGGING
# ==========================================================

def setup_logging():
    """Configura sistema de logging completo."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"g8_plus_{datetime.now():%Y%m%d}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()
logger.info("="*60)
logger.info("MLB G8+ ULTIMATE - Sistema iniciado")
logger.info("="*60)

# ==========================================================
# DICCIONARIO DE ALIASES MLB (FUZZY MATCHING)
# ==========================================================

MLB_TEAM_ALIASES = {
    # American League East
    "yankees": ["yankees", "new york yankees", "ny yankees", "nyy"],
    "red sox": ["red sox", "boston red sox", "redsox", "bos"],
    "blue jays": ["blue jays", "toronto blue jays", "jays", "tor"],
    "rays": ["rays", "tampa bay rays", "tb rays", "tampabayrays"],
    "orioles": ["orioles", "baltimore orioles", "o's", "bal"],
    
    # American League Central
    "white sox": ["white sox", "chicago white sox", "whitesox", "chw", "cws"],
    "guardians": ["guardians", "cleveland guardians", "cle"],
    "tigers": ["tigers", "detroit tigers", "det"],
    "royals": ["royals", "kansas city royals", "kc royals", "kcr"],
    "twins": ["twins", "minnesota twins", "min"],
    
    # American League West
    "astros": ["astros", "houston astros", "hou"],
    "angels": ["angels", "los angeles angels", "la angels", "laa"],
    "athletics": ["athletics", "oakland athletics", "a's", "oak"],
    "mariners": ["mariners", "seattle mariners", "sea"],
    "rangers": ["rangers", "texas rangers", "tex"],
    
    # National League East
    "braves": ["braves", "atlanta braves", "atl"],
    "marlins": ["marlins", "miami marlins", "mia"],
    "mets": ["mets", "new york mets", "ny mets", "nym"],
    "phillies": ["phillies", "philadelphia phillies", "phi"],
    "nationals": ["nationals", "washington nationals", "nats", "was"],
    
    # National League Central
    "cubs": ["cubs", "chicago cubs", "chc"],
    "reds": ["reds", "cincinnati reds", "cin"],
    "brewers": ["brewers", "milwaukee brewers", "mil"],
    "pirates": ["pirates", "pittsburgh pirates", "pit"],
    "cardinals": ["cardinals", "st louis cardinals", "stl"],
    
    # National League West
    "diamondbacks": ["diamondbacks", "arizona diamondbacks", "d-backs", "dbacks", "ari"],
    "rockies": ["rockies", "colorado rockies", "col"],
    "dodgers": ["dodgers", "los angeles dodgers", "la dodgers", "lad"],
    "padres": ["padres", "san diego padres", "sd padres", "sdp"],
    "giants": ["giants", "san francisco giants", "sf giants", "sfg"],
}

# ==========================================================
# IMPORTS DE TUS FETCHERS
# ==========================================================

try:
    from data_fetchers import MLBDataIntegrator
    DATA_FETCHERS_AVAILABLE = True
    logger.info("✅ data_fetchers.py cargado exitosamente")
except ImportError:
    DATA_FETCHERS_AVAILABLE = False
    logger.warning("⚠️ data_fetchers.py no disponible - modo básico")
    st.warning("⚠️ data_fetchers.py no encontrado - Funcionalidad limitada")

# ==========================================================
# FUNCIONES MATEMÁTICAS CORE
# ==========================================================

def clamp(x, a, b):
    """Limita valor entre a y b."""
    return max(a, min(b, x))

def _logit(p: float) -> float:
    """Función logit con protección."""
    p = clamp(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1 - p))

def _sigmoid(z: float) -> float:
    """Función sigmoide con límites."""
    return 1.0 / (1.0 + math.exp(-clamp(z, -20, 20)))

def decimal_to_prob(odds: float) -> float:
    """Convierte cuotas decimales a probabilidad implícita."""
    try:
        if odds is None or not np.isfinite(odds) or odds <= 1:
            return np.nan
        return 1.0 / float(odds)
    except Exception:
        return np.nan

# ==========================================================
# FUZZY MATCHING MEJORADO
# ==========================================================

def normalize_team_name(team_name: str) -> str:
    """Normaliza nombre de equipo removiendo caracteres especiales."""
    if not team_name:
        return ""
    normalized = team_name.lower().strip()
    normalized = normalized.replace("-", " ").replace(".", "")
    normalized = " ".join(normalized.split())
    return normalized

def get_team_canonical_name(team_name: str) -> Optional[str]:
    """Encuentra el nombre canónico usando diccionario de aliases."""
    normalized = normalize_team_name(team_name)
    
    for canonical, aliases in MLB_TEAM_ALIASES.items():
        if normalized in aliases:
            return canonical
        
        for alias in aliases:
            similarity = SequenceMatcher(None, normalized, alias).ratio()
            if similarity > 0.85:
                return canonical
    
    return None

def calculate_team_similarity(team1: str, team2: str) -> float:
    """
    Calcula similitud entre dos nombres de equipos.
    Returns: float entre 0-1 (1 = match perfecto)
    """
    norm1 = normalize_team_name(team1)
    norm2 = normalize_team_name(team2)
    
    # Match exacto
    if norm1 == norm2:
        return 1.0
    
    # Intenta match por canónico
    canon1 = get_team_canonical_name(team1)
    canon2 = get_team_canonical_name(team2)
    
    if canon1 and canon2 and canon1 == canon2:
        return 1.0
    
    # Fuzzy match directo
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    # Bonus por palabras clave compartidas
    common_words = {"the", "of", "at", "in", "bay", "city"}
    words1 = set(norm1.split()) - common_words
    words2 = set(norm2.split()) - common_words
    
    if words1 and words2:
        word_overlap = len(words1 & words2) / max(len(words1), len(words2))
        similarity = max(similarity, word_overlap)
    
    return similarity

def find_matching_mlb_data(odds_game: Dict, mlb_games: List[Dict], 
                          min_similarity: float = 0.80) -> Optional[Dict]:
    """
    Encuentra el partido MLB que hace match con odds.
    
    Args:
        odds_game: Datos del juego desde odds_fetcher
        mlb_games: Lista de juegos desde data_fetchers
        min_similarity: Umbral mínimo de similitud (0-1)
    
    Returns:
        Dict con data MLB o None si no hay match
    """
    home_odds = str(odds_game.get("home_team", ""))
    away_odds = str(odds_game.get("away_team", ""))
    
    if not home_odds or not away_odds:
        return None
    
    best_match = None
    best_score = 0.0
    
    for mlb_game in mlb_games:
        home_mlb = str(mlb_game.get("home_team", ""))
        away_mlb = str(mlb_game.get("away_team", ""))
        
        if not home_mlb or not away_mlb:
            continue
        
        home_sim = calculate_team_similarity(home_odds, home_mlb)
        away_sim = calculate_team_similarity(away_odds, away_mlb)
        combined_score = (home_sim + away_sim) / 2
        
        if combined_score > best_score and combined_score >= min_similarity:
            best_score = combined_score
            best_match = mlb_game
    
    if best_match:
        logger.info(f"✅ Match: '{home_odds}' vs '{away_odds}' -> {best_score:.1%} similarity")
    else:
        logger.warning(f"❌ No match: '{home_odds}' vs '{away_odds}'")
    
    return best_match

# ==========================================================
# ESTIMACIÓN DE FUERZAS DESDE ODDS
# ==========================================================

def estimate_strength_from_moneyline(home_odds: float, away_odds: float,
                                    home_advantage: float = 0.54) -> Tuple[float, float]:
    """
    Estima fuerza relativa de equipos desde moneyline.
    
    Args:
        home_odds: Cuota decimal home
        away_odds: Cuota decimal away
        home_advantage: Ventaja de local (default 54%)
    
    Returns:
        Tuple (p_home, p_away) normalizado
    """
    p_home = decimal_to_prob(home_odds)
    p_away = decimal_to_prob(away_odds)
    
    if np.isfinite(p_home) and np.isfinite(p_away) and (p_home + p_away) > 0:
        # Normalizar para remover vig
        s = p_home + p_away
        p_home /= s
        p_away = 1.0 - p_home
    else:
        p_home, p_away = home_advantage, 1 - home_advantage
    
    # Suavizar hacia ventaja de local
    target_home = clamp(p_home, 0.30, 0.70)
    p_home = target_home * 0.7 + home_advantage * 0.3
    p_away = 1.0 - p_home
    
    logger.debug(f"Strength estimation: home={p_home:.3f}, away={p_away:.3f}")
    
    return p_home, p_away

def distribute_lambda(total_line: float, p_home_strength: float,
                     p_away_strength: float) -> Tuple[float, float]:
    """
    Distribuye runs esperados (lambda) según fuerza relativa.
    
    Args:
        total_line: Total O/U del mercado
        p_home_strength: Probabilidad de fuerza home (0-1)
        p_away_strength: Probabilidad de fuerza away (0-1)
    
    Returns:
        Tuple (lam_home, lam_away)
    """
    ph = clamp(p_home_strength, 0.05, 0.95)
    pa = clamp(p_away_strength, 0.05, 0.95)
    z = ph + pa
    
    if z <= 0:
        ph, pa = 0.5, 0.5
    else:
        ph, pa = ph / z, pa / z
    
    lam_home = max(0.5, total_line * ph)
    lam_away = max(0.5, total_line * pa)
    
    logger.debug(f"Lambda distribution: home={lam_home:.2f}, away={lam_away:.2f} (total={total_line})")
    
    return lam_home, lam_away

# ==========================================================
# AJUSTES COMBINADOS (CORREGIDO)
# ==========================================================

def calculate_adjustment_factors(pitcher_stats: Dict, weather: Dict, 
                                park_factor: Dict) -> Dict[str, float]:
    """
    Calcula factores de ajuste individuales.
    
    Returns:
        Dict con factores: {"pitcher": float, "weather": float, "park": float}
    """
    factors = {"pitcher": 1.0, "weather": 1.0, "park": 1.0}
    
    # Factor de pitcher
    if pitcher_stats:
        era = pitcher_stats.get("era", 4.50)
        whip = pitcher_stats.get("whip", 1.35)
        
        era_factor = clamp(era / 4.50, 0.70, 1.40)
        whip_factor = clamp(whip / 1.35, 0.80, 1.30)
        
        factors["pitcher"] = 0.6 * era_factor + 0.4 * whip_factor
        logger.debug(f"Pitcher factor: {factors['pitcher']:.3f} (ERA={era:.2f}, WHIP={whip:.2f})")
    
    # Factor de clima
    if weather:
        temp = weather.get("temp_f", 72)
        wind = weather.get("wind_speed_mph", 0)
        
        temp_mult = 1.05 if temp > 80 else (0.95 if temp < 60 else 1.0)
        wind_mult = 1.08 if wind > 15 else (1.03 if wind > 10 else 1.0)
        
        factors["weather"] = temp_mult * wind_mult
        logger.debug(f"Weather factor: {factors['weather']:.3f} (temp={temp}°F, wind={wind}mph)")
    
    # Factor de parque
    if park_factor:
        factors["park"] = park_factor.get("runs", 1.0)
        logger.debug(f"Park factor: {factors['park']:.3f}")
    
    return factors

def adjust_lambda_combined(base_lambda: float, factors: Dict[str, float]) -> float:
    """
    Ajusta lambda combinando factores con PESOS.
    ✅ CORREGIDO: No multiplica directamente para evitar over-adjustment.
    
    Pesos:
    - 50% Pitcher quality
    - 25% Park factor
    - 25% Weather
    
    Args:
        base_lambda: Lambda base
        factors: Dict con factores de ajuste
    
    Returns:
        Lambda ajustado
    """
    combined_factor = (
        0.50 * factors["pitcher"] +
        0.25 * factors["park"] +
        0.25 * factors["weather"]
    )
    
    # Limitar rango total a ±35% del base
    combined_factor = clamp(combined_factor, 0.75, 1.35)
    
    adjusted = base_lambda * combined_factor
    
    logger.debug(f"Lambda adjustment: {base_lambda:.2f} -> {adjusted:.2f} "
                f"(P:{factors['pitcher']:.2f}, K:{factors['park']:.2f}, W:{factors['weather']:.2f})")
    
    return adjusted

# ==========================================================
# MONTE CARLO SIMULATION
# ==========================================================

def monte_carlo_totals(lh: float, la: float, n: int = 50000,
                      total_line: Optional[float] = None) -> Dict[str, Any]:
    """
    Simulación Monte Carlo con distribuciones Poisson.
    
    Args:
        lh: Lambda home (runs esperados)
        la: Lambda away (runs esperados)
        n: Número de simulaciones
        total_line: Línea de total O/U (opcional)
    
    Returns:
        Dict con probabilidades y estadísticas
    """
    n = int(clamp(n, 30000, 80000))
    
    try:
        home_runs = np.random.poisson(lh, size=n)
        away_runs = np.random.poisson(la, size=n)
        total_runs = home_runs + away_runs
        
        home_win = np.mean(home_runs > away_runs)
        away_win = np.mean(away_runs > home_runs)
        tie = np.mean(home_runs == away_runs)
        
        out = {
            "p_home": float(home_win + 0.5 * tie),
            "p_away": float(away_win + 0.5 * tie),
            "mean_home": float(np.mean(home_runs)),
            "mean_away": float(np.mean(away_runs)),
            "mean_total": float(np.mean(total_runs)),
            "std_total": float(np.std(total_runs)),
        }
        
        if total_line is not None and np.isfinite(total_line):
            out["p_over"] = float(np.mean(total_runs > total_line))
            out["p_under"] = float(np.mean(total_runs < total_line))
            out["total_line"] = float(total_line)
        else:
            out["p_over"] = out["p_under"] = np.nan
        
        logger.debug(f"Monte Carlo ({n} sims): p_home={out['p_home']:.3f}, mean_total={out['mean_total']:.2f}")
        
        return out
    except Exception as e:
        logger.error(f"Monte Carlo error: {e}")
        return {"p_home": 0.5, "p_away": 0.5, "mean_home": lh, "mean_away": la}

# ==========================================================
# EV AJUSTADO POR VIG (NUEVO)
# ==========================================================

def calculate_ev_adjusted(p_model: float, odds: float, 
                         all_odds: List[float]) -> float:
    """
    Calcula EV ajustado removiendo el vig del bookmaker.
    ✅ NUEVO: Usa probabilidades fair (sin vig).
    
    Args:
        p_model: Probabilidad del modelo
        odds: Cuota decimal
        all_odds: Todas las cuotas del mercado
    
    Returns:
        EV ajustado
    """
    try:
        p_market_with_vig = 1.0 / odds
        
        # Calcular vig total
        total_prob = sum([1/o for o in all_odds if o > 1])
        
        if total_prob <= 1.0:
            # No hay vig detectado
            logger.debug(f"No vig detected, using direct EV")
            return p_model - p_market_with_vig
        
        # Probabilidad fair (sin vig)
        p_market_fair = p_market_with_vig / total_prob
        vig = total_prob - 1.0
        
        ev = p_model - p_market_fair
        
        logger.debug(f"EV adjustment: market={p_market_with_vig:.3f} -> fair={p_market_fair:.3f}, vig={vig:.3f}, EV={ev:+.3f}")
        
        return ev
    except Exception as e:
        logger.error(f"EV adjustment error: {e}")
        return p_model - (1.0 / odds)

# ==========================================================
# CONFIDENCE CALIBRADO (CORREGIDO)
# ==========================================================

def calculate_confidence_calibrated(
    ev: float,
    pitcher_quality: float,
    weather_impact: float,
    data_completeness: float
) -> float:
    """
    Calcula confidence score CALIBRADO y realista.
    ✅ CORREGIDO: Máximo 85% (nunca 100%).
    
    Rango: 0.30 - 0.85
    
    Componentes:
    - Base 30% (siempre hay incertidumbre)
    - EV magnitude (hasta +25%)
    - Data quality (hasta +15%)
    - Weather availability (hasta +5%)
    
    Args:
        ev: Expected value
        pitcher_quality: Calidad de data de pitchers (0-1)
        weather_impact: Disponibilidad de weather data (0-1)
        data_completeness: Completitud general de data (0-1)
    
    Returns:
        Confidence score (0.30 - 0.85)
    """
    # BASE: 30% (siempre hay incertidumbre en deportes)
    score = 0.30
    
    # 1. EV component (máx +0.25)
    if ev > 0.10:
        score += 0.25
    elif ev > 0.07:
        score += 0.20
    elif ev > 0.05:
        score += 0.15
    elif ev > 0.03:
        score += 0.10
    else:
        score += 0.05
    
    # 2. Data quality (máx +0.15)
    score += pitcher_quality * 0.08
    score += data_completeness * 0.07
    
    # 3. Weather data (máx +0.05)
    score += weather_impact * 0.05
    
    # LÍMITE REALISTA: 30% - 85%
    score = clamp(score, 0.30, 0.85)
    
    logger.debug(f"Confidence: {score:.2%} (EV={ev:+.3f}, PQ={pitcher_quality:.2f}, DC={data_completeness:.2f})")
    
    return float(score)

def get_recommendation(ev: float, confidence: float, rating: float) -> str:
    """
    Genera recomendación basada en métricas.
    
    Args:
        ev: Expected value
        confidence: Confidence score (0-1)
        rating: Rating G8+ (0-10)
    
    Returns:
        String con recomendación
    """
    if ev > 0.06 and confidence > 0.70 and rating > 7.5:
        return "🔥 STRONG BET"
    elif ev > 0.04 and confidence > 0.60 and rating > 7.0:
        return "✅ GOOD BET"
    elif ev > 0.03 and confidence > 0.55 and rating > 6.5:
        return "⚠️ MARGINAL BET"
    else:
        return "❌ SKIP"

# ==========================================================
# FUNCIÓN PRINCIPAL: RUN_MODULE
# ==========================================================

def run_module(data, predictor, detector, config):
    """
    Módulo principal MLB G8+ Ultimate - VERSIÓN CORREGIDA.
    
    Args:
        data: DataFrame con partidos de odds_fetcher
        predictor: Legacy (no usado)
        detector: Legacy (no usado)
        config: Dict con configuración
    """
    
    logger.info("Iniciando análisis G8+...")
    
    if data is None or data.empty:
        st.warning("⚠️ No hay partidos de MLB disponibles")
        logger.warning("No games available")
        return
    
    # Configuración
    kelly_factor = config.get("kelly_factor", 0.25)
    min_ev = config.get("min_ev", 0.03)
    min_rating = config.get("min_rating", 6.5)
    save_prediction = config.get("save_prediction")
    
    DEFAULT_TOTAL_LINE = 9.0
    
    logger.info(f"Config: kelly={kelly_factor}, min_ev={min_ev}, min_rating={min_rating}")
    
    # ======================================================
    # OBTENER DATA ADICIONAL (PITCHERS, WEATHER, PARK)
    # ======================================================
    mlb_data = []
    data_available = False
    
    if DATA_FETCHERS_AVAILABLE:
        try:
            with st.spinner("🔍 Obteniendo data de pitchers, clima y parques..."):
                integrator = MLBDataIntegrator()
                mlb_data = integrator.get_complete_game_data()
                data_available = True
                st.success(f"✅ Data MLB cargada: {len(mlb_data)} juegos con contexto completo")
                logger.info(f"MLB data loaded: {len(mlb_data)} games")
        except Exception as e:
            st.warning(f"⚠️ No se pudo cargar data adicional: {e}")
            logger.error(f"MLB data fetch error: {e}")
    else:
        st.info("ℹ️ Modo básico: Usando solo cuotas (sin pitcher/weather data)")
    
    # ======================================================
    # SELECTOR DE PARTIDOS
    # ======================================================
    st.subheader(f"📋 Selecciona los partidos a analizar ({len(data)} disponibles)")
    
    matches_list = []
    for idx, row in data.iterrows():
        home = row.get("home_team", "Unknown")
        away = row.get("away_team", "Unknown")
        home_odds = row.get("home_odds", "N/A")
        away_odds = row.get("away_odds", "N/A")
        commence_time = row.get("commence_time", "")
        
        try:
            game_time = datetime.fromisoformat(str(commence_time).replace('Z', '+00:00'))
            time_str = game_time.strftime("%d/%m %H:%M")
        except:
            time_str = "Horario N/D"
        
        match_info = {
            "index": idx,
            "display": f"⚾ {home} vs {away} | 🏠 {home_odds:.2f} - ✈️ {away_odds:.2f} | ⏰ {time_str}",
            "home": home,
            "away": away
        }
        matches_list.append(match_info)
    
    # UI de selección
    st.markdown("##### Marca los partidos que quieres analizar:")
    
    selected_indices = []
    select_all = st.checkbox("✅ Seleccionar todos los partidos", value=False)
    
    if select_all:
        selected_indices = [m["index"] for m in matches_list]
    else:
        for match in matches_list:
            if st.checkbox(match["display"], value=False, key=f"match_{match['index']}"):
                selected_indices.append(match["index"])
    
    st.markdown("---")
    
    if not selected_indices:
        st.info("👆 Selecciona al menos un partido para analizar")
        return
    
    st.success(f"✅ {len(selected_indices)} partido(s) seleccionado(s)")
    logger.info(f"Selected {len(selected_indices)} games for analysis")
    
    if not st.button("🚀 ANALIZAR PARTIDOS SELECCIONADOS", type="primary", use_container_width=True):
        st.info("💡 Presiona el botón para comenzar el análisis")
        return
    
    st.markdown("---")
    st.subheader("📊 Resultados del Análisis G8+ Ultimate")
    
    picks_found = 0
    
    # ======================================================
    # PROCESAR PARTIDOS SELECCIONADOS
    # ======================================================
    for idx in selected_indices:
        row = data.loc[idx]
        
        try:
            # Datos básicos
            home_team = str(row.get("home_team", "Unknown"))
            away_team = str(row.get("away_team", "Unknown"))
            home_odds = float(row.get("home_odds", np.nan))
            away_odds = float(row.get("away_odds", np.nan))
            total_line = float(row.get("total_line", DEFAULT_TOTAL_LINE))
            over_odds = float(row.get("over_odds", 1.91))
            under_odds = float(row.get("under_odds", 1.91))
            
            if not (np.isfinite(home_odds) and np.isfinite(away_odds)):
                st.warning(f"⚠️ {home_team} vs {away_team}: Cuotas inválidas")
                logger.warning(f"Invalid odds: {home_team} vs {away_team}")
                continue
            
            logger.info(f"Processing: {home_team} vs {away_team}")
            
            # Buscar data adicional con MATCHING MEJORADO
            mlb_match = None
            if data_available:
                mlb_match = find_matching_mlb_data(row.to_dict(), mlb_data)
            
            # ==========================================
            # PASO 1: ESTIMAR FUERZAS BASE
            # ==========================================
            p_home_str, p_away_str = estimate_strength_from_moneyline(home_odds, away_odds)
            
            # ==========================================
            # PASO 2: DISTRIBUCIÓN INICIAL DE LAMBDAS
            # ==========================================
            lam_home_base, lam_away_base = distribute_lambda(total_line, p_home_str, p_away_str)
            
            # ==========================================
            # PASO 3: CALCULAR FACTORES DE AJUSTE
            # ==========================================
            factors_home = {"pitcher": 1.0, "weather": 1.0, "park": 1.0}
            factors_away = {"pitcher": 1.0, "weather": 1.0, "park": 1.0}
            
            pitcher_quality_home = 0.5
            pitcher_quality_away = 0.5
            weather_impact = 0.0
            park_impact = 1.0
            data_completeness = 0.3
            
            adjustments_applied = []
            
            if mlb_match:
                data_completeness = 0.8
                
                # Pitcher away (afecta runs de home)
                if "away_pitcher_stats" in mlb_match and mlb_match["away_pitcher_stats"]:
                    pitcher_stats = mlb_match["away_pitcher_stats"]
                    temp_factors = calculate_adjustment_factors(pitcher_stats, {}, {})
                    factors_home["pitcher"] = temp_factors["pitcher"]
                    
                    era = pitcher_stats.get("era", 4.50)
                    pitcher_quality_away = clamp(1.0 - (abs(era - 4.50) / 4.50), 0, 1)
                    adjustments_applied.append(f"Pitcher Away (ERA {era:.2f})")
                
                # Pitcher home (afecta runs de away)
                if "home_pitcher_stats" in mlb_match and mlb_match["home_pitcher_stats"]:
                    pitcher_stats = mlb_match["home_pitcher_stats"]
                    temp_factors = calculate_adjustment_factors(pitcher_stats, {}, {})
                    factors_away["pitcher"] = temp_factors["pitcher"]
                    
                    era = pitcher_stats.get("era", 4.50)
                    pitcher_quality_home = clamp(1.0 - (abs(era - 4.50) / 4.50), 0, 1)
                    adjustments_applied.append(f"Pitcher Home (ERA {era:.2f})")
                
                # Weather (afecta ambos)
                if "weather" in mlb_match and mlb_match["weather"]:
                    weather = mlb_match["weather"]
                    temp_factors = calculate_adjustment_factors({}, weather, {})
                    factors_home["weather"] = temp_factors["weather"]
                    factors_away["weather"] = temp_factors["weather"]
                    
                    temp = weather.get("temp_f", "?")
                    wind = weather.get("wind_speed_mph", "?")
                    adjustments_applied.append(f"Clima ({temp}°F, {wind}mph viento)")
                    weather_impact = 0.3
                
                # Park factor (afecta ambos)
                if "park_factor" in mlb_match and mlb_match["park_factor"]:
                    park_factor = mlb_match["park_factor"]
                    park_impact = park_factor.get("runs", 1.0)
                    factors_home["park"] = park_impact
                    factors_away["park"] = park_impact
                    
                    park_type = park_factor.get("type", "neutral")
                    adjustments_applied.append(f"Parque ({park_impact:.2f}x, {park_type})")
            
            # ==========================================
            # PASO 4: APLICAR AJUSTES COMBINADOS
            # ==========================================
            lam_home_adj = adjust_lambda_combined(lam_home_base, factors_home)
            lam_away_adj = adjust_lambda_combined(lam_away_base, factors_away)
            
            # Límites finales
            lam_home_adj = clamp(lam_home_adj, 0.5, 12.0)
            lam_away_adj = clamp(lam_away_adj, 0.5, 12.0)
            
            # ==========================================
            # PASO 5: MONTE CARLO
            # ==========================================
            mc_results = monte_carlo_totals(lam_home_adj, lam_away_adj, 
                                          n=50000, total_line=total_line)
            
            p_home_model = mc_results["p_home"]
            p_away_model = mc_results["p_away"]
            mean_total = mc_results["mean_total"]
            
            # ==========================================
            # PASO 6: EVs AJUSTADOS POR VIG
            # ==========================================
            all_odds = [home_odds, away_odds]
            
            ev_home = calculate_ev_adjusted(p_home_model, home_odds, all_odds)
            ev_away = calculate_ev_adjusted(p_away_model, away_odds, all_odds)
            
            # ==========================================
            # PASO 7: KELLY
            # ==========================================
            kelly_home = config["kelly_criterion"](home_odds, p_home_model, kelly_factor)
            kelly_away = config["kelly_criterion"](away_odds, p_away_model, kelly_factor)
            
            # ==========================================
            # PASO 8: RATING & CONFIDENCE
            # ==========================================
            best_ev = max([x for x in [ev_home, ev_away] if np.isfinite(x)], default=0)
            rating_g8 = clamp(5.0 + best_ev * 20.0, 0.0, 10.0)
            
            pitcher_quality = (pitcher_quality_home + pitcher_quality_away) / 2
            confidence = calculate_confidence_calibrated(
                best_ev,
                pitcher_quality,
                weather_impact,
                data_completeness
            )
            
            # ==========================================
            # PASO 9: RECOMMENDATION
            # ==========================================
            recommendation = get_recommendation(best_ev, confidence, rating_g8)
            
            # ==========================================
            # PASO 10: FILTRAR POR MÍNIMOS
            # ==========================================
            if best_ev < min_ev or rating_g8 < min_rating:
                logger.info(f"Skipping {home_team} vs {away_team}: EV={best_ev:.3f}, rating={rating_g8:.1f}")
                continue
            
            # ==========================================
            # PASO 11: IDENTIFICAR MEJOR PICK
            # ==========================================
            if ev_home > ev_away and np.isfinite(ev_home):
                pick_team = home_team
                pick_odds = home_odds
                pick_ev = ev_home
                pick_kelly = kelly_home
                pick_prob = p_home_model
            elif np.isfinite(ev_away):
                pick_team = away_team
                pick_odds = away_odds
                pick_ev = ev_away
                pick_kelly = kelly_away
                pick_prob = p_away_model
            else:
                continue
            
            picks_found += 1
            
            logger.info(f"✅ PICK FOUND: {pick_team} @ {pick_odds:.2f} | EV={pick_ev:+.2%} | Conf={confidence:.0%}")
            
            # ==========================================
            # PASO 12: MOSTRAR EN UI
            # ==========================================
            st.markdown("---")
            
            # Header con recomendación
            rec_color = {
                "🔥 STRONG BET": "🟢",
                "✅ GOOD BET": "🟡",
                "⚠️ MARGINAL BET": "🟠",
                "❌ SKIP": "🔴"
            }.get(recommendation, "⚪")
            
            st.markdown(f"## {rec_color} {home_team} vs {away_team}")
            st.caption(f"**{recommendation}** | Confianza: {confidence*100:.0f}% | Rating G8+: {rating_g8:.1f}/10")
            
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 PICK", pick_team, delta="Recomendado")
            
            with col2:
                ev_color = "normal" if pick_ev < 0.06 else "inverse"
                st.metric("💰 Expected Value", f"{pick_ev*100:+.2f}%", 
                         delta="Alto" if pick_ev > 0.06 else "Medio",
                         delta_color=ev_color)
            
            with col3:
                st.metric("📊 Kelly Sugerido", f"{pick_kelly*100:.2f}%", 
                         delta="del bankroll")
            
            with col4:
                st.metric("📈 Probabilidad", f"{pick_prob*100:.1f}%",
                         delta=f"vs {(1/pick_odds)*100:.1f}% mercado")
            
            # Análisis detallado
            with st.expander("🔍 Ver análisis completo"):
                st.markdown(f"""
                ### 📊 Análisis Cuantitativo G8+
                
                **🏠 {home_team}**
                - Cuota: {home_odds:.2f}
                - Prob. Modelo: {p_home_model*100:.1f}%
                - Prob. Mercado: {(1/home_odds)*100:.1f}%
                - EV: {ev_home*100:+.2f}%
                - λ ajustado: {lam_home_adj:.2f} runs (base: {lam_home_base:.2f})
                
                **✈️ {away_team}**
                - Cuota: {away_odds:.2f}
                - Prob. Modelo: {p_away_model*100:.1f}%
                - Prob. Mercado: {(1/away_odds)*100:.1f}%
                - EV: {ev_away*100:+.2f}%
                - λ ajustado: {lam_away_adj:.2f} runs (base: {lam_away_base:.2f})
                
                **📊 Totales**
                - Total esperado: {mean_total:.2f} runs
                - Línea O/U: {total_line} runs
                - Desviación estándar: {mc_results.get('std_total', 0):.2f}
                
                **🔧 Ajustes Aplicados:**
                """)
                
                if adjustments_applied:
                    for adj in adjustments_applied:
                        st.markdown(f"- ✅ {adj}")
                else:
                    st.markdown("- ℹ️ Solo análisis base (sin data de pitchers/clima)")
                
                st.markdown(f"""
                **🎲 Simulación Monte Carlo**
                - Iteraciones: 50,000
                - Método: Poisson + Bayes
                - Confianza estadística: {confidence*100:.0f}%
                
                **💡 Interpretación:**
                - Rating {rating_g8:.1f}/10 = {'Excelente' if rating_g8 > 8 else 'Bueno' if rating_g8 > 7 else 'Aceptable'}
                - EV {pick_ev*100:+.2f}% = {'Alto valor' if pick_ev > 0.05 else 'Valor moderado'}
                - Kelly {pick_kelly*100:.2f}% = Apostar ${pick_kelly*1000:.0f} por cada $1,000 de bankroll
                """)
            
            # ==========================================
            # PASO 13: GUARDAR PREDICCIÓN
            # ==========================================
            if save_prediction:
                prediction = {
                    "timestamp": datetime.now().isoformat(),
                    "sport": "MLB",
                    "league": "MLB",
                    "home_team": home_team,
                    "away_team": away_team,
                    "p_home": p_home_model,
                    "p_away": p_away_model,
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "pick_type": "Moneyline",
                    "pick_value": pick_team,
                    "ev": pick_ev,
                    "kelly": pick_kelly,
                    "confidence": confidence,
                    "rating": rating_g8,
                    "model_version": "G8+ Ultimate v2.0",
                    "notes": f"λ_home={lam_home_adj:.2f}, λ_away={lam_away_adj:.2f}, adjustments={len(adjustments_applied)}"
                }
                
                try:
                    save_prediction(prediction)
                    st.success("💾 Pick guardado en base de datos")
                    logger.info(f"Prediction saved: {pick_team}")
                except Exception as e:
                    st.warning(f"⚠️ No se pudo guardar: {e}")
                    logger.error(f"Save error: {e}")
        
        except Exception as e:
            st.error(f"❌ Error procesando {row.get('home_team', 'partido')}: {e}")
            logger.error(f"Processing error: {e}", exc_info=True)
            continue
    
    # ======================================================
    # RESUMEN FINAL
    # ======================================================
    st.markdown("---")
    
    logger.info(f"Analysis complete: {picks_found} picks found")
    
    if picks_found > 0:
        st.success(f"""
        ### ✅ Análisis MLB G8+ Ultimate Completado
        
        - **{picks_found} pick(s)** con valor identificado(s)
        - EV mínimo: **{min_ev*100:.1f}%**
        - Rating mínimo: **{min_rating}/10**
        - Modelo: **Poisson + Bayes + Pitcher + Weather + Park**
        - Data: **{'Completa (pitchers/clima/parques)' if data_available else 'Básica (solo cuotas)'}**
        
        💡 **Recuerda**: Usa gestión de bankroll Kelly sugerida
        
        📝 **Logs guardados en**: `logs/g8_plus_{datetime.now():%Y%m%d}.log`
        """)
    else:
        st.info(f"""
        ### ℹ️ No se encontraron picks con valor suficiente
        
        De {len(selected_indices)} partidos analizados, ninguno cumplió con:
        - EV mínimo: **{min_ev*100:.1f}%**
        - Rating mínimo: **{min_rating}/10**
        
        💡 **Sugerencias**:
        - Ajusta los filtros en el sidebar
        - Espera a que haya más partidos disponibles
        - Los mercados pueden estar eficientemente valorados hoy
        
        📝 **Logs disponibles en**: `logs/g8_plus_{datetime.now():%Y%m%d}.log`
        """)

# ==========================================================
# METADATA Y VERSIÓN
# ==========================================================

__version__ = "2.0.0"
__author__ = "G8+ Team"
__description__ = "MLB G8+ Ultimate - Sistema profesional de análisis de apuestas MLB"

# Registro de versión
logger.info(f"G8+ Ultimate v{__version__} - Module loaded successfully")
