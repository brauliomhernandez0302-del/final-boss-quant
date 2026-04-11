# ============================================================
# SISTEMA DE APUESTAS INTELIGENTE - APP PRINCIPAL
# FINAL BOSS QUANT G8+ MULTIDEPORTE
# Versión Final v2.1 - MLB + NBA + UFC
# ============================================================
"""
FINAL BOSS QUANT G8+ - Sistema Cuantitativo Multideporte

Arquitectura mejorada con:
- Configuración centralizada
- Componentes UI reutilizables
- Manejo robusto de errores
- Base de datos optimizada
- Código DRY (Don't Repeat Yourself)
- Soporte completo: MLB, NBA, UFC

Autor: Braulio
Versión: 2.1
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================


class ThemeColors(Enum):
    """Paleta de colores del tema."""

    PRIMARY = "#00C2FF"
    SUCCESS = "#00C853"
    DANGER = "#FF5252"
    WARNING = "#FDD835"
    NEUTRAL = "#90A4AE"
    BG_DARK = "#0C0F12"
    BG_CARD = "rgba(255,255,255,0.05)"


@dataclass
class AppConfig:
    """Configuración de la aplicación."""

    APP_NAME: str = "FINAL BOSS QUANT G8+"
    APP_VERSION: str = "2.1"
    PAGE_ICON: str = "🎯"

    # Valores por defecto
    DEFAULT_KELLY_FACTOR: float = 0.25
    DEFAULT_MIN_EV: float = 3.0  # porcentaje
    DEFAULT_MIN_RATING: float = 6.5

    # Simulaciones por deporte
    MLB_SIMULATIONS: int = 5_000_000
    NBA_SIMULATIONS: int = 50_000
    UFC_SIMULATIONS: int = 100_000

    # Game IDs de fallback para testing
    MLB_FALLBACK_GAME_ID: int = 746_929  # World Series 2024

    # Cache TTL
    ODDS_CACHE_TTL: int = 300  # segundos

    # Límites
    MAX_HISTORY_RECORDS: int = 200
    MAX_DISPLAY_RECORDS: int = 50

    def __post_init__(self) -> None:
        """Inicializa directorios derivados."""
        self.BASE_DIR = Path(__file__).parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.CACHE_DIR = self.BASE_DIR / ".cache"
        self.MODULES_DIR = self.BASE_DIR / "modules"


# Instancia global de configuración
CONFIG = AppConfig()

# Crear directorios necesarios
for directory in [CONFIG.DATA_DIR, CONFIG.CACHE_DIR, CONFIG.MODULES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Agregar módulos al path
sys.path.insert(0, str(CONFIG.MODULES_DIR))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# TYPE DEFINITIONS
# ============================================================


class GameData(TypedDict, total=False):
    """Estructura de datos de un juego/pelea."""

    home: str  # equipo local o Fighter 1
    away: str  # equipo visitante o Fighter 2
    home_odds: float
    away_odds: float
    commence_time: str
    raw_row: pd.Series


class PredictionData(TypedDict, total=False):
    """Estructura de datos de una predicción."""

    timestamp: str
    sport: str
    league: str
    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    pick_type: str
    pick_value: str
    ev: float
    kelly: float
    confidence: float
    rating: float
    model_version: str
    notes: str


class AnalysisResult(TypedDict, total=False):
    """Resultado genérico del análisis."""

    status: str
    game_info: Dict[str, Any]
    fight_info: Dict[str, Any]  # Para UFC
    probabilities: Dict[str, float]
    lambdas_history: Dict[str, Any]
    predictions: Dict[str, float]
    best_bets: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    error: str


# ============================================================
# CONFIGURACIÓN DE DEPORTES
# ============================================================


@dataclass
class SportConfig:
    """Configuración de un deporte."""

    name: str
    display_name: str
    module_name: str
    keywords: List[str]
    icon: str
    enabled: bool = True
    simulations: int = 50_000


SPORT_CONFIGS: Dict[str, SportConfig] = {
    "MLB": SportConfig(
        name="MLB",
        display_name="⚾ Baseball (MLB)",
        module_name="mlb_predictor",
        keywords=["baseball", "mlb"],
        icon="⚾",
        simulations=CONFIG.MLB_SIMULATIONS,
    ),
    "NBA": SportConfig(
        name="NBA",
        display_name="🏀 Basketball (NBA)",
        module_name="basketball_module",
        keywords=["basketball", "nba"],
        icon="🏀",
        simulations=CONFIG.NBA_SIMULATIONS,
    ),
    "UFC": SportConfig(
        name="UFC",
        display_name="🥊 UFC / MMA",
        module_name="ufc_module",
        keywords=["mma", "ufc"],
        icon="🥊",
        simulations=CONFIG.UFC_SIMULATIONS,
    ),
    "SOCCER": SportConfig(
        name="SOCCER",
        display_name="⚽ Soccer",
        module_name="football_module",
        keywords=["soccer", "football"],
        icon="⚽",
        enabled=False,  # En desarrollo
    ),
}


# ============================================================
# IMPORTS DINÁMICOS
# ============================================================


def safe_import(module_name: str, fallback: Any = None) -> Any:
    """Importa un módulo de forma segura con fallback."""
    try:
        import importlib

        return importlib.import_module(module_name)
    except ImportError as e:
        logger.warning(f"No se pudo importar {module_name}: {e}")
        return fallback


# Cargar odds_fetcher si existe
odds_fetcher = safe_import("odds_fetcher")
get_odds_data: Optional[Callable] = (
    getattr(odds_fetcher, "get_odds_data", None) if odds_fetcher else None
)


# ============================================================
# BASE DE DATOS
# ============================================================


class PredictionsDB:
    """Manejador de base de datos de predicciones optimizado."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Obtiene una conexión a la base de datos."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Inicializa el esquema con índices optimizados."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    sport TEXT NOT NULL,
                    league TEXT,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    p_home REAL,
                    p_draw REAL,
                    p_away REAL,
                    pick_type TEXT,
                    pick_value TEXT,
                    ev REAL,
                    kelly REAL,
                    confidence REAL,
                    rating REAL,
                    model_version TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Índices para consultas frecuentes
                CREATE INDEX IF NOT EXISTS idx_predictions_sport 
                    ON predictions(sport);
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
                    ON predictions(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_predictions_ev 
                    ON predictions(ev);
                CREATE INDEX IF NOT EXISTS idx_predictions_rating 
                    ON predictions(rating);
            """)

    def save(self, pred: PredictionData) -> int:
        """Guarda una predicción y retorna el ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions (
                    timestamp, sport, league, home_team, away_team,
                    p_home, p_draw, p_away, pick_type, pick_value,
                    ev, kelly, confidence, rating, model_version, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pred.get("timestamp", datetime.now().isoformat()),
                    pred.get("sport", ""),
                    pred.get("league", ""),
                    pred.get("home_team", ""),
                    pred.get("away_team", ""),
                    pred.get("p_home"),
                    pred.get("p_draw"),
                    pred.get("p_away"),
                    pred.get("pick_type", ""),
                    pred.get("pick_value", ""),
                    pred.get("ev"),
                    pred.get("kelly"),
                    pred.get("confidence"),
                    pred.get("rating"),
                    pred.get("model_version", CONFIG.APP_VERSION),
                    pred.get("notes", ""),
                ),
            )
            return int(cursor.lastrowid or 0)

    def read(
        self,
        sport: Optional[str] = None,
        limit: int = 100,
        min_ev: Optional[float] = None,
        min_rating: Optional[float] = None,
    ) -> pd.DataFrame:
        """Lee predicciones con filtros opcionales."""
        conditions: List[str] = []
        params: List[Any] = []

        if sport:
            conditions.append("sport = ?")
            params.append(sport)
        if min_ev is not None:
            conditions.append("ev >= ?")
            params.append(min_ev)
        if min_rating is not None:
            conditions.append("rating >= ?")
            params.append(min_rating)

        query = "SELECT * FROM predictions"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_stats(self, sport: str) -> Dict[str, Any]:
        """Obtiene estadísticas agregadas por deporte."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_picks,
                    COALESCE(AVG(ev), 0) as avg_ev,
                    COALESCE(AVG(rating), 0) as avg_rating,
                    COALESCE(MAX(ev), 0) as max_ev,
                    COALESCE(MIN(ev), 0) as min_ev,
                    COALESCE(SUM(CASE WHEN ev > 0 THEN 1 ELSE 0 END), 0) as positive_ev_count
                FROM predictions 
                WHERE sport = ?
                """,
                (sport,),
            ).fetchone()

            total = row["total_picks"]
            positive = row["positive_ev_count"]

            return {
                "total_picks": total,
                "avg_ev": row["avg_ev"],
                "avg_rating": row["avg_rating"],
                "max_ev": row["max_ev"],
                "min_ev": row["min_ev"],
                "positive_ev_count": positive,
                "positive_ev_rate": (positive / total if total > 0 else 0.0),
            }


# Instancia global de DB
db = PredictionsDB(CONFIG.DATA_DIR / "predictions_history.db")


# ============================================================
# FUNCIONES DE CÁLCULO
# ============================================================


def calculate_ev(odds: Optional[float], prob: Optional[float]) -> float:
    """
    Calcula el Expected Value (EV) de una apuesta.

    Args:
        odds: Cuota decimal del mercado
        prob: Probabilidad estimada por el modelo (0-1)

    Returns:
        EV como decimal (ej: 0.05 = 5% de edge)
    """
    if odds is None or prob is None:
        return float("nan")

    try:
        odds_f = float(odds)
        prob_f = float(prob)

        if np.isnan(odds_f) or np.isnan(prob_f):
            return float("nan")
        if odds_f <= 1.0 or not (0 < prob_f < 1):
            return float("nan")

        return (odds_f * prob_f) - 1.0
    except (ValueError, TypeError):
        return float("nan")


def calculate_kelly(
    odds: Optional[float],
    prob: Optional[float],
    fraction: float = 0.25,
    max_stake: float = 0.05,
) -> float:
    """
    Calcula el tamaño óptimo de apuesta usando Kelly Criterion.

    Args:
        odds: Cuota decimal del mercado
        prob: Probabilidad estimada (0-1)
        fraction: Fracción de Kelly (0.25 = Quarter Kelly)
        max_stake: Stake máximo como fracción del bankroll

    Returns:
        Stake recomendado (0 a max_stake)
    """
    if odds is None or prob is None:
        return 0.0

    try:
        odds_f = float(odds)
        prob_f = float(prob)

        if odds_f <= 1.0 or not (0 < prob_f < 1):
            return 0.0

        b = odds_f - 1.0
        q = 1.0 - prob_f
        kelly = ((prob_f * b) - q) / b

        return max(0.0, min(kelly * fraction, max_stake))
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


def calculate_rating(ev: float) -> float:
    """Calcula el rating (1-10) basado en el EV."""
    if np.isnan(ev):
        return 0.0
    if ev > 0.10:
        return 10.0
    if ev > 0.05:
        return 8.0
    if ev > 0.03:
        return 7.0
    if ev > 0.01:
        return 6.0
    return 5.0


# ============================================================
# COMPONENTES UI
# ============================================================


class UIComponents:
    """Componentes de UI reutilizables para Streamlit."""

    @staticmethod
    def apply_theme() -> None:
        """Aplica el tema oscuro personalizado."""
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-color: {ThemeColors.BG_DARK.value};
                    color: #E0E6ED;
                }}
                h1, h2, h3 {{
                    color: {ThemeColors.PRIMARY.value} !important;
                }}
                [data-testid="stMetricValue"] {{
                    color: {ThemeColors.PRIMARY.value} !important;
                    font-weight: 700 !important;
                }}
                .stButton>button {{
                    background-color: {ThemeColors.PRIMARY.value} !important;
                    color: black !important;
                    font-weight: 700 !important;
                    border-radius: 8px !important;
                    border: none !important;
                }}
                .stSelectbox, .stTextInput, .stNumberInput {{
                    background-color: #12161B !important;
                }}
                .value-card {{
                    padding: 10px;
                    background: {ThemeColors.BG_CARD.value};
                    border-radius: 8px;
                    margin: 5px 0;
                }}
                .value-card-success {{
                    border-left: 4px solid {ThemeColors.SUCCESS.value};
                }}
                .value-card-danger {{
                    border-left: 4px solid {ThemeColors.DANGER.value};
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_header() -> None:
        """Renderiza el header de la aplicación."""
        st.markdown(
            f"""
            <div style="text-align:center; padding: 20px 0;">
                <h1 style="color:{ThemeColors.PRIMARY.value}; margin:0;">
                    {CONFIG.PAGE_ICON} {CONFIG.APP_NAME}
                </h1>
                <p style="opacity:0.8; margin:5px 0;">
                    Sistema Cuantitativo Multideporte de Predicción
                </p>
                <p style="opacity:0.6; font-size:14px;">
                    Poisson · Bayes · Monte Carlo · Auto-ML · EV · Kelly
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_status_bar() -> None:
        """Renderiza la barra de estado de conexión."""
        cache_file = CONFIG.CACHE_DIR / "odds_last.json"
        api_key = os.getenv("ODDS_API_KEY", "").strip()

        if cache_file.exists() and api_key:
            status, color = "🌐 API Online + Caché", ThemeColors.SUCCESS.value
        elif api_key:
            status, color = "🌐 API Online", ThemeColors.PRIMARY.value
        elif cache_file.exists():
            status, color = "♻️ Solo Caché Local", ThemeColors.WARNING.value
        else:
            status, color = "❌ Sin Datos", ThemeColors.DANGER.value

        st.markdown(
            f"""
            <div style='text-align:center; background:{color}; padding:8px; 
                border-radius:8px; margin:10px 0;'>
                <b>{status}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_value_card(
        title: str,
        ev: float,
        kelly: float,
        odds: float,
        prob: float,
        min_ev: float,
    ) -> None:
        """Renderiza una tarjeta de valor para una apuesta."""
        is_value = ev > min_ev / 100
        card_class = "value-card-success" if is_value else "value-card-danger"
        color = ThemeColors.SUCCESS.value if is_value else ThemeColors.DANGER.value

        st.markdown(f"#### {title}")
        st.metric("Market Odds", f"{odds:.2f}")
        st.metric("Model Probability", f"{prob:.1%}")
        st.markdown(
            f"""
            <div class='value-card {card_class}'>
                <b>Expected Value:</b> 
                <span style='color:{color}'>{ev:+.1%}</span><br>
                <b>Kelly Stake:</b> {kelly:.1%}
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_footer() -> None:
        """Renderiza el footer de la aplicación."""
        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align:center; opacity:0.6; padding:20px;'>
                <p>{CONFIG.APP_NAME} © 2025 | BetMindex Dark Edition</p>
                <p style='font-size:12px;'>
                    Desarrollado con Streamlit + Python v{CONFIG.APP_VERSION} | 
                    Modelos: Poisson, Bayes, Monte Carlo
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FUNCIONES DE DATOS
# ============================================================


def safe_to_dataframe(data: Any) -> pd.DataFrame:
    """Convierte datos a DataFrame de forma segura."""
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data
    try:
        return pd.DataFrame(data)
    except Exception as e:
        logger.warning(f"Error convirtiendo a DataFrame: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=CONFIG.ODDS_CACHE_TTL)
def load_odds_data() -> pd.DataFrame:
    """Carga datos de odds con caché."""
    if get_odds_data is None:
        logger.warning("odds_fetcher no disponible")
        return pd.DataFrame()

    cache_file = CONFIG.CACHE_DIR / "odds_last.json"

    try:
        with st.spinner("Cargando odds..."):
            raw_data = get_odds_data()
            df = safe_to_dataframe(raw_data)

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(raw_data if isinstance(raw_data, list) else [], f)

            return df

    except Exception as e:
        logger.error(f"Error cargando odds: {e}")
        st.error(f"❌ Error cargando odds: {e}")

        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    cached = json.load(f)
                st.warning("⚠️ Usando datos en caché")
                return safe_to_dataframe(cached)
            except Exception as cache_error:
                logger.error(f"Error leyendo caché: {cache_error}")

        return pd.DataFrame()


def filter_odds_by_sport(df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    """Filtra DataFrame de odds por deporte usando keywords."""
    if df.empty:
        return df

    sport_columns = ["sport_key", "sport", "sport_title", "key", "league"]
    sport_col = next((col for col in sport_columns if col in df.columns), None)

    if sport_col is None:
        logger.warning("No se encontró columna de deporte en odds")
        return df

    try:
        mask = df[sport_col].astype(str).str.lower().apply(
            lambda x: any(keyword in x for keyword in keywords)
        )
        return df[mask].copy()
    except Exception as e:
        logger.error(f"Error filtrando odds: {e}")
        return df


def parse_game_datetime(commence_time: Any) -> str:
    """Parsea y formatea la fecha/hora de un juego."""
    try:
        if isinstance(commence_time, str):
            dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        return str(commence_time)
    except Exception:
        return str(commence_time)


def build_game_selector(
    sport_df: pd.DataFrame,
) -> tuple[List[str], Dict[str, GameData]]:
    """Construye las opciones del selector de juegos."""
    game_options: List[str] = []
    game_mapping: Dict[str, GameData] = {}

    for _, row in sport_df.iterrows():
        home = row.get("home_team", "Home")
        away = row.get("away_team", "Away")
        commence = row.get("commence_time", "")
        date_str = parse_game_datetime(commence)

        label = f"{away} @ {home} — {date_str}"
        game_options.append(label)

        game_mapping[label] = GameData(
            home=str(home),
            away=str(away),
            home_odds=float(row.get("home_odds", row.get("draw_odds", 2.0))),
            away_odds=float(row.get("away_odds", 2.0)),
            commence_time=str(commence),
            raw_row=row,
        )

    return game_options, game_mapping


# ============================================================
# ANALYZERS
# ============================================================


class BaseAnalyzer:
    """Clase base para analizadores de deportes."""

    def __init__(self, sport_config: SportConfig, settings: Dict[str, float]):
        self.config = sport_config
        self.settings = settings

    def analyze(self, game_data: GameData) -> AnalysisResult:
        """Método abstracto a implementar por subclases."""
        raise NotImplementedError

    def save_value_picks(
        self,
        game_info: Dict[str, str],
        probabilities: Dict[str, float],
        odds: Dict[str, float],
        notes: str = "",
    ) -> int:
        """Guarda picks con valor positivo (home/away ML)."""
        picks_saved = 0
        min_ev = self.settings.get("min_ev", CONFIG.DEFAULT_MIN_EV) / 100
        kelly_factor = self.settings.get("kelly_factor", CONFIG.DEFAULT_KELLY_FACTOR)

        home = game_info["home"]
        away = game_info["away"]
        p_home = probabilities["home_win"]
        p_away = probabilities["away_win"]

        # Evaluar Home
        ev_home = calculate_ev(odds["home"], p_home)
        if not np.isnan(ev_home) and ev_home > min_ev:
            db.save(
                PredictionData(
                    timestamp=datetime.now().isoformat(),
                    sport=self.config.name,
                    home_team=home,
                    away_team=away,
                    p_home=p_home,
                    p_away=p_away,
                    pick_type=f"{home} ML",
                    pick_value=str(odds["home"]),
                    ev=ev_home,
                    kelly=calculate_kelly(odds["home"], p_home, kelly_factor),
                    confidence=p_home,
                    rating=calculate_rating(ev_home),
                    model_version=f"{self.config.name} {CONFIG.APP_VERSION}",
                    notes=notes,
                )
            )
            picks_saved += 1

        # Evaluar Away
        ev_away = calculate_ev(odds["away"], p_away)
        if not np.isnan(ev_away) and ev_away > min_ev:
            db.save(
                PredictionData(
                    timestamp=datetime.now().isoformat(),
                    sport=self.config.name,
                    home_team=home,
                    away_team=away,
                    p_home=p_home,
                    p_away=p_away,
                    pick_type=f"{away} ML",
                    pick_value=str(odds["away"]),
                    ev=ev_away,
                    kelly=calculate_kelly(odds["away"], p_away, kelly_factor),
                    confidence=p_away,
                    rating=calculate_rating(ev_away),
                    model_version=f"{self.config.name} {CONFIG.APP_VERSION}",
                    notes=notes,
                )
            )
            picks_saved += 1

        return picks_saved


class MLBAnalyzer(BaseAnalyzer):
    """Analizador especializado para MLB."""

    @staticmethod
    def _fuzzy_match(name1: str, name2: str) -> bool:
        """Compara nombres de equipos de forma flexible."""
        n1, n2 = name1.lower(), name2.lower()
        return n1 in n2 or n2 in n1 or any(word in n2 for word in n1.split())

    def find_game_id(self, game_data: GameData) -> Optional[int]:
        """Busca el Game ID en MLB Stats API."""
        try:
            from data_fetchers import MLBDataIntegrator
            from datetime import datetime, timedelta
            integrator = MLBDataIntegrator()
            today = datetime.now().strftime("%Y-%m-%d")
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            today_games = integrator.mlb_api.get_todays_games(date=today) or []
            tomorrow_games = integrator.mlb_api.get_todays_games(date=tomorrow) or []
            all_games = today_games + tomorrow_games
            for game in all_games:
                mlb_home = game.get('home_team', '')
                mlb_away = game.get('away_team', '')
                home_match = self._fuzzy_match(game_data["home"], mlb_home)
                away_match = self._fuzzy_match(game_data["away"], mlb_away)
                if home_match and away_match:
                    return int(game.get('game_pk', 0))
            return None
        except Exception as e:
            logger.error(f"Error buscando Game ID: {e}")
            return None

    def analyze(self, game_data: GameData) -> AnalysisResult:
        """Ejecuta el análisis completo de MLB."""
        try:
            from modules.baseball_module.core.run_module import (
                run_module as run_mlb_analysis,
            )

            with st.spinner("🔍 Buscando Game ID en MLB Stats API..."):
                game_id = self.find_game_id(game_data)
                if game_id:
                    st.success(f"✅ Game ID encontrado: {game_id}")
                else:
                    st.warning("⚠️ No se encontró Game ID - usando modo simulado")
                    game_id = CONFIG.MLB_FALLBACK_GAME_ID

            with st.spinner("⚡ Ejecutando análisis MLB G10 Ultra Pro..."):
                result = run_mlb_analysis(
                    game_id=game_id,
                    use_calibration=True,
                    use_hfa=True,
                    use_pitcher=True,
                    use_regression=True,
                    analyze_f5=True,
                    n_max=self.config.simulations,
                )

            return result

        except ImportError as e:
            return AnalysisResult(status="error", error=f"Módulo no disponible: {e}")
        except Exception as e:
            logger.exception("Error en análisis MLB")
            return AnalysisResult(status="error", error=str(e))


class NBAAnalyzer(BaseAnalyzer):
    """Analizador especializado para NBA."""

    def analyze(self, game_data: GameData) -> AnalysisResult:
        """Ejecuta el análisis completo de NBA."""
        try:
            import basketball_module

            # Construir payload para el módulo
            data_payload = {
                "home_team": {"name": game_data["home"]},
                "away_team": {"name": game_data["away"]},
                "game_context": {
                    "game_date": game_data.get("commence_time", ""),
                    "home_b2b": False,
                    "away_b2b": False,
                    "away_travel_miles": 0,
                    "home_days_rest": 1,
                    "away_days_rest": 1,
                    "home_injuries": [],
                    "away_injuries": [],
                },
                "n_simulations": self.config.simulations,
            }

            with st.spinner("⚡ Ejecutando NBA MODULE G8+..."):
                # Intenta ambas firmas de función para compatibilidad
                try:
                    result = basketball_module.run_module(data=data_payload)
                except TypeError:
                    # Firma alternativa
                    result = basketball_module.run_module(
                        home_team=data_payload["home_team"],
                        away_team=data_payload["away_team"],
                        game_context=data_payload["game_context"],
                        n_simulations=data_payload["n_simulations"],
                    )

            return result

        except ImportError as e:
            return AnalysisResult(status="error", error=f"Módulo no disponible: {e}")
        except Exception as e:
            logger.exception("Error en análisis NBA")
            return AnalysisResult(status="error", error=str(e))


class UFCAnalyzer(BaseAnalyzer):
    """Analizador especializado para UFC/MMA."""

    def analyze(self, game_data: GameData) -> AnalysisResult:
        """Ejecuta el análisis completo de una pelea UFC."""
        try:
            import ufc_module

            fighter1_data = {"name": game_data["home"]}
            fighter2_data = {"name": game_data["away"]}

            with st.spinner("⚡ Ejecutando UFC MODULE G8+ Ultra..."):
                result = ufc_module.run_module(
                    fighter1_data=fighter1_data,
                    fighter2_data=fighter2_data,
                    weight_class="TBD",  # Se puede inferir de odds API
                    is_title_fight=False,
                    n_simulations=self.config.simulations,
                )

            return result

        except ImportError as e:
            return AnalysisResult(status="error", error=f"Módulo no disponible: {e}")
        except Exception as e:
            logger.exception("Error en análisis UFC")
            return AnalysisResult(status="error", error=str(e))


# ============================================================
# RENDERIZADO DE RESULTADOS
# ============================================================


def render_value_analysis(
    home: str,
    away: str,
    p_home: float,
    p_away: float,
    home_odds: float,
    away_odds: float,
    settings: Dict[str, float],
) -> None:
    """Renderiza el análisis de valor para ambos lados."""
    min_ev = settings.get("min_ev", CONFIG.DEFAULT_MIN_EV)
    kelly_factor = settings.get("kelly_factor", CONFIG.DEFAULT_KELLY_FACTOR)

    ev_home = calculate_ev(home_odds, p_home)
    ev_away = calculate_ev(away_odds, p_away)
    kelly_home = calculate_kelly(home_odds, p_home, kelly_factor)
    kelly_away = calculate_kelly(away_odds, p_away, kelly_factor)

    col1, col2 = st.columns(2)

    with col1:
        UIComponents.render_value_card(
            title=f"🏠 {home}",
            ev=ev_home,
            kelly=kelly_home,
            odds=home_odds,
            prob=p_home,
            min_ev=min_ev,
        )

    with col2:
        UIComponents.render_value_card(
            title=f"✈️ {away}",
            ev=ev_away,
            kelly=kelly_away,
            odds=away_odds,
            prob=p_away,
            min_ev=min_ev,
        )


def render_mlb_results(
    result: AnalysisResult,
    game_data: GameData,
    settings: Dict[str, float],
) -> None:
    """Renderiza resultados MLB."""
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

    lambdas = result.get("lambdas_history", {}).get("regression", {})
    lh = float(lambdas.get("lh", 0.0))
    la = float(lambdas.get("la", 0.0))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("λ Home Final", f"{lh:.3f}")
    with col2:
        st.metric("λ Away Final", f"{la:.3f}")

    st.markdown("---")

    probs = result.get("probabilities", {})
    p_home = float(probs.get("p_home", probs.get("home_win", 0.0)))
    p_away = float(probs.get("p_away", probs.get("away_win", 0.0)))
    total = float(probs.get("mean_total", probs.get("total_expected", lh + la)))

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
        home=home,
        away=away,
        p_home=p_home,
        p_away=p_away,
        home_odds=game_data["home_odds"],
        away_odds=game_data["away_odds"],
        settings=settings,
    )

    analyzer = MLBAnalyzer(SPORT_CONFIGS["MLB"], settings)
    picks_saved = analyzer.save_value_picks(
        game_info={"home": home, "away": away},
        probabilities={"home_win": p_home, "away_win": p_away},
        odds={"home": game_data["home_odds"], "away": game_data["away_odds"]},
        notes=f"λh={lh:.3f}, λa={la:.3f}",
    )

    if picks_saved > 0:
        st.success(f"✅ {picks_saved} picks guardadas en base de datos")
    else:
        st.info("ℹ️ No se encontraron value bets que cumplan los criterios mínimos")

    best_bets = result.get("best_bets", [])
    if best_bets:
        st.markdown("---")
        st.markdown("### 🎯 Additional Value Bets (from Model)")
        for i, bet in enumerate(best_bets[:3], 1):
            market = bet.get("market", "Unknown")
            ev = bet.get("ev", 0.0)
            rating = bet.get("rating", "C")
            color = (
                ThemeColors.SUCCESS.value
                if rating in ["A+", "A"]
                else ThemeColors.PRIMARY.value
            )
            st.markdown(
                f"""
                <div class='value-card' style='border-left:4px solid {color};'>
                    <b>#{i} {market}</b> [{rating}] - EV: 
                    <b style='color:{color}'>{ev:+.1f}%</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("🔍 Ver detalles técnicos"):
        st.json(result.get("metadata", {}))


def render_nba_results(
    result: AnalysisResult,
    game_data: GameData,
    settings: Dict[str, float],
) -> None:
    """Renderiza resultados NBA."""
    if result.get("status") != "success":
        st.error(f"❌ Error en módulo NBA: {result.get('error', 'Unknown')}")
        return

    st.success("✅ Análisis NBA completado")

    info = result["game_info"]
    preds = result["predictions"]
    probs = result["probabilities"]

    home = info["home_team"]
    away = info["away_team"]

    st.markdown(f"### 🏀 {away} @ {home} ({info.get('game_date', '')})")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"{home} pts", f"{preds['home_points']:.1f}")
    with c2:
        st.metric(f"{away} pts", f"{preds['away_points']:.1f}")
    with c3:
        st.metric("Total", f"{preds['total']:.1f}")

    st.markdown("---")
    st.markdown("### 🎯 Probabilidades (Moneyline)")

    p_home = float(probs["home_win"])
    p_away = float(probs["away_win"])

    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"🏠 {home}", f"{p_home:.1%}", delta=f"{(p_home-0.5)*100:+.1f}%")
    with c2:
        st.metric(f"✈️ {away}", f"{p_away:.1%}", delta=f"{(p_away-0.5)*100:+.1f}%")

    st.markdown("---")
    st.markdown("### 💰 Value Analysis (Moneyline)")

    render_value_analysis(
        home=home,
        away=away,
        p_home=p_home,
        p_away=p_away,
        home_odds=game_data["home_odds"],
        away_odds=game_data["away_odds"],
        settings=settings,
    )

    analyzer = NBAAnalyzer(SPORT_CONFIGS["NBA"], settings)
    picks_saved = analyzer.save_value_picks(
        game_info={"home": home, "away": away},
        probabilities={"home_win": p_home, "away_win": p_away},
        odds={"home": game_data["home_odds"], "away": game_data["away_odds"]},
    )

    if picks_saved > 0:
        st.success(f"✅ {picks_saved} picks NBA guardadas")
    else:
        st.info("ℹ️ No hubo EV suficiente para guardar picks NBA")

    with st.expander("🔍 Detalles técnicos (NBA)"):
        st.json(result.get("metadata", {}))


def render_ufc_results(
    result: AnalysisResult,
    game_data: GameData,
    settings: Dict[str, float],
) -> None:
    """Renderiza resultados UFC/MMA."""
    if result.get("status") != "success":
        st.error(f"❌ Error en módulo UFC: {result.get('error', 'Unknown')}")
        return

    st.success("✅ Análisis UFC completado")

    fight_info = result.get("fight_info", {})
    probs = result.get("probabilities", {})

    f1 = fight_info.get("fighter1", game_data["home"])
    f2 = fight_info.get("fighter2", game_data["away"])

    st.markdown(f"### 🥊 {f1} vs {f2}")

    # Probabilidades de victoria
    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"{f1} Win", f"{probs.get('fighter1_win', 0.0):.1%}")
    with c2:
        st.metric(f"{f2} Win", f"{probs.get('fighter2_win', 0.0):.1%}")

    st.markdown("---")
    st.markdown("### 💥 Método de Victoria")

    # Fighter 1 métodos
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(f"{f1} KO/TKO", f"{probs.get('fighter1_ko_tko', 0.0):.1%}")
    with c2:
        st.metric(f"{f1} SUB", f"{probs.get('fighter1_submission', 0.0):.1%}")
    with c3:
        st.metric(f"{f1} DEC", f"{probs.get('fighter1_decision', 0.0):.1%}")

    # Fighter 2 métodos
    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric(f"{f2} KO/TKO", f"{probs.get('fighter2_ko_tko', 0.0):.1%}")
    with c5:
        st.metric(f"{f2} SUB", f"{probs.get('fighter2_submission', 0.0):.1%}")
    with c6:
        st.metric(f"{f2} DEC", f"{probs.get('fighter2_decision', 0.0):.1%}")

    st.markdown("---")
    st.metric("Fight goes the distance", f"{probs.get('goes_distance', 0.0):.1%}")

    st.markdown("---")
    st.markdown("### 💰 Value Analysis (Moneyline)")

    o1 = float(game_data["home_odds"])
    o2 = float(game_data["away_odds"])
    p1 = float(probs.get("fighter1_win", 0.0))
    p2 = float(probs.get("fighter2_win", 0.0))

    min_ev = settings.get("min_ev", CONFIG.DEFAULT_MIN_EV)
    kelly_factor = settings.get("kelly_factor", CONFIG.DEFAULT_KELLY_FACTOR)

    ev1 = calculate_ev(o1, p1)
    ev2 = calculate_ev(o2, p2)
    k1 = calculate_kelly(o1, p1, kelly_factor)
    k2 = calculate_kelly(o2, p2, kelly_factor)

    c1, c2 = st.columns(2)
    with c1:
        UIComponents.render_value_card(
            title=f"🥊 {f1}",
            ev=ev1,
            kelly=k1,
            odds=o1,
            prob=p1,
            min_ev=min_ev,
        )
    with c2:
        UIComponents.render_value_card(
            title=f"🥊 {f2}",
            ev=ev2,
            kelly=k2,
            odds=o2,
            prob=p2,
            min_ev=min_ev,
        )

    # Guardar picks
    analyzer = UFCAnalyzer(SPORT_CONFIGS["UFC"], settings)
    picks_saved = analyzer.save_value_picks(
        game_info={"home": f1, "away": f2},
        probabilities={"home_win": p1, "away_win": p2},
        odds={"home": o1, "away": o2},
        notes="UFC MODULE G8+ Ultra",
    )

    if picks_saved > 0:
        st.success(f"✅ {picks_saved} picks UFC guardadas")
    else:
        st.info("ℹ️ No hubo EV suficiente para guardar picks UFC")

    # Best bets del modelo
    best_bets = result.get("best_bets", [])
    if best_bets:
        with st.expander("🎯 Best Bets del módulo UFC"):
            for bet in best_bets:
                st.write(
                    f"[{bet.get('rating', '')}] "
                    f"{bet.get('market', '')} "
                    f"({bet.get('probability', 0):.1%})"
                )

    with st.expander("🔍 Detalles técnicos UFC"):
        st.json(result.get("metadata", {}))


# ============================================================
# HISTORIAL
# ============================================================


def render_history(sport: str, settings: Dict[str, float]) -> None:
    """Renderiza el historial de predicciones."""
    st.subheader("📜 Historial de Predicciones")

    history_df = db.read(sport=sport, limit=CONFIG.MAX_HISTORY_RECORDS)

    if history_df.empty:
        st.info("No hay historial para este deporte")
        if st.button("❌ Cerrar Historial"):
            st.session_state["show_history"] = False
            st.rerun()  # ← CORRECTO (no experimental_rerun)
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Picks", len(history_df))
    with c2:
        avg_ev = history_df["ev"].mean() * 100 if "ev" in history_df.columns else 0.0
        st.metric("EV Promedio", f"{avg_ev:.2f}%")
    with c3:
        avg_rating = (
            history_df["rating"].mean() if "rating" in history_df.columns else 0.0
        )
        st.metric("Rating Promedio", f"{avg_rating:.1f}/10")
    with c4:
        if st.button("❌ Cerrar"):
            st.session_state["show_history"] = False
            st.rerun()  # ← CORRECTO (no experimental_rerun)

    display_cols = [
        "timestamp",
        "sport",
        "home_team",
        "away_team",
        "pick_type",
        "ev",
        "kelly",
        "rating",
    ]
    display_cols = [c for c in display_cols if c in history_df.columns]
    st.dataframe(
        history_df[display_cols].head(CONFIG.MAX_DISPLAY_RECORDS),
        use_container_width=True,
    )

    # Gráfico de EV
    if "ev" in history_df.columns and len(history_df) > 10:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            history_df.index,
            history_df["ev"] * 100,
            linewidth=2,
            color=ThemeColors.PRIMARY.value,
        )
        ax.axhline(y=0, linestyle="--", alpha=0.5, color="white")
        ax.set_xlabel("Pick #", color="white")
        ax.set_ylabel("EV (%)", color="white")
        ax.set_facecolor(ThemeColors.BG_DARK.value)
        fig.patch.set_facecolor(ThemeColors.BG_DARK.value)
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        st.pyplot(fig)
        plt.close(fig)


# ============================================================
# SIDEBAR
# ============================================================


def render_sidebar(sport_name: str) -> Dict[str, float]:
    """Renderiza el sidebar y retorna la configuración."""
    with st.sidebar:
        st.header("📊 Estadísticas Generales")
        stats = db.get_stats(sport_name)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Picks", stats["total_picks"])
        with c2:
            st.metric("EV+ Rate", f"{stats['positive_ev_rate']:.0%}")

        st.metric("EV Promedio", f"{stats['avg_ev']*100:.2f}%")
        st.metric("Rating Promedio", f"{stats['avg_rating']:.1f}/10")

        st.markdown("---")
        st.header("⚙️ Configuración")

        kelly_factor = st.slider(
            "Factor Kelly",
            min_value=0.1,
            max_value=0.5,
            value=CONFIG.DEFAULT_KELLY_FACTOR,
            step=0.05,
            help="Fracción del Kelly Criterion (0.25 = Quarter Kelly)",
        )

        min_ev = st.slider(
            "EV Mínimo (%)",
            min_value=0.0,
            max_value=10.0,
            value=CONFIG.DEFAULT_MIN_EV,
            step=0.5,
            help="Expected Value mínimo para considerar apuesta",
        )

        min_rating = st.slider(
            "Rating Mínimo",
            min_value=0.0,
            max_value=10.0,
            value=CONFIG.DEFAULT_MIN_RATING,
            step=0.5,
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
            "min_ev": min_ev,
            "min_rating": min_rating,
        }


# ============================================================
# MAIN APP
# ============================================================


def main() -> None:
    """Función principal de la aplicación."""
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

    # Cargar odds
    odds_df = load_odds_data()

    # Métricas generales
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("📦 Eventos Totales", len(odds_df))
    with c2:
        sports_count = (
            odds_df.get("sport_key", pd.Series(dtype=str)).nunique()
            if not odds_df.empty
            else 0
        )
        st.metric("🎯 Deportes Disponibles", int(sports_count))
    with c3:
        if st.button("🔄 Recargar Datos"):
            st.cache_data.clear()
            st.rerun()  # ← CORRECTO (no experimental_rerun)

    st.markdown("---")

    # Selector de deporte
    st.subheader("🎯 Selecciona el Deporte")

    sport_options = {cfg.display_name: name for name, cfg in SPORT_CONFIGS.items()}
    selected_display = st.selectbox("Deporte:", list(sport_options.keys()), index=0)
    sport_name = sport_options[selected_display]
    sport_config = SPORT_CONFIGS[sport_name]

    # Sidebar
    settings = render_sidebar(sport_name)

    # Filtrar odds
    sport_df = filter_odds_by_sport(odds_df, sport_config.keywords)
    st.info(f"🔍 **{sport_name}**: {len(sport_df)} eventos encontrados")

    if not sport_df.empty:
        with st.expander("📊 Ver datos crudos"):
            st.dataframe(sport_df.head(20), use_container_width=True)

    st.markdown("---")

    # Análisis
    st.subheader(f"{sport_config.icon} Análisis de {sport_name}")

    if not sport_config.enabled:
        st.warning(f"⚠️ Módulo de {sport_name} en desarrollo")
    elif sport_df.empty:
        st.warning(f"⚠️ No hay eventos {sport_name} disponibles en Odds API")
    else:
        st.success(f"✅ {len(sport_df)} eventos {sport_name} encontrados")

        game_options, game_mapping = build_game_selector(sport_df)

        if game_options:
            selected_label = st.selectbox(
                f"🎯 Selecciona el evento {sport_name} a analizar:",
                options=game_options,
                index=0,
            )

            game_data = game_mapping[selected_label]

            st.markdown("### 📊 Evento Seleccionado:")
            st.info(f"**{selected_label}**")

            c1, c2 = st.columns(2)
            with c1:
                st.metric(f"💵 {game_data['away']} Odds", f"{game_data['away_odds']:.2f}")
            with c2:
                st.metric(f"💵 {game_data['home']} Odds", f"{game_data['home_odds']:.2f}")

            st.markdown("---")

            if st.button(
                f"{sport_config.icon} Analizar Evento ({sport_name})",
                use_container_width=True,
            ):
                try:
                    if sport_name == "MLB":
                        analyzer = MLBAnalyzer(sport_config, settings)
                        result = analyzer.analyze(game_data)
                        render_mlb_results(result, game_data, settings)

                    elif sport_name == "NBA":
                        analyzer = NBAAnalyzer(sport_config, settings)
                        result = analyzer.analyze(game_data)
                        render_nba_results(result, game_data, settings)

                    elif sport_name == "UFC":
                        analyzer = UFCAnalyzer(sport_config, settings)
                        result = analyzer.analyze(game_data)
                        render_ufc_results(result, game_data, settings)

                    else:
                        st.warning(f"Analizador para {sport_name} no implementado")

                except Exception as e:
                    st.error(f"❌ Error ejecutando análisis: {e}")
                    logger.exception(f"Error en análisis {sport_name}")
                    st.exception(e)

    st.markdown("---")

    # Historial
    if st.session_state.get("show_history", False):
        render_history(sport_name, settings)

    UIComponents.render_footer()


if __name__ == "__main__":
    main()
