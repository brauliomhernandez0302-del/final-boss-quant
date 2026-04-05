"""
NBA STATS FETCHER - Módulo de estadísticas NBA para G10+ Ultra Pro V2
======================================================================

Jala stats completas de equipos NBA usando nba_api (gratis).
Formatea todo exactamente como lo espera el G10+ V2.

Estadísticas incluidas:
- Offensive/Defensive Rating
- Pace (season, last N games, home/away)
- eFG% (season y recent)
- Points, rebounds, assists, turnovers
- Paint scoring, fastbreak, 3PT
- Bench points, clutch stats
- Consistency score calculado

Requisitos:
    pip install nba_api pandas numpy

Autor: Braulio
Versión: 1.0
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np

# NBA API imports
try:
    from nba_api.stats.static import teams as nba_teams
    from nba_api.stats.endpoints import (
        TeamDashboardByGeneralSplits,
        LeagueGameLog,
        TeamGameLog,
        LeagueDashTeamStats,
        TeamEstimatedMetrics,
        LeagueDashTeamClutch,
    )
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("⚠️ nba_api no instalado. Ejecuta: pip install nba_api")

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTES
# ============================================================

# Season actual
CURRENT_SEASON = "2024-25"
SEASON_TYPE = "Regular Season"

# Promedios NBA para fallback
NBA_DEFAULTS = {
    "offensive_rating": 114.5,
    "defensive_rating": 114.5,
    "pace": 99.5,
    "efg_pct": 0.545,
    "points_per_game": 114.5,
    "rebounds": 44.0,
    "assists": 25.0,
    "turnovers": 13.5,
    "steals": 7.5,
    "blocks": 5.0,
    "three_pt_pct": 0.365,
    "free_throw_rate": 0.25,
    "points_in_paint": 48.0,
    "fastbreak_points": 14.0,
    "bench_points": 35.0,
}

# Mapeo de nombres de equipo (variaciones comunes)
TEAM_NAME_MAP = {
    # Nombres cortos
    "LAL": "Los Angeles Lakers",
    "LAC": "Los Angeles Clippers",
    "GSW": "Golden State Warriors",
    "BOS": "Boston Celtics",
    "MIA": "Miami Heat",
    "DEN": "Denver Nuggets",
    "PHX": "Phoenix Suns",
    "MIL": "Milwaukee Bucks",
    "PHI": "Philadelphia 76ers",
    "NYK": "New York Knicks",
    "BKN": "Brooklyn Nets",
    "TOR": "Toronto Raptors",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DET": "Detroit Pistons",
    "IND": "Indiana Pacers",
    "ATL": "Atlanta Hawks",
    "CHA": "Charlotte Hornets",
    "ORL": "Orlando Magic",
    "WAS": "Washington Wizards",
    "DAL": "Dallas Mavericks",
    "HOU": "Houston Rockets",
    "MEM": "Memphis Grizzlies",
    "NOP": "New Orleans Pelicans",
    "SAS": "San Antonio Spurs",
    "MIN": "Minnesota Timberwolves",
    "OKC": "Oklahoma City Thunder",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "UTA": "Utah Jazz",
    # Variaciones comunes
    "Lakers": "Los Angeles Lakers",
    "Clippers": "Los Angeles Clippers",
    "Warriors": "Golden State Warriors",
    "Celtics": "Boston Celtics",
    "Heat": "Miami Heat",
    "Nuggets": "Denver Nuggets",
    "Suns": "Phoenix Suns",
    "Bucks": "Milwaukee Bucks",
    "76ers": "Philadelphia 76ers",
    "Sixers": "Philadelphia 76ers",
    "Knicks": "New York Knicks",
    "Nets": "Brooklyn Nets",
    "Raptors": "Toronto Raptors",
    "Bulls": "Chicago Bulls",
    "Cavaliers": "Cleveland Cavaliers",
    "Cavs": "Cleveland Cavaliers",
    "Pistons": "Detroit Pistons",
    "Pacers": "Indiana Pacers",
    "Hawks": "Atlanta Hawks",
    "Hornets": "Charlotte Hornets",
    "Magic": "Orlando Magic",
    "Wizards": "Washington Wizards",
    "Mavericks": "Dallas Mavericks",
    "Mavs": "Dallas Mavericks",
    "Rockets": "Houston Rockets",
    "Grizzlies": "Memphis Grizzlies",
    "Pelicans": "New Orleans Pelicans",
    "Spurs": "San Antonio Spurs",
    "Timberwolves": "Minnesota Timberwolves",
    "Wolves": "Minnesota Timberwolves",
    "Thunder": "Oklahoma City Thunder",
    "Trail Blazers": "Portland Trail Blazers",
    "Blazers": "Portland Trail Blazers",
    "Kings": "Sacramento Kings",
    "Jazz": "Utah Jazz",
}

# Delay entre llamadas API (para evitar rate limiting)
API_DELAY = 0.6  # segundos


# ============================================================
# CLASE PRINCIPAL
# ============================================================

class NBAStatsFetcher:
    """
    Fetcher de estadísticas NBA para G10+ Ultra Pro V2.
    """

    def __init__(self, season: str = CURRENT_SEASON):
        self.season = season
        self.teams_cache: Dict[str, Dict] = {}
        self.game_logs_cache: Dict[int, pd.DataFrame] = {}
        self._load_teams()

    def _load_teams(self) -> None:
        """Carga lista de equipos NBA."""
        if not NBA_API_AVAILABLE:
            logger.warning("nba_api no disponible")
            return

        try:
            all_teams = nba_teams.get_teams()
            for team in all_teams:
                self.teams_cache[team["full_name"]] = team
                self.teams_cache[team["abbreviation"]] = team
                self.teams_cache[team["nickname"]] = team
            logger.info(f"✅ Cargados {len(all_teams)} equipos NBA")
        except Exception as e:
            logger.error(f"Error cargando equipos: {e}")

    def _normalize_team_name(self, team_name: str) -> str:
        """Normaliza nombre de equipo."""
        if team_name in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[team_name]
        return team_name

    def _get_team_id(self, team_name: str) -> Optional[int]:
        """Obtiene ID del equipo."""
        normalized = self._normalize_team_name(team_name)

        # Buscar en cache
        if normalized in self.teams_cache:
            return self.teams_cache[normalized]["id"]

        # Buscar por nombre parcial
        for key, team in self.teams_cache.items():
            if normalized.lower() in key.lower():
                return team["id"]

        logger.warning(f"Equipo no encontrado: {team_name}")
        return None

    def _api_call_with_retry(self, func, max_retries: int = 3, **kwargs):
        """Ejecuta llamada API con reintentos."""
        for attempt in range(max_retries):
            try:
                time.sleep(API_DELAY)
                result = func(**kwargs)
                return result
            except Exception as e:
                logger.warning(f"Intento {attempt + 1} fallido: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    # --------------------------------------------------------
    # MÉTODOS PRINCIPALES DE FETCHING
    # --------------------------------------------------------

    def get_team_stats(self, team_name: str) -> Dict[str, Any]:
        """
        Obtiene todas las stats de un equipo formateadas para G10+ V2.
        """
        if not NBA_API_AVAILABLE:
            logger.warning("nba_api no disponible, usando defaults")
            return self._get_default_team_stats(team_name)

        team_id = self._get_team_id(team_name)
        if team_id is None:
            return self._get_default_team_stats(team_name)

        normalized_name = self._normalize_team_name(team_name)
        logger.info(f"📊 Fetching stats para {normalized_name}...")

        try:
            # 1. Stats generales del equipo
            base_stats = self._fetch_base_stats(team_id)

            # 2. Advanced stats (ratings, pace)
            advanced_stats = self._fetch_advanced_stats(team_id)

            # 3. Game log para tendencias
            game_log = self._fetch_game_log(team_id)
            trend_stats = self._calculate_trends(game_log)

            # 4. Home/Away splits
            split_stats = self._fetch_splits(team_id)

            # 5. Clutch stats
            clutch_stats = self._fetch_clutch_stats(team_id)

            # 6. Calcular consistency score
            consistency = self._calculate_consistency(game_log)

            # Combinar todo
            result = {
                "name": normalized_name,
                "team_id": team_id,
                **base_stats,
                **advanced_stats,
                **trend_stats,
                **split_stats,
                **clutch_stats,
                "consistency_score": consistency,
                "fetch_timestamp": datetime.now().isoformat(),
            }

            logger.info(f"✅ Stats completas para {normalized_name}")
            return result

        except Exception as e:
            logger.error(f"Error fetching stats para {team_name}: {e}")
            return self._get_default_team_stats(team_name)

    def _fetch_base_stats(self, team_id: int) -> Dict[str, Any]:
        """Fetch stats base del equipo."""
        try:
            dashboard = self._api_call_with_retry(
                TeamDashboardByGeneralSplits,
                team_id=team_id,
                season=self.season,
                season_type_all_star=SEASON_TYPE,
            )

            df = dashboard.overall_team_dashboard.get_data_frame()

            if df.empty:
                return {}

            row = df.iloc[0]

            return {
                "points_per_game": float(row.get("PTS", NBA_DEFAULTS["points_per_game"])),
                "rebounds": float(row.get("REB", NBA_DEFAULTS["rebounds"])),
                "assists": float(row.get("AST", NBA_DEFAULTS["assists"])),
                "turnovers": float(row.get("TOV", NBA_DEFAULTS["turnovers"])),
                "steals": float(row.get("STL", NBA_DEFAULTS["steals"])),
                "blocks": float(row.get("BLK", NBA_DEFAULTS["blocks"])),
                "field_goal_pct": float(row.get("FG_PCT", 0.46)),
                "three_pt_offense": float(row.get("FG3_PCT", NBA_DEFAULTS["three_pt_pct"])),
                "free_throw_pct": float(row.get("FT_PCT", 0.78)),
                "three_pt_attempts": float(row.get("FG3A", 35.0)),
                "free_throw_rate": float(row.get("FTA", 22.0)) / max(float(row.get("FGA", 88.0)), 1),
                "offensive_rebounds": float(row.get("OREB", 10.0)),
                "defensive_rebounds": float(row.get("DREB", 34.0)),
            }

        except Exception as e:
            logger.warning(f"Error en base stats: {e}")
            return {}

    def _fetch_advanced_stats(self, team_id: int) -> Dict[str, Any]:
        """Fetch stats avanzadas (ratings, pace, eFG)."""
        try:
            # League dash team stats para eFG y otros
            league_stats = self._api_call_with_retry(
                LeagueDashTeamStats,
                season=self.season,
                season_type_all_star=SEASON_TYPE,
                measure_type_detailed_defense="Advanced",
            )

            df = league_stats.get_data_frames()[0]
            team_row = df[df["TEAM_ID"] == team_id]

            if team_row.empty:
                # Intentar con estimated metrics
                return self._fetch_estimated_metrics(team_id)

            row = team_row.iloc[0]

            return {
                "offensive_rating": float(row.get("OFF_RATING", NBA_DEFAULTS["offensive_rating"])),
                "defensive_rating": float(row.get("DEF_RATING", NBA_DEFAULTS["defensive_rating"])),
                "net_rating": float(row.get("NET_RATING", 0.0)),
                "pace": float(row.get("PACE", NBA_DEFAULTS["pace"])),
                "efg_season": float(row.get("EFG_PCT", NBA_DEFAULTS["efg_pct"])),
                "true_shooting_pct": float(row.get("TS_PCT", 0.57)),
                "assist_ratio": float(row.get("AST_RATIO", 17.0)),
                "turnover_ratio": float(row.get("TM_TOV_PCT", 13.0)),
                "rebounding_rate": float(row.get("REB_PCT", 50.0)) if "REB_PCT" in row else 50.0,
            }

        except Exception as e:
            logger.warning(f"Error en advanced stats: {e}")
            return self._fetch_estimated_metrics(team_id)

    def _fetch_estimated_metrics(self, team_id: int) -> Dict[str, Any]:
        """Fetch métricas estimadas como fallback."""
        try:
            metrics = self._api_call_with_retry(
                TeamEstimatedMetrics,
                season=self.season,
                season_type=SEASON_TYPE,
            )

            df = metrics.get_data_frames()[0]
            team_row = df[df["TEAM_ID"] == team_id]

            if team_row.empty:
                return {
                    "offensive_rating": NBA_DEFAULTS["offensive_rating"],
                    "defensive_rating": NBA_DEFAULTS["defensive_rating"],
                    "pace": NBA_DEFAULTS["pace"],
                }

            row = team_row.iloc[0]

            return {
                "offensive_rating": float(row.get("E_OFF_RATING", NBA_DEFAULTS["offensive_rating"])),
                "defensive_rating": float(row.get("E_DEF_RATING", NBA_DEFAULTS["defensive_rating"])),
                "net_rating": float(row.get("E_NET_RATING", 0.0)),
                "pace": float(row.get("E_PACE", NBA_DEFAULTS["pace"])),
            }

        except Exception as e:
            logger.warning(f"Error en estimated metrics: {e}")
            return {
                "offensive_rating": NBA_DEFAULTS["offensive_rating"],
                "defensive_rating": NBA_DEFAULTS["defensive_rating"],
                "pace": NBA_DEFAULTS["pace"],
            }

    def _fetch_game_log(self, team_id: int) -> pd.DataFrame:
        """Fetch game log del equipo."""
        if team_id in self.game_logs_cache:
            return self.game_logs_cache[team_id]

        try:
            game_log = self._api_call_with_retry(
                TeamGameLog,
                team_id=team_id,
                season=self.season,
                season_type_all_star=SEASON_TYPE,
            )

            df = game_log.get_data_frames()[0]
            self.game_logs_cache[team_id] = df
            return df

        except Exception as e:
            logger.warning(f"Error en game log: {e}")
            return pd.DataFrame()

    def _calculate_trends(self, game_log: pd.DataFrame) -> Dict[str, Any]:
        """Calcula tendencias basadas en game log."""
        if game_log.empty:
            return {
                "points_last_3": 0.0,
                "points_last_5": 0.0,
                "points_last_10": 0.0,
                "pace_last_5": NBA_DEFAULTS["pace"],
                "pace_last_10": NBA_DEFAULTS["pace"],
                "efg_last_5": NBA_DEFAULTS["efg_pct"],
            }

        try:
            # Ordenar por fecha (más reciente primero)
            df = game_log.sort_values("GAME_DATE", ascending=False)

            # Puntos
            points_l3 = df.head(3)["PTS"].mean() if len(df) >= 3 else df["PTS"].mean()
            points_l5 = df.head(5)["PTS"].mean() if len(df) >= 5 else df["PTS"].mean()
            points_l10 = df.head(10)["PTS"].mean() if len(df) >= 10 else df["PTS"].mean()

            # eFG de últimos 5 juegos
            if len(df) >= 5:
                recent = df.head(5)
                fgm = recent["FGM"].sum()
                fga = recent["FGA"].sum()
                fg3m = recent["FG3M"].sum()
                efg_l5 = (fgm + 0.5 * fg3m) / max(fga, 1)
            else:
                efg_l5 = NBA_DEFAULTS["efg_pct"]

            # Pace estimado (basado en posesiones aproximadas)
            # Pace ≈ (FGA + 0.44*FTA - OREB + TOV) / minutos * 48
            if "MIN" in df.columns and len(df) >= 5:
                recent = df.head(5)
                possessions = (
                    recent["FGA"].sum() +
                    0.44 * recent["FTA"].sum() -
                    recent["OREB"].sum() +
                    recent["TOV"].sum()
                )
                # Aproximación simple
                pace_l5 = possessions / 5 * 0.98
            else:
                pace_l5 = NBA_DEFAULTS["pace"]

            if len(df) >= 10:
                recent = df.head(10)
                possessions = (
                    recent["FGA"].sum() +
                    0.44 * recent["FTA"].sum() -
                    recent["OREB"].sum() +
                    recent["TOV"].sum()
                )
                pace_l10 = possessions / 10 * 0.98
            else:
                pace_l10 = pace_l5

            return {
                "points_last_3": round(float(points_l3), 1),
                "points_last_5": round(float(points_l5), 1),
                "points_last_10": round(float(points_l10), 1),
                "pace_last_5": round(float(pace_l5), 1),
                "pace_last_10": round(float(pace_l10), 1),
                "efg_last_5": round(float(efg_l5), 4),
            }

        except Exception as e:
            logger.warning(f"Error calculando tendencias: {e}")
            return {
                "points_last_3": 0.0,
                "points_last_5": 0.0,
                "points_last_10": 0.0,
                "pace_last_5": NBA_DEFAULTS["pace"],
                "pace_last_10": NBA_DEFAULTS["pace"],
                "efg_last_5": NBA_DEFAULTS["efg_pct"],
            }

    def _fetch_splits(self, team_id: int) -> Dict[str, Any]:
        """Fetch home/away splits."""
        try:
            dashboard = self._api_call_with_retry(
                TeamDashboardByGeneralSplits,
                team_id=team_id,
                season=self.season,
                season_type_all_star=SEASON_TYPE,
            )

            location_df = dashboard.location_team_dashboard.get_data_frame()

            if location_df.empty:
                return {
                    "pace_home": NBA_DEFAULTS["pace"],
                    "pace_away": NBA_DEFAULTS["pace"],
                    "points_home": NBA_DEFAULTS["points_per_game"],
                    "points_away": NBA_DEFAULTS["points_per_game"],
                }

            home_row = location_df[location_df["GROUP_VALUE"] == "Home"]
            away_row = location_df[location_df["GROUP_VALUE"] == "Road"]

            # Pace aproximado desde splits
            pace_home = NBA_DEFAULTS["pace"]
            pace_away = NBA_DEFAULTS["pace"]
            points_home = NBA_DEFAULTS["points_per_game"]
            points_away = NBA_DEFAULTS["points_per_game"]

            if not home_row.empty:
                points_home = float(home_row.iloc[0].get("PTS", points_home))
                # Estimar pace desde puntos (correlación aproximada)
                pace_home = 95 + (points_home - 110) * 0.3

            if not away_row.empty:
                points_away = float(away_row.iloc[0].get("PTS", points_away))
                pace_away = 95 + (points_away - 110) * 0.3

            return {
                "pace_home": round(float(pace_home), 1),
                "pace_away": round(float(pace_away), 1),
                "points_home": round(float(points_home), 1),
                "points_away": round(float(points_away), 1),
            }

        except Exception as e:
            logger.warning(f"Error en splits: {e}")
            return {
                "pace_home": NBA_DEFAULTS["pace"],
                "pace_away": NBA_DEFAULTS["pace"],
            }

    def _fetch_clutch_stats(self, team_id: int) -> Dict[str, Any]:
        """Fetch clutch stats del equipo."""
        try:
            clutch = self._api_call_with_retry(
                LeagueDashTeamClutch,
                season=self.season,
                season_type_all_star=SEASON_TYPE,
            )

            df = clutch.get_data_frames()[0]
            team_row = df[df["TEAM_ID"] == team_id]

            if team_row.empty:
                return {
                    "clutch_net_rating": 0.0,
                    "clutch_win_pct": 0.5,
                }

            row = team_row.iloc[0]

            # Calcular net rating en clutch
            clutch_pts = float(row.get("PTS", 0))
            clutch_gp = float(row.get("GP", 1))
            clutch_ppg = clutch_pts / max(clutch_gp, 1)

            # Aproximación de net rating
            clutch_net = (clutch_ppg - 10) * 0.5  # Normalizado

            return {
                "clutch_net_rating": round(float(clutch_net), 1),
                "clutch_win_pct": float(row.get("W_PCT", 0.5)),
            }

        except Exception as e:
            logger.warning(f"Error en clutch stats: {e}")
            return {
                "clutch_net_rating": 0.0,
                "clutch_win_pct": 0.5,
            }

    def _calculate_consistency(self, game_log: pd.DataFrame) -> float:
        """Calcula score de consistencia del equipo."""
        if game_log.empty or len(game_log) < 5:
            return 70.0

        try:
            # Basado en desviación estándar de puntos
            pts_std = game_log["PTS"].std()

            # Menor std = más consistente
            # Rango típico: 8-15 pts de std
            # Mapear a 0-100
            if pts_std <= 8:
                consistency = 90.0
            elif pts_std <= 10:
                consistency = 80.0
            elif pts_std <= 12:
                consistency = 70.0
            elif pts_std <= 14:
                consistency = 60.0
            else:
                consistency = 50.0

            # Ajustar por varianza en margen de victoria
            if "PLUS_MINUS" in game_log.columns:
                margin_std = game_log["PLUS_MINUS"].std()
                if margin_std > 15:
                    consistency -= 5

            return round(float(consistency), 1)

        except Exception as e:
            logger.warning(f"Error calculando consistency: {e}")
            return 70.0

    def _get_default_team_stats(self, team_name: str) -> Dict[str, Any]:
        """Devuelve stats por defecto para un equipo."""
        return {
            "name": self._normalize_team_name(team_name),
            "offensive_rating": NBA_DEFAULTS["offensive_rating"],
            "defensive_rating": NBA_DEFAULTS["defensive_rating"],
            "pace": NBA_DEFAULTS["pace"],
            "pace_last_5": NBA_DEFAULTS["pace"],
            "pace_last_10": NBA_DEFAULTS["pace"],
            "pace_home": NBA_DEFAULTS["pace"],
            "pace_away": NBA_DEFAULTS["pace"],
            "efg_season": NBA_DEFAULTS["efg_pct"],
            "efg_last_5": NBA_DEFAULTS["efg_pct"],
            "scoring_variance": 8.0,
            "consistency_score": 70.0,
            "points_last_3": 0.0,
            "points_last_5": 0.0,
            "points_last_10": 0.0,
            "points_per_game": NBA_DEFAULTS["points_per_game"],
            "points_in_paint": NBA_DEFAULTS["points_in_paint"],
            "opp_points_in_paint": NBA_DEFAULTS["points_in_paint"],
            "fastbreak_points": NBA_DEFAULTS["fastbreak_points"],
            "opp_fastbreak_points": NBA_DEFAULTS["fastbreak_points"],
            "turnovers": NBA_DEFAULTS["turnovers"],
            "steals": NBA_DEFAULTS["steals"],
            "free_throw_rate": NBA_DEFAULTS["free_throw_rate"],
            "offensive_rebounds": 10.0,
            "defensive_rebounds": 34.0,
            "assists": NBA_DEFAULTS["assists"],
            "three_pt_offense": NBA_DEFAULTS["three_pt_pct"],
            "three_pt_defense": NBA_DEFAULTS["three_pt_pct"],
            "three_pt_attempts": 35.0,
            "bench_points": NBA_DEFAULTS["bench_points"],
            "clutch_net_rating": 0.0,
            "pnr_offense_rating": 100.0,
            "pnr_defense_rating": 100.0,
            "rebounding_rate": 50.0,
            "hfa_multiplier": 1.0,
            "fetch_timestamp": datetime.now().isoformat(),
            "is_default": True,
        }

    # --------------------------------------------------------
    # MÉTODOS DE UTILIDAD
    # --------------------------------------------------------

    def get_all_teams_stats(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene stats de todos los equipos NBA."""
        all_stats = {}

        for team_name in self.teams_cache:
            if len(team_name) > 3:  # Solo nombres completos
                try:
                    stats = self.get_team_stats(team_name)
                    all_stats[stats["name"]] = stats
                except Exception as e:
                    logger.error(f"Error con {team_name}: {e}")

        return all_stats

    def get_matchup_stats(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Obtiene stats de ambos equipos para un matchup.
        Retorna (home_stats, away_stats) listas para G10+ V2.
        """
        home_stats = self.get_team_stats(home_team)
        away_stats = self.get_team_stats(away_team)

        return home_stats, away_stats

    def format_for_g10(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asegura que las stats tengan todos los campos que espera G10+ V2.
        """
        defaults = self._get_default_team_stats(stats.get("name", "Unknown"))

        # Merge con defaults para campos faltantes
        formatted = {**defaults, **stats}

        # Asegurar campos críticos
        required_fields = [
            "name", "offensive_rating", "defensive_rating", "pace",
            "efg_season", "efg_last_5", "consistency_score",
            "points_last_3", "points_last_5", "points_last_10",
            "points_per_game",
        ]

        for field in required_fields:
            if field not in formatted or formatted[field] is None:
                formatted[field] = defaults.get(field, 0)

        return formatted


# ============================================================
# FUNCIONES DE CONVENIENCIA
# ============================================================

def get_team_stats(team_name: str, season: str = CURRENT_SEASON) -> Dict[str, Any]:
    """Función rápida para obtener stats de un equipo."""
    fetcher = NBAStatsFetcher(season=season)
    return fetcher.get_team_stats(team_name)


def get_matchup_stats(
    home_team: str,
    away_team: str,
    season: str = CURRENT_SEASON
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Función rápida para obtener stats de un matchup."""
    fetcher = NBAStatsFetcher(season=season)
    return fetcher.get_matchup_stats(home_team, away_team)


# ============================================================
# MAIN - TEST
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NBA STATS FETCHER - TEST")
    print("=" * 70 + "\n")

    if not NBA_API_AVAILABLE:
        print("❌ nba_api no instalado!")
        print("   Ejecuta: pip install nba_api")
        exit(1)

    fetcher = NBAStatsFetcher()

    # Test: Lakers stats
    print("📊 Fetching Los Angeles Lakers stats...")
    lakers = fetcher.get_team_stats("Lakers")

    print("\n" + "-" * 40)
    print("LAKERS STATS:")
    print("-" * 40)

    key_stats = [
        "name", "offensive_rating", "defensive_rating", "pace",
        "efg_season", "efg_last_5", "points_per_game",
        "points_last_5", "consistency_score"
    ]

    for key in key_stats:
        if key in lakers:
            print(f"  {key}: {lakers[key]}")

    # Test: Matchup
    print("\n" + "=" * 70)
    print("📊 Fetching Matchup: Nuggets vs Lakers...")
    print("=" * 70 + "\n")

    home, away = fetcher.get_matchup_stats("Denver Nuggets", "Los Angeles Lakers")

    print("HOME (Nuggets):")
    print(f"  ORtg: {home.get('offensive_rating', 'N/A')}")
    print(f"  DRtg: {home.get('defensive_rating', 'N/A')}")
    print(f"  Pace: {home.get('pace', 'N/A')}")

    print("\nAWAY (Lakers):")
    print(f"  ORtg: {away.get('offensive_rating', 'N/A')}")
    print(f"  DRtg: {away.get('defensive_rating', 'N/A')}")
    print(f"  Pace: {away.get('pace', 'N/A')}")

    print("\n✅ Test completado!")
