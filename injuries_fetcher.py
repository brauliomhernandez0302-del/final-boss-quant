"""
NBA INJURIES FETCHER - Módulo de lesiones para G10+ Ultra Pro V2
=================================================================

Scrapea lesiones de ESPN y otras fuentes.
Clasifica jugadores por nivel de impacto automáticamente.
Detecta B2B, días de descanso, y contexto de schedule.

Funcionalidades:
- Scraping de ESPN NBA Injury Report
- Clasificación automática de jugadores (MVP, superstar, star, etc.)
- Detección de Back-to-Back games
- Cálculo de días de descanso
- Detección de viajes y timezone
- Formato listo para G10+ V2

Requisitos:
    pip install requests beautifulsoup4 pandas

Autor: Braulio
Versión: 1.0
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    print("⚠️ Instala: pip install requests beautifulsoup4")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTES
# ============================================================

# URLs de fuentes
ESPN_INJURIES_URL = "https://www.espn.com/nba/injuries"
ESPN_SCHEDULE_URL = "https://www.espn.com/nba/schedule"
ESPN_TEAM_SCHEDULE = "https://www.espn.com/nba/team/schedule/_/name/{team_abbr}"

# Headers para requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Clasificación de jugadores por nivel
# MVP Candidates / Top 10 players
MVP_PLAYERS = [
    "Nikola Jokic", "Luka Doncic", "Giannis Antetokounmpo", "Joel Embiid",
    "Shai Gilgeous-Alexander", "Jayson Tatum", "Kevin Durant", "Stephen Curry",
    "LeBron James", "Anthony Davis", "Ja Morant", "Trae Young",
]

# All-Stars / Superstar level
SUPERSTAR_PLAYERS = [
    "Damian Lillard", "Jimmy Butler", "Kyrie Irving", "Kawhi Leonard",
    "Paul George", "Donovan Mitchell", "Devin Booker", "Zion Williamson",
    "Tyrese Haliburton", "De'Aaron Fox", "Domantas Sabonis", "Bam Adebayo",
    "Anthony Edwards", "Lauri Markkanen", "Paolo Banchero", "Chet Holmgren",
    "Victor Wembanyama", "Tyrese Maxey", "Jalen Brunson", "Julius Randle",
    "Karl-Anthony Towns", "Rudy Gobert", "Jaren Jackson Jr.", "Desmond Bane",
]

# Star players (high impact starters)
STAR_PLAYERS = [
    "Brandon Ingram", "CJ McCollum", "Khris Middleton", "Brook Lopez",
    "Mikal Bridges", "OG Anunoby", "Pascal Siakam", "Scottie Barnes",
    "Fred VanVleet", "Alperen Sengun", "Jalen Green", "Cade Cunningham",
    "Franz Wagner", "Dejounte Murray", "Trey Murphy III", "Herbert Jones",
    "Derrick White", "Jaylen Brown", "Al Horford", "Kristaps Porzingis",
    "Myles Turner", "Buddy Hield", "Terry Rozier", "LaMelo Ball",
    "Miles Bridges", "Jalen Williams", "Chet Holmgren", "Josh Giddey",
    "Anfernee Simons", "Jerami Grant", "Deandre Ayton", "Bradley Beal",
    "Chris Paul", "Draymond Green", "Andrew Wiggins", "Jordan Poole",
    "RJ Barrett", "Immanuel Quickley", "Evan Mobley", "Jarrett Allen",
    "Darius Garland", "Coby White", "Zach LaVine", "DeMar DeRozan",
    "Nikola Vucevic", "Alex Caruso",
]

# Mapeo de equipos ESPN
TEAM_ABBR_MAP = {
    "Los Angeles Lakers": "lal",
    "Los Angeles Clippers": "lac",
    "Golden State Warriors": "gs",
    "Boston Celtics": "bos",
    "Miami Heat": "mia",
    "Denver Nuggets": "den",
    "Phoenix Suns": "phx",
    "Milwaukee Bucks": "mil",
    "Philadelphia 76ers": "phi",
    "New York Knicks": "ny",
    "Brooklyn Nets": "bkn",
    "Toronto Raptors": "tor",
    "Chicago Bulls": "chi",
    "Cleveland Cavaliers": "cle",
    "Detroit Pistons": "det",
    "Indiana Pacers": "ind",
    "Atlanta Hawks": "atl",
    "Charlotte Hornets": "cha",
    "Orlando Magic": "orl",
    "Washington Wizards": "wsh",
    "Dallas Mavericks": "dal",
    "Houston Rockets": "hou",
    "Memphis Grizzlies": "mem",
    "New Orleans Pelicans": "no",
    "San Antonio Spurs": "sa",
    "Minnesota Timberwolves": "min",
    "Oklahoma City Thunder": "okc",
    "Portland Trail Blazers": "por",
    "Sacramento Kings": "sac",
    "Utah Jazz": "utah",
    # Abreviaciones
    "LAL": "lal", "LAC": "lac", "GSW": "gs", "BOS": "bos",
    "MIA": "mia", "DEN": "den", "PHX": "phx", "MIL": "mil",
    "PHI": "phi", "NYK": "ny", "BKN": "bkn", "TOR": "tor",
    "CHI": "chi", "CLE": "cle", "DET": "det", "IND": "ind",
    "ATL": "atl", "CHA": "cha", "ORL": "orl", "WAS": "wsh",
    "DAL": "dal", "HOU": "hou", "MEM": "mem", "NOP": "no",
    "SAS": "sa", "MIN": "min", "OKC": "okc", "POR": "por",
    "SAC": "sac", "UTA": "utah",
    # Variaciones comunes
    "Lakers": "lal", "Clippers": "lac", "Warriors": "gs",
    "Celtics": "bos", "Heat": "mia", "Nuggets": "den",
    "Suns": "phx", "Bucks": "mil", "76ers": "phi", "Sixers": "phi",
    "Knicks": "ny", "Nets": "bkn", "Raptors": "tor",
    "Bulls": "chi", "Cavaliers": "cle", "Cavs": "cle",
    "Pistons": "det", "Pacers": "ind", "Hawks": "atl",
    "Hornets": "cha", "Magic": "orl", "Wizards": "wsh",
    "Mavericks": "dal", "Mavs": "dal", "Rockets": "hou",
    "Grizzlies": "mem", "Pelicans": "no", "Spurs": "sa",
    "Timberwolves": "min", "Wolves": "min", "Thunder": "okc",
    "Trail Blazers": "por", "Blazers": "por", "Kings": "sac",
    "Jazz": "utah",
}

# Ciudades y timezone
TEAM_LOCATIONS = {
    "lal": {"city": "Los Angeles", "timezone": "PST", "tz_offset": -8},
    "lac": {"city": "Los Angeles", "timezone": "PST", "tz_offset": -8},
    "gs": {"city": "San Francisco", "timezone": "PST", "tz_offset": -8},
    "phx": {"city": "Phoenix", "timezone": "MST", "tz_offset": -7},
    "sac": {"city": "Sacramento", "timezone": "PST", "tz_offset": -8},
    "por": {"city": "Portland", "timezone": "PST", "tz_offset": -8},
    "utah": {"city": "Salt Lake City", "timezone": "MST", "tz_offset": -7},
    "den": {"city": "Denver", "timezone": "MST", "tz_offset": -7},
    "okc": {"city": "Oklahoma City", "timezone": "CST", "tz_offset": -6},
    "dal": {"city": "Dallas", "timezone": "CST", "tz_offset": -6},
    "sa": {"city": "San Antonio", "timezone": "CST", "tz_offset": -6},
    "hou": {"city": "Houston", "timezone": "CST", "tz_offset": -6},
    "no": {"city": "New Orleans", "timezone": "CST", "tz_offset": -6},
    "mem": {"city": "Memphis", "timezone": "CST", "tz_offset": -6},
    "min": {"city": "Minneapolis", "timezone": "CST", "tz_offset": -6},
    "mil": {"city": "Milwaukee", "timezone": "CST", "tz_offset": -6},
    "chi": {"city": "Chicago", "timezone": "CST", "tz_offset": -6},
    "ind": {"city": "Indianapolis", "timezone": "EST", "tz_offset": -5},
    "det": {"city": "Detroit", "timezone": "EST", "tz_offset": -5},
    "cle": {"city": "Cleveland", "timezone": "EST", "tz_offset": -5},
    "atl": {"city": "Atlanta", "timezone": "EST", "tz_offset": -5},
    "cha": {"city": "Charlotte", "timezone": "EST", "tz_offset": -5},
    "orl": {"city": "Orlando", "timezone": "EST", "tz_offset": -5},
    "mia": {"city": "Miami", "timezone": "EST", "tz_offset": -5},
    "wsh": {"city": "Washington", "timezone": "EST", "tz_offset": -5},
    "phi": {"city": "Philadelphia", "timezone": "EST", "tz_offset": -5},
    "ny": {"city": "New York", "timezone": "EST", "tz_offset": -5},
    "bkn": {"city": "Brooklyn", "timezone": "EST", "tz_offset": -5},
    "bos": {"city": "Boston", "timezone": "EST", "tz_offset": -5},
    "tor": {"city": "Toronto", "timezone": "EST", "tz_offset": -5},
}

# Distancias aproximadas entre ciudades (en millas)
# Simplificado: costa a costa ~2500, mismo timezone ~500
TRAVEL_DISTANCES = {
    ("PST", "EST"): 2500,
    ("PST", "CST"): 1500,
    ("PST", "MST"): 800,
    ("MST", "EST"): 1700,
    ("MST", "CST"): 900,
    ("CST", "EST"): 800,
    ("EST", "PST"): 2500,
    ("CST", "PST"): 1500,
    ("MST", "PST"): 800,
    ("EST", "MST"): 1700,
    ("CST", "MST"): 900,
    ("EST", "CST"): 800,
}

# Status de lesiones
INJURY_STATUS_MAP = {
    "out": "OUT",
    "doubtful": "DOUBTFUL",
    "questionable": "QUESTIONABLE",
    "probable": "PROBABLE",
    "day-to-day": "DAY-TO-DAY",
    "gtd": "GAME-TIME DECISION",
}


# ============================================================
# CLASE PRINCIPAL
# ============================================================

class InjuriesFetcher:
    """
    Fetcher de lesiones NBA para G10+ Ultra Pro V2.
    """

    def __init__(self):
        self.injuries_cache: Dict[str, List[Dict]] = {}
        self.schedule_cache: Dict[str, List[Dict]] = {}
        self.last_fetch: Optional[datetime] = None

    # --------------------------------------------------------
    # CLASIFICACIÓN DE JUGADORES
    # --------------------------------------------------------

    def classify_player(self, player_name: str) -> str:
        """
        Clasifica el nivel de impacto de un jugador.
        """
        # Normalizar nombre
        name_lower = player_name.lower().strip()

        # Buscar en listas
        for mvp in MVP_PLAYERS:
            if mvp.lower() in name_lower or name_lower in mvp.lower():
                return "mvp"

        for superstar in SUPERSTAR_PLAYERS:
            if superstar.lower() in name_lower or name_lower in superstar.lower():
                return "superstar"

        for star in STAR_PLAYERS:
            if star.lower() in name_lower or name_lower in star.lower():
                return "star"

        # Default basado en heurísticas
        # Si no está en ninguna lista, es starter o rotation
        return "starter"  # Asumimos starter por defecto para jugadores no listados

    def get_player_level(self, player_name: str, position: str = "", 
                          team_context: Optional[Dict] = None) -> str:
        """
        Obtiene nivel del jugador con contexto adicional.
        """
        base_level = self.classify_player(player_name)

        # Si tenemos contexto del equipo, podemos ajustar
        if team_context:
            # Si el jugador es el líder en puntos/asistencias, subir nivel
            if player_name in team_context.get("top_scorers", [])[:2]:
                if base_level == "star":
                    return "superstar"
                elif base_level == "starter":
                    return "star"

        return base_level

    # --------------------------------------------------------
    # SCRAPING DE LESIONES
    # --------------------------------------------------------

    def fetch_injuries_espn(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scrapea lesiones de ESPN.
        """
        if not SCRAPING_AVAILABLE:
            logger.warning("requests/beautifulsoup no disponible")
            return {}

        logger.info("🏥 Fetching injuries from ESPN...")

        try:
            response = requests.get(ESPN_INJURIES_URL, headers=HEADERS, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            injuries_by_team: Dict[str, List[Dict]] = {}

            # Buscar tablas de lesiones
            # ESPN estructura: divs con clase específica por equipo
            team_sections = soup.find_all("div", class_="ResponsiveTable")

            if not team_sections:
                # Intentar estructura alternativa
                team_sections = soup.find_all("section", class_="Card")

            for section in team_sections:
                try:
                    # Obtener nombre del equipo
                    team_header = section.find(["h2", "h3", "span"], class_=re.compile(r"team|Team|headline"))
                    if not team_header:
                        team_header = section.find("a", href=re.compile(r"/nba/team/"))

                    if not team_header:
                        continue

                    team_name = team_header.get_text(strip=True)

                    # Buscar filas de jugadores
                    rows = section.find_all("tr")
                    team_injuries = []

                    for row in rows[1:]:  # Skip header
                        cells = row.find_all(["td", "th"])
                        if len(cells) >= 2:
                            player_cell = cells[0]
                            status_cell = cells[1] if len(cells) > 1 else None

                            player_name = player_cell.get_text(strip=True)
                            status = status_cell.get_text(strip=True) if status_cell else "Unknown"

                            # Limpiar status
                            status_lower = status.lower()
                            injury_status = "OUT"
                            for key, val in INJURY_STATUS_MAP.items():
                                if key in status_lower:
                                    injury_status = val
                                    break

                            # Obtener descripción de lesión si existe
                            injury_desc = ""
                            if len(cells) > 2:
                                injury_desc = cells[2].get_text(strip=True)

                            if player_name and player_name != "NAME":
                                team_injuries.append({
                                    "player": player_name,
                                    "status": injury_status,
                                    "injury": injury_desc,
                                    "level": self.classify_player(player_name),
                                })

                    if team_injuries:
                        injuries_by_team[team_name] = team_injuries

                except Exception as e:
                    logger.warning(f"Error parsing team section: {e}")
                    continue

            self.injuries_cache = injuries_by_team
            self.last_fetch = datetime.now()

            logger.info(f"✅ Found injuries for {len(injuries_by_team)} teams")
            return injuries_by_team

        except requests.RequestException as e:
            logger.error(f"Error fetching ESPN injuries: {e}")
            return {}

    def get_team_injuries(self, team_name: str) -> List[Dict[str, Any]]:
        """
        Obtiene lesiones de un equipo específico.
        """
        # Refresh cache si es viejo (más de 1 hora)
        if (self.last_fetch is None or 
            datetime.now() - self.last_fetch > timedelta(hours=1)):
            self.fetch_injuries_espn()

        # Buscar equipo en cache
        normalized = self._normalize_team_name(team_name)

        for cached_team, injuries in self.injuries_cache.items():
            if (normalized.lower() in cached_team.lower() or 
                cached_team.lower() in normalized.lower()):
                return injuries

        # Si no encontramos, intentar con abreviación
        abbr = TEAM_ABBR_MAP.get(team_name, "").upper()
        for cached_team, injuries in self.injuries_cache.items():
            if abbr and abbr in cached_team.upper():
                return injuries

        return []

    def _normalize_team_name(self, team_name: str) -> str:
        """Normaliza nombre de equipo."""
        # Mapeo de variaciones comunes a nombres completos
        name_map = {
            "LAL": "Los Angeles Lakers",
            "Lakers": "Los Angeles Lakers",
            "LAC": "Los Angeles Clippers",
            "Clippers": "Los Angeles Clippers",
            "GSW": "Golden State Warriors",
            "Warriors": "Golden State Warriors",
            "BOS": "Boston Celtics",
            "Celtics": "Boston Celtics",
            "MIA": "Miami Heat",
            "Heat": "Miami Heat",
            "DEN": "Denver Nuggets",
            "Nuggets": "Denver Nuggets",
            # ... agregar más según necesites
        }
        return name_map.get(team_name, team_name)

    # --------------------------------------------------------
    # SCHEDULE Y CONTEXTO
    # --------------------------------------------------------

    def get_team_schedule_context(
        self, 
        team_name: str, 
        game_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtiene contexto de schedule para un equipo.
        Incluye B2B, días de descanso, viajes.
        """
        if game_date is None:
            game_date = datetime.now().strftime("%Y-%m-%d")

        game_dt = datetime.strptime(game_date, "%Y-%m-%d")

        # Por ahora retornamos valores por defecto
        # En producción, esto scrapearía el schedule real
        context = {
            "team": team_name,
            "game_date": game_date,
            "is_b2b": False,
            "is_3in4": False,
            "is_4in6": False,
            "is_5in7": False,
            "days_rest": 1,
            "previous_game_date": None,
            "previous_opponent": None,
            "travel_miles": 0,
            "timezone_direction": "none",
        }

        return context

    def calculate_rest_days(
        self, 
        current_game_date: str, 
        previous_game_date: str
    ) -> int:
        """Calcula días de descanso entre juegos."""
        try:
            current = datetime.strptime(current_game_date, "%Y-%m-%d")
            previous = datetime.strptime(previous_game_date, "%Y-%m-%d")
            delta = (current - previous).days - 1
            return max(0, delta)
        except:
            return 1  # Default

    def detect_b2b(
        self, 
        current_game_date: str, 
        previous_game_date: str
    ) -> bool:
        """Detecta si es back-to-back."""
        try:
            current = datetime.strptime(current_game_date, "%Y-%m-%d")
            previous = datetime.strptime(previous_game_date, "%Y-%m-%d")
            return (current - previous).days == 1
        except:
            return False

    def calculate_travel(
        self, 
        from_team: str, 
        to_team: str
    ) -> Tuple[int, str]:
        """
        Calcula distancia y dirección de viaje.
        Returns: (miles, direction)
        """
        from_abbr = TEAM_ABBR_MAP.get(from_team, "").lower()
        to_abbr = TEAM_ABBR_MAP.get(to_team, "").lower()

        if not from_abbr or not to_abbr:
            return 0, "none"

        from_loc = TEAM_LOCATIONS.get(from_abbr, {})
        to_loc = TEAM_LOCATIONS.get(to_abbr, {})

        if not from_loc or not to_loc:
            return 0, "none"

        from_tz = from_loc.get("timezone", "EST")
        to_tz = to_loc.get("timezone", "EST")

        # Calcular distancia
        distance = TRAVEL_DISTANCES.get((from_tz, to_tz), 500)

        # Determinar dirección
        from_offset = from_loc.get("tz_offset", -5)
        to_offset = to_loc.get("tz_offset", -5)

        if from_offset > to_offset:
            direction = "west_to_east"
        elif from_offset < to_offset:
            direction = "east_to_west"
        else:
            direction = "none"

        return distance, direction

    # --------------------------------------------------------
    # FORMATO PARA G10+ V2
    # --------------------------------------------------------

    def format_injuries_for_g10(
        self, 
        team_name: str,
        only_out: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Formatea lesiones para el formato que espera G10+ V2.
        """
        raw_injuries = self.get_team_injuries(team_name)
        formatted = []

        for injury in raw_injuries:
            status = injury.get("status", "OUT")

            # Filtrar solo OUT si se especifica
            if only_out and status not in ["OUT", "DOUBTFUL"]:
                continue

            formatted.append({
                "player": injury.get("player", "Unknown"),
                "level": injury.get("level", "rotation"),
                "status": status,
                "injury": injury.get("injury", ""),
            })

        return formatted

    def format_context_for_g10(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[str] = None,
        home_previous_game: Optional[str] = None,
        away_previous_game: Optional[str] = None,
        home_previous_location: Optional[str] = None,
        away_previous_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Crea el game_context completo para G10+ V2.
        """
        if game_date is None:
            game_date = datetime.now().strftime("%Y-%m-%d")

        # Calcular días de descanso
        home_rest = 1
        away_rest = 1

        if home_previous_game:
            home_rest = self.calculate_rest_days(game_date, home_previous_game)

        if away_previous_game:
            away_rest = self.calculate_rest_days(game_date, away_previous_game)

        # Detectar B2B
        home_b2b = home_rest == 0
        away_b2b = away_rest == 0

        # Calcular viaje del away team
        travel_miles = 0
        timezone_direction = "none"

        if away_previous_location:
            travel_miles, timezone_direction = self.calculate_travel(
                away_previous_location, home_team
            )

        # Obtener lesiones
        home_injuries = self.format_injuries_for_g10(home_team)
        away_injuries = self.format_injuries_for_g10(away_team)

        context = {
            "game_date": game_date,
            "home_b2b": home_b2b,
            "away_b2b": away_b2b,
            "home_3in4": False,  # Requiere más datos de schedule
            "away_3in4": False,
            "home_4in6": False,
            "away_4in6": False,
            "home_5in7": False,
            "away_5in7": False,
            "home_days_rest": home_rest,
            "away_days_rest": away_rest,
            "away_travel_miles": travel_miles,
            "timezone_direction": timezone_direction,
            "home_win_streak": 0,  # Requiere datos adicionales
            "away_win_streak": 0,
            "home_loss_streak": 0,
            "away_loss_streak": 0,
            "rivalry_game": False,
            "home_desperation": False,
            "away_desperation": False,
            "elimination_game": False,
            "national_tv": False,
            "playoff_game": False,
            "home_injuries": home_injuries,
            "away_injuries": away_injuries,
        }

        return context

    # --------------------------------------------------------
    # MÉTODOS DE CONVENIENCIA
    # --------------------------------------------------------

    def get_all_injuries(self) -> Dict[str, List[Dict[str, Any]]]:
        """Obtiene todas las lesiones de todos los equipos."""
        return self.fetch_injuries_espn()

    def get_injury_report_summary(self) -> str:
        """Genera un resumen de lesiones."""
        if not self.injuries_cache:
            self.fetch_injuries_espn()

        summary_lines = [
            "=" * 50,
            "NBA INJURY REPORT SUMMARY",
            "=" * 50,
            ""
        ]

        for team, injuries in sorted(self.injuries_cache.items()):
            if injuries:
                summary_lines.append(f"📋 {team}:")
                for inj in injuries:
                    level_emoji = {
                        "mvp": "⭐",
                        "superstar": "🌟",
                        "star": "✨",
                        "starter": "👤",
                        "rotation": "📌",
                        "bench": "🪑",
                    }.get(inj.get("level", "rotation"), "•")

                    summary_lines.append(
                        f"   {level_emoji} {inj['player']} [{inj['level'].upper()}] - {inj['status']}"
                    )
                summary_lines.append("")

        return "\n".join(summary_lines)


# ============================================================
# FUNCIONES DE CONVENIENCIA
# ============================================================

def get_injuries(team_name: str) -> List[Dict[str, Any]]:
    """Función rápida para obtener lesiones de un equipo."""
    fetcher = InjuriesFetcher()
    return fetcher.format_injuries_for_g10(team_name)


def get_game_context(
    home_team: str,
    away_team: str,
    game_date: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Función rápida para obtener contexto de juego."""
    fetcher = InjuriesFetcher()
    return fetcher.format_context_for_g10(
        home_team=home_team,
        away_team=away_team,
        game_date=game_date,
        **kwargs
    )


def classify_player(player_name: str) -> str:
    """Función rápida para clasificar un jugador."""
    fetcher = InjuriesFetcher()
    return fetcher.classify_player(player_name)


# ============================================================
# DATOS MANUALES (FALLBACK)
# ============================================================

def get_manual_injuries(team_name: str, injuries_list: List[Dict]) -> List[Dict[str, Any]]:
    """
    Permite ingresar lesiones manualmente.
    
    Uso:
        injuries = get_manual_injuries("Lakers", [
            {"player": "Anthony Davis", "level": "superstar"},
            {"player": "Austin Reaves", "level": "starter"},
        ])
    """
    fetcher = InjuriesFetcher()
    formatted = []

    for injury in injuries_list:
        player = injury.get("player", "Unknown")
        level = injury.get("level", None)

        if level is None:
            level = fetcher.classify_player(player)

        formatted.append({
            "player": player,
            "level": level,
            "status": injury.get("status", "OUT"),
            "injury": injury.get("injury", ""),
        })

    return formatted


def create_context_manual(
    home_team: str,
    away_team: str,
    game_date: str,
    home_b2b: bool = False,
    away_b2b: bool = False,
    home_rest: int = 1,
    away_rest: int = 1,
    away_travel_miles: int = 0,
    timezone_direction: str = "none",
    home_injuries: Optional[List[Dict]] = None,
    away_injuries: Optional[List[Dict]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Crea contexto de juego manualmente.
    
    Uso:
        context = create_context_manual(
            home_team="Nuggets",
            away_team="Lakers",
            game_date="2024-12-05",
            away_b2b=True,
            away_travel_miles=850,
            timezone_direction="west_to_east",
            away_injuries=[{"player": "Anthony Davis", "level": "superstar"}],
        )
    """
    fetcher = InjuriesFetcher()

    # Formatear lesiones si se proveen
    formatted_home_injuries = []
    formatted_away_injuries = []

    if home_injuries:
        formatted_home_injuries = get_manual_injuries(home_team, home_injuries)

    if away_injuries:
        formatted_away_injuries = get_manual_injuries(away_team, away_injuries)

    context = {
        "game_date": game_date,
        "home_b2b": home_b2b,
        "away_b2b": away_b2b,
        "home_3in4": kwargs.get("home_3in4", False),
        "away_3in4": kwargs.get("away_3in4", False),
        "home_4in6": kwargs.get("home_4in6", False),
        "away_4in6": kwargs.get("away_4in6", False),
        "home_5in7": kwargs.get("home_5in7", False),
        "away_5in7": kwargs.get("away_5in7", False),
        "home_days_rest": home_rest,
        "away_days_rest": away_rest,
        "away_travel_miles": away_travel_miles,
        "timezone_direction": timezone_direction,
        "home_win_streak": kwargs.get("home_win_streak", 0),
        "away_win_streak": kwargs.get("away_win_streak", 0),
        "home_loss_streak": kwargs.get("home_loss_streak", 0),
        "away_loss_streak": kwargs.get("away_loss_streak", 0),
        "rivalry_game": kwargs.get("rivalry_game", False),
        "home_desperation": kwargs.get("home_desperation", False),
        "away_desperation": kwargs.get("away_desperation", False),
        "elimination_game": kwargs.get("elimination_game", False),
        "national_tv": kwargs.get("national_tv", False),
        "playoff_game": kwargs.get("playoff_game", False),
        "home_injuries": formatted_home_injuries,
        "away_injuries": formatted_away_injuries,
    }

    return context


# ============================================================
# MAIN - TEST
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NBA INJURIES FETCHER - TEST")
    print("=" * 60 + "\n")

    fetcher = InjuriesFetcher()

    # Test 1: Clasificación de jugadores
    print("🎯 TEST 1: Player Classification")
    print("-" * 40)

    test_players = [
        "LeBron James",
        "Stephen Curry",
        "Anthony Davis",
        "Austin Reaves",
        "Random Player",
    ]

    for player in test_players:
        level = fetcher.classify_player(player)
        print(f"  {player}: {level.upper()}")

    # Test 2: Fetch injuries (si hay conexión)
    print("\n" + "-" * 40)
    print("🏥 TEST 2: Fetching ESPN Injuries")
    print("-" * 40)

    if SCRAPING_AVAILABLE:
        injuries = fetcher.fetch_injuries_espn()
        if injuries:
            print(f"  Found injuries for {len(injuries)} teams")
            # Mostrar primeros 2 equipos
            for i, (team, inj_list) in enumerate(list(injuries.items())[:2]):
                print(f"\n  📋 {team}:")
                for inj in inj_list[:3]:
                    print(f"     - {inj['player']} [{inj['level']}]: {inj['status']}")
        else:
            print("  No injuries found (might be offseason or connection issue)")
    else:
        print("  ⚠️ Scraping not available - install requests beautifulsoup4")

    # Test 3: Manual context creation
    print("\n" + "-" * 40)
    print("📝 TEST 3: Manual Context Creation")
    print("-" * 40)

    context = create_context_manual(
        home_team="Denver Nuggets",
        away_team="Los Angeles Lakers",
        game_date="2024-12-05",
        away_b2b=True,
        away_rest=0,
        home_rest=2,
        away_travel_miles=850,
        timezone_direction="west_to_east",
        national_tv=True,
        away_injuries=[
            {"player": "Anthony Davis", "level": "superstar"},
            {"player": "D'Angelo Russell", "level": "starter"},
        ],
    )

    print(f"  Game: Lakers @ Nuggets")
    print(f"  Away B2B: {context['away_b2b']}")
    print(f"  Away Rest: {context['away_days_rest']} days")
    print(f"  Travel: {context['away_travel_miles']} miles")
    print(f"  Direction: {context['timezone_direction']}")
    print(f"  Away Injuries: {len(context['away_injuries'])}")
    for inj in context['away_injuries']:
        print(f"    - {inj['player']} [{inj['level']}]")

    # Test 4: Travel calculation
    print("\n" + "-" * 40)
    print("✈️ TEST 4: Travel Calculation")
    print("-" * 40)

    test_routes = [
        ("Lakers", "Celtics"),
        ("Nuggets", "Lakers"),
        ("Heat", "Knicks"),
    ]

    for from_team, to_team in test_routes:
        miles, direction = fetcher.calculate_travel(from_team, to_team)
        print(f"  {from_team} → {to_team}: {miles} miles ({direction})")

    print("\n" + "=" * 60)
    print("✅ Tests completed!")
    print("=" * 60 + "\n")
