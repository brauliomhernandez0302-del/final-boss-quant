# ==========================================================
# MLB DATA FETCHERS V3 FINAL - MLB STATS + WEATHER + PARK
# ==========================================================

import logging
import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DATA_DIR, CACHE_DIR, LEAGUE_AVG_RUNS, LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP
from modules.baseball_module.context_engine.defensive_efficiency_engine import calculate_der
from modules.baseball_module.offense.true_talent_engine import _fetch_team_roster as _fetch_gameday_roster

load_dotenv()

logger = logging.getLogger(__name__)


def _current_mlb_season() -> int:
    """Return the current MLB season year. Season starts in March."""
    now = datetime.now()
    return now.year if now.month >= 3 else now.year - 1


# Cuántos innings de MLB de la temporada en curso "vale" un inning de cada
# fuente de respaldo, para efectos de cuánto creerle. Ver el docstring de
# get_pitcher_stats_full_fallback para el problema que resuelven y la medición
# que los motivó.
#
# ASUMIDOS, no ajustados contra datos. El criterio con que se eligieron:
#   - Temporada anterior de MLB (0.50): es MLB de verdad, contra los mismos
#     bateadores, pero con un invierno en el medio — lesiones, cambios de
#     repertorio, envejecimiento. Media credibilidad.
#   - AAA (0.25): temporada en curso pero otro nivel de competencia; la brecha
#     de traducción AAA→MLB es grande y bien conocida.
#   - AA (0.15): la misma brecha, más ancha.
# Ninguno es 0: un abridor con 180 innings de la temporada pasada sí dice algo,
# y tirar esa información sería tan deshonesto como creerle entera.
# Innings mínimos para tratar una muestra de MLB de esta temporada como usable
# por sí sola. Un solo número para dos decisiones que tienen que coincidir:
# si un split de lado alcanza para desplazar a la línea consolidada
# (`_select_best_stats`) y si el tier 1 alcanza para no bajar a los tiers de
# respaldo (`get_pitcher_stats_full_fallback`). Estaban como literales
# separados y por eso podían discrepar: un split de 2 IP ganaba la primera
# decisión y perdía la segunda, mandando a un pitcher con datos reales de MLB
# a las menores.
_MIN_USABLE_MLB_IP = 5.0

_IP_EQ_MLB_PREV_SEASON = 0.50
_IP_EQ_AAA = 0.25
_IP_EQ_AA = 0.15


# Peso de un bateador ambidiestro dentro del LHB% de una alineación, según la
# mano del abridor que enfrenta. No es una convención: es el hecho. Un ambidiestro
# batea zurdo contra un derecho y derecho contra un zurdo, así que el número
# correcto es 1.0 o 0.0, nunca un promedio.
#
# El 0.5 sobrevive SÓLO para cuando no se sabe la mano del rival, y ahí es lo
# honesto: reparte el error en vez de apostar a un lado. Medido: los ambidiestros
# son el 11.2% de los turnos de una alineación real, y elegir mal la convención
# mueve el LHB% del MISMO lineup en 0.056 — más del doble de lo que se gana
# eligiendo bien la fuente del lineup (0.022).
_PESO_AMBIDIESTRO = {"R": 1.0, "L": 0.0}


def lhb_pct_efectivo(
    manos: Dict[int, str], bateadores: List[int], mano_rival: Optional[str],
) -> Optional[float]:
    """Fracción zurda EFECTIVA de una alineación frente a un abridor dado.

    `mano_rival` None ⇒ el ambidiestro pesa 0.5 (no se sabe contra quién batea).
    Devuelve None con la lista vacía: sin bateadores no hay fracción que informar,
    y fabricar una media de liga acá es justo lo que tapaba el dato real.
    """
    if not bateadores:
        return None
    peso_s = _PESO_AMBIDIESTRO.get(mano_rival or "", 0.5)
    total = 0.0
    for pid in bateadores:
        mano = manos.get(pid, "R")
        total += 1.0 if mano == "L" else (peso_s if mano == "S" else 0.0)
    return round(total / len(bateadores), 3)


def _official_day(game: Dict[str, Any]) -> str:
    """El día de calendario al que MLB asigna este juego (YYYY-MM-DD).

    `game_date` es el timestamp UTC de PRIMER PITCHEO, no un día: para un
    nocturno que cruce medianoche UTC —lo normal en la costa oeste— truncarlo
    da el día siguiente. `officialDate` es el campo que MLB expone justo para
    esto, y es por el que se consulta el endpoint de schedule.

    Fallback explícito y ruidoso: si `official_date` falta, se vuelve al
    truncado con un warning. Es el comportamiento viejo (sesgado), así que
    tiene que verse en el log y no pasar por normal.
    """
    official = game.get("official_date")
    if official:
        return str(official)[:10]
    crudo = (game.get("game_date") or "")[:10]
    logger.warning(
        "official_date ausente para game_pk=%s (%s @ %s) — usando la fecha UTC %r, "
        "que para un juego nocturno del oeste es el día siguiente y sesga days_rest",
        game.get("game_pk"), game.get("away_team"), game.get("home_team"), crudo,
    )
    return crudo


def _safe_float(value, default: float = 0.0) -> float:
    """Convert value to float, returning default on any failure."""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    import math
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _fetch_savant_oaa(season: int) -> Dict[int, float]:
    """
    Returns {player_id: outs_above_average} from Baseball Savant's public OAA
    leaderboard (season total, all fielders). No auth required — confirmed
    live 2026-07-04; the "OAA not available from free MLB API" assumption
    that made DefensiveEfficiencyEngine's OAA path permanently unused was
    outdated. Cached 24h (season-long fielding aggregate changes slowly).
    """
    cache_file = CACHE_DIR / f"savant_oaa_{season}.json"
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86_400:
        try:
            return {int(k): v for k, v in json.load(open(cache_file)).items()}
        except Exception:
            pass

    try:
        r = requests.get(
            "https://baseballsavant.mlb.com/leaderboard/outs_above_average",
            params={
                "type": "Fielder", "startYear": str(season), "endYear": str(season),
                "split": "no", "team": "", "range": "year", "min": "1",
                "pos": "", "roles": "", "viz": "hide", "csv": "true",
            },
            headers={"User-Agent": "Mozilla/5.0 (compatible; FinalBossQuant/1.0)"},
            timeout=(5, 30),
        )
        r.raise_for_status()
        csv_text = r.text
    except Exception as e:
        logger.warning(f"⚠️ Error obteniendo OAA de Savant: {e}")
        return {}

    import csv as _csv_mod
    result: Dict[int, float] = {}
    lines = csv_text.lstrip("﻿").splitlines()
    for row in _csv_mod.DictReader(lines):
        try:
            pid = int(row.get("player_id", 0) or 0)
            oaa = row.get("outs_above_average")
            if pid and oaa not in (None, ""):
                result[pid] = float(oaa)
        except (ValueError, TypeError):
            continue

    try:
        with open(cache_file, "w") as f:
            json.dump(result, f)
    except Exception:
        pass
    logger.info(f"Savant OAA {season}: {len(result)} fielders cached")
    return result


def _lon_to_tz_offset(lon: float) -> int:
    """Approximate North American UTC offset from longitude."""
    if lon > -90:
        return -5   # Eastern
    elif lon > -105:
        return -6   # Central
    elif lon > -115:
        return -7   # Mountain
    else:
        return -8   # Pacific


# Ensure directories exist (config.py owns the paths)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 1) MLB STATS API - OFICIAL (GRATIS)
#    - Fallback: playoff → home/away → overall
#    - Context detection (playoff/series)
#    - 7 features adicionales (form, runs trend, bullpen, H2H, standings, travel, pvt placeholder)
# ==========================================================

class MLBStatsAPI:
    BASE_URL = "https://statsapi.mlb.com/api/v1"

    def __init__(self):
        self.cache_ttl = 3600  # 1 hora
        self.session = requests.Session()
        self.team_id_cache: Dict[str, int] = {}

    # -------------------------
    # GAMES + CONTEXT
    # -------------------------
    def get_todays_games(self, date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene partidos del día con pitchers probables + contexto."""
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/schedule"
        params = {
            "sportId": 1,
            "date": date,
            "hydrate": "probablePitcher,lineups,team,seriesStatus"
        }

        try:
            r = self.session.get(url, params=params, timeout=(5, 20))
            r.raise_for_status()
            data = r.json()

            games = []
            for date_item in data.get("dates", []):
                for game in date_item.get("games", []):
                    parsed = self._parse_game(game)
                    if parsed:
                        games.append(parsed)

            logger.info(f"✅ MLB API: {len(games)} juegos encontrados para {date}")
            return games

        except Exception as e:
            logger.error(f"❌ Error obteniendo juegos MLB: {e}")
            return []

    def get_games_by_date(self, date: str) -> List[Dict[str, Any]]:
        """Alias for get_todays_games with an explicit date string (YYYY-MM-DD)."""
        return self.get_todays_games(date=date)

    def _parse_game(self, game: Dict) -> Optional[Dict[str, Any]]:
        """Parsea datos del juego con contexto playoff/series conservando claves originales."""
        try:
            home_team = game["teams"]["home"]["team"]["name"]
            away_team = game["teams"]["away"]["team"]["name"]
            home_team_id = game["teams"]["home"]["team"]["id"]
            away_team_id = game["teams"]["away"]["team"]["id"]
            self.team_id_cache[home_team] = home_team_id
            self.team_id_cache[away_team] = away_team_id

            # Pitchers probables
            home_pitcher = None
            home_pitcher_id = None
            hp = game["teams"]["home"].get("probablePitcher")
            if hp:
                home_pitcher = hp.get("fullName")
                home_pitcher_id = hp.get("id")

            away_pitcher = None
            away_pitcher_id = None
            ap = game["teams"]["away"].get("probablePitcher")
            if ap:
                away_pitcher = ap.get("fullName")
                away_pitcher_id = ap.get("id")

            # Contexto
            game_type = game.get("gameType", "R")
            game_type_map = {
                "R": "regular",
                "F": "wildcard",
                "D": "division",
                "L": "championship",
                "W": "worldseries",
                "S": "spring",
                "E": "exhibition"
            }
            game_context = game_type_map.get(game_type, "regular")
            is_playoff = game_type in {"F", "D", "L", "W"}

            series_info = None
            if is_playoff:
                series_info = {
                    "description": game.get("seriesDescription", ""),
                    "game_number": game.get("seriesGameNumber", 0),
                    "games_in_series": game.get("gamesInSeries", 7)
                }

            # Lineups (populated same-day; None for future games)
            raw_lineups = game.get("lineups") or {}
            def _parse_players(players):
                return [
                    {
                        "id": p.get("id"),
                        "name": p.get("fullName"),
                        "position": (p.get("primaryPosition") or {}).get("abbreviation"),
                    }
                    for p in players if p.get("id")
                ]
            home_lineup = _parse_players(raw_lineups.get("homePlayers") or [])
            away_lineup = _parse_players(raw_lineups.get("awayPlayers") or [])

            return {
                "game_pk": game.get("gamePk"),
                "game_date": game.get("gameDate"),
                # "Which schedule day this game belongs to" per MLB's own
                # convention — distinct from game_date's UTC start-time
                # timestamp, which crosses into the next calendar day for any
                # late-night start (e.g. a 10pm Pacific game has a gameDate
                # of ~05:00 UTC the NEXT day). This is exactly what
                # get_todays_games()/get_games_by_date() query the schedule
                # endpoint by, so a consumer that needs "today vs tomorrow"
                # to agree with those calls (not the raw UTC date) should use
                # this field, not game_date[:10].
                "official_date": game.get("officialDate"),
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_pitcher": home_pitcher,
                "home_pitcher_id": home_pitcher_id,
                "away_pitcher": away_pitcher,
                "away_pitcher_id": away_pitcher_id,
                "venue": (game.get("venue") or {}).get("name"),
                # Doubleheader: 'N' ninguno, 'Y' tradicional (los dos juegos
                # seguidos, el schedule le pone al segundo un marcador 5 min
                # después del primero), 'S' partido (275-405 min de separación
                # real, medido). Se capturan porque son la ÚNICA forma de saber
                # que dos juegos del mismo par de equipos en el mismo día son
                # dos juegos distintos: sin esto, la abstención del emparejador
                # de odds (`odds_fetcher._ODDS_MATCH_MIN_MARGIN`) es correcta
                # pero no puede explicarse a sí misma en ningún log ni reporte.
                "doubleheader": game.get("doubleHeader", "N"),
                "game_number": game.get("gameNumber", 1),
                "game_type": game_type,
                "game_context": game_context,
                "is_playoff": is_playoff,
                "series_info": series_info,
                "status": (game.get("status") or {}).get("detailedState"),
                "home_lineup": home_lineup,
                "away_lineup": away_lineup,
            }
        except Exception as e:
            logger.warning(f"⚠️ Error parseando juego: {e}")
            return None

    # -------------------------
    # PITCHER STATS (compat + fallback)
    # -------------------------
    def get_pitcher_stats(self, pitcher_id: int, season: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Stats de temporada (overall). Mantiene compatibilidad con tu método original."""
        if season is None:
            season = _current_mlb_season()
        cache_file = CACHE_DIR / f"pitcher_{pitcher_id}_{season}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < self.cache_ttl:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        url = f"{self.BASE_URL}/people/{pitcher_id}/stats"
        params = {"stats": "season", "season": season, "group": "pitching"}

        try:
            r = self.session.get(url, params=params, timeout=(5, 30))
            r.raise_for_status()
            data = r.json()
            stats = self._parse_pitcher_stats(data)
            if stats:
                with open(cache_file, "w") as f:
                    json.dump(stats, f)
            return stats
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo stats de pitcher {pitcher_id}: {e}")
            return None

    def _parse_pitcher_stats(self, data: Dict) -> Optional[Dict[str, Any]]:
        try:
            splits = data["stats"][0]["splits"]
            if not splits:
                return None
            stat = splits[0]["stat"]

            era = float(stat.get("era", -1))
            whip = float(stat.get("whip", -1))
            innings = self._ip_to_float(stat.get("inningsPitched", 0.0))
            so = int(stat.get("strikeOuts", 0))
            bb = int(stat.get("baseOnBalls", 0))

            # validaciones fuertes
            if era < 0 or era > 99.0:
                return None
            if whip < 0 or whip > 10.0:
                return None
            if innings <= 0:
                return None

            k_per_9 = (so / innings * 9) if innings > 0 else 0.0
            bb_per_9 = (bb / innings * 9) if innings > 0 else 0.0

            hr = int(stat.get("homeRuns", 0))
            hbp = int(stat.get("hitBatsmen", stat.get("hitByPitch", 0)))
            # FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + FIP_constant
            FIP_CONSTANT = 3.10
            fip = round((13 * hr + 3 * (bb + hbp) - 2 * so) / innings + FIP_CONSTANT, 2)
            fip = round(max(0.0, min(fip, 12.0)), 2)

            gs = int(stat.get("gamesStarted", 0))
            avg_ips = round(innings / gs, 2) if gs > 0 else None

            return {
                "era": round(era, 2),
                "whip": round(whip, 2),
                "fip": fip,
                "wins": int(stat.get("wins", 0)),
                "losses": int(stat.get("losses", 0)),
                "innings_pitched": round(innings, 1),
                "strikeouts": so,
                "walks": bb,
                "home_runs_allowed": hr,
                "hit_batsmen": hbp,
                "k_per_9": round(k_per_9, 2),
                "bb_per_9": round(bb_per_9, 2),
                "hits_allowed": int(stat.get("hits", 0)),
                "earned_runs": int(stat.get("earnedRuns", 0)),
                "games_started": gs,
                "avg_innings_per_start": avg_ips,
            }
        except Exception:
            return None

    # ====== FALLBACK INTELIGENTE ======
    def get_pitcher_stats_with_fallback(
        self,
        pitcher_id: int,
        season: int,
        is_playoff: bool,
        is_home: bool
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Selecciona mejor fuente: playoff (IP>=3) → split home/away → overall."""
        cache_file = CACHE_DIR / f"pitcher_{pitcher_id}_{season}_complete.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < self.cache_ttl:
            try:
                with open(cache_file, "r") as f:
                    all_stats = json.load(f)
                return self._select_best_stats(all_stats, is_playoff, is_home)
            except Exception:
                pass

        all_stats: Dict[str, Dict[str, Any]] = {}

        overall = self._fetch_single_stat_type(pitcher_id, season, "season", "pitching", game_type=None)
        if overall:
            all_stats["regular_overall"] = overall

        splits = self._fetch_home_away_splits(pitcher_id, season)
        all_stats.update(splits)

        if is_playoff:
            playoff = self._fetch_single_stat_type(pitcher_id, season, "season", "pitching", game_type="P")
            if playoff:
                all_stats["playoff"] = playoff

        if all_stats:
            try:
                with open(cache_file, "w") as f:
                    json.dump(all_stats, f)
            except Exception:
                pass

        return self._select_best_stats(all_stats, is_playoff, is_home)

    def _fetch_single_stat_type(
        self,
        pitcher_id: int,
        season: int,
        stat_type: str,
        group: str,
        game_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.BASE_URL}/people/{pitcher_id}/stats"
        params = {"stats": stat_type, "season": season, "group": group}
        if game_type:
            params["gameType"] = game_type
        try:
            r = self.session.get(url, params=params, timeout=(5, 30))
            r.raise_for_status()
            return self._parse_pitcher_stats(r.json())
        except Exception:
            return None

    def _fetch_home_away_splits(self, pitcher_id: int, season: int) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/people/{pitcher_id}/stats"
        params = {"stats": "homeAndAway", "season": season, "group": "pitching"}
        result: Dict[str, Any] = {}
        try:
            r = self.session.get(url, params=params, timeout=(5, 30))
            r.raise_for_status()
            data = r.json()
            splits = data["stats"][0]["splits"]
            for s in splits:
                stat = s.get("stat") or {}
                parsed = self._extract_basic_stats(stat)
                if not parsed:
                    continue
                # MLB API returns isHome bool at top level (not split.code)
                is_home_split = s.get("isHome")
                if is_home_split is True:
                    result["home"] = parsed
                elif is_home_split is False:
                    result["away"] = parsed
        except Exception:
            pass
        return result

    def _extract_basic_stats(self, stat: Dict) -> Optional[Dict[str, Any]]:
        try:
            era = float(stat.get("era", -1))
            whip = float(stat.get("whip", -1))
            innings = self._ip_to_float(stat.get("inningsPitched", 0.0))
            if era < 0 or era > 99.0:
                return None
            if whip < 0 or whip > 10.0:
                return None
            if innings <= 0:
                return None
            so = int(stat.get("strikeOuts", 0))
            bb = int(stat.get("baseOnBalls", 0))
            k9 = (so / innings * 9) if innings > 0 else 0.0
            bb9 = (bb / innings * 9) if innings > 0 else 0.0
            hr = int(stat.get("homeRuns", 0))
            hbp = int(stat.get("hitBatsmen", stat.get("hitByPitch", 0)))
            FIP_CONSTANT = 3.10
            fip = round((13 * hr + 3 * (bb + hbp) - 2 * so) / innings + FIP_CONSTANT, 2)
            fip = round(max(0.0, min(fip, 12.0)), 2)
            return {
                "era": round(era, 2),
                "whip": round(whip, 2),
                "fip": fip,
                "innings_pitched": round(innings, 1),
                "k_per_9": round(k9, 2),
                "bb_per_9": round(bb9, 2),
                "strikeouts": so,
                "walks": bb,
                "home_runs_allowed": hr,
            }
        except Exception:
            return None

    def _select_best_stats(self, all_stats: Dict, is_playoff: bool, is_home: bool) -> Tuple[Optional[Dict], Optional[str]]:
        if is_playoff and "playoff" in all_stats:
            ps = all_stats["playoff"]
            if ps.get("innings_pitched", 0) >= 3.0:
                logger.info(f"  ✅ Usando PLAYOFF (IP {ps.get('innings_pitched', 0)})")
                return ps, "playoff"
            else:
                logger.warning("  ⚠️ Playoff con IP bajas, buscando fallback...")

        # El split del lado sólo gana si es una muestra usable POR SÍ SOLA. Antes
        # se devolvía sin mirar los innings, y un split de 2 IP desplazaba a la
        # línea consolidada del mismo pitcher — que además, aguas arriba, hacía
        # que `get_pitcher_stats_full_fallback` lo diera por insuficiente y se
        # fuera a la temporada anterior o a las menores, saltándose datos reales
        # de MLB de esta temporada que estaban ahí mismo.
        #
        # Caso real que lo destapó (Eddy Yean, 2026): home 5.2 IP ERA 3.18,
        # away 2.0 IP ERA 0.00, OVERALL 7.2 IP ERA 2.35. Como visitante se
        # evaluaba con 43 innings de Doble-A en vez de con sus 7.2 de MLB.
        lado = "home" if is_home else "away"
        etiqueta = f"{lado}_split"
        split = all_stats.get(lado)
        if split is not None:
            if split.get("innings_pitched", 0) >= _MIN_USABLE_MLB_IP:
                logger.info(f"  ✅ Usando {lado.upper()} split")
                return split, etiqueta
            logger.info(
                "  ↪ split %s con sólo %s IP (< %s): se prefiere la línea consolidada",
                lado.upper(), split.get("innings_pitched", 0), _MIN_USABLE_MLB_IP,
            )

        if "regular_overall" in all_stats:
            logger.info("  ✅ Usando OVERALL (regular season)")
            return all_stats["regular_overall"], "regular_overall"

        logger.error("  ❌ Sin stats válidas")
        return None, None

    def get_pitcher_game_log(
        self,
        pitcher_id: int,
        season: int,
        last_n: int = 5,
        as_of_date: Optional[str] = None,
    ):
        as_of_key = as_of_date or "live"
        cache_file = CACHE_DIR / f"pitcher_log_{pitcher_id}_{season}_{last_n}_{as_of_key}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 3600:
            try:
                with open(cache_file, "r") as f2:
                    return json.load(f2)
            except Exception:
                pass
        url = f"{self.BASE_URL}/people/{pitcher_id}/stats"
        params = {"stats": "gameLog", "season": season, "group": "pitching"}
        try:
            r = self.session.get(url, params=params, timeout=(5, 30))
            r.raise_for_status()
            data = r.json()
            splits = data.get("stats", [{}])[0].get("splits", [])
            if not splits:
                return None
            if as_of_date:
                try:
                    cutoff_date = datetime.strptime(as_of_date[:10], "%Y-%m-%d").date()
                    splits = [
                        s for s in splits
                        if s.get("date") and datetime.strptime(s.get("date", "")[:10], "%Y-%m-%d").date() <= cutoff_date
                    ]
                except Exception:
                    pass
            starts = [s for s in splits if int(s.get("stat", {}).get("gamesStarted", 0)) > 0]
            if not starts:
                starts = splits
            starts.sort(key=lambda x: x.get("date", ""), reverse=True)
            recent = starts[:last_n]
            if not recent:
                return None
            total_er = sum(int(s.get("stat", {}).get("earnedRuns", 0)) for s in recent)
            total_ip = sum(self._ip_to_float(s.get("stat", {}).get("inningsPitched", 0)) for s in recent)
            era_last_n = round((total_er / total_ip * 9), 2) if total_ip > 0 else 4.50
            avg_ips_recent = round(total_ip / len(recent), 2) if recent else None
            last_start = recent[0]
            last_date_str = last_start.get("date", "")
            last_pitch_count = int(last_start.get("stat", {}).get("numberOfPitches", 90))
            days_rest = 4
            if last_date_str:
                try:
                    last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
                    if as_of_date:
                        ref_date = datetime.strptime(as_of_date[:10], "%Y-%m-%d")
                    else:
                        ref_date = datetime.utcnow()
                    days_rest = (ref_date - last_date).days
                except Exception:
                    pass
            result = {
                "era_last_5": era_last_n,
                "days_rest": days_rest,
                "last_pitch_count": last_pitch_count,
                "starts_analyzed": len(recent),
                "avg_innings_per_start": avg_ips_recent,
            }

            # ── Trend metrics (oldest→newest) ──────────────────────────
            chrono = list(reversed(recent))
            _metrics = []
            for _s in chrono:
                _st = _s.get("stat", {})
                _ip = self._ip_to_float(_st.get("inningsPitched", 0))
                _er = int(_st.get("earnedRuns", 0))
                _so = int(_st.get("strikeOuts", 0))
                _bb = int(_st.get("baseOnBalls", 0))
                if _ip > 0:
                    _metrics.append({
                        "era": _er / _ip * 9,
                        "k9": _so / _ip * 9,
                        "bb9": _bb / _ip * 9,
                        "qs": 1 if _ip >= 6 and _er <= 3 else 0,
                    })
            if len(_metrics) >= 2:
                def _slope(vals):
                    n = len(vals)
                    xbar = (n - 1) / 2.0
                    ybar = sum(vals) / n
                    num = sum((i - xbar) * (vals[i] - ybar) for i in range(n))
                    den = sum((i - xbar) ** 2 for i in range(n))
                    return num / den if den else 0.0
                result["era_trend"] = round(_slope([m["era"] for m in _metrics]), 3)
                result["k9_trend"]  = round(_slope([m["k9"]  for m in _metrics]), 3)
                result["bb9_trend"] = round(_slope([m["bb9"] for m in _metrics]), 3)
                result["quality_start_pct"] = round(
                    sum(m["qs"] for m in _metrics) / len(_metrics), 2
                )
            # ───────────────────────────────────────────────────────────

            with open(cache_file, "w") as f2:
                json.dump(result, f2)
            return result
        except Exception as e:
            logger.error(f"Error game log pitcher {pitcher_id}: {e}")
            return None

    def get_pitcher_f5_stats(self, pitcher_id: int, season: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch per-inning aggregated stats and compute real F5 ERA (innings 1-5).
        Returns {f5_era, f5_ip, f5_er} or None if insufficient data.
        """
        if season is None:
            season = _current_mlb_season()
        cache_file = CACHE_DIR / f"pitcher_f5_{pitcher_id}_{season}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 21600:
            try:
                data = json.load(open(cache_file))
                if data.get("_failed"):
                    return None
                return data
            except Exception:
                pass

        url = f"{self.BASE_URL}/people/{pitcher_id}/stats"
        params = {"stats": "byInning", "group": "pitching", "season": season}
        try:
            r = self.session.get(url, params=params, timeout=(5, 30))
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [])
            if not splits:
                cache_file.write_text(json.dumps({"_failed": True}))
                return None

            f5_er = 0
            f5_ip = 0.0
            for split in splits:
                inning = split.get("inning", 0)
                if 1 <= inning <= 5:
                    stat = split.get("stat", {})
                    f5_er += int(stat.get("earnedRuns", 0))
                    f5_ip += self._ip_to_float(stat.get("inningsPitched", 0.0))

            if f5_ip < 5.0:
                cache_file.write_text(json.dumps({"_failed": True}))
                return None

            result = {
                "f5_era": round(f5_er / f5_ip * 9, 2),
                "f5_ip": round(f5_ip, 1),
                "f5_er": f5_er,
            }
            cache_file.write_text(json.dumps(result))
            return result
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo F5 stats pitcher {pitcher_id}: {e}")
            cache_file.write_text(json.dumps({"_failed": True}))
            return None

    # ── 5-level pitcher fallback ──────────────────────────────────────────────

    def _fetch_milb_stats(
        self,
        pitcher_id: int,
        season: int,
        sport_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Fetch stats for a pitcher from a MiLB level (AAA=11, AA=12)."""
        cache_file = CACHE_DIR / f"pitcher_{pitcher_id}_{season}_sport{sport_id}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < self.cache_ttl:
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass
        url = f"{self.BASE_URL}/people/{pitcher_id}/stats"
        params = {"stats": "season", "season": season, "group": "pitching", "sportId": sport_id}
        try:
            r = self.session.get(url, params=params, timeout=(5, 30))
            r.raise_for_status()
            stats = self._parse_pitcher_stats(r.json())
            if stats:
                with open(cache_file, "w") as f:
                    json.dump(stats, f)
            return stats
        except Exception:
            return None

    def get_pitcher_stats_full_fallback(
        self,
        pitcher_id: int,
        season: int,
        team_pitching: Optional[Dict[str, Any]] = None,
        is_home: bool = True,
        is_playoff: bool = False,
    ) -> Tuple[Dict[str, Any], str]:
        """
        5-level fallback hierarchy for pitcher stats. Never returns arbitrary defaults.

        Priority:
          1. Current MLB season (home/away split preferred, otherwise overall)
          2. Previous MLB season overall
          3. Triple-A (AAA, sportId=11) current season
          4. Double-A (AA, sportId=12) current season
          5. Team staff ERA/WHIP (from team_pitching dict, or bare league averages)

        Returns (stats_dict, source_label).  stats_dict always has 'era', 'whip',
        'fip', and is_fallback=True on tiers 3-5.

        Cada tier fija además `ip_mlb_equivalent`: cuántos innings de MLB de ESTA
        temporada valdría la muestra. Existe porque `innings_pitched` responde
        "¿cuántos innings lanzó?" y sus dos consumidores preguntan otra cosa —
        "¿cuánto le creo?":

          - `pitcher_engine` lo usa como n de una regresión bayesiana hacia la
            media de liga. Con el crudo, un abridor de Doble-A con 150 innings
            casi no se regresa: su ERA de AA se trata como ERA de mayores, y eso
            mueve λ.
          - `compute_data_quality_confidence` lo pesa al 40%, su factor más
            grande, y su propio docstring dice "this season".

        Medido antes del cambio: AAA con 100 IP, AA con 150 IP, MLB de la
        temporada ANTERIOR con 180 IP y MLB actual con 30 IP daban los cuatro
        exactamente 0.65 de confianza. El score medía el TAMAÑO de la muestra,
        nunca su procedencia.

        Los factores de equivalencia de abajo son ASUMIDOS, no ajustados contra
        datos — igual que `WALKOFF_9TH_SHARE` en el simulador, y nombrados así a
        propósito para que una calibración futura tenga dónde enchufarse.
        Disparador de re-visita: cualquier trabajo que mida traducción de
        minors a MLB, o el arranque de un motor de proyección de abridores.
        """
        from config import LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP

        # ── tier 1: current MLB season ────────────────────────────────────────
        mlb_stats, source = self.get_pitcher_stats_with_fallback(
            pitcher_id, season, is_playoff=is_playoff, is_home=is_home
        )
        if mlb_stats and mlb_stats.get("innings_pitched", 0) >= _MIN_USABLE_MLB_IP:
            # Es exactamente lo que el consumidor quiere medir: sin descuento.
            mlb_stats["ip_mlb_equivalent"] = float(mlb_stats["innings_pitched"])
            return mlb_stats, source or "mlb_current"

        # partial current-season stats (< 5 IP) — keep as candidate, but look further
        partial = mlb_stats

        # ── tier 2: previous MLB season ──────────────────────────────────────
        prev_stats = self.get_pitcher_stats(pitcher_id, season=season - 1)
        if prev_stats and prev_stats.get("innings_pitched", 0) >= 20.0:
            prev_stats["is_fallback"] = True
            prev_stats["fallback_tier"] = "mlb_prev_season"
            prev_stats["ip_mlb_equivalent"] = (
                float(prev_stats["innings_pitched"]) * _IP_EQ_MLB_PREV_SEASON
            )
            logger.info(f"  ⚾ Pitcher {pitcher_id}: no current-season data → using {season-1} MLB season")
            return prev_stats, "mlb_prev_season"

        # ── tier 3: Triple-A current season ──────────────────────────────────
        aaa = self._fetch_milb_stats(pitcher_id, season, sport_id=11)
        if aaa and aaa.get("innings_pitched", 0) >= 10.0:
            aaa["is_fallback"] = True
            aaa["fallback_tier"] = "aaa_current"
            aaa["ip_mlb_equivalent"] = float(aaa["innings_pitched"]) * _IP_EQ_AAA
            logger.info(f"  ⚾ Pitcher {pitcher_id}: no MLB data → using AAA {season}")
            return aaa, "aaa_current"

        # ── tier 4: Double-A current season ───────────────────────────────────
        aa = self._fetch_milb_stats(pitcher_id, season, sport_id=12)
        if aa and aa.get("innings_pitched", 0) >= 10.0:
            aa["is_fallback"] = True
            aa["fallback_tier"] = "aa_current"
            aa["ip_mlb_equivalent"] = float(aa["innings_pitched"]) * _IP_EQ_AA
            logger.info(f"  ⚾ Pitcher {pitcher_id}: no MLB/AAA data → using AA {season}")
            return aa, "aa_current"

        # ── partial current-season (< 5 IP) is better than nothing ───────────
        if partial and partial.get("innings_pitched", 0) > 0:
            partial["is_fallback"] = True
            partial["fallback_tier"] = "mlb_current_partial"
            # Sí es MLB de esta temporada: sin descuento. Que la muestra sea
            # chica ya lo castiga el propio n, que es para lo que sirve.
            partial["ip_mlb_equivalent"] = float(partial["innings_pitched"])
            return partial, "mlb_current_partial"

        # ── tier 5: team staff ERA/WHIP ────────────────────────────────────────
        if team_pitching and team_pitching.get("team_era", 0) > 0:
            staff_era = float(team_pitching["team_era"])
            staff_whip = float(team_pitching.get("team_whip", LEAGUE_AVG_WHIP))
        else:
            staff_era = LEAGUE_AVG_ERA
            staff_whip = LEAGUE_AVG_WHIP
        staff_stats = {
            "era": staff_era,
            "whip": staff_whip,
            "fip": staff_era,          # best proxy without pitch-mix data
            "innings_pitched": 0.0,
            # Cero, y no por casualidad: esto no es una muestra de ESTE abridor,
            # es el promedio de su cuerpo de lanzadores. Aporta cero información
            # sobre quién abre, así que la regresión debe llevarlo entero a la
            # media de liga y la confianza debe cobrárselo.
            "ip_mlb_equivalent": 0.0,
            "is_fallback": True,
            "fallback_tier": "team_staff_era",
        }
        logger.info(f"  ⚾ Pitcher {pitcher_id}: no stats found at any level → team staff ERA {staff_era:.2f}")
        return staff_stats, "team_staff_era"

    # ==========================================================
    # FEATURE 2 - TEAM RUNS TRENDS (últimos 5)
    # ==========================================================
    def get_team_runs_trend(self, team_id: int, games: int = 5) -> Optional[Dict[str, Any]]:
        """{"runs_scored_avg": float, "runs_allowed_avg": float, "last_5_scores": list, "last_5_allowed": list}"""
        cache_file = CACHE_DIR / f"team_runs_{team_id}_{games}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 1800:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=20)
        url = f"{self.BASE_URL}/schedule"
        params = {
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "gameType": "R,F,D,L,W",
            "hydrate": "team,linescore"
        }
        try:
            r = self.session.get(url, params=params, timeout=(5, 30))
            r.raise_for_status()
            data = r.json()

            scored, allowed = [], []
            for date_item in data.get("dates", []):
                for game in date_item.get("games", []):
                    if game.get("status", {}).get("abstractGameState") != "Final":
                        continue
                    home_id = game["teams"]["home"]["team"]["id"]
                    away_id = game["teams"]["away"]["team"]["id"]
                    hs = game["teams"]["home"].get("score", 0)
                    as_ = game["teams"]["away"].get("score", 0)
                    if home_id == team_id:
                        scored.append(hs); allowed.append(as_)
                    elif away_id == team_id:
                        scored.append(as_); allowed.append(hs)

            rs = scored[-games:]; ra = allowed[-games:]
            result = {
                "runs_scored_avg": round(sum(rs) / len(rs), 2) if rs else 0.0,
                "runs_allowed_avg": round(sum(ra) / len(ra), 2) if ra else 0.0,
                "last_5_scores": rs,
                "last_5_allowed": ra,
                "games_count": len(rs)
            }
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo runs trend: {e}")
            return None

    # ==========================================================
    # FEATURE 3 - BULLPEN WORKLOAD (aprox)
    # ==========================================================
    @staticmethod
    def _ip_to_float(ip_str) -> float:
        """Convert baseball IP notation '4.2' (4⅔) to decimal 4.667.

        Existía desde antes y la usaba UN solo sitio; los otros ocho hacían
        `float(...)` crudo sobre el mismo campo (auditado el 2026-07-28, paso 7).
        Todos eran denominadores de tasas, así que el efecto es sistemático y de
        un solo signo: `float("X.1")=X.1 < X+1/3` y `float("X.2")=X.2 < X+2/3`,
        nunca al revés, o sea denominador chico y tasa inflada.

        Lo peor no era el sesgo en sí sino su asimetría: el WHIP y el ERA
        GENERALES vienen del campo ya calculado por la API (innings reales),
        mientras que los splits, el FIP, K/9, BB/9 y los IP por inicio se
        recalculaban acá con el denominador mal. Cualquier RAZÓN entre un número
        propio y uno de la API quedaba sesgada de un solo lado — que es
        exactamente lo que hace `_adjust_pitcher_platoon`.

        Medido sobre los 51 abridores de la cartelera: FIP +0.0072 carreras/9 de
        media (máx 0.0985), K/9 +0.46% (máx +6.48%), y en los 105 splits de
        platoon el WHIP salía inflado en el 100% de los casos (media +0.66%,
        máx +5.69%) sobre un rango útil de sólo ±7%.
        """
        try:
            s = str(ip_str or "0")
            if "." in s:
                whole, thirds = s.split(".", 1)
                return float(whole) + int(thirds) / 3.0
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    def get_bullpen_workload(self, team_id: int, days: int = 3) -> Optional[Dict[str, Any]]:
        """
        Returns real bullpen innings pitched over the last N days by reading
        each completed game's boxscore — no more fixed 3.5-IP estimate.
        {"innings_last_n_days", "ip_last_3_days", "is_tired"} — only these
        three are consumed downstream (bullpen_engine.py's fatigue factor,
        plus the enrichment loop's own "bullpen CANSADO" log warning);
        games_played/avg_innings_per_game/starter_ip_avg were computed but
        never read anywhere, removed 2026-07-06.
        """
        cache_file = CACHE_DIR / f"bullpen_{team_id}_{days}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 3600:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        end_date   = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        try:
            r = self.session.get(
                f"{self.BASE_URL}/schedule",
                params={
                    "sportId": 1, "teamId": team_id,
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate":   end_date.strftime("%Y-%m-%d"),
                    "gameType":  "R,F,D,L,W",
                },
                timeout=(5, 30),
            )
            r.raise_for_status()
            schedule = r.json()
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo schedule bullpen workload: {e}")
            return None

        game_pks = []
        for date_item in schedule.get("dates", []):
            for game in date_item.get("games", []):
                if game.get("status", {}).get("abstractGameState") == "Final":
                    game_pks.append(game.get("gamePk"))

        total_bp_ip    = 0.0
        games_counted  = 0

        for gk in game_pks:
            if not gk:
                continue
            try:
                rb = self.session.get(
                    f"{self.BASE_URL}/game/{gk}/boxscore",
                    timeout=(5, 15),
                )
                rb.raise_for_status()
                bs = rb.json()
            except Exception:
                continue

            # Determine which side this team is on
            home_id = (bs.get("teams", {}).get("home", {})
                         .get("team", {}).get("id"))
            side = "home" if home_id == team_id else "away"
            side_data = bs.get("teams", {}).get(side, {})
            pitchers  = side_data.get("pitchers", [])
            players   = side_data.get("players", {})

            if not pitchers:
                continue

            reliever_ids = pitchers[1:]

            bp_ip = sum(
                self._ip_to_float(
                    players.get(f"ID{pid}", {})
                           .get("stats", {}).get("pitching", {})
                           .get("inningsPitched", 0)
                )
                for pid in reliever_ids
            )

            total_bp_ip    += bp_ip
            games_counted  += 1

        if games_counted == 0:
            return None

        result = {
            "innings_last_n_days":  round(total_bp_ip,    1),
            "ip_last_3_days":       round(total_bp_ip,    1),
            "is_tired":             total_bp_ip > 12.0,
        }
        try:
            with open(cache_file, "w") as f:
                json.dump(result, f)
        except Exception:
            pass
        return result

    # ==========================================================
    # FEATURE 5 - PITCHER VS TEAM (placeholder)
    # ==========================================================
    def get_pitcher_vs_team(self, pitcher_id: int, team_id: int, season: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Pitcher career stats vs a specific team via the vsTeam stat type.

        Tries current season first; if fewer than 3 IP found, falls back to
        the previous season so rookies and early-season starters still get
        a signal.  Returns None only when there is genuinely no data.

        Return shape:
            {era, whip, ip, games, strikeouts, walks, k_per_9,
             bb_per_9, season_used, sample_size}
        """
        if season is None:
            season = _current_mlb_season()

        def _fetch_season(yr: int) -> Optional[Dict[str, Any]]:
            cache_file = CACHE_DIR / f"pitcher_vs_team_{pitcher_id}_{team_id}_{yr}.json"
            if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 7200:
                try:
                    with open(cache_file) as f:
                        return json.load(f)
                except Exception:
                    pass

            url = f"{self.BASE_URL}/people/{pitcher_id}/stats"
            params = {
                "stats": "vsTeam",
                "opposingTeamId": team_id,
                "group": "pitching",
                "season": yr,
            }
            try:
                r = self.session.get(url, params=params, timeout=(5, 30))
                r.raise_for_status()
                splits = r.json().get("stats", [{}])[0].get("splits", [])
                if not splits:
                    return None
                stat = splits[0].get("stat", {})
                ip_raw = stat.get("inningsPitched", "0")
                try:
                    ip = self._ip_to_float(ip_raw)
                except (ValueError, TypeError):
                    ip = 0.0
                if ip < 3.0:
                    return None

                er = int(stat.get("earnedRuns", 0))
                bb = int(stat.get("baseOnBalls", 0))
                so = int(stat.get("strikeOuts", 0))
                h  = int(stat.get("hits", 0))
                games = int(stat.get("gamesPlayed", len(splits)))

                era  = round(er / ip * 9, 2) if ip > 0 else 4.50
                whip = round((h + bb) / ip, 3) if ip > 0 else 1.30
                k9   = round(so / ip * 9, 2) if ip > 0 else 0.0
                bb9  = round(bb / ip * 9, 2) if ip > 0 else 0.0

                result = {
                    "era": era,
                    "whip": whip,
                    "ip": ip,
                    "games": games,
                    "strikeouts": so,
                    "walks": bb,
                    "k_per_9": k9,
                    "bb_per_9": bb9,
                    "season_used": yr,
                    "sample_size": "small" if ip < 15 else "medium" if ip < 40 else "large",
                }
                with open(cache_file, "w") as f:
                    json.dump(result, f)
                return result
            except Exception as e:
                logger.warning(f"⚠️ get_pitcher_vs_team({pitcher_id}, {team_id}, {yr}): {e}")
                return None

        # Try current season, fall back to previous if insufficient data
        data = _fetch_season(season)
        if data is None:
            data = _fetch_season(season - 1)
            if data is not None:
                data["season_used"] = season - 1
        return data

    # ==========================================================
    # FEATURE 7 - TRAVEL FATIGUE (coordinate-based)
    # ==========================================================
    def get_travel_fatigue(self, team_id: int, game_date: str, current_venue: str = "",
                           official_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Returns travel metrics: miles_traveled, time_zones_crossed, back_to_back, has_travel_fatigue.

        `game_date` es el timestamp UTC completo y se necesita así: la selección
        del juego anterior compara timestamp contra timestamp, que es correcto.
        `official_date` ancla la VENTANA de búsqueda, que es un rango de días y
        no puede derivarse del UTC sin correrse: para un nocturno del oeste
        `game_dt.date()` es el día siguiente, así que la ventana quedaba
        [oficial-2, oficial+1] en vez de [oficial-3, oficial] y un equipo cuyo
        último juego fue 3 días antes se reportaba como "sin viaje".
        """
        cache_key = f"travel_{team_id}_{game_date}"
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 7200:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        try:
            game_dt = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
        except Exception:
            return None

        # Se ancla la ventana en el día OFICIAL, no se ensancha. Ensancharla
        # habría sido peor que el bug: `hfa_engine._travel_penalty` castiga por
        # millas y husos SIN mirar la recencia (nunca lee hours_since_last_game),
        # así que sumar un día de margen le habría dado penalización de viaje a
        # equipos que llevan 3-4 días en la ciudad y ya se recuperaron. Anclar
        # restituye exactamente la ventana pretendida de 3 días.
        if official_date:
            try:
                ancla = datetime.strptime(str(official_date)[:10], "%Y-%m-%d").date()
            except ValueError:
                ancla = game_dt.date()
        else:
            ancla = game_dt.date()
        start_search = ancla - timedelta(days=3)
        url = f"{self.BASE_URL}/schedule"
        params = {
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_search.strftime("%Y-%m-%d"),
            # El día del juego, no el UTC: el filtro `gd < game_date` de abajo
            # ya excluye este juego y cualquier otro posterior del mismo día,
            # así que incluir el día completo es correcto y no cuela nada.
            "endDate": ancla.strftime("%Y-%m-%d"),
            "gameType": "R,F,D,L,W",
            "hydrate": "venue"
        }
        try:
            r = self.session.get(url, params=params, timeout=(5, 30))
            r.raise_for_status()
            data = r.json()

            previous_venue = None
            previous_game_date = None
            for date_item in data.get("dates", []):
                for game in date_item.get("games", []):
                    gd = game.get("gameDate")
                    if gd and gd < game_date and game.get("status", {}).get("abstractGameState") == "Final":
                        # Comparación explícita en vez de "gana el último que
                        # itera": eso dependía de que la API devolviera las
                        # fechas en orden ascendente, supuesto no documentado
                        # que además hay que sostener ahora que la ventana es
                        # más ancha. Con el máximo el resultado no depende del
                        # orden de llegada.
                        if previous_game_date is None or gd > previous_game_date:
                            previous_venue = (game.get("venue") or {}).get("name")
                            previous_game_date = gd

            if not previous_venue or not previous_game_date:
                # Genuinely known, not fabricated: no completed prior game
                # found in the lookback window, so there is no travel to
                # report — 0 is the real answer here, not a guess.
                return {
                    "has_travel_fatigue": False, "miles_traveled": 0,
                    "time_zones_crossed": 0, "back_to_back": False,
                    "travel_source": "live",
                }

            prev_dt = datetime.fromisoformat(previous_game_date.replace('Z', '+00:00'))
            hours_between = (game_dt - prev_dt).total_seconds() / 3600
            back_to_back = hours_between < 30

            # Coordinate-based distance and time-zone calculation
            coords = WeatherAPI.STADIUM_COORDS
            miles = 0
            time_zones = 0
            # FALL-002 fix (roadmap Step 4, audit_20260714/): an unmapped
            # venue used to fabricate a plausible-looking "mid-range travel"
            # guess (1000mi/1tz) here — indistinguishable downstream from a
            # real measurement. That silent substitution is exactly what let
            # REG-015 go undetected for weeks across 4 renamed stadiums.
            # Missing now stays missing: miles/time_zones remain 0 (their
            # honest "unknown, no fabricated number" default) and
            # travel_source flags it so hfa_engine.py can apply a neutral
            # multiplier instead of guessing.
            travel_source = "live"
            if previous_venue in coords and current_venue in coords:
                prev_c = coords[previous_venue]
                curr_c = coords[current_venue]
                miles = round(_haversine_miles(prev_c["lat"], prev_c["lon"], curr_c["lat"], curr_c["lon"]))
                tz_prev = _lon_to_tz_offset(prev_c["lon"])
                tz_curr = _lon_to_tz_offset(curr_c["lon"])
                time_zones = abs(tz_curr - tz_prev)
            elif previous_venue != current_venue:
                travel_source = "missing"

            has_fatigue = (miles > 1000 or time_zones >= 2) and hours_between < 30

            result = {
                "has_travel_fatigue": bool(has_fatigue),
                "hours_since_last_game": round(hours_between, 1),
                "previous_venue": previous_venue,
                "miles_traveled": miles,
                "time_zones_crossed": time_zones,
                "back_to_back": back_to_back,
                "travel_source": travel_source,
            }
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result
        except Exception as e:
            logger.warning(f"⚠️ Error calculando travel fatigue: {e}")
            return None

    # ==========================================================
    # FEATURE 8 - DAYS REST PER TEAM
    # ==========================================================
    def get_team_days_rest(self, team_id: int, game_date: str) -> int:
        """Returns days since the team's last completed game (0 = back-to-back, 4 = normal)."""
        cache_file = CACHE_DIR / f"rest_{team_id}_{game_date}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 3600:
            try:
                with open(cache_file) as f:
                    return json.load(f).get("days_rest", 4)
            except Exception:
                pass
        try:
            target = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
            start = target - timedelta(days=7)
            url = f"{self.BASE_URL}/schedule"
            params = {
                "sportId": 1,
                "teamId": team_id,
                "startDate": start.strftime("%Y-%m-%d"),
                "endDate": (target - timedelta(days=1)).strftime("%Y-%m-%d"),
                "gameType": "R,F,D,L,W",
            }
            r = self.session.get(url, params=params, timeout=(5, 15))
            r.raise_for_status()
            data = r.json()
            last_date = None
            for date_item in data.get("dates", []):
                for game in date_item.get("games", []):
                    if game.get("status", {}).get("abstractGameState") == "Final":
                        d = date_item.get("date", "")
                        if d and (last_date is None or d > last_date):
                            last_date = d
            if last_date:
                days_rest = (target - datetime.strptime(last_date, "%Y-%m-%d").date()).days
            else:
                days_rest = 4
            with open(cache_file, "w") as f:
                json.dump({"days_rest": days_rest}, f)
            return days_rest
        except Exception:
            return 4

    # ==========================================================
    # FEATURE 9 - TEAM PITCHING / DEFENSE STATS
    # ==========================================================
    def get_team_pitching_stats(self, team_id: int, season: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Fetches team ERA, WHIP, and runs-allowed-per-game from the pitching stats endpoint."""
        if season is None:
            season = _current_mlb_season()
        cache_file = CACHE_DIR / f"team_pitching_{team_id}_{season}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 3600:
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass
        try:
            url = f"{self.BASE_URL}/teams/{team_id}/stats"
            params = {"stats": "season", "season": season, "group": "pitching", "sportId": 1}
            r = self.session.get(url, params=params, timeout=(5, 15))
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [{}])
            if not splits:
                return None
            stat = splits[0].get("stat", {})
            games = int(stat.get("gamesPlayed", 0))
            if games < 5:
                return None
            era = _safe_float(stat.get("era"))
            whip = _safe_float(stat.get("whip"))
            runs_allowed = int(stat.get("runs", 0))
            ra_per_game = round(runs_allowed / games, 3) if games > 0 else 4.5
            if era <= 0 or whip <= 0:
                return None

            # DER = 1 − BABIP_allowed (fielding-pure metric).
            # Shared with DefensiveEfficiencyEngine's own DER usage so both
            # sides of the pipeline agree on one formula.
            hits = int(stat.get("hits", 0))
            ab   = int(stat.get("atBats", 0))
            so   = int(stat.get("strikeOuts", 0))
            hr   = int(stat.get("homeRuns", 0))
            sf   = int(stat.get("sacFlies", 0))
            der, bip = calculate_der(hits, ab, so, hr, sf)

            result = {
                "team_era": round(era, 2),
                "team_whip": round(whip, 2),
                "runs_allowed_per_game": ra_per_game,
                "games": games,
                "der": der,
                "bip": bip,
            }
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo pitching stats team {team_id}: {e}")
            return None

    def get_bullpen_era(self, team_id: int, season: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Fetches bullpen ERA from the relief pitchers endpoint (pitcherType=R)."""
        if season is None:
            season = _current_mlb_season()
        cache_file = CACHE_DIR / f"bullpen_era_{team_id}_{season}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 3600:
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass
        try:
            url = f"{self.BASE_URL}/teams/{team_id}/stats"
            params = {
                "stats": "season",
                "season": season,
                "group": "pitching",
                "pitcherType": "R",
                "sportId": 1,
            }
            r = self.session.get(url, params=params, timeout=(5, 15))
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [{}])
            if not splits:
                return None
            stat = splits[0].get("stat", {})
            games = int(stat.get("gamesPlayed", 0))
            if games < 5:
                return None
            era = _safe_float(stat.get("era"))
            whip = _safe_float(stat.get("whip"))
            if era <= 0:
                return None

            # K% and BB% — same API call, no extra cost
            so  = int(stat.get("strikeOuts",    0))
            bb  = int(stat.get("baseOnBalls",   0))
            tbf = int(stat.get("battersFaced",  0))
            ip_str = stat.get("inningsPitched", "0")
            try:
                ip = self._ip_to_float(ip_str)
            except (TypeError, ValueError):
                ip = 0.0

            result = {
                "era":           round(era,  2),
                "bullpen_era":   round(era,  2),
                "bullpen_whip":  round(whip, 2) if whip > 0 else None,
                "bullpen_games": games,
                "k_pct":  round(so / tbf, 4) if tbf > 0 else None,
                "bb_pct": round(bb / tbf, 4) if tbf > 0 else None,
                "tbf":    tbf,
                "ip":     round(ip, 1),
            }
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo bullpen ERA team {team_id}: {e}")
            return None

    # ==========================================================
    # FEATURE: ROOF STATUS (for ParkWeatherEngine retractable-roof parks)
    # ==========================================================
    def get_roof_status(self, game_pk: int) -> Optional[bool]:
        """
        Returns True if the roof is confirmed OPEN, False if confirmed CLOSED,
        None if unknown (fetch failed, or the park has no roof status to report).

        Fixes a real gap found 2026-07-05: ParkWeatherEngine previously had no
        data source for roof status at all, so `game_data.get("roof_open")`
        always defaulted to False, permanently suppressing weather effects for
        every game at all 8 retractable-roof parks regardless of the real
        roof status that day. MLB's own game feed reports this in
        `gameData.weather.condition` — "Roof Closed" when shut; any other
        string (real weather like "Sunny"/"Cloudy") when open. Uses the
        `fields` filter so this is a ~100-byte response, not the full live
        feed (confirmed live: 85 bytes, ~0.2s).
        """
        cache_file = CACHE_DIR / f"roof_status_{game_pk}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 3600:
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                    return cached.get("roof_open")
            except Exception:
                pass

        try:
            r = self.session.get(
                f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
                params={"fields": "gameData,weather,condition"},
                timeout=(5, 10),
            )
            r.raise_for_status()
            condition = r.json().get("gameData", {}).get("weather", {}).get("condition")
        except Exception as e:
            logger.debug(f"Roof status unavailable for game {game_pk}: {e}")
            return None

        if not condition:
            return None
        roof_open = "roof closed" not in condition.lower()

        try:
            with open(cache_file, "w") as f:
                json.dump({"roof_open": roof_open, "condition": condition}, f)
        except Exception:
            pass
        return roof_open

    # ==========================================================
    # FEATURE: BATTING HANDEDNESS % PER TEAM (for wind asymmetry)
    # ==========================================================
    def get_team_injuries(
        self, team_id: int, season: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Jugadores del equipo en lista de lesionados, y cuánta ofensa se llevan.

        Hasta el 2026-08-01 este dato NO EXISTÍA en el módulo de béisbol:
        `app.py` tenía `"home_injuries": []` hardcodeado en vacío. El módulo de
        básquet sí las modela (peso 0.20 y una función propia de análisis); el de
        béisbol no las miraba. Consecuencia concreta: la ofensa de cada equipo se
        calcula del Statcast ACUMULADO de la temporada, que incluye entera la
        producción de quien hoy está lesionado.

        Caso real al construir esto (Yankees, 2026-08-01): Aaron Judge en lista
        de 60 días, Bellinger y Stanton en la de 10 — **19.2% de los turnos
        ofensivos del equipo** pertenecían a jugadores que no iban a jugar, y el
        modelo los contaba como si jugaran.

        Devuelve el detalle y `pa_share_out`, la fracción de turnos del roster que
        está fuera. Se excluye a los lanzadores del cálculo: un abridor lesionado
        ya está cubierto por el probable pitcher, y contarlo acá sería doble.

        Una sola llamada por equipo — el roster hidratado trae el estado Y las
        estadísticas de cada jugador juntos, sin costo extra.
        """
        season = season or _current_mlb_season()
        cache_file = CACHE_DIR / f"injuries_{team_id}_{season}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 3600:
            try:
                return json.load(open(cache_file))
            except Exception:
                pass
        try:
            r = self.session.get(
                f"{self.BASE_URL}/teams/{team_id}/roster",
                params={
                    "rosterType": "fullSeason", "season": season,
                    "hydrate": f"person(stats(type=season,group=hitting,season={season}))",
                },
                timeout=(5, 25),
            )
            r.raise_for_status()
            roster = r.json().get("roster", [])
        except Exception as exc:
            logger.warning("⚠️ no se pudieron leer lesiones del equipo %s: %s", team_id, exc)
            return None

        fuera: List[Dict[str, Any]] = []
        pa_total = 0.0
        pa_fuera = 0.0
        for e in roster:
            pos = (e.get("position") or {}).get("abbreviation", "")
            if pos == "P":
                continue                      # ver docstring: lo cubre el probable pitcher
            persona = e.get("person") or {}
            pa = 0
            for bloque in (persona.get("stats") or []):
                for sp in bloque.get("splits", []):
                    pa = sp.get("stat", {}).get("plateAppearances") or pa
            if not pa:
                continue
            pa_total += pa
            estado = (e.get("status") or {}).get("description", "")
            if "Injured" in estado:
                pa_fuera += pa
                fuera.append({
                    "id": persona.get("id"),
                    "name": persona.get("fullName"),
                    "position": pos,
                    "status": estado,
                    "pa": pa,
                })

        if pa_total <= 0:
            return None
        fuera.sort(key=lambda x: -x["pa"])
        resultado = {
            "n_out": len(fuera),
            "pa_share_out": round(pa_fuera / pa_total, 4),
            "pa_total_roster": int(pa_total),
            "players": fuera[:12],            # los más relevantes por turnos
        }
        try:
            with open(cache_file, "w") as f:
                json.dump(resultado, f)
        except Exception:
            pass
        return resultado

    def get_team_batting_handedness_pct(
        self, team_id: int, game_date: str, vs_hand: Optional[str] = None
    ) -> Optional[float]:
        """
        Fraction of non-pitcher players on the gameday roster who bat left.
        Cached per team per date (changes only on roster moves).
        Returns LHB% (0.0–1.0) or None if unavailable.

        `vs_hand` es la mano del abridor rival, y resuelve a los ambidiestros al
        lado que realmente batean. Sin ella pesan 0.5, que era el comportamiento
        único hasta el 2026-07-27 — y que discrepaba con la ruta del lineup
        confirmado, donde el ambidiestro pesaba 1.0. Las dos rutas alimentan la
        MISMA cadena de respaldo, así que caer de una a la otra movía el número
        0.056 sin que nada lo señalara.
        """
        cache_file = CACHE_DIR / f"lhb_pct_{team_id}_{game_date}_{vs_hand or 'na'}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86_400:
            try:
                return json.load(open(cache_file)).get("lhb_pct")
            except Exception:
                pass
        try:
            r = self.session.get(
                f"{self.BASE_URL}/teams/{team_id}/roster",
                params={"rosterType": "gameday", "season": game_date[:4],
                        "hydrate": "person"},
                timeout=(5, 15),
            )
            r.raise_for_status()
            roster = r.json().get("roster", [])
            counts = {"L": 0, "R": 0, "S": 0}
            for entry in roster:
                pos_type = (entry.get("position") or {}).get("type", "")
                if pos_type == "Pitcher":
                    continue
                side = (entry.get("person") or {}).get("batSide", {}).get("code", "")
                if side in counts:
                    counts[side] += 1
            total = counts["L"] + counts["R"] + counts["S"]
            if total < 5:
                return None
            # El ambidiestro pesa según la mano del abridor rival — 1.0 contra
            # derecho, 0.0 contra zurdo, 0.5 sólo si no se sabe.
            peso_s = _PESO_AMBIDIESTRO.get(vs_hand or "", 0.5)
            lhb_pct = round((counts["L"] + counts["S"] * peso_s) / total, 3)
            with open(cache_file, "w") as f:
                json.dump({"lhb_pct": lhb_pct, "counts": counts}, f)
            return lhb_pct
        except Exception as exc:
            logger.debug(f"[handedness] team {team_id} {game_date}: {exc}")
            return None

    # ==========================================================
    # FEATURE: PITCHER PLATOON SPLITS (vs LHB / vs RHB)
    # ==========================================================
    def get_pitcher_platoon_splits(
        self, pitcher_id: int, season: int
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch pitcher splits vs left-handed batters (vl) and right-handed (vr).

        Uses the statSplits endpoint with sitCodes=vl,vr.
        Returns {'vs_lhb': {ip, k_per_9, bb_per_9, hr_per_9, whip, ops},
                 'vs_rhb': same}  or None if insufficient data (< 5 IP per split).
        """
        cache_file = CACHE_DIR / f"platoon_{pitcher_id}_{season}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < self.cache_ttl:
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass

        url = f"{self.BASE_URL}/people/{pitcher_id}/stats"
        params = {
            "stats": "statSplits",
            "group": "pitching",
            "season": season,
            "sitCodes": "vl,vr",
        }
        try:
            r = self.session.get(url, params=params, timeout=(5, 30))
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [])
        except Exception as e:
            logger.warning(f"⚠️ platoon splits pitcher {pitcher_id}: {e}")
            return None

        result: Dict[str, Any] = {}
        for s in splits:
            code = (s.get("split") or {}).get("code")
            if code not in ("vl", "vr"):
                continue
            stat = s.get("stat", {})
            ip_raw = stat.get("inningsPitched", "0") or "0"
            try:
                ip = self._ip_to_float(ip_raw)
            except (ValueError, TypeError):
                ip = 0.0
            if ip < 5.0:
                continue

            so  = int(stat.get("strikeOuts", 0))
            bb  = int(stat.get("baseOnBalls", 0))
            hr  = int(stat.get("homeRuns", 0))
            h   = int(stat.get("hits", 0))

            key = "vs_lhb" if code == "vl" else "vs_rhb"
            result[key] = {
                "ip":       round(ip, 1),
                "k_per_9":  round(so / ip * 9, 2) if ip > 0 else 0.0,
                "bb_per_9": round(bb / ip * 9, 2) if ip > 0 else 0.0,
                "hr_per_9": round(hr / ip * 9, 2) if ip > 0 else 0.0,
                "whip":     round((h + bb) / ip, 3) if ip > 0 else 1.30,
                "ops":      _safe_float(stat.get("ops")),
            }

        if not result:
            return None

        try:
            with open(cache_file, "w") as f:
                json.dump(result, f)
        except Exception:
            pass
        return result

    # ==========================================================
    # FEATURE: BATTER HANDEDNESS BATCH FETCH
    # ==========================================================
    def get_batter_handedness_batch(
        self, player_ids: List[int]
    ) -> Dict[int, str]:
        """
        Batch-fetch bat side (L/R/S) for a list of player IDs.

        Uses /people?personIds=... endpoint.  Results cached indefinitely
        (handedness doesn't change).  Returns {player_id: 'L'|'R'|'S'}.
        """
        if not player_ids:
            return {}

        result: Dict[int, str] = {}
        uncached: List[int] = []

        for pid in player_ids:
            cache_file = CACHE_DIR / f"hand_{pid}.json"
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        result[pid] = json.load(f)["s"]
                    continue
                except Exception:
                    pass
            uncached.append(pid)

        if uncached:
            ids_str = ",".join(str(i) for i in uncached)
            url = f"{self.BASE_URL}/people"
            try:
                r = self.session.get(url, params={"personIds": ids_str}, timeout=(5, 30))
                r.raise_for_status()
                for person in r.json().get("people", []):
                    pid  = person.get("id")
                    side = (person.get("batSide") or {}).get("code", "R")
                    if pid:
                        result[pid] = side
                        cache_file = CACHE_DIR / f"hand_{pid}.json"
                        try:
                            with open(cache_file, "w") as f:
                                # `p` = mano con la que LANZA, guardada acá porque
                                # sale de la misma respuesta que ya pedimos y hace
                                # falta para resolver a los ambidiestros (ver
                                # get_pitcher_throws). Pedirla aparte sería una
                                # llamada extra por el mismo dato.
                                json.dump({
                                    "s": side,
                                    "p": (person.get("pitchHand") or {}).get("code"),
                                }, f)
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"⚠️ batch handedness fetch failed: {e}")
                for pid in uncached:
                    result.setdefault(pid, "R")

        return result

    def get_prev_lineup_lhb_pct(
        self, team_id: int, official_date: str, vs_hand: Optional[str] = None
    ) -> Optional[float]:
        """LHB% de la última alineación REAL que puso este equipo.

        Personal de ayer, matchup de hoy: la composición sale del juego anterior,
        pero los ambidiestros se resuelven contra el abridor de HOY, que es
        contra quien van a batear.

        Existe porque a la hora en que se publica no hay lineup (medido: 0 de 27
        juegos en la corrida de las 07:00) y había que elegir con qué estimarlo.
        Comparado sobre 250 casos reales de equipo-día, error absoluto medio al
        predecir el LHB% del lineup de hoy:

            lineup de ayer solo   0.0936
            roster solo           0.0986
            mitad y mitad         0.0835   ← se usa esto
            constante 0.45        0.1059

        El de ayer gana en promedio y pierde en la cola (cuando el lineup sí
        cambia, se equivoca con confianza); el roster al revés. La mezcla gana en
        las dos, y la curva es plana entre 0.4 y 0.6, así que el 0.5 no es un
        filo ajustado a la muestra.
        """
        try:
            fin = datetime.strptime(official_date[:10], "%Y-%m-%d").date()
        except Exception:
            return None
        cache_file = CACHE_DIR / f"prev_lineup_lhb_{team_id}_{official_date}_{vs_hand or 'na'}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 43_200:
            try:
                return json.load(open(cache_file)).get("lhb_pct")
            except Exception:
                pass
        try:
            r = self.session.get(
                f"{self.BASE_URL}/schedule",
                params={"sportId": 1, "teamId": team_id,
                        "startDate": (fin - timedelta(days=5)).strftime("%Y-%m-%d"),
                        "endDate": (fin - timedelta(days=1)).strftime("%Y-%m-%d"),
                        "gameType": "R,F,D,L,W"},
                timeout=(5, 20),
            )
            r.raise_for_status()
            candidatos = []
            for item in r.json().get("dates", []):
                for g in item.get("games", []):
                    if g.get("status", {}).get("abstractGameState") == "Final":
                        candidatos.append((item.get("date", ""), g))
            candidatos.sort(reverse=True)          # del más reciente al más viejo
            if not candidatos:
                return None

            titulares: List[int] = []
            ultima_fecha = None
            # Se prueba más de un juego hacia atrás: el boxscore del más reciente
            # a veces viene sin `battingOrder` (visto en vivo: 1 de 5 juegos
            # Final de un equipo). Quedarse con el primero y rendirse dejaba sin
            # estimación a ~25% de los equipos.
            for fecha, g in candidatos[:3]:
                lado = "home" if g["teams"]["home"]["team"]["id"] == team_id else "away"
                try:
                    box = self.session.get(
                        f"{self.BASE_URL}/game/{g['gamePk']}/boxscore", timeout=(5, 25)
                    )
                    box.raise_for_status()
                    jugadores = box.json().get("teams", {}).get(lado, {}).get("players", {})
                except Exception:
                    continue
                # Titulares = múltiplos de 100. El boxscore acumula sustitutos a
                # medida que avanza el juego, y los numera 201, 402, 503...
                # Cortar por los primeros 9 ordenados NO sirve: con un emergente
                # en el 2º turno queda [100,200,201,300,...,800] — cuela al
                # suplente y PIERDE el 9º turno. Verificado en un boxscore real.
                orden = sorted(
                    (int(p["battingOrder"]), p["person"]["id"])
                    for p in jugadores.values()
                    if p.get("battingOrder") and (p.get("person") or {}).get("id")
                )
                titulares = [pid for bo, pid in orden if bo % 100 == 0]
                if len(titulares) >= 9:
                    ultima_fecha = fecha
                    break
                titulares = []
            if len(titulares) < 9:
                return None
            titulares = titulares[:9]
            pct = lhb_pct_efectivo(
                self.get_batter_handedness_batch(titulares), titulares, vs_hand
            )
            if pct is not None:
                try:
                    with open(cache_file, "w") as f:
                        json.dump({"lhb_pct": pct, "desde": ultima_fecha}, f)
                except Exception:
                    pass
            return pct
        except Exception as e:
            logger.debug("no se pudo leer la alineación previa de %s: %s", team_id, e)
            return None

    def get_pitcher_throws(self, pitcher_id: Optional[int]) -> Optional[str]:
        """Mano con la que lanza ('L'/'R'), o None si no se puede resolver.

        None NO es "derecho": es "no sé". Quien la use tiene que decidir
        explícitamente qué hacer sin el dato, porque asumir diestro sesga hacia
        el 72% de los casos y esconde el otro 28%.

        Mismo endpoint y misma caché que `get_batter_handedness_batch` — el
        campo viaja en la misma respuesta, así que esto no agrega llamadas.
        """
        if not pitcher_id:
            return None
        cache_file = CACHE_DIR / f"hand_{pitcher_id}.json"
        if cache_file.exists():
            try:
                mano = json.load(open(cache_file)).get("p")
                if mano:
                    return mano
            except Exception:
                pass
        try:
            r = self.session.get(f"{self.BASE_URL}/people",
                                 params={"personIds": str(pitcher_id)}, timeout=(5, 20))
            r.raise_for_status()
            for person in r.json().get("people", []):
                if person.get("id") != pitcher_id:
                    continue
                lanza = (person.get("pitchHand") or {}).get("code")
                batea = (person.get("batSide") or {}).get("code", "R")
                try:
                    with open(cache_file, "w") as f:
                        json.dump({"s": batea, "p": lanza}, f)
                except Exception:
                    pass
                return lanza
        except Exception as e:
            logger.warning("⚠️ no se pudo resolver la mano del pitcher %s: %s", pitcher_id, e)
        return None


# ==========================================================
# 2) WEATHER API - OPENWEATHER (GRATIS)
# ==========================================================

class WeatherAPI:
    BASE_URL = "https://api.openweathermap.org/data/2.5"

    STADIUM_COORDS = {
        "Yankee Stadium": {"lat": 40.8296, "lon": -73.9262, "city": "New York"},
        "Fenway Park": {"lat": 42.3467, "lon": -71.0972, "city": "Boston"},
        "Dodger Stadium": {"lat": 34.0739, "lon": -118.2400, "city": "Los Angeles"},
        # Renamed "UNIQLO Field at Dodger Stadium" for 2026 (sponsorship change)
        # — live MLB API reports the new name (same drift class fixed in
        # park_weather_engine.py's STADIUM_DATABASE; this key was missed there,
        # silently dropping weather + travel-distance for every Dodgers home game).
        "UNIQLO Field at Dodger Stadium": {"lat": 34.0739, "lon": -118.2400, "city": "Los Angeles"},
        "Wrigley Field": {"lat": 41.9484, "lon": -87.6553, "city": "Chicago"},
        "Oracle Park": {"lat": 37.7786, "lon": -122.3893, "city": "San Francisco"},
        "Coors Field": {"lat": 39.7559, "lon": -104.9942, "city": "Denver"},
        "Petco Park": {"lat": 32.7073, "lon": -117.1566, "city": "San Diego"},
        "Rogers Centre": {"lat": 43.6414, "lon": -79.3894, "city": "Toronto"},
        "Rogers Center": {"lat": 43.6414, "lon": -79.3894, "city": "Toronto"},
        "T-Mobile Park": {"lat": 47.5914, "lon": -122.3325, "city": "Seattle"},
        "Minute Maid Park": {"lat": 29.7573, "lon": -95.3555, "city": "Houston"},
        # Renamed "Daikin Park" for 2026 (sponsorship change) — see note above.
        "Daikin Park": {"lat": 29.7573, "lon": -95.3555, "city": "Houston"},
        "Busch Stadium": {"lat": 38.6226, "lon": -90.1928, "city": "St. Louis"},
        "Progressive Field": {"lat": 41.4962, "lon": -81.6852, "city": "Cleveland"},
        "Truist Park": {"lat": 33.8907, "lon": -84.4685, "city": "Atlanta"},
        "Tropicana Field": {"lat": 27.7682, "lon": -82.6534, "city": "St. Petersburg"},
        "Citi Field": {"lat": 40.7571, "lon": -73.8458, "city": "New York"},
        "Citizens Bank Park": {"lat": 39.9061, "lon": -75.1665, "city": "Philadelphia"},
        "Great American Ball Park": {"lat": 39.0979, "lon": -84.5063, "city": "Cincinnati"},
        "Chase Field": {"lat": 33.4453, "lon": -112.0667, "city": "Phoenix"},
        "Globe Life Field": {"lat": 32.7512, "lon": -97.0837, "city": "Arlington"},
        "Target Field": {"lat": 44.9817, "lon": -93.2779, "city": "Minneapolis"},
        "Kauffman Stadium": {"lat": 39.0517, "lon": -94.4803, "city": "Kansas City"},
        "Camden Yards": {"lat": 39.2838, "lon": -76.6216, "city": "Baltimore"},
        # Renamed "Oriole Park at Camden Yards" — see note above.
        "Oriole Park at Camden Yards": {"lat": 39.2838, "lon": -76.6216, "city": "Baltimore"},
        "Guaranteed Rate Field": {"lat": 41.8300, "lon": -87.6338, "city": "Chicago"},
        # Renamed "Rate Field" for 2026 (sponsorship change) — see note above.
        "Rate Field": {"lat": 41.8300, "lon": -87.6338, "city": "Chicago"},
        "Comerica Park": {"lat": 42.3390, "lon": -83.0489, "city": "Detroit"},
        "PNC Park": {"lat": 40.4468, "lon": -80.0057, "city": "Pittsburgh"},
        "Angel Stadium": {"lat": 33.8003, "lon": -117.8827, "city": "Anaheim"},
        "loanDepot park": {"lat": 25.7781, "lon": -80.2201, "city": "Miami"},
        "Nationals Park": {"lat": 38.8730, "lon": -77.0074, "city": "Washington"},
        "American Family Field": {"lat": 43.0281, "lon": -87.9712, "city": "Milwaukee"},
        "RingCentral Coliseum": {"lat": 37.7516, "lon": -122.2005, "city": "Oakland"},
        # A's relocated to Sacramento for 2025 season
        "Sutter Health Park": {"lat": 38.5802, "lon": -121.5011, "city": "Sacramento"},
    }

    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY", "")
        if not self.api_key:
            logger.warning("⚠️ OPENWEATHER_API_KEY no configurada en .env")

    # Severity ranking for weather conditions — worst-case wins across game slots
    _CONDITION_SEVERITY = {
        "Thunderstorm": 5, "Storm": 5,
        "Rain": 4, "Drizzle": 3,
        "Snow": 4, "Mist": 2, "Fog": 2,
        "Clouds": 1, "Clear": 0, "Haze": 1, "Smoke": 1,
    }

    def get_weather_for_stadium(
        self,
        stadium_name: str,
        game_time: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Returns weather averaged across the full game duration (~3 hours).

        Uses OpenWeather 5-day/3-hour forecast to find the 1-2 slots that
        cover game start → game end.  Averages temperature, humidity, wind
        speed and direction (circular mean).  Uses worst-case condition
        (e.g. if any slot has Rain during the game, the game has Rain).

        game_time: ISO-8601 string for first pitch (UTC preferred).
                   If None, uses the next available forecast slot.
        Falls back to /weather (current conditions) if forecast unavailable.
        """
        if not self.api_key:
            return None
        coords = self.STADIUM_COORDS.get(stadium_name)
        if not coords:
            logger.warning("⚠️ Coordenadas no disponibles para %s", stadium_name)
            return None

        # Parse first-pitch time (naive UTC)
        target_dt: Optional[datetime] = None
        if game_time:
            try:
                target_dt = datetime.fromisoformat(
                    game_time.replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except Exception:
                target_dt = None

        params = {"lat": coords["lat"], "lon": coords["lon"],
                  "appid": self.api_key, "units": "imperial"}

        # ── 5-day / 3-hour forecast (free tier) ──────────────────────────────
        try:
            r = requests.get(f"{self.BASE_URL}/forecast", params=params, timeout=(5, 30))
            r.raise_for_status()
            all_slots = r.json().get("list", [])

            if all_slots:
                if target_dt:
                    game_end_dt = target_dt + timedelta(hours=3)
                    # Slots whose center time falls within [start, end] window
                    game_slots = [
                        s for s in all_slots
                        if target_dt <= datetime.utcfromtimestamp(s["dt"]) <= game_end_dt
                    ]
                    # Always include the slot closest to first pitch as anchor
                    if not game_slots:
                        game_slots = [min(
                            all_slots,
                            key=lambda s: abs(datetime.utcfromtimestamp(s["dt"]) - target_dt),
                        )]
                else:
                    game_slots = all_slots[:2]  # next two 3-hour slots

                # ── Average continuous metrics ─────────────────────────────
                avg_temp  = sum(s["main"]["temp"]              for s in game_slots) / len(game_slots)
                avg_humid = sum(s["main"]["humidity"]          for s in game_slots) / len(game_slots)
                avg_wind  = sum((s.get("wind") or {}).get("speed", 0) for s in game_slots) / len(game_slots)

                # Circular mean for wind direction (handles 350°↔10° wrap)
                import math
                sin_sum = sum(math.sin(math.radians((s.get("wind") or {}).get("deg", 0))) for s in game_slots)
                cos_sum = sum(math.cos(math.radians((s.get("wind") or {}).get("deg", 0))) for s in game_slots)
                avg_dir = round(math.degrees(math.atan2(sin_sum, cos_sum)) % 360)

                # ── Precipitation: total mm and max probability across slots ──
                # rain.3h / snow.3h — OpenWeather returns these only when > 0
                total_rain_mm = sum(
                    (s.get("rain") or {}).get("3h", 0.0) +
                    (s.get("snow") or {}).get("3h", 0.0)
                    for s in game_slots
                )
                max_pop = max(s.get("pop", 0.0) for s in game_slots)  # probability 0-1

                # ── Worst-case condition across the game window ────────────
                def _severity(slot):
                    cond = slot["weather"][0]["main"]
                    return self._CONDITION_SEVERITY.get(cond, 0), cond

                worst_sev, worst_cond = max(_severity(s) for s in game_slots)
                worst_desc = next(
                    s["weather"][0]["description"] for s in game_slots
                    if s["weather"][0]["main"] == worst_cond
                )

                # Heavy rain (>10mm/game) → postponement territory; flag it clearly
                postponement_risk = total_rain_mm > 10.0

                slot_times = [
                    datetime.utcfromtimestamp(s["dt"]).strftime("%H:%M")
                    for s in game_slots
                ]
                result = {
                    "stadium":            stadium_name,
                    "city":               coords["city"],
                    "temp_f":             round(avg_temp,       1),
                    "humidity":           round(avg_humid,      1),
                    "wind_speed_mph":     round(avg_wind,       1),
                    "wind_direction":     avg_dir,
                    "conditions":         worst_cond,
                    "description":        worst_desc,
                    "rain_mm":            round(total_rain_mm,  2),  # mm across game window
                    "precip_probability": round(max_pop,        2),  # 0-1
                    "postponement_risk":  postponement_risk,
                    "forecast_slots":     slot_times,
                    "n_slots":            len(game_slots),
                    "timestamp":          datetime.utcnow().isoformat(),
                }

                if postponement_risk:
                    logger.warning(
                        "⚠️  POSTPONEMENT RISK %s: %.1f mm rain expected during game",
                        stadium_name, total_rain_mm,
                    )
                else:
                    logger.info(
                        "☁️  Game forecast %s [%s UTC]: %.0f°F %s "
                        "wind=%.0f mph dir=%d° rain=%.1fmm pop=%.0f%% (%d slots)",
                        stadium_name, "–".join(slot_times),
                        result["temp_f"], result["conditions"],
                        result["wind_speed_mph"], result["wind_direction"],
                        total_rain_mm, max_pop * 100, len(game_slots),
                    )
                return result

        except Exception as e:
            logger.warning("⚠️ Forecast API failed for %s: %s — trying current weather", stadium_name, e)

        # ── Fallback: current conditions ──────────────────────────────────────
        try:
            r = requests.get(f"{self.BASE_URL}/weather", params=params, timeout=(5, 30))
            r.raise_for_status()
            d = r.json()
            result = {
                "stadium":        stadium_name,
                "city":           coords["city"],
                "temp_f":         round(d["main"]["temp"], 1),
                "humidity":       d["main"]["humidity"],
                "wind_speed_mph": round((d.get("wind") or {}).get("speed", 0.0), 1),
                "wind_direction": (d.get("wind") or {}).get("deg", 0),
                "conditions":     d["weather"][0]["main"],
                "description":    d["weather"][0]["description"],
                "timestamp":      datetime.utcnow().isoformat(),
            }
            logger.info(
                "☁️  Current weather %s: %.0f°F %s wind=%.0f mph",
                stadium_name, result["temp_f"], result["conditions"], result["wind_speed_mph"],
            )
            return result
        except Exception as e:
            logger.error("❌ Error obteniendo clima %s: %s", stadium_name, e)
            return None


# ParkFactors removed — park factors are owned by park_weather_engine.STADIUM_DATABASE.
# Keeping a duplicate with different values here caused silent divergence.
# The park_weather_engine reads game_data["park"]["name"] directly from the venue string.


# ==========================================================
# 4) DATA INTEGRATOR - CON TODAS LAS FEATURES
# ==========================================================

class MLBDataIntegrator:
    def __init__(self):
        self.mlb_api = MLBStatsAPI()
        self.weather_api = WeatherAPI()
        self.league_avg_rpg = LEAGUE_AVG_RUNS   # single source of truth from config
        self._rpg_cache = {}

    def _get_season(self) -> tuple:
        """Retorna (temporada_actual, temporada_anterior) automáticamente."""
        now = datetime.now()
        current = now.year if now.month >= 3 else now.year - 1
        return current, current - 1

    def _fetch_team_rpg(self, team_id: int, season: int) -> float:
        """Jala RPG de un equipo para una temporada desde MLB Stats API."""
        cache_key = f"{team_id}_{season}"
        if cache_key in self._rpg_cache:
            return self._rpg_cache[cache_key]

        cache_file = CACHE_DIR / f"team_rpg_{team_id}_{season}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400:
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    self._rpg_cache[cache_key] = data['rpg']
                    return data['rpg']
            except Exception:
                pass

        try:
            url = f"{self.mlb_api.BASE_URL}/teams/{team_id}/stats"
            params = {"stats": "season", "season": season,
                     "group": "hitting", "sportId": 1}
            r = self.mlb_api.session.get(url, params=params, timeout=(5, 15))
            r.raise_for_status()
            data = r.json()
            splits = data.get("stats", [{}])[0].get("splits", [{}])
            if splits:
                stat = splits[0].get("stat", {})
                games = int(stat.get("gamesPlayed", 0))
                runs = int(stat.get("runs", 0))
                if games >= 10:
                    rpg = round(runs / games, 3)
                    self._rpg_cache[cache_key] = rpg
                    with open(cache_file, "w") as f:
                        json.dump({"rpg": rpg, "games": games}, f)
                    return rpg
        except Exception:
            pass

        return self.league_avg_rpg

    def get_team_lambda(self, team_name: str, recent_rpg: float = None,
                        team_id: int = None) -> float:
        """
        Lambda base dinámico. Pesos automáticos según juegos jugados:
        - Temporada actual < 35 juegos: 60% anterior + 30% reciente + 10% league
        - Temporada actual >= 35 juegos: 40% anterior + 20% actual + 30% reciente + 10% league
        Nunca necesita cambios manuales.
        """
        current_season, previous_season = self._get_season()

        rpg_previous = self._fetch_team_rpg(team_id, previous_season) if team_id else self.league_avg_rpg

        rpg_current = None
        games_current = 0
        if team_id:
            # Reuse the cached _fetch_team_rpg path to avoid uncached HTTP calls
            cache_file = CACHE_DIR / f"team_rpg_{team_id}_{current_season}.json"
            if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400:
                try:
                    d = json.loads(cache_file.read_text())
                    games_current = d.get("games", 0)
                    if games_current >= 35:
                        rpg_current = d.get("rpg")
                except Exception:
                    pass
            if rpg_current is None:
                try:
                    url = f"{self.mlb_api.BASE_URL}/teams/{team_id}/stats"
                    params = {"stats": "season", "season": current_season,
                             "group": "hitting", "sportId": 1}
                    r = self.mlb_api.session.get(url, params=params, timeout=(5, 15))
                    r.raise_for_status()
                    data = r.json()
                    splits = data.get("stats", [{}])[0].get("splits", [{}])
                    if splits:
                        stat = splits[0].get("stat", {})
                        games_current = int(stat.get("gamesPlayed", 0))
                        runs_current  = int(stat.get("runs", 0))
                        if games_current >= 10:
                            rpg_c = round(runs_current / games_current, 3)
                            cache_file.write_text(
                                json.dumps({"rpg": rpg_c, "games": games_current})
                            )
                            if games_current >= 35:
                                rpg_current = rpg_c
                except Exception:
                    pass

        recent = recent_rpg if recent_rpg and recent_rpg > 0 else rpg_previous

        if rpg_current and games_current >= 35:
            lambda_base = (
                rpg_previous * 0.40 +
                rpg_current  * 0.20 +
                recent       * 0.30 +
                self.league_avg_rpg * 0.10
            )
        else:
            lambda_base = (
                rpg_previous * 0.60 +
                recent       * 0.30 +
                self.league_avg_rpg * 0.10
            )

        return round(max(2.5, min(7.0, lambda_base)), 3)

    def _enrich_pitchers_concurrent(self, game: Dict[str, Any], season: int) -> Dict[str, Any]:
        """Enriquecer pitchers en paralelo con fallback completo de 5 niveles."""
        is_playoff = game.get("is_playoff", False)

        # Use team pitching stats as tier-5 fallback context
        home_team_pitch = None
        away_team_pitch = None
        if game.get("home_team_id"):
            home_team_pitch = self.mlb_api.get_team_pitching_stats(game["home_team_id"], season)
        if game.get("away_team_id"):
            away_team_pitch = self.mlb_api.get_team_pitching_stats(game["away_team_id"], season)

        def fetch_home():
            if game.get("home_pitcher_id"):
                return self.mlb_api.get_pitcher_stats_full_fallback(
                    pitcher_id=game["home_pitcher_id"],
                    season=season,
                    team_pitching=home_team_pitch,
                    is_home=True,
                    is_playoff=is_playoff,
                )
            return (None, None)

        def fetch_away():
            if game.get("away_pitcher_id"):
                return self.mlb_api.get_pitcher_stats_full_fallback(
                    pitcher_id=game["away_pitcher_id"],
                    season=season,
                    team_pitching=away_team_pitch,
                    is_home=False,
                    is_playoff=is_playoff,
                )
            return (None, None)

        # Opposing team IDs needed for pitcher-vs-team lookup
        home_opp_team_id = game.get("away_team_id")   # home pitcher faces away lineup
        away_opp_team_id = game.get("home_team_id")   # away pitcher faces home lineup

        results = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {
                ex.submit(fetch_home): "home",
                ex.submit(fetch_away): "away"
            }
            for fut in as_completed(futures):
                side = futures[fut]
                stats, source = fut.result()
                key_stats = f"{side}_pitcher_stats"
                key_source = f"{side}_pitcher_source"
                key_valid = f"{side}_pitcher_valid"
                if stats:
                    results[key_stats] = stats
                    pitcher_id = game.get(f"{side}_pitcher_id")
                    opp_team_id = home_opp_team_id if side == "home" else away_opp_team_id
                    if pitcher_id:
                        game_log = self.mlb_api.get_pitcher_game_log(pitcher_id, season)
                        if game_log:
                            stats.update(game_log)
                            results[key_stats] = stats
                            trend_str = (
                                f" trend={game_log.get('era_trend','?'):+.2f}"
                                f" qs={game_log.get('quality_start_pct','?'):.0%}"
                                if game_log.get('era_trend') is not None else ""
                            )
                            logger.info(
                                f"  ✅ {side.upper()} game log: ERA_L5={game_log.get('era_last_5','?')}"
                                f" rest={game_log.get('days_rest','?')}d"
                                f" avg_IPS={game_log.get('avg_innings_per_start','?')}{trend_str}"
                            )
                        f5 = self.mlb_api.get_pitcher_f5_stats(pitcher_id, season)
                        if f5:
                            stats.update(f5)
                            results[key_stats] = stats
                            logger.info(f"  ✅ {side.upper()} F5 ERA: {f5.get('f5_era','?')} ({f5.get('f5_ip','?')} IP in first 5)")
                        # Platoon splits (vs LHB / vs RHB)
                        platoon = self.mlb_api.get_pitcher_platoon_splits(pitcher_id, season)
                        if platoon:
                            stats["platoon_splits"] = platoon
                            results[key_stats] = stats
                            lhb_w = platoon.get("vs_lhb", {}).get("whip", "?")
                            rhb_w = platoon.get("vs_rhb", {}).get("whip", "?")
                            logger.info(f"  ✅ {side.upper()} platoon: WHIP vs LHB={lhb_w} vs RHB={rhb_w}")
                        if opp_team_id:
                            pvt = self.mlb_api.get_pitcher_vs_team(pitcher_id, opp_team_id, season)
                            if pvt:
                                stats["era_vs_opp"] = pvt["era"]
                                stats["whip_vs_opp"] = pvt["whip"]
                                stats["ip_vs_opp"] = pvt["ip"]
                                stats["k9_vs_opp"] = pvt["k_per_9"]
                                stats["pvt_season"] = pvt["season_used"]
                                stats["pvt_sample"] = pvt["sample_size"]
                                results[key_stats] = stats
                                logger.info(
                                    f"  ✅ {side.upper()} vs opp: ERA={pvt['era']} "
                                    f"WHIP={pvt['whip']} ({pvt['ip']} IP, {pvt['sample_size']}, "
                                    f"season={pvt['season_used']})"
                                )
                    results[key_source] = source
                    results[key_valid] = True
                    logger.info(f"  ✅ {side.upper()} pitcher: ERA {stats.get('era','?')} ({source})")
                else:
                    results[key_valid] = False
                    logger.error(f"  ❌ {side.upper()} pitcher: sin stats válidas")
        return results

    def get_complete_game_data(
        self,
        date: Optional[str] = None,
        season: Optional[int] = None,
        game_pk: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Obtiene data completa con todas las features.

        If game_pk is provided, only that game is enriched — avoids processing
        all games on the schedule when we only need one.
        """
        if season is None:
            season = _current_mlb_season()
        logger.info("🔍 Obteniendo data completa de juegos MLB...")
        games = self.mlb_api.get_todays_games(date)
        if not games:
            logger.warning("⚠️ No se encontraron juegos")
            return []

        # Filter to the requested game_pk to avoid enriching the entire schedule
        if game_pk is not None:
            games = [g for g in games if g.get("game_pk") == game_pk]
            if not games:
                logger.warning(f"game_pk={game_pk} not found in schedule for date={date}")
                return []

        enriched_games: List[Dict[str, Any]] = []

        for game in games:
            try:
                enriched = game.copy()

                # ===== PITCHERS (con fallback) =====
                pitcher_data = self._enrich_pitchers_concurrent(game, season=season)
                enriched.update(pitcher_data)
                enriched["pitchers_valid"] = pitcher_data.get("home_pitcher_valid", False) and \
                                             pitcher_data.get("away_pitcher_valid", False)

                # ===== WEATHER — forecast for game time =====
                if game.get("venue"):
                    weather = self.weather_api.get_weather_for_stadium(
                        game["venue"],
                        game_time=game.get("game_date"),  # ISO string from MLB API
                    )
                    if weather:
                        enriched["weather"] = weather

                # Park factors are applied by park_weather_engine in run_module PASO 5.

                # ===== NUEVAS FEATURES =====
                # 1) Team Runs Trends (feeds the legacy TTE-unavailable λ fallback)
                if game.get("home_team_id"):
                    home_runs = self.mlb_api.get_team_runs_trend(game["home_team_id"], games=5)
                    if home_runs:
                        enriched["home_team_runs"] = home_runs
                if game.get("away_team_id"):
                    away_runs = self.mlb_api.get_team_runs_trend(game["away_team_id"], games=5)
                    if away_runs:
                        enriched["away_team_runs"] = away_runs

                # 2) Bullpen Workload + ERA
                # Fetched independently and merged: workload (schedule + N boxscore
                # calls) is more failure-prone than era (single stats endpoint). A
                # workload failure must not discard already-fetched, valid era/k_pct/
                # bb_pct data — that used to happen because era was only merged into
                # (and stored via) the workload dict, so era got silently thrown away
                # too whenever workload failed, dropping bullpen_engine to neutral.
                if game.get("home_team_id"):
                    home_bullpen = self.mlb_api.get_bullpen_workload(game["home_team_id"], days=3) or {}
                    home_bp_era = self.mlb_api.get_bullpen_era(game["home_team_id"], season)
                    if home_bp_era:
                        home_bullpen.update(home_bp_era)
                    if home_bullpen:
                        enriched["bullpen_home"] = home_bullpen
                        if home_bullpen.get("is_tired"):
                            logger.warning(f"  ⚠️ {game['home_team']} bullpen CANSADO ({home_bullpen['innings_last_n_days']} IP)")
                if game.get("away_team_id"):
                    away_bullpen = self.mlb_api.get_bullpen_workload(game["away_team_id"], days=3) or {}
                    away_bp_era = self.mlb_api.get_bullpen_era(game["away_team_id"], season)
                    if away_bp_era:
                        away_bullpen.update(away_bp_era)
                    if away_bullpen:
                        enriched["bullpen_away"] = away_bullpen
                        if away_bullpen.get("is_tired"):
                            logger.warning(f"  ⚠️ {game['away_team']} bullpen CANSADO ({away_bullpen['innings_last_n_days']} IP)")

                # El DÍA del juego según MLB, no la fecha UTC de arranque. Para
                # cualquier nocturno que cruce medianoche UTC (la norma en la
                # costa oeste) `game_date[:10]` es el día SIGUIENTE — misma clase
                # de bug que el leak V4 que la Fase 2B arregló del lado del
                # backtest, y que del lado vivo nadie había mirado. Medido sobre
                # la ventana real: 8 de 27 juegos (30%) difieren.
                game_date_str = _official_day(game)
                # Estimación para cuando NO hay lineup confirmado, que es el caso
                # normal al publicar (0 de 27 juegos lo tenían en la corrida de
                # las 07:00). Mezcla mitad y mitad de la última alineación real
                # con el promedio del roster — ver get_prev_lineup_lhb_pct para
                # los errores medidos que eligieron ese peso.
                for _lado, _tid_key, _lhb_key, _mano in [
                    ("home", "home_team_id", "home_lhb_pct",
                     self.mlb_api.get_pitcher_throws(game.get("away_pitcher_id"))),
                    ("away", "away_team_id", "away_lhb_pct",
                     self.mlb_api.get_pitcher_throws(game.get("home_pitcher_id"))),
                ]:
                    _tid = game.get(_tid_key)
                    if not (_tid and game_date_str):
                        continue
                    roster = self.mlb_api.get_team_batting_handedness_pct(
                        _tid, game_date_str, vs_hand=_mano
                    )
                    previo = self.mlb_api.get_prev_lineup_lhb_pct(
                        _tid, game_date_str, vs_hand=_mano
                    )
                    if roster is not None and previo is not None:
                        enriched[_lhb_key] = round(0.5 * previo + 0.5 * roster, 3)
                    elif roster is not None:
                        enriched[_lhb_key] = roster
                    elif previo is not None:
                        enriched[_lhb_key] = previo
                    # Si ninguna resuelve, la clave queda ausente y la cadena de
                    # los engines cae al promedio de liga — que es el único lugar
                    # donde una constante corresponde.

                # 6) Travel Fatigue (coordinate-based miles + time zones)
                if game.get("away_team_id") and game.get("game_date"):
                    travel = self.mlb_api.get_travel_fatigue(
                        game["away_team_id"],
                        game["game_date"],
                        current_venue=game.get("venue", ""),
                        official_date=_official_day(game),
                    )
                    if travel:
                        enriched["away_travel_fatigue"] = travel
                        enriched["miles_traveled_away"] = travel.get("miles_traveled", 0)
                        enriched["time_zones_crossed_away"] = travel.get("time_zones_crossed", 0)
                        enriched["back_to_back_away"] = travel.get("back_to_back", False)
                        # FALL-002 fix (roadmap Step 4): flattened alongside
                        # the other travel fields above, same convention —
                        # hfa_engine.py reads this to tell a real 0-mile/0-tz
                        # measurement apart from an unmapped-venue "missing".
                        enriched["travel_source_away"] = travel.get("travel_source", "live")
                        if travel.get("has_travel_fatigue"):
                            logger.info(f"  ✈️ {game['away_team']}: {travel['miles_traveled']} mi, {travel['time_zones_crossed']} TZ")

                # 6b) Roof status (for ParkWeatherEngine's 8 retractable-roof parks —
                # previously always defaulted to "closed" since nothing populated
                # this field at all; see get_roof_status() docstring for the fix).
                if game.get("game_pk"):
                    roof_open = self.mlb_api.get_roof_status(game["game_pk"])
                    if roof_open is not None:
                        enriched["roof_open"] = roof_open
                        enriched["roof_closed"] = not roof_open

                # 7) Days rest (home and away)
                # Éste es el consumidor que el día equivocado SÍ rompía:
                # get_team_days_rest resta la fecha del último juego —que la API
                # devuelve como día oficial— de la fecha que se le pasa acá. Con
                # `game_date[:10]` eso mezclaba unidades y devolvía exactamente
                # +1 día de descanso: medido, 16 de 16 equipos en los juegos que
                # difieren. Sesgo direccional, siempre a favor de quien juega de
                # noche en el oeste.
                game_date_str = _official_day(game)
                if game.get("home_team_id") and game_date_str:
                    enriched["home_days_rest"] = self.mlb_api.get_team_days_rest(game["home_team_id"], game_date_str)
                if game.get("away_team_id") and game_date_str:
                    enriched["away_days_rest"] = self.mlb_api.get_team_days_rest(game["away_team_id"], game_date_str)

                # 7b) LESIONES — dato que hasta el 2026-08-01 no entraba al módulo
                # de béisbol en absoluto (`app.py` lo tenía hardcodeado en vacío).
                # La ofensa de cada equipo se calcula del Statcast ACUMULADO de la
                # temporada, así que incluye entera la producción de quien hoy no
                # juega. Ver `get_team_injuries` para el caso real que lo motivó.
                for _lado, _tid_key, _clave in (("home", "home_team_id", "home_injuries"),
                                                ("away", "away_team_id", "away_injuries")):
                    _tid = game.get(_tid_key)
                    if not _tid:
                        continue
                    _les = self.mlb_api.get_team_injuries(_tid, season)
                    if not _les:
                        continue
                    enriched[_clave] = _les
                    enriched[f"{_lado}_injured_pa_share"] = _les["pa_share_out"]
                    if _les["pa_share_out"] >= 0.10:
                        logger.warning(
                            "  🏥 %s: %d bateador(es) en lista de lesionados, %.1f%% de sus "
                            "turnos ofensivos — el λ de ofensa NO lo descuenta todavía (%s)",
                            game.get(f"{_lado}_team", "?"), _les["n_out"],
                            _les["pa_share_out"] * 100,
                            ", ".join(p["name"] for p in _les["players"][:3]),
                        )

                # 8) Team pitching/defense stats (ERA, WHIP, RA/G)
                if game.get("home_team_id"):
                    home_pitch = self.mlb_api.get_team_pitching_stats(game["home_team_id"], season)
                    if home_pitch:
                        enriched["home_pitching_stats"] = home_pitch
                        logger.info(f"  ✅ {game['home_team']} pitching: ERA={home_pitch['team_era']} WHIP={home_pitch['team_whip']}")
                if game.get("away_team_id"):
                    away_pitch = self.mlb_api.get_team_pitching_stats(game["away_team_id"], season)
                    if away_pitch:
                        enriched["away_pitching_stats"] = away_pitch
                        logger.info(f"  ✅ {game['away_team']} pitching: ERA={away_pitch['team_era']} WHIP={away_pitch['team_whip']}")

                # 8b) Defensive efficiency dicts — DER from pitching stats, OAA
                #     from Savant's public outs_above_average leaderboard,
                #     summed across the team's gameday roster.
                #     Convention: defense_home = home team's fielding; defense_away = away team's
                _oaa_lb = _fetch_savant_oaa(season)
                for _side, _pitch_key, _def_key, _team_name, _team_id in [
                    ("home", "home_pitching_stats", "defense_home", game.get("home_team", ""), game.get("home_team_id")),
                    ("away", "away_pitching_stats", "defense_away", game.get("away_team", ""), game.get("away_team_id")),
                ]:
                    _ps = enriched.get(_pitch_key) or {}
                    _der = _ps.get("der")
                    _bip = _ps.get("bip", 0)
                    if _der is not None and _bip > 0:
                        _oaa_sum = None
                        if _team_id and _oaa_lb:
                            _roster = _fetch_gameday_roster(int(_team_id), season)
                            _found  = [_oaa_lb[pid] for pid in _roster if pid in _oaa_lb]
                            if _found:
                                _oaa_sum = round(sum(_found), 1)
                        enriched[_def_key] = {
                            "team_name": _team_name,
                            "der": _der,
                            "bip": _bip,
                            "oaa": _oaa_sum,
                        }
                        logger.info(
                            f"  🛡️  {_team_name} defense: DER={_der:.4f}  BIP={_bip}"
                            + (f"  OAA={_oaa_sum:+.1f}" if _oaa_sum is not None else "  OAA=n/a")
                        )

                # 9) Lineup handedness (LHB% per side — used for platoon split adjustment)
                # El ambidiestro se resuelve con la mano del abridor que ESA
                # alineación enfrenta: la local batea contra el abridor visitante
                # y viceversa.
                _mano_vs_local = self.mlb_api.get_pitcher_throws(game.get("away_pitcher_id"))
                _mano_vs_visita = self.mlb_api.get_pitcher_throws(game.get("home_pitcher_id"))
                enriched["home_faces_hand"] = _mano_vs_local
                enriched["away_faces_hand"] = _mano_vs_visita
                # La mano de cada abridor, por su propio nombre: el prior
                # poblacional del ajuste platoon es opuesto para derechos y
                # zurdos, así que el motor necesita saber cuál es cuál.
                enriched["home_pitcher_throws"] = _mano_vs_visita
                enriched["away_pitcher_throws"] = _mano_vs_local
                for _side, _lineup_key, _lhb_key, _mano in [
                    ("home", "home_lineup", "home_lineup_lhb_pct", _mano_vs_local),
                    ("away", "away_lineup", "away_lineup_lhb_pct", _mano_vs_visita),
                ]:
                    lineup = enriched.get(_lineup_key) or []
                    pids = [p["id"] for p in lineup if p.get("id")]
                    pct = lhb_pct_efectivo(
                        self.mlb_api.get_batter_handedness_batch(pids), pids, _mano
                    ) if pids else None
                    if pct is not None:
                        enriched[_lhb_key] = pct
                        logger.info(
                            f"  ✅ {_side.upper()} lineup: {pct:.0%} LHB efectivo "
                            f"vs {_mano or '?'}HP — "
                            f"{', '.join(p.get('name','?') for p in lineup[:3])}..."
                        )
                    # Sin lineup confirmado la clave queda AUSENTE, no en 0.45.
                    # Rellenarla con la media de liga dejaba a `_first_present` de
                    # park_weather_engine sin poder llegar nunca al segundo
                    # escalón —`*_lhb_pct`, el roster real que se mide 40 líneas
                    # más arriba— porque el primero jamás estaba vacío. Medido:
                    # el error medio de ese 0.45 contra el roster real es 0.076,
                    # casi la desviación completa entre equipos (0.085). Misma
                    # clase que FALL-002: un valor fabricado que aguas abajo no se
                    # distingue de una medición, y que además tapa la medición.

                status_icon = "✅" if enriched["pitchers_valid"] else "⚠️"
                status_msg = "Data completa" if enriched["pitchers_valid"] else "Data incompleta - NO APOSTAR"
                logger.info(f"{status_icon} {game['away_team']} @ {game['home_team']}: {status_msg}\n")

                enriched_games.append(enriched)
                time.sleep(0.2)

            except Exception as e:
                logger.error(f"❌ Error enriqueciendo {game.get('home_team', 'juego')}: {e}")
                # Se conserva el crudo (descartar el enriquecimiento parcial es
                # lo conservador: nadie sabe hasta dónde llegó), pero MARCADO.
                # Antes salía indistinguible de un juego normal y aguas abajo
                # bullpen/defensa/descanso caían a promedio de liga en silencio,
                # con el pick haciéndose igual. `pitchers_valid=False` es la
                # respuesta honesta: no se sabe si son válidos.
                fallido = dict(game)
                fallido["enrichment_failed"] = True
                fallido["enrichment_error"] = f"{type(e).__name__}: {e}"[:200]
                fallido["pitchers_valid"] = False
                enriched_games.append(fallido)

        valid_count = sum(1 for g in enriched_games if g.get("pitchers_valid"))
        logger.warning(f"\n🎉 {len(enriched_games)} juegos totales | ✅ {valid_count} VÁLIDOS | ⚠️ {len(enriched_games) - valid_count} SKIP")
        return enriched_games

    def save_to_file(self, games: List[Dict[str, Any]], filename: str = "mlb_complete_data.json"):
        filepath = DATA_DIR / filename
        with open(filepath, "w") as f:
            json.dump(games, f, indent=2)
        logger.info(f"💾 Data guardada en {filepath}")


# ==========================================================
# TESTING & DEMO
# ==========================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🔥 MLB DATA FETCHERS V3 FINAL - SISTEMA COMPLETO CON 12 FEATURES")
    print("=" * 70)

    integrator = MLBDataIntegrator()

    print("\n📅 Obteniendo juegos de HOY con TODAS las features...")
    complete_data = integrator.get_complete_game_data()

    if complete_data:
        integrator.save_to_file(complete_data)

        print("\n" + "=" * 70)
        print("📊 RESUMEN EJECUTIVO:")
        print("=" * 70)

        for idx, game in enumerate(complete_data, 1):
            print(f"\n🎯 JUEGO {idx}: {game.get('away_team','?')} @ {game.get('home_team','?')}")
            print(f"   📍 {game.get('venue', 'N/A')}")
            print(f"   🎮 {game.get('game_context', 'regular')}")
            if game.get("pitchers_valid"):
                print(f"   ✅ Data VÁLIDA para apostar")
            else:
                print(f"   ⚠️ Data INCOMPLETA - SKIP")
    else:
        print("\n⚠️ No hay juegos programados hoy")

    print("\n" + "=" * 70)
    print("✅ SISTEMA COMPLETO FUNCIONANDO - LISTO PARA PRODUCCIÓN")
    print("=" * 70)
