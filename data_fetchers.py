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

load_dotenv()

logger = logging.getLogger(__name__)


def _current_mlb_season() -> int:
    """Return the current MLB season year. Season starts in March."""
    now = datetime.now()
    return now.year if now.month >= 3 else now.year - 1


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
            "hydrate": "probablePitcher,lineups,officials,team,seriesStatus"
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

            # Home plate umpire (populated for in-progress / Final games)
            hp_umpire_id = None
            hp_umpire_name = None
            for off in (game.get("officials") or []):
                if off.get("officialType") == "Home Plate":
                    hp_umpire_id   = (off.get("official") or {}).get("id")
                    hp_umpire_name = (off.get("official") or {}).get("fullName")
                    break

            return {
                "game_pk": game.get("gamePk"),
                "game_date": game.get("gameDate"),
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_pitcher": home_pitcher,
                "home_pitcher_id": home_pitcher_id,
                "away_pitcher": away_pitcher,
                "away_pitcher_id": away_pitcher_id,
                "venue": (game.get("venue") or {}).get("name"),
                "game_type": game_type,
                "game_context": game_context,
                "is_playoff": is_playoff,
                "series_info": series_info,
                "status": (game.get("status") or {}).get("detailedState"),
                "home_lineup": home_lineup,
                "away_lineup": away_lineup,
                "hp_umpire_id": hp_umpire_id,
                "hp_umpire_name": hp_umpire_name,
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
            innings = float(stat.get("inningsPitched", 0.0))
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
            innings = float(stat.get("inningsPitched", 0.0))
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

        if is_home and "home" in all_stats:
            logger.info("  ✅ Usando HOME split")
            return all_stats["home"], "home_split"
        if (not is_home) and "away" in all_stats:
            logger.info("  ✅ Usando AWAY split")
            return all_stats["away"], "away_split"

        if "regular_overall" in all_stats:
            logger.info("  ✅ Usando OVERALL (regular season)")
            return all_stats["regular_overall"], "regular_overall"

        logger.error("  ❌ Sin stats válidas")
        return None, None

    def get_pitcher_game_log(self, pitcher_id: int, season: int, last_n: int = 5):
        cache_file = CACHE_DIR / f"pitcher_log_{pitcher_id}_{season}_{last_n}.json"
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
            starts = [s for s in splits if int(s.get("stat", {}).get("gamesStarted", 0)) > 0]
            if not starts:
                starts = splits
            starts.sort(key=lambda x: x.get("date", ""), reverse=True)
            recent = starts[:last_n]
            if not recent:
                return None
            total_er = sum(int(s.get("stat", {}).get("earnedRuns", 0)) for s in recent)
            total_ip = sum(float(s.get("stat", {}).get("inningsPitched", 0)) for s in recent)
            era_last_n = round((total_er / total_ip * 9), 2) if total_ip > 0 else 4.50
            avg_ips_recent = round(total_ip / len(recent), 2) if recent else None
            last_start = recent[0]
            last_date_str = last_start.get("date", "")
            last_pitch_count = int(last_start.get("stat", {}).get("numberOfPitches", 90))
            days_rest = 4
            if last_date_str:
                try:
                    last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
                    days_rest = (datetime.utcnow() - last_date).days
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
                _ip = float(_st.get("inningsPitched", 0))
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
                    f5_ip += float(stat.get("inningsPitched", 0.0))

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
        """
        from config import LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP

        # ── tier 1: current MLB season ────────────────────────────────────────
        mlb_stats, source = self.get_pitcher_stats_with_fallback(
            pitcher_id, season, is_playoff=is_playoff, is_home=is_home
        )
        if mlb_stats and mlb_stats.get("innings_pitched", 0) >= 5.0:
            return mlb_stats, source or "mlb_current"

        # partial current-season stats (< 5 IP) — keep as candidate, but look further
        partial = mlb_stats

        # ── tier 2: previous MLB season ──────────────────────────────────────
        prev_stats = self.get_pitcher_stats(pitcher_id, season=season - 1)
        if prev_stats and prev_stats.get("innings_pitched", 0) >= 20.0:
            prev_stats["is_fallback"] = True
            prev_stats["fallback_tier"] = "mlb_prev_season"
            logger.info(f"  ⚾ Pitcher {pitcher_id}: no current-season data → using {season-1} MLB season")
            return prev_stats, "mlb_prev_season"

        # ── tier 3: Triple-A current season ──────────────────────────────────
        aaa = self._fetch_milb_stats(pitcher_id, season, sport_id=11)
        if aaa and aaa.get("innings_pitched", 0) >= 10.0:
            aaa["is_fallback"] = True
            aaa["fallback_tier"] = "aaa_current"
            logger.info(f"  ⚾ Pitcher {pitcher_id}: no MLB data → using AAA {season}")
            return aaa, "aaa_current"

        # ── tier 4: Double-A current season ───────────────────────────────────
        aa = self._fetch_milb_stats(pitcher_id, season, sport_id=12)
        if aa and aa.get("innings_pitched", 0) >= 10.0:
            aa["is_fallback"] = True
            aa["fallback_tier"] = "aa_current"
            logger.info(f"  ⚾ Pitcher {pitcher_id}: no MLB/AAA data → using AA {season}")
            return aa, "aa_current"

        # ── partial current-season (< 5 IP) is better than nothing ───────────
        if partial and partial.get("innings_pitched", 0) > 0:
            partial["is_fallback"] = True
            partial["fallback_tier"] = "mlb_current_partial"
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
            "is_fallback": True,
            "fallback_tier": "team_staff_era",
        }
        logger.info(f"  ⚾ Pitcher {pitcher_id}: no stats found at any level → team staff ERA {staff_era:.2f}")
        return staff_stats, "team_staff_era"

    def get_team_recent_form(self, team_id: int, games: int = 10) -> Optional[Dict[str, Any]]:
        """{"wins": int, "losses": int, "win_pct": float, "streak": str, "last_10": "WLWL..."}"""
        cache_file = CACHE_DIR / f"team_form_{team_id}_{games}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 1800:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
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

            results = []
            for date_item in data.get("dates", []):
                for game in date_item.get("games", []):
                    if game.get("status", {}).get("abstractGameState") != "Final":
                        continue
                    home_id = game["teams"]["home"]["team"]["id"]
                    away_id = game["teams"]["away"]["team"]["id"]
                    hs = game["teams"]["home"].get("score", 0)
                    as_ = game["teams"]["away"].get("score", 0)
                    if home_id == team_id:
                        results.append("W" if hs > as_ else "L")
                    elif away_id == team_id:
                        results.append("W" if as_ > hs else "L")

            recent = results[-games:]
            wins = recent.count("W")
            losses = recent.count("L")
            win_pct = round(wins / len(recent), 3) if recent else 0.0
            if recent:
                last = recent[-1]
                c = 1
                for i in range(len(recent) - 2, -1, -1):
                    if recent[i] == last: c += 1
                    else: break
                streak = f"{last}{c}"
            else:
                streak = "N/A"

            result = {"wins": wins, "losses": losses, "games_played": len(recent),
                      "win_pct": win_pct, "streak": streak, "last_10": "".join(recent)}
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo forma reciente: {e}")
            return None

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
        """Convert baseball IP notation '4.2' (4⅔) to decimal 4.667."""
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
        {"innings_last_n_days", "ip_last_3_days", "games_played",
         "is_tired", "avg_innings_per_game", "starter_ip_avg"}
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
        total_start_ip = 0.0
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

            starter_id  = pitchers[0]
            reliever_ids = pitchers[1:]

            sp_data = players.get(f"ID{starter_id}", {})
            sp_ip   = self._ip_to_float(
                sp_data.get("stats", {}).get("pitching", {}).get("inningsPitched", 0)
            )
            bp_ip = sum(
                self._ip_to_float(
                    players.get(f"ID{pid}", {})
                           .get("stats", {}).get("pitching", {})
                           .get("inningsPitched", 0)
                )
                for pid in reliever_ids
            )

            total_start_ip += sp_ip
            total_bp_ip    += bp_ip
            games_counted  += 1

        if games_counted == 0:
            return None

        avg_bp_ip = round(total_bp_ip / games_counted, 2)
        result = {
            "innings_last_n_days":  round(total_bp_ip,    1),
            "ip_last_3_days":       round(total_bp_ip,    1),
            "games_played":         games_counted,
            "is_tired":             total_bp_ip > 12.0,
            "avg_innings_per_game": avg_bp_ip,
            "starter_ip_avg":       round(total_start_ip / games_counted, 2),
        }
        try:
            with open(cache_file, "w") as f:
                json.dump(result, f)
        except Exception:
            pass
        return result

    # ==========================================================
    # FEATURE 4 - H2H HISTÓRICO
    # ==========================================================
    def get_head_to_head(self, team1_id: int, team2_id: int, season: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """{"team1_wins": int, "team2_wins": int, "total_games": int, "avg_total_runs": float, "has_history": bool}"""
        if season is None:
            season = _current_mlb_season()
        cache_key = f"h2h_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}_{season}"
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 7200:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        url = f"{self.BASE_URL}/schedule"
        params = {
            "sportId": 1,
            "teamId": team1_id,
            "opponentId": team2_id,
            "season": season,
            "gameType": "R",
            "hydrate": "team,linescore"
        }
        try:
            r = self.session.get(url, params=params, timeout=(5, 30))
            r.raise_for_status()
            data = r.json()

            t1, t2, total_runs = 0, 0, []
            for date_item in data.get("dates", []):
                for game in date_item.get("games", []):
                    if game.get("status", {}).get("abstractGameState") != "Final":
                        continue
                    home_id = game["teams"]["home"]["team"]["id"]
                    hs = game["teams"]["home"].get("score", 0)
                    as_ = game["teams"]["away"].get("score", 0)
                    if home_id == team1_id:
                        t1 += 1 if hs > as_ else 0
                        t2 += 1 if as_ > hs else 0
                    else:
                        t1 += 1 if as_ > hs else 0
                        t2 += 1 if hs > as_ else 0
                    total_runs.append(hs + as_)

            total_games = t1 + t2
            result = {
                "team1_wins": t1,
                "team2_wins": t2,
                "total_games": total_games,
                "avg_total_runs": round(sum(total_runs)/len(total_runs), 2) if total_runs else 0.0,
                "has_history": total_games >= 5
            }
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo H2H: {e}")
            return None

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
                    ip = float(ip_raw)
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
    # FEATURE 6 - STANDINGS STATUS
    # ==========================================================
    def get_standings_status(self, team_id: int, season: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """{"status": "clinched|eliminated|in_race", "games_back": float, "clinched": bool, "eliminated": bool, "win_pct": float}"""
        if season is None:
            season = _current_mlb_season()
        cache_key = f"standings_{team_id}_{season}"
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 3600:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        url = f"{self.BASE_URL}/standings"
        params = {"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"}
        try:
            r = self.session.get(url, params=params, timeout=(5, 30))
            r.raise_for_status()
            data = r.json()

            for record in data.get("records", []):
                for team_record in record.get("teamRecords", []):
                    if team_record["team"]["id"] == team_id:
                        clinched = bool(team_record.get("clinched", False))
                        eliminated = bool(team_record.get("eliminated", False))
                        # algunos payloads usan wildCardEliminationNumber="E" cuando está eliminado
                        wc_elim = (team_record.get("wildCardEliminationNumber") == "E")
                        try:
                            games_back = float(team_record.get("wildCardGamesBack", 0.0))
                        except (ValueError, TypeError):
                            games_back = 0.0
                        if clinched:
                            status = "clinched"
                        elif eliminated or wc_elim:
                            status = "eliminated"
                        else:
                            status = "in_race"
                        result = {
                            "status": status,
                            "games_back": games_back,
                            "clinched": clinched,
                            "eliminated": eliminated,
                            "win_pct": _safe_float(team_record.get("winningPercentage"))
                        }
                        with open(cache_file, "w") as f:
                            json.dump(result, f)
                        return result
            return None
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo standings: {e}")
            return None

    # ==========================================================
    # FEATURE 6b - TEAM OFFENSIVE STATS (wOBA, OPS)
    # ==========================================================
    def get_team_offensive_stats(self, team_id: int, season: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Fetches wOBA (computed from components) and OPS from the team hitting endpoint."""
        if season is None:
            season = _current_mlb_season()
        cache_file = CACHE_DIR / f"team_offense_{team_id}_{season}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 3600:
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass

        try:
            url = f"{self.BASE_URL}/teams/{team_id}/stats"
            params = {"stats": "season", "season": season, "group": "hitting", "sportId": 1}
            r = self.session.get(url, params=params, timeout=(5, 15))
            r.raise_for_status()
            splits = r.json().get("stats", [{}])[0].get("splits", [{}])
            if not splits:
                return None
            stat = splits[0].get("stat", {})

            games = int(stat.get("gamesPlayed", 0))
            if games < 5:
                return None

            bb = int(stat.get("baseOnBalls", 0))
            ibb = int(stat.get("intentionalWalks", 0))
            hbp = int(stat.get("hitByPitch", 0))
            hits = int(stat.get("hits", 0))
            doubles = int(stat.get("doubles", 0))
            triples = int(stat.get("triples", 0))
            hr = int(stat.get("homeRuns", 0))
            ab = int(stat.get("atBats", 0))
            sf = int(stat.get("sacrificeFlies", 0))

            ubb = bb - ibb
            singles = hits - doubles - triples - hr
            denom = ab + ubb + hbp + sf
            if denom > 0:
                woba = round(
                    (0.690 * ubb + 0.722 * hbp + 0.888 * singles +
                     1.271 * doubles + 1.616 * triples + 2.101 * hr) / denom,
                    3
                )
            else:
                woba = 0.320

            obp = _safe_float(stat.get("obp", 0.0))
            slg = _safe_float(stat.get("slg", 0.0))
            ops = round(obp + slg, 3) if obp > 0 and slg > 0 else _safe_float(stat.get("ops", 0.735))

            # Approximate wRC+ from wOBA (no park adjustment; close enough for lambda scaling)
            # wRC+ = ((wOBA - lgwOBA) / wOBAscale) * 100 + 100
            LG_WOBA, WOBA_SCALE = 0.320, 1.157
            wrc_plus = round(((woba - LG_WOBA) / WOBA_SCALE) * 100 + 100, 1)
            wrc_plus = max(50.0, min(165.0, wrc_plus))

            result = {"woba": woba, "ops": ops, "obp": round(obp, 3), "slg": round(slg, 3),
                      "wrc_plus": wrc_plus, "games": games}
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo offensive stats team {team_id}: {e}")
            return None

    # ==========================================================
    # FEATURE 7 - TRAVEL FATIGUE (coordinate-based)
    # ==========================================================
    def get_travel_fatigue(self, team_id: int, game_date: str, current_venue: str = "") -> Optional[Dict[str, Any]]:
        """Returns travel metrics: miles_traveled, time_zones_crossed, back_to_back, has_travel_fatigue."""
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

        start_search = game_dt - timedelta(days=3)
        url = f"{self.BASE_URL}/schedule"
        params = {
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_search.strftime("%Y-%m-%d"),
            "endDate": game_dt.strftime("%Y-%m-%d"),
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
                        previous_venue = (game.get("venue") or {}).get("name")
                        previous_game_date = gd

            if not previous_venue or not previous_game_date:
                return {"has_travel_fatigue": False, "miles_traveled": 0, "time_zones_crossed": 0, "back_to_back": False}

            prev_dt = datetime.fromisoformat(previous_game_date.replace('Z', '+00:00'))
            hours_between = (game_dt - prev_dt).total_seconds() / 3600
            back_to_back = hours_between < 30

            # Coordinate-based distance and time-zone calculation
            coords = WeatherAPI.STADIUM_COORDS
            miles = 0
            time_zones = 0
            if previous_venue in coords and current_venue in coords:
                prev_c = coords[previous_venue]
                curr_c = coords[current_venue]
                miles = round(_haversine_miles(prev_c["lat"], prev_c["lon"], curr_c["lat"], curr_c["lon"]))
                tz_prev = _lon_to_tz_offset(prev_c["lon"])
                tz_curr = _lon_to_tz_offset(curr_c["lon"])
                time_zones = abs(tz_curr - tz_prev)
            elif previous_venue != current_venue:
                # Fallback: unknown stadium → assume mid-range travel
                miles = 1000
                time_zones = 1

            has_fatigue = (miles > 1000 or time_zones >= 2) and hours_between < 30

            result = {
                "has_travel_fatigue": bool(has_fatigue),
                "hours_since_last_game": round(hours_between, 1),
                "previous_venue": previous_venue,
                "miles_traveled": miles,
                "time_zones_crossed": time_zones,
                "back_to_back": back_to_back,
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

            # DER = 1 − BABIP_allowed (fielding-pure metric)
            # BIP = AB − K − HR + SF
            hits = int(stat.get("hits", 0))
            ab   = int(stat.get("atBats", 0))
            so   = int(stat.get("strikeOuts", 0))
            hr   = int(stat.get("homeRuns", 0))
            sf   = int(stat.get("sacFlies", 0))
            bip  = ab - so - hr + sf
            if bip > 0:
                babip_allowed = (hits - hr) / bip
                der = round(1.0 - babip_allowed, 4)
            else:
                der = 0.715   # league average fallback
                bip = 0

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
                ip = float(ip_str)
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
                ip = float(ip_raw)
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
                                json.dump({"s": side}, f)
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"⚠️ batch handedness fetch failed: {e}")
                for pid in uncached:
                    result.setdefault(pid, "R")

        return result

    # ==========================================================
    # FEATURE: HOME PLATE UMPIRE ZONE STATS
    # ==========================================================
    def get_umpire_historical_stats(
        self, umpire_id: int, lookback_days: int = 21
    ) -> Optional[Dict[str, Any]]:
        """
        Build a zone-tendency profile for an umpire from recent completed games.

        Scans lookback_days of the schedule for games where umpire_id was the
        HP umpire, then aggregates ball/strike counts from each boxscore.

        Returns:
          {games_worked, strike_pct, k_rate, rpg, zone_factor}

        zone_factor: 1.0 = average zone.
          < 1.0  = pitcher-friendly (high strike%, depresses scoring).
          > 1.0  = hitter-friendly (low strike%, inflates scoring).
        Capped at ±4%.  Cached 3 days.
        """
        cache_file = CACHE_DIR / f"umpire_{umpire_id}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400 * 3:
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass

        LEAGUE_STRIKE_PCT = 0.635

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)

        # One schedule request for the whole range with officials hydrated
        try:
            r = self.session.get(
                f"{self.BASE_URL}/schedule",
                params={
                    "sportId": 1,
                    "startDate": start_date.strftime("%Y-%m-%d"),
                    "endDate":   end_date.strftime("%Y-%m-%d"),
                    "hydrate":   "officials",
                },
                timeout=(5, 30),
            )
            r.raise_for_status()
            sched_data = r.json()
        except Exception as e:
            logger.warning(f"⚠️ umpire schedule scan failed: {e}")
            return None

        # Identify game_pks where this ump was home plate ump
        hp_game_pks: List[int] = []
        for date_item in sched_data.get("dates", []):
            for game in date_item.get("games", []):
                if (game.get("status") or {}).get("abstractGameState") != "Final":
                    continue
                for off in (game.get("officials") or []):
                    if (off.get("officialType") == "Home Plate" and
                            (off.get("official") or {}).get("id") == umpire_id):
                        pk = game.get("gamePk")
                        if pk:
                            hp_game_pks.append(pk)
                        break

        if not hp_game_pks:
            result = {
                "games_worked": 0,
                "zone_factor": 1.0,
                "strike_pct": LEAGUE_STRIKE_PCT,
            }
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result

        # Aggregate ball/strike stats from each boxscore (cap at 15 games)
        total_strikes = 0
        total_pitches = 0
        total_k = 0
        total_bf = 0
        total_runs = 0
        games_counted = 0

        for game_pk in hp_game_pks[:15]:
            try:
                r2 = self.session.get(
                    f"{self.BASE_URL}/game/{game_pk}/boxscore",
                    timeout=(5, 15),
                )
                r2.raise_for_status()
                bs = r2.json()
                for side in ("home", "away"):
                    ps = (
                        bs.get("teams", {})
                          .get(side, {})
                          .get("teamStats", {})
                          .get("pitching", {})
                    )
                    s = int(ps.get("strikes", 0))
                    b = int(ps.get("balls",   0))
                    total_strikes += s
                    total_pitches += s + b
                    total_k    += int(ps.get("strikeOuts",   0))
                    total_bf   += int(ps.get("battersFaced", 0))
                    total_runs += int(ps.get("runs",         0))
                games_counted += 1
            except Exception:
                continue
            time.sleep(0.1)

        if games_counted == 0 or total_pitches == 0:
            result = {
                "games_worked": 0,
                "zone_factor": 1.0,
                "strike_pct": LEAGUE_STRIKE_PCT,
            }
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result

        strike_pct = total_strikes / total_pitches
        k_rate     = total_k / total_bf if total_bf > 0 else 0.225
        rpg        = total_runs / games_counted

        # Pitcher-friendly (high strike%) → lower scoring → zone_factor < 1.0
        zone_factor = round(1.0 - (strike_pct - LEAGUE_STRIKE_PCT) * 0.60, 4)
        zone_factor = max(0.96, min(zone_factor, 1.04))

        result = {
            "games_worked": games_counted,
            "strike_pct":   round(strike_pct, 4),
            "k_rate":       round(k_rate, 4),
            "rpg":          round(rpg, 2),
            "zone_factor":  zone_factor,
        }
        with open(cache_file, "w") as f:
            json.dump(result, f)
        return result


# ==========================================================
# 2) WEATHER API - OPENWEATHER (GRATIS)
# ==========================================================

class WeatherAPI:
    BASE_URL = "https://api.openweathermap.org/data/2.5"

    STADIUM_COORDS = {
        "Yankee Stadium": {"lat": 40.8296, "lon": -73.9262, "city": "New York"},
        "Fenway Park": {"lat": 42.3467, "lon": -71.0972, "city": "Boston"},
        "Dodger Stadium": {"lat": 34.0739, "lon": -118.2400, "city": "Los Angeles"},
        "Wrigley Field": {"lat": 41.9484, "lon": -87.6553, "city": "Chicago"},
        "Oracle Park": {"lat": 37.7786, "lon": -122.3893, "city": "San Francisco"},
        "Coors Field": {"lat": 39.7559, "lon": -104.9942, "city": "Denver"},
        "Petco Park": {"lat": 32.7073, "lon": -117.1566, "city": "San Diego"},
        "Rogers Centre": {"lat": 43.6414, "lon": -79.3894, "city": "Toronto"},
        "Rogers Center": {"lat": 43.6414, "lon": -79.3894, "city": "Toronto"},
        "T-Mobile Park": {"lat": 47.5914, "lon": -122.3325, "city": "Seattle"},
        "Minute Maid Park": {"lat": 29.7573, "lon": -95.3555, "city": "Houston"},
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
        "Guaranteed Rate Field": {"lat": 41.8300, "lon": -87.6338, "city": "Chicago"},
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

                # Park factors are applied by park_weather_engine in run_module PASO 2.

                # ===== NUEVAS FEATURES =====
                # 1) Team Recent Form
                if game.get("home_team_id"):
                    home_form = self.mlb_api.get_team_recent_form(game["home_team_id"], games=10)
                    if home_form:
                        enriched["home_team_form"] = home_form
                        logger.info(f"  ✅ {game['home_team']} form: {home_form['wins']}-{home_form['losses']} (L10)")
                if game.get("away_team_id"):
                    away_form = self.mlb_api.get_team_recent_form(game["away_team_id"], games=10)
                    if away_form:
                        enriched["away_team_form"] = away_form
                        logger.info(f"  ✅ {game['away_team']} form: {away_form['wins']}-{away_form['losses']} (L10)")

                # 2) Team Runs Trends
                if game.get("home_team_id"):
                    home_runs = self.mlb_api.get_team_runs_trend(game["home_team_id"], games=5)
                    if home_runs:
                        enriched["home_team_runs"] = home_runs
                if game.get("away_team_id"):
                    away_runs = self.mlb_api.get_team_runs_trend(game["away_team_id"], games=5)
                    if away_runs:
                        enriched["away_team_runs"] = away_runs

                # 3) Bullpen Workload + ERA
                if game.get("home_team_id"):
                    home_bullpen = self.mlb_api.get_bullpen_workload(game["home_team_id"], days=3)
                    if home_bullpen:
                        home_bp_era = self.mlb_api.get_bullpen_era(game["home_team_id"], season)
                        if home_bp_era:
                            home_bullpen.update(home_bp_era)
                        enriched["bullpen_home"] = home_bullpen
                        if home_bullpen.get("is_tired"):
                            logger.warning(f"  ⚠️ {game['home_team']} bullpen CANSADO ({home_bullpen['innings_last_n_days']} IP)")
                if game.get("away_team_id"):
                    away_bullpen = self.mlb_api.get_bullpen_workload(game["away_team_id"], days=3)
                    if away_bullpen:
                        away_bp_era = self.mlb_api.get_bullpen_era(game["away_team_id"], season)
                        if away_bp_era:
                            away_bullpen.update(away_bp_era)
                        enriched["bullpen_away"] = away_bullpen
                        if away_bullpen.get("is_tired"):
                            logger.warning(f"  ⚠️ {game['away_team']} bullpen CANSADO ({away_bullpen['innings_last_n_days']} IP)")

                # 4) H2H Histórico
                if game.get("home_team_id") and game.get("away_team_id"):
                    h2h = self.mlb_api.get_head_to_head(game["home_team_id"], game["away_team_id"], season)
                    if h2h and h2h.get("has_history"):
                        enriched["head_to_head"] = h2h
                        logger.info(f"  ✅ H2H: {game['home_team']} {h2h['team1_wins']}-{h2h['team2_wins']} {game['away_team']}")

                # 5) Standings Status
                if game.get("home_team_id"):
                    home_standings = self.mlb_api.get_standings_status(game["home_team_id"], season)
                    if home_standings:
                        enriched["home_standings"] = home_standings
                        if home_standings["status"] in ["clinched", "eliminated"]:
                            logger.info(f"  📊 {game['home_team']}: {home_standings['status'].upper()}")
                if game.get("away_team_id"):
                    away_standings = self.mlb_api.get_standings_status(game["away_team_id"], season)
                    if away_standings:
                        enriched["away_standings"] = away_standings
                        if away_standings["status"] in ["clinched", "eliminated"]:
                            logger.info(f"  📊 {game['away_team']}: {away_standings['status'].upper()}")

                # 5b) Team Offensive Stats (wOBA, OPS)
                if game.get("home_team_id"):
                    home_off = self.mlb_api.get_team_offensive_stats(game["home_team_id"], season)
                    if home_off:
                        enriched["home_offensive_stats"] = home_off
                        logger.info(f"  ✅ {game['home_team']} offense: wOBA={home_off['woba']} OPS={home_off['ops']}")
                if game.get("away_team_id"):
                    away_off = self.mlb_api.get_team_offensive_stats(game["away_team_id"], season)
                    if away_off:
                        enriched["away_offensive_stats"] = away_off
                        logger.info(f"  ✅ {game['away_team']} offense: wOBA={away_off['woba']} OPS={away_off['ops']}")

                # 6) Travel Fatigue (coordinate-based miles + time zones)
                if game.get("away_team_id") and game.get("game_date"):
                    travel = self.mlb_api.get_travel_fatigue(
                        game["away_team_id"],
                        game["game_date"],
                        current_venue=game.get("venue", "")
                    )
                    if travel:
                        enriched["away_travel_fatigue"] = travel
                        enriched["miles_traveled_away"] = travel.get("miles_traveled", 0)
                        enriched["time_zones_crossed_away"] = travel.get("time_zones_crossed", 0)
                        enriched["back_to_back_away"] = travel.get("back_to_back", False)
                        if travel.get("has_travel_fatigue"):
                            logger.info(f"  ✈️ {game['away_team']}: {travel['miles_traveled']} mi, {travel['time_zones_crossed']} TZ")

                # 7) Days rest (home and away)
                game_date_str = (game.get("game_date") or "")[:10]
                if game.get("home_team_id") and game_date_str:
                    enriched["home_days_rest"] = self.mlb_api.get_team_days_rest(game["home_team_id"], game_date_str)
                if game.get("away_team_id") and game_date_str:
                    enriched["away_days_rest"] = self.mlb_api.get_team_days_rest(game["away_team_id"], game_date_str)

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

                # 8b) Defensive efficiency dicts (DER from pitching stats, OAA absent by default)
                #     Convention: defense_home = home team's fielding; defense_away = away team's
                for _side, _pitch_key, _def_key, _team_name in [
                    ("home", "home_pitching_stats", "defense_home", game.get("home_team", "")),
                    ("away", "away_pitching_stats", "defense_away", game.get("away_team", "")),
                ]:
                    _ps = enriched.get(_pitch_key) or {}
                    _der = _ps.get("der")
                    _bip = _ps.get("bip", 0)
                    if _der is not None and _bip > 0:
                        enriched[_def_key] = {
                            "team_name": _team_name,
                            "der": _der,
                            "bip": _bip,
                            "oaa": None,   # OAA not available from free MLB API
                        }
                        logger.info(f"  🛡️  {_team_name} defense: DER={_der:.4f}  BIP={_bip}")

                # 9) Lineup handedness (LHB% per side — used for platoon split adjustment)
                for _side, _lineup_key, _lhb_key in [
                    ("home", "home_lineup", "home_lineup_lhb_pct"),
                    ("away", "away_lineup", "away_lineup_lhb_pct"),
                ]:
                    lineup = enriched.get(_lineup_key) or []
                    if lineup:
                        pids = [p["id"] for p in lineup if p.get("id")]
                        hand_map = self.mlb_api.get_batter_handedness_batch(pids)
                        lhb_n = sum(
                            1 for p in lineup
                            if hand_map.get(p.get("id"), "R") in ("L", "S")
                        )
                        total_n = len(lineup)
                        enriched[_lhb_key] = round(lhb_n / total_n, 3) if total_n else 0.45
                        logger.info(
                            f"  ✅ {_side.upper()} lineup: "
                            f"{lhb_n}L/{total_n - lhb_n}R "
                            f"({enriched[_lhb_key]:.0%} LHB) — "
                            f"{', '.join(p.get('name','?') for p in lineup[:3])}..."
                        )
                    else:
                        enriched[_lhb_key] = 0.45  # league average fallback

                # 10) Home plate umpire zone tendency
                hp_ump_id = enriched.get("hp_umpire_id")
                if hp_ump_id:
                    ump_stats = self.mlb_api.get_umpire_historical_stats(hp_ump_id)
                    if ump_stats and ump_stats.get("games_worked", 0) >= 4:
                        enriched["umpire_stats"] = ump_stats
                        logger.info(
                            f"  ⚖️  HP Ump {enriched.get('hp_umpire_name','?')}: "
                            f"zone_factor={ump_stats['zone_factor']:.3f} "
                            f"strike%={ump_stats['strike_pct']:.1%} "
                            f"({ump_stats['games_worked']} games)"
                        )

                status_icon = "✅" if enriched["pitchers_valid"] else "⚠️"
                status_msg = "Data completa" if enriched["pitchers_valid"] else "Data incompleta - NO APOSTAR"
                logger.info(f"{status_icon} {game['away_team']} @ {game['home_team']}: {status_msg}\n")

                enriched_games.append(enriched)
                time.sleep(0.2)

            except Exception as e:
                logger.error(f"❌ Error enriqueciendo {game.get('home_team', 'juego')}: {e}")
                enriched_games.append(game)

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
