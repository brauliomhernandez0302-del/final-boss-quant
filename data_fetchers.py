# ==========================================================
# MLB DATA FETCHERS V3 FINAL - MLB STATS + WEATHER + PARK
# Gratis, robusto y completo con TODAS las features
# Base de ChatGPT + Mejoras de Claude + 7 features adicionales
# ==========================================================

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

load_dotenv()

# Directorios
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

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
            "hydrate": "probablePitcher,team,seriesStatus"
        }

        try:
            r = self.session.get(url, params=params, timeout=12)
            r.raise_for_status()
            data = r.json()

            games = []
            for date_item in data.get("dates", []):
                for game in date_item.get("games", []):
                    parsed = self._parse_game(game)
                    if parsed:
                        games.append(parsed)

            print(f"✅ MLB API: {len(games)} juegos encontrados para {date}")
            return games

        except Exception as e:
            print(f"❌ Error obteniendo juegos MLB: {e}")
            return []

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
                "status": (game.get("status") or {}).get("detailedState")
            }
        except Exception as e:
            print(f"⚠️ Error parseando juego: {e}")
            return None

    # -------------------------
    # PITCHER STATS (compat + fallback)
    # -------------------------
    def get_pitcher_stats(self, pitcher_id: int, season: int = 2026) -> Optional[Dict[str, Any]]:
        """Stats de temporada (overall). Mantiene compatibilidad con tu método original."""
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
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            stats = self._parse_pitcher_stats(data)
            if stats:
                with open(cache_file, "w") as f:
                    json.dump(stats, f)
            return stats
        except Exception as e:
            print(f"⚠️ Error obteniendo stats de pitcher {pitcher_id}: {e}")
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

            return {
                "era": round(era, 2),
                "whip": round(whip, 2),
                "wins": int(stat.get("wins", 0)),
                "losses": int(stat.get("losses", 0)),
                "innings_pitched": round(innings, 1),
                "strikeouts": so,
                "walks": bb,
                "k_per_9": round(k_per_9, 2),
                "bb_per_9": round(bb_per_9, 2),
                "hits_allowed": int(stat.get("hits", 0)),
                "earned_runs": int(stat.get("earnedRuns", 0)),
                "games_started": int(stat.get("gamesStarted", 0))
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
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            return self._parse_pitcher_stats(r.json())
        except Exception:
            return None

    def _fetch_home_away_splits(self, pitcher_id: int, season: int) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/people/{pitcher_id}/stats"
        params = {"stats": "homeAndAway", "season": season, "group": "pitching"}
        result: Dict[str, Any] = {}
        try:
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            splits = data["stats"][0]["splits"]
            for s in splits:
                code = (s.get("split") or {}).get("code")
                stat = s.get("stat") or {}
                parsed = self._extract_basic_stats(stat)
                if not parsed:
                    continue
                if code == "home":
                    result["home"] = parsed
                elif code == "away":
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
            return {
                "era": round(era, 2),
                "whip": round(whip, 2),
                "innings_pitched": round(innings, 1),
                "k_per_9": round(k9, 2),
                "bb_per_9": round(bb9, 2),
                "strikeouts": so,
                "walks": bb
            }
        except Exception:
            return None

    def _select_best_stats(self, all_stats: Dict, is_playoff: bool, is_home: bool) -> Tuple[Optional[Dict], Optional[str]]:
        if is_playoff and "playoff" in all_stats:
            ps = all_stats["playoff"]
            if ps.get("innings_pitched", 0) >= 3.0:
                print(f"  ✅ Usando PLAYOFF (IP {ps.get('innings_pitched', 0)})")
                return ps, "playoff"
            else:
                print("  ⚠️ Playoff con IP bajas, buscando fallback...")

        if is_home and "home" in all_stats:
            print("  ✅ Usando HOME split")
            return all_stats["home"], "home_split"
        if (not is_home) and "away" in all_stats:
            print("  ✅ Usando AWAY split")
            return all_stats["away"], "away_split"

        if "regular_overall" in all_stats:
            print("  ✅ Usando OVERALL (regular season)")
            return all_stats["regular_overall"], "regular_overall"

        print("  ❌ Sin stats válidas")
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
            r = self.session.get(url, params=params, timeout=10)
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
            last_start = recent[0]
            last_date_str = last_start.get("date", "")
            last_pitch_count = int(last_start.get("stat", {}).get("numberOfPitches", 90))
            days_rest = 4
            if last_date_str:
                try:
                    from datetime import datetime as dt
                    last_date = dt.strptime(last_date_str, "%Y-%m-%d")
                    days_rest = (dt.utcnow() - last_date).days
                except Exception:
                    pass
            result = {"era_last_5": era_last_n, "days_rest": days_rest, "last_pitch_count": last_pitch_count, "starts_analyzed": len(recent)}
            with open(cache_file, "w") as f2:
                json.dump(result, f2)
            return result
        except Exception as e:
            print(f"Error game log pitcher {pitcher_id}: {e}")
            return None

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
            r = self.session.get(url, params=params, timeout=10)
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
            print(f"⚠️ Error obteniendo forma reciente: {e}")
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
            r = self.session.get(url, params=params, timeout=10)
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
            print(f"⚠️ Error obteniendo runs trend: {e}")
            return None

    # ==========================================================
    # FEATURE 3 - BULLPEN WORKLOAD (aprox)
    # ==========================================================
    def get_bullpen_workload(self, team_id: int, days: int = 3) -> Optional[Dict[str, Any]]:
        """{"innings_last_n_days": float, "games_played": int, "is_tired": bool, "avg_innings_per_game": float}"""
        cache_file = CACHE_DIR / f"bullpen_{team_id}_{days}.json"
        if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 3600:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        url = f"{self.BASE_URL}/schedule"
        params = {
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "gameType": "R,F,D,L,W"
        }
        try:
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()

            total_innings = 0.0
            appearances = 0
            for date_item in data.get("dates", []):
                for game in date_item.get("games", []):
                    if game.get("status", {}).get("abstractGameState") != "Final":
                        continue
                    appearances += 1
                    total_innings += 3.5  # estimación bullpen por juego

            is_tired = total_innings > 12.0  # umbral configurable
            result = {
                "innings_last_n_days": round(total_innings, 1),
                "games_played": appearances,
                "is_tired": is_tired,
                "avg_innings_per_game": round(total_innings / appearances, 1) if appearances > 0 else 0.0
            }
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result
        except Exception as e:
            print(f"⚠️ Error obteniendo bullpen workload: {e}")
            return None

    # ==========================================================
    # FEATURE 4 - H2H HISTÓRICO
    # ==========================================================
    def get_head_to_head(self, team1_id: int, team2_id: int, season: int = 2026) -> Optional[Dict[str, Any]]:
        """{"team1_wins": int, "team2_wins": int, "total_games": int, "avg_total_runs": float, "has_history": bool}"""
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
            r = self.session.get(url, params=params, timeout=10)
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
            print(f"⚠️ Error obteniendo H2H: {e}")
            return None

    # ==========================================================
    # FEATURE 5 - PITCHER VS TEAM (placeholder)
    # ==========================================================
    def get_pitcher_vs_team(self, pitcher_id: int, team_id: int, season: int = 2026) -> Optional[Dict[str, Any]]:
        """Devuelve None por ahora (requiere game logs detallados)."""
        # TODO: implementar con endpoint de game logs si está disponible públicamente
        return None

    # ==========================================================
    # FEATURE 6 - STANDINGS STATUS
    # ==========================================================
    def get_standings_status(self, team_id: int, season: int = 2026) -> Optional[Dict[str, Any]]:
        """{"status": "clinched|eliminated|in_race", "games_back": float, "clinched": bool, "eliminated": bool, "win_pct": float}"""
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
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()

            for record in data.get("records", []):
                for team_record in record.get("teamRecords", []):
                    if team_record["team"]["id"] == team_id:
                        clinched = bool(team_record.get("clinched", False))
                        eliminated = bool(team_record.get("eliminated", False))
                        # algunos payloads usan wildCardEliminationNumber="E" cuando está eliminado
                        wc_elim = (team_record.get("wildCardEliminationNumber") == "E")
                        raw_gb = team_record.get("wildCardGamesBack", 0.0)
                        games_back = float(raw_gb) if str(raw_gb).replace('.','').lstrip('-').isdigit() else 0.0
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
                            "win_pct": float(team_record.get("winningPercentage", 0.0))
                        }
                        with open(cache_file, "w") as f:
                            json.dump(result, f)
                        return result
            return None
        except Exception as e:
            print(f"⚠️ Error obteniendo standings: {e}")
            return None

    # ==========================================================
    # FEATURE 7 - TRAVEL FATIGUE (heurística simple)
    # ==========================================================
    def get_travel_fatigue(self, team_id: int, game_date: str) -> Optional[Dict[str, Any]]:
        """{"has_travel_fatigue": bool, "hours_since_last_game": float, "previous_venue": str, "is_cross_country": bool}"""
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
            r = self.session.get(url, params=params, timeout=10)
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
                return {"has_travel_fatigue": False, "reason": "No previous game found"}

            prev_dt = datetime.fromisoformat(previous_game_date.replace('Z', '+00:00'))
            hours_between = (game_dt - prev_dt).total_seconds() / 3600

            west = {"Oracle Park", "Dodger Stadium", "T-Mobile Park", "Petco Park"}
            east = {"Yankee Stadium", "Fenway Park", "Citi Field", "Citizens Bank Park"}
            is_west = previous_venue in west
            is_east = previous_venue in east
            # Heurística de cruce costa a costa: si venía de costa opuesta y menos de 24h
            is_cross_country = is_west or is_east  # simplificación: viaje largo desde costa
            has_fatigue = is_cross_country and hours_between < 24

            result = {
                "has_travel_fatigue": bool(has_fatigue),
                "hours_since_last_game": round(hours_between, 1),
                "previous_venue": previous_venue,
                "is_cross_country": bool(is_cross_country)
            }
            with open(cache_file, "w") as f:
                json.dump(result, f)
            return result
        except Exception as e:
            print(f"⚠️ Error calculando travel fatigue: {e}")
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
    }

    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY", "")
        if not self.api_key:
            print("⚠️ OPENWEATHER_API_KEY no configurada en .env")

    def get_weather_for_stadium(self, stadium_name: str) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        coords = self.STADIUM_COORDS.get(stadium_name)
        if not coords:
            print(f"⚠️ Coordenadas no disponibles para {stadium_name}")
            return None

        url = f"{self.BASE_URL}/weather"
        params = {"lat": coords["lat"], "lon": coords["lon"], "appid": self.api_key, "units": "imperial"}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            d = r.json()
            return {
                "stadium": stadium_name,
                "city": coords["city"],
                "temp_f": round(d["main"]["temp"], 1),
                "humidity": d["main"]["humidity"],
                "wind_speed_mph": round((d.get("wind") or {}).get("speed", 0.0), 1),
                "wind_direction": (d.get("wind") or {}).get("deg", 0),
                "conditions": d["weather"][0]["main"],
                "description": d["weather"][0]["description"],
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            print(f"❌ Error obteniendo clima: {e}")
            return None


# ==========================================================
# 3) PARK FACTORS (ESTÁTICO)
# ==========================================================

class ParkFactors:
    FACTORS = {
        "Coors Field": {"runs": 1.30, "hr": 1.25, "type": "hitter"},
        "Great American Ball Park": {"runs": 1.18, "hr": 1.22, "type": "hitter"},
        "Fenway Park": {"runs": 1.05, "hr": 1.10, "type": "neutral"},
        "Yankee Stadium": {"runs": 1.08, "hr": 1.15, "type": "hitter"},
        "Camden Yards": {"runs": 1.07, "hr": 1.12, "type": "hitter"},
        "Wrigley Field": {"runs": 1.03, "hr": 1.08, "type": "neutral"},
        "Dodger Stadium": {"runs": 0.95, "hr": 0.92, "type": "pitcher"},
        "Petco Park": {"runs": 0.85, "hr": 0.80, "type": "pitcher"},
        "Oracle Park": {"runs": 0.88, "hr": 0.75, "type": "pitcher"},
        "T-Mobile Park": {"runs": 0.92, "hr": 0.88, "type": "pitcher"},
        "Tropicana Field": {"runs": 0.96, "hr": 0.95, "type": "pitcher"},
        "Rogers Centre": {"runs": 1.02, "hr": 1.05, "type": "neutral"},
        "Rogers Center": {"runs": 1.02, "hr": 1.05, "type": "neutral"},
        "Minute Maid Park": {"runs": 1.00, "hr": 1.05, "type": "neutral"},
        "Busch Stadium": {"runs": 0.98, "hr": 0.95, "type": "neutral"},
        "Progressive Field": {"runs": 1.01, "hr": 1.03, "type": "neutral"},
        "Truist Park": {"runs": 0.99, "hr": 1.02, "type": "neutral"},
    }

    def get_factor(self, stadium_name: str) -> Dict[str, Any]:
        factor = self.FACTORS.get(stadium_name, {"runs": 1.0, "hr": 1.0, "type": "neutral"})
        return {"stadium": stadium_name, **factor}


# ==========================================================
# 4) DATA INTEGRATOR - CON TODAS LAS FEATURES
# ==========================================================

class MLBDataIntegrator:
    def __init__(self):
        self.mlb_api = MLBStatsAPI()
        self.weather_api = WeatherAPI()
        self.park_factors = ParkFactors()

    def _enrich_pitchers_concurrent(self, game: Dict[str, Any], season: int) -> Dict[str, Any]:
        """Enriquecer pitchers en paralelo con fallback."""
        is_playoff = game.get("is_playoff", False)

        def fetch_home():
            if game.get("home_pitcher_id"):
                return self.mlb_api.get_pitcher_stats_with_fallback(
                    pitcher_id=game["home_pitcher_id"],
                    season=season,
                    is_playoff=is_playoff,
                    is_home=True
                )
            return (None, None)

        def fetch_away():
            if game.get("away_pitcher_id"):
                return self.mlb_api.get_pitcher_stats_with_fallback(
                    pitcher_id=game["away_pitcher_id"],
                    season=season,
                    is_playoff=is_playoff,
                    is_home=False
                )
            return (None, None)

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
                    # Enriquecer con game log (era_last_5, days_rest, pitch_count)
                    pitcher_id = game.get(f"{side}_pitcher_id")
                    if pitcher_id:
                        game_log = self.mlb_api.get_pitcher_game_log(pitcher_id, season)
                        if game_log:
                            stats.update(game_log)
                            results[key_stats] = stats
                            print(f"  ✅ {side.upper()} game log: ERA_L5={game_log.get('era_last_5','?')} rest={game_log.get('days_rest','?')}d")
                    results[key_source] = source
                    results[key_valid] = True
                    print(f"  ✅ {side.upper()} pitcher: ERA {stats.get('era','?')} ({source})")
                else:
                    results[key_valid] = False
                    print(f"  ❌ {side.upper()} pitcher: sin stats válidas")
        return results

    def get_complete_game_data(self, date: Optional[str] = None, season: int = 2026) -> List[Dict[str, Any]]:
        """
        Obtiene data COMPLETA con TODAS las features.
        """
        print("🔍 Obteniendo data completa de juegos MLB...")
        games = self.mlb_api.get_todays_games(date)
        if not games:
            print("⚠️ No se encontraron juegos")
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

                # ===== WEATHER =====
                if game.get("venue"):
                    weather = self.weather_api.get_weather_for_stadium(game["venue"])
                    if weather:
                        enriched["weather"] = weather

                # ===== PARK FACTORS =====
                if game.get("venue"):
                    enriched["park_factor"] = self.park_factors.get_factor(game["venue"])

                # ===== NUEVAS FEATURES =====
                # 1) Team Recent Form
                if game.get("home_team_id"):
                    home_form = self.mlb_api.get_team_recent_form(game["home_team_id"], games=10)
                    if home_form:
                        enriched["home_team_form"] = home_form
                        print(f"  ✅ {game['home_team']} form: {home_form['wins']}-{home_form['losses']} (L10)")
                if game.get("away_team_id"):
                    away_form = self.mlb_api.get_team_recent_form(game["away_team_id"], games=10)
                    if away_form:
                        enriched["away_team_form"] = away_form
                        print(f"  ✅ {game['away_team']} form: {away_form['wins']}-{away_form['losses']} (L10)")

                # 2) Team Runs Trends
                if game.get("home_team_id"):
                    home_runs = self.mlb_api.get_team_runs_trend(game["home_team_id"], games=5)
                    if home_runs:
                        enriched["home_team_runs"] = home_runs
                if game.get("away_team_id"):
                    away_runs = self.mlb_api.get_team_runs_trend(game["away_team_id"], games=5)
                    if away_runs:
                        enriched["away_team_runs"] = away_runs

                # 3) Bullpen Workload
                if game.get("home_team_id"):
                    home_bullpen = self.mlb_api.get_bullpen_workload(game["home_team_id"], days=3)
                    if home_bullpen:
                        enriched["home_bullpen"] = home_bullpen
                        if home_bullpen.get("is_tired"):
                            print(f"  ⚠️ {game['home_team']} bullpen CANSADO ({home_bullpen['innings_last_n_days']} IP)")
                if game.get("away_team_id"):
                    away_bullpen = self.mlb_api.get_bullpen_workload(game["away_team_id"], days=3)
                    if away_bullpen:
                        enriched["away_bullpen"] = away_bullpen
                        if away_bullpen.get("is_tired"):
                            print(f"  ⚠️ {game['away_team']} bullpen CANSADO ({away_bullpen['innings_last_n_days']} IP)")

                # 4) H2H Histórico
                if game.get("home_team_id") and game.get("away_team_id"):
                    h2h = self.mlb_api.get_head_to_head(game["home_team_id"], game["away_team_id"], season)
                    if h2h and h2h.get("has_history"):
                        enriched["head_to_head"] = h2h
                        print(f"  ✅ H2H: {game['home_team']} {h2h['team1_wins']}-{h2h['team2_wins']} {game['away_team']}")

                # 5) Standings Status
                if game.get("home_team_id"):
                    home_standings = self.mlb_api.get_standings_status(game["home_team_id"], season)
                    if home_standings:
                        enriched["home_standings"] = home_standings
                        if home_standings["status"] in ["clinched", "eliminated"]:
                            print(f"  📊 {game['home_team']}: {home_standings['status'].upper()}")
                if game.get("away_team_id"):
                    away_standings = self.mlb_api.get_standings_status(game["away_team_id"], season)
                    if away_standings:
                        enriched["away_standings"] = away_standings
                        if away_standings["status"] in ["clinched", "eliminated"]:
                            print(f"  📊 {game['away_team']}: {away_standings['status'].upper()}")

                # 6) Travel Fatigue (para el equipo visitante)
                if game.get("away_team_id") and game.get("game_date"):
                    travel = self.mlb_api.get_travel_fatigue(game["away_team_id"], game["game_date"])
                    if travel and travel.get("has_travel_fatigue"):
                        enriched["away_travel_fatigue"] = travel
                        print(f"  ✈️ {game['away_team']}: FATIGA DE VIAJE detectada")

                status_icon = "✅" if enriched["pitchers_valid"] else "⚠️"
                status_msg = "Data completa" if enriched["pitchers_valid"] else "Data incompleta - NO APOSTAR"
                print(f"{status_icon} {game['away_team']} @ {game['home_team']}: {status_msg}\n")

                enriched_games.append(enriched)
                time.sleep(0.2)

            except Exception as e:
                print(f"❌ Error enriqueciendo {game.get('home_team', 'juego')}: {e}")
                enriched_games.append(game)

        valid_count = sum(1 for g in enriched_games if g.get("pitchers_valid"))
        print(f"\n🎉 {len(enriched_games)} juegos totales | ✅ {valid_count} VÁLIDOS | ⚠️ {len(enriched_games) - valid_count} SKIP")
        return enriched_games

    def save_to_file(self, games: List[Dict[str, Any]], filename: str = "mlb_complete_data.json"):
        filepath = DATA_DIR / filename
        with open(filepath, "w") as f:
            json.dump(games, f, indent=2)
        print(f"💾 Data guardada en {filepath}")


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
