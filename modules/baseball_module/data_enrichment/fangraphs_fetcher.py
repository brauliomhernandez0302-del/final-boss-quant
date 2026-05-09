"""
FanGraphs pitcher leaderboard fetcher.

Uses the free public JSON API (no auth required). Returns real xFIP, SIERA,
WAR, FIP, K%, BB%, SwStr% etc. keyed by MLBAM player_id (xMLBAMID field).

Name and Team columns contain HTML anchors — stripped before returning.

Endpoint:
  GET https://www.fangraphs.com/api/leaders/major-league/data
      ?pos=all&stats=pit&lg=all&qual=0&season=YEAR&season1=YEAR
      &ind=0&team=0&pageitems=600&pagenum=1&type=8
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

_FG_URL    = "https://www.fangraphs.com/api/leaders/major-league/data"
_CACHE_TTL = 86_400   # 24 h
_TIMEOUT   = 15
_TAG_RE    = re.compile(r"<[^>]+>")


class FanGraphsFetcher:
    """Fetches and caches FanGraphs pitcher leaderboard data."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = Path(cache_dir) if cache_dir else Path("/tmp")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; FinalBossQuant/1.0)"})

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_pitcher_stats(self, mlbam_id: int, year: int) -> Dict:
        """Return FanGraphs stats dict for one pitcher (empty dict if not found)."""
        all_stats = self.get_all_pitcher_stats(year)
        return all_stats.get(int(mlbam_id), {})

    def get_all_pitcher_stats(self, year: int) -> Dict[int, Dict]:
        """Return dict keyed by MLBAM player_id for all pitchers in leaderboard."""
        cache_key = f"fg_pitchers_{year}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"[fangraphs] downloading pitcher leaderboard ({year})...")
        try:
            r = self._session.get(
                _FG_URL,
                params={
                    "pos": "all", "stats": "pit", "lg": "all", "qual": "0",
                    "season": str(year), "season1": str(year),
                    "ind": "0", "team": "0", "pageitems": "600", "pagenum": "1",
                    "type": "8",
                },
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            rows = data.get("data", [])
            result = self._parse_rows(rows)
            self._save_cache(cache_key, result)
            logger.info(f"[fangraphs] leaderboard: {len(result)} pitchers")
            return result
        except Exception as exc:
            logger.warning(f"[fangraphs] fetch failed: {exc}")
            return {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_rows(self, rows) -> Dict[int, Dict]:
        out: Dict[int, Dict] = {}
        for row in rows:
            mlbam_id = row.get("xMLBAMID")
            if not mlbam_id:
                continue
            try:
                mlbam_id = int(mlbam_id)
            except (TypeError, ValueError):
                continue

            out[mlbam_id] = {
                # Core ERA estimators — the three most predictive metrics
                "xfip":         _f(row.get("xFIP")),
                "siera":        _f(row.get("SIERA")),
                "fip":          _f(row.get("FIP")),
                "xera":         _f(row.get("xERA")),
                "era":          _f(row.get("ERA")),
                # Wins above replacement
                "war":          _f(row.get("WAR")),
                # Rate stats
                "k_pct":        _f(row.get("K%")),
                "bb_pct":       _f(row.get("BB%")),
                "k_bb_pct":     _f(row.get("K-BB%")),
                "swstr_pct":    _f(row.get("SwStr%")),
                "babip":        _f(row.get("BABIP")),
                "lob_pct":      _f(row.get("LOB%")),
                "hr_fb":        _f(row.get("HR/FB")),
                "gb_pct":       _f(row.get("GB%")),
                "fb_pct":       _f(row.get("FB%")),
                "ld_pct":       _f(row.get("LD%")),
                # Volume
                "ip":           _f(row.get("IP")),
                "whip":         _f(row.get("WHIP")),
                # Contact quality from bat tracking / Statcast fields
                "avg_bat_speed":    _f(row.get("AvgBatSpeed")),
                "hard_contact_pct": _f(row.get("Hard%")),
                # Pitcher IDs
                "fg_playerid":  row.get("playerid"),
                "fg_name":      _strip_html(row.get("Name", "")),
                "fg_team":      _strip_html(row.get("Team", "")),
                "fg_year":      _i(row.get("Season")),
            }
        return out

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[Dict]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > _CACHE_TTL:
            return None
        try:
            raw = json.loads(p.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:
            return None

    def _save_cache(self, key: str, data: Dict[int, Dict]) -> None:
        try:
            self._cache_path(key).write_text(json.dumps({str(k): v for k, v in data.items()}))
        except Exception as exc:
            logger.debug(f"[fangraphs] cache write failed: {exc}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _strip_html(s: str) -> str:
    return _TAG_RE.sub("", s).strip()


def _f(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "", "null", "NULL") else None
    except (TypeError, ValueError):
        return None


def _i(v) -> Optional[int]:
    try:
        return int(float(v)) if v not in (None, "", "null", "NULL") else None
    except (TypeError, ValueError):
        return None
