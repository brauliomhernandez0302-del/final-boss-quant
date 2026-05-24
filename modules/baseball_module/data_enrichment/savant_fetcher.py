"""
Baseball Savant (Statcast) leaderboard fetcher.

Pulls two free CSV endpoints (no auth) and merges them into a per-pitcher
dict keyed by MLBAM player_id. Results are cached for CACHE_TTL_HOURS.

Endpoints used:
  /leaderboard/expected_statistics  → xERA, xwOBA (est_woba), est_ba, est_slg
  /leaderboard/statcast             → barrel%, avg exit velo, EV95%
"""

import csv
import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

_EXPECTED_URL = "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
_EV_URL       = "https://baseballsavant.mlb.com/leaderboard/statcast"
_CACHE_TTL    = 86_400  # 24 h in seconds
_TIMEOUT      = (5, 30)


class SavantFetcher:
    """Fetches and caches Savant pitcher leaderboards."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = Path(cache_dir) if cache_dir else Path("/tmp")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; FinalBossQuant/1.0)"})

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_pitcher_stats(self, mlbam_id: int, year: int) -> Dict:
        """Return merged Savant stats dict for one pitcher (empty dict if not found)."""
        all_stats = self.get_all_pitcher_stats(year)
        return all_stats.get(int(mlbam_id), {})

    def get_all_pitcher_stats(self, year: int) -> Dict[int, Dict]:
        """Return dict keyed by MLBAM player_id for all qualified pitchers."""
        expected = self._fetch_expected(year)
        ev       = self._fetch_ev(year)
        merged: Dict[int, Dict] = {}
        for pid in set(expected) | set(ev):
            merged[pid] = {**expected.get(pid, {}), **ev.get(pid, {})}
        return merged

    # ------------------------------------------------------------------
    # Internal — expected stats (xERA, xwOBA)
    # ------------------------------------------------------------------

    def _fetch_expected(self, year: int) -> Dict[int, Dict]:
        cache_key = f"savant_expected_{year}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"[savant] downloading expected stats ({year})...")
        try:
            r = self._session.get(
                _EXPECTED_URL,
                params={"type": "pitcher", "year": str(year), "csv": "true"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            result = self._parse_expected_csv(r.text)
            self._save_cache(cache_key, result)
            logger.info(f"[savant] expected stats: {len(result)} pitchers")
            return result
        except Exception as exc:
            logger.warning(f"[savant] expected stats fetch failed: {exc}")
            return {}

    def _parse_expected_csv(self, text: str) -> Dict[int, Dict]:
        out: Dict[int, Dict] = {}
        reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
        for row in reader:
            try:
                pid = int(row.get("player_id", 0) or 0)
                if not pid:
                    continue
                out[pid] = {
                    "est_woba":             _f(row.get("est_woba")),
                    "xera":                 _f(row.get("xera")),
                    "era":                  _f(row.get("era")),
                    "woba":                 _f(row.get("woba")),
                    "est_ba":               _f(row.get("est_ba")),
                    "est_slg":              _f(row.get("est_slg")),
                    "era_minus_xera_diff":  _f(row.get("era_minus_xera_diff") or row.get("era_minu")),
                    "pa":                   _i(row.get("pa")),
                    "bip":                  _i(row.get("bip")),
                    "savant_year":          _i(row.get("year")),
                }
            except (ValueError, KeyError):
                continue
        return out

    # ------------------------------------------------------------------
    # Internal — EV / barrel stats
    # ------------------------------------------------------------------

    def _fetch_ev(self, year: int) -> Dict[int, Dict]:
        cache_key = f"savant_ev_{year}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"[savant] downloading EV/barrel stats ({year})...")
        try:
            r = self._session.get(
                _EV_URL,
                params={"type": "pitcher", "year": str(year), "csv": "true"},
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            result = self._parse_ev_csv(r.text)
            self._save_cache(cache_key, result)
            logger.info(f"[savant] EV stats: {len(result)} pitchers")
            return result
        except Exception as exc:
            logger.warning(f"[savant] EV stats fetch failed: {exc}")
            return {}

    def _parse_ev_csv(self, text: str) -> Dict[int, Dict]:
        out: Dict[int, Dict] = {}
        reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
        for row in reader:
            try:
                pid = int(row.get("player_id", 0) or 0)
                if not pid:
                    continue
                out[pid] = {
                    "avg_hit_speed":        _f(row.get("avg_hit_speed")),
                    "brl_percent":          _f(row.get("brl_percent")),
                    "brl_pa":               _f(row.get("brl_pa")),
                    "ev95percent":          _f(row.get("ev95percent")),
                    "ev95plus":             _i(row.get("ev95plus")),
                    "max_hit_speed":        _f(row.get("max_hit_speed")),
                    "ev50":                 _f(row.get("ev50")),
                    "avg_hit_angle":        _f(row.get("avg_hit_angle")),
                    "sweet_spot_pct":       _f(row.get("anglesweetspotpercent")),
                    "attempts":             _i(row.get("attempts")),
                }
            except (ValueError, KeyError):
                continue
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
            logger.debug(f"[savant] cache write failed: {exc}")


# ------------------------------------------------------------------
# Type helpers
# ------------------------------------------------------------------

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
