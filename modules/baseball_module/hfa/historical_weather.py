"""
Historical weather fetcher for backtest pipeline.

Uses Open-Meteo archive API (ERA5 reanalysis — free, no API key required).
Pre-fetches entire MLB seasons per venue in one API call each and caches
to disk. The returned dict matches the format consumed by park_weather_engine.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

log = logging.getLogger(__name__)

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# UTC hours covering the MLB game window:
# 16:00-22:00 UTC = noon-6pm Eastern = 9am-3pm Pacific
# Captures afternoon day games and evening starts across all time zones.
_GAME_HOURS_UTC = set(range(16, 23))

# Stadium coordinates (lat, lon).  Matches WeatherAPI.STADIUM_COORDS in data_fetchers.py.
_STADIUM_COORDS: Dict[str, Tuple[float, float]] = {
    "Fenway Park":                (42.3467, -71.0972),
    "Yankee Stadium":             (40.8296, -73.9262),
    "Camden Yards":               (39.2838, -76.6216),
    "Tropicana Field":            (27.7682, -82.6534),
    "Rogers Centre":              (43.6414, -79.3894),
    "Guaranteed Rate Field":      (41.8300, -87.6338),
    "Progressive Field":          (41.4962, -81.6852),
    "Comerica Park":              (42.3390, -83.0489),
    "Kauffman Stadium":           (39.0517, -94.4803),
    "Target Field":               (44.9817, -93.2779),
    "Minute Maid Park":           (29.7573, -95.3555),
    "Globe Life Field":           (32.7512, -97.0837),
    "Angel Stadium":              (33.8003, -117.8827),
    "T-Mobile Park":              (47.5914, -122.3325),
    "RingCentral Coliseum":       (37.7516, -122.2005),
    "Sutter Health Park":         (38.5802, -121.5011),
    "Citizens Bank Park":         (39.9061, -75.1665),
    "Citi Field":                 (40.7571, -73.8458),
    "Nationals Park":             (38.8730, -77.0074),
    "Truist Park":                (33.8907, -84.4685),
    "loanDepot park":             (25.7781, -80.2201),
    "Wrigley Field":              (41.9484, -87.6553),
    "Great American Ball Park":   (39.0979, -84.5063),
    "American Family Field":      (43.0281, -87.9712),
    "PNC Park":                   (40.4468, -80.0057),
    "Busch Stadium":              (38.6226, -90.1928),
    "Dodger Stadium":             (34.0739, -118.2400),
    "Chase Field":                (33.4453, -112.0667),
    "Oracle Park":                (37.7786, -122.3893),
    "Petco Park":                 (32.7073, -117.1566),
    "Coors Field":                (39.7559, -104.9942),
}


class HistoricalWeatherFetcher:
    """
    Fetch and cache historical game-day weather for MLB stadiums.

    Usage:
        fetcher = HistoricalWeatherFetcher(Path(".cache/historical_weather_cache.json"))
        fetcher.prefetch_all(seasons=[2024, 2025, 2026])
        wx = fetcher.get("Fenway Park", "2024-07-04")
        # wx → {"temp_f": 78.2, "wind_speed_mph": 12.5, "wind_direction": 225.0, ...}
    """

    def __init__(self, cache_path: Path):
        self._cache_path = cache_path
        self._cache: Dict[str, Any] = {}
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text())
                log.info("Historical weather cache: %d entries loaded", len(self._cache))
            except Exception as exc:
                log.warning("Could not load weather cache: %s", exc)
                self._cache = {}

    # ── public API ──────────────────────────────────────────────────────────────

    def prefetch_all(self, seasons: list, delay_s: float = 0.5) -> None:
        """
        Pre-fetch weather for all venues × all seasons.

        Skips venue+season pairs already in the cache.  Sleeps `delay_s`
        between API calls to stay within Open-Meteo rate limits (10 req/s).
        """
        all_venues = list(_STADIUM_COORDS.keys())
        to_fetch = [
            (v, s) for v in all_venues for s in seasons
            if f"__season__{v}__{s}" not in self._cache
        ]
        if not to_fetch:
            log.info("Historical weather: all %d venue×seasons already cached",
                     len(all_venues) * len(seasons))
            return

        log.info("Historical weather: pre-fetching %d venue×season pairs via Open-Meteo …",
                 len(to_fetch))
        n_ok, n_err = 0, 0
        for venue, season in to_fetch:
            n = self._fetch_season(venue, season)
            if n >= 0:
                n_ok += 1
            else:
                n_err += 1
            if delay_s > 0:
                time.sleep(delay_s)

        self._save()
        log.info("Historical weather: prefetch done — %d ok, %d errors, %d total entries cached",
                 n_ok, n_err, len(self._cache))

    def get(self, venue: str, date_str: str) -> Dict[str, Any]:
        """Return weather dict for venue on date_str (YYYY-MM-DD), or {} on miss."""
        return dict(self._cache.get(f"{venue}:{date_str}", {}))

    # ── private ─────────────────────────────────────────────────────────────────

    def _fetch_season(self, venue: str, season: int) -> int:
        """
        Fetch a full season for one venue from Open-Meteo archive.
        Returns number of new daily records cached, or -1 on error.
        """
        coords = _STADIUM_COORDS.get(venue)
        if not coords:
            return 0

        lat, lon = coords
        start_date = f"{season}-03-01"
        end_date   = f"{season}-11-30"

        try:
            resp = requests.get(
                _ARCHIVE_URL,
                params={
                    "latitude":        lat,
                    "longitude":       lon,
                    "start_date":      start_date,
                    "end_date":        end_date,
                    "hourly":          "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation",
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                    "timezone":        "UTC",
                },
                timeout=(10, 90),
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("Open-Meteo error for %s %d: %s", venue, season, exc)
            return -1

        n = self._parse_and_cache(venue, data)
        self._cache[f"__season__{venue}__{season}"] = True   # mark as fully fetched
        log.debug("  %s %d → %d daily records cached", venue, season, n)
        return n

    def _parse_and_cache(self, venue: str, api_data: Dict) -> int:
        """Parse Open-Meteo hourly response into daily game-window summaries."""
        hourly = api_data.get("hourly", {})
        times  = hourly.get("time", [])
        temps  = hourly.get("temperature_2m", [])
        winds  = hourly.get("wind_speed_10m", [])
        wdirs  = hourly.get("wind_direction_10m", [])
        precips = hourly.get("precipitation", [])

        from collections import defaultdict
        by_date: Dict[str, list] = defaultdict(list)
        for i, ts in enumerate(times):
            try:
                d_str, h_str = ts.split("T")
                h = int(h_str.split(":")[0])
            except Exception:
                continue
            if h not in _GAME_HOURS_UTC:
                continue
            by_date[d_str].append({
                "temp_f":    temps[i]  if i < len(temps)   else None,
                "wind_mph":  winds[i]  if i < len(winds)   else None,
                "wind_dir":  wdirs[i]  if i < len(wdirs)   else None,
                "precip_mm": precips[i] if i < len(precips) else None,
            })

        n = 0
        for d_str, slots in by_date.items():
            key = f"{venue}:{d_str}"
            if key in self._cache:
                continue
            valid = [s for s in slots if s["temp_f"] is not None]
            if not valid:
                continue

            avg_temp = sum(s["temp_f"] for s in valid) / len(valid)
            avg_wind = sum(s["wind_mph"] for s in valid if s["wind_mph"] is not None) / max(
                1, sum(1 for s in valid if s["wind_mph"] is not None)
            )
            # Circular mean for wind direction (handles 350°/10° wraparound correctly)
            valid_dirs = [s["wind_dir"] for s in valid if s["wind_dir"] is not None]
            if valid_dirs:
                sin_s = sum(math.sin(math.radians(d)) for d in valid_dirs)
                cos_s = sum(math.cos(math.radians(d)) for d in valid_dirs)
                avg_dir = math.degrees(math.atan2(sin_s, cos_s)) % 360
            else:
                avg_dir = 0.0

            total_precip = sum(
                s["precip_mm"] for s in valid if s["precip_mm"] is not None
            )
            # Conditions string — keep it simple for engine compatibility
            if total_precip > 5.0:
                conditions = "Rain"
            elif total_precip > 1.0:
                conditions = "Drizzle"
            else:
                conditions = "Clear"

            self._cache[key] = {
                "temp_f":            round(avg_temp, 1),
                "wind_speed_mph":    round(avg_wind, 1),
                "wind_direction":    round(avg_dir),
                "rain_mm":           round(total_precip, 2),
                "conditions":        conditions,
                "precip_probability": min(1.0, total_precip / 3.0) if total_precip > 0 else 0.0,
            }
            n += 1
        return n

    def _save(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache, separators=(",", ":")))
