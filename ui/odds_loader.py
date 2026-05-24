"""
ui/odds_loader.py — Odds data pipeline for Streamlit.

Responsibilities:
  - Fetch + cache odds from odds_fetcher
  - Filter by sport keyword
  - Build the game selector dropdown mapping
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

import config as _cfg
from db.predictions_db import GameData

logger = logging.getLogger(__name__)

# Import odds_fetcher gracefully — app still works without it (cache fallback)
try:
    from odds_fetcher import get_odds_data as _get_odds_data
except Exception:
    _get_odds_data = None


# ── Utilities ─────────────────────────────────────────────────────────────


def safe_to_dataframe(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data
    try:
        return pd.DataFrame(data)
    except Exception as exc:
        logger.warning("Error converting to DataFrame: %s", exc)
        return pd.DataFrame()


def parse_game_datetime(commence_time: Any) -> str:
    try:
        if isinstance(commence_time, str):
            dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        return str(commence_time)
    except Exception:
        return str(commence_time)


# ── Main loader (Streamlit-cached) ────────────────────────────────────────


@st.cache_data(ttl=_cfg.ODDS_CACHE_TTL)
def load_odds_data() -> pd.DataFrame:
    """Fetch odds and return as DataFrame.  Uses odds_fetcher's own on-disk cache."""
    if _get_odds_data is None:
        logger.warning("odds_fetcher not available")
        return pd.DataFrame()

    cache_file = _cfg.CACHE_DIR / "odds_last.json"

    try:
        with st.spinner("Cargando odds..."):
            raw = _get_odds_data()
            return safe_to_dataframe(raw)

    except Exception as exc:
        logger.error("Error loading odds: %s", exc)
        st.error(f"❌ Error cargando odds: {exc}")

        # Fallback: read whatever odds_fetcher left on disk
        if cache_file.exists():
            import json
            try:
                with open(cache_file, encoding="utf-8") as f:
                    cached = json.load(f)
                st.warning("⚠️ Usando datos en caché")
                data = cached if isinstance(cached, list) else cached.get("data", [])
                return safe_to_dataframe(data)
            except Exception as cache_exc:
                logger.error("Error reading cache: %s", cache_exc)

        return pd.DataFrame()


# ── Filtering ─────────────────────────────────────────────────────────────


def filter_odds_by_sport(df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    """Keep only rows whose sport column contains one of the keywords."""
    if df.empty:
        return df

    sport_columns = ["sport_key", "sport", "sport_title", "key", "league"]
    sport_col = next((c for c in sport_columns if c in df.columns), None)

    if sport_col is None:
        logger.warning("No sport column found in odds DataFrame")
        return df

    try:
        mask = df[sport_col].astype(str).str.lower().apply(
            lambda x: any(kw in x for kw in keywords)
        )
        return df[mask].copy()
    except Exception as exc:
        logger.error("Error filtering odds: %s", exc)
        return df


# ── Game selector ─────────────────────────────────────────────────────────


def _safe_float(val: Any) -> Optional[float]:
    """Return float(val) or None if val is missing/zero/invalid."""
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def build_game_selector(
    sport_df: pd.DataFrame,
) -> Tuple[List[str], Dict[str, GameData]]:
    """Build dropdown labels and a label→GameData mapping from an odds DataFrame.

    Populates the full GameData — ML odds, Pinnacle reference, totals, and
    runline — so the caller can pass them straight into run_module() without
    a second API call.
    """
    options: List[str] = []
    mapping: Dict[str, GameData] = {}

    for _, row in sport_df.iterrows():
        home     = row.get("home_team", "Home")
        away     = row.get("away_team", "Away")
        commence = row.get("commence_time", "")
        date_str = parse_game_datetime(commence)

        label = f"{away} @ {home} — {date_str}"
        options.append(label)

        mapping[label] = GameData(
            home          = str(home),
            away          = str(away),
            home_odds     = _safe_float(row.get("home_odds"))  or 2.0,
            away_odds     = _safe_float(row.get("away_odds"))  or 2.0,
            pin_home      = _safe_float(row.get("pin_home")),
            pin_away      = _safe_float(row.get("pin_away")),
            total_line    = _safe_float(row.get("total_line")),
            total_over    = _safe_float(row.get("over_odds")),
            total_under   = _safe_float(row.get("under_odds")),
            runline_home  = _safe_float(row.get("runline_home")),
            runline_away  = _safe_float(row.get("runline_away")),
            commence_time = str(commence),
            raw_row       = row,
        )

    return options, mapping
