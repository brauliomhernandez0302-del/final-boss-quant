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
    """Return float(val) or None if val is missing/zero/invalid.

    El descarte de `<= 0` es deliberado y correcto para PRECIOS y líneas de
    total: una cuota decimal de 0 o negativa no existe, y dejarla pasar la
    haría indistinguible de una real aguas abajo. NO sirve para magnitudes
    con signo — ver `_safe_signed_float`.
    """
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _safe_signed_float(val: Any) -> Optional[float]:
    """Igual que `_safe_float` pero CONSERVA el signo (rechaza sólo lo no numérico).

    Existe por el punto firmado del runline: el local favorito se cotiza en
    −1.5, así que pasarlo por `_safe_float` lo convertía en None justo en el
    caso más común, borrando exactamente el dato que distingue al favorito.
    Un cero sí se descarta: un runline de 0 no es un mercado válido.
    """
    try:
        f = float(val)
        return f if f != 0 else None
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
        if label in mapping:
            # Doubleheader / duplicate event with an identical formatted
            # label — disambiguate instead of silently overwriting the
            # earlier game in the mapping (it would become unreachable).
            suffix = 2
            while f"{label} ({suffix})" in mapping:
                suffix += 1
            label = f"{label} ({suffix})"
        options.append(label)

        mapping[label] = GameData(
            home          = str(home),
            away          = str(away),
            # No fallback here on purpose: a fabricated even-money price would
            # be indistinguishable from a real one downstream — ui/mlb.py's
            # `if game_data.get("home_odds") and ...` treats it as real market
            # data and skips run_module()'s own live odds fetch entirely.
            # Missing odds must stay None so callers know to fall back.
            home_odds     = _safe_float(row.get("home_odds")),
            away_odds     = _safe_float(row.get("away_odds")),
            pin_home      = _safe_float(row.get("pin_home")),
            pin_away      = _safe_float(row.get("pin_away")),
            total_line    = _safe_float(row.get("total_line")),
            total_over    = _safe_float(row.get("over_odds")),
            total_under   = _safe_float(row.get("under_odds")),
            runline_home  = _safe_float(row.get("runline_home")),
            runline_away  = _safe_float(row.get("runline_away")),
            runline_line  = _safe_float(row.get("runline_line")),
            # El punto FIRMADO, no sólo su magnitud. `_normalize_event()` ya lo
            # emitía; esta función lo dejaba caer, así que el análisis lanzado
            # desde el selector llegaba a `analyze_runline` sin él y caía al
            # supuesto "local favorito" — la causa raíz de PURP-1. El camino de
            # cron nunca tuvo el problema porque get_best_odds_for_teams() sí lo
            # entrega. Ver el comentario del campo en db/predictions_db.py.
            # `_safe_signed_float`, NO `_safe_float`: éste último descarta todo
            # `<= 0`, y el punto del local favorito es −1.5 — el caso mayoritario.
            # Pasarlo por el helper de precios lo anulaba en silencio y dejaba
            # este arreglo sin efecto justo donde más importa.
            runline_home_point = _safe_signed_float(row.get("runline_home_point")),
            # Par propio de Pinnacle para los derivados: es contra lo que se
            # desvigoriza para obtener la línea justa, igual que pin_home/
            # pin_away en el moneyline. Los PUNTOS van por `_safe_signed_float`
            # por la misma razón que runline_home_point — el del local favorito
            # es −1.5 y `_safe_float` lo anularía.
            pin_total_over  = _safe_float(row.get("pin_total_over")),
            pin_total_under = _safe_float(row.get("pin_total_under")),
            pin_total_point = _safe_signed_float(row.get("pin_total_point")),
            pin_runline_home = _safe_float(row.get("pin_runline_home")),
            pin_runline_away = _safe_float(row.get("pin_runline_away")),
            pin_runline_home_point = _safe_signed_float(row.get("pin_runline_home_point")),
            # odds_fetcher.py::_normalize_event()'s own F5 naming scheme
            # (f5_home_odds/f5_over_odds/f5_under_odds) is a THIRD,
            # independent convention from GameOdds' f5_ml_home/f5_total_over
            # — mapped here rather than left as a silent mismatch, the same
            # class of bug just fixed one boundary over in
            # get_best_odds_for_teams(). f5_total_line already matches.
            f5_ml_home    = _safe_float(row.get("f5_home_odds")),
            f5_ml_away    = _safe_float(row.get("f5_away_odds")),
            f5_total_line = _safe_float(row.get("f5_total_line")),
            f5_total_over = _safe_float(row.get("f5_over_odds")),
            f5_total_under= _safe_float(row.get("f5_under_odds")),
            commence_time = str(commence),
            raw_row       = row,
        )

    return options, mapping
