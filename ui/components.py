"""
ui/components.py — Visual primitives and Streamlit config.

Contains:
  - ThemeColors: color palette constants
  - AppConfig: Streamlit-specific runtime config (wraps config.py constants)
  - SportConfig: per-sport UI configuration
  - UIComponents: reusable Streamlit HTML/metric blocks
  - EV / Kelly math helpers used exclusively for display
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import streamlit as st

import config as _cfg


# ── Palette ───────────────────────────────────────────────────────────────


class ThemeColors(Enum):
    PRIMARY  = "#00C2FF"
    SUCCESS  = "#00C853"
    DANGER   = "#FF5252"
    WARNING  = "#FDD835"
    NEUTRAL  = "#90A4AE"
    BG_DARK  = "#0C0F12"
    BG_CARD  = "rgba(255,255,255,0.05)"


# ── AppConfig ─────────────────────────────────────────────────────────────


@dataclass
class AppConfig:
    """Streamlit-specific runtime config sourced from config.py."""

    APP_NAME:    str = _cfg.APP_NAME
    APP_VERSION: str = _cfg.APP_VERSION
    PAGE_ICON:   str = _cfg.PAGE_ICON

    DEFAULT_KELLY_FACTOR: float = _cfg.KELLY_FRACTION
    DEFAULT_MIN_EV:       float = _cfg.DEFAULT_MIN_EV
    DEFAULT_MIN_RATING:   float = _cfg.DEFAULT_MIN_RATING

    MLB_SIMULATIONS: int = _cfg.MLB_SIMULATIONS
    NBA_SIMULATIONS: int = _cfg.NBA_SIMULATIONS
    UFC_SIMULATIONS: int = _cfg.UFC_SIMULATIONS

    MLB_FALLBACK_GAME_ID: int = _cfg.MLB_FALLBACK_GAME_ID
    ODDS_CACHE_TTL:       int = _cfg.ODDS_CACHE_TTL

    MAX_HISTORY_RECORDS: int = _cfg.MAX_HISTORY_RECORDS
    MAX_DISPLAY_RECORDS: int = _cfg.MAX_DISPLAY_RECORDS

    def __post_init__(self) -> None:
        self.BASE_DIR   = Path(_cfg.__file__).parent
        self.DATA_DIR   = self.BASE_DIR / "data"
        self.CACHE_DIR  = self.BASE_DIR / ".cache"
        self.MODULES_DIR = self.BASE_DIR / "modules"


# Module-level singleton — import CONFIG from here everywhere in ui/
CONFIG = AppConfig()


# ── SportConfig ───────────────────────────────────────────────────────────


@dataclass
class SportConfig:
    """Per-sport UI configuration."""
    name:         str
    display_name: str
    module_name:  str
    keywords:     List[str]
    icon:         str
    enabled:      bool = True
    simulations:  int  = 50_000


# ── Math helpers (UI-scoped) ──────────────────────────────────────────────
# These are intentionally separate from core/utils.py: different argument
# order, different scale, and used only for Streamlit display.


def calculate_ev(odds: Optional[float], prob: Optional[float]) -> float:
    """Return EV as a decimal (0.05 = 5 %).  NaN on bad inputs."""
    if odds is None or prob is None:
        return float("nan")
    try:
        o, p = float(odds), float(prob)
        if math.isnan(o) or math.isnan(p) or o <= 1.0 or not (0 < p < 1):
            return float("nan")
        return (o * p) - 1.0
    except (ValueError, TypeError):
        return float("nan")


def calculate_kelly(
    odds: Optional[float],
    prob: Optional[float],
    fraction: float = 0.25,
) -> float:
    """Fractional Kelly stake as a fraction of bankroll.

    Delegates to core.value_detector.kelly_criterion so the display cards
    use the same MIN_KELLY/MAX_KELLY clip as the pipeline's best_bets —
    previously this hardcoded its own 5% cap while best_bets used 15%,
    showing two different Kelly numbers for the same pick.
    """
    if odds is None or prob is None:
        return 0.0
    try:
        o, p = float(odds), float(prob)
        if o <= 1.0 or not (0 < p < 1):
            return 0.0
        from core.value_detector import kelly_criterion
        return kelly_criterion(p, o, fractional=fraction)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


def calculate_rating(ev: float) -> float:
    """Map EV (decimal) to a display rating 0–10."""
    if math.isnan(ev):
        return 0.0
    if ev > 0.10: return 10.0
    if ev > 0.05: return 8.0
    if ev > 0.03: return 7.0
    if ev > 0.01: return 6.0
    return 5.0


# ── UIComponents ──────────────────────────────────────────────────────────


class UIComponents:
    """Static Streamlit building blocks shared across all tabs."""

    @staticmethod
    def apply_theme() -> None:
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-color: {ThemeColors.BG_DARK.value};
                    color: #E0E6ED;
                }}
                h1, h2, h3 {{
                    color: {ThemeColors.PRIMARY.value} !important;
                }}
                [data-testid="stMetricValue"] {{
                    color: {ThemeColors.PRIMARY.value} !important;
                    font-weight: 700 !important;
                }}
                .stButton>button {{
                    background-color: {ThemeColors.PRIMARY.value} !important;
                    color: black !important;
                    font-weight: 700 !important;
                    border-radius: 8px !important;
                    border: none !important;
                }}
                .stSelectbox, .stTextInput, .stNumberInput {{
                    background-color: #12161B !important;
                }}
                .value-card {{
                    padding: 10px;
                    background: {ThemeColors.BG_CARD.value};
                    border-radius: 8px;
                    margin: 5px 0;
                }}
                .value-card-success {{
                    border-left: 4px solid {ThemeColors.SUCCESS.value};
                }}
                .value-card-danger {{
                    border-left: 4px solid {ThemeColors.DANGER.value};
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_header() -> None:
        st.markdown(
            f"""
            <div style="text-align:center; padding: 20px 0;">
                <h1 style="color:{ThemeColors.PRIMARY.value}; margin:0;">
                    {CONFIG.PAGE_ICON} {CONFIG.APP_NAME}
                </h1>
                <p style="opacity:0.8; margin:5px 0;">
                    Sistema Cuantitativo Multideporte de Predicción
                </p>
                <p style="opacity:0.6; font-size:14px;">
                    Poisson · Bayes · Monte Carlo · Auto-ML · EV · Kelly
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_status_bar() -> None:
        cache_file = CONFIG.CACHE_DIR / "odds_last.json"
        import os
        api_key = os.getenv("ODDS_API_KEY", "").strip()

        if cache_file.exists() and api_key:
            status, color = "🌐 API Online + Caché", ThemeColors.SUCCESS.value
        elif api_key:
            status, color = "🌐 API Online", ThemeColors.PRIMARY.value
        elif cache_file.exists():
            status, color = "♻️ Solo Caché Local", ThemeColors.WARNING.value
        else:
            status, color = "❌ Sin Datos", ThemeColors.DANGER.value

        st.markdown(
            f"""
            <div style='text-align:center; background:{color}; padding:8px;
                border-radius:8px; margin:10px 0;'>
                <b>{status}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_value_card(
        title: str,
        ev: float,
        kelly: float,
        odds: float,
        prob: float,
        min_ev: float,
    ) -> None:
        is_value  = ev > min_ev / 100
        card_class = "value-card-success" if is_value else "value-card-danger"
        color      = ThemeColors.SUCCESS.value if is_value else ThemeColors.DANGER.value

        st.markdown(f"#### {title}")
        st.metric("Market Odds",       f"{odds:.2f}")
        st.metric("Model Probability", f"{prob:.1%}")
        st.markdown(
            f"""
            <div class='value-card {card_class}'>
                <b>Expected Value:</b>
                <span style='color:{color}'>{ev:+.1%}</span><br>
                <b>Kelly Stake:</b> {kelly:.1%}
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_footer() -> None:
        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align:center; opacity:0.6; padding:20px;'>
                <p>{CONFIG.APP_NAME} © 2025 | BetMindex Dark Edition</p>
                <p style='font-size:12px;'>
                    Desarrollado con Streamlit + Python v{CONFIG.APP_VERSION} |
                    Modelos: Poisson, Bayes, Monte Carlo
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
