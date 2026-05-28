"""
Central configuration — single source of truth for all constants.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
CACHE_DIR  = BASE_DIR / ".cache"
MODULES_DIR = BASE_DIR / "modules"

# ── App metadata ───────────────────────────────────────────────────────────
APP_NAME    = "FINAL BOSS QUANT G8+"
APP_VERSION = "2.1"
PAGE_ICON   = "🎯"

# ── Simulation counts ──────────────────────────────────────────────────────
MLB_SIMULATIONS = 5_000_000
NBA_SIMULATIONS = 50_000
UFC_SIMULATIONS = 100_000

# ── Kelly / bankroll ───────────────────────────────────────────────────────
KELLY_FRACTION  = 0.25   # fractional Kelly (quarter-Kelly default)
MAX_RISK_PCT    = 0.05   # max 5 % of bankroll per bet
MIN_STAKE       = 1.0    # minimum absolute stake (USD)

# ── Value detection ────────────────────────────────────────────────────────
MIN_KELLY       = 0.01
MAX_KELLY       = 0.15
MIN_CONFIDENCE  = 0.65
MIN_EDGE        = 0.5    # percentage points
VIG_METHODS     = ["multiplicative", "power", "shin"]
BOOTSTRAP_SAMPLES = 1_000
CI_LEVEL        = 0.95

# ── EV / rating thresholds ─────────────────────────────────────────────────
DEFAULT_MIN_EV     = 3.0   # % minimum EV to flag a bet
DEFAULT_MIN_RATING = 6.5

# ── MLB league averages (2024 season) ──────────────────────────────────────
LEAGUE_AVG_RUNS = 4.5    # D2 reverted: empirical 4.427 degraded Brier 0.24209→0.24307; hfa_mult formula was tuned at 4.5
LEAGUE_AVG_OPS  = 0.735
LEAGUE_AVG_ERA  = 4.15
LEAGUE_AVG_WHIP = 1.30
LEAGUE_AVG_WOBA = 0.310  # FanGraphs Guts! 2024

# ── Odds API / caching ─────────────────────────────────────────────────────
ODDS_CACHE_TTL       = 300   # seconds — Streamlit function-cache TTL
ODDS_FILE_CACHE_TTL  = 600   # seconds — on-disk JSON cache TTL (odds_fetcher)
MLB_FALLBACK_GAME_ID = 746_929  # World Series 2024

# ── UI limits ──────────────────────────────────────────────────────────────
MAX_HISTORY_RECORDS = 200
MAX_DISPLAY_RECORDS = 50

# ── Pitcher engine weights ──────────────────────────────────────────────────
# 5 signals only; park/travel/bullpen removed (double-counted or data-free).
# Proportionally redistributed from the original 5-signal sum of 0.78.
PITCHER_ENGINE_WEIGHTS = {
    'pitcher_quality': 0.321,   # SIERA/xFIP/FIP/ERA composite + Savant overlays
    'pitcher_form':    0.256,   # era_last_5 level, era_trend slope, QS%
    'pitcher_matchup': 0.192,   # historical ERA vs this opponent
    'pitcher_fatigue': 0.128,   # days rest + last pitch count
    'pitcher_platoon': 0.103,   # L/R split × opposing lineup handedness
}
