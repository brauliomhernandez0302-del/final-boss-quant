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
LEAGUE_AVG_RUNS = 4.5
LEAGUE_AVG_OPS  = 0.735
LEAGUE_AVG_ERA  = 4.15
LEAGUE_AVG_WHIP = 1.30

# ── Odds API / caching ─────────────────────────────────────────────────────
ODDS_CACHE_TTL       = 300   # seconds
MLB_FALLBACK_GAME_ID = 746_929  # World Series 2024

# ── UI limits ──────────────────────────────────────────────────────────────
MAX_HISTORY_RECORDS = 200
MAX_DISPLAY_RECORDS = 50
