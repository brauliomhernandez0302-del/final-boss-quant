"""
Central configuration — single source of truth for all constants.
"""

import os
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

# ── Value detection ────────────────────────────────────────────────────────
MIN_KELLY       = 0.01
MAX_KELLY       = 0.15   # single source of truth for the max stake fraction per bet
MIN_CONFIDENCE  = 0.65
MIN_EDGE        = 0.5    # percentage points
VIG_METHODS     = ["multiplicative", "power", "shin"]
BOOTSTRAP_SAMPLES = 1_000
CI_LEVEL        = 0.95

# ── EV / rating thresholds ─────────────────────────────────────────────────
DEFAULT_MIN_EV     = 3.0   # % minimum EV to flag a bet
DEFAULT_MIN_RATING = 6.5

# ── Track record publish mode (Fase 2A commit 4) ───────────────────────────
# Every pick the pipeline generates publishes as 'quarantine' — visible in
# track_record's UI with an explicit badge, closing lines captured for all
# of it, but not presented as the public official record. The public band
# (an EV/tier cutoff for what counts as a real pick) is Fase 2C's job, fixed
# against a freshly re-measured baseline — not decided here. Set
# QUARANTINE_MODE=false in .env to flip this once that band exists; not
# meant to be flipped casually before then.
QUARANTINE_MODE = os.getenv("QUARANTINE_MODE", "true").strip().lower() != "false"

# ── MLB league averages (2024 season) ──────────────────────────────────────
LEAGUE_AVG_RUNS = 4.5    # D2 reverted: empirical 4.427 degraded Brier 0.24209→0.24307; hfa_mult formula was tuned at 4.5
LEAGUE_AVG_OPS  = 0.735
LEAGUE_AVG_ERA  = 4.15
LEAGUE_AVG_WHIP = 1.30
LEAGUE_AVG_WOBA = 0.310  # FanGraphs Guts! 2024
# Statcast expected wOBA (≠ traditional wOBA above) — empirically measured
# 2026-07-05 via Savant's expected_statistics leaderboard with min=1 PA
# (matches true_talent_engine.py's own fetch filter exactly): PA-weighted
# league avg was 0.3157 (2024) and 0.3163 (2025), both meaningfully above the
# 0.312 this constant used to duplicate across 4 files (true_talent_engine.py,
# pitcher_engine.py, bullpen_engine.py, tte_pit_adapter.py) — same
# systematic-stale-constant pattern as the LG_DER fix in
# defensive_efficiency_engine.py. Centralized here as the single source of
# truth so it can't drift out of sync across files again.
LEAGUE_AVG_XWOBA = 0.316

# Fracción zurda de una alineación típica. Último recurso de la cadena
# lineup confirmado → estimación del equipo → esto; sólo se llega acá cuando
# ninguna de las dos primeras resuelve.
#
# Era 0.45 en dos archivos distintos (`park_weather_engine._AVG_LHB_PCT` y un
# literal suelto en `pitcher_engine`) — mismo patrón de constante duplicada que
# LEAGUE_AVG_XWOBA. Medido el 2026-07-27 sobre las 24 alineaciones reales de la
# cartelera: media 0.406, mediana 0.417, rango 0.267-0.538. El 0.45 estaba
# sesgado ~4pp hacia la izquierda contra el dato real.
LEAGUE_AVG_LHB_PCT = 0.406

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
#
# ⚠️ `pitcher_form` PESA 0.256 PERO APORTA CERO. Su factor
# (`pitcher_engine._adjust_pitcher_form`) está neutralizado desde el paso 8 de
# la auditoría y devuelve 1.0 fijo: sus tres señales salían de la misma lista
# de cinco arranques —contando un dato tres veces— y con errores agrupados por
# pitcher ninguna sobrevive al control por nivel; medían regresión a la media y
# el motor las leía como persistencia. La combinación del motor es ADITIVA
# sobre deltas (`total = 1 + Σ wᵢ·(fᵢ−1)`), así que un factor en identidad
# aporta exactamente 0 sin importar su peso y NO diluye a los otros cuatro —
# por eso el 0.256 se dejó donde está en vez de redistribuirlo. Se anota acá
# porque leyendo sólo esta tabla parecería la segunda señal más importante del
# motor, y es la única que no hace nada. Evidencia completa y condiciones para
# reinstaurarla: docstring de `_adjust_pitcher_form`.
PITCHER_ENGINE_WEIGHTS = {
    'pitcher_quality': 0.321,   # SIERA/xFIP/FIP/ERA composite + Savant overlays
    'pitcher_form':    0.256,   # INERTE — ver la nota de arriba (devuelve 1.0)
    'pitcher_matchup': 0.192,   # historical ERA vs this opponent
    'pitcher_fatigue': 0.128,   # days rest + last pitch count
    'pitcher_platoon': 0.103,   # L/R split × opposing lineup handedness
}
