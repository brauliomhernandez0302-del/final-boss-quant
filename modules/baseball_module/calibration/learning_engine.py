"""
Learning engine — records model predictions and actual outcomes,
drives adaptive calibration through six mechanisms:

  1. Team bias            — mean(actual / predicted_λ) per team, per season
  2. Multi-dim bias       — same metric sliced by home/away, optionally also by month
  3. Kalman filter        — tracks each team's true run rate as a hidden state
  4. Platt recalibration  — weekly logistic-regression fit on p_home → home_won
  5. Platt-2D             — expanding-window fit on p_home + market_prob → home_won,
                             shrinks edge toward the market to fix high-edge overconfidence
  6. Pipeline weights     — gradient-descent scaling on each engine's λ adjustment

Tables (all in predictions_history.db):
  game_outcomes   — one row per prediction; actual runs filled in post-game
  kalman_state    — Kalman filter state (x_est, p_est) per team/context/season
  ml_state        — key-value store for biases, Platt params, pipeline weights
"""

import json
import logging
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_MLB_API_BASE   = "https://statsapi.mlb.com/api/v1"
_MIN_SAMPLES    = 10
# Recorte del sesgo de equipo. NO SUBIR NI BAJAR sin pasar el gate de ROI — ver
# `audit_20260714/paso11/reporte.md`.
#
# Se midió (2026-07-28) qué pasa al neutralizarlo (_BIAS_CLAMP = 0.0). Por Brier
# parecía excelente: 0.24624 → 0.24501, y la ventaja del modelo sobre el azar de
# 1.51% a 2.00% (+32% relativo). El ROI dijo lo contrario, en los CINCO umbrales y
# cada vez peor cuanto más alto el edge:
#
#     edge>=0%   -2.05% -> -3.47%      edge>=8%   -5.58% -> -10.59%
#     edge>=5%   -1.63% -> -6.72%      edge>=10%  -6.69% -> -19.89%
#
# Y el conteo de apuestas se derrumbó (edge>=8%: 1034 -> 642). La lectura: el Brier
# mide la calidad PROMEDIO de las probabilidades; el ROI mide la COLA, los juegos
# donde el modelo cree apartarse del mercado. Quitar el sesgo volvió al modelo más
# "promedio" — mejor calibrado en el centro, incapaz de encontrar dónde apostar.
# Para un sistema que sólo apuesta con edge, es el intercambio exactamente
# equivocado.
#
# Nota sobre su fuerza actual: desde que `_KALMAN_BLEND` pasó a 0.0 (paso 10), el
# amortiguamiento `raw/((1-B)+B·raw)` es identidad, así que este sesgo se aplica
# CRUDO — más fuerte que antes (un sesgo de 1.20 se aplicaba como 1.1215, ahora
# como 1.2000). Es correcto (sin Kalman no hay solapamiento que remover) pero hay
# que saberlo al leer el baseline: 0.24624 es el neto de quitar un canal de
# corrección-hacia-el-resultado y reforzar el otro.
_BIAS_CLAMP     = 0.30
# A team plays at most one game/day, so a bias cache shorter than 24h buys
# nothing — there's no new data to pick up until that day's game is scored.
_BIAS_CACHE_HRS = 24

# Kalman filter hyper-parameters
_KF_Q = 0.025   # process noise variance (team quality changes ~0.16 R/G per game)
_KF_R = 9.0     # observation noise variance (game-to-game σ ≈ 3 R/G)

# Home-runs walk-off truncation correction (added 2026-07-11, audit finding).
# montecarlo/simulator.py draws full-game Poisson/NB for both teams with no
# bottom-9th walk-off logic, but real box scores ARE truncated: the home
# team skips its half-inning once already leading. Empirically (2024-2025,
# 4,830 games): mean(actual_home_runs / lambda_home) = 0.964-0.969, while
# mean(actual_away_runs / lambda_away) = 0.989-0.994 — the away side has no
# equivalent truncation (it always completes the innings it's due to bat).
# Every learner below (Kalman, team-bias, multidim-bias, gradient-descent)
# trains against raw observed runs, so without this correction they read
# the truncation gap as a real "home is overrated" signal and fight any
# upstream fix (e.g. hfa_engine.py's _UNIFORM_HOME_MULT) that corrects
# lambda_home toward the untruncated latent rate the simulator needs.
# Applied ONLY to values feeding these learners — never to the stored
# actual_home_runs column (kept as true ground truth for accuracy/Brier
# grading) and never to home_won (win/loss isn't affected by how many
# runs were left unscored in a truncated bottom 9th).
_HOME_RUNS_TRUNCATION_FACTOR = 0.967   # empirical mean of 0.964-0.969, 2024-2025


# CHRON-001 fix (2026-07-17, audit_20260714/, roadmap Step 1): single source
# of truth mapping a logical prediction field to its actual game_outcomes
# column, per provenance. 'live' is the pre-existing column set (lambda_home,
# p_home, ...) that record_prediction() authors and that every learning
# function below defaulted to reading directly. 'backtest' is the shadow
# column set (backtest_lambda_home, backtest_p_home, ...) that
# backtest_and_retrain.py::update_game_outcomes() now writes to instead of
# clobbering the live columns — see that function's docstring. Every
# function in this file that reads a prediction column from game_outcomes
# takes a `prediction_source: str = "live"` parameter and resolves column
# names through this map, so a backtest run reads only what it just wrote
# (never a live game's real prediction) while the live path's behavior is
# provably unchanged (same column names as before this fix).
_PREDICTION_COLUMNS: Dict[str, Dict[str, str]] = {
    "live": {
        "lambda_home":        "lambda_home",
        "lambda_away":        "lambda_away",
        "p_home":             "p_home",
        "p_away":             "p_away",
        "p_home_raw":         "p_home_raw",
        "p_away_raw":         "p_away_raw",
        "stage_factors_json": "stage_factors_json",
    },
    "backtest": {
        "lambda_home":        "backtest_lambda_home",
        "lambda_away":        "backtest_lambda_away",
        "p_home":             "backtest_p_home",
        "p_away":             "backtest_p_away",
        "p_home_raw":         "backtest_p_home_raw",
        "p_away_raw":         "backtest_p_away_raw",
        "stage_factors_json": "backtest_stage_factors_json",
    },
}


def _pred_col(prediction_source: str, logical_name: str) -> str:
    """Resolve a logical prediction field to its real column name for this source."""
    try:
        return _PREDICTION_COLUMNS[prediction_source][logical_name]
    except KeyError:
        raise ValueError(
            f"unknown prediction_source={prediction_source!r} or "
            f"logical column={logical_name!r}"
        )


def _l0_ratio(
    actual_runs: float,
    lambda_final: float,
    is_home: bool,
) -> Optional[float]:
    """Compute O/λ_final for one game-side (untruncated on the home side).

    MATH-001 fix (roadmap Step 3, 2026-07): dropped the `stage_factors_json`
    parameter — vestigial from the two reverted denominator-change attempts
    below (attempts 1/2 needed it to extract a logged L0 value; the kept,
    final version never reads it). See the postmortem immediately below,
    preserved verbatim — it is why no third attempt should touch this
    function without new information.

    2026-07-11/12 postmortem, kept for anyone reading git blame or old
    session notes — two attempts were made this session to change this
    function's denominator, both reverted after real backtest regressions:

    Attempt 1 (2026-07-11) divided by a logged "L0" value meant to be
    "pre-bias, pre-Kalman λ" — actually captured AFTER Kalman and BEFORE
    the 6 downstream engines, so it implicitly attributed those engines'
    entire effect to "team bias" (a real double-count with e.g.
    hfa_engine.py's uniform home multiplier). Backtest Brier regressed
    0.24479 → 0.24550.

    Attempt 2 (2026-07-12) divided out only the bias's own already-applied
    multiplier (`bias_on_*_lambda`), leaving Kalman and the 6 engines in
    the denominator — correct in isolation, but it converted the bias
    estimator from closed-loop (this function's ratio already nets out any
    PREVIOUSLY applied bias, so an overshoot shows up as a sub-1.0 ratio
    next time and self-corrects) to open-loop (dividing the bias back out
    means an overshoot never appears in the signal and is never damped).
    Combined with `compute_multidim_bias`'s small samples (min_samples=8,
    ±24% standard error on a ratio-of-means at that n) and the dampening
    formula's removal (see compute_team_bias_kalman_adjusted), this applied
    high-variance noise at full gain every time. Backtest Brier regressed
    further, to 0.24636 — worse than attempt 1.

    Conclusion: plain O/λ_final (this function, as currently implemented)
    is a closed-loop residual — it already nets out whatever bias was
    applied last time, so it re-measures only what's LEFT uncorrected, and
    naturally self-extinguishes rather than compounding. Combined with
    compute_team_bias_kalman_adjusted's dampening, the net effect functions
    as implicit shrinkage appropriate to the real signal-to-noise of 8-30
    game slices — a good empirical-Bayes-style estimator that a "more
    exact" derivation (attempts 1 and 2) accidentally broke each time by
    removing dampening or feedback. If this is revisited, the correct
    framing is "does EXPLICIT, tuned shrinkage on a net-of-bias ratio beat
    this implicit-shrinkage design" — an experiment requiring its own
    validation, not a one-line "fix."
    """
    if not lambda_final or lambda_final <= 0:
        return None
    numer = untruncate_home_runs(actual_runs) if is_home else actual_runs
    return numer / lambda_final


def untruncate_home_runs(home_runs: float) -> float:
    """Inflate observed (walk-off-truncated) home runs toward the latent,
    full-game-equivalent rate that lambda_home is meant to represent.
    Use at every point where actual_home_runs becomes a LEARNING target
    (Kalman "offense_home"/"defense_away" roles, team-bias/multidim-bias
    home-side ratios, gradient-descent's home-role target) — not for
    grading, storage, or the away side (which isn't truncated)."""
    return home_runs / _HOME_RUNS_TRUNCATION_FACTOR

# Platt recalibration
_PLATT_MIN_SAMPLES  = 50
_PLATT_RECAL_DAYS   = 7
_PLATT_A_DEFAULT    = 1.0
_PLATT_B_DEFAULT    = 0.0

# Platt-2D (edge-vs-outcome) recalibration — see AUDIT_FINDINGS.md Value
# Detector BUG #2. Fit on an EXPANDING multi-season window (not per-season
# like the 1D fit above) because the >10%-edge bucket this exists to fix has
# only ~150-160 games per season, too thin to refit reliably in isolation.
_PLATT2D_MIN_SAMPLES = 400
_PLATT2D_RECAL_DAYS  = 7

# LEARN-002 fix (audit_20260714/, roadmap Step 2 Commit C) — "is calibration
# alive" monitor. Three independent, confirmed silent-calibration-failure
# incidents predate this: REG-001 (team-bias look-ahead leak), REG-003
# (Platt reset to identity by an interrupted concurrent backtest launch),
# REG-007 (the live odds fetch's REGION/bookmaker-key bug that silently
# disabled Platt-2D correction in production for months, zero errors). None
# of the three would have tripped any alarm — this constant pair is that
# alarm's threshold, calibrated conservatively (a real mechanism should be
# active on the vast majority of predictions, not just >5%; 5% is "so low
# it can only mean broken," not a tuned optimum).
_CALIBRATION_HEALTH_MIN_ROWS      = 20    # below this, a sample-noise false alarm is as likely as a real one
_CALIBRATION_HEALTH_PCT_THRESHOLD = 5.0   # percent

# Kalman blend fraction — must be identical in both get_kalman_lambda_adjustment
# and compute_team_bias_kalman_adjusted so the bias dampening formula matches
# the actual blend applied.
# Fracción de la λ que se toma del Kalman de carreras observadas.
#
# 0.35 → 0.0 el 2026-07-28 (auditoría paso 10, decisión del dueño con la
# evidencia de abajo; reporte completo en `audit_20260714/paso10/reporte.md`).
#
# EL PROBLEMA: el Kalman observa CARRERAS REALES ANOTADAS, y el TTE que produce
# la λ que se está corrigiendo es una estimación MERECIDA (xwOBA/barrel/
# disciplina) que por construcción filtra la suerte en pelotas en juego. Tirar
# esa λ hacia el resultado real le devuelve exactamente lo que el filtrado había
# quitado. Segundo mecanismo, independiente: λ_base es NEUTRA DE PARQUE por
# diseño, las carreras reales no lo son, y el factor de parque se vuelve a
# aplicar en el PASO 5 del pipeline.
#
# MEDIDO POR DOS CAMINOS. Datos vivos, 236 obs (30 equipos × 4 cortes × 2
# temporadas), recursión de Kalman real, prediciendo carreras/juego del resto de
# temporada:  w=0.00 r=+0.5303 | 0.15 +0.5267 | 0.35 +0.4713 | 1.00 +0.3455 —
# monótona decreciente. Backtest completo, tres corridas de 4.825 juegos:
#
#     w=0.35   Brier 0.24675   vs azar 1.300%   accuracy 55.05%
#     w=0.15   Brier 0.24642   vs azar 1.430%   accuracy 54.86%
#     w=0.00   Brier 0.24624   vs azar 1.510%   accuracy 54.84%
#
# Brier y log-loss mejoran monótonamente (+16.2% de ventaja sobre el azar en
# términos relativos); la accuracy cae 0.21pp. El criterio elegido son las reglas
# de puntuación propias y no la accuracy, porque acá no se apuesta a quién gana
# sino a que `p × cuota > 1` — lo que importa es que la probabilidad esté bien,
# no acertar por encima del 50%.
#
# ⚠️ LA SALVEDAD, que no se borra: la ganancia de Brier está concentrada en 2024
# (-0.00095) con 2025 casi plano (-0.00008) — una temporada aporta ~92%. La
# pérdida de accuracy, en cambio, es consistente en las dos. Si una tercera
# temporada no reproduce la ganancia, este cambio es candidato a revisarse.
#
# EFECTOS COLATERALES, verificados:
#  - El amortiguamiento del sesgo de equipo usa esta misma constante
#    (`compute_team_bias_kalman_adjusted`): con 0.0 el denominador queda en 1.0 y
#    devuelve el sesgo crudo intacto, que es lo correcto — sin Kalman no hay
#    solapamiento que remover. No toca la parte frágil de esa fórmula.
#  - `update_kalman` SIGUE corriendo y manteniendo el estado. Es barato, deja el
#    camino abierto para revertir, y su `n_obs` alimenta la confianza (ver
#    `get_kalman_n_obs`), donde sigue siendo un indicador válido de "cuántos
#    datos de esta temporada tenemos de este equipo" aunque ya no corrija λ.
_KALMAN_BLEND = 0.0

# Pipeline gradient-descent weights — one per engine stage. These stage
# names are combined with a role ("home"/"away") into the actual
# stage_factors_json key as "{stage}_on_{role}_lambda" (see _gradient_step) —
# must match what run_module.py and backtest_and_retrain.py emit. Add a
# stage here only when its factors are actually recorded; otherwise it stays
# at 1.0 and wastes gradient cycles.
_STAGE_KEYS = [
    "park",         # Park + Weather engine
    "hfa",          # Home Field Advantage engine
    "defense",      # Defensive Efficiency engine
    "pitcher",      # Pitcher engine
    "bullpen",      # Bullpen engine
    "context",      # Contextual engine (rest / B2B / umpire)
]
_LR         = 0.01    # gradient step size
_MIN_WEIGHT = 0.30    # floor: never fully bypass a stage
_MAX_WEIGHT = 1.50    # ceiling


class LearningEngine:
    """
    Adaptive calibration engine:
      record_prediction()       → called before the game
      fetch_pending_outcomes()  → auto-fetches actual scores for past games
      update_outcome()          → fill in score + trigger all adaptive updates
      compute_team_bias()       → per-team λ correction (cached)
      compute_multidim_bias()   → correction sliced by home/away, month, venue
      get_kalman_estimate()     → Kalman-filtered team run rate
      get_platt_params()        → current (a, b) for probability shrinkage
      recalibrate_platt()       → refit logistic regression from outcomes
      get_pipeline_weights()    → learned stage weights for λ blending
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_tables(self) -> None:
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS game_outcomes (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_pk          INTEGER NOT NULL UNIQUE,
                    game_date        TEXT    NOT NULL,
                    season           INTEGER NOT NULL,
                    home_team        TEXT    NOT NULL,
                    away_team        TEXT    NOT NULL,
                    venue            TEXT,
                    month            INTEGER,
                    lambda_home      REAL,
                    lambda_away      REAL,
                    p_home           REAL,
                    p_away           REAL,
                    ml_home_pin      REAL,
                    ml_away_pin      REAL,
                    stage_factors_json TEXT,
                    actual_home_runs INTEGER,
                    actual_away_runs INTEGER,
                    home_won         INTEGER,
                    created_at       TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_go_home_team
                    ON game_outcomes(home_team, season);
                CREATE INDEX IF NOT EXISTS idx_go_away_team
                    ON game_outcomes(away_team, season);
                CREATE INDEX IF NOT EXISTS idx_go_pending
                    ON game_outcomes(game_date)
                    WHERE actual_home_runs IS NULL;

                CREATE TABLE IF NOT EXISTS kalman_state (
                    team       TEXT    NOT NULL,
                    context    TEXT    NOT NULL,
                    season     INTEGER NOT NULL,
                    x_est      REAL    NOT NULL,
                    p_est      REAL    NOT NULL,
                    n_obs      INTEGER DEFAULT 0,
                    updated_at TEXT    NOT NULL,
                    PRIMARY KEY (team, context, season)
                );

                CREATE TABLE IF NOT EXISTS ml_state (
                    key          TEXT    NOT NULL,
                    scope        TEXT    NOT NULL,
                    season       INTEGER NOT NULL,
                    value_json   TEXT    NOT NULL,
                    sample_count INTEGER DEFAULT 0,
                    updated_at   TEXT    NOT NULL,
                    PRIMARY KEY (key, scope, season)
                );
            """)
            # Migrate: add columns silently if they don't exist yet
            for col, typedef in [
                ("venue", "TEXT"),
                ("month", "INTEGER"),
                ("stage_factors_json", "TEXT"),
                ("p_home_raw", "REAL"),   # pre-Platt MC probability (for clean Platt refitting)
                ("p_away_raw", "REAL"),
                # Pinnacle moneyline price at prediction time — without these,
                # recalibrate_platt_2d()'s query (which reads them) raises
                # OperationalError on a fresh DB, silently caught by its
                # caller, permanently disabling Platt-2D with only a log
                # warning. On this repo's actual DB these already exist
                # (added ad-hoc by fetch_historical_odds.py for backfilled
                # historical seasons) — this ALTER is then a no-op, caught
                # below same as any other already-present column.
                ("ml_home_pin", "REAL"),
                ("ml_away_pin", "REAL"),
                # CHRON-001 fix (2026-07-17, audit_20260714/08_chronology_audit.md):
                # game_outcomes used to be shared, unprotected, between live-production
                # writes (record_prediction()/update_outcome()) and backtest overwrites
                # (backtest_and_retrain.py::update_game_outcomes(), a plain
                # UPDATE...WHERE game_pk=? with no provenance guard) — a routine backtest
                # run touching a reconciled live game silently destroyed the live
                # prediction record with no audit trail. `source` records who authored
                # the CURRENT values in the live prediction columns (lambda_home, p_home,
                # etc.) and is set once, never flipped. The backtest_* columns are where
                # update_game_outcomes() now writes instead of clobbering the live ones —
                # see its docstring and get_learning_rows()/apply-side usage below.
                # backtest_run_at itself is normally added by backtest_and_retrain.py's
                # own _add_backtest_col() — duplicated here (same no-op-if-present
                # pattern as ml_home_pin/ml_away_pin above) because _backfill_chron001_
                # source() below needs it to exist unconditionally, including on a
                # fresh DB that a live-only deployment created without ever running a
                # backtest.
                ("backtest_run_at", "TEXT"),
                ("source", "TEXT"),
                ("backtest_lambda_home", "REAL"),
                ("backtest_lambda_away", "REAL"),
                ("backtest_p_home", "REAL"),
                ("backtest_p_away", "REAL"),
                ("backtest_p_home_raw", "REAL"),
                ("backtest_p_away_raw", "REAL"),
                ("backtest_stage_factors_json", "TEXT"),
                # Fase 2B commit B1 (audit_20260714/fase2b/): game_date is the
                # raw UTC gameDate timestamp truncated to a date — for any
                # night game crossing midnight UTC (the norm for west-coast
                # teams), that's one calendar day AHEAD of the MLB schedule's
                # own officialDate. Every PIT walk-forward cutoff derived
                # from game_date was therefore requesting one day too late,
                # and with PITCache.get_latest()'s inclusive `<=` comparison,
                # silently including the target game's own game-day in its
                # own PIT snapshot — a confirmed real leak (see
                # audit_20260714/verificacion_operativa/reporte.md, V4, and
                # b0_verificacion_previa.md for the empirical proof against
                # game_pk=745199). This column is the correct field for any
                # future chronological comparison; game_date itself is left
                # untouched (still needed wherever the actual UTC start time
                # matters, e.g. weather-forecast fetch alignment).
                ("official_date", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE game_outcomes ADD COLUMN {col} {typedef}")
                except sqlite3.OperationalError:
                    pass
            self._backfill_chron001_source(conn)
            self._migrate_chron002_state_source(conn)

    def _migrate_chron002_state_source(self, conn: sqlite3.Connection) -> None:
        """CHRON-002 fix (roadmap Step 2, Commit A): ml_state and kalman_state
        used to be shared, unprotected, between live and backtest — a
        backtest refit of Platt/team-bias/pipeline-weights/Kalman state
        overwrote the exact keys production reads, the residual CHRON-001
        left open (see CONTRACTS.md's game_outcomes entry). Adds a
        `state_source` column to both tables' PRIMARY KEY.

        SQLite can't ALTER a PRIMARY KEY, so this rebuilds each table
        (CREATE new schema -> copy -> DROP old -> RENAME) — a one-time
        operation, guarded by checking whether `state_source` is already a
        column (idempotent: every LearningEngine() instantiation, live and
        backtest alike, calls this, so the guard must make every call after
        the first a cheap no-op).

        Migration is copy-to-both, not heuristic backfill: every existing
        row is duplicated into state_source='live' AND state_source=
        'backtest' with byte-identical values. Historical provenance of a
        given key's value is exactly as irreconstructible as it was for
        game_outcomes (CHRON-001) — copying to both guarantees zero
        behavior change at cutover: whichever path reads a key next gets
        precisely what was there before this migration, and the two
        namespaces only diverge from new writes made after this point.
        """
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ml_state)")}
        if "state_source" not in cols:
            conn.executescript(
                """
                CREATE TABLE ml_state_chron002_new (
                    key          TEXT    NOT NULL,
                    scope        TEXT    NOT NULL,
                    season       INTEGER NOT NULL,
                    state_source TEXT    NOT NULL,
                    value_json   TEXT    NOT NULL,
                    sample_count INTEGER DEFAULT 0,
                    updated_at   TEXT    NOT NULL,
                    PRIMARY KEY (key, scope, season, state_source)
                );
                INSERT INTO ml_state_chron002_new
                    (key, scope, season, state_source, value_json, sample_count, updated_at)
                    SELECT key, scope, season, 'live', value_json, sample_count, updated_at
                    FROM ml_state;
                INSERT INTO ml_state_chron002_new
                    (key, scope, season, state_source, value_json, sample_count, updated_at)
                    SELECT key, scope, season, 'backtest', value_json, sample_count, updated_at
                    FROM ml_state;
                DROP TABLE ml_state;
                ALTER TABLE ml_state_chron002_new RENAME TO ml_state;
                """
            )

        cols = {r[1] for r in conn.execute("PRAGMA table_info(kalman_state)")}
        if "state_source" not in cols:
            conn.executescript(
                """
                CREATE TABLE kalman_state_chron002_new (
                    team         TEXT    NOT NULL,
                    context      TEXT    NOT NULL,
                    season       INTEGER NOT NULL,
                    state_source TEXT    NOT NULL,
                    x_est        REAL    NOT NULL,
                    p_est        REAL    NOT NULL,
                    n_obs        INTEGER DEFAULT 0,
                    updated_at   TEXT    NOT NULL,
                    PRIMARY KEY (team, context, season, state_source)
                );
                INSERT INTO kalman_state_chron002_new
                    (team, context, season, state_source, x_est, p_est, n_obs, updated_at)
                    SELECT team, context, season, 'live', x_est, p_est, n_obs, updated_at
                    FROM kalman_state;
                INSERT INTO kalman_state_chron002_new
                    (team, context, season, state_source, x_est, p_est, n_obs, updated_at)
                    SELECT team, context, season, 'backtest', x_est, p_est, n_obs, updated_at
                    FROM kalman_state;
                DROP TABLE kalman_state;
                ALTER TABLE kalman_state_chron002_new RENAME TO kalman_state;
                """
            )

    def _backfill_chron001_source(self, conn: sqlite3.Connection) -> None:
        """One-time backfill for the CHRON-001 fix (audit_20260714/, roadmap Step 1).

        Idempotent: every statement is gated on `source IS NULL`, so calling
        this on every LearningEngine instantiation (live AND backtest both
        construct one) is a cheap no-op once the backfill has run once.

        Semantics — deliberately ordered, each step only touching rows the
        previous step left with `source IS NULL`:
          1. season=2026 AND backtest_run_at IS NULL -> 'live'
             (the pure live-production rows, never touched by a backtest).
          2. backtest_run_at IS NOT NULL -> 'backtest', and their CURRENT
             live-prediction-column values are copied into the new
             backtest_* columns. Provenance for these rows was already lost
             before this fix existed — see
             audit_20260714/chron001_forensics_report.md (0 of 563 rows
             recoverable from any backup on disk) — this just records that
             their live columns are backtest-authored so FASE 4's read path
             finds the (already-backtest) data where it expects it, and so
             a *future* backtest run editing these rows again writes only
             to backtest_* rather than re-touching lambda_home/p_home.
          3. Everything left (older-season bulk imports that never went
             through record_prediction() or a backtest run) -> 'import'.
        """
        conn.execute(
            "UPDATE game_outcomes SET source = 'live' "
            "WHERE source IS NULL AND season = 2026 AND backtest_run_at IS NULL"
        )
        conn.execute(
            """
            UPDATE game_outcomes SET
                source = 'backtest',
                backtest_lambda_home = lambda_home,
                backtest_lambda_away = lambda_away,
                backtest_p_home = p_home,
                backtest_p_away = p_away,
                backtest_p_home_raw = p_home_raw,
                backtest_p_away_raw = p_away_raw,
                backtest_stage_factors_json = stage_factors_json
            WHERE source IS NULL AND backtest_run_at IS NOT NULL
            """
        )
        conn.execute(
            "UPDATE game_outcomes SET source = 'import' WHERE source IS NULL"
        )

    # ------------------------------------------------------------------
    # Prediction recording
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        game_pk: int,
        game_date: str,
        season: int,
        home_team: str,
        away_team: str,
        lambda_home: float,
        lambda_away: float,
        p_home: float,
        p_away: float,
        venue: Optional[str] = None,
        stage_factors: Optional[Dict[str, Any]] = None,
        p_home_raw: Optional[float] = None,  # pre-Platt MC probability
        p_away_raw: Optional[float] = None,
        ml_home_pin: Optional[float] = None,  # Pinnacle moneyline price at prediction time
        ml_away_pin: Optional[float] = None,
        official_date: Optional[str] = None,
    ) -> bool:
        """Insert a pre-game prediction row. Returns True if newly inserted.

        p_home / p_away  — post-Platt calibrated probabilities (for display).
        p_home_raw / p_away_raw — raw Monte Carlo probabilities before Platt
            scaling.  recalibrate_platt() prefers these so Platt is fitted on
            its own input signal rather than its own output (circular dependency).
        ml_home_pin / ml_away_pin — Pinnacle's moneyline price, if fetched for
            this game. Without persisting these, recalibrate_platt_2d()'s
            expanding-window query (season < ? AND ml_home_pin IS NOT NULL)
            can only ever see backfilled historical seasons — no live season
            ever accumulates enough rows to be included in a future refit,
            regardless of how long the app runs.

        For rows that already exist (historical bulk imports), we backfill any
        NULL fields that we now have values for — crucially stage_factors_json
        (required by gradient descent) and p_home_raw (required by clean Platt
        refitting).  INSERT OR IGNORE alone would silently skip this, leaving
        gradient descent permanently starved of stage factor data.

        Pin-backfill semantics (2026-07-19, Fase 2A commit 1): the COALESCE
        backfill below also applies to ml_home_pin/ml_away_pin, which in
        practice is its most common trigger — a "tomorrow" game is often
        first analyzed before Pinnacle has posted a line for it, so the
        first call records the prediction with pins NULL, and a later same-
        day re-run (once the line exists) fills them in. This means
        ml_home_pin/ml_away_pin means "the earliest pre-game Pinnacle price
        this pipeline observed for this game", NOT "the price at the exact
        moment p_home/p_away were computed" — those two calls can be hours
        apart. The backfill NEVER touches p_home, p_away, lambda_home, or
        lambda_away (not in the UPDATE below) — a prediction, once recorded,
        is immutable; only telemetry fields fill in gaps afterward.

        official_date (Fase 2B commit B1, audit_20260714/fase2b/): the MLB
        schedule's own officialDate, distinct from game_date (a raw UTC
        timestamp truncated to a date — one day ahead of officialDate for
        any night game crossing midnight UTC). This is the field every PIT
        walk-forward cutoff and chronological bias-window comparison should
        use going forward; game_date itself stays as-is (still needed where
        the actual UTC start time matters, e.g. weather-forecast alignment).
        """
        month = None
        try:
            month = int(game_date[5:7]) if game_date else None
        except (IndexError, ValueError):
            pass

        sf_json = json.dumps(stage_factors) if stage_factors else None

        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO game_outcomes
                    (game_pk, game_date, season, home_team, away_team, venue, month,
                     lambda_home, lambda_away, p_home, p_away,
                     p_home_raw, p_away_raw, ml_home_pin, ml_away_pin, stage_factors_json,
                     official_date, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'live')
                """,
                (game_pk, game_date, season, home_team, away_team, venue, month,
                 lambda_home, lambda_away, p_home, p_away,
                 p_home_raw, p_away_raw, ml_home_pin, ml_away_pin, sf_json, official_date),
            )
            inserted = cursor.rowcount == 1
            if not inserted:
                # Row already exists (e.g. historical bulk import).  Backfill
                # any NULL fields we now have — never overwrite existing data.
                # CHRON-001 (audit_20260714/): source = COALESCE(source, 'live')
                # — this call path (record_prediction) is exclusively the live
                # writer (backtest_and_retrain.py never calls it, only
                # update_game_outcomes(), which never touches `source` at
                # all — see its docstring). So the only way `source` could
                # still be NULL here is a pre-CHRON-001-migration row or a
                # future bulk-import path that didn't set it; in that case
                # this backfill call — by definition live — is the first
                # real authorship information available, so 'live' is
                # correct. If `source` is already set (the normal case,
                # true for every row after the one-time migration backfill),
                # COALESCE leaves it untouched, exactly as "never overwrite
                # existing data" already requires for every other field here.
                conn.execute(
                    """
                    UPDATE game_outcomes SET
                        stage_factors_json = COALESCE(stage_factors_json, ?),
                        p_home_raw         = COALESCE(p_home_raw, ?),
                        p_away_raw         = COALESCE(p_away_raw, ?),
                        ml_home_pin        = COALESCE(ml_home_pin, ?),
                        ml_away_pin        = COALESCE(ml_away_pin, ?),
                        official_date      = COALESCE(official_date, ?),
                        source             = COALESCE(source, 'live')
                    WHERE game_pk = ?
                    """,
                    (sf_json, p_home_raw, p_away_raw, ml_home_pin, ml_away_pin, official_date, game_pk),
                )
        logger.debug(f"[learning] recorded prediction game_pk={game_pk} inserted={inserted}")
        return inserted

    # ------------------------------------------------------------------
    # Outcome fetching
    # ------------------------------------------------------------------

    def fetch_pending_outcomes(self, lookback_days: int = 7) -> int:
        import requests as _req

        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._get_conn() as conn:
            pending = conn.execute(
                """
                SELECT game_pk, game_date FROM game_outcomes
                WHERE actual_home_runs IS NULL
                  AND game_date < ?
                  AND game_date >= ?
                ORDER BY game_date DESC
                """,
                (today, cutoff),
            ).fetchall()

        if not pending:
            return 0

        updated = 0
        session = _req.Session()
        for row in pending:
            try:
                r = session.get(f"{_MLB_API_BASE}/game/{row['game_pk']}/linescore", timeout=8)
                r.raise_for_status()
                data = r.json()
                home_runs = data.get("teams", {}).get("home", {}).get("runs")
                away_runs = data.get("teams", {}).get("away", {}).get("runs")
                if home_runs is None or away_runs is None:
                    continue

                # /linescore alone never says whether the game is actually
                # over — it updates live, mid-game. A West Coast night game
                # still in progress past UTC midnight would otherwise have
                # its PARTIAL score locked in here, and update_outcome()'s
                # own idempotency guard (WHERE actual_home_runs IS NULL)
                # means that wrong score can never be corrected afterward.
                # Same two-step pattern as track_record/reconciler.py's
                # _fetch_mlb_final_v2().
                sched = session.get(
                    f"{_MLB_API_BASE}/schedule?gamePk={row['game_pk']}&hydrate=linescore",
                    timeout=8,
                )
                sched.raise_for_status()
                sdates = sched.json().get("dates", [])
                sgames = sdates[0].get("games", []) if sdates else []
                state = sgames[0].get("status", {}).get("abstractGameState", "") if sgames else ""
                if state != "Final":
                    continue

                if self.update_outcome(row["game_pk"], int(home_runs), int(away_runs)):
                    updated += 1
                    logger.info(f"[learning] game {row['game_pk']}: {away_runs}–{home_runs}")
            except Exception as exc:
                logger.debug(f"[learning] could not fetch {row['game_pk']}: {exc}")

        if updated:
            logger.info(f"[learning] fetched {updated} outcome(s)")
        return updated

    def update_outcome(
        self,
        game_pk: int,
        actual_home_runs: int,
        actual_away_runs: int,
        prediction_source: str = "live",
    ) -> bool:
        """Fill in actual runs, trigger Kalman + gradient-descent updates.

        Idempotent: the UPDATE only matches rows that haven't been scored yet
        (`actual_home_runs IS NULL`), so a second call for an already-scored
        game_pk is a no-op — it returns False rather than silently re-running
        4 Kalman updates and a gradient-descent step for the same game.
        `fetch_pending_outcomes` already avoids this via its own pending-only
        query, but this guard protects any future caller (e.g. a track-record
        reconciler) that doesn't know to check first.

        Returns True only if this call actually scored the game (row existed
        and was previously unscored).

        `prediction_source` — CHRON-001 fix (audit_20260714/): which
        prediction columns _post_outcome_update()'s gradient-descent step
        reads (live vs backtest_*). Actual-runs/home_won columns themselves
        stay shared regardless — ground truth isn't provenance-split, only
        model predictions are. Defaults to 'live'; currently the only real
        caller is fetch_pending_outcomes() (live path — the backtest loop
        calls update_kalman()/_gradient_step() directly, bypassing this
        method entirely), so this default is a no-op change in practice,
        added for defensive consistency with every other learning function
        in this file.
        """
        home_won = 1 if actual_home_runs > actual_away_runs else 0

        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE game_outcomes
                SET actual_home_runs = ?,
                    actual_away_runs = ?,
                    home_won         = ?
                WHERE game_pk = ? AND actual_home_runs IS NULL
                """,
                (actual_home_runs, actual_away_runs, home_won, game_pk),
            )
            existed = cursor.rowcount > 0

        if existed:
            self._post_outcome_update(game_pk, actual_home_runs, actual_away_runs, prediction_source)
        return existed

    def _post_outcome_update(
        self,
        game_pk: int,
        home_runs: int,
        away_runs: int,
        prediction_source: str = "live",
    ) -> None:
        """Update Kalman states and maybe trigger Platt recalibration."""
        lam_home_col = _pred_col(prediction_source, "lambda_home")
        lam_away_col = _pred_col(prediction_source, "lambda_away")
        with self._get_conn() as conn:
            row = conn.execute(
                f"SELECT home_team, away_team, season, "
                f"{lam_home_col} AS lambda_home, {lam_away_col} AS lambda_away "
                f"FROM game_outcomes WHERE game_pk = ?",
                (game_pk,),
            ).fetchone()
        if not row:
            return

        season = row["season"]
        self.update_kalman(row["home_team"], "offense_home", season, untruncate_home_runs(float(home_runs)), prediction_source)
        self.update_kalman(row["away_team"], "offense_away", season, float(away_runs), prediction_source)
        self.update_kalman(row["home_team"], "defense_home", season, float(away_runs), prediction_source)
        self.update_kalman(row["away_team"], "defense_away", season, untruncate_home_runs(float(home_runs)), prediction_source)

        # Gradient-descent weight update
        if row["lambda_home"] and row["lambda_away"]:
            self._gradient_step(game_pk, home_runs, away_runs, season, prediction_source)

    # ------------------------------------------------------------------
    # Team bias (simple 1-D)
    # ------------------------------------------------------------------

    def compute_team_bias(
        self,
        team: str,
        season: int,
        min_samples: int = _MIN_SAMPLES,
        before_date: Optional[str] = None,
        prediction_source: str = "live",
    ) -> float:
        """mean(actual_runs / predicted_λ) for the team. Returns 1.0 when insufficient data.

        `prediction_source` — CHRON-001 fix (audit_20260714/): selects which
        game_outcomes columns (live vs backtest_*) supply lambda_home/
        lambda_away/stage_factors_json. Also selects the ml_state
        `state_source` namespace for the team_bias cache below (CHRON-002
        fix, roadmap Step 2 Commit A) — closes the residual CHRON-001 left
        open, where a source-unaware cache meant a backtest's final
        cache-populating call could still leave a value a live call
        (before_date=None) would read. The cache is also still bypassed
        whenever `before_date` is given (see below), which is how
        backtest_and_retrain.py's per-game walk-forward calls avoid it
        entirely regardless.

        FIX (2026-07-08): added `before_date` — a walk-forward cutoff (exclusive,
        "YYYY-MM-DD"). Without it, this query only filters by season, so a
        backtest replay of an early-season game (e.g. Opening Day) pulls in
        actual_home_runs from every OTHER game of that team's season already
        sitting in game_outcomes — including games chronologically AFTER the
        one being predicted, since the backtest loader pre-populates the whole
        season's ground truth before the per-game loop starts. Confirmed via
        isolated repro: a team with neutral early games and extreme "future"
        games produced bias=1.3 (the clamp ceiling) when queried as of its
        first game of the season. When before_date is given, the 24h ml_state
        cache is bypassed entirely (the cache has no date dimension, so a
        cached value from one as-of-date is not valid for another) and the
        result is computed fresh from the date-bounded query every call —
        this is the correct walk-forward behavior for a backtest replay.
        Live callers (before_date=None) keep the original cached, season-wide
        behavior unchanged — there's no leak risk live since future games
        genuinely have no actual_home_runs yet.
        """
        if before_date is None:
            cache_key = f"team_bias:{team}"
            cached = self.load_state(cache_key, "team_bias", season, prediction_source)
            if cached and cached.get("sample_count", 0) >= min_samples:
                if self._hours_since(cached.get("updated_at", "")) < _BIAS_CACHE_HRS:
                    return float(cached["bias"])

        lam_home_col = _pred_col(prediction_source, "lambda_home")
        lam_away_col = _pred_col(prediction_source, "lambda_away")
        query = f"""
            SELECT home_team, away_team,
                   {lam_home_col} AS lambda_home, {lam_away_col} AS lambda_away,
                   actual_home_runs, actual_away_runs
            FROM game_outcomes
            WHERE season = ?
              AND actual_home_runs IS NOT NULL
              AND (home_team = ? OR away_team = ?)
        """
        params: List[Any] = [season, team, team]
        if before_date is not None:
            # Fase 2B commit B2 (audit_20260714/fase2b/): compare against
            # official_date, not game_date — before_date is now itself an
            # official_date (see backtest_and_retrain.py's
            # _bias_before_date), and mixing bases (official on one side,
            # raw UTC-timestamp game_date on the other) would silently
            # reintroduce the same leak this whole phase exists to close.
            # COALESCE falls back to game_date only for a row the Fase 2B
            # backfill couldn't resolve (expected to be zero).
            query += " AND COALESCE(official_date, game_date) < ?"
            params.append(before_date)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        if len(rows) < min_samples:
            return 1.0

        ratios = []
        for r in rows:
            if r["home_team"] == team:
                ratio = _l0_ratio(r["actual_home_runs"], r["lambda_home"], is_home=True)
            elif r["away_team"] == team:
                ratio = _l0_ratio(r["actual_away_runs"], r["lambda_away"], is_home=False)
            else:
                continue
            if ratio is not None:
                ratios.append(ratio)

        if not ratios:
            return 1.0

        raw  = sum(ratios) / len(ratios)
        bias = max(1.0 - _BIAS_CLAMP, min(1.0 + _BIAS_CLAMP, raw))
        if before_date is None:
            self.save_state(
                cache_key, "team_bias", {"bias": bias}, len(ratios), season,
                prediction_source=prediction_source,
            )
        logger.debug(f"[learning] {team} bias={bias:.4f} (n={len(ratios)})")
        return bias

    def compute_team_bias_kalman_adjusted(
        self,
        team: str,
        season: int,
        context: str,
        month: Optional[int] = None,
        before_date: Optional[str] = None,
        prediction_source: str = "live",
    ) -> float:
        """Team bias dampened by Kalman's fractional coverage to prevent double-correction.

        Both Kalman and team bias draw from the same actual-vs-predicted history.
        When Kalman is active it already corrects `blend` (35%) of the model error;
        applying the raw bias on top overcorrects that share of the signal.

        Dampening formula:
          adjusted = raw_bias / (1 − blend + blend × raw_bias)

        2026-07-11/12 postmortem (kept — this constant's exact theoretical
        justification is wrong, but the mechanism itself is empirically
        load-bearing, confirmed by two real regressions this session):
        this formula's original derivation claimed `raw_bias` measures
        O/L0 with L0 the pre-bias, PRE-KALMAN λ — that premise was never
        actually true (`raw_bias` comes from `compute_multidim_bias`,
        whose ratio always divides by the game's FINAL λ via `_l0_ratio`,
        i.e. post-Kalman, post-bias, post all 6 downstream engines, not
        L0). Two same-day attempts to "fix" this premise by changing
        `_l0_ratio`'s denominator — first to a mislabeled L0 value, then to
        λ_final with only the bias's own contribution divided out — both
        produced measured backtest Brier regressions (0.24479 → 0.24550 →
        0.24636) and were reverted. Root cause of both regressions: this
        function's dampening, combined with `_l0_ratio`'s use of the FINAL
        λ (which already nets out any previously-applied bias), forms a
        closed-loop, implicitly-shrunk residual estimator appropriate to
        the real signal-to-noise of small per-team slices (min_samples=8,
        ±24% standard error on the ratio-of-means at that n) — removing
        either the dampening or the closed-loop property converts it into
        an open-loop, high-variance estimator that injects noise at full
        gain. So: keep this formula exactly as originally written, despite
        its stated derivation being false — it is functioning as intended
        empirically, just for a different reason than documented. A future
        session revisiting this should treat it as "does explicit, tuned
        shrinkage beat this implicit-shrinkage design" — an experiment
        requiring its own validation — not a one-line derivation fix.

        Example — team scores 3.5 R/G, model says 4.5 (raw_bias = 0.778):
          Kalman (blend=0.35): 0.65×4.5 + 0.35×3.5 = 4.15  (−7.8%)
          Old bias on 4.15:    4.15 × 0.778 = 3.23          (−22% extra → total −28%)
          Dampened bias: 0.778 / (0.65 + 0.35×0.778) = 0.8434
          New bias on 4.15:    4.15 × 0.8434 = 3.50          (−15.7% → total ≈ exact)

        FIX C1: compute_team_bias mezclaba home+away juegos en una sola media,
        contaminando la señal del Kalman para equipos con asimetría HFA real.
        compute_multidim_bias filtra por contexto (home vs away games separados).
        Esto resuelve el bucket 40-45% donde equipos como Atlanta (Bias_home=0.93
        vs Bias_away=1.08) tenían su señal correcta neutralizada por el bias
        agregado (~1.005). El dampening se mantiene igual: ambas fuentes (Kalman
        y multidim_bias) siguen usando los mismos datos home-only o away-only.
        """
        # FIX C1: use context-specific bias instead of aggregate home+away mix.
        home_away = "home" if context == "offense_home" else "away"
        raw_bias = self.compute_multidim_bias(
            team, season, home_away, month=month, before_date=before_date,
            prediction_source=prediction_source,
        )
        if raw_bias == 1.0:
            return 1.0

        state = self._get_kalman_state(team, context, season, prediction_source)
        if not state or state["n_obs"] < 10:
            return raw_bias  # Kalman cold-start: no overlap to remove

        denom = (1.0 - _KALMAN_BLEND) + _KALMAN_BLEND * raw_bias
        dampened = raw_bias / denom if abs(denom) > 1e-9 else raw_bias
        logger.debug(
            "[learning] %s/%s bias Kalman-adjusted: raw=%.4f → dampened=%.4f (n_obs=%d)",
            team, home_away, raw_bias, dampened, state["n_obs"],
        )
        return max(1.0 - _BIAS_CLAMP, min(1.0 + _BIAS_CLAMP, dampened))

    # ------------------------------------------------------------------
    # Multi-dimensional bias
    # ------------------------------------------------------------------

    def compute_multidim_bias(
        self,
        team: str,
        season: int,
        home_away: str = "home",       # "home" or "away"
        month: Optional[int] = None,   # 3-10; None = skip this tier, use home_away only
        min_samples: int = 8,
        before_date: Optional[str] = None,
        prediction_source: str = "live",
    ) -> float:
        """
        Bias correction refined along up to two dimensions.

        Priority:
          1. team × home_away × month (most specific — only tried when a real
             month is given; previously this tier was always queried with
             month=None, making it an exact duplicate of tier 2 and leaving
             the month-specific cache key ("...:mNone") permanently unused)
          2. team × home_away
          3. simple team bias (fallback)

        FIX (2026-07-08): `before_date` threads a walk-forward cutoff down to
        `_compute_multidim`'s SQL — see `compute_team_bias`'s docstring for
        the full leak this closes. When given, the ml_state cache (which has
        no date dimension) is bypassed and every tier is computed fresh.
        """
        dims = []
        if month is not None:
            dims.append((f"bias:{team}:{home_away}:m{month}", home_away, month))
        dims.append((f"bias:{team}:{home_away}", home_away, None))

        for scope_key, ha, mo in dims:
            if before_date is None:
                cached = self.load_state(scope_key, "multidim_bias", season, prediction_source)
                if cached and cached.get("sample_count", 0) >= min_samples:
                    if self._hours_since(cached.get("updated_at", "")) < _BIAS_CACHE_HRS:
                        return float(cached["bias"])
            result = self._compute_multidim(
                team, season, ha, mo, min_samples, before_date=before_date,
                prediction_source=prediction_source,
            )
            if result is not None:
                if before_date is None:
                    self.save_state(
                        scope_key, "multidim_bias", {"bias": result[0]}, result[1], season,
                        prediction_source=prediction_source,
                    )
                return result[0]

        # Final fallback: simple team bias
        return self.compute_team_bias(
            team, season, before_date=before_date, prediction_source=prediction_source,
        )

    def _compute_multidim(
        self,
        team: str,
        season: int,
        home_away: str,
        month: Optional[int],
        min_samples: int,
        before_date: Optional[str] = None,
        prediction_source: str = "live",
    ) -> Optional[Tuple[float, int]]:
        is_home = (home_away == "home")
        team_col    = "home_team" if is_home else "away_team"
        lambda_col  = _pred_col(prediction_source, "lambda_home" if is_home else "lambda_away")
        actual_col  = "actual_home_runs" if is_home else "actual_away_runs"

        where = f"season = ? AND actual_home_runs IS NOT NULL AND {team_col} = ?"
        params: List[Any] = [season, team]
        if month is not None:
            where += " AND month = ?"
            params.append(month)
        if before_date is not None:
            # Fase 2B commit B2 — same reasoning as compute_team_bias()'s
            # identical change: before_date is now an official_date, and
            # comparing it against the raw game_date column would mix bases.
            where += " AND COALESCE(official_date, game_date) < ?"
            params.append(before_date)

        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT {lambda_col} AS lambda_val, {actual_col} "
                f"FROM game_outcomes WHERE {where}",
                params,
            ).fetchall()

        if len(rows) < min_samples:
            return None

        ratios = [
            r for r in (
                _l0_ratio(row[actual_col], row["lambda_val"], is_home)
                for row in rows
            )
            if r is not None
        ]
        if not ratios:
            return None

        raw  = sum(ratios) / len(ratios)
        bias = max(1.0 - _BIAS_CLAMP, min(1.0 + _BIAS_CLAMP, raw))
        return bias, len(ratios)

    # ------------------------------------------------------------------
    # Kalman filter
    # ------------------------------------------------------------------

    def reset_kalman_for_seasons(
        self, seasons: List[int], prediction_source: str = "live",
    ) -> int:
        """Delete all Kalman states for the given seasons.

        Called by the backtest before it starts processing so that the
        walk-forward loop begins from a neutral state rather than from
        whatever stale (and potentially look-ahead-biased) states the
        live system accumulated.

        CHRON-002 fix: `prediction_source` scopes the delete to one
        state_source namespace — a backtest reset can never delete
        production's live Kalman states, or vice versa.

        Returns the number of rows deleted.
        """
        if not seasons:
            return 0
        placeholders = ",".join("?" * len(seasons))
        with self._get_conn() as conn:
            cursor = conn.execute(
                f"DELETE FROM kalman_state WHERE season IN ({placeholders}) AND state_source = ?",
                [*seasons, prediction_source],
            )
            deleted = cursor.rowcount
        logger.info(
            "[learning] Kalman reset: deleted %d states for seasons %s (state_source=%s)",
            deleted, seasons, prediction_source,
        )
        return deleted

    def reset_pipeline_weights(
        self, seasons: List[int], prediction_source: str = "live",
    ) -> int:
        """Reset learned pipeline weights to 1.0 for the given seasons.

        Called by the backtest before the walk-forward loop so that weights
        trained on Kalman-look-ahead-biased data are discarded.  Gradient
        descent re-learns optimal weights from clean walk-forward predictions.

        Returns the number of seasons reset.
        """
        if not seasons:
            return 0
        defaults = {k: 1.0 for k in _STAGE_KEYS}
        for season in seasons:
            self.save_state(
                "pipeline_weights", "weights", defaults, 0, season,
                prediction_source=prediction_source,
            )
        logger.info(
            "[learning] Pipeline weights reset to 1.0 for seasons %s (state_source=%s)",
            seasons, prediction_source,
        )
        return len(seasons)

    def reset_platt_params(
        self, seasons: List[int], prediction_source: str = "live",
    ) -> int:
        """Seed identity Platt params (a=1.0, b=0.0) for the given seasons.

        We UPSERT rather than DELETE.  If we deleted, get_platt_params() would
        immediately try to recalibrate_platt() and find all the historical
        home_won rows — fitting Platt on the *old* pipeline's p_home values
        (p_home_raw is NULL at the start of the backtest).  By writing fresh
        identity params, the 7-day TTL in get_platt_params() keeps them in
        place for the entire backtest forward pass.

        The explicit recalibrate_platt() call at the end of the backtest then
        overwrites these with correctly fitted params using the new p_home_raw.

        Returns the number of seasons reset.
        """
        if not seasons:
            return 0
        for season in seasons:
            self.save_state(
                "platt_params", "calibration",
                {"a": _PLATT_A_DEFAULT, "b": _PLATT_B_DEFAULT, "n": 0},
                sample_count=0,
                season=season,
                prediction_source=prediction_source,
            )
        logger.info(
            "[learning] Platt params reset to identity (a=1.0, b=0.0) for seasons %s (state_source=%s)",
            seasons, prediction_source,
        )
        return len(seasons)

    def get_kalman_n_obs(
        self, team: str, context: str, season: int, prediction_source: str = "live",
    ) -> int:
        """Return the number of observations backing this team/context's Kalman
        state (0 if cold-start / no state yet). Used to gauge confidence in
        whether the Kalman correction is actually active for this game.
        """
        state = self._get_kalman_state(team, context, season, prediction_source)
        return state["n_obs"] if state else 0

    def get_kalman_estimate(
        self,
        team: str,
        context: str,
        season: int,
        prediction_source: str = "live",
    ) -> Optional[float]:
        """Return Kalman-filtered estimate of team's expected runs for this context.

        context: 'offense_home' | 'offense_away' | 'defense_home' | 'defense_away'
        Returns None when no observations exist yet.
        """
        state = self._get_kalman_state(team, context, season, prediction_source)
        return state["x_est"] if state else None

    def update_kalman(
        self,
        team: str,
        context: str,
        season: int,
        observed: float,
        prediction_source: str = "live",
    ) -> float:
        """Kalman update step with new run observation. Returns updated estimate.

        CHRON-002 fix: `prediction_source` selects the state_source
        namespace for both the read and the write below, so a backtest's
        walk-forward Kalman update can never read or overwrite production's
        live Kalman state.
        """
        state = self._get_kalman_state(team, context, season, prediction_source)

        if state is None:
            x_est = observed
            p_est = _KF_R
            n_obs = 1
        else:
            x_pred = state["x_est"]
            p_pred = state["p_est"] + _KF_Q
            K      = p_pred / (p_pred + _KF_R)
            x_est  = x_pred + K * (observed - x_pred)
            p_est  = (1.0 - K) * p_pred
            n_obs  = state["n_obs"] + 1

        self._save_kalman_state(team, context, season, x_est, p_est, n_obs, prediction_source)
        return x_est

    def _get_kalman_state(
        self, team: str, context: str, season: int, prediction_source: str = "live",
    ) -> Optional[Dict]:
        """CHRON-002 fix: `prediction_source` selects the state_source namespace."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT x_est, p_est, n_obs FROM kalman_state "
                "WHERE team = ? AND context = ? AND season = ? AND state_source = ?",
                (team, context, season, prediction_source),
            ).fetchone()
        if not row:
            return None
        return {"x_est": row["x_est"], "p_est": row["p_est"], "n_obs": row["n_obs"]}

    def _save_kalman_state(
        self,
        team: str,
        context: str,
        season: int,
        x_est: float,
        p_est: float,
        n_obs: int,
        prediction_source: str = "live",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO kalman_state (team, context, season, state_source, x_est, p_est, n_obs, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team, context, season, state_source) DO UPDATE SET
                    x_est = excluded.x_est,
                    p_est = excluded.p_est,
                    n_obs = excluded.n_obs,
                    updated_at = excluded.updated_at
                """,
                (team, context, season, prediction_source, x_est, p_est, n_obs, now),
            )

    def get_kalman_lambda_adjustment(
        self,
        team: str,
        context: str,
        season: int,
        model_lambda: float,
        prediction_source: str = "live",
    ) -> float:
        """
        Return an adjusted λ blending the model estimate with the Kalman estimate.

        Uses the module-level _KALMAN_BLEND (35% Kalman, 65% model), which must
        match the fraction used in compute_team_bias_kalman_adjusted so the bias
        dampening formula removes exactly the Kalman share of the correction.
        Returns model_lambda when fewer than 10 Kalman observations exist.
        """
        state = self._get_kalman_state(team, context, season, prediction_source)
        if not state or state["n_obs"] < 10:
            return model_lambda
        kalman_lam = state["x_est"]
        result = (1.0 - _KALMAN_BLEND) * model_lambda + _KALMAN_BLEND * kalman_lam
        # Clip to realistic single-game MLB run range.
        # 1.0 floor (not 2.0) allows elite-pitcher/weak-offense matchups through.
        return max(1.0, min(12.0, result))

    # ------------------------------------------------------------------
    # Platt recalibration
    # ------------------------------------------------------------------

    def get_platt_params(self, season: int, prediction_source: str = "live") -> Tuple[float, float]:
        """Return current (a, b) Platt logistic shrinkage coefficients.

        `prediction_source` — CHRON-001 fix (audit_20260714/): forwarded to
        recalibrate_platt() if a refit is triggered below, so a backtest
        caller never refits against live columns. CHRON-002 fix (roadmap
        Step 2 Commit A): also selects the ml_state state_source namespace
        for the cache reads/writes below, so a backtest refit's fitted
        params can never be read by (or overwrite) production's live cache.
        """
        cached = self.load_state("platt_params", "calibration", season, prediction_source)
        if cached:
            last_recal = cached.get("updated_at", "")
            age_days = self._hours_since(last_recal) / 24.0
            if age_days < _PLATT_RECAL_DAYS:
                return float(cached.get("a", _PLATT_A_DEFAULT)), float(cached.get("b", _PLATT_B_DEFAULT))
        # Try to refit — recalibrate_platt returns identity (1.0, 0.0) when n < MIN_SAMPLES.
        try:
            a, b = self.recalibrate_platt(season, prediction_source=prediction_source)
        except Exception as exc:
            logger.warning(f"[learning] Platt recalibration failed: {exc}")
            a, b = _PLATT_A_DEFAULT, _PLATT_B_DEFAULT
        if (a, b) != (_PLATT_A_DEFAULT, _PLATT_B_DEFAULT):
            return a, b
        # recalibrate_platt returned identity — not enough current-season data yet.
        # Warm-start: carry forward previous season's fitted params.
        prev = self.load_state("platt_params", "calibration", season - 1, prediction_source)
        if prev and prev.get("n", 0) >= _PLATT_MIN_SAMPLES:
            logger.info("[learning] Platt warm-start from season %d (a=%.4f b=%.4f)",
                        season - 1, prev.get("a", _PLATT_A_DEFAULT), prev.get("b", _PLATT_B_DEFAULT))
            return float(prev.get("a", _PLATT_A_DEFAULT)), float(prev.get("b", _PLATT_B_DEFAULT))
        return _PLATT_A_DEFAULT, _PLATT_B_DEFAULT

    def recalibrate_platt(self, season: int, prediction_source: str = "live") -> Tuple[float, float]:
        """
        Refit logistic regression mapping raw p_home → home_won on this season's data.
        Requires sklearn. Falls back to defaults if insufficient data.

        `prediction_source` — CHRON-001 fix (audit_20260714/): live reads
        p_home_raw/p_home as before (no behavior change); backtest reads
        backtest_p_home_raw/backtest_p_home instead, so a backtest refit
        never trains on a live game's actual prediction (or vice versa).
        """
        p_home_col     = _pred_col(prediction_source, "p_home")
        p_home_raw_col = _pred_col(prediction_source, "p_home_raw")
        with self._get_conn() as conn:
            rows = conn.execute(
                f"""
                SELECT COALESCE({p_home_raw_col}, {p_home_col}) AS p_home, home_won
                FROM game_outcomes
                WHERE season = ? AND home_won IS NOT NULL AND {p_home_col} IS NOT NULL
                """,
                (season,),
            ).fetchall()

        n = len(rows)
        if n < _PLATT_MIN_SAMPLES:
            logger.info(f"[learning] Platt: only {n} samples (need {_PLATT_MIN_SAMPLES}), keeping defaults")
            return _PLATT_A_DEFAULT, _PLATT_B_DEFAULT

        try:
            from sklearn.linear_model import LogisticRegression
            import numpy as np

            probs    = np.array([r["p_home"] for r in rows], dtype=float)
            outcomes = np.array([r["home_won"] for r in rows], dtype=int)

            # Clip to avoid log(0)
            probs = np.clip(probs, 0.01, 0.99)
            logits = np.log(probs / (1.0 - probs)).reshape(-1, 1)

            lr = LogisticRegression(solver="lbfgs", max_iter=500, C=1e6)
            lr.fit(logits, outcomes)
            a = float(lr.coef_[0][0])
            b = float(lr.intercept_[0])

            self.save_state("platt_params", "calibration",
                            {"a": a, "b": b, "n": n}, n, season,
                            prediction_source=prediction_source)
            logger.info(f"[learning] Platt recalibrated: a={a:.4f}, b={b:.4f} (n={n})")
            return a, b

        except ImportError:
            # sklearn not available — use moment-matching on training set
            a, b = self._platt_moment_match(rows)
            # Save result so TTL cache kicks in and prevents repeated recomputation.
            self.save_state("platt_params", "calibration",
                            {"a": a, "b": b, "n": n}, n, season,
                            prediction_source=prediction_source)
            return a, b

    def _platt_moment_match(self, rows) -> Tuple[float, float]:
        """Fallback: shrink toward 50% using empirical calibration error."""
        probs    = [max(0.01, min(0.99, r["p_home"])) for r in rows]
        outcomes = [r["home_won"] for r in rows]
        n        = len(rows)
        obs_win  = sum(outcomes) / n
        mean_p   = sum(probs) / n

        if abs(mean_p - obs_win) < 0.005:
            return _PLATT_A_DEFAULT, _PLATT_B_DEFAULT

        # Scale logit space to match empirical win rate
        logits   = [math.log(p / (1 - p)) for p in probs]
        mean_logit = sum(logits) / n
        target_logit = math.log(obs_win / (1 - obs_win)) if 0 < obs_win < 1 else 0.0
        b = target_logit - mean_logit
        a = _PLATT_A_DEFAULT
        return a, float(b)

    # ------------------------------------------------------------------
    # Platt-2D recalibration (edge-vs-outcome)
    # ------------------------------------------------------------------

    def get_platt_2d_params(
        self, season: int, prediction_source: str = "live",
        training_columns: str = "backtest_preferred",
    ) -> Optional[Tuple[float, float, float]]:
        """Return (a, b, c) Platt-2D coefficients for `season`, or None if
        there isn't enough historical data yet to fit one.

            logit(p_corrected_home) = a + b*logit(p_home) + c*logit(market_prob_home)

        Callers MUST treat None as "leave p_home unchanged" — never fall back
        to identity coefficients silently, since (a=0,b=1,c=0) is a real,
        meaningful fit outcome, not a safe default.

        `prediction_source` — CHRON-001 fix (audit_20260714/): forwarded to
        recalibrate_platt_2d()'s training query. CHRON-002 fix (roadmap
        Step 2 Commit A): the ("platt2d_params") cache itself is now also
        state_source-scoped — closes the residual CHRON-001 left open (see
        that fix's audit note; this was "out of scope" there, in scope here).

        `training_columns` — roadmap Step 2, Commit B: see
        recalibrate_platt_2d()'s docstring for what this selects.
        """
        cached = self.load_state("platt2d_params", "calibration", season, prediction_source)
        if cached:
            age_days = self._hours_since(cached.get("updated_at", "")) / 24.0
            if age_days < _PLATT2D_RECAL_DAYS:
                return float(cached["a"]), float(cached["b"]), float(cached["c"])
        try:
            result = self.recalibrate_platt_2d(
                season, prediction_source=prediction_source, training_columns=training_columns,
            )
        except Exception as exc:
            logger.warning(f"[learning] Platt-2D recalibration failed: {exc}")
            result = None
        if result is not None:
            return result
        # Not enough current data — warm-start from the most recent prior
        # season that had a successful fit.
        for prior_season in range(season - 1, season - 5, -1):
            prev = self.load_state("platt2d_params", "calibration", prior_season, prediction_source)
            if prev and prev.get("n", 0) >= _PLATT2D_MIN_SAMPLES:
                logger.info(f"[learning] Platt-2D warm-start from season {prior_season}")
                return float(prev["a"]), float(prev["b"]), float(prev["c"])
        return None

    def recalibrate_platt_2d(
        self, season: int, prediction_source: str = "live",
        training_columns: str = "backtest_preferred",
    ) -> Optional[Tuple[float, float, float]]:
        """Refit logit(home_won) ~ a + b*logit(p_home) + c*logit(market_prob_home)
        on all seasons strictly before `season` (expanding window). Returns
        None — not identity — when there isn't enough data; the caller must
        not apply an unfit model.

        `prediction_source` — CHRON-001 fix (audit_20260714/): selects the
        state_source namespace and the plain-column fallback in
        `training_columns='live_only'` mode. ml_home_pin/ml_away_pin (the
        market price) are NOT provenance-split — they record a real
        historical fact (what Pinnacle actually quoted at prediction time),
        populated independently by fetch_historical_odds.py::
        enrich_game_outcomes(), not by record_prediction()/
        update_game_outcomes() — so they stay shared regardless of source,
        same as actual_home_runs/home_won.

        `training_columns` — roadmap Step 2, Commit B (audit_20260714/
        14_remediation_roadmap.md). This is the ONLY cross-season reader of
        game_outcomes prediction columns anywhere in the live-reachable code
        path (confirmed by repo-wide enumeration,
        audit_20260714/chron002_commitB_enumeration.md) — it trains on
        `WHERE season < ?`, i.e. seasons that, since CHRON-001, have their
        live prediction columns permanently FROZEN at whatever they were
        before that fix landed (a backtest can no longer refresh them).
        The backtest_* columns, by contrast, ARE refreshed by every
        validated backtest re-run and reflect the CURRENT model — exactly
        what a calibration layer training on "how well does today's model
        agree with reality" wants as input, not a frozen historical
        snapshot of a possibly-superseded model version.
          - 'backtest_preferred' (default): reads
            COALESCE(backtest_p_home, p_home) — prefers the fresher
            backtest value, falling back to the live column only for rows a
            backtest has never touched (there both columns already agree,
            since backtest_p_home is NULL and COALESCE picks p_home).
          - 'live_only': the pre-Commit-B behavior — reads the plain
            `prediction_source`-selected column only. Kept for the
            equality-freeze test (today, on this DB, backtest_p_home ==
            p_home for every row a backtest has touched, so both modes
            produce byte-identical fits — this mode exists to prove that,
            not because it should be used going forward).
        """
        if training_columns == "backtest_preferred":
            _primary = _pred_col("backtest", "p_home")
            _fallback = _pred_col(prediction_source, "p_home")
            p_home_col = f"COALESCE({_primary}, {_fallback})"
        elif training_columns == "live_only":
            p_home_col = _pred_col(prediction_source, "p_home")
        else:
            raise ValueError(f"unknown training_columns={training_columns!r}")
        with self._get_conn() as conn:
            rows = conn.execute(
                f"""
                SELECT {p_home_col} AS p_home, home_won, ml_home_pin, ml_away_pin
                FROM game_outcomes
                WHERE season < ? AND home_won IS NOT NULL AND {p_home_col} IS NOT NULL
                  AND ml_home_pin IS NOT NULL AND ml_away_pin IS NOT NULL
                  AND ml_home_pin > 1 AND ml_away_pin > 1
                """,
                (season,),
            ).fetchall()

        n = len(rows)
        if n < _PLATT2D_MIN_SAMPLES:
            logger.info(
                f"[learning] Platt-2D: only {n} samples before season {season} "
                f"(need {_PLATT2D_MIN_SAMPLES}), skipping fit"
            )
            return None

        try:
            from sklearn.linear_model import LogisticRegression
            import numpy as np
        except ImportError:
            logger.warning("[learning] Platt-2D requires sklearn — skipping fit")
            return None

        p_home_vals: List[float] = []
        market_prob_vals: List[float] = []
        outcomes: List[int] = []
        for r in rows:
            ph_imp = 1.0 / r["ml_home_pin"]
            pa_imp = 1.0 / r["ml_away_pin"]
            fair_home = ph_imp / (ph_imp + pa_imp)
            p_home_vals.append(min(0.99, max(0.01, r["p_home"])))
            market_prob_vals.append(min(0.99, max(0.01, fair_home)))
            outcomes.append(int(r["home_won"]))

        p_home_arr = np.array(p_home_vals)
        market_arr = np.array(market_prob_vals)
        X = np.column_stack([
            np.log(p_home_arr / (1 - p_home_arr)),
            np.log(market_arr / (1 - market_arr)),
        ])
        y = np.array(outcomes, dtype=int)

        lr = LogisticRegression(fit_intercept=True, penalty="l2", C=1.0)
        lr.fit(X, y)
        a = float(lr.intercept_[0])
        b, c = (float(v) for v in lr.coef_[0])

        self.save_state(
            "platt2d_params", "calibration",
            {"a": a, "b": b, "c": c, "n": n}, n, season,
            prediction_source=prediction_source,
        )
        logger.info(
            f"[learning] Platt-2D recalibrated for season {season}: "
            f"a={a:.4f} b={b:.4f} c={c:.4f} (n={n}, seasons<{season})"
        )
        return a, b, c

    def apply_platt_2d(self, p_home: float, market_prob_home: float, season: int) -> float:
        """Return the Platt-2D-corrected p_home for `season`, or p_home
        unchanged if no fit is available yet (never apply untrained coefficients).
        """
        params = self.get_platt_2d_params(season)
        if params is None:
            return p_home
        a, b, c = params
        ph = min(0.999, max(0.001, p_home))
        mp = min(0.999, max(0.001, market_prob_home))
        logit_p = math.log(ph / (1 - ph))
        logit_m = math.log(mp / (1 - mp))
        z = a + b * logit_p + c * logit_m
        return 1.0 / (1.0 + math.exp(-z))

    # ------------------------------------------------------------------
    # Pipeline weights (gradient descent)
    # ------------------------------------------------------------------

    def get_pipeline_weights(
        self, season: int, prediction_source: str = "live",
    ) -> Dict[str, float]:
        """Return learned weight (0.30–1.50) for each pipeline stage."""
        cached = self.load_state("pipeline_weights", "weights", season, prediction_source)
        if cached:
            return {k: float(cached.get(k, 1.0)) for k in _STAGE_KEYS}
        # Warm-start: carry forward previous season's learned weights.
        # GD takes months to converge from 1.0; prior season is a strong prior.
        prev = self.load_state("pipeline_weights", "weights", season - 1, prediction_source)
        if prev:
            logger.info(
                "[learning] weights warm-start from season %d (state_source=%s)",
                season - 1, prediction_source,
            )
            return {k: float(prev.get(k, 1.0)) for k in _STAGE_KEYS}
        return {k: 1.0 for k in _STAGE_KEYS}

    def _gradient_step(
        self,
        game_pk: int,
        actual_home: int,
        actual_away: int,
        season: int,
        prediction_source: str = "live",
    ) -> None:
        """
        One gradient-descent step on pipeline weights using Poisson log-likelihood.

        Reads stored stage_factors_json to know each stage's contribution.
        Skips silently if stage_factors are missing.

        `prediction_source` — CHRON-001 fix (audit_20260714/): live reads
        lambda_home/lambda_away/stage_factors_json as before; backtest reads
        the backtest_* shadow columns that update_game_outcomes() writes to
        instead of the live ones, so a backtest gradient step trains on the
        SAME λ/stage-factors it just computed this run, not on a live game's
        real prediction (or stale data from a previous backtest run).

        Key format is "{stage}_on_{role}_lambda" (renamed 2026-07-06 — see
        run_module.py's inline comments at each _stage_factors[...] assignment
        for the full rationale: the old "{role}_{stage}" format, e.g.
        "home_pitcher", read like "home team's own pitcher" but actually meant
        "the ratio applied to λ_home", which for Pitcher/Defense/Bullpen is
        driven by the OPPOSING team's engine output — a real, previously
        confirmed source of confusion for external analysis, not a bug in the
        gradient descent itself). Falls back to the old key format for any
        `stage_factors_json` blob written before the rename (e.g. today's
        batched backtest) so historical games aren't silently treated as
        neutral until the next full re-run regenerates them with the new
        format — remove this fallback once a batched backtest has run with
        the renamed keys and no more old-format blobs remain in game_outcomes.
        """
        lam_home_col = _pred_col(prediction_source, "lambda_home")
        lam_away_col = _pred_col(prediction_source, "lambda_away")
        sfj_col      = _pred_col(prediction_source, "stage_factors_json")
        with self._get_conn() as conn:
            row = conn.execute(
                f"SELECT {lam_home_col} AS lambda_home, {lam_away_col} AS lambda_away, "
                f"{sfj_col} AS stage_factors_json FROM game_outcomes "
                f"WHERE game_pk = ?",
                (game_pk,),
            ).fetchone()

        if not row or not row["stage_factors_json"]:
            return

        try:
            factors = json.loads(row["stage_factors_json"])
        except (json.JSONDecodeError, TypeError):
            return

        state = self.load_state("pipeline_weights", "weights", season, prediction_source)
        weights = {k: float(state.get(k, 1.0)) for k in _STAGE_KEYS} if state else {k: 1.0 for k in _STAGE_KEYS}
        n_steps = (state.get("sample_count", 0) if state else 0) + 1
        updated = dict(weights)

        for role, lam_final, actual in [
            ("home", row["lambda_home"], untruncate_home_runs(actual_home)),
            ("away", row["lambda_away"], actual_away),
        ]:
            if not lam_final or lam_final <= 0:
                continue

            for stage in _STAGE_KEYS:
                _new_key = f"{stage}_on_{role}_lambda"
                _old_key = f"{role}_{stage}"
                if _new_key in factors:
                    adj = factors[_new_key]
                elif _old_key in factors:
                    adj = factors[_old_key]
                    logger.debug(
                        "[stage_factors_naming] game_pk=%s stage=%s used legacy "
                        "key '%s' (pre-2026-07-06 rename) — will disappear once "
                        "a batched backtest re-run regenerates stage_factors_json "
                        "with the new format", game_pk, stage, _old_key,
                    )
                else:
                    adj = 1.0
                if adj == 1.0:
                    continue  # stage had no effect on this game
                w = weights[stage]
                # Model: λ_final = λ_base × ∏_s (1 + w_s × (adj_s − 1))
                # ∂NLL/∂λ  = (λ − k) / λ           [Poisson NLL gradient]
                # ∂λ/∂w_s  = λ_final × (adj_s − 1) / (1 + w_s × (adj_s − 1))
                # ∂NLL/∂w_s = (λ − k) × (adj_s − 1) / denom   [chain rule]
                #
                # Previous code used (λ−k)/λ × (adj−1)/denom, which dropped the
                # λ_final factor and made the effective LR ~4.5× too small.
                denom = 1.0 + w * (adj - 1.0)
                if abs(denom) < 1e-6:
                    continue
                grad = (lam_final - actual) * (adj - 1.0) / denom
                updated[stage] = updated[stage] - _LR * grad

        # Clamp
        for stage in _STAGE_KEYS:
            updated[stage] = max(_MIN_WEIGHT, min(_MAX_WEIGHT, updated[stage]))

        self.save_state("pipeline_weights", "weights", updated,
                        sample_count=n_steps, season=season,
                        prediction_source=prediction_source)

    # ------------------------------------------------------------------
    # Generic state persistence
    # ------------------------------------------------------------------

    def save_state(
        self,
        key: str,
        scope: str,
        value: Dict[str, Any],
        sample_count: int,
        season: int,
        prediction_source: str = "live",
    ) -> None:
        """CHRON-002 fix (roadmap Step 2, Commit A): `prediction_source`
        selects the `state_source` namespace ('live' default, unchanged
        behavior; 'backtest' for every backtest_and_retrain.py caller) so a
        backtest refit can never overwrite the exact ml_state key
        production reads — the residual CHRON-001 left open."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO ml_state (key, scope, season, state_source, value_json, sample_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key, scope, season, state_source) DO UPDATE SET
                    value_json   = excluded.value_json,
                    sample_count = excluded.sample_count,
                    updated_at   = excluded.updated_at
                """,
                (key, scope, season, prediction_source, json.dumps(value), sample_count, now),
            )

    def load_state(
        self,
        key: str,
        scope: str,
        season: int,
        prediction_source: str = "live",
    ) -> Optional[Dict[str, Any]]:
        """See save_state()'s docstring for `prediction_source`."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT value_json, sample_count, updated_at "
                "FROM ml_state WHERE key = ? AND scope = ? AND season = ? AND state_source = ?",
                (key, scope, season, prediction_source),
            ).fetchone()
        if not row:
            return None
        data = json.loads(row["value_json"])
        data["sample_count"] = row["sample_count"]
        data["updated_at"]   = row["updated_at"]
        return data

    # ------------------------------------------------------------------
    # Calibration health monitor (LEARN-002, roadmap Step 2 Commit C)
    # ------------------------------------------------------------------

    def calibration_health(self, window_days: int = 14) -> Dict[str, Any]:
        """"Is calibration alive?" — a cheap, real-time production check.

        Scoped to `source='live'` ONLY (CHRON-001 made this possible for
        the first time — before that fix, game_outcomes couldn't
        distinguish a live prediction from a backtest-recomputed one, so
        this exact query would have silently mixed the two and produced a
        meaningless number).

        Returns:
          pct_platt_active     — % of live rows in the window where
                                  p_home != p_home_raw (Platt actually
                                  changed the raw MC probability).
          pct_pinnacle_present — % of live rows in the window with
                                  ml_home_pin populated (Platt-2D's fair-
                                  line input is actually arriving).
          n_rows                — window sample size.

        Logs a warning (not an exception — this must never block a live
        prediction) when n_rows >= _CALIBRATION_HEALTH_MIN_ROWS and either
        percentage falls below _CALIBRATION_HEALTH_PCT_THRESHOLD. This is
        the check that would have caught REG-001/REG-003/REG-007 on day
        one instead of running silently broken for weeks to months.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime("%Y-%m-%d")
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT p_home, p_home_raw, ml_home_pin FROM game_outcomes "
                "WHERE source = 'live' AND game_date >= ?",
                (cutoff,),
            ).fetchall()

        n_rows = len(rows)
        if n_rows == 0:
            pct_platt_active = 0.0
            pct_pinnacle_present = 0.0
        else:
            n_platt_active = sum(
                1 for r in rows
                if r["p_home"] is not None and r["p_home_raw"] is not None
                and abs(r["p_home"] - r["p_home_raw"]) > 1e-9
            )
            n_pinnacle_present = sum(1 for r in rows if r["ml_home_pin"] is not None)
            pct_platt_active = 100.0 * n_platt_active / n_rows
            pct_pinnacle_present = 100.0 * n_pinnacle_present / n_rows

        if n_rows >= _CALIBRATION_HEALTH_MIN_ROWS:
            if pct_platt_active < _CALIBRATION_HEALTH_PCT_THRESHOLD:
                logger.warning(
                    "[calibration_health] Platt appears DEAD in production: only "
                    "%.1f%% of %d live predictions in the last %d days show "
                    "p_home != p_home_raw. This is the exact silent-failure "
                    "signature behind REG-001 (team-bias look-ahead leak) and "
                    "REG-003 (Platt reset to identity by an interrupted "
                    "concurrent backtest launch) — check ml_state's "
                    "'platt_params' row (state_source='live') before trusting "
                    "any live prediction's calibrated probability.",
                    pct_platt_active, n_rows, window_days,
                )
            if pct_pinnacle_present < _CALIBRATION_HEALTH_PCT_THRESHOLD:
                logger.warning(
                    "[calibration_health] Pinnacle fair-line data appears "
                    "MISSING in production: only %.1f%% of %d live predictions "
                    "in the last %d days have ml_home_pin populated. This is "
                    "the exact silent-failure signature behind REG-007 (the "
                    "live odds fetch's REGION/bookmaker-key bug that silently "
                    "disabled Platt-2D correction in production for months, "
                    "zero errors) — check odds_fetcher.py's live fetch path "
                    "before trusting Platt-2D's correction is actually applying.",
                    pct_pinnacle_present, n_rows, window_days,
                )

        return {
            "pct_platt_active": pct_platt_active,
            "pct_pinnacle_present": pct_pinnacle_present,
            "n_rows": n_rows,
            "window_days": window_days,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hours_since(iso_str: str) -> float:
        if not iso_str:
            return float("inf")
        try:
            dt = datetime.fromisoformat(iso_str)
            now = datetime.now(timezone.utc)
            # Make both tz-aware or both tz-naive before subtracting
            if dt.tzinfo is None:
                now = now.replace(tzinfo=None)
            return (now - dt).total_seconds() / 3600.0
        except (ValueError, TypeError):
            return float("inf")
