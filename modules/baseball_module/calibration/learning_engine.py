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


def _l0_ratio(
    actual_runs: float,
    lambda_final: float,
    stage_factors_json: Optional[str],
    is_home: bool,
) -> Optional[float]:
    """Compute O/λ_final for one game-side (untruncated on the home side).

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

# Kalman blend fraction — must be identical in both get_kalman_lambda_adjustment
# and compute_team_bias_kalman_adjusted so the bias dampening formula matches
# the actual blend applied.
_KALMAN_BLEND = 0.35

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
            ]:
                try:
                    conn.execute(f"ALTER TABLE game_outcomes ADD COLUMN {col} {typedef}")
                except sqlite3.OperationalError:
                    pass

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
                     p_home_raw, p_away_raw, ml_home_pin, ml_away_pin, stage_factors_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (game_pk, game_date, season, home_team, away_team, venue, month,
                 lambda_home, lambda_away, p_home, p_away,
                 p_home_raw, p_away_raw, ml_home_pin, ml_away_pin, sf_json),
            )
            inserted = cursor.rowcount == 1
            if not inserted:
                # Row already exists (e.g. historical bulk import).  Backfill
                # any NULL fields we now have — never overwrite existing data.
                conn.execute(
                    """
                    UPDATE game_outcomes SET
                        stage_factors_json = COALESCE(stage_factors_json, ?),
                        p_home_raw         = COALESCE(p_home_raw, ?),
                        p_away_raw         = COALESCE(p_away_raw, ?),
                        ml_home_pin        = COALESCE(ml_home_pin, ?),
                        ml_away_pin        = COALESCE(ml_away_pin, ?)
                    WHERE game_pk = ?
                    """,
                    (sf_json, p_home_raw, p_away_raw, ml_home_pin, ml_away_pin, game_pk),
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
            self._post_outcome_update(game_pk, actual_home_runs, actual_away_runs)
        return existed

    def _post_outcome_update(
        self,
        game_pk: int,
        home_runs: int,
        away_runs: int,
    ) -> None:
        """Update Kalman states and maybe trigger Platt recalibration."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT home_team, away_team, season, lambda_home, lambda_away "
                "FROM game_outcomes WHERE game_pk = ?",
                (game_pk,),
            ).fetchone()
        if not row:
            return

        season = row["season"]
        self.update_kalman(row["home_team"], "offense_home", season, untruncate_home_runs(float(home_runs)))
        self.update_kalman(row["away_team"], "offense_away", season, float(away_runs))
        self.update_kalman(row["home_team"], "defense_home", season, float(away_runs))
        self.update_kalman(row["away_team"], "defense_away", season, untruncate_home_runs(float(home_runs)))

        # Gradient-descent weight update
        if row["lambda_home"] and row["lambda_away"]:
            self._gradient_step(game_pk, home_runs, away_runs, season)

    # ------------------------------------------------------------------
    # Team bias (simple 1-D)
    # ------------------------------------------------------------------

    def compute_team_bias(
        self,
        team: str,
        season: int,
        min_samples: int = _MIN_SAMPLES,
        before_date: Optional[str] = None,
    ) -> float:
        """mean(actual_runs / predicted_λ) for the team. Returns 1.0 when insufficient data.

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
            cached = self.load_state(cache_key, "team_bias", season)
            if cached and cached.get("sample_count", 0) >= min_samples:
                if self._hours_since(cached.get("updated_at", "")) < _BIAS_CACHE_HRS:
                    return float(cached["bias"])

        query = """
            SELECT home_team, away_team,
                   lambda_home, lambda_away,
                   actual_home_runs, actual_away_runs,
                   stage_factors_json
            FROM game_outcomes
            WHERE season = ?
              AND actual_home_runs IS NOT NULL
              AND (home_team = ? OR away_team = ?)
        """
        params: List[Any] = [season, team, team]
        if before_date is not None:
            query += " AND game_date < ?"
            params.append(before_date)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        if len(rows) < min_samples:
            return 1.0

        ratios = []
        for r in rows:
            if r["home_team"] == team:
                ratio = _l0_ratio(r["actual_home_runs"], r["lambda_home"], r["stage_factors_json"], is_home=True)
            elif r["away_team"] == team:
                ratio = _l0_ratio(r["actual_away_runs"], r["lambda_away"], r["stage_factors_json"], is_home=False)
            else:
                continue
            if ratio is not None:
                ratios.append(ratio)

        if not ratios:
            return 1.0

        raw  = sum(ratios) / len(ratios)
        bias = max(1.0 - _BIAS_CLAMP, min(1.0 + _BIAS_CLAMP, raw))
        if before_date is None:
            self.save_state(cache_key, "team_bias", {"bias": bias}, len(ratios), season)
        logger.debug(f"[learning] {team} bias={bias:.4f} (n={len(ratios)})")
        return bias

    def compute_team_bias_kalman_adjusted(
        self,
        team: str,
        season: int,
        context: str,
        month: Optional[int] = None,
        before_date: Optional[str] = None,
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
        raw_bias = self.compute_multidim_bias(team, season, home_away, month=month, before_date=before_date)
        if raw_bias == 1.0:
            return 1.0

        state = self._get_kalman_state(team, context, season)
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
                cached = self.load_state(scope_key, "multidim_bias", season)
                if cached and cached.get("sample_count", 0) >= min_samples:
                    if self._hours_since(cached.get("updated_at", "")) < _BIAS_CACHE_HRS:
                        return float(cached["bias"])
            result = self._compute_multidim(team, season, ha, mo, min_samples, before_date=before_date)
            if result is not None:
                if before_date is None:
                    self.save_state(scope_key, "multidim_bias", {"bias": result[0]}, result[1], season)
                return result[0]

        # Final fallback: simple team bias
        return self.compute_team_bias(team, season, before_date=before_date)

    def _compute_multidim(
        self,
        team: str,
        season: int,
        home_away: str,
        month: Optional[int],
        min_samples: int,
        before_date: Optional[str] = None,
    ) -> Optional[Tuple[float, int]]:
        is_home = (home_away == "home")
        team_col    = "home_team" if is_home else "away_team"
        lambda_col  = "lambda_home" if is_home else "lambda_away"
        actual_col  = "actual_home_runs" if is_home else "actual_away_runs"

        where = f"season = ? AND actual_home_runs IS NOT NULL AND {team_col} = ?"
        params: List[Any] = [season, team]
        if month is not None:
            where += " AND month = ?"
            params.append(month)
        if before_date is not None:
            where += " AND game_date < ?"
            params.append(before_date)

        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT {lambda_col}, {actual_col}, stage_factors_json "
                f"FROM game_outcomes WHERE {where}",
                params,
            ).fetchall()

        if len(rows) < min_samples:
            return None

        ratios = [
            r for r in (
                _l0_ratio(row[actual_col], row[lambda_col], row["stage_factors_json"], is_home)
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

    def reset_kalman_for_seasons(self, seasons: List[int]) -> int:
        """Delete all Kalman states for the given seasons.

        Called by the backtest before it starts processing so that the
        walk-forward loop begins from a neutral state rather than from
        whatever stale (and potentially look-ahead-biased) states the
        live system accumulated.

        Returns the number of rows deleted.
        """
        if not seasons:
            return 0
        placeholders = ",".join("?" * len(seasons))
        with self._get_conn() as conn:
            cursor = conn.execute(
                f"DELETE FROM kalman_state WHERE season IN ({placeholders})",
                seasons,
            )
            deleted = cursor.rowcount
        logger.info(
            "[learning] Kalman reset: deleted %d states for seasons %s",
            deleted, seasons,
        )
        return deleted

    def reset_pipeline_weights(self, seasons: List[int]) -> int:
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
            self.save_state("pipeline_weights", "weights", defaults, 0, season)
        logger.info(
            "[learning] Pipeline weights reset to 1.0 for seasons %s", seasons
        )
        return len(seasons)

    def reset_platt_params(self, seasons: List[int]) -> int:
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
            )
        logger.info(
            "[learning] Platt params reset to identity (a=1.0, b=0.0) for seasons %s",
            seasons,
        )
        return len(seasons)

    def get_kalman_n_obs(self, team: str, context: str, season: int) -> int:
        """Return the number of observations backing this team/context's Kalman
        state (0 if cold-start / no state yet). Used to gauge confidence in
        whether the Kalman correction is actually active for this game.
        """
        state = self._get_kalman_state(team, context, season)
        return state["n_obs"] if state else 0

    def get_kalman_estimate(
        self,
        team: str,
        context: str,
        season: int,
    ) -> Optional[float]:
        """Return Kalman-filtered estimate of team's expected runs for this context.

        context: 'offense_home' | 'offense_away' | 'defense_home' | 'defense_away'
        Returns None when no observations exist yet.
        """
        state = self._get_kalman_state(team, context, season)
        return state["x_est"] if state else None

    def update_kalman(
        self,
        team: str,
        context: str,
        season: int,
        observed: float,
    ) -> float:
        """Kalman update step with new run observation. Returns updated estimate."""
        state = self._get_kalman_state(team, context, season)

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

        self._save_kalman_state(team, context, season, x_est, p_est, n_obs)
        return x_est

    def _get_kalman_state(self, team: str, context: str, season: int) -> Optional[Dict]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT x_est, p_est, n_obs FROM kalman_state "
                "WHERE team = ? AND context = ? AND season = ?",
                (team, context, season),
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
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO kalman_state (team, context, season, x_est, p_est, n_obs, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team, context, season) DO UPDATE SET
                    x_est = excluded.x_est,
                    p_est = excluded.p_est,
                    n_obs = excluded.n_obs,
                    updated_at = excluded.updated_at
                """,
                (team, context, season, x_est, p_est, n_obs, now),
            )

    def get_kalman_lambda_adjustment(
        self,
        team: str,
        context: str,
        season: int,
        model_lambda: float,
    ) -> float:
        """
        Return an adjusted λ blending the model estimate with the Kalman estimate.

        Uses the module-level _KALMAN_BLEND (35% Kalman, 65% model), which must
        match the fraction used in compute_team_bias_kalman_adjusted so the bias
        dampening formula removes exactly the Kalman share of the correction.
        Returns model_lambda when fewer than 10 Kalman observations exist.
        """
        state = self._get_kalman_state(team, context, season)
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

    def get_platt_params(self, season: int) -> Tuple[float, float]:
        """Return current (a, b) Platt logistic shrinkage coefficients."""
        cached = self.load_state("platt_params", "calibration", season)
        if cached:
            last_recal = cached.get("updated_at", "")
            age_days = self._hours_since(last_recal) / 24.0
            if age_days < _PLATT_RECAL_DAYS:
                return float(cached.get("a", _PLATT_A_DEFAULT)), float(cached.get("b", _PLATT_B_DEFAULT))
        # Try to refit — recalibrate_platt returns identity (1.0, 0.0) when n < MIN_SAMPLES.
        try:
            a, b = self.recalibrate_platt(season)
        except Exception as exc:
            logger.warning(f"[learning] Platt recalibration failed: {exc}")
            a, b = _PLATT_A_DEFAULT, _PLATT_B_DEFAULT
        if (a, b) != (_PLATT_A_DEFAULT, _PLATT_B_DEFAULT):
            return a, b
        # recalibrate_platt returned identity — not enough current-season data yet.
        # Warm-start: carry forward previous season's fitted params.
        prev = self.load_state("platt_params", "calibration", season - 1)
        if prev and prev.get("n", 0) >= _PLATT_MIN_SAMPLES:
            logger.info("[learning] Platt warm-start from season %d (a=%.4f b=%.4f)",
                        season - 1, prev.get("a", _PLATT_A_DEFAULT), prev.get("b", _PLATT_B_DEFAULT))
            return float(prev.get("a", _PLATT_A_DEFAULT)), float(prev.get("b", _PLATT_B_DEFAULT))
        return _PLATT_A_DEFAULT, _PLATT_B_DEFAULT

    def recalibrate_platt(self, season: int) -> Tuple[float, float]:
        """
        Refit logistic regression mapping raw p_home → home_won on this season's data.
        Requires sklearn. Falls back to defaults if insufficient data.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(p_home_raw, p_home) AS p_home, home_won
                FROM game_outcomes
                WHERE season = ? AND home_won IS NOT NULL AND p_home IS NOT NULL
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
                            {"a": a, "b": b, "n": n}, n, season)
            logger.info(f"[learning] Platt recalibrated: a={a:.4f}, b={b:.4f} (n={n})")
            return a, b

        except ImportError:
            # sklearn not available — use moment-matching on training set
            a, b = self._platt_moment_match(rows)
            # Save result so TTL cache kicks in and prevents repeated recomputation.
            self.save_state("platt_params", "calibration",
                            {"a": a, "b": b, "n": n}, n, season)
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

    def get_platt_2d_params(self, season: int) -> Optional[Tuple[float, float, float]]:
        """Return (a, b, c) Platt-2D coefficients for `season`, or None if
        there isn't enough historical data yet to fit one.

            logit(p_corrected_home) = a + b*logit(p_home) + c*logit(market_prob_home)

        Callers MUST treat None as "leave p_home unchanged" — never fall back
        to identity coefficients silently, since (a=0,b=1,c=0) is a real,
        meaningful fit outcome, not a safe default.
        """
        cached = self.load_state("platt2d_params", "calibration", season)
        if cached:
            age_days = self._hours_since(cached.get("updated_at", "")) / 24.0
            if age_days < _PLATT2D_RECAL_DAYS:
                return float(cached["a"]), float(cached["b"]), float(cached["c"])
        try:
            result = self.recalibrate_platt_2d(season)
        except Exception as exc:
            logger.warning(f"[learning] Platt-2D recalibration failed: {exc}")
            result = None
        if result is not None:
            return result
        # Not enough current data — warm-start from the most recent prior
        # season that had a successful fit.
        for prior_season in range(season - 1, season - 5, -1):
            prev = self.load_state("platt2d_params", "calibration", prior_season)
            if prev and prev.get("n", 0) >= _PLATT2D_MIN_SAMPLES:
                logger.info(f"[learning] Platt-2D warm-start from season {prior_season}")
                return float(prev["a"]), float(prev["b"]), float(prev["c"])
        return None

    def recalibrate_platt_2d(self, season: int) -> Optional[Tuple[float, float, float]]:
        """Refit logit(home_won) ~ a + b*logit(p_home) + c*logit(market_prob_home)
        on all seasons strictly before `season` (expanding window). Returns
        None — not identity — when there isn't enough data; the caller must
        not apply an unfit model.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT p_home, home_won, ml_home_pin, ml_away_pin
                FROM game_outcomes
                WHERE season < ? AND home_won IS NOT NULL AND p_home IS NOT NULL
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

    def get_pipeline_weights(self, season: int) -> Dict[str, float]:
        """Return learned weight (0.30–1.50) for each pipeline stage."""
        cached = self.load_state("pipeline_weights", "weights", season)
        if cached:
            return {k: float(cached.get(k, 1.0)) for k in _STAGE_KEYS}
        # Warm-start: carry forward previous season's learned weights.
        # GD takes months to converge from 1.0; prior season is a strong prior.
        prev = self.load_state("pipeline_weights", "weights", season - 1)
        if prev:
            logger.info("[learning] weights warm-start from season %d", season - 1)
            return {k: float(prev.get(k, 1.0)) for k in _STAGE_KEYS}
        return {k: 1.0 for k in _STAGE_KEYS}

    def _gradient_step(
        self,
        game_pk: int,
        actual_home: int,
        actual_away: int,
        season: int,
    ) -> None:
        """
        One gradient-descent step on pipeline weights using Poisson log-likelihood.

        Reads stored stage_factors_json to know each stage's contribution.
        Skips silently if stage_factors are missing.

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
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT lambda_home, lambda_away, stage_factors_json FROM game_outcomes "
                "WHERE game_pk = ?",
                (game_pk,),
            ).fetchone()

        if not row or not row["stage_factors_json"]:
            return

        try:
            factors = json.loads(row["stage_factors_json"])
        except (json.JSONDecodeError, TypeError):
            return

        state = self.load_state("pipeline_weights", "weights", season)
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
                        sample_count=n_steps, season=season)

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
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO ml_state (key, scope, season, value_json, sample_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key, scope, season) DO UPDATE SET
                    value_json   = excluded.value_json,
                    sample_count = excluded.sample_count,
                    updated_at   = excluded.updated_at
                """,
                (key, scope, season, json.dumps(value), sample_count, now),
            )

    def load_state(
        self,
        key: str,
        scope: str,
        season: int,
    ) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT value_json, sample_count, updated_at "
                "FROM ml_state WHERE key = ? AND scope = ? AND season = ?",
                (key, scope, season),
            ).fetchone()
        if not row:
            return None
        data = json.loads(row["value_json"])
        data["sample_count"] = row["sample_count"]
        data["updated_at"]   = row["updated_at"]
        return data

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
