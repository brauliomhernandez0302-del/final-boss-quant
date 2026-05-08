"""
Learning engine — records model predictions and actual outcomes,
computes per-team lambda bias, and persists model state to SQLite.

Tables (both live in predictions_history.db):
  game_outcomes  — one row per prediction; actual runs filled in post-game
  ml_state       — key-value store for cached bias values and other state
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
_MIN_SAMPLES = 10          # min finished games before bias is applied
_BIAS_CLAMP = 0.30         # max ±30% correction
_BIAS_CACHE_HOURS = 6      # invalidate cached bias after this many hours


class LearningEngine:
    """
    Closes the prediction loop:
      1. record_prediction()       — called before the game
      2. fetch_pending_outcomes()  — auto-fetches actual scores for past games
      3. compute_team_bias()       — mean(actual / predicted_λ), cached in ml_state
      4. LambdaCalibrator uses the bias as a final multiplicative correction
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
                    lambda_home      REAL,
                    lambda_away      REAL,
                    p_home           REAL,
                    p_away           REAL,
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
    ) -> bool:
        """
        Insert a pre-game prediction row.
        Returns True if a new row was created, False if game_pk already existed.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO game_outcomes
                    (game_pk, game_date, season, home_team, away_team,
                     lambda_home, lambda_away, p_home, p_away)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (game_pk, game_date, season, home_team, away_team,
                 lambda_home, lambda_away, p_home, p_away),
            )
            inserted = cursor.rowcount == 1
        logger.debug(f"   [learning] recorded prediction game_pk={game_pk}")
        return inserted

    # ------------------------------------------------------------------
    # Outcome fetching
    # ------------------------------------------------------------------

    def fetch_pending_outcomes(self, lookback_days: int = 7) -> int:
        """
        For any game_outcome rows without actual runs whose game_date is in the past,
        fetch the final score from MLB Stats API and fill it in.
        Returns the number of rows updated.
        """
        import requests

        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        today = datetime.utcnow().strftime("%Y-%m-%d")

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
        session = requests.Session()
        for row in pending:
            try:
                url = f"{_MLB_API_BASE}/game/{row['game_pk']}/linescore"
                r = session.get(url, timeout=8)
                r.raise_for_status()
                data = r.json()
                home_runs = data.get("teams", {}).get("home", {}).get("runs")
                away_runs = data.get("teams", {}).get("away", {}).get("runs")
                if home_runs is None or away_runs is None:
                    continue
                if self.update_outcome(row["game_pk"], int(home_runs), int(away_runs)):
                    updated += 1
                    logger.info(
                        f"   [learning] game {row['game_pk']} outcome: "
                        f"{away_runs}–{home_runs}"
                    )
            except Exception as exc:
                logger.debug(f"   [learning] could not fetch {row['game_pk']}: {exc}")

        if updated:
            logger.info(f"   [learning] fetched {updated} outcome(s) from MLB API")
        return updated

    def update_outcome(
        self,
        game_pk: int,
        actual_home_runs: int,
        actual_away_runs: int,
    ) -> bool:
        """Fill in actual runs for a finished game. Returns True if the row existed."""
        home_won = 1 if actual_home_runs > actual_away_runs else 0
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE game_outcomes
                SET actual_home_runs = ?,
                    actual_away_runs = ?,
                    home_won         = ?
                WHERE game_pk = ?
                """,
                (actual_home_runs, actual_away_runs, home_won, game_pk),
            )
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Bias computation
    # ------------------------------------------------------------------

    def compute_team_bias(
        self,
        team: str,
        season: int,
        min_samples: int = _MIN_SAMPLES,
    ) -> float:
        """
        Returns mean(actual_runs / predicted_lambda) for the team over the season.

        Interpretation:
          1.0  — model is well-calibrated for this team
          >1.0 — model under-predicts (team scores more than expected)
          <1.0 — model over-predicts

        Returns 1.0 (neutral) when fewer than min_samples finished games exist.
        Bias is clamped to [1-_BIAS_CLAMP, 1+_BIAS_CLAMP] = [0.80, 1.20].
        """
        cache_key = f"team_bias:{team}"
        cached = self.load_state(cache_key, "team_bias", season)
        if cached and cached.get("sample_count", 0) >= min_samples:
            age_hours = self._hours_since(cached.get("updated_at", ""))
            if age_hours < _BIAS_CACHE_HOURS:
                return float(cached["bias"])

        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT home_team, away_team,
                       lambda_home, lambda_away,
                       actual_home_runs, actual_away_runs
                FROM game_outcomes
                WHERE season = ?
                  AND actual_home_runs IS NOT NULL
                  AND (home_team = ? OR away_team = ?)
                """,
                (season, team, team),
            ).fetchall()

        if len(rows) < min_samples:
            return 1.0

        ratios = []
        for r in rows:
            if r["home_team"] == team and r["lambda_home"] and r["lambda_home"] > 0:
                ratios.append(r["actual_home_runs"] / r["lambda_home"])
            elif r["away_team"] == team and r["lambda_away"] and r["lambda_away"] > 0:
                ratios.append(r["actual_away_runs"] / r["lambda_away"])

        if not ratios:
            return 1.0

        raw = sum(ratios) / len(ratios)
        bias = max(1.0 - _BIAS_CLAMP, min(1.0 + _BIAS_CLAMP, raw))
        self.save_state(cache_key, "team_bias", {"bias": bias}, len(ratios), season)
        logger.debug(f"   [learning] {team} bias={bias:.4f} (n={len(ratios)})")
        return bias

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
        now = datetime.utcnow().isoformat()
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
                """
                SELECT value_json, sample_count, updated_at
                FROM ml_state
                WHERE key = ? AND scope = ? AND season = ?
                """,
                (key, scope, season),
            ).fetchone()
        if not row:
            return None
        data = json.loads(row["value_json"])
        data["sample_count"] = row["sample_count"]
        data["updated_at"] = row["updated_at"]
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
            return (datetime.utcnow() - dt).total_seconds() / 3600
        except ValueError:
            return float("inf")
