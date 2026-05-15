"""
db/predictions_db.py — Predictions table + shared TypedDicts.

Owns ONLY the `predictions` table.
The ML learning tables (game_outcomes, ml_state, kalman_state) are owned
exclusively by LearningEngine in modules/baseball_module/calibration/learning_engine.py.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

import pandas as pd

import config as _cfg


# ── Shared TypedDicts ──────────────────────────────────────────────────────


class GameData(TypedDict, total=False):
    """One game row from the odds API, after normalization."""
    home: str
    away: str
    home_odds: float
    away_odds: float
    commence_time: str
    raw_row: Any  # pd.Series — kept as Any to avoid pandas coupling


class PredictionData(TypedDict, total=False):
    """Row saved to the predictions table."""
    timestamp: str
    sport: str
    league: str
    home_team: str
    away_team: str
    p_home: float
    p_draw: float
    p_away: float
    pick_type: str
    pick_value: str
    ev: float
    kelly: float
    confidence: float
    rating: float
    model_version: str
    notes: str


class AnalysisResult(TypedDict, total=False):
    """Generic return value from any sport run_module()."""
    status: str
    game_info: Dict[str, Any]
    fight_info: Dict[str, Any]
    probabilities: Dict[str, float]
    lambdas_history: Dict[str, Any]
    predictions: Dict[str, float]
    best_bets: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    error: str


# ── PredictionsDB ──────────────────────────────────────────────────────────


class PredictionsDB:
    """SQLite wrapper — manages only the `predictions` table.

    ML tables (game_outcomes, ml_state, kalman_state) are intentionally
    absent here; LearningEngine creates and owns them.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT    NOT NULL,
                    sport         TEXT    NOT NULL,
                    league        TEXT,
                    home_team     TEXT    NOT NULL,
                    away_team     TEXT    NOT NULL,
                    p_home        REAL,
                    p_draw        REAL,
                    p_away        REAL,
                    pick_type     TEXT,
                    pick_value    TEXT,
                    ev            REAL,
                    kelly         REAL,
                    confidence    REAL,
                    rating        REAL,
                    model_version TEXT,
                    notes         TEXT,
                    created_at    TEXT    DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_sport
                    ON predictions(sport);
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
                    ON predictions(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_predictions_ev
                    ON predictions(ev);
                CREATE INDEX IF NOT EXISTS idx_predictions_rating
                    ON predictions(rating);
            """)

    def save(self, pred: PredictionData) -> int:
        """Insert a prediction row; returns the new row id."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions (
                    timestamp, sport, league, home_team, away_team,
                    p_home, p_draw, p_away, pick_type, pick_value,
                    ev, kelly, confidence, rating, model_version, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pred.get("timestamp", datetime.now().isoformat()),
                    pred.get("sport", ""),
                    pred.get("league", ""),
                    pred.get("home_team", ""),
                    pred.get("away_team", ""),
                    pred.get("p_home"),
                    pred.get("p_draw"),
                    pred.get("p_away"),
                    pred.get("pick_type", ""),
                    pred.get("pick_value", ""),
                    pred.get("ev"),
                    pred.get("kelly"),
                    pred.get("confidence"),
                    pred.get("rating"),
                    pred.get("model_version", _cfg.APP_VERSION),
                    pred.get("notes", ""),
                ),
            )
            return int(cursor.lastrowid or 0)

    def read(
        self,
        sport: Optional[str] = None,
        limit: int = _cfg.MAX_HISTORY_RECORDS,
        min_ev: Optional[float] = None,
        min_rating: Optional[float] = None,
    ) -> pd.DataFrame:
        conditions: List[str] = []
        params: List[Any] = []

        if sport:
            conditions.append("sport = ?")
            params.append(sport)
        if min_ev is not None:
            conditions.append("ev >= ?")
            params.append(min_ev)
        if min_rating is not None:
            conditions.append("rating >= ?")
            params.append(min_rating)

        query = "SELECT * FROM predictions"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_stats(self, sport: str) -> Dict[str, Any]:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total_picks,
                    COALESCE(AVG(ev), 0)  as avg_ev,
                    COALESCE(AVG(rating), 0) as avg_rating,
                    COALESCE(MAX(ev), 0)  as max_ev,
                    COALESCE(MIN(ev), 0)  as min_ev,
                    COALESCE(SUM(CASE WHEN ev > 0 THEN 1 ELSE 0 END), 0) as positive_ev_count
                FROM predictions
                WHERE sport = ?
                """,
                (sport,),
            ).fetchone()

            total    = row["total_picks"]
            positive = row["positive_ev_count"]

            return {
                "total_picks":      total,
                "avg_ev":           row["avg_ev"],
                "avg_rating":       row["avg_rating"],
                "max_ev":           row["max_ev"],
                "min_ev":           row["min_ev"],
                "positive_ev_count": positive,
                "positive_ev_rate": (positive / total if total > 0 else 0.0),
            }
