"""
track_record/db.py — SQLite schema and CRUD for the live track record.

Separate from predictions_history.db so the track record is a clean,
auditable log: every pick has a pre-game published_at timestamp and
is resolved automatically after the game ends.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

DB_PATH = Path(__file__).parent.parent / "data" / "track_record.db"

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- One row per published pick (written PRE-game, resolved POST-game)
CREATE TABLE IF NOT EXISTS picks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    pick_uid         TEXT    UNIQUE NOT NULL,   -- {game_pk}:{market} deterministic
    published_at     TEXT    NOT NULL,           -- ISO-8601 UTC, proof pick is pre-game
    game_date        TEXT    NOT NULL,           -- YYYY-MM-DD
    sport            TEXT    NOT NULL,           -- MLB | NBA | UFC
    game_pk          INTEGER,                    -- sport internal game ID
    home_team        TEXT    NOT NULL,
    away_team        TEXT    NOT NULL,
    market           TEXT    NOT NULL,           -- ML_HOME|ML_AWAY|RL_HOME|RL_AWAY|OVER|UNDER|F5_HOME|F5_AWAY
    model_prob       REAL    NOT NULL,
    implied_prob     REAL,
    ev_pct           REAL    NOT NULL,
    kelly_fraction   REAL,
    confidence_tier  TEXT,
    odds_decimal     REAL,
    stake_units      REAL,
    -- resolved after game
    resolved_at      TEXT,
    actual_home_score INTEGER,
    actual_away_score INTEGER,
    result           TEXT,                       -- WIN | LOSS | PUSH | VOID
    profit_loss_units REAL,
    notes            TEXT,
    pipeline_json    TEXT                        -- full pipeline snapshot for audit
);

-- Running bankroll ledger (one row per resolved pick)
CREATE TABLE IF NOT EXISTS bankroll (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    pick_id      INTEGER NOT NULL REFERENCES picks(id),
    resolved_at  TEXT    NOT NULL,
    units_staked REAL    NOT NULL,
    units_pnl    REAL    NOT NULL,
    running_total REAL   NOT NULL,
    notes        TEXT
);

-- Daily roll-up snapshots
CREATE TABLE IF NOT EXISTS daily_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    snap_date       TEXT    UNIQUE NOT NULL,
    picks_count     INTEGER NOT NULL DEFAULT 0,
    wins            INTEGER NOT NULL DEFAULT 0,
    losses          INTEGER NOT NULL DEFAULT 0,
    pushes          INTEGER NOT NULL DEFAULT 0,
    units_wagered   REAL    NOT NULL DEFAULT 0.0,
    units_pnl       REAL    NOT NULL DEFAULT 0.0,
    cumulative_pnl  REAL    NOT NULL DEFAULT 0.0,
    roi_pct         REAL    NOT NULL DEFAULT 0.0,
    updated_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_picks_game_date  ON picks(game_date);
CREATE INDEX IF NOT EXISTS idx_picks_sport      ON picks(sport);
CREATE INDEX IF NOT EXISTS idx_picks_result     ON picks(result);
CREATE INDEX IF NOT EXISTS idx_picks_published  ON picks(published_at);
CREATE INDEX IF NOT EXISTS idx_picks_game_pk    ON picks(game_pk);
"""


class TrackRecordDB:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ writes

    def publish_pick(
        self,
        pick_uid: str,
        game_date: str,
        sport: str,
        game_pk: Optional[int],
        home_team: str,
        away_team: str,
        market: str,
        model_prob: float,
        ev_pct: float,
        implied_prob: Optional[float] = None,
        kelly_fraction: Optional[float] = None,
        confidence_tier: Optional[str] = None,
        odds_decimal: Optional[float] = None,
        stake_units: Optional[float] = None,
        notes: Optional[str] = None,
        pipeline_json: Optional[str] = None,
        published_at: Optional[str] = None,
    ) -> int:
        """Insert a pre-game pick. Returns the new row id (0 if already exists)."""
        ts = published_at or datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO picks (
                    pick_uid, published_at, game_date, sport, game_pk,
                    home_team, away_team, market,
                    model_prob, implied_prob, ev_pct, kelly_fraction,
                    confidence_tier, odds_decimal, stake_units,
                    notes, pipeline_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    pick_uid, ts, game_date, sport, game_pk,
                    home_team, away_team, market,
                    model_prob, implied_prob, ev_pct, kelly_fraction,
                    confidence_tier, odds_decimal, stake_units,
                    notes, pipeline_json,
                ),
            )
            return int(cur.lastrowid or 0)

    def resolve_pick(
        self,
        pick_uid: str,
        actual_home_score: int,
        actual_away_score: int,
        result: str,
        profit_loss_units: float,
        resolved_at: Optional[str] = None,
    ) -> bool:
        """Fill in the post-game outcome. Returns True if a row was updated."""
        ts = resolved_at or datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """
                UPDATE picks
                SET resolved_at       = ?,
                    actual_home_score = ?,
                    actual_away_score = ?,
                    result            = ?,
                    profit_loss_units = ?
                WHERE pick_uid = ? AND result IS NULL
                """,
                (ts, actual_home_score, actual_away_score, result,
                 profit_loss_units, pick_uid),
            )
            if cur.rowcount == 0:
                return False

            # update bankroll ledger
            row = conn.execute(
                "SELECT id, stake_units FROM picks WHERE pick_uid = ?", (pick_uid,)
            ).fetchone()
            if row:
                prev = conn.execute(
                    "SELECT COALESCE(MAX(running_total), 0.0) FROM bankroll"
                ).fetchone()[0]
                conn.execute(
                    """
                    INSERT INTO bankroll (pick_id, resolved_at, units_staked,
                                         units_pnl, running_total)
                    VALUES (?,?,?,?,?)
                    """,
                    (
                        row["id"], ts,
                        row["stake_units"] or 0.0,
                        profit_loss_units,
                        round(prev + profit_loss_units, 6),
                    ),
                )
            return True

    # ------------------------------------------------------------------ reads

    def get_pending(self, sport: Optional[str] = None) -> List[sqlite3.Row]:
        """Return picks not yet resolved."""
        with self._conn() as conn:
            if sport:
                return conn.execute(
                    "SELECT * FROM picks WHERE result IS NULL AND sport = ? "
                    "ORDER BY game_date",
                    (sport,),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM picks WHERE result IS NULL ORDER BY game_date"
            ).fetchall()

    def get_picks(
        self,
        sport: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        result: Optional[str] = None,
        limit: int = 500,
    ) -> List[sqlite3.Row]:
        clauses, params = [], []
        if sport:
            clauses.append("sport = ?"); params.append(sport)
        if from_date:
            clauses.append("game_date >= ?"); params.append(from_date)
        if to_date:
            clauses.append("game_date <= ?"); params.append(to_date)
        if result:
            clauses.append("result = ?"); params.append(result)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._conn() as conn:
            return conn.execute(
                f"SELECT * FROM picks {where} ORDER BY published_at DESC LIMIT ?",
                params,
            ).fetchall()

    def get_bankroll_history(self) -> List[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT b.*, p.game_date, p.sport, p.market, p.home_team, p.away_team, p.result "
                "FROM bankroll b JOIN picks p ON b.pick_id = p.id "
                "ORDER BY b.resolved_at",
            ).fetchall()

    def get_all_time_stats(self, sport: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate resolved picks into headline numbers."""
        clauses = ["result IS NOT NULL"]
        params: list = []
        if sport:
            clauses.append("sport = ?"); params.append(sport)
        where = "WHERE " + " AND ".join(clauses)
        with self._conn() as conn:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*)                                             AS total,
                    SUM(CASE WHEN result='WIN'  THEN 1 ELSE 0 END)      AS wins,
                    SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END)      AS losses,
                    SUM(CASE WHEN result='PUSH' THEN 1 ELSE 0 END)      AS pushes,
                    COALESCE(SUM(profit_loss_units), 0)                  AS total_units,
                    COALESCE(SUM(stake_units), 0)                        AS total_staked,
                    COALESCE(AVG(ev_pct), 0)                             AS avg_ev,
                    COALESCE(AVG(model_prob), 0)                         AS avg_model_prob
                FROM picks {where}
                """,
                params,
            ).fetchone()
            d = dict(row)
            t = d["total"] or 0
            staked = d["total_staked"] or 0
            d["win_rate"] = round(d["wins"] / t, 4) if t else 0.0
            d["roi_pct"] = round(d["total_units"] / staked * 100, 2) if staked > 0 else 0.0
            return d

    def already_published(self, pick_uid: str) -> bool:
        with self._conn() as conn:
            return conn.execute(
                "SELECT 1 FROM picks WHERE pick_uid = ?", (pick_uid,)
            ).fetchone() is not None

    def upsert_daily_snapshot(self, snap_date: str) -> None:
        """Recompute and upsert the daily snapshot for snap_date."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                                             AS cnt,
                    SUM(CASE WHEN result='WIN'  THEN 1 ELSE 0 END)      AS wins,
                    SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END)      AS losses,
                    SUM(CASE WHEN result='PUSH' THEN 1 ELSE 0 END)      AS pushes,
                    COALESCE(SUM(stake_units),    0)                     AS wagered,
                    COALESCE(SUM(profit_loss_units), 0)                  AS pnl
                FROM picks
                WHERE game_date = ? AND result IS NOT NULL
                """,
                (snap_date,),
            ).fetchone()
            cum = conn.execute(
                "SELECT COALESCE(MAX(running_total), 0) FROM bankroll "
                "WHERE resolved_at <= ? || 'T23:59:59Z'",
                (snap_date,),
            ).fetchone()[0]
            wagered = row["wagered"] or 0
            roi = round(row["pnl"] / wagered * 100, 2) if wagered > 0 else 0.0
            conn.execute(
                """
                INSERT INTO daily_snapshots
                    (snap_date, picks_count, wins, losses, pushes,
                     units_wagered, units_pnl, cumulative_pnl, roi_pct, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(snap_date) DO UPDATE SET
                    picks_count    = excluded.picks_count,
                    wins           = excluded.wins,
                    losses         = excluded.losses,
                    pushes         = excluded.pushes,
                    units_wagered  = excluded.units_wagered,
                    units_pnl      = excluded.units_pnl,
                    cumulative_pnl = excluded.cumulative_pnl,
                    roi_pct        = excluded.roi_pct,
                    updated_at     = excluded.updated_at
                """,
                (
                    snap_date,
                    row["cnt"] or 0,
                    row["wins"] or 0,
                    row["losses"] or 0,
                    row["pushes"] or 0,
                    wagered,
                    row["pnl"] or 0.0,
                    cum,
                    roi,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
