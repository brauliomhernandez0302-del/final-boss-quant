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


def _parse_iso(value: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (any 'Z'-suffixed or offset-aware form)
    into a tz-aware datetime. None on any parse failure."""
    if not value:
        return None
    try:
        v = value[:-1] + "+00:00" if value.endswith("Z") else value
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError, AttributeError):
        return None


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
    market           TEXT    NOT NULL,           -- ML_HOME|ML_AWAY|RL_HOME|RL_AWAY|OVER|UNDER|F5_HOME|F5_AWAY|F5_OVER|F5_UNDER
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
            try:
                conn.execute("ALTER TABLE picks ADD COLUMN total_line REAL")
            except Exception:
                pass  # column already exists
            # 2026-07-12: real closing-line value (CLV) capture. Added because
            # the only prior "CLV" anywhere in this codebase (backtest_and_
            # retrain.py's clv_home/clv_away) is edge-as-ratio, not a real
            # bet-time-vs-closing-time price pair — see feedback_clv_misnomer
            # memory. This is the actual thing: closing_pin_home/away are the
            # Pinnacle price for each side captured shortly before first
            # pitch; clv_pct compares the price this pick was published at to
            # the closing price on the SAME side. CLV converges to a
            # skill/no-skill answer in ~100 picks, an order of magnitude
            # faster than live ROI (see project memory for the reasoning).
            for ddl in (
                "ALTER TABLE picks ADD COLUMN closing_odds_decimal REAL",
                "ALTER TABLE picks ADD COLUMN closing_pin_home REAL",
                "ALTER TABLE picks ADD COLUMN closing_pin_away REAL",
                "ALTER TABLE picks ADD COLUMN closing_captured_at TEXT",
                "ALTER TABLE picks ADD COLUMN clv_pct REAL",
            ):
                try:
                    conn.execute(ddl)
                except Exception:
                    pass  # column already exists
            # 2026-07-19: without a per-pick game start time, a closing-line
            # sweep run any time after publish captures WHATEVER price is
            # live at that moment — including a tonight's-game pick published
            # this morning, hours before its real close, with no way to tell
            # a stale early snapshot from a true closing one. This column is
            # what get_picks_needing_closing_capture() uses to know a pick's
            # game hasn't started yet (still capturable) and what
            # capture_closing_line() checks to reject any capture attempted
            # at or after the game's own start — see both methods' docstrings
            # for the "last pre-start capture wins" design this enables.
            # Nullable/backfill-free: old rows and non-MLB picks (whose
            # publishers may not populate it) are still returned by the query
            # above rather than being silently stranded forever — the caller
            # decides whether to skip them with a warning.
            try:
                conn.execute("ALTER TABLE picks ADD COLUMN commence_time TEXT")
            except Exception:
                pass  # column already exists
            # 2026-07-19: per-capture staleness, so CLV analysis can report
            # its own distribution of "how close to first pitch was this
            # actually captured" instead of assuming every snapshot is
            # equally close to the true close (see capture_closing_line()).
            try:
                conn.execute("ALTER TABLE picks ADD COLUMN minutes_before_start REAL")
            except Exception:
                pass  # column already exists
            # 2026-07-19 (Fase 2A commit 4): every pick publishes tagged
            # 'quarantine' until the public band (an EV/tier cutoff) is
            # fixed against a freshly re-measured baseline (Fase 2C) — see
            # config.QUARANTINE_MODE. 'public' is the only other value, set
            # by a future migration/promotion step, never by publish_pick()
            # itself while QUARANTINE_MODE is true.
            try:
                conn.execute("ALTER TABLE picks ADD COLUMN publish_mode TEXT NOT NULL DEFAULT 'quarantine'")
            except Exception:
                pass  # column already exists
            # 2026-07-20 (docs/PROTOCOLO_CLV_V1.md): the CLV evaluation
            # protocol's engine-freeze validity condition needs a record of
            # which prediction-engine commit each pick was actually made
            # under — if the engine changes mid-window, the protocol
            # requires the primary sample to reset. This is what lets a
            # future audit tell whether that condition held.
            try:
                conn.execute("ALTER TABLE picks ADD COLUMN engine_commit TEXT")
            except Exception:
                pass  # column already exists

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
        total_line: Optional[float] = None,
        commence_time: Optional[str] = None,
        publish_mode: str = "quarantine",
        engine_commit: Optional[str] = None,
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
                    notes, pipeline_json, total_line, commence_time, publish_mode,
                    engine_commit
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    pick_uid, ts, game_date, sport, game_pk,
                    home_team, away_team, market,
                    model_prob, implied_prob, ev_pct, kelly_fraction,
                    confidence_tier, odds_decimal, stake_units,
                    notes, pipeline_json, total_line, commence_time, publish_mode,
                    engine_commit,
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
                # Latest-by-insertion-order, NOT MAX(running_total): the
                # cumulative total legitimately dips below a prior peak
                # after any loss, so MAX() silently freezes in a stale,
                # inflated baseline the next time a pick resolves (found +
                # reproduced 2026-07-06: pnl sequence +10,-15,+3 should give
                # running_total 10,-5,-2 but MAX() gave 10,-5,13).
                prev = conn.execute(
                    "SELECT running_total FROM bankroll ORDER BY id DESC LIMIT 1"
                ).fetchone()
                prev = prev[0] if prev else 0.0
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

    # ------------------------------------------------------------------ CLV

    def get_picks_needing_closing_capture(
        self, sport: Optional[str] = None, game_date: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        """Picks still capturable — i.e. their game hasn't started yet.

        2026-07-19 (Fase 2A commit 3, V-closing-lines audit): replaced the
        earlier "closing_captured_at IS NULL" one-shot-capture design with
        "last pre-start capture wins" — a sweep is expected to re-capture and
        overwrite an already-captured pick as long as its game hasn't
        started, so this returns EVERY not-yet-started pick on every call,
        not just never-captured ones. Once commence_time passes, a pick
        drops out of this query permanently — its last pre-start capture IS
        the closing line, nothing touches it again.

        Picks with no `commence_time` (a sport/publisher that doesn't
        populate it) are still returned — the caller decides whether to skip
        them with a warning (see capture_closing_lines.py) rather than this
        query silently stranding them forever.
        """
        clauses = ["(commence_time IS NULL OR datetime(commence_time) > datetime('now'))"]
        params: list = []
        if sport:
            clauses.append("sport = ?"); params.append(sport)
        if game_date:
            clauses.append("game_date = ?"); params.append(game_date)
        where = "WHERE " + " AND ".join(clauses)
        with self._conn() as conn:
            return conn.execute(
                f"SELECT * FROM picks {where} ORDER BY game_date", params,
            ).fetchall()

    def capture_closing_line(
        self,
        pick_uid: str,
        *,
        closing_odds_decimal: Optional[float] = None,
        closing_pin_home: Optional[float] = None,
        closing_pin_away: Optional[float] = None,
        captured_at: Optional[str] = None,
    ) -> bool:
        """Record the closing line for a pick and compute clv_pct.

        clv_pct = (odds_decimal_at_publish / closing_price_same_side - 1) * 100
        — positive means the pick was published at a better (higher decimal)
        price than the market closed at, i.e. beat the close. Only computed
        for ML_HOME/ML_AWAY, where the Pinnacle closing price on the matching
        side is unambiguous; other markets store the closing snapshot for
        future use but leave clv_pct NULL (v1 scope — see track_record
        project memory).

        "Last pre-start capture wins" (2026-07-19, Fase 2A commit 3): this
        OVERWRITES any prior capture, not just the first one — the caller
        (a scheduled sweep) is expected to call this repeatedly as first
        pitch approaches, and each call updates closing_* to the freshest
        pre-game snapshot. A capture attempted AT OR AFTER the pick's own
        commence_time is REJECTED outright (returns False, row untouched) —
        once the game has started there is no more "closing" line to
        capture, and honoring a late call here would silently mislabel an
        in-game or post-game price as the close. Also stores
        minutes_before_start (commence_time - captured_at, in minutes) so
        CLV analysis can report its own staleness distribution instead of
        assuming every snapshot is equally close to the true close.
        """
        ts = captured_at or datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT market, odds_decimal, commence_time FROM picks WHERE pick_uid = ?",
                (pick_uid,),
            ).fetchone()
            if row is None:
                return False

            minutes_before_start = None
            if row["commence_time"]:
                commence_dt = _parse_iso(row["commence_time"])
                captured_dt = _parse_iso(ts)
                if commence_dt is not None and captured_dt is not None:
                    if captured_dt >= commence_dt:
                        return False  # post-start — reject, never capture
                    minutes_before_start = round(
                        (commence_dt - captured_dt).total_seconds() / 60.0, 2
                    )

            clv_pct = None
            odds_decimal = row["odds_decimal"]
            market = row["market"]
            closing_ref = None
            if market == "ML_HOME":
                closing_ref = closing_pin_home
            elif market == "ML_AWAY":
                closing_ref = closing_pin_away
            if odds_decimal and closing_ref and closing_ref > 1.0:
                clv_pct = round((odds_decimal / closing_ref - 1.0) * 100.0, 4)

            cur = conn.execute(
                """
                UPDATE picks
                SET closing_odds_decimal  = ?,
                    closing_pin_home      = ?,
                    closing_pin_away      = ?,
                    closing_captured_at   = ?,
                    minutes_before_start  = ?,
                    clv_pct               = ?
                WHERE pick_uid = ?
                """,
                (closing_odds_decimal, closing_pin_home, closing_pin_away,
                 ts, minutes_before_start, clv_pct, pick_uid),
            )
            return cur.rowcount > 0

    def get_clv_stats(self, sport: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate CLV across all picks with a computed clv_pct (ML markets
        only, v1). This is the fast skill signal — see capture_closing_line's
        docstring; meaningful well before enough picks exist for an honest
        ROI read."""
        clauses = ["clv_pct IS NOT NULL"]
        params: list = []
        if sport:
            clauses.append("sport = ?"); params.append(sport)
        where = "WHERE " + " AND ".join(clauses)
        with self._conn() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS n,
                       AVG(clv_pct) AS mean_clv_pct,
                       SUM(CASE WHEN clv_pct > 0 THEN 1 ELSE 0 END) AS n_positive
                FROM picks {where}
                """,
                params,
            ).fetchone()
            d = dict(row)
            d["pct_positive"] = round(100.0 * d["n_positive"] / d["n"], 2) if d["n"] else 0.0
            return d

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
            # Same latest-by-insertion-order fix as resolve_pick() — MAX()
            # is wrong here too, for the identical reason.
            cum_row = conn.execute(
                "SELECT running_total FROM bankroll "
                "WHERE resolved_at <= ? || 'T23:59:59Z' "
                "ORDER BY id DESC LIMIT 1",
                (snap_date,),
            ).fetchone()
            cum = cum_row[0] if cum_row else 0.0
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
