#!/usr/bin/env python3
"""
post_game.py — daily post-game reconciliation script.

Steps:
  1. fetch_pending_outcomes() — fill actual scores in game_outcomes
  2. Match filled game_outcomes to pending predictions (by team names + date)
  3. Insert into results, update predictions.result / predictions.profit_loss
  4. Recompute sport_stats for MLB
  5. Refresh team bias cache in ml_state
"""

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modules.baseball_module.calibration.learning_engine import LearningEngine

DB_PATH = ROOT / "data" / "predictions_history.db"
CURRENT_SEASON = 2026

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("post_game")


def _get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _extract_date(ts: str) -> str:
    """Return YYYY-MM-DD from a timestamp string."""
    return ts[:10]


def _side_from_pick(pick_value: str) -> str | None:
    """Return 'home' or 'away' from a pick_value like '... (Home)'/'... (Away)'."""
    pv = (pick_value or "").lower()
    if "(home)" in pv:
        return "home"
    if "(away)" in pv:
        return "away"
    return None


def step1_fetch_outcomes(engine: LearningEngine, lookback: int) -> int:
    log.info("Step 1 — fetching pending outcomes from MLB Stats API")
    n = engine.fetch_pending_outcomes(lookback_days=lookback)
    log.info(f"         {n} game(s) updated with actual scores")
    return n


def step2_match_and_record(conn: sqlite3.Connection) -> dict:
    """
    Scan pending MLB predictions, match to filled game_outcomes by team names
    and date (±1 day window), compute correctness and P&L, write to results.
    """
    stats = {"matched": 0, "wins": 0, "losses": 0, "skipped": 0}

    pending = conn.execute(
        """
        SELECT id, timestamp, home_team, away_team,
               pick_type, pick_value, kelly,
               home_odds, away_odds, over_odds, under_odds,
               total_line, ev
        FROM predictions
        WHERE sport = 'MLB' AND result IS NULL
        ORDER BY timestamp
        """
    ).fetchall()

    if not pending:
        log.info("Step 2 — no pending MLB predictions to reconcile")
        return stats

    log.info(f"Step 2 — reconciling {len(pending)} pending prediction(s)")

    for pred in pending:
        game_date = _extract_date(pred["timestamp"])
        home_team = pred["home_team"]
        away_team = pred["away_team"]

        # Match by team names within a ±1-day window (handles late data ingestion)
        outcome = conn.execute(
            """
            SELECT actual_home_runs, actual_away_runs, home_won
            FROM game_outcomes
            WHERE home_team = ? AND away_team = ?
              AND actual_home_runs IS NOT NULL
              AND game_date BETWEEN date(?, '-1 day') AND date(?, '+1 day')
            ORDER BY ABS(julianday(game_date) - julianday(?)) ASC
            LIMIT 1
            """,
            (home_team, away_team, game_date, game_date, game_date),
        ).fetchone()

        if outcome is None:
            stats["skipped"] += 1
            log.debug(f"   no outcome yet: {away_team} @ {home_team} ({game_date})")
            continue

        # Idempotent: skip if already written
        if conn.execute(
            "SELECT 1 FROM results WHERE prediction_id = ?", (pred["id"],)
        ).fetchone():
            log.debug(f"   already recorded: prediction {pred['id']}")
            continue

        home_runs = outcome["actual_home_runs"]
        away_runs = outcome["actual_away_runs"]
        home_won = bool(outcome["home_won"])
        winner = home_team if home_won else away_team

        pick_type = (pred["pick_type"] or "").lower()
        pick_value = pred["pick_value"] or ""
        side = _side_from_pick(pick_value)

        if "over" in pick_type:
            total = home_runs + away_runs
            was_correct = total > (pred["total_line"] or 0)
            odds = pred["over_odds"] or 1.90
        elif "under" in pick_type:
            total = home_runs + away_runs
            was_correct = total < (pred["total_line"] or 0)
            odds = pred["under_odds"] or 1.90
        elif side == "home":
            was_correct = home_won
            odds = pred["home_odds"] or 2.0
        elif side == "away":
            was_correct = not home_won
            odds = pred["away_odds"] or 2.0
        else:
            stats["skipped"] += 1
            log.debug(f"   unknown side for prediction {pred['id']}: {pick_value!r}")
            continue

        kelly = pred["kelly"] or 0.0
        actual_pl = round(kelly * (odds - 1), 5) if was_correct else round(-kelly, 5)
        result_str = "Win" if was_correct else "Loss"
        now = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """
            INSERT OR IGNORE INTO results
                (prediction_id, timestamp, sport, home_team, away_team,
                 home_score, away_score, total_score, winner,
                 was_correct, actual_profit_loss)
            VALUES (?, ?, 'MLB', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pred["id"], now, home_team, away_team,
                home_runs, away_runs, home_runs + away_runs,
                winner, int(was_correct), actual_pl,
            ),
        )
        conn.execute(
            "UPDATE predictions SET result = ?, profit_loss = ? WHERE id = ?",
            (result_str, actual_pl, pred["id"]),
        )

        stats["matched"] += 1
        if was_correct:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        log.info(
            f"   [{result_str}] {away_team} @ {home_team}  "
            f"{away_runs}-{home_runs}  PL={actual_pl:+.4f}"
        )

    conn.commit()
    return stats


def step3_update_sport_stats(conn: sqlite3.Connection) -> None:
    log.info("Step 3 — recomputing MLB sport_stats")

    row = conn.execute(
        """
        SELECT
            COUNT(*)                                        AS total_picks,
            SUM(CASE WHEN result = 'Win'  THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN result = 'Loss' THEN 1 ELSE 0 END) AS losses,
            AVG(ev)                                         AS avg_ev,
            COALESCE(SUM(profit_loss), 0.0)                AS total_roi
        FROM predictions
        WHERE sport = 'MLB' AND result IS NOT NULL
        """
    ).fetchone()

    total = row["total_picks"] or 0
    wins = row["wins"] or 0
    losses = row["losses"] or 0
    win_rate = wins / total if total > 0 else 0.0
    avg_ev = row["avg_ev"] or 0.0
    total_roi = row["total_roi"] or 0.0

    conn.execute(
        """
        INSERT INTO sport_stats
            (sport, total_picks, wins, losses, win_rate, avg_ev, total_roi, updated_at)
        VALUES ('MLB', ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(sport) DO UPDATE SET
            total_picks = excluded.total_picks,
            wins        = excluded.wins,
            losses      = excluded.losses,
            win_rate    = excluded.win_rate,
            avg_ev      = excluded.avg_ev,
            total_roi   = excluded.total_roi,
            updated_at  = excluded.updated_at
        """,
        (total, wins, losses, win_rate, avg_ev, total_roi,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    log.info(
        f"         {total} picks | {wins}W-{losses}L | "
        f"WR={win_rate:.1%} | ROI={total_roi:+.4f}"
    )


def step4_refresh_bias(
    engine: LearningEngine, conn: sqlite3.Connection, season: int
) -> None:
    log.info(f"Step 4 — refreshing team bias cache (season={season})")

    teams = conn.execute(
        """
        SELECT DISTINCT home_team AS team FROM game_outcomes WHERE season = ?
        UNION
        SELECT DISTINCT away_team AS team FROM game_outcomes WHERE season = ?
        """,
        (season, season),
    ).fetchall()

    non_neutral = 0
    for t in teams:
        bias = engine.compute_team_bias(t["team"], season)
        if bias != 1.0:
            non_neutral += 1
            log.debug(f"   {t['team']}: bias={bias:.4f}")

    log.info(f"         {len(teams)} teams | {non_neutral} with non-neutral bias")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-game reconciliation — fetches scores, records results, "
                    "updates stats, refreshes bias cache."
    )
    parser.add_argument(
        "--lookback", type=int, default=7,
        help="Days to look back for pending game outcomes (default: 7)",
    )
    parser.add_argument(
        "--season", type=int, default=CURRENT_SEASON,
        help=f"MLB season year for bias refresh (default: {CURRENT_SEASON})",
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help="Path to predictions_history.db",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("=" * 60)
    log.info("post_game.py — daily reconciliation")
    log.info(f"DB:       {args.db}")
    log.info(f"Season:   {args.season}   Lookback: {args.lookback}d")
    log.info("=" * 60)

    engine = LearningEngine(db_path=args.db)
    conn = _get_conn(args.db)

    try:
        step1_fetch_outcomes(engine, args.lookback)
        stats = step2_match_and_record(conn)
        step3_update_sport_stats(conn)
        step4_refresh_bias(engine, conn, args.season)
    finally:
        conn.close()

    log.info("=" * 60)
    log.info(
        f"Done.  matched={stats['matched']}  "
        f"wins={stats['wins']}  losses={stats['losses']}  "
        f"skipped={stats['skipped']}"
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
