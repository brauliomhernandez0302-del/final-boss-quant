"""Explicit, auditable promotion of backtest-fitted calibration into the live namespace.

CHRON-002 fix (audit_20260714/14_remediation_roadmap.md, roadmap Step 2,
Commit A): before this script existed, a validated backtest's Platt/team-
bias/pipeline-weight/Kalman fits could only reach production the way
CHRON-001 originally worked — a backtest run silently overwriting the exact
ml_state/kalman_state keys production reads. Both CHRON-001 and CHRON-002
close that path entirely: `ml_state`/`kalman_state` now carry a
`state_source` column ('live' | 'backtest'), and every backtest write lands
in the 'backtest' namespace, never touching 'live'.

This script is the *only* sanctioned way to move a backtest's fitted
calibration into the namespace production actually reads. It is
deliberately a separate, manual, human-triggered act — building the tool
does not imply anyone should run it; that decision is explicitly out of
scope for the roadmap step that built it (see the commit message).

Usage:
  # Preview only — nothing written, shows current live value vs. what would
  # replace it:
  python3 scripts/promote_calibration.py --season 2025 --mechanism platt

  # Actually promote:
  python3 scripts/promote_calibration.py --season 2025 --mechanism platt --confirm

  # Every mechanism at once:
  python3 scripts/promote_calibration.py --season 2025 --mechanism all --confirm
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DB_PATH = REPO_ROOT / "data" / "predictions_history.db"

# mechanism -> [(key, scope), ...] for single-key ml_state mechanisms.
_ML_STATE_MECHANISMS = {
    "platt":   [("platt_params", "calibration")],
    "platt2d": [("platt2d_params", "calibration")],
    "weights": [("pipeline_weights", "weights")],
}

_ALL_MECHANISMS = ["platt", "platt2d", "team_bias", "weights", "kalman"]


def _get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _promote_ml_state_row(
    conn: sqlite3.Connection,
    key: str,
    scope: str,
    season: int,
    confirm: bool,
) -> bool:
    """Promote one (key, scope, season) ml_state row from backtest -> live.
    Returns True if a promotion happened (or would happen, dry-run)."""
    src = conn.execute(
        "SELECT value_json, sample_count, updated_at FROM ml_state "
        "WHERE key = ? AND scope = ? AND season = ? AND state_source = 'backtest'",
        (key, scope, season),
    ).fetchone()
    if src is None:
        print(f"  [skip] {scope}/{key} season={season}: no backtest value found")
        return False

    dst = conn.execute(
        "SELECT value_json, sample_count, updated_at FROM ml_state "
        "WHERE key = ? AND scope = ? AND season = ? AND state_source = 'live'",
        (key, scope, season),
    ).fetchone()

    print(f"  {scope}/{key} season={season}:")
    print(f"    live (before) : {dict(dst) if dst else '(none)'}")
    print(f"    backtest (src): {dict(src)}")

    if not confirm:
        print("    -> DRY RUN, not written")
        return True

    conn.execute(
        """
        INSERT INTO ml_state (key, scope, season, state_source, value_json, sample_count, updated_at)
        VALUES (?, ?, ?, 'live', ?, ?, ?)
        ON CONFLICT(key, scope, season, state_source) DO UPDATE SET
            value_json   = excluded.value_json,
            sample_count = excluded.sample_count,
            updated_at   = excluded.updated_at
        """,
        (key, scope, season, src["value_json"], src["sample_count"], src["updated_at"]),
    )
    print("    -> promoted to live")
    return True


def _promote_team_bias(conn: sqlite3.Connection, season: int, confirm: bool) -> int:
    """team_bias covers many ml_state keys (per-team "team_bias:<team>" and
    per-team/home-away/month "bias:<team>:<home|away>[:m<month>]" under the
    'team_bias' and 'multidim_bias' scopes respectively) — enumerate every
    distinct one that has a backtest value for this season."""
    rows = conn.execute(
        "SELECT DISTINCT key, scope FROM ml_state "
        "WHERE scope IN ('team_bias', 'multidim_bias') AND season = ? AND state_source = 'backtest'",
        (season,),
    ).fetchall()
    n = 0
    for row in rows:
        if _promote_ml_state_row(conn, row["key"], row["scope"], season, confirm):
            n += 1
    if not rows:
        print(f"  [skip] team_bias season={season}: no backtest values found")
    return n


def _promote_kalman(conn: sqlite3.Connection, season: int, confirm: bool) -> int:
    rows = conn.execute(
        "SELECT team, context, x_est, p_est, n_obs, updated_at FROM kalman_state "
        "WHERE season = ? AND state_source = 'backtest'",
        (season,),
    ).fetchall()
    n = 0
    for row in rows:
        dst = conn.execute(
            "SELECT x_est, p_est, n_obs, updated_at FROM kalman_state "
            "WHERE team = ? AND context = ? AND season = ? AND state_source = 'live'",
            (row["team"], row["context"], season),
        ).fetchone()
        print(f"  kalman {row['team']}/{row['context']} season={season}:")
        print(f"    live (before) : {dict(dst) if dst else '(none)'}")
        print(f"    backtest (src): {dict(row)}")
        if not confirm:
            print("    -> DRY RUN, not written")
            n += 1
            continue
        conn.execute(
            """
            INSERT INTO kalman_state (team, context, season, state_source, x_est, p_est, n_obs, updated_at)
            VALUES (?, ?, ?, 'live', ?, ?, ?, ?)
            ON CONFLICT(team, context, season, state_source) DO UPDATE SET
                x_est = excluded.x_est,
                p_est = excluded.p_est,
                n_obs = excluded.n_obs,
                updated_at = excluded.updated_at
            """,
            (row["team"], row["context"], season, row["x_est"], row["p_est"], row["n_obs"], row["updated_at"]),
        )
        print("    -> promoted to live")
        n += 1
    if not rows:
        print(f"  [skip] kalman season={season}: no backtest values found")
    return n


def promote(db_path: Path, season: int, mechanisms: List[str], confirm: bool) -> int:
    conn = _get_conn(db_path)
    total = 0
    try:
        for mech in mechanisms:
            print(f"[{mech}]")
            if mech in _ML_STATE_MECHANISMS:
                for key, scope in _ML_STATE_MECHANISMS[mech]:
                    if _promote_ml_state_row(conn, key, scope, season, confirm):
                        total += 1
            elif mech == "team_bias":
                total += _promote_team_bias(conn, season, confirm)
            elif mech == "kalman":
                total += _promote_kalman(conn, season, confirm)
            else:
                raise ValueError(f"unknown mechanism {mech!r}")
        if confirm:
            conn.commit()
    finally:
        conn.close()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a backtest run's fitted calibration into the live namespace. "
                     "Without --confirm, only previews what would change.",
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--mechanism", required=True,
        choices=[*_ALL_MECHANISMS, "all"],
        help="Which calibration mechanism to promote, or 'all' for every one.",
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--confirm", action="store_true",
        help="Actually write the promotion. Without this flag the script only "
             "prints a before/after preview and writes nothing.",
    )
    args = parser.parse_args()

    mechanisms = _ALL_MECHANISMS if args.mechanism == "all" else [args.mechanism]

    print(f"Promoting calibration: season={args.season} mechanisms={mechanisms} "
          f"db={args.db_path} confirm={args.confirm}")
    if not args.confirm:
        print("*** DRY RUN — pass --confirm to actually write. Nothing will be modified. ***")
    print()

    n = promote(args.db_path, args.season, mechanisms, args.confirm)

    print()
    if args.confirm:
        print(f"Done — {n} state row(s) promoted from backtest to live.")
    else:
        print(f"Dry run complete — {n} state row(s) would be promoted. Re-run with --confirm to apply.")


if __name__ == "__main__":
    main()
