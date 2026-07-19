"""Fase 2B commit B1 — backfill game_outcomes.official_date from the MLB
Stats API's own schedule officialDate field (audit_20260714/fase2b/).

game_outcomes.game_date is the raw UTC gameDate timestamp truncated to a
date — for any night game crossing midnight UTC, that's one calendar day
AHEAD of the real schedule day. This backfills the correct field for every
existing row from the actual source of truth: the /schedule endpoint,
queried once per calendar day across each season's real date range (cheap —
one call returns every game for that whole day, not one call per game_pk).

No heuristics (no "subtract N hours from game_date" guessing) — every row
either resolves from a real schedule lookup or is reported as unresolved
explicitly; unresolved rows are expected to be zero.

Usage:
    python3 scripts/backfill_official_date.py [--db data/predictions_history.db] [--dry-run]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_fetchers import MLBStatsAPI
from modules.baseball_module.calibration.learning_engine import LearningEngine


def _daterange(start: str, end: str):
    d = date.fromisoformat(start)
    e = date.fromisoformat(end)
    while d <= e:
        yield d.isoformat()
        d += timedelta(days=1)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_official_date_map(con: sqlite3.Connection, api: MLBStatsAPI) -> dict[int, str]:
    """One /schedule call per calendar day across each season's real date
    range (padded by 1 day on each side, since the contamination this fixes
    is at most a 1-day shift) — returns {game_pk: officialDate} for every
    game the API returns in that window, not just ones already in our DB."""
    seasons = con.execute(
        "SELECT season, MIN(game_date), MAX(game_date) FROM game_outcomes GROUP BY season ORDER BY season"
    ).fetchall()

    pk_to_official: dict[int, str] = {}
    for season, min_date, max_date in seasons:
        start = (date.fromisoformat(str(min_date)[:10]) - timedelta(days=1)).isoformat()
        end = (date.fromisoformat(str(max_date)[:10]) + timedelta(days=1)).isoformat()
        days = list(_daterange(start, end))
        _log(f"season={season}: {len(days)} days ({start}..{end})")
        for i, day in enumerate(days, 1):
            try:
                games = api.get_todays_games(date=day) or []
            except Exception as exc:
                _log(f"  ERROR fetching schedule for {day}: {exc}")
                continue
            for g in games:
                pk = g.get("game_pk")
                official = g.get("official_date")
                if pk is not None and official:
                    pk_to_official[int(pk)] = official
            if i % 30 == 0 or i == len(days):
                _log(f"  [{i}/{len(days)}] {day} — {len(pk_to_official)} game_pks resolved so far")
    return pk_to_official


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "predictions_history.db")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Triggers LearningEngine._init_tables()'s idempotent migration (adds
    # official_date if missing) — a raw sqlite3.connect() below bypasses it
    # entirely, since that migration only runs via this class's __init__.
    LearningEngine(db_path=args.db)

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    api = MLBStatsAPI()

    pk_to_official = build_official_date_map(con, api)

    all_pks = [r[0] for r in con.execute("SELECT DISTINCT game_pk FROM game_outcomes").fetchall()]
    _log(f"total distinct game_pk in game_outcomes: {len(all_pks)}")

    updated = 0
    unresolved: list[int] = []
    for pk in all_pks:
        official = pk_to_official.get(pk)
        if official is None:
            unresolved.append(pk)
            continue
        if not args.dry_run:
            con.execute(
                "UPDATE game_outcomes SET official_date = ? WHERE game_pk = ?",
                (official, pk),
            )
        updated += 1

    if not args.dry_run:
        con.commit()

    _log(f"resolved={updated} unresolved={len(unresolved)}")
    if unresolved:
        _log(f"UNRESOLVED game_pks (expected empty): {unresolved}")

    # Sanity: % of rows where official_date != date(game_date) — the whole
    # point of this backfill; expected to be substantial (night games), not
    # ~0% (which would mean something's wrong with the fetch, not the thesis).
    if not args.dry_run:
        mismatch = con.execute(
            "SELECT COUNT(*), SUM(official_date != substr(game_date, 1, 10)) "
            "FROM game_outcomes WHERE official_date IS NOT NULL"
        ).fetchone()
        total, n_mismatch = mismatch
        pct = 100.0 * (n_mismatch or 0) / total if total else 0.0
        _log(f"official_date != date(game_date): {n_mismatch}/{total} ({pct:.1f}%)")

        row_745199 = con.execute(
            "SELECT game_pk, game_date, official_date FROM game_outcomes WHERE game_pk = 745199"
        ).fetchone()
        _log(f"game_pk=745199 check: {dict(row_745199) if row_745199 else 'NOT FOUND'}")

    return 0 if not unresolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
