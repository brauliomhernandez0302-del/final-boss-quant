#!/usr/bin/env python3
"""
train_historical.py
===================
Bootstrap the learning engine with MLB 2024-2025 historical results.

Downloads every final game score from MLB Stats API, inserts rows into
game_outcomes using each team's season RPG as the predicted lambda, then
computes and caches per-team bias in ml_state.

Re-running is safe — game_pk is UNIQUE so duplicates are silently skipped.

Usage
-----
    source mi_entorno/bin/activate
    python train_historical.py                   # 2024 + 2025
    python train_historical.py --seasons 2024    # single season
    python train_historical.py --dry-run         # fetch + compute, no DB writes
    python train_historical.py --min-samples 5   # lower threshold for small samples
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, LEAGUE_AVG_RUNS
from modules.baseball_module.calibration.learning_engine import LearningEngine

# ── Constants ─────────────────────────────────────────────────────────────────

MLB_BASE = "https://statsapi.mlb.com/api/v1"
REQUEST_DELAY = 0.2   # polite pause between API calls (seconds)

SEASON_WINDOWS: Dict[int, Tuple[str, str]] = {
    2024: ("2024-03-20", "2024-11-01"),
    2025: ("2025-03-18", date.today().strftime("%Y-%m-%d")),
}

DEFAULT_SEASONS = [2024, 2025]
DEFAULT_MIN_SAMPLES = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(done: int, total: int, width: int = 40) -> str:
    pct = done / total if total else 0
    filled = int(width * pct)
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total} ({pct:.0%})"


def _request(session: requests.Session, url: str, params: dict, retries: int = 3) -> dict:
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries:
                raise
            wait = attempt * 2
            print(f"      ⚠️  attempt {attempt} failed ({exc}), retrying in {wait}s...")
            time.sleep(wait)
    return {}


# ── Season fetching ───────────────────────────────────────────────────────────

GameRow = Dict  # {game_pk, game_date, season, home_team, away_team, home_score, away_score}


def fetch_season(session: requests.Session, season: int) -> List[GameRow]:
    """
    Return all Final regular-season games for the given season.
    One API call — MLB returns the full season in a single response.
    """
    start, end = SEASON_WINDOWS[season]
    print(f"\n   Fetching {season} season ({start} → {end})...")

    data = _request(session, f"{MLB_BASE}/schedule", {
        "sportId":   1,
        "season":    season,
        "gameType":  "R",
        "startDate": start,
        "endDate":   end,
        "hydrate":   "linescore",
    })
    time.sleep(REQUEST_DELAY)

    rows: List[GameRow] = []
    skipped = 0

    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            state = game.get("status", {}).get("abstractGameState", "")
            if state != "Final":
                skipped += 1
                continue

            home_score = game["teams"]["home"].get("score")
            away_score = game["teams"]["away"].get("score")
            if home_score is None or away_score is None:
                skipped += 1
                continue

            rows.append({
                "game_pk":    int(game["gamePk"]),
                "game_date":  game["gameDate"][:10],
                "season":     season,
                "home_team":  game["teams"]["home"]["team"]["name"],
                "away_team":  game["teams"]["away"]["team"]["name"],
                "home_score": int(home_score),
                "away_score": int(away_score),
            })

    print(f"   ✅ {len(rows)} final games  ({skipped} skipped — not Final or missing score)")
    return rows


# ── Season RPG computation ────────────────────────────────────────────────────

def compute_season_rpg(rows: List[GameRow]) -> Dict[str, float]:
    """
    Per-team average runs scored over the season.
    Used as the 'predicted lambda' when inserting historical rows.
    """
    scored: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        scored[r["home_team"]].append(r["home_score"])
        scored[r["away_team"]].append(r["away_score"])

    return {
        team: sum(runs) / len(runs)
        for team, runs in scored.items()
        if runs
    }


# ── DB population ─────────────────────────────────────────────────────────────

def populate_outcomes(
    le: LearningEngine,
    rows: List[GameRow],
    rpg: Dict[str, float],
    dry_run: bool,
) -> Tuple[int, int]:
    """
    Insert game_outcomes rows.

    Predicted lambda = LEAGUE_AVG_RUNS for all historical games.
    Using the team's own season RPG as predicted lambda causes bias to cancel
    to 1.0 by definition (mean(actual / team_avg) ≈ 1.0).  Using league avg
    as the baseline lets high-scoring teams accumulate bias > 1.0 and
    low-scoring teams < 1.0 — a useful prior that the calibrator's live
    predictions will eventually refine over time.

    Returns (inserted, skipped_duplicate).
    """
    inserted = skipped = 0

    for i, r in enumerate(rows, 1):
        lh = LEAGUE_AVG_RUNS
        la = LEAGUE_AVG_RUNS

        if not dry_run:
            is_new = le.record_prediction(
                game_pk=r["game_pk"],
                game_date=r["game_date"],
                season=r["season"],
                home_team=r["home_team"],
                away_team=r["away_team"],
                lambda_home=lh,
                lambda_away=la,
                p_home=0.5,
                p_away=0.5,
            )
            if is_new:
                le.update_outcome(
                    r["game_pk"],
                    actual_home_runs=r["home_score"],
                    actual_away_runs=r["away_score"],
                )
                inserted += 1
            else:
                skipped += 1
        else:
            inserted += 1   # count as would-be inserts in dry-run mode

        if i % 250 == 0 or i == len(rows):
            print(f"      {_bar(i, len(rows))}", end="\r")

    print()  # newline after progress bar
    return inserted, skipped


# ── Bias computation ──────────────────────────────────────────────────────────

def compute_all_biases(
    le: LearningEngine,
    teams: List[str],
    season: int,
    min_samples: int,
    dry_run: bool,
) -> Dict[str, float]:
    """
    Compute and cache bias for every team seen this season.
    Returns {team: bias}.
    """
    biases: Dict[str, float] = {}
    activated = neutral = 0

    for team in sorted(teams):
        if dry_run:
            # Can't compute without real DB rows in dry-run mode
            continue
        bias = le.compute_team_bias(team, season, min_samples=min_samples)
        biases[team] = bias
        if abs(bias - 1.0) > 0.005:
            activated += 1
        else:
            neutral += 1

    if not dry_run:
        print(f"   {activated} teams with active bias correction  |  {neutral} neutral")

    return biases


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the MLB learning engine with historical game results."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        metavar="YEAR",
        help=f"Seasons to download (default: {DEFAULT_SEASONS})",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        metavar="N",
        help=f"Minimum games before bias activates (default: {DEFAULT_MIN_SAMPLES})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and compute but do not write to the database",
    )
    args = parser.parse_args()

    db_path = DATA_DIR / "predictions_history.db"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MLB Historical Training  —  Learning Engine Bootstrap")
    print("=" * 60)
    print(f"  Seasons:     {args.seasons}")
    print(f"  Min samples: {args.min_samples}")
    print(f"  DB path:     {db_path}")
    print(f"  Dry run:     {args.dry_run}")
    print("=" * 60)

    le = LearningEngine(db_path=db_path)
    session = requests.Session()
    session.headers["User-Agent"] = "FinalBossQuantG8/train_historical"

    grand_total_inserted = 0
    grand_total_skipped  = 0
    all_teams: Dict[int, List[str]] = {}

    for season in args.seasons:
        if season not in SEASON_WINDOWS:
            print(f"\n⚠️  Unknown season {season} — supported: {sorted(SEASON_WINDOWS)}")
            continue

        print(f"\n{'─' * 60}")
        print(f"  Season {season}")
        print(f"{'─' * 60}")

        # 1. Fetch all final games
        rows = fetch_season(session, season)
        if not rows:
            print(f"   No data returned for {season}, skipping.")
            continue

        # 2. Compute per-team season RPG
        rpg = compute_season_rpg(rows)
        teams = sorted(rpg)
        all_teams[season] = teams

        print(f"\n   Season RPG summary ({len(teams)} teams):")
        sorted_rpg = sorted(rpg.items(), key=lambda x: -x[1])
        for team, avg in sorted_rpg[:5]:
            print(f"     {team:<30}  {avg:.3f} R/G")
        print(f"     {'...':<30}")
        for team, avg in sorted_rpg[-3:]:
            print(f"     {team:<30}  {avg:.3f} R/G")
        print(f"     League avg:                     {sum(rpg.values())/len(rpg):.3f} R/G")

        # 3. Populate game_outcomes
        print(f"\n   Populating game_outcomes{' (DRY RUN)' if args.dry_run else ''}...")
        inserted, skipped = populate_outcomes(le, rows, rpg, args.dry_run)
        print(f"   {'Would insert' if args.dry_run else 'Inserted'}: {inserted}  |  "
              f"Already existed: {skipped}")
        grand_total_inserted += inserted
        grand_total_skipped  += skipped

        # 4. Compute and cache biases
        print(f"\n   Computing team biases (min_samples={args.min_samples})...")
        biases = compute_all_biases(le, teams, season, args.min_samples, args.dry_run)

        if biases:
            active = {t: b for t, b in biases.items() if abs(b - 1.0) > 0.005}
            if active:
                print(f"\n   Top corrections:")
                for team, bias in sorted(active.items(), key=lambda x: -abs(x[1] - 1.0))[:10]:
                    direction = "↑" if bias > 1.0 else "↓"
                    print(f"     {direction} {team:<30}  bias={bias:.4f}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Training complete")
    print(f"{'=' * 60}")
    print(f"  Rows {'processed' if args.dry_run else 'inserted'}:  {grand_total_inserted}")
    if not args.dry_run:
        print(f"  Rows skipped:   {grand_total_skipped}  (already in DB)")

    if not args.dry_run:
        # Quick sanity check against the DB
        import sqlite3
        conn = sqlite3.connect(db_path)
        n_outcomes = conn.execute("SELECT COUNT(*) FROM game_outcomes").fetchone()[0]
        n_state    = conn.execute("SELECT COUNT(*) FROM ml_state").fetchone()[0]
        conn.close()
        print(f"\n  DB state:")
        print(f"    game_outcomes rows: {n_outcomes}")
        print(f"    ml_state rows:      {n_state}")
        print(f"\n  The learning engine will apply bias corrections automatically")
        print(f"  when ≥{DEFAULT_MIN_SAMPLES} games are available for a team.")

    print()


if __name__ == "__main__":
    main()
