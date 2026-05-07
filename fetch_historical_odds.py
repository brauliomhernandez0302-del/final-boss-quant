#!/usr/bin/env python3
"""
fetch_historical_odds.py
========================
Download 2024-2025 MLB pre-game moneylines from The Odds API.

Strategy
--------
One bulk GET /historical/sports/baseball_mlb/odds/ call per game-day,
snapshot at 17:00 UTC (noon ET) — safely pre-game for every MLB game.

Key detail: game_outcomes stores UTC dates (gameDate[:10]), but evening
games start at midnight UTC so their UTC date is local_date + 1.  This
script rebuilds local ET dates from the MLB schedule's "date" blocks
before querying the Odds API, ensuring pre-game odds for all games.

Regions: us + eu  → 27 bookmakers including Pinnacle
Markets: h2h (moneyline only)
Cost:    ~369 game-days × 20 requests = ~7,380 total

Usage
-----
  python fetch_historical_odds.py              # 2024 + 2025
  python fetch_historical_odds.py --seasons 2024
  python fetch_historical_odds.py --dry-run
  python fetch_historical_odds.py --enrich-only  # skip fetch, update game_outcomes
"""

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR

# ── Constants ─────────────────────────────────────────────────────────────────

_ODDS_KEY  = "b6823bc4d3c308218f71835ab183e6a2"
_ODDS_BASE = "https://api.the-odds-api.com/v4"
_MLB_BASE  = "https://statsapi.mlb.com/api/v1"

SNAPSHOT_TIME = "T17:00:00Z"   # noon ET, pre-game for all MLB games
REGIONS       = "us,eu"        # 27 books including Pinnacle
MARKETS       = "h2h"
ODDS_FORMAT   = "decimal"
REQUEST_DELAY = 0.25           # seconds between Odds API calls

SEASON_WINDOWS: Dict[int, Tuple[str, str]] = {
    2024: ("2024-03-20", "2024-11-01"),
    2025: ("2025-03-18", date.today().strftime("%Y-%m-%d")),
}
DEFAULT_SEASONS = [2024, 2025]

# Bookmaker keys that represent Pinnacle
_PINNACLE_KEYS = {"pinnacle"}

# MLB Stats API team names → Odds API team names (where they differ)
_MLB_TO_ODDS_NAME: Dict[str, str] = {
    "athletics": "oakland athletics",   # 2025: moved to Sacramento but Odds API kept "Oakland Athletics"
}


# ── Math helpers ──────────────────────────────────────────────────────────────

def remove_vig(ml_home: float, ml_away: float) -> Tuple[float, float]:
    """
    Multiplicative vig removal.
    Returns (fair_prob_home, fair_prob_away) that sum to 1.0.
    """
    if ml_home <= 1.0 or ml_away <= 1.0:
        return 0.5, 0.5
    impl_h = 1.0 / ml_home
    impl_a = 1.0 / ml_away
    total  = impl_h + impl_a
    return round(impl_h / total, 6), round(impl_a / total, 6)


def extract_odds(game: dict) -> dict:
    """
    Parse a single Odds API game object.
    Returns dict with best, consensus, Pinnacle lines and fair probs.
    """
    home = game["home_team"]
    away = game["away_team"]
    best_home = best_away = 0.0
    best_home_bk = best_away_bk = None
    home_prices: List[float] = []
    away_prices: List[float] = []
    pin_home = pin_away = None

    for bk in game.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            if mkt["key"] != "h2h":
                continue
            prices = {o["name"]: o["price"] for o in mkt["outcomes"]}
            hp = prices.get(home, 0.0)
            ap = prices.get(away, 0.0)

            if hp > 1.0 and ap > 1.0:   # valid line (no placeholder 1.0 odds)
                if hp > best_home:
                    best_home, best_home_bk = hp, bk["title"]
                if ap > best_away:
                    best_away, best_away_bk = ap, bk["title"]
                home_prices.append(hp)
                away_prices.append(ap)

                if bk["key"] in _PINNACLE_KEYS:
                    pin_home, pin_away = hp, ap

    cons_home = round(sum(home_prices) / len(home_prices), 4) if home_prices else None
    cons_away = round(sum(away_prices) / len(away_prices), 4) if away_prices else None

    # Fair (no-vig) probabilities — Pinnacle preferred, fall back to consensus
    if pin_home and pin_away:
        fair_h, fair_a = remove_vig(pin_home, pin_away)
    elif cons_home and cons_away:
        fair_h, fair_a = remove_vig(cons_home, cons_away)
    else:
        fair_h = fair_a = None

    return {
        "odds_api_id":   game["id"],
        "home_team":     home,
        "away_team":     away,
        "ml_home_best":  best_home or None,
        "ml_away_best":  best_away or None,
        "ml_home_best_bk": best_home_bk,
        "ml_away_best_bk": best_away_bk,
        "ml_home_cons":  cons_home,
        "ml_away_cons":  cons_away,
        "ml_home_pin":   pin_home,
        "ml_away_pin":   pin_away,
        "fair_prob_home": fair_h,
        "fair_prob_away": fair_a,
        "n_bookmakers":  len(game.get("bookmakers", [])),
    }


# ── API helpers ───────────────────────────────────────────────────────────────

def _request(session: requests.Session, url: str, params: dict, retries: int = 3) -> dict:
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json(), r.headers
        except Exception as exc:
            if attempt == retries:
                raise
            time.sleep(attempt * 2)
    return {}, {}


def _bar(done: int, total: int, width: int = 38) -> str:
    pct = done / total if total else 0
    filled = int(width * pct)
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total} ({pct:.0%})"


# ── MLB schedule → local dates ────────────────────────────────────────────────

LocalDateMap = Dict[int, Tuple[str, str, str]]   # game_pk → (local_date, home, away)


def fetch_mlb_local_dates(session: requests.Session, season: int) -> LocalDateMap:
    """
    Fetch the full MLB schedule for a season and return a mapping of
        game_pk → (local_ET_date, home_team, away_team)

    Uses the "date" field from each date-block (local schedule date),
    NOT gameDate[:10] which is the UTC datetime string.
    """
    start, end = SEASON_WINDOWS[season]
    data, _ = _request(session, f"{_MLB_BASE}/schedule", {
        "sportId":   1,
        "season":    season,
        "gameType":  "R",
        "startDate": start,
        "endDate":   end,
        "hydrate":   "linescore",
    })

    mapping: LocalDateMap = {}
    for date_block in data.get("dates", []):
        local_date = date_block["date"]   # "YYYY-MM-DD" in local ET
        for game in date_block.get("games", []):
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            gk = game["gamePk"]
            mapping[gk] = (
                local_date,
                game["teams"]["home"]["team"]["name"],
                game["teams"]["away"]["team"]["name"],
            )
    return mapping


# ── Odds API fetch ────────────────────────────────────────────────────────────

def fetch_odds_for_date(session: requests.Session, local_date: str) -> Tuple[List[dict], dict]:
    """One bulk call — returns (list_of_game_objects, response_headers)."""
    snapshot_ts = local_date + SNAPSHOT_TIME
    data, headers = _request(session, f"{_ODDS_BASE}/historical/sports/baseball_mlb/odds/", {
        "apiKey":     _ODDS_KEY,
        "regions":    REGIONS,
        "markets":    MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "date":       snapshot_ts,
    })
    time.sleep(REQUEST_DELAY)
    return data.get("data", []), headers


# ── Game matching ─────────────────────────────────────────────────────────────

MatchRow = Tuple[dict, int]   # (odds_game_dict, game_pk)


def match_games(
    odds_games: List[dict],
    date_games: List[Tuple[int, str, str]],   # [(game_pk, home, away)]
) -> List[MatchRow]:
    """
    Match Odds API games to MLB game_pks by exact team name (case-insensitive).
    Handles doubleheaders: when two games share the same matchup on one day,
    sorts both by commence_time (Odds) and game_pk (MLB) and zips them.

    Returns list of (odds_game, game_pk) pairs.
    """
    def _norm(name: str) -> str:
        low = name.lower()
        return _MLB_TO_ODDS_NAME.get(low, low)

    # Index date_games by normalised (home, away)
    db_by_matchup: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for gk, home, away in date_games:
        db_by_matchup[(_norm(home), _norm(away))].append(gk)
    # Sort game_pks within each matchup (lower pk = earlier game in doubleheader)
    for key in db_by_matchup:
        db_by_matchup[key].sort()

    # Index odds games by normalised (home, away)
    odds_by_matchup: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for og in odds_games:
        key = (og["home_team"].lower(), og["away_team"].lower())
        odds_by_matchup[key].append(og)
    for key in odds_by_matchup:
        odds_by_matchup[key].sort(key=lambda g: g["commence_time"])

    results: List[MatchRow] = []
    for key, og_list in odds_by_matchup.items():
        db_list = db_by_matchup.get(key, [])
        if not db_list:
            # Try reversed home/away — neutral-site games (e.g. Seoul Series)
            # sometimes have opposite designations between MLB and Odds API
            reversed_key = (key[1], key[0])
            db_list = db_by_matchup.get(reversed_key, [])
        if not db_list:
            continue
        for odds_game, gk in zip(og_list, db_list):
            results.append((odds_game, gk))

    return results


# ── Schema management ─────────────────────────────────────────────────────────

def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS historical_odds (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            game_pk          INTEGER NOT NULL UNIQUE,
            game_date        TEXT    NOT NULL,
            season           INTEGER NOT NULL,
            odds_api_id      TEXT,
            home_team        TEXT    NOT NULL,
            away_team        TEXT    NOT NULL,
            snapshot_ts      TEXT    NOT NULL,
            ml_home_best     REAL,
            ml_away_best     REAL,
            ml_home_best_bk  TEXT,
            ml_away_best_bk  TEXT,
            ml_home_cons     REAL,
            ml_away_cons     REAL,
            ml_home_pin      REAL,
            ml_away_pin      REAL,
            fair_prob_home   REAL,
            fair_prob_away   REAL,
            n_bookmakers     INTEGER,
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_ho_season
            ON historical_odds(season);
        CREATE INDEX IF NOT EXISTS idx_ho_game_date
            ON historical_odds(game_date);
    """)

    # Add enrichment columns to game_outcomes (idempotent via try/except)
    for col_def in [
        "ml_home_open     REAL",
        "ml_away_open     REAL",
        "ml_home_cons     REAL",
        "ml_away_cons     REAL",
        "ml_home_pin      REAL",
        "ml_away_pin      REAL",
        "market_prob_home REAL",
        "market_prob_away REAL",
    ]:
        col_name = col_def.split()[0]
        try:
            conn.execute(f"ALTER TABLE game_outcomes ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass   # column already exists


# ── DB writes ─────────────────────────────────────────────────────────────────

def insert_historical_odds(conn: sqlite3.Connection, rows: List[dict]) -> int:
    """INSERT OR IGNORE into historical_odds. Returns number of new rows."""
    inserted = 0
    for r in rows:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO historical_odds (
                game_pk, game_date, season, odds_api_id,
                home_team, away_team, snapshot_ts,
                ml_home_best, ml_away_best, ml_home_best_bk, ml_away_best_bk,
                ml_home_cons, ml_away_cons,
                ml_home_pin,  ml_away_pin,
                fair_prob_home, fair_prob_away,
                n_bookmakers
            ) VALUES (
                :game_pk, :game_date, :season, :odds_api_id,
                :home_team, :away_team, :snapshot_ts,
                :ml_home_best, :ml_away_best, :ml_home_best_bk, :ml_away_best_bk,
                :ml_home_cons, :ml_away_cons,
                :ml_home_pin,  :ml_away_pin,
                :fair_prob_home, :fair_prob_away,
                :n_bookmakers
            )
            """,
            r,
        )
        if cur.rowcount == 1:
            inserted += 1
    conn.commit()
    return inserted


def enrich_game_outcomes(conn: sqlite3.Connection) -> int:
    """
    Copy odds columns from historical_odds → game_outcomes for every matched game_pk.
    Returns number of rows updated.
    """
    cur = conn.execute("""
        UPDATE game_outcomes
        SET
            ml_home_open     = h.ml_home_best,
            ml_away_open     = h.ml_away_best,
            ml_home_cons     = h.ml_home_cons,
            ml_away_cons     = h.ml_away_cons,
            ml_home_pin      = h.ml_home_pin,
            ml_away_pin      = h.ml_away_pin,
            market_prob_home = h.fair_prob_home,
            market_prob_away = h.fair_prob_away
        FROM historical_odds h
        WHERE game_outcomes.game_pk = h.game_pk
          AND h.ml_home_best IS NOT NULL
    """)
    conn.commit()
    return cur.rowcount


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical MLB pre-game odds from The Odds API."
    )
    parser.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and parse, do not write to DB")
    parser.add_argument("--enrich-only", action="store_true",
                        help="Skip fetch, only run enrich_game_outcomes()")
    args = parser.parse_args()

    db_path = DATA_DIR / "predictions_history.db"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  MLB Historical Odds Downloader")
    print("=" * 62)
    print(f"  Seasons:    {args.seasons}")
    print(f"  Regions:    {REGIONS}  (Pinnacle + US books)")
    print(f"  Snapshot:   17:00 UTC (noon ET)")
    print(f"  DB:         {db_path}")
    print(f"  Dry run:    {args.dry_run}")
    print("=" * 62)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)

    if args.enrich_only:
        print("\nRunning enrich_game_outcomes()...")
        n = enrich_game_outcomes(conn)
        print(f"  Updated {n} game_outcomes rows")
        conn.close()
        return

    session = requests.Session()
    session.headers["User-Agent"] = "FinalBossQuantG8/fetch_historical_odds"

    grand_inserted = grand_skipped = grand_no_odds = grand_unmatched = 0

    for season in args.seasons:
        if season not in SEASON_WINDOWS:
            print(f"\n⚠️  Unknown season {season}")
            continue

        print(f"\n{'─' * 62}")
        print(f"  Season {season}")
        print(f"{'─' * 62}")

        # Build local-date → [(game_pk, home, away)] from MLB schedule
        print("  Building local-date index from MLB schedule...", end=" ", flush=True)
        local_date_map = fetch_mlb_local_dates(session, season)
        print(f"{len(local_date_map)} games")

        # Group by local date
        date_index: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
        for gk, (local_date, home, away) in local_date_map.items():
            date_index[local_date].append((gk, home, away))
        sorted_dates = sorted(date_index)

        # Check which game_pks already have odds in the DB
        existing_pks: set = set()
        for row in conn.execute(
            "SELECT game_pk FROM historical_odds WHERE season = ?", (season,)
        ).fetchall():
            existing_pks.add(row[0])

        total_dates = len(sorted_dates)
        remaining_api = None
        inserted = skipped = no_odds = unmatched = 0

        print(f"  Game days: {total_dates}  |  Already in DB: {len(existing_pks)} games")
        print(f"\n  Fetching odds ({REGIONS}, cost ~20/day)...\n")

        batch: List[dict] = []

        for i, local_date in enumerate(sorted_dates, 1):
            date_games = date_index[local_date]   # [(game_pk, home, away)]
            snapshot_ts = local_date + SNAPSHOT_TIME

            # Skip date if all games already in DB
            new_pks_on_date = [gk for gk, _, _ in date_games if gk not in existing_pks]
            if not new_pks_on_date:
                skipped += len(date_games)
                if i % 20 == 0 or i == total_dates:
                    print(f"  {_bar(i, total_dates)}", end="\r")
                continue

            # Fetch odds snapshot for this date
            try:
                odds_games, headers = fetch_odds_for_date(session, local_date)
                remaining_api = headers.get("x-requests-remaining", "?")
            except Exception as exc:
                print(f"\n  ⚠️  {local_date}: fetch failed — {exc}")
                continue

            if not odds_games:
                no_odds += len(date_games)
                if i % 20 == 0 or i == total_dates:
                    print(f"  {_bar(i, total_dates)}", end="\r")
                continue

            # Match odds_games → game_pks
            matches = match_games(odds_games, date_games)
            unmatched += len(date_games) - len(matches)

            for odds_game, game_pk in matches:
                if game_pk in existing_pks:
                    skipped += 1
                    continue
                odds_info = extract_odds(odds_game)
                if odds_info["ml_home_best"] is None:
                    no_odds += 1
                    continue

                row = {
                    "game_pk":        game_pk,
                    "game_date":      local_date,
                    "season":         season,
                    **odds_info,
                    "snapshot_ts":    snapshot_ts,
                }
                batch.append(row)

            # Flush batch every 50 dates
            if not args.dry_run and (len(batch) >= 200 or i == total_dates):
                n = insert_historical_odds(conn, batch)
                inserted += n
                skipped  += len(batch) - n
                batch.clear()

            if i % 5 == 0 or i == total_dates:
                print(f"  {_bar(i, total_dates)}  rem={remaining_api}", end="\r")

        if args.dry_run and batch:
            inserted = len(batch)   # count as would-be inserts

        print(f"\n\n  {'─' * 50}")
        print(f"  {season} results:")
        print(f"    Inserted:    {inserted}")
        print(f"    Skipped:     {skipped}  (already in DB)")
        print(f"    No odds:     {no_odds}  (game not yet in Odds API)")
        print(f"    Unmatched:   {unmatched}  (team name mismatch)")
        print(f"    API remaining: {remaining_api}")

        grand_inserted  += inserted
        grand_skipped   += skipped
        grand_no_odds   += no_odds
        grand_unmatched += unmatched

    # ── Enrich game_outcomes ──────────────────────────────────────────────────
    if not args.dry_run and grand_inserted > 0:
        print(f"\n{'─' * 62}")
        print("  Enriching game_outcomes with pre-game odds...")
        enriched = enrich_game_outcomes(conn)
        print(f"  Updated {enriched} game_outcomes rows")

    conn.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print("  Complete")
    print(f"{'=' * 62}")
    print(f"  Total inserted:   {grand_inserted}")
    print(f"  Total skipped:    {grand_skipped}")
    print(f"  Games no odds:    {grand_no_odds}")
    print(f"  Unmatched teams:  {grand_unmatched}")

    if not args.dry_run:
        conn2 = sqlite3.connect(DATA_DIR / "predictions_history.db")
        n_ho  = conn2.execute("SELECT COUNT(*) FROM historical_odds").fetchone()[0]
        n_enr = conn2.execute(
            "SELECT COUNT(*) FROM game_outcomes WHERE ml_home_open IS NOT NULL"
        ).fetchone()[0]
        n_pin = conn2.execute(
            "SELECT COUNT(*) FROM historical_odds WHERE ml_home_pin IS NOT NULL"
        ).fetchone()[0]
        conn2.close()
        print(f"\n  DB state:")
        print(f"    historical_odds rows:            {n_ho}")
        print(f"    game_outcomes enriched:          {n_enr}")
        print(f"    games with Pinnacle line:        {n_pin}")
    print()


if __name__ == "__main__":
    main()
