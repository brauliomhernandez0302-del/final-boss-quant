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

import os

import requests
from dotenv import load_dotenv

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

_ODDS_KEY  = os.getenv("ODDS_API_KEY", "").strip()
_ODDS_BASE = "https://api.the-odds-api.com/v4"
_MLB_BASE  = "https://statsapi.mlb.com/api/v1"

SNAPSHOT_TIME = "T17:00:00Z"   # noon ET, pre-game for all MLB games
# Región y mercados: el COSTE de cuota es (nº mercados × nº regiones × 10) por
# llamada histórica, así que la combinación importa mucho. Medido el 2026-08-04
# contra el endpoint real:
#
#     h2h,totals,spreads  en us,eu   → 60 por día  (369 días = 22.140, NO cabe)
#     totals,spreads      en us      → 20 por día  pero SIN Pinnacle
#     totals,spreads      en eu      → 20 por día  CON Pinnacle  ← ésta
#
# Pinnacle es indispensable: su par es contra el que se desvigoriza para
# obtener la probabilidad justa. Vive en la región `eu`, así que pedir sólo esa
# baja el coste a un tercio y no pierde nada que se use.
REGIONS       = "us,eu"        # 27 books including Pinnacle
MARKETS       = "h2h"

# Backfill de derivados: se piden APARTE de la corrida de moneyline, para no
# re-descargar 369 días de h2h que ya están en la tabla desde hace meses.
REGIONS_DERIVADOS = "eu"           # Pinnacle vive acá; us no lo trae
MARKETS_DERIVADOS = "totals,spreads"
ODDS_FORMAT   = "decimal"
REQUEST_DELAY = 0.25           # seconds between Odds API calls

SEASON_WINDOWS: Dict[int, Tuple[str, str]] = {
    2024: ("2024-03-20", "2024-11-01"),
    2025: ("2025-03-18", "2025-11-01"),
    2026: ("2026-03-25", date.today().strftime("%Y-%m-%d")),
}
DEFAULT_SEASONS = [2024, 2025, 2026]

# Bookmaker keys that represent Pinnacle
_PINNACLE_KEYS = {"pinnacle"}

# MLB Stats API team names → Odds API team names (where they differ)
# Alias → nombre canónico, aplicado a AMBOS lados del emparejamiento.
#
# Antes esto era un mapa MLB→Odds de un solo sentido (`"athletics" →
# "oakland athletics"`) aplicado sólo al lado de MLB. Funcionó mientras la
# Odds API dijo "Oakland Athletics", y se rompió en silencio cuando pasó a
# decir "Athletics": la traducción llevaba el nombre de MLB LEJOS del de la
# API en vez de acercarlo. Costo medido el 2026-08-04: 162 juegos de 2025 y
# 114 de 2026 sin precio, todos de este equipo, atribuidos a "team name
# mismatch" sin que nadie mirara cuál.
#
# Canonizar los dos lados es inmune a que cualquiera de las dos fuentes cambie
# el nombre: alcanza con agregar el alias nuevo acá.
_TEAM_ALIASES: Dict[str, str] = {
    "oakland athletics":    "athletics",
    "sacramento athletics": "athletics",
    "las vegas athletics":  "athletics",
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

    derivados = _extract_derived(game, home, away)

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
        **derivados,
    }


def _extract_derived(game: dict, home: str, away: str) -> dict:
    """Total y runline, con el PUNTO FIRMADO de cada lado.

    Se guarda el punto además del precio porque sin él un precio de runline no
    significa nada: `analyze_runline` necesita saber quién pone el −1.5 para
    decidir cuál es el evento de cobertura de cada lado. Asumir local favorito
    es exactamente PURP-1, que publicó 55 picks con EV falso.

    Se prefiere Pinnacle para el par (su margen es el más limpio y es contra lo
    que se desvigoriza), y se guarda además la mejor línea disponible, que es a
    lo que realmente se apostaría. Los dos, no uno: la primera decide la
    probabilidad justa y la segunda el retorno.

    Un total sólo se acepta si Over y Under vienen en el MISMO punto de la
    MISMA casa; un runline, si los dos lados suman cero. Un par tomado de
    puntos distintos no es un mercado, es dos mercados mezclados.
    """
    tot_pin_over = tot_pin_under = tot_pin_point = None
    tot_best_over = tot_best_under = tot_best_point = None
    rl_pin_home = rl_pin_away = rl_pin_home_point = None
    rl_best_home = rl_best_away = rl_best_home_point = None

    for bk in game.get("bookmakers", []):
        es_pin = bk["key"] in _PINNACLE_KEYS
        for mkt in bk.get("markets", []):
            outs = mkt.get("outcomes", [])

            if mkt["key"] == "totals":
                over = next((o for o in outs if o["name"] == "Over"), None)
                under = next((o for o in outs if o["name"] == "Under"), None)
                if not (over and under):
                    continue
                if over.get("point") is None or over.get("point") != under.get("point"):
                    continue  # par de puntos distintos: no es un mercado
                po, pu, pt = over["price"], under["price"], float(over["point"])
                if not (po > 1.0 and pu > 1.0):
                    continue
                if es_pin:
                    tot_pin_over, tot_pin_under, tot_pin_point = po, pu, pt
                # "mejor" = el par con menor overround, no el mejor precio de un
                # lado: mezclar casas da un par que nadie puede apostar junto.
                if tot_best_over is None or (1/po + 1/pu) < (1/tot_best_over + 1/tot_best_under):
                    tot_best_over, tot_best_under, tot_best_point = po, pu, pt

            elif mkt["key"] == "spreads":
                oh = next((o for o in outs if o["name"] == home), None)
                oa = next((o for o in outs if o["name"] == away), None)
                if not (oh and oa):
                    continue
                if oh.get("point") is None or oa.get("point") is None:
                    continue
                if abs(float(oh["point"]) + float(oa["point"])) > 1e-9:
                    continue  # los dos lados deben sumar cero
                ph, pa_, pt = oh["price"], oa["price"], float(oh["point"])
                if not (ph > 1.0 and pa_ > 1.0):
                    continue
                if es_pin:
                    rl_pin_home, rl_pin_away, rl_pin_home_point = ph, pa_, pt
                if rl_best_home is None or (1/ph + 1/pa_) < (1/rl_best_home + 1/rl_best_away):
                    rl_best_home, rl_best_away, rl_best_home_point = ph, pa_, pt

    return {
        "total_point_pin":    tot_pin_point,
        "total_over_pin":     tot_pin_over,
        "total_under_pin":    tot_pin_under,
        "total_point_best":   tot_best_point,
        "total_over_best":    tot_best_over,
        "total_under_best":   tot_best_under,
        "rl_home_point_pin":  rl_pin_home_point,
        "rl_home_pin":        rl_pin_home,
        "rl_away_pin":        rl_pin_away,
        "rl_home_point_best": rl_best_home_point,
        "rl_home_best":       rl_best_home,
        "rl_away_best":       rl_best_away,
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

def fetch_odds_for_date(session: requests.Session, local_date: str,
                        solo_derivados: bool = False) -> Tuple[List[dict], dict]:
    """One bulk call — returns (list_of_game_objects, response_headers).

    `solo_derivados` pide únicamente totals+spreads de la región `eu`. Cuesta
    20 de cuota por día en vez de 60, y sigue trayendo Pinnacle — ver la nota
    de REGIONS_DERIVADOS. Se usa para rellenar los derivados de días cuyo
    moneyline ya está en la tabla, sin volver a pagar por el h2h.
    """
    snapshot_ts = local_date + SNAPSHOT_TIME
    data, headers = _request(session, f"{_ODDS_BASE}/historical/sports/baseball_mlb/odds/", {
        "apiKey":     _ODDS_KEY,
        "regions":    REGIONS_DERIVADOS if solo_derivados else REGIONS,
        "markets":    MARKETS_DERIVADOS if solo_derivados else MARKETS,
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
        low = name.strip().lower()
        return _TEAM_ALIASES.get(low, low)

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
        # _norm en AMBOS lados — ver la nota de _TEAM_ALIASES. Antes acá iba
        # un .lower() suelto, así que la canonización del lado de MLB no tenía
        # con qué encontrarse.
        key = (_norm(og["home_team"]), _norm(og["away_team"]))
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

    # Derivados (2026-08-04) — idempotente, mismo patrón que las columnas de
    # game_outcomes de abajo. Hasta ahora esta tabla era moneyline puro, y por
    # eso `backtest_and_retrain.py` sólo podía puntuar moneyline mientras el
    # 69 % de los picks publicados son runline o total.
    #
    # El PUNTO va junto al precio en todos los casos: un precio de runline sin
    # saber quién pone el −1.5 no identifica ningún evento, que es exactamente
    # la confusión que causó PURP-1.
    #
    # Se guardan DOS pares por mercado y no uno: el de Pinnacle, que es contra
    # el que se desvigoriza para obtener la probabilidad justa, y el mejor par
    # disponible, que es a lo que realmente se apostaría. El primero decide si
    # hay ventaja; el segundo, cuánto rinde.
    for col_def in [
        "total_point_pin    REAL", "total_over_pin     REAL", "total_under_pin    REAL",
        "total_point_best   REAL", "total_over_best    REAL", "total_under_best   REAL",
        "rl_home_point_pin  REAL", "rl_home_pin        REAL", "rl_away_pin        REAL",
        "rl_home_point_best REAL", "rl_home_best       REAL", "rl_away_best       REAL",
    ]:
        try:
            conn.execute(f"ALTER TABLE historical_odds ADD COLUMN {col_def}")
        except sqlite3.OperationalError:
            pass  # ya existe

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

_COLS_DERIVADOS = (
    "total_point_pin", "total_over_pin", "total_under_pin",
    "total_point_best", "total_over_best", "total_under_best",
    "rl_home_point_pin", "rl_home_pin", "rl_away_pin",
    "rl_home_point_best", "rl_home_best", "rl_away_best",
)


def insert_historical_odds(conn: sqlite3.Connection, rows: List[dict]) -> tuple:
    """Inserta filas nuevas y rellena los derivados de las que ya existían.

    Devuelve (nuevas, actualizadas).

    El `ON CONFLICT` toca EXCLUSIVAMENTE las columnas de derivados. Las de
    moneyline de una fila existente no se rozan: llevan meses ahí, el backtest
    canónico se midió con ellas, y re-escribirlas desde un snapshot nuevo
    cambiaría datos históricos por un efecto colateral de un backfill. Es la
    misma disciplina que CHRON-001 impuso en `game_outcomes`.
    """
    set_clause = ", ".join(f"{c}=excluded.{c}" for c in _COLS_DERIVADOS)
    cols_der = ", ".join(_COLS_DERIVADOS)
    vals_der = ", ".join(f":{c}" for c in _COLS_DERIVADOS)

    nuevas = actualizadas = 0
    for r in rows:
        ya_existia = conn.execute(
            "SELECT 1 FROM historical_odds WHERE game_pk=?", (r["game_pk"],)
        ).fetchone() is not None
        conn.execute(
            f"""
            INSERT INTO historical_odds (
                game_pk, game_date, season, odds_api_id,
                home_team, away_team, snapshot_ts,
                ml_home_best, ml_away_best, ml_home_best_bk, ml_away_best_bk,
                ml_home_cons, ml_away_cons,
                ml_home_pin,  ml_away_pin,
                fair_prob_home, fair_prob_away,
                n_bookmakers,
                {cols_der}
            ) VALUES (
                :game_pk, :game_date, :season, :odds_api_id,
                :home_team, :away_team, :snapshot_ts,
                :ml_home_best, :ml_away_best, :ml_home_best_bk, :ml_away_best_bk,
                :ml_home_cons, :ml_away_cons,
                :ml_home_pin,  :ml_away_pin,
                :fair_prob_home, :fair_prob_away,
                :n_bookmakers,
                {vals_der}
            )
            ON CONFLICT(game_pk) DO UPDATE SET {set_clause}
            """,
            r,
        )
        if ya_existia:
            actualizadas += 1
        else:
            nuevas += 1
    conn.commit()
    return nuevas, actualizadas


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
    parser.add_argument("--derivados", action="store_true",
                        help="rellenar SOLO total/runline en las filas ya existentes "
                             "(20 de cuota por día en vez de 60; no re-descarga h2h)")
    parser.add_argument("--limite-dias", type=int, default=0,
                        help="con --derivados: procesar como mucho N días (piloto)")
    parser.add_argument("--enrich-only", action="store_true",
                        help="Skip fetch, only run enrich_game_outcomes()")
    args = parser.parse_args()

    if not _ODDS_KEY:
        print("❌ ODDS_API_KEY not set in .env — aborting.")
        sys.exit(1)

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

        # Modo derivados: sólo días que YA están en la tabla — son los que
        # tienen moneyline y a los que les falta total/runline. Un día sin
        # moneyline no se toca: rellenar derivados de una fila que no existe
        # crearía una fila a medias.
        if args.derivados:
            sorted_dates = [d for d in sorted_dates
                            if any(gk in existing_pks for gk, _, _ in date_index[d])]
            if args.limite_dias:
                sorted_dates = sorted_dates[:args.limite_dias]

        total_dates = len(sorted_dates)
        remaining_api = None
        inserted = updated = skipped = no_odds = unmatched = 0

        print(f"  Game days: {total_dates}  |  Already in DB: {len(existing_pks)} games")
        if args.derivados:
            print(f"  Modo DERIVADOS: {MARKETS_DERIVADOS} en '{REGIONS_DERIVADOS}' "
                  f"— 20 de cuota/día, coste estimado {total_dates * 20:,}")
        else:
            print(f"\n  Fetching odds ({REGIONS}, cost ~20/day)...\n")

        batch: List[dict] = []

        for i, local_date in enumerate(sorted_dates, 1):
            date_games = date_index[local_date]   # [(game_pk, home, away)]
            snapshot_ts = local_date + SNAPSHOT_TIME

            # Skip date if all games already in DB. En modo derivados es al
            # revés: interesan justo los días que YA están.
            new_pks_on_date = [gk for gk, _, _ in date_games if gk not in existing_pks]
            if not args.derivados and not new_pks_on_date:
                skipped += len(date_games)
                if i % 20 == 0 or i == total_dates:
                    print(f"  {_bar(i, total_dates)}", end="\r")
                continue

            # Fetch odds snapshot for this date
            try:
                odds_games, headers = fetch_odds_for_date(
                    session, local_date, solo_derivados=args.derivados)
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
                # En modo derivados las filas que YA están son justamente el
                # objetivo: se les rellena total/runline vía el ON CONFLICT.
                # Una que NO esté se salta — insertarla sin moneyline dejaría
                # una fila a medias que el backtest no puede usar.
                if args.derivados:
                    if game_pk not in existing_pks:
                        skipped += 1
                        continue
                elif game_pk in existing_pks:
                    skipped += 1
                    continue

                odds_info = extract_odds(odds_game)
                # El gate de moneyline sólo aplica al modo normal: en derivados
                # no se pide h2h, así que ml_home_best viene None por diseño.
                if args.derivados:
                    if odds_info["total_over_pin"] is None and odds_info["rl_home_pin"] is None:
                        no_odds += 1
                        continue
                elif odds_info["ml_home_best"] is None:
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
                n, upd = insert_historical_odds(conn, batch)
                inserted += n
                updated  += upd
                skipped  += len(batch) - n - upd
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
