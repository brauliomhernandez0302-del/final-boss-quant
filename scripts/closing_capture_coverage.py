#!/usr/bin/env python3
"""Cobertura de captura de cierre por familia de mercado — medición, no decisión.

Responde una sola pregunta, repetible: **de los picks que hoy existen, ¿a
cuántos les podría guardar un cierre gradable la captura actual?**, desglosado
por familia (moneyline / runline / total).

Por qué existe: hasta el 2026-07-26 `capture_closing_lines.py` solo tenía ramas
de moneyline, así que runline y total tenían atrición del 100% por construcción
(ver `fix(clv): dejar de tirar el precio de cierre...`). Antes de reescribir el
protocolo hace falta saber qué cobertura real da el arreglo — no la teórica.

Dos modos, ninguno escribe en la DB:

  --stored   (default) lee lo YA guardado en `picks`: qué familias tienen
             precio de cierre, par de Pinnacle de su propio mercado, punto, y
             movimiento de línea. Es la foto histórica, incluida la pérdida
             previa al arreglo.

  --live     además consulta el mercado actual (una sola llamada cacheada a
             `_get_raw_events`, cero quota extra por pick) para los picks cuyo
             juego todavía no empezó, y reporta cuántos TENDRÍAN un cierre
             gradable si la barrida corriera ahora. Es la prueba de que el
             arreglo cubre lo que dice cubrir, sobre datos reales.

"Gradable" acá = existe precio del lado del pick. "Devig-able" = existen además
AMBOS lados de Pinnacle para ese mismo mercado. Se reportan por separado a
propósito: la métrica primaria del protocolo se define sobre el devig de
Pinnacle, así que la segunda es la que manda para una muestra primaria y la
primera es el piso.

Uso:
    python3 scripts/closing_capture_coverage.py
    python3 scripts/closing_capture_coverage.py --live
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "track_record.db"

FAMILY = {
    "ML_HOME": "MONEYLINE", "ML_AWAY": "MONEYLINE",
    "RL_HOME": "RUNLINE", "RL_AWAY": "RUNLINE",
    "OVER": "TOTAL", "UNDER": "TOTAL",
}
ORDER = ["MONEYLINE", "RUNLINE", "TOTAL", "OTRO"]


def _family(market: str) -> str:
    return FAMILY.get(market, "OTRO")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def report_stored(conn: sqlite3.Connection, since: str | None) -> None:
    where = "WHERE date(published_at) >= ?" if since else ""
    params = [since] if since else []
    rows = conn.execute(f"SELECT * FROM picks {where}", params).fetchall()

    # Una DB que todavía no corrió la migración del 2026-07-26 no tiene las
    # columnas nuevas. Eso es información, no un error: significa que ninguna
    # barrida corrió aún con el arreglo.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(picks)")}
    has_new = {"closing_pin_side", "closing_pin_opposite", "closing_point_moved"} <= cols
    if not has_new:
        print("\n[aviso] esta DB aún no tiene las columnas de cierre de derivados "
              "(la migración corre al instanciar TrackRecordDB) — se reporta lo que hay.")

    def _get(row, key):
        return row[key] if has_new else None

    agg: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        f = _family(r["market"])
        agg[f]["picks"] += 1
        agg[f]["snapshot"] += bool(r["closing_captured_at"])
        agg[f]["precio_lado"] += bool(r["closing_odds_decimal"])
        agg[f]["pin_par_propio"] += bool(_get(r, "closing_pin_side") and _get(r, "closing_pin_opposite"))
        agg[f]["clv"] += bool(r["clv_pct"] is not None)
        moved = _get(r, "closing_point_moved")
        if moved is not None:
            agg[f]["con_punto"] += 1
            agg[f]["punto_movido"] += int(moved)

    print("\n== GUARDADO (histórico real, incluye la pérdida previa al arreglo) ==")
    print(f"{'familia':<10} {'picks':>6} {'snapshot':>9} {'precio lado':>12} "
          f"{'par Pinnacle':>13} {'clv_pct':>8} {'punto movido':>13}")
    for f in ORDER:
        if f not in agg:
            continue
        a = agg[f]
        movido = f"{a['punto_movido']}/{a['con_punto']}" if a["con_punto"] else "—"
        print(f"{f:<10} {a['picks']:>6} {a['snapshot']:>9} {a['precio_lado']:>12} "
              f"{a['pin_par_propio']:>13} {a['clv']:>8} {movido:>13}")


def report_live(conn: sqlite3.Connection) -> None:
    from odds_fetcher import get_best_odds_for_teams

    now = datetime.now(timezone.utc).isoformat()
    picks = conn.execute(
        "SELECT * FROM picks WHERE commence_time IS NOT NULL AND commence_time > ? "
        "ORDER BY commence_time",
        (now,),
    ).fetchall()

    if not picks:
        print("\n== EN VIVO ==\nNingún pick con juego pendiente ahora mismo — nada que medir.")
        return

    # Precios por (game_pk) — una consulta al mercado por juego, no por pick.
    odds_by_game: dict = {}
    agg: dict = defaultdict(lambda: defaultdict(int))

    for p in picks:
        gpk = p["game_pk"]
        if gpk not in odds_by_game:
            odds_by_game[gpk] = get_best_odds_for_teams(
                p["home_team"], p["away_team"], p["commence_time"]
            ) or {}
        o = odds_by_game[gpk]
        f = _family(p["market"])
        agg[f]["picks"] += 1
        if not o:
            continue
        agg[f]["evento_encontrado"] += 1

        m = p["market"]
        if m == "ML_HOME":
            side, pin_a, pin_b, point = o.get("ml_home"), o.get("pin_home"), o.get("pin_away"), None
        elif m == "ML_AWAY":
            side, pin_a, pin_b, point = o.get("ml_away"), o.get("pin_away"), o.get("pin_home"), None
        elif m == "OVER":
            side, pin_a, pin_b = o.get("total_over"), o.get("pin_total_over"), o.get("pin_total_under")
            point = o.get("pin_total_point") or o.get("total_line")
        elif m == "UNDER":
            side, pin_a, pin_b = o.get("total_under"), o.get("pin_total_under"), o.get("pin_total_over")
            point = o.get("pin_total_point") or o.get("total_line")
        elif m == "RL_HOME":
            side, pin_a, pin_b = o.get("runline_home"), o.get("pin_runline_home"), o.get("pin_runline_away")
            point = o.get("pin_runline_home_point") or o.get("runline_home_point")
        elif m == "RL_AWAY":
            side, pin_a, pin_b = o.get("runline_away"), o.get("pin_runline_away"), o.get("pin_runline_home")
            point = o.get("pin_runline_away_point") or o.get("runline_away_point")
        else:
            continue

        agg[f]["gradable"] += bool(side)
        agg[f]["devigable"] += bool(pin_a and pin_b)

        taken = p["total_line"] if f == "TOTAL" else (p["runline_point"] if f == "RUNLINE" else None)
        if point is not None and taken is not None:
            agg[f]["comparable_punto"] += 1
            if abs(abs(point) - abs(taken)) > 0.01:
                agg[f]["punto_movido"] += 1

    print(f"\n== EN VIVO (si la barrida corriera ahora, {len(picks)} picks pendientes) ==")
    print(f"{'familia':<10} {'picks':>6} {'evento':>8} {'gradable':>10} {'devig-able':>11} {'punto movido':>13}")
    for f in ORDER:
        if f not in agg:
            continue
        a = agg[f]
        movido = f"{a['punto_movido']}/{a['comparable_punto']}" if a["comparable_punto"] else "—"
        print(f"{f:<10} {a['picks']:>6} {a['evento_encontrado']:>8} {a['gradable']:>10} "
              f"{a['devigable']:>11} {movido:>13}")
    print("\n  gradable   = hay precio de cierre del lado del pick")
    print("  devig-able = hay AMBOS lados de Pinnacle en el mercado del pick "
          "(lo que pide la métrica primaria)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true",
                    help="además consulta el mercado actual para los picks pendientes")
    ap.add_argument("--since", default=None, help="solo picks publicados desde esta fecha (YYYY-MM-DD)")
    args = ap.parse_args()

    conn = _connect()
    print("Cobertura de captura de cierre — medición, no decide nada. Solo lectura.")
    report_stored(conn, args.since)
    if args.live:
        report_live(conn)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
