"""evaluator/detalle.py — una fila por juego, con la pérdida y de qué precio salió.

Existe por una discrepancia real. El informe del 2026-09-06 publicó tres Brier
—todos 0,241333 · prepartido 0,242124 · en vivo 0,227976— presentados de forma
que invitaban a leerlos como una partición. No lo son, y la media ponderada de
los dos grupos da 0,241809, no 0,241333.

La causa no está en la aritmética ni en los datos: está en las COHORTES. El
marco "todos" no es "los mismos juegos más 139". Para los juegos que además
tienen captura en vivo propia, `solo_pregame=False` se queda con la última
cotización, que es la EN VIVO, así que cambia el precio del mismo juego. Son 25
juegos, y su Brier pasa de 0,234231 (precio prepartido) a 0,115213 (precio en
vivo) — un precio que ya vio parte del partido.

Un resumen que no se puede reconstruir desde su detalle no es un resumen, es
una afirmación. Este módulo publica el detalle para que la reconstrucción sea
mecánica y la próxima discrepancia se vea sola.

**La media es simple, por juego.** No hay pesos por temporada ni de ninguna
otra clase: `brier()` es `mean((p-y)**2)` sobre las filas del marco. Las
temporadas pesan lo que pesan por su número de juegos, que es lo que hace que
2026 —temporada incompleta— pese menos.

Sólo lectura: nada de este archivo escribe en ninguna base.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fbq.evaluator.frame import DB_MERCADO, DB_RESULTADOS, _devig_multiplicativo

CAMPOS = [
    "game_pk", "official_date", "season", "home_team", "away_team",
    "inicio_utc",
    "snapshot_ts", "p_home", "brier", "clasificacion_temporal",
    "snapshot_ts_prepartido", "p_home_prepartido", "brier_prepartido",
    "resultado_local", "en_marco_prepartido", "precio_cambia_al_filtrar",
]

_SQL = """
WITH ultima AS (
    SELECT s.event_id, s.side, s.price_dec, s.captured_at, s.commence_time,
           ROW_NUMBER() OVER (PARTITION BY s.event_id, s.side
                              ORDER BY s.captured_at DESC, s.id DESC) AS rn
    FROM mkt.odds_snapshot s
    WHERE s.sport_key = ? AND s.market = 'h2h' AND s.book = ?
),
ultima_pre AS (
    SELECT s.event_id, s.side, s.price_dec, s.captured_at,
           ROW_NUMBER() OVER (PARTITION BY s.event_id, s.side
                              ORDER BY s.captured_at DESC, s.id DESC) AS rn
    FROM mkt.odds_snapshot s
    WHERE s.sport_key = ? AND s.market = 'h2h' AND s.book = ?
      AND s.captured_at < s.commence_time
)
SELECT r.game_pk, r.official_date, r.season, r.home_team, r.away_team,
       r.home_won,
       l.commence_time AS inicio_utc,
       MAX(CASE WHEN u.side='home' THEN u.price_dec END)   AS pin_home,
       MAX(CASE WHEN u.side='away' THEN u.price_dec END)   AS pin_away,
       MAX(CASE WHEN u.side='home' THEN u.captured_at END) AS cap_home,
       MAX(CASE WHEN p.side='home' THEN p.price_dec END)   AS pre_home,
       MAX(CASE WHEN p.side='away' THEN p.price_dec END)   AS pre_away,
       MAX(CASE WHEN p.side='home' THEN p.captured_at END) AS cap_pre_home
FROM resultado r
JOIN mkt.event_link l ON l.game_pk = r.game_pk
LEFT JOIN ultima     u ON u.event_id = l.event_id AND u.rn = 1
LEFT JOIN ultima_pre p ON p.event_id = l.event_id AND p.rn = 1
WHERE r.season IN ({temporadas})
GROUP BY r.game_pk
ORDER BY r.official_date, r.game_pk
"""


def detalle_por_juego(
    seasons: Sequence[int] = (2024, 2025, 2026),
    *,
    db_mercado: Path = DB_MERCADO,
    db_resultados: Path = DB_RESULTADOS,
    book: str = "pinnacle",
    sport_key: str = "baseball_mlb",
) -> List[Dict[str, Any]]:
    """Una fila por juego del universo "todos" (el que tiene par sin filtrar).

    Cada fila lleva las DOS lecturas del mismo juego —la última cotización y la
    última PRE-JUEGO— para que se vea cuándo son distintas en vez de tener que
    deducirlo de dos resúmenes.
    """
    con = sqlite3.connect(f"file:{db_resultados}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        con.execute("ATTACH DATABASE ? AS mkt", (f"file:{db_mercado}?mode=ro",))
        sql = _SQL.format(temporadas=",".join("?" * len(seasons)))
        crudas = con.execute(
            sql, (sport_key, book, sport_key, book, *seasons)).fetchall()
    finally:
        con.close()

    filas: List[Dict[str, Any]] = []
    for r in crudas:
        if not (r["pin_home"] and r["pin_away"]):
            continue  # sin par no hay línea justa: mismo criterio que el marco
        y = float(r["home_won"])
        p, _ = _devig_multiplicativo(1.0 * r["pin_home"], 1.0 * r["pin_away"])
        cap = r["cap_home"]
        inicio = r["inicio_utc"]
        en_vivo = bool(cap and inicio and cap >= inicio)

        hay_pre = bool(r["pre_home"] and r["pre_away"])
        p_pre = (_devig_multiplicativo(1.0 * r["pre_home"], 1.0 * r["pre_away"])[0]
                 if hay_pre else None)

        filas.append({
            "game_pk": r["game_pk"],
            "official_date": r["official_date"],
            "season": r["season"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "inicio_utc": inicio,
            "snapshot_ts": cap,
            "p_home": p,
            "brier": (p - y) ** 2,
            "clasificacion_temporal": "en_vivo" if en_vivo else "prepartido",
            "snapshot_ts_prepartido": r["cap_pre_home"],
            "p_home_prepartido": p_pre,
            "brier_prepartido": (p_pre - y) ** 2 if hay_pre else None,
            "resultado_local": int(y),
            "en_marco_prepartido": hay_pre,
            "precio_cambia_al_filtrar": bool(hay_pre and abs(p - p_pre) > 1e-12),
        })
    return filas


def resumen_desde_detalle(filas: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Las tres cohortes, calculadas desde el detalle y no aparte de él.

    `todos` y `prepartido` NO son un grupo y su complemento: comparten juegos
    que aparecen en los dos con precios distintos. Por eso se publica también
    `prepartido_de_los_mismos_juegos`, que sí permite comparar manzanas con
    manzanas, y el puente aritmético que explica la diferencia.
    """
    def media(vals):
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    todos = list(filas)
    solo_pre = [f for f in filas if f["clasificacion_temporal"] == "prepartido"]
    solo_vivo = [f for f in filas if f["clasificacion_temporal"] == "en_vivo"]
    marco_pre = [f for f in filas if f["en_marco_prepartido"]]
    cambian = [f for f in filas if f["precio_cambia_al_filtrar"]]

    n_pre, b_pre = len(marco_pre), media(f["brier_prepartido"] for f in marco_pre)
    n_vivo_solo = len([f for f in solo_vivo if not f["en_marco_prepartido"]])
    b_vivo_solo = media(f["brier"] for f in solo_vivo if not f["en_marco_prepartido"])

    reconstruido = ((n_pre * b_pre + n_vivo_solo * b_vivo_solo) / (n_pre + n_vivo_solo)
                    if b_pre is not None and b_vivo_solo is not None else None)

    return {
        "media": "simple por juego, sin pesos de ninguna clase",
        "todos": {"n": len(todos), "brier": media(f["brier"] for f in todos)},
        "marco_prepartido": {"n": n_pre, "brier": b_pre},
        "solo_prepartido_en_todos": {
            "n": len(solo_pre), "brier": media(f["brier"] for f in solo_pre)},
        "solo_en_vivo_en_todos": {
            "n": len(solo_vivo), "brier": media(f["brier"] for f in solo_vivo)},
        "sin_precio_prepartido": {"n": n_vivo_solo, "brier": b_vivo_solo},
        "juegos_en_ambos_con_precio_distinto": {
            "n": len(cambian),
            "brier_con_precio_de_todos": media(f["brier"] for f in cambian),
            "brier_con_precio_prepartido": media(f["brier_prepartido"] for f in cambian),
        },
        "puente": {
            "reconstruccion_de_las_dos_cohortes": reconstruido,
            "brier_todos_publicado": media(f["brier"] for f in todos),
            "diferencia": (media(f["brier"] for f in todos) - reconstruido
                           if reconstruido is not None else None),
        },
        "por_temporada": {
            int(s): {
                "n": len([f for f in marco_pre if f["season"] == s]),
                "brier": media(f["brier_prepartido"] for f in marco_pre if f["season"] == s),
            }
            for s in sorted({f["season"] for f in marco_pre})
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--salida", type=Path, default=None,
                    help="CSV con una fila por juego")
    args = ap.parse_args()

    filas = detalle_por_juego(args.seasons)
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        with args.salida.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CAMPOS)
            w.writeheader()
            w.writerows(filas)
        print(f"detalle: {len(filas)} juegos → {args.salida}")

    import json
    print(json.dumps(resumen_desde_detalle(filas), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
