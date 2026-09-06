"""Importa los precios históricos del sistema anterior al almacén propio.

Una sola vez. Corta la última dependencia de datos con `legacy/`: después de
esto, `evaluator/` y `features/` leen sólo de almacenes propios.

Lo que se importa son HECHOS —precios observados el día del juego— y por eso
sobreviven a la invalidación del sistema anterior. Lo que NO se importa es
nada derivado de un modelo.

## Tres decisiones que quedan escritas acá

**El consenso entra con nombre `_consensus`.** No es una casa: es un agregado
calculado sobre ~28 libros al momento de la captura. El guion bajo marca que es
derivado, para que ninguna consulta lo confunda con un precio real cotizado.
Se importa porque es un dato medido que no se puede reconstruir —sólo tenemos
dos libros por juego del histórico, no el board completo.

**El mejor precio entra con el nombre REAL de su casa**, que el histórico
guarda por lado (`ml_home_best_bk`, `ml_away_best_bk`). Pueden ser dos casas
distintas para el mismo juego, y así queda.

**Todas las filas llevan `captured_at` = 17:00Z del día del juego**, que es lo
que el sistema anterior pidió al proveedor. No es el cierre ni una trayectoria:
es una foto diaria, y hay que tratarla como tal. Se conserva el instante real
para que el filtro pre-juego funcione.

Uso:
    python3 -m fbq.market.importar_historico --dry-run
    python3 -m fbq.market.importar_historico
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fbq.market.store import MarketStore

log = logging.getLogger(__name__)

ORIGEN = Path(__file__).parent.parent.parent / "data" / "predictions_history.db"
DEPORTE = "baseball_mlb"

# Marca de agregado derivado. El guion bajo es deliberado: ninguna casa real
# empieza así, y una consulta que filtre por libro no lo va a confundir.
CONSENSO = "_consensus"


def _filas(origen: Path) -> List[sqlite3.Row]:
    con = sqlite3.connect(f"file:{origen}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        # `game_outcomes.game_date` es el timestamp UTC de inicio: hace falta
        # para el filtro pre-juego del almacén. El histórico no lo trae.
        return con.execute(
            """SELECT h.*, g.game_date AS inicio_utc
               FROM historical_odds h
               LEFT JOIN game_outcomes g ON g.game_pk = h.game_pk
               ORDER BY h.snapshot_ts, h.game_pk"""
        ).fetchall()
    finally:
        con.close()


def _cotizaciones(r: sqlite3.Row) -> List[Tuple[str, str, str, Optional[float], float]]:
    """(libro, mercado, lado, punto, precio) de una fila del histórico.

    El punto va FIRMADO y pegado a su lado, igual que en la captura en vivo:
    el visitante cotiza el complemento del punto del local.
    """
    out: List[Tuple[str, str, str, Optional[float], float]] = []

    def agregar(libro, mercado, lado, punto, precio):
        if libro and precio and precio > 1.0:
            out.append((str(libro), mercado, lado, punto, float(precio)))

    # Moneyline
    agregar("pinnacle", "h2h", "home", None, r["ml_home_pin"])
    agregar("pinnacle", "h2h", "away", None, r["ml_away_pin"])
    agregar(CONSENSO, "h2h", "home", None, r["ml_home_cons"])
    agregar(CONSENSO, "h2h", "away", None, r["ml_away_cons"])
    agregar(r["ml_home_best_bk"], "h2h", "home", None, r["ml_home_best"])
    agregar(r["ml_away_best_bk"], "h2h", "away", None, r["ml_away_best"])

    # Total — el punto es el mismo para los dos lados
    p = r["total_point_pin"]
    agregar("pinnacle", "totals", "over", p, r["total_over_pin"])
    agregar("pinnacle", "totals", "under", p, r["total_under_pin"])
    pb = r["total_point_best"]
    # El histórico no guarda QUÉ casa dio el mejor total, así que entra como
    # agregado derivado y no como precio cotizado por alguien.
    agregar(CONSENSO, "totals", "over", pb, r["total_over_best"])
    agregar(CONSENSO, "totals", "under", pb, r["total_under_best"])

    # Runline — el visitante lleva el complemento
    ph = r["rl_home_point_pin"]
    agregar("pinnacle", "spreads", "home", ph, r["rl_home_pin"])
    agregar("pinnacle", "spreads", "away", -ph if ph is not None else None, r["rl_away_pin"])
    phb = r["rl_home_point_best"]
    agregar(CONSENSO, "spreads", "home", phb, r["rl_home_best"])
    agregar(CONSENSO, "spreads", "away", -phb if phb is not None else None, r["rl_away_best"])

    return out


def importar(store: MarketStore, *, origen: Path = ORIGEN,
             dry_run: bool = False) -> Dict[str, int]:
    filas = _filas(origen)
    resumen = {"juegos": 0, "sin_event_id": 0, "sin_inicio": 0, "cotizaciones": 0}

    lote: List[Tuple] = []
    for r in filas:
        ev = r["odds_api_id"]
        if not ev:
            # Sin identidad del proveedor no hay serie temporal posible. Misma
            # regla que la captura en vivo: se cuenta, no se inventa una llave.
            resumen["sin_event_id"] += 1
            continue
        inicio = r["inicio_utc"]
        if not inicio:
            resumen["sin_inicio"] += 1
            continue

        resumen["juegos"] += 1
        for libro, mercado, lado, punto, precio in _cotizaciones(r):
            lote.append((
                r["snapshot_ts"], None, DEPORTE, ev, inicio,
                r["home_team"], r["away_team"], libro, mercado, lado,
                punto, precio,
            ))
    resumen["cotizaciones"] = len(lote)

    if dry_run or not lote:
        return resumen

    with store._conn() as conn:
        conn.executemany(
            """INSERT INTO odds_snapshot
               (captured_at, book_update, sport_key, event_id, commence_time,
                home_team, away_team, book, market, side, point, price_dec)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            lote,
        )
        conn.execute(
            """INSERT INTO sweep (captured_at, sport_key, n_events, n_rows_new,
                                  n_unchanged, n_skipped, note)
               VALUES (?,?,?,?,?,?,?)""",
            (filas[0]["snapshot_ts"] if filas else "", DEPORTE, resumen["juegos"],
             len(lote), 0, resumen["sin_event_id"] + resumen["sin_inicio"],
             "import histórico del sistema anterior — foto diaria 17:00Z, no trayectoria"),
        )
    return resumen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origen", type=Path, default=ORIGEN)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    store = MarketStore()
    antes = store.summary()
    r = importar(store, origen=args.origen, dry_run=args.dry_run)
    log.info("Import: %s", r)
    if not args.dry_run:
        log.info("almacén: %s filas → %s", antes["rows"], store.summary()["rows"])


if __name__ == "__main__":
    main()
