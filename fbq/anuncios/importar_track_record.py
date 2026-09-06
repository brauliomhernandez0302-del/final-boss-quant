"""Rescata al almacén de anuncios lo único capturado prospectivamente que existe.

Origen: `track_record.db`, tabla `picks`, campo `pipeline_json`, desde el commit
`127bac6` del sistema anterior (2026-08-02) — el que empezó a guardar las
ENTRADAS del pipeline y no sólo sus resultados.

Cada fila trae `home_pitcher_id` / `away_pitcher_id` junto a un `published_at`
que es **el instante real en que el sistema lo observó**, no una reconstrucción.
Eso lo vuelve la única evidencia histórica utilizable: la API no guarda historial
del anuncio, así que este dato no se puede volver a obtener de ninguna fuente.

**Se importa el instante, no sólo la identidad.** Un anuncio sin su hora de
observación no sirve para predecir, porque no se puede afirmar que estuviera
disponible antes del corte.

Uso:
    python3 -m fbq.anuncios.importar_track_record --origen /ruta/track_record.db
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fbq.anuncios.store import AnunciosStore
from fbq.core.clock import normalizar_utc, tiene_hora

log = logging.getLogger(__name__)

ORIGEN = Path("/home/raulio/anterior/data/track_record.db")
FUENTE = "track_record.pipeline_json (sistema anterior, commit 127bac6)"


def _observaciones(origen: Path) -> List[Dict[str, Any]]:
    """Una observación por (pick, instante), en orden cronológico.

    NO se deduplica por juego acá: si el sistema publicó dos veces el mismo
    juego con abridores distintos, las dos observaciones son datos y el almacén
    —append-only, deduplicado por CAMBIO— resuelve cuál conservar.
    """
    con = sqlite3.connect(f"file:{origen}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        filas = con.execute(
            """SELECT game_pk, published_at, commence_time, game_date, pipeline_json
               FROM picks WHERE pipeline_json IS NOT NULL AND published_at IS NOT NULL
               ORDER BY published_at"""
        ).fetchall()
    finally:
        con.close()

    salida: List[Dict[str, Any]] = []
    for r in filas:
        try:
            entradas = (json.loads(r["pipeline_json"]) or {}).get("inputs") or {}
        except (ValueError, TypeError):
            continue
        if not entradas.get("home_pitcher_id") and not entradas.get("away_pitcher_id"):
            continue
        if not tiene_hora(r["published_at"] or ""):
            continue
        salida.append({
            "observado_en": normalizar_utc(r["published_at"]),
            "game_pk": int(r["game_pk"]),
            "official_date": entradas.get("official_date") or r["game_date"],
            "commence_time": r["commence_time"],
            "estado": FUENTE,
            "home_pitcher_id": entradas.get("home_pitcher_id"),
            "home_pitcher": entradas.get("home_pitcher"),
            "away_pitcher_id": entradas.get("away_pitcher_id"),
            "away_pitcher": entradas.get("away_pitcher"),
        })
    return salida


def importar(origen: Path = ORIGEN, *, store: Optional[AnunciosStore] = None) -> Dict[str, int]:
    store = store or AnunciosStore()
    obs = _observaciones(origen)
    resumen = {"observaciones": len(obs), "nuevos": 0, "sin_cambio": 0}
    # Una llamada por instante: el almacén deduplica por cambio dentro de cada
    # barrida, y mezclar instantes distintos en una sola perdería el orden.
    for o in obs:
        r = store.registrar([o], fecha=o["official_date"] or "",
                            observado_en=o["observado_en"])
        resumen["nuevos"] += r["nuevos"]
        resumen["sin_cambio"] += r["sin_cambio"]
    return resumen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origen", type=Path, default=ORIGEN)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("import: %s", importar(args.origen))
    log.info("almacén: %s", AnunciosStore().resumen())


if __name__ == "__main__":
    main()
