"""Captura los abridores anunciados. Gratis, sin clave, sin cuota de pago.

    python3 -m fbq.anuncios.capturar --dias 3

Pensado para correr junto a la captura de precios: lo que importa es tener
observaciones ANTES de cada corte, así que la frecuencia útil es la misma que
la del precio de referencia.

**No instala ningún cron.** Programarlo es una decisión operativa del dueño, y
este proyecto convive con otro que ya tiene su propia programación.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta

from fbq.anuncios.store import AnunciosStore
from fbq.sources import mlb_stats

log = logging.getLogger(__name__)


def capturar(dias: int = 3, *, store: AnunciosStore | None = None) -> dict:
    store = store or AnunciosStore()
    total = {"juegos": 0, "nuevos": 0, "sin_cambio": 0}
    hoy = date.today()
    for i in range(dias):
        fecha = (hoy + timedelta(days=i)).isoformat()
        filas = mlb_stats.probables(fecha)
        r = store.registrar(filas, fecha=fecha)
        log.info("  %s: %s", fecha, r)
        for k in total:
            total[k] += r[k]
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dias", type=int, default=3,
                    help="cuántos días hacia adelante barrer (default 3)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("total: %s", capturar(args.dias))
    log.info("almacén: %s", AnunciosStore().resumen())


if __name__ == "__main__":
    main()
