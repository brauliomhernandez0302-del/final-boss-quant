"""Captura los abridores anunciados. Gratis, sin clave, sin cuota de pago.

    python3 -m fbq.anuncios.capturar --dias 3

Pensado para correr junto a la captura de precios: lo que importa es tener
observaciones ANTES de cada corte, así que la frecuencia útil es la misma que
la del precio de referencia.

## Dos protecciones, y las dos por defectos que este proyecto ya pagó

**Cerrojo de archivo.** Dos corridas simultáneas insertarían la misma barrida
dos veces y, peor, cada una vería un estado distinto del almacén al deduplicar
por cambio. El sistema anterior perdió una calibración entera por escritura
concurrente sin protección —`ml_state`, 2026-07-11— y la respuesta entonces fue
un lockfile; ésta es la misma respuesta. Si el cerrojo está tomado, la corrida
**sale sin hacer nada y sin error**: es lo correcto para algo que corre cada
media hora.

**Registro de errores.** Un fallo de red no puede quedarse en el código de
salida: el cron lo perdería. Se escribe a `logs/anuncios.log` con el detalle, y
el proceso termina con código distinto de cero para que cualquier supervisor lo
vea. Un fallo silencioso en una captura prospectiva es la peor clase de fallo,
porque el hueco que deja no se puede rellenar después.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from fbq.anuncios.store import AnunciosStore
from fbq.core.cerrojo import Cerrojo, Ocupado
from fbq.sources import mlb_stats

log = logging.getLogger(__name__)

RAIZ = Path(__file__).parent.parent.parent
LOG = RAIZ / "logs" / "anuncios.log"
CERROJO = RAIZ / "logs" / "anuncios.lock"


# `Ocupado` y `_Cerrojo` viven en `fbq.core.cerrojo`: el mismo modo de falla
# —dos escritores contra un almacén que deduplica por cambio— lo tienen también
# la incorporación de resultados y cualquier otra barrida futura. Se re-exportan
# acá para no romper a quien los importe de este módulo.
_Cerrojo = Cerrojo


def capturar(dias: int = 3, *, store: Optional[AnunciosStore] = None) -> dict:
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


def _configurar_log(verboso: bool) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    formato = "%(asctime)s %(levelname)s %(name)s %(message)s"
    manejadores: list = [logging.FileHandler(LOG, encoding="utf-8")]
    if verboso:
        manejadores.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, format=formato, handlers=manejadores,
                        force=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dias", type=int, default=3,
                    help="cuántos días hacia adelante barrer (default 3)")
    ap.add_argument("-v", "--verboso", action="store_true",
                    help="además del archivo, escribe a stderr")
    args = ap.parse_args()
    _configurar_log(args.verboso)

    try:
        with Cerrojo(CERROJO, "captura"):
            total = capturar(args.dias)
            log.info("total: %s | almacén: %s", total, AnunciosStore().resumen())
        return 0
    except Ocupado as exc:
        # No es un error: es el cerrojo haciendo su trabajo.
        log.info("salteada: %s", exc)
        return 0
    except Exception:                                     # noqa: BLE001
        # Un hueco en una captura prospectiva no se rellena después, así que el
        # fallo se deja escrito con traza completa y se sale con código != 0.
        log.exception("FALLÓ la captura de anuncios")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
