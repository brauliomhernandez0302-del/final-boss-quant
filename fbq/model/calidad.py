"""fbq/model/calidad.py — con qué historial se calculó cada emisión.

Separa dos cosas que se confunden fácil y no son lo mismo:

| | pregunta |
|---|---|
| `origen` | ¿la predicción se **escribió** antes del primer lanzamiento? |
| `calidad_datos` | ¿con qué **entradas** se calculó? |

Una emisión puede ser impecablemente prospectiva y haber usado un historial
incompleto. Las dos cosas son ciertas a la vez y **ninguna anula a la otra**:
degradar la primera por la segunda escondería que la predicción sí se hizo
antes; ignorar la segunda haría pasar por buena una entrada que no lo era.

## El caso que obligó a construir esto

`results.db` no tenía nada entre **2026-08-07 y 2026-09-04** — 28 jornadas. Se
descubrió al automatizar la incorporación de resultados, el 2026-09-07 a las
13:13 UTC, y se rellenó (377 resultados). Pero las emisiones de los primeros
15 partidos ya se habían calculado **con ese hueco abierto**, así que sus
perfiles de equipo usaron una ventana de 162 partidos a la que le faltaba casi
un mes.

**No es una fuga** —todo lo usado era anterior al corte— pero sí una
degradación, y la regla del primer par la congela. No se sustituyen: se
**anotan**.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fbq.evaluator.frame import DB_RESULTADOS
from fbq.model.prospectiva import DB_PATH as DB_PRED
from fbq.model.prospectiva import Prospectiva

log = logging.getLogger(__name__)

# Cuántas jornadas vacías en los últimos 60 días bastan para llamar al
# historial incompleto. Una jornada sin juegos es normal (día libre de toda la
# liga, ~4 al año); veinte no lo son.
MAX_DIAS_VACIOS = 5


def huella_historial(*, db: Path = DB_RESULTADOS, hasta: Optional[str] = None,
                     ventana: int = 60) -> Dict[str, Any]:
    """Estado del almacén de resultados: hasta dónde llega y qué le falta."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        filas, ultimo = con.execute(
            "SELECT COUNT(*), MAX(official_date) FROM resultado").fetchone()
        fin = date.fromisoformat(hasta or ultimo)
        desde = fin - timedelta(days=ventana)
        con_datos = {r[0] for r in con.execute(
            "SELECT DISTINCT official_date FROM resultado "
            "WHERE official_date BETWEEN ? AND ?", (desde.isoformat(), fin.isoformat()))}
    finally:
        con.close()
    vacios = sum(1 for i in range(ventana + 1)
                 if (desde + timedelta(days=i)).isoformat() not in con_datos)
    return {"historial_hasta": ultimo, "historial_filas": filas,
            "dias_sin_datos": vacios,
            "calidad": "completo" if vacios <= MAX_DIAS_VACIOS else "historial_incompleto"}


def anotar_pendientes(*, db_pred: Path = DB_PRED, calidad: Optional[Dict[str, Any]] = None,
                      motivo: str = "") -> Dict[str, int]:
    """Anota todas las emisiones que todavía no tienen calidad registrada."""
    store = Prospectiva(db_pred)
    with store._conn() as c:
        filas = [dict(r) for r in c.execute(
            """SELECT game_pk, version, corte, modelo_sha FROM prediccion p
               WHERE NOT EXISTS (SELECT 1 FROM calidad_datos q
                                 WHERE q.game_pk=p.game_pk AND q.version=p.version
                                   AND q.corte=p.corte AND q.modelo_sha=p.modelo_sha)""")]
    huella = calidad or huella_historial()
    for f in filas:
        f.update(huella)
        f["motivo"] = motivo or (
            "anotación retroactiva: el estado del almacén se conoce hoy, no al "
            "momento de emitir" if huella["calidad"] == "completo" else
            "results.db sin datos entre 2026-08-07 y 2026-09-04 al momento de emitir")
    return {"anotadas": store.anotar_calidad(filas), "candidatas": len(filas)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calidad", choices=["completo", "historial_incompleto"],
                    default=None, help="fuerza la calidad (anotación retroactiva)")
    ap.add_argument("--motivo", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    h = huella_historial()
    if args.calidad:
        h = {**h, "calidad": args.calidad}
    log.info("huella del historial: %s", h)
    log.info("%s", anotar_pendientes(calidad=h, motivo=args.motivo))
    log.info("almacén: %s", json.dumps(Prospectiva().resumen(), ensure_ascii=False))


if __name__ == "__main__":
    main()
