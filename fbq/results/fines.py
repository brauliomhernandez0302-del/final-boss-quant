"""fbq/results/fines.py — cuándo terminó realmente cada partido.

Un hecho, no una estimación, y por eso vive en `results/`.

## Por qué existe

El candidato v1 aproximaba el fin de un partido con `inicio + 8 horas`, una
cota elegida sin datos: ninguna fuente propia guardaba la hora de fin. Medido
después sobre los 7.664 partidos de los almacenes, con el instante de la última
jugada del feed en vivo:

| | reloj de pared inicio → fin |
|---|---|
| mediana | 2,69 h |
| p90 | 3,22 h |
| p99 | 4,96 h |
| p99,9 | 6,73 h |
| **máximo** | **9,04 h** |

**Dos partidos de 7.664 (0,026%) exceden las 8 horas.** O sea que la cota era
casi siempre holgada de más —lo que cuesta cobertura— y en dos casos corta de
menos, que es lo que cuesta una fuga. Con el fin medido las dos cosas
desaparecen.

## Cómo se tratan las demoras

El instante que se guarda es el de la **última jugada**, o sea reloj de pared,
así que una demora por lluvia ya está adentro. El boxscore lo confirma por
separado en su campo `T`: el partido más largo por demora fue
`T=2:35 (5:00 delay)` — dos horas y media de juego dentro de siete horas y media
de reloj. Para un corte temporal importa el reloj, no el tiempo de juego.

## Casos inciertos

Un partido sin fin medido **no cae en una estimación optimista**: se usa la cota
`inicio + DURACION_MAXIMA`, y queda marcado como `cota` para que cualquier
análisis pueda separarlos. Uno sin inicio ni fin **nunca está disponible**. En
la descarga del 2026-09-06 los 7.664 partidos trajeron fin y ninguno falló, así
que hoy no hay ninguno en ese estado — pero el camino existe porque un partido
en curso, o uno que la API todavía no publicó, sí lo estará.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from fbq.core.clock import normalizar_utc, tiene_hora
from fbq.results.store import DB_PATH as DB_RESULTADOS
from fbq.sources import mlb_stats

log = logging.getLogger(__name__)

CACHE = Path(__file__).parent.parent.parent / "data" / "fines_de_juego.json"

# Cota de respaldo, sólo para partidos sin fin medido. Ver la nota del módulo:
# el máximo observado sobre 7.664 partidos es 9,04 h, así que 8 h NO es una cota
# superior — es una aproximación que falla en el 0,026% de los casos. Se
# conserva únicamente como respaldo y marcada como tal.
DURACION_MAXIMA = timedelta(hours=8)

# Margen sobre el fin MEDIDO. El instante que publica el feed es el de la última
# JUGADA, que no es exactamente el cierre oficial del partido, y `T` está
# redondeado al minuto. Contrastando el fin contra la estimación independiente
# `inicio + T` sobre los 7.656 partidos no suspendidos —en los suspendidos `T`
# es el tiempo de juego de los DOS días y la comparación no aplica—, la
# estimación independiente supera al fin medido en 302 casos, con un **máximo de
# 15,7 minutos**. 20 minutos los cubre todos.
#
# El margen sólo puede costar cobertura, nunca causar una fuga, así que se
# redondea hacia arriba. Medido: no excluye ni un juego del conjunto evaluable.
MARGEN_FIN = timedelta(minutes=20)


def cargar(cache: Path = CACHE) -> Dict[int, Dict[str, Any]]:
    if not cache.exists():
        return {}
    datos = json.loads(cache.read_text(encoding="utf-8"))
    return {int(k): v for k, v in datos.get("juegos", {}).items()}


def disponible_desde(
    game_pk: int, inicio_schedule: Optional[str], medidos: Dict[int, Dict[str, Any]],
) -> tuple[Optional[str], str]:
    """`(instante, procedencia)` a partir del cual el resultado se puede usar.

    Procedencia: `"medido"` (última jugada del feed), `"cota"` (inicio + 8 h) o
    `"desconocido"` (nunca disponible). Se devuelve para que ningún consumidor
    tenga que adivinar de dónde salió el número.
    """
    m = medidos.get(int(game_pk)) or {}
    fin = m.get("fin")
    if fin and tiene_hora(fin):
        return (datetime.fromisoformat(normalizar_utc(fin))
                + MARGEN_FIN).isoformat(), "medido"
    inicio = m.get("reanudacion") or m.get("inicio") or inicio_schedule
    if inicio and tiene_hora(inicio):
        return (datetime.fromisoformat(normalizar_utc(inicio))
                + DURACION_MAXIMA).isoformat(), "cota"
    return None, "desconocido"


def descargar(
    *, db_resultados: Path = DB_RESULTADOS, cache: Path = CACHE,
    hilos: int = 8, solo_faltantes: bool = True,
) -> Dict[str, Any]:
    """Baja el fin de cada partido terminado y lo guarda. Gratis, sin clave."""
    con = sqlite3.connect(f"file:{db_resultados}?mode=ro", uri=True)
    try:
        pks = [r[0] for r in con.execute(
            "SELECT game_pk FROM resultado ORDER BY official_date")]
    finally:
        con.close()

    previos = cargar(cache)
    pendientes = [p for p in pks if not solo_faltantes or p not in previos]
    log.info("partidos a fechar: %s (de %s)", len(pendientes), len(pks))

    salida = {int(k): v for k, v in previos.items()}
    fallos: list = []

    def uno(pk):
        for intento in range(3):
            try:
                return mlb_stats.fin_de_juego(pk)
            except Exception as exc:                      # noqa: BLE001
                if intento == 2:
                    return {"game_pk": pk, "error": str(exc)[:160]}
                time.sleep(1.5 * (intento + 1))

    if pendientes:
        with ThreadPoolExecutor(max_workers=hilos) as ex:
            for r in ex.map(uno, pendientes):
                if r.get("error"):
                    fallos.append(r)
                else:
                    salida[int(r["game_pk"])] = r
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "fuente": "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
        "campo": "liveData.plays.currentPlay.about.endTime (última jugada)",
        "descargado_en": datetime.now().astimezone().isoformat(),
        "juegos": {str(k): v for k, v in salida.items()},
        "fallos": fallos,
    }, indent=1), encoding="utf-8")
    return {"fechados": len(salida), "fallos": len(fallos)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--todos", action="store_true",
                    help="re-baja también los que ya están en la caché")
    ap.add_argument("--hilos", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("%s", descargar(hilos=args.hilos, solo_faltantes=not args.todos))


if __name__ == "__main__":
    main()
