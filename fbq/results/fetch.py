"""fbq/results/fetch.py — traer los finales del día y guardarlos.

Corre en cron después de que terminan los juegos. Es deliberadamente aburrido:
pide el schedule del día, se queda con los que terminaron, y los guarda.

Toda la inteligencia está en `Final`, que se niega a construirse con un estado
que no sea final o con un marcador empatado. Eso deja este archivo sin ninguna
condición sutil que revisar — si algo no es un final, el constructor lo rechaza
y acá sólo se cuenta.

Uso:
    python3 -m fbq.results.fetch                    # ayer y hoy
    python3 -m fbq.results.fetch --fecha 2026-08-01
    python3 -m fbq.results.fetch --desde 2026-03-25 --hasta 2026-08-04
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterator, List

from fbq.results.store import ESTADOS_FINALES, Final, ResultsStore
from fbq.sources import mlb_stats

log = logging.getLogger(__name__)


def _dias(desde: str, hasta: str) -> Iterator[str]:
    d, e = date.fromisoformat(desde), date.fromisoformat(hasta)
    while d <= e:
        yield d.isoformat()
        d += timedelta(days=1)


def finales_del_dia(fecha: str) -> List[Final]:
    """Los juegos terminados de esa fecha de schedule.

    Un juego no terminado, pospuesto o empatado no llega a la lista: `Final` lo
    rechaza al construirse y acá se descarta con un debug. La distinción entre
    "no lo miramos" y "lo miramos y no había final" queda en el resumen del
    llamador, no en un None silencioso.
    """
    salida: List[Final] = []
    for g in mlb_stats.schedule(fecha=fecha):
        estado = (g.get("status") or {}).get("detailedState", "")
        if estado not in ESTADOS_FINALES:
            continue
        marcador = mlb_stats.linescore_final(g)
        if marcador is None:
            continue
        local, visita, innings = marcador
        equipos = g.get("teams") or {}
        try:
            salida.append(Final(
                game_pk=int(g["gamePk"]),
                official_date=str(g.get("officialDate") or fecha)[:10],
                season=int(str(g.get("officialDate") or fecha)[:4]),
                home_team=((equipos.get("home") or {}).get("team") or {}).get("name", ""),
                away_team=((equipos.get("away") or {}).get("team") or {}).get("name", ""),
                home_runs=local, away_runs=visita,
                detailed_state=estado, innings=innings,
            ))
        except (ValueError, KeyError) as exc:
            log.debug("game_pk=%s descartado: %s", g.get("gamePk"), exc)
    return salida


def traer(store: ResultsStore, desde: str, hasta: str) -> Dict[str, int]:
    total = {"dias": 0, "finales": 0, "nuevos": 0, "corregidos": 0, "sin_cambio": 0}
    for dia in _dias(desde, hasta):
        finales = finales_del_dia(dia)
        r = store.registrar(finales)
        total["dias"] += 1
        total["finales"] += len(finales)
        for k in ("nuevos", "corregidos", "sin_cambio"):
            total[k] += r[k]
        if r["corregidos"]:
            log.warning("%s: %d marcador(es) CORREGIDO(s) respecto de lo guardado",
                        dia, r["corregidos"])
    return total


def main() -> None:
    hoy = datetime.now(timezone.utc).date()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fecha", help="un solo día (YYYY-MM-DD)")
    ap.add_argument("--dias", type=int, default=None,
                    help="ventana hacia atrás desde hoy; evita aritmética de "
                         "fechas en el cron, que se escribe distinto en cada shell")
    ap.add_argument("--desde", default=(hoy - timedelta(days=1)).isoformat())
    ap.add_argument("--hasta", default=hoy.isoformat())
    args = ap.parse_args()
    if args.dias:
        args.desde = (hoy - timedelta(days=int(args.dias))).isoformat()
        args.hasta = hoy.isoformat()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    desde = hasta = args.fecha
    if not args.fecha:
        desde, hasta = args.desde, args.hasta

    resumen = traer(ResultsStore(), desde, hasta)
    log.info("Resultados %s..%s: %s", desde, hasta, resumen)


if __name__ == "__main__":
    main()
