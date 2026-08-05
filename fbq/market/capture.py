"""market/capture.py — barrida del board completo hacia el almacén append-only.

Corre en cron, tan seguido como la cuota lo permita. No recibe picks, no
consulta el ledger y no le importa si el juego está apostado: captura todo lo
que el proveedor devuelva.

Por qué el board completo del DEPORTE y no sólo los juegos con pick: cuál es el
subconjunto interesante es una decisión del modelo, y el modelo cambia. Los
precios no se pueden volver a comprar. Filtrar por juego en la captura es tomar
hoy una decisión irreversible en nombre del criterio de mañana.

Por qué sólo MLB por defecto, en cambio, SÍ es una decisión de cuota: cada
deporte pedido cuesta (mercados × regiones) = 6 créditos por barrida, así que
los nueve deportes salen 54 por barrida y 648 por día. MLB solo sale 72. Se
amplía con `--sport-keys` cuando haya un consumidor real para otro deporte;
capturar precios de deportes que nadie analiza es gastar la cuota que MLB
necesita.

Cuota: `fbq.sources.odds_api.board()` cachea en archivo durante 10 minutos, así
que dos corridas dentro de esa ventana gastan una sola llamada. Es la misma
razón por la que el cron corre a los :51, justo después de la barrida de cierre
del sistema anterior: cae dentro de su ventana y no gasta nada nuevo.

Uso:
    python3 -m fbq.market.capture
    python3 -m fbq.market.capture --sport-keys baseball_mlb
    python3 -m fbq.market.capture --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fbq.market.store import MarketStore

log = logging.getLogger(__name__)


def capture(
    store: MarketStore,
    *,
    sport_keys: list[str] | None = None,
    dedupe: bool = True,
    dry_run: bool = False,
    note: str | None = None,
) -> dict:
    """Trae el board y lo guarda. Devuelve el resumen de la barrida."""
    from fbq.sources import odds_api

    try:
        events = odds_api.board(deportes=tuple(sport_keys or odds_api.DEPORTES))
    except odds_api.SinClave as exc:
        log.error("%s — sin clave no hay board que capturar", exc)
        return {"events": 0, "rows_new": 0, "unchanged": 0, "skipped": 0,
                "empty_board": True}
    if not events:
        # Falla ruidosa: una barrida vacía es indistinguible de "no había
        # juegos" si no se grita, y el cron la ocultaría para siempre.
        log.error("Board vacío: el fetcher no devolvió eventos (¿API caída, "
                  "cuota agotada, o ODDS_API_KEY sin definir?)")
        return {"events": 0, "rows_new": 0, "unchanged": 0, "skipped": 0,
                "empty_board": True}

    if dry_run:
        n = len(events)
        if sport_keys:
            n = sum(1 for e in events if e.get("sport_key") in set(sport_keys))
        log.info("[dry-run] %d eventos en el board (%d tras filtrar), sin escribir", len(events), n)
        return {"events": n, "rows_new": 0, "unchanged": 0, "skipped": 0,
                "dry_run": True}

    summary = store.record_board(
        events, sport_keys=sport_keys, dedupe=dedupe, note=note,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport-keys", nargs="*", default=["baseball_mlb"],
                        help="Deportes a capturar (default: sólo baseball_mlb — "
                             "cada deporte extra cuesta 6 créditos por barrida)")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Guardar toda cotización aunque no se haya movido")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--note", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    store = MarketStore()
    summary = capture(
        store, sport_keys=args.sport_keys, dedupe=not args.no_dedupe,
        dry_run=args.dry_run, note=args.note,
    )
    log.info("Barrida de mercado: %s", summary)
    if summary.get("empty_board"):
        sys.exit(1)


if __name__ == "__main__":
    main()
