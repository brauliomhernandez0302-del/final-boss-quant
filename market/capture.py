"""market/capture.py — barrida del board completo hacia el almacén append-only.

Corre en cron, tan seguido como la cuota lo permita. No recibe picks, no
consulta el ledger y no le importa si el juego está apostado: captura todo lo
que el proveedor devuelva.

Por qué el board completo y no solo los juegos con pick: cuál es el
subconjunto interesante es una decisión del modelo, y el modelo cambia. Los
precios no se pueden volver a comprar. Filtrar en la captura es tomar hoy una
decisión irreversible en nombre del criterio de mañana.

Cuota: reutiliza `odds_fetcher._get_raw_events()`, la misma función y la misma
caché de 10 minutos que ya usan `run_daily_picks` y la barrida de cierre. Una
corrida de este script pegada a una de esas no gasta una llamada extra — sirve
la misma respuesta desde la caché. El costo real de agregar esta captura es
cero llamadas nuevas: la data ya se estaba trayendo doce veces al día y
tirando once.

Uso:
    python3 -m market.capture
    python3 -m market.capture --sport-keys baseball_mlb
    python3 -m market.capture --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from market.store import MarketStore

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
    # `_get_raw_events` es privada a propósito en odds_fetcher (nadie más
    # necesita los eventos crudos), pero es exactamente lo que hace falta acá:
    # la versión pública, `get_odds_data()`, ya viene normalizada y aplanada,
    # y normalizar antes de guardar es cómo se perdieron los precios de los
    # derivados hasta el 2026-07-26.
    from odds_fetcher import _get_raw_events

    events = _get_raw_events()
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
    parser.add_argument("--sport-keys", nargs="*", default=None,
                        help="Filtrar deportes (default: todos los del board)")
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
