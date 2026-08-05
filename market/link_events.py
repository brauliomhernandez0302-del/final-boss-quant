"""market/link_events.py — resolver la identidad evento-de-odds ↔ juego de MLB.

Va en un script aparte de la captura, y la separación es deliberada: el precio
es irrecuperable, el enlace no. Si esto falla, se vuelve a correr mañana sobre
los mismos eventos ya guardados y se resuelve igual. Por eso la captura nunca
espera a que el enlace exista — un evento con precios y sin `game_pk` es un
estado válido y transitorio, no un error.

La regla de identidad es la misma que ya usa `get_best_odds_for_teams()`, con
sus mismas constantes importadas (no copiadas): mismo par de equipos, inicio
dentro de la ventana, y **abstención** si los dos mejores candidatos quedan
más cerca entre sí que el margen mínimo. El caso real que fuerza la
abstención es el doubleheader tradicional, cuyos dos juegos el schedule de MLB
separa por 5 minutos. Preferimos un `game_pk` NULL que uno equivocado: un
enlace errado mete los precios de un juego en el otro y no deja rastro.

`official_date` se copia del schedule de MLB, nunca se deriva de
`date(commence_time)` — para un nocturno de la costa oeste el timestamp UTC
cae un día adelante, y ese desfase ya costó un leak entero en el camino PIT.

Uso:
    python3 -m market.link_events
    python3 -m market.link_events --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from market.store import MarketStore

log = logging.getLogger(__name__)

_SPORT = "baseball_mlb"


def _parse_utc(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _teams_match(a: str, b: str) -> bool:
    """Misma regla de contención que usa el emparejador de odds vigente.

    No es fuzzy matching: es contención de cadenas en minúsculas, que resuelve
    "Athletics" vs "Oakland Athletics" sin abrir la puerta a que dos equipos
    distintos se parezcan lo suficiente.
    """
    a, b = (a or "").lower(), (b or "").lower()
    return bool(a) and bool(b) and (a in b or b in a)


def link_events(
    store: MarketStore,
    *,
    dry_run: bool = False,
    max_days_back: int = 14,
) -> Dict[str, int]:
    """Resuelve `game_pk`/`official_date` para los eventos aún sin enlazar."""
    from data_fetchers import MLBStatsAPI
    from odds_fetcher import _ODDS_MATCH_WINDOW, _ODDS_MATCH_MIN_MARGIN

    api = MLBStatsAPI()
    pending = store.unlinked_events(sport_key=_SPORT)
    summary = {"pending": len(pending), "linked": 0, "ambiguous": 0,
               "no_candidate": 0, "unparseable": 0}
    if not pending:
        return summary

    # Un evento puede empezar en un día UTC y pertenecer al día oficial
    # anterior, así que se consulta el schedule de ambos días. Se cachea por
    # día para no repetir la llamada por evento.
    schedule_cache: Dict[str, List[dict]] = {}

    def schedule_for(day: str) -> List[dict]:
        if day not in schedule_cache:
            try:
                schedule_cache[day] = api.get_todays_games(date=day) or []
            except Exception as exc:
                log.warning("Fallo al pedir el schedule de %s: %s", day, exc)
                schedule_cache[day] = []
        return schedule_cache[day]

    horizon = datetime.now(timezone.utc) - timedelta(days=max_days_back)

    for ev in pending:
        target = _parse_utc(ev["commence_time"])
        if target is None:
            log.warning("commence_time ilegible para event_id=%s (%s @ %s)",
                        ev["event_id"], ev["away_team"], ev["home_team"])
            summary["unparseable"] += 1
            continue
        if target < horizon:
            # Eventos viejos sin enlazar: el schedule sigue disponible, pero
            # no se barren en cada corrida. `--max-days-back` los recupera.
            continue

        days = {(target - timedelta(days=1)).date().isoformat(),
                target.date().isoformat()}

        candidates = []
        for day in sorted(days):
            for game in schedule_for(day):
                if not (_teams_match(ev["home_team"], game.get("home_team", ""))
                        and _teams_match(ev["away_team"], game.get("away_team", ""))):
                    continue
                start = _parse_utc(game.get("game_date", ""))
                if start is None:
                    continue
                delta = abs(start - target)
                if delta <= _ODDS_MATCH_WINDOW:
                    candidates.append((delta, game))

        if not candidates:
            log.info("Sin candidato para event_id=%s (%s @ %s, %s)",
                     ev["event_id"], ev["away_team"], ev["home_team"],
                     ev["commence_time"])
            summary["no_candidate"] += 1
            continue

        candidates.sort(key=lambda c: c[0])
        if (len(candidates) > 1
                and (candidates[1][0] - candidates[0][0]) < _ODDS_MATCH_MIN_MARGIN):
            log.warning(
                "Ambiguo para event_id=%s (%s @ %s): dos juegos casi igual de "
                "cerca (%s y %s) — no se adivina",
                ev["event_id"], ev["away_team"], ev["home_team"],
                candidates[0][0], candidates[1][0],
            )
            summary["ambiguous"] += 1
            if not dry_run:
                # Se deja constancia del intento fallido: sin esto, un evento
                # ambiguo se re-intenta para siempre sin que nadie sepa por qué
                # nunca se enlaza.
                store.link_event(
                    ev["event_id"], sport_key=_SPORT, game_pk=None,
                    official_date=None, commence_time=ev["commence_time"],
                    home_team=ev["home_team"], away_team=ev["away_team"],
                    method="ambiguous",
                )
            continue

        game = candidates[0][1]
        if dry_run:
            log.info("[dry-run] %s (%s @ %s) → game_pk=%s official_date=%s",
                     ev["event_id"], ev["away_team"], ev["home_team"],
                     game.get("game_pk"), game.get("official_date"))
        else:
            store.link_event(
                ev["event_id"], sport_key=_SPORT,
                game_pk=game.get("game_pk"),
                official_date=game.get("official_date"),
                commence_time=ev["commence_time"],
                home_team=ev["home_team"], away_team=ev["away_team"],
                method="schedule_match",
            )
        summary["linked"] += 1

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-days-back", type=int, default=14)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    store = MarketStore()
    summary = link_events(store, dry_run=args.dry_run,
                          max_days_back=args.max_days_back)
    log.info("Enlace de eventos: %s", summary)


if __name__ == "__main__":
    main()
