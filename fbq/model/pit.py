"""fbq/model/pit.py — la compuerta temporal. Qué hecho estaba disponible, y cuándo.

Todo hecho que entra en una fila pasa por acá. No es una convención que el
constructor de features deba recordar: es una compuerta que **levanta una
excepción** si le piden un hecho que al corte no existía.

La regla, del preregistro §2:

    un partido anterior está disponible en el corte T si
        inicio_utc(partido) + DURACION_MAXIMA ≤ T

`DURACION_MAXIMA = 8 horas` es una cota deliberadamente holgada. La mediana de
un partido de MLB ronda las 3 horas y los extremos de entradas extra no llegan a
7; ninguna fuente propia guarda la hora de FIN, y la alternativa a una cota es
adivinar. **Errar por exceso cuesta cobertura; errar por defecto cuesta una
fuga**, y este proyecto ya pagó siete baselines por fugas.

**Suspendidos**: un partido suspendido y reanudado tiene dos entradas en el
schedule con el mismo `game_pk`. La disponibilidad se calcula sobre la MÁS
TARDÍA — un partido suspendido el día D y terminado el D+1 no estaba disponible
el D, aunque su `official_date` diga D.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from fbq.core.clock import normalizar_utc, tiene_hora
from fbq.evaluator.frame import DB_RESULTADOS

RAIZ = Path(__file__).parent.parent.parent
CACHE_SCHEDULE = RAIZ / "data" / "schedule_inicios.json"

# Ver la nota del módulo. Cota, no estimación.
DURACION_MAXIMA = timedelta(hours=8)


class FugaDetectada(Exception):
    """Se pidió un hecho que en el corte declarado no estaba disponible."""


@dataclass(frozen=True)
class Partido:
    """Un partido terminado, con el instante a partir del cual su resultado se
    puede usar sin fugar."""

    game_pk: int
    official_date: str
    season: int
    home_team: str
    away_team: str
    home_runs: int
    away_runs: int
    disponible_desde: Optional[str]   # None = nunca se pudo fechar el fin

    @property
    def home_won(self) -> int:
        return 1 if self.home_runs > self.away_runs else 0


def _inicios(cache: Path = CACHE_SCHEDULE) -> Dict[int, List[str]]:
    """`{game_pk: [instantes de inicio]}` del schedule oficial.

    Una LISTA porque un suspendido tiene dos. Se usa el máximo: el partido no
    terminó antes de haber empezado por última vez.
    """
    if not cache.exists():
        raise FileNotFoundError(
            f"falta {cache}. Se genera con "
            f"`python3 -m fbq.market.importar_historico` — el mismo schedule "
            f"oficial que fecha las cotizaciones fecha los partidos.")
    datos = json.loads(cache.read_text(encoding="utf-8"))
    out: Dict[int, List[str]] = defaultdict(list)
    for j in datos["juegos"]:
        g = j.get("game_date")
        if g and tiene_hora(g):
            out[int(j["game_pk"])].append(normalizar_utc(g))
    return dict(out)


def cargar_partidos(
    seasons: Sequence[int],
    *,
    db_resultados: Path = DB_RESULTADOS,
    cache_schedule: Path = CACHE_SCHEDULE,
) -> List[Partido]:
    """Los partidos terminados, con su instante de disponibilidad.

    Un partido sin hora de inicio en el schedule queda con
    `disponible_desde=None`: **nunca** se usa como hecho previo. No se le
    inventa una hora — es el mismo criterio que la importación del histórico.
    """
    inicios = _inicios(cache_schedule)
    marcas = ",".join("?" * len(seasons))
    con = sqlite3.connect(f"file:{db_resultados}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        filas = con.execute(
            f"""SELECT game_pk, official_date, season, home_team, away_team,
                       home_runs, away_runs
                FROM resultado WHERE season IN ({marcas})
                ORDER BY official_date, game_pk""", tuple(seasons)).fetchall()
    finally:
        con.close()

    out = []
    for r in filas:
        arranques = inicios.get(int(r["game_pk"]))
        # El más tardío: un suspendido no terminó antes de su reanudación.
        fin = (normalizar_utc(
            (max(arranques) if arranques else None) or "1970-01-01T00:00:00Z")
            if arranques else None)
        if fin is not None:
            from datetime import datetime
            fin = (datetime.fromisoformat(fin) + DURACION_MAXIMA).isoformat()
        out.append(Partido(
            game_pk=int(r["game_pk"]), official_date=r["official_date"],
            season=int(r["season"]), home_team=r["home_team"],
            away_team=r["away_team"], home_runs=int(r["home_runs"]),
            away_runs=int(r["away_runs"]), disponible_desde=fin))
    return out


class VentanaPIT:
    """Los partidos de cada equipo que YA habían terminado en un corte dado.

    Es la única puerta por la que un hecho entra a una fila. `de_equipo()`
    filtra; `exigir_disponible()` es la compuerta explícita para cuando alguien
    quiere usar un partido concreto — y es la que levanta `FugaDetectada`.
    """

    def __init__(self, partidos: Iterable[Partido]) -> None:
        self._por_equipo: Dict[str, List[Partido]] = defaultdict(list)
        self._por_pk: Dict[int, Partido] = {}
        for p in partidos:
            self._por_pk[p.game_pk] = p
            self._por_equipo[p.home_team].append(p)
            self._por_equipo[p.away_team].append(p)
        for lista in self._por_equipo.values():
            lista.sort(key=lambda p: (p.disponible_desde or "", p.game_pk))

    def disponible(self, partido: Partido, corte: str) -> bool:
        return (partido.disponible_desde is not None
                and partido.disponible_desde <= corte)

    def exigir_disponible(self, game_pk: int, corte: str) -> Partido:
        """El partido, si en `corte` ya había terminado. Si no, levanta.

        Éste es el detector estructural. Cualquier intento de meter en una fila
        un partido que al corte seguía en juego —incluido el propio partido que
        se está prediciendo— muere acá.
        """
        p = self._por_pk.get(int(game_pk))
        if p is None:
            raise FugaDetectada(f"game_pk={game_pk} no está en la ventana")
        if p.disponible_desde is None:
            raise FugaDetectada(
                f"game_pk={game_pk} no tiene hora de inicio en el schedule, así "
                f"que no se puede afirmar que hubiera terminado en {corte}")
        if p.disponible_desde > corte:
            raise FugaDetectada(
                f"FUGA: game_pk={game_pk} recién estaba disponible en "
                f"{p.disponible_desde}, y el corte de la fila es {corte}. "
                f"Un hecho posterior al corte no puede entrar en la predicción.")
        return p

    def de_equipo(self, equipo: str, corte: str, *, ventana: int) -> List[Partido]:
        """Los últimos `ventana` partidos del equipo disponibles en `corte`.

        Cruza el borde de temporada a propósito (preregistro §4): así abril
        tiene historia en vez de un fallback.
        """
        previos = [p for p in self._por_equipo.get(equipo, ())
                   if self.disponible(p, corte)]
        return previos[-ventana:] if ventana else previos
