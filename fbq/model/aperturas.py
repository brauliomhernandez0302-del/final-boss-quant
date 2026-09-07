"""fbq/model/aperturas.py — K, BB y bateadores enfrentados, apertura por apertura.

Existe porque las dos fuentes del almacén PIT **no son combinables**, y eso se
midió antes de combinarlas:

| comprobación (300 instantáneas al azar) | resultado |
|---|---|
| `K%`/`BB%` de `fangraphs.pitcher.daily` == reconstruido del Statcast crudo | **221 / 300** |
| `pa` de `savant.pitcher.rolling` == bateadores enfrentados contados en el crudo | **129 / 300** |
| instantáneas que incluyen juegos POSTERIORES a su `as_of` | **0 / 300** |

La última fila es la buena noticia y es lo que demuestra la disponibilidad
temporal —no el nombre "PIT diaria", sino que ninguna instantánea contiene un
juego posterior a su fecha, verificado contra el registro de lanzamientos—. Las
dos primeras son el problema: el numerador y el denominador vienen de fuentes
que **no cuentan la misma población de turnos**, así que dividir uno por el otro
mezcla dos definiciones.

Además, **ninguna de las dos ventanas es la del preregistro**. Las dos son
acumuladas de temporada (`2024-03-28 → as_of`), y el preregistro fija una
ventana de **40 aperturas** que cruza el borde de temporada. Ese solo hecho ya
obligaba a reconstruir.

## La reconstrucción

Una fila por (lanzador, apertura), con `k`, `bb` y `bf` **contados sobre la
misma población de turnos**, del `gameLog` oficial de MLB — gratis, sin clave,
y la misma fuente para entrenar y para predecir, así que no hay paridad que
vigilar entre dos caminos.

`bf` es `battersFaced` del propio boxscore oficial: el denominador que
corresponde a `strikeOuts` y `baseOnBalls` del mismo registro.

## Disponibilidad temporal

Cada apertura lleva su `game_date`, y la ventana se arma con
`game_date < día del juego a predecir`, estricto. Una apertura del propio día no
entra: es la misma regla que rige todo el proyecto.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

import requests

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "aperturas.db"
BASE = "https://statsapi.mlb.com/api/v1"
TIMEOUT = (5, 25)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS apertura (
    pitcher_id   INTEGER NOT NULL,
    game_pk      INTEGER NOT NULL,
    game_date    TEXT    NOT NULL,
    season       INTEGER NOT NULL,
    es_apertura  INTEGER NOT NULL,   -- gamesStarted == 1
    bf           INTEGER NOT NULL,   -- battersFaced del boxscore oficial
    k            INTEGER NOT NULL,
    bb           INTEGER NOT NULL,
    PRIMARY KEY (pitcher_id, game_pk)
);
CREATE INDEX IF NOT EXISTS ix_ap_ventana ON apertura (pitcher_id, game_date);
"""


@contextmanager
def _conn(db: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db, timeout=30.0)
    c.row_factory = sqlite3.Row
    try:
        c.executescript(_SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def _game_log(pitcher_id: int, season: int) -> List[Dict[str, Any]]:
    r = requests.get(f"{BASE}/people/{int(pitcher_id)}/stats",
                     params={"stats": "gameLog", "season": int(season),
                             "group": "pitching"}, timeout=TIMEOUT)
    r.raise_for_status()
    datos = r.json().get("stats") or []
    if not datos:
        return []
    filas = []
    for sp in datos[0].get("splits") or []:
        st = sp.get("stat") or {}
        juego = sp.get("game") or {}
        if not juego.get("gamePk") or not sp.get("date"):
            continue
        filas.append({
            "pitcher_id": int(pitcher_id), "game_pk": int(juego["gamePk"]),
            "game_date": sp["date"], "season": int(season),
            "es_apertura": int(st.get("gamesStarted") or 0),
            "bf": int(st.get("battersFaced") or 0),
            "k": int(st.get("strikeOuts") or 0),
            "bb": int(st.get("baseOnBalls") or 0),
        })
    return filas


def descargar(pitchers: Iterable[int], seasons: Sequence[int], *,
              db: Path = DB_PATH, hilos: int = 8) -> Dict[str, int]:
    tareas = [(p, s) for p in sorted(set(int(x) for x in pitchers)) for s in seasons]
    log.info("game logs a pedir: %s", len(tareas))

    def uno(t):
        p, s = t
        for i in range(3):
            try:
                return _game_log(p, s)
            except Exception:                             # noqa: BLE001
                if i == 2:
                    return []
                time.sleep(1.5 * (i + 1))

    filas: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=hilos) as ex:
        for r in ex.map(uno, tareas):
            filas.extend(r)
    with _conn(db) as c:
        c.executemany(
            """INSERT OR REPLACE INTO apertura
               (pitcher_id, game_pk, game_date, season, es_apertura, bf, k, bb)
               VALUES (:pitcher_id,:game_pk,:game_date,:season,:es_apertura,:bf,:k,:bb)""",
            filas)
        n = c.execute("SELECT COUNT(*) FROM apertura").fetchone()[0]
    return {"filas_traidas": len(filas), "en_almacen": n}


def ventana(pitcher_id: int, antes_de: str, *, n: int = 40,
            solo_aperturas: bool = True, db: Path = DB_PATH,
            ) -> Tuple[int, int, int, int]:
    """`(k, bb, bf, n_aperturas)` de las últimas `n` aperturas ANTES de `antes_de`.

    El corte es estricto: una apertura del propio día del juego no entra. Los
    tres conteos se suman sobre **exactamente el mismo conjunto de aperturas**,
    que es lo que evita mezclar poblaciones.
    """
    with _conn(db) as c:
        filas = c.execute(
            f"""SELECT k, bb, bf FROM apertura
                WHERE pitcher_id=? AND game_date < ? {'AND es_apertura=1' if solo_aperturas else ''}
                ORDER BY game_date DESC, game_pk DESC LIMIT ?""",
            (int(pitcher_id), antes_de, int(n))).fetchall()
    return (sum(f["k"] for f in filas), sum(f["bb"] for f in filas),
            sum(f["bf"] for f in filas), len(filas))


def resumen(db: Path = DB_PATH) -> Dict[str, Any]:
    with _conn(db) as c:
        q = c.execute
        return {
            "filas": q("SELECT COUNT(*) FROM apertura").fetchone()[0],
            "lanzadores": q("SELECT COUNT(DISTINCT pitcher_id) FROM apertura").fetchone()[0],
            "aperturas": q("SELECT COUNT(*) FROM apertura WHERE es_apertura=1").fetchone()[0],
            "rango": tuple(q("SELECT MIN(game_date), MAX(game_date) FROM apertura").fetchone()),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pitchers", type=Path, required=True,
                    help="JSON con la lista de pitcher_id")
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ids = json.loads(args.pitchers.read_text(encoding="utf-8"))
    log.info("%s", descargar(ids, args.seasons))
    log.info("almacén: %s", resumen())


if __name__ == "__main__":
    main()
