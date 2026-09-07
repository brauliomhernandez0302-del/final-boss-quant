"""fbq/model/bullpen.py — lanzamientos de relevo por equipo y partido.

Existe porque **ninguna fuente local los tiene**, y eso se comprobó antes de
descargar nada:

| almacén local | qué tiene | por qué no sirve para esto |
|---|---|---|
| `data/aperturas.db` | 27.958 filas, **517 lanzadores** | los 517 abrieron alguna vez (`MAX(es_apertura)=1` en los 517). Un brazo que nunca abrió NO está: el almacén se construyó desde los abridores anunciados |
| `data/pit_cache_pitcher.db` | 463.967 instantáneas, 1.302 entidades | acumulados de temporada por LANZADOR, sin conteo de lanzamientos y sin el corte por partido |
| `data/results.db` | 8.087 partidos | marcador y fin medido; nada de pitcheo |

Cobertura real del almacén local sobre lo que hace falta: **13.743 de 16.174
equipos-partido (85,0 %) tienen abridor conocido**, y ni siquiera esos traen el
conteo de lanzamientos. Reconstruir la carga del bullpen restando el abridor al
total del equipo dejaría fuera 2.431 equipos-partido y encima mezclaría dos
fuentes que no cuentan la misma población — el error que este proyecto ya pagó
al intentar combinar `fangraphs.pitcher.daily` con `savant.pitcher.rolling`
(221/300 y 129/300 de coincidencia).

## La fuente, y por qué ésta

El **boxscore oficial de MLB** (`/api/v1/game/{pk}/boxscore`), gratis y sin
clave. Una llamada por partido da, del mismo documento:

- `teams.{lado}.pitchers` — la lista ORDENADA. El primero es el abridor; todos
  los demás son relevo. No hace falta clasificar roles ni consultar rósters.
- `numberOfPitches` de cada uno.

Es la misma fuente que ya fecha los resultados y que da `bf` en
`fbq/model/aperturas.py`, así que entrenar y predecir leen lo mismo.

**Un brazo que jamás abrió entra igual**, porque no se enumera a los lanzadores:
se lee quién lanzó en ESE partido.

## Disponibilidad temporal

No se compara ninguna fecha acá. La fila guarda el partido, y quien la use pasa
por `VentanaPIT`, que ordena por `disponible_desde` = fin medido de la última
jugada + margen preregistrado (`fbq/results/fines.py`). La misma compuerta del
resto del proyecto, heredada por construcción y no por disciplina.

## Lo que NO decide este módulo

No define la ventana, ni el tratamiento de faltantes, ni la variable. Eso vive
en `docs/PREREGISTRO_V1_5_BULLPEN_2026-09-07.md` y se aplica en
`fbq/model/carga.py`. Acá sólo se guarda el hecho contado.
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
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence

import requests

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "bullpen.db"
DB_RESULTADOS = Path(__file__).parent.parent.parent / "data" / "results.db"
BASE = "https://statsapi.mlb.com/api/v1"
TIMEOUT = (5, 25)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS relevo (
    game_pk         INTEGER NOT NULL,
    es_local        INTEGER NOT NULL,   -- 1 local, 0 visita
    team_id         INTEGER NOT NULL,
    team_nombre     TEXT    NOT NULL,
    lanzadores      INTEGER NOT NULL,   -- todos los del equipo en ese partido
    relevistas      INTEGER NOT NULL,   -- lanzadores - 1
    pitches_total   INTEGER NOT NULL,
    pitches_abridor INTEGER NOT NULL,
    pitches_relevo  INTEGER NOT NULL,
    bf_relevo       INTEGER NOT NULL,
    outs_relevo     INTEGER NOT NULL,
    abridor_id      INTEGER NOT NULL,
    abridor_gs      INTEGER NOT NULL,   -- gamesStarted del primero de la lista
    PRIMARY KEY (game_pk, es_local)
);
CREATE INDEX IF NOT EXISTS ix_relevo_juego ON relevo (game_pk);
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


def _outs(ip: Any) -> int:
    """`"5.2"` son 5 entradas y 2 outs = 17 outs, no 5,2 entradas."""
    try:
        s = str(ip or "0")
        entero, _, tercios = s.partition(".")
        return int(entero or 0) * 3 + int(tercios or 0)
    except (TypeError, ValueError):
        return 0


def filas_de_boxscore(pk: int, box: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Las dos filas (local y visita) de un boxscore ya descargado.

    Un equipo sin lanzadores en el documento **no produce fila**: es un partido
    del que no se puede afirmar nada, no un partido con cero relevo.
    """
    filas = []
    for lado, es_local in (("home", 1), ("away", 0)):
        t = (box.get("teams") or {}).get(lado) or {}
        orden = t.get("pitchers") or []
        jugadores = t.get("players") or {}
        if not orden:
            continue

        def stat(pid: int) -> Dict[str, Any]:
            return ((jugadores.get(f"ID{pid}") or {}).get("stats") or {}
                    ).get("pitching") or {}

        def np_(pid: int) -> int:
            s = stat(pid)
            v = s.get("numberOfPitches")
            if v is None:
                v = s.get("pitchesThrown")
            return int(v or 0)

        # La regla de rol reutilizada dice "el primero de la lista es el
        # abridor". En 3 de 16.174 equipos-partido el primero figura con CERO
        # lanzamientos —anunciado y retirado antes de lanzarle a nadie— y el
        # abridor real (`gamesStarted = 1`) es el segundo. Contarlo como relevo
        # metería una apertura entera en la carga del bullpen.
        #
        # No es una excepción a la regla de rol: una entrada de cero
        # lanzamientos no es una aparición. Se salta, y en todos los demás
        # casos esto ES la regla original, porque el primero ya lanzó.
        i0 = next((i for i, pid in enumerate(orden) if np_(int(pid)) > 0), 0)
        abridor = int(orden[i0])
        relevistas = [int(p) for p in orden[i0 + 1:]]
        filas.append({
            "game_pk": int(pk), "es_local": es_local,
            "team_id": int((t.get("team") or {}).get("id") or 0),
            "team_nombre": (t.get("team") or {}).get("name") or "",
            "lanzadores": len(orden), "relevistas": len(relevistas),
            "pitches_total": sum(np_(int(p)) for p in orden),
            "pitches_abridor": np_(abridor),
            "pitches_relevo": sum(np_(p) for p in relevistas),
            "bf_relevo": sum(int(stat(p).get("battersFaced") or 0)
                             for p in relevistas),
            "outs_relevo": sum(_outs(stat(p).get("inningsPitched"))
                               for p in relevistas),
            "abridor_id": abridor,
            "abridor_gs": int(stat(abridor).get("gamesStarted") or 0),
        })
    return filas


def _boxscore(pk: int) -> Optional[Dict[str, Any]]:
    r = requests.get(f"{BASE}/game/{int(pk)}/boxscore", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def descargar(game_pks: Iterable[int], *, db: Path = DB_PATH,
              hilos: int = 8) -> Dict[str, int]:
    """Trae los boxscores que falten y guarda sus dos filas por partido."""
    pks = sorted({int(p) for p in game_pks})
    with _conn(db) as c:
        ya = {int(r[0]) for r in c.execute(
            "SELECT game_pk FROM relevo GROUP BY game_pk HAVING COUNT(*)=2")}
    faltan = [p for p in pks if p not in ya]
    log.info("boxscores: %s pedidos · %s ya estaban · %s a bajar",
             len(pks), len(pks) - len(faltan), len(faltan))

    fallidos: List[int] = []

    def uno(pk: int) -> List[Dict[str, Any]]:
        for i in range(3):
            try:
                b = _boxscore(pk)
                return filas_de_boxscore(pk, b or {})
            except Exception:                                  # noqa: BLE001
                if i == 2:
                    fallidos.append(pk)
                    return []
                time.sleep(1.5 * (i + 1))
        return []

    filas: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=hilos) as ex:
        for i, r in enumerate(ex.map(uno, faltan), 1):
            filas.extend(r)
            if i % 500 == 0:
                log.info("  %s/%s", i, len(faltan))

    with _conn(db) as c:
        c.executemany(
            """INSERT OR REPLACE INTO relevo
               (game_pk, es_local, team_id, team_nombre, lanzadores, relevistas,
                pitches_total, pitches_abridor, pitches_relevo, bf_relevo,
                outs_relevo, abridor_id, abridor_gs)
               VALUES (:game_pk,:es_local,:team_id,:team_nombre,:lanzadores,
                       :relevistas,:pitches_total,:pitches_abridor,
                       :pitches_relevo,:bf_relevo,:outs_relevo,:abridor_id,
                       :abridor_gs)""", filas)
        n = c.execute("SELECT COUNT(*) FROM relevo").fetchone()[0]
    return {"pedidos": len(pks), "bajados": len(faltan), "filas": len(filas),
            "fallidos": len(fallidos), "en_almacen": n}


def cargar(db: Path = DB_PATH) -> Dict[tuple, Dict[str, Any]]:
    """`{(game_pk, es_local): fila}` — todo el almacén, para armar ventanas."""
    if not Path(db).exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return {(int(r["game_pk"]), int(r["es_local"])): dict(r)
                for r in con.execute("SELECT * FROM relevo")}
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--hilos", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    con = sqlite3.connect(f"file:{DB_RESULTADOS}?mode=ro", uri=True)
    pks = [int(r[0]) for r in con.execute(
        "SELECT game_pk FROM resultado WHERE season IN (%s)"
        % ",".join("?" * len(a.seasons)), a.seasons)]
    con.close()
    log.info("%s", json.dumps(descargar(pks, hilos=a.hilos), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
