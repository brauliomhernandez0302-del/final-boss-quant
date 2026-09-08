"""fbq/model/relevistas.py — carga OBSERVADA, relevista por relevista.

Una fila por `(game_pk, team_id, pitcher_id)`, construida **releyendo las
respuestas guardadas** en `data/boxscores/`. Sin red: el documento que ya se
pidió contiene todo esto y no hay que volver a pedirlo.

## Qué es este dato, y qué NO es

Es **carga observada**: cuántos lanzamientos hizo ese brazo, en ese partido, y
desde qué instante ese hecho se pudo usar sin fugar.

**No es disponibilidad.** Que un relevista haya lanzado 40 lanzamientos en dos
días no demuestra que hoy no esté disponible, ni que esté lesionado, ni que el
entrenador vaya a usarlo o dejar de usarlo. Un brazo puede no aparecer por
lesión, por descanso programado, por rol —un cerrador que no entra porque el
juego no está cerrado—, por una bajada a ligas menores o simplemente porque el
partido no lo pidió. **Ninguna de esas cosas está en este dato**, y ninguna se
infiere de él. Lo único que dice una fila es: este brazo lanzó esto, acá.

La distinción no es retórica. El sistema anterior convirtió carga en un
multiplicador de fatiga (`bullpen_engine._workload_mult`) con pendientes
elegidas a mano, tratando «lanzó mucho» como «rinde peor». Eso es una hipótesis,
y este almacén no la contiene: la deja medible.

## Reglas reutilizadas, no reinventadas

| regla | de dónde | qué hace acá |
|---|---|---|
| rol por orden | `bullpen_relief_appearance_builder.ROLE_RULE_VERSION` | el primer lanzador **que lanzó** es `abridor`; los demás, `relevo` |
| entradas de cero lanzamientos | arreglo del 2026-09-07 | no son apariciones: se saltan para elegir al abridor y se marcan `rol = "sin_lanzar"` |
| `disponible_desde` | `fbq/results/fines.py` | fin medido de la última jugada + `MARGEN_FIN` (20 min), con `procedencia` |

`procedencia` viaja en la fila **a propósito**: `medido` (fin real del feed),
`cota` (inicio + 8 h) o `desconocido`. Ningún consumidor tiene que adivinar de
dónde salió el instante, y una fila `desconocido` nunca está disponible.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "relevistas.db"
DB_RESULTADOS = Path(__file__).parent.parent.parent / "data" / "results.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS aparicion (
    game_pk          INTEGER NOT NULL,
    team_id          INTEGER NOT NULL,
    pitcher_id       INTEGER NOT NULL,
    es_local         INTEGER NOT NULL,
    official_date    TEXT    NOT NULL,
    season           INTEGER NOT NULL,
    orden            INTEGER NOT NULL,   -- posición en la lista del boxscore
    rol              TEXT    NOT NULL,   -- abridor | relevo | sin_lanzar
    pitches          INTEGER NOT NULL,
    bf               INTEGER NOT NULL,
    outs             INTEGER NOT NULL,
    k                INTEGER NOT NULL,
    bb               INTEGER NOT NULL,
    fin_medido       TEXT,               -- fin de la última jugada, o NULL
    disponible_desde TEXT,               -- fin + MARGEN_FIN, o cota, o NULL
    procedencia      TEXT    NOT NULL,   -- medido | cota | desconocido
    PRIMARY KEY (game_pk, team_id, pitcher_id)
);
CREATE INDEX IF NOT EXISTS ix_ap_brazo ON aparicion (pitcher_id, disponible_desde);
CREATE INDEX IF NOT EXISTS ix_ap_equipo ON aparicion (team_id, disponible_desde);
CREATE INDEX IF NOT EXISTS ix_ap_juego ON aparicion (game_pk);
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


def filas_de_boxscore(pk: int, box: Dict[str, Any], *,
                      official_date: str, season: int,
                      fin: Optional[str], disponible: Optional[str],
                      procedencia: str) -> List[Dict[str, Any]]:
    """Una fila por lanzador de cada equipo, con su rol y su carga."""
    from fbq.model.bullpen import _outs

    filas: List[Dict[str, Any]] = []
    for lado, es_local in (("home", 1), ("away", 0)):
        t = (box.get("teams") or {}).get(lado) or {}
        orden = [int(p) for p in (t.get("pitchers") or [])]
        jugadores = t.get("players") or {}
        if not orden:
            continue
        team_id = int((t.get("team") or {}).get("id") or 0)

        def st(pid: int) -> Dict[str, Any]:
            return ((jugadores.get(f"ID{pid}") or {}).get("stats") or {}
                    ).get("pitching") or {}

        def np_(pid: int) -> int:
            s = st(pid)
            v = s.get("numberOfPitches")
            if v is None:
                v = s.get("pitchesThrown")
            return int(v or 0)

        # El abridor es el PRIMERO QUE LANZÓ. Ver el arreglo del 2026-09-07: en
        # 3 de 16.174 equipos-partido el primero de la lista tiró cero.
        i0 = next((i for i, pid in enumerate(orden) if np_(pid) > 0), None)
        for i, pid in enumerate(orden):
            s = st(pid)
            rol = ("sin_lanzar" if np_(pid) == 0 and i0 is not None and i < i0
                   else "abridor" if i == i0
                   else "relevo" if i0 is not None and i > i0
                   else "sin_lanzar")
            filas.append({
                "game_pk": int(pk), "team_id": team_id, "pitcher_id": int(pid),
                "es_local": es_local, "official_date": official_date,
                "season": int(season), "orden": i, "rol": rol,
                "pitches": np_(pid),
                "bf": int(s.get("battersFaced") or 0),
                "outs": _outs(s.get("inningsPitched")),
                "k": int(s.get("strikeOuts") or 0),
                "bb": int(s.get("baseOnBalls") or 0),
                "fin_medido": fin, "disponible_desde": disponible,
                "procedencia": procedencia,
            })
    return filas


def construir(game_pks: Optional[Iterable[int]] = None, *, db: Path = DB_PATH,
              db_resultados: Path = DB_RESULTADOS,
              carpeta: Optional[Path] = None) -> Dict[str, int]:
    """Reconstruye la tabla desde las respuestas guardadas. Nunca pide a la red."""
    from fbq.model.bullpen import RESPUESTAS, leer_respuesta
    from fbq.results import fines as _f

    carpeta = Path(carpeta or RESPUESTAS)
    con = sqlite3.connect(f"file:{db_resultados}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        juegos = {int(r["game_pk"]): dict(r) for r in con.execute(
            "SELECT game_pk, official_date, season FROM resultado")}
    finally:
        con.close()
    pks = sorted(set(int(p) for p in game_pks) & set(juegos)) if game_pks else sorted(juegos)

    medidos = _f.cargar()
    filas: List[Dict[str, Any]] = []
    sin_respuesta: List[int] = []
    for pk in pks:
        box = leer_respuesta(pk, carpeta)
        if box is None:
            sin_respuesta.append(pk)
            continue
        disponible, procedencia = _f.disponible_desde(pk, None, medidos)
        fin = (medidos.get(pk) or {}).get("fin")
        filas.extend(filas_de_boxscore(
            pk, box, official_date=juegos[pk]["official_date"],
            season=int(juegos[pk]["season"]), fin=fin,
            disponible=disponible, procedencia=procedencia))

    with _conn(db) as c:
        c.executemany(
            """INSERT OR REPLACE INTO aparicion
               (game_pk, team_id, pitcher_id, es_local, official_date, season,
                orden, rol, pitches, bf, outs, k, bb, fin_medido,
                disponible_desde, procedencia)
               VALUES (:game_pk,:team_id,:pitcher_id,:es_local,:official_date,
                       :season,:orden,:rol,:pitches,:bf,:outs,:k,:bb,
                       :fin_medido,:disponible_desde,:procedencia)""", filas)
        n = c.execute("SELECT COUNT(*) FROM aparicion").fetchone()[0]
    if sin_respuesta:
        log.warning("sin respuesta guardada: %s partidos (ej. %s)",
                    len(sin_respuesta), sin_respuesta[:5])
    return {"partidos": len(pks), "sin_respuesta": len(sin_respuesta),
            "filas_escritas": len(filas), "en_almacen": n}


def conciliar(*, db: Path = DB_PATH, db_bullpen: Optional[Path] = None) -> Dict[str, Any]:
    """Suma las apariciones y las compara contra los agregados por equipo.

    Dos almacenes construidos del mismo documento por caminos distintos tienen
    que dar lo mismo. Si no dan, uno de los dos está mal y hay que saber cuál
    ANTES de usar ninguno — es la comprobación que este proyecto no hizo cuando
    combinó `fangraphs.pitcher.daily` con `savant.pitcher.rolling`.
    """
    from fbq.model.bullpen import DB_PATH as DB_BULLPEN

    a = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    b = sqlite3.connect(f"file:{db_bullpen or DB_BULLPEN}?mode=ro", uri=True)
    try:
        suma = {(int(pk), int(loc)): (int(rel or 0), int(tot or 0), int(nrel or 0))
                for pk, loc, rel, tot, nrel in a.execute(
                    """SELECT game_pk, es_local,
                              SUM(CASE WHEN rol='relevo' THEN pitches ELSE 0 END),
                              SUM(pitches),
                              SUM(CASE WHEN rol='relevo' THEN 1 ELSE 0 END)
                       FROM aparicion GROUP BY game_pk, es_local""")}
        agg = {(int(pk), int(loc)): (int(rel), int(tot), int(nrel))
               for pk, loc, rel, tot, nrel in b.execute(
                   "SELECT game_pk, es_local, pitches_relevo, pitches_total, "
                   "relevistas FROM relevo")}
    finally:
        a.close(); b.close()

    comunes = set(suma) & set(agg)
    dif = {"pitches_relevo": [], "pitches_total": [], "relevistas": []}
    for k in sorted(comunes):
        for i, campo in enumerate(("pitches_relevo", "pitches_total", "relevistas")):
            if suma[k][i] != agg[k][i]:
                dif[campo].append({"game_pk": k[0], "es_local": k[1],
                                   "por_relevista": suma[k][i], "agregado": agg[k][i]})
    return {
        "equipos_partido_en_relevistas": len(suma),
        "equipos_partido_en_agregado": len(agg),
        "comunes": len(comunes),
        "solo_en_relevistas": len(set(suma) - set(agg)),
        "solo_en_agregado": len(set(agg) - set(suma)),
        "discrepancias": {k: len(v) for k, v in dif.items()},
        "ejemplos": {k: v[:5] for k, v in dif.items() if v},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conciliar", action="store_true",
                    help="sólo concilia contra los agregados, sin reconstruir")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if not a.conciliar:
        log.info("%s", json.dumps(construir(), ensure_ascii=False))
    log.info("conciliación: %s", json.dumps(conciliar(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
