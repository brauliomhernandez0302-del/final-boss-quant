"""fbq/results/store.py — el almacén de hechos.

Cuatro reglas, todas nacidas de un defecto medido:

1. **Sólo hechos.** Ninguna columna de modelo. Ver la nota del paquete.

2. **Sólo se escribe un FINAL.** El estado se juzga por `detailed_state`, no por
   el abstracto: `abstractGameState` vale "Final" también para un juego
   POSPUESTO. Y cuando un juego se pospone y se rejuega, el schedule devuelve
   dos entradas con el mismo `game_pk` — hay que quedarse con la que terminó,
   no con la primera.

3. **Un empate no se guarda.** En MLB no existe un final empatado: un 5–5
   significa que lo que llegó no es un final. Escribirlo como derrota del local
   —que es lo que hace `1 if home > away else 0`— convierte un juego sin
   terminar en verdad de terreno.

4. **Una corrección agrega, no pisa.** El marcador oficial puede cambiar (una
   decisión de anotación, un protesto). La fila vigente es la última observada;
   las anteriores quedan. Sin esto, "el marcador cambió" es indistinguible de
   "lo escribimos mal", y la diferencia importa cuando algo entrenó con el viejo.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence

from fbq.core.clock import ahora
from fbq.core.identity import ESTADOS_FINALES

DB_PATH = Path(__file__).parent.parent.parent / "data" / "results.db"

# `ESTADOS_FINALES` se re-exporta desde `core.identity`, donde vive: la misma
# regla decide qué entrada del schedule es el partido cuando un juego se
# pospone y se rejuega, y eso lo necesitan tanto este almacén como `market/`.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observacion (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    observado_en   TEXT    NOT NULL,   -- cuándo lo vimos NOSOTROS
    game_pk        INTEGER NOT NULL,
    official_date  TEXT    NOT NULL,   -- día de schedule, NO date(UTC)
    season         INTEGER NOT NULL,
    home_team      TEXT    NOT NULL,
    away_team      TEXT    NOT NULL,
    home_runs      INTEGER NOT NULL,
    away_runs      INTEGER NOT NULL,
    detailed_state TEXT    NOT NULL,
    innings        INTEGER
);

CREATE INDEX IF NOT EXISTS ix_obs_game ON observacion (game_pk, id);
CREATE INDEX IF NOT EXISTS ix_obs_fecha ON observacion (official_date);

CREATE TRIGGER IF NOT EXISTS trg_obs_no_update
BEFORE UPDATE ON observacion
BEGIN
    SELECT RAISE(ABORT, 'observacion es append-only: una corrección agrega una fila nueva');
END;

CREATE TRIGGER IF NOT EXISTS trg_obs_no_delete
BEFORE DELETE ON observacion
BEGIN
    SELECT RAISE(ABORT, 'observacion es append-only: borrar destruye la historia de la corrección');
END;

-- La vista es la cara normal del almacén: un juego, su resultado vigente.
-- La tabla de abajo conserva cómo se llegó a él.
CREATE VIEW IF NOT EXISTS resultado AS
SELECT o.game_pk, o.official_date, o.season, o.home_team, o.away_team,
       o.home_runs, o.away_runs,
       CASE WHEN o.home_runs > o.away_runs THEN 1 ELSE 0 END AS home_won,
       o.detailed_state, o.innings, o.observado_en
FROM observacion o
JOIN (SELECT game_pk, MAX(id) AS ult FROM observacion GROUP BY game_pk) u
  ON u.ult = o.id;
"""


@dataclass(frozen=True)
class Final:
    """Un juego terminado. Construirlo ya valida que sea un final legítimo."""

    game_pk: int
    official_date: str
    season: int
    home_team: str
    away_team: str
    home_runs: int
    away_runs: int
    detailed_state: str
    innings: Optional[int] = None

    def __post_init__(self) -> None:
        if self.detailed_state not in ESTADOS_FINALES:
            raise ValueError(
                f"game_pk={self.game_pk}: estado {self.detailed_state!r} no es "
                f"un final. Los pospuestos y suspendidos comparten "
                f"abstractGameState='Final' con los terminados de verdad."
            )
        if self.home_runs == self.away_runs:
            raise ValueError(
                f"game_pk={self.game_pk}: marcador empatado "
                f"{self.away_runs}–{self.home_runs}. En MLB no hay finales "
                f"empatados, así que esto no es un final."
            )

    @property
    def home_won(self) -> int:
        return 1 if self.home_runs > self.away_runs else 0


class ResultsStore:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── Escritura ────────────────────────────────────────────────────────

    def registrar(self, finales: Iterable[Final]) -> Dict[str, int]:
        """Guarda finales. Sólo agrega una observación si CAMBIA el marcador.

        Re-observar el mismo resultado no genera fila: no es información nueva.
        Un marcador distinto sí la genera, y las dos quedan — la vista
        `resultado` devuelve la última y la tabla conserva que hubo corrección.
        """
        nuevos = corregidos = sin_cambio = 0
        ts = ahora()
        with self._conn() as conn:
            for f in finales:
                previo = conn.execute(
                    "SELECT home_runs, away_runs FROM observacion "
                    "WHERE game_pk=? ORDER BY id DESC LIMIT 1", (f.game_pk,),
                ).fetchone()
                if previo is not None:
                    if (previo["home_runs"], previo["away_runs"]) == (f.home_runs, f.away_runs):
                        sin_cambio += 1
                        continue
                    corregidos += 1
                else:
                    nuevos += 1
                conn.execute(
                    """INSERT INTO observacion
                       (observado_en, game_pk, official_date, season, home_team,
                        away_team, home_runs, away_runs, detailed_state, innings)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (ts, f.game_pk, f.official_date, f.season, f.home_team,
                     f.away_team, f.home_runs, f.away_runs, f.detailed_state,
                     f.innings),
                )
        return {"nuevos": nuevos, "corregidos": corregidos, "sin_cambio": sin_cambio}

    # ── Lectura ──────────────────────────────────────────────────────────

    def resultado(self, game_pk: int) -> Optional[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM resultado WHERE game_pk=?", (game_pk,)
            ).fetchone()

    def por_temporada(self, seasons: Sequence[int]) -> List[sqlite3.Row]:
        marcas = ",".join("?" * len(seasons))
        with self._conn() as conn:
            return conn.execute(
                f"SELECT * FROM resultado WHERE season IN ({marcas}) "
                f"ORDER BY official_date, game_pk", tuple(seasons),
            ).fetchall()

    def correcciones(self) -> List[sqlite3.Row]:
        """Juegos con más de una observación: dónde el marcador cambió."""
        with self._conn() as conn:
            return conn.execute(
                """SELECT game_pk, COUNT(*) n, MIN(observado_en) primera,
                          MAX(observado_en) ultima
                   FROM observacion GROUP BY game_pk HAVING n > 1
                   ORDER BY n DESC"""
            ).fetchall()

    def resumen(self) -> Dict[str, object]:
        with self._conn() as conn:
            q = conn.execute
            return {
                "juegos": q("SELECT COUNT(*) FROM resultado").fetchone()[0],
                "observaciones": q("SELECT COUNT(*) FROM observacion").fetchone()[0],
                "con_correccion": q(
                    "SELECT COUNT(*) FROM (SELECT game_pk FROM observacion "
                    "GROUP BY game_pk HAVING COUNT(*)>1)").fetchone()[0],
                "temporadas": [r[0] for r in q(
                    "SELECT DISTINCT season FROM resultado ORDER BY season")],
            }
