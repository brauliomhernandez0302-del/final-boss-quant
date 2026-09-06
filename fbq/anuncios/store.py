"""fbq/anuncios/store.py — almacén append-only de abridores anunciados.

Mismas reglas que `market/`, y por el mismo motivo: lo valioso es la serie de
observaciones, no la última. Un anuncio que cambia —un abridor adelantado, una
cancelación— es la información más cara de todas, y se pierde entera si la
captura pisa el valor anterior.

    `anuncio` es append-only por TRIGGER. Un UPDATE aborta.

Se deduplica por cambio: si el anuncio no se movió entre dos barridas no se
inserta una fila nueva, pero la barrida queda registrada en `barrida`, así que
"no cambió" siempre se distingue de "no miramos". Es la misma decisión que
`market.sweep`, tomada por la misma razón.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

from fbq.core.clock import ahora, normalizar_utc

DB_PATH = Path(__file__).parent.parent.parent / "data" / "anuncios.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS anuncio (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    observado_en   TEXT    NOT NULL,   -- cuándo lo vimos NOSOTROS. Lo único irrepetible.
    game_pk        INTEGER NOT NULL,
    official_date  TEXT,
    commence_time  TEXT,
    lado           TEXT    NOT NULL,   -- home | away
    pitcher_id     INTEGER,            -- NULL = sin anuncio todavía, y eso es información
    pitcher_nombre TEXT,
    estado         TEXT
);

CREATE INDEX IF NOT EXISTS ix_anuncio_juego ON anuncio (game_pk, lado, observado_en);
CREATE INDEX IF NOT EXISTS ix_anuncio_tiempo ON anuncio (observado_en);

CREATE TRIGGER IF NOT EXISTS trg_anuncio_no_update
BEFORE UPDATE ON anuncio
BEGIN
    SELECT RAISE(ABORT, 'anuncio es append-only: un anuncio que cambia es el dato más valioso y pisarlo lo destruye');
END;

CREATE TRIGGER IF NOT EXISTS trg_anuncio_no_delete
BEFORE DELETE ON anuncio
BEGIN
    SELECT RAISE(ABORT, 'anuncio es append-only: borrar destruye la serie de observaciones');
END;

CREATE TABLE IF NOT EXISTS barrida (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    observado_en TEXT    NOT NULL,
    fecha        TEXT    NOT NULL,   -- qué día del schedule se pidió
    n_juegos     INTEGER NOT NULL,
    n_nuevos     INTEGER NOT NULL,
    n_sin_cambio INTEGER NOT NULL
);
"""


class AnunciosStore:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def registrar(self, filas: Iterable[Dict[str, Any]], *, fecha: str,
                  observado_en: Optional[str] = None) -> Dict[str, int]:
        """Guarda una barrida. Sólo inserta lo que cambió."""
        ts = normalizar_utc(observado_en or ahora())
        filas = list(filas)
        nuevas, sin_cambio = [], 0
        with self._conn() as conn:
            for f in filas:
                for lado in ("home", "away"):
                    pid = f.get(f"{lado}_pitcher_id")
                    ult = conn.execute(
                        "SELECT pitcher_id FROM anuncio WHERE game_pk=? AND lado=? "
                        "ORDER BY id DESC LIMIT 1", (f["game_pk"], lado)).fetchone()
                    if ult is not None and ult["pitcher_id"] == pid:
                        sin_cambio += 1
                        continue
                    nuevas.append((ts, f["game_pk"], f.get("official_date"),
                                   f.get("commence_time"), lado, pid,
                                   f.get(f"{lado}_pitcher"), f.get("estado")))
            if nuevas:
                conn.executemany(
                    """INSERT INTO anuncio (observado_en, game_pk, official_date,
                       commence_time, lado, pitcher_id, pitcher_nombre, estado)
                       VALUES (?,?,?,?,?,?,?,?)""", nuevas)
            conn.execute(
                "INSERT INTO barrida (observado_en, fecha, n_juegos, n_nuevos, n_sin_cambio)"
                " VALUES (?,?,?,?,?)", (ts, fecha, len(filas), len(nuevas), sin_cambio))
        return {"juegos": len(filas), "nuevos": len(nuevas), "sin_cambio": sin_cambio}

    def vigente_antes(self, game_pk: int, lado: str, corte: str) -> Optional[sqlite3.Row]:
        """El último anuncio observado ESTRICTAMENTE antes del corte.

        Es la única lectura que un modelo tiene permitido hacer: devuelve lo que
        se sabía en ese instante, no lo que terminó pasando. Sin observación
        previa al corte devuelve None — y el juego se excluye, no se rellena.
        """
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM anuncio WHERE game_pk=? AND lado=? AND observado_en < ? "
                "ORDER BY observado_en DESC, id DESC LIMIT 1",
                (int(game_pk), lado, corte)).fetchone()

    def resumen(self) -> Dict[str, Any]:
        with self._conn() as conn:
            q = conn.execute
            return {
                "anuncios": q("SELECT COUNT(*) FROM anuncio").fetchone()[0],
                "juegos": q("SELECT COUNT(DISTINCT game_pk) FROM anuncio").fetchone()[0],
                "barridas": q("SELECT COUNT(*) FROM barrida").fetchone()[0],
                "primera": q("SELECT MIN(observado_en) FROM anuncio").fetchone()[0],
                "ultima": q("SELECT MAX(observado_en) FROM anuncio").fetchone()[0],
            }
