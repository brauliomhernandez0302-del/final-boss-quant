"""fbq/model/prospectiva.py — predicciones pareadas de v1.2 y v1.4, guardadas ANTES del partido.

Es lo único que produce cifras que nadie escribió conociendo el resultado. Todo
lo demás de este proyecto es histórico y, sobre 2024-2026, además explorado.

## Las tres reglas

1. **Los dos modelos, el MISMO corte, la MISMA lectura.** Se calculan juntos, en
   una sola pasada, sobre el mismo instante y los mismos datos. Dos procesos
   separados no lo garantizan: cada uno llamaría al reloj por su cuenta y
   leería el almacén en un instante distinto, y ahí las dos versiones dejan de
   ser comparables sin que nadie lo note.
2. **El ajuste está CONGELADO.** Los coeficientes se entrenan una vez, se
   escriben a un archivo con su fecha y su muestra, y se leen de ahí. Un modelo
   que se re-entrena solo cada noche no produce una serie prospectiva: produce
   una sucesión de modelos distintos evaluados una vez cada uno.
3. **Antes del primer lanzamiento, o no se guarda.** Una predicción con corte
   posterior al inicio no es una predicción.

`prediccion` es append-only por trigger y tiene una llave única por
`(game_pk, version, corte)`: repetir la corrida no duplica, y el cron puede
disparar cada hora sin pensarlo.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence

import numpy as np

from fbq.core.clock import ahora, normalizar_utc

RAIZ = Path(__file__).parent.parent.parent
DB_PATH = RAIZ / "data" / "prospectiva.db"
CONGELADO = RAIZ / "docs" / "modelos_congelados_2026-09-06.json"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prediccion (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    corte          TEXT    NOT NULL,   -- el instante en que se predijo
    game_pk        INTEGER NOT NULL,
    official_date  TEXT,
    commence_time  TEXT    NOT NULL,
    home_team      TEXT,
    away_team      TEXT,
    version        TEXT    NOT NULL,   -- v1.2 | v1.4
    p_home         REAL    NOT NULL,
    variables_json TEXT    NOT NULL,   -- las x usadas, para poder auditarla
    cohorte        TEXT    NOT NULL,   -- prospectiva | historica_42
    modelo_sha     TEXT    NOT NULL,   -- huella del ajuste congelado
    UNIQUE (game_pk, version, corte)
);

CREATE INDEX IF NOT EXISTS ix_pred_juego ON prediccion (game_pk, version);
CREATE INDEX IF NOT EXISTS ix_pred_corte ON prediccion (corte);

CREATE TRIGGER IF NOT EXISTS trg_pred_no_update
BEFORE UPDATE ON prediccion
BEGIN
    SELECT RAISE(ABORT, 'prediccion es append-only: una predicción publicada no se corrige');
END;

CREATE TRIGGER IF NOT EXISTS trg_pred_no_delete
BEFORE DELETE ON prediccion
BEGIN
    SELECT RAISE(ABORT, 'prediccion es append-only: borrarla destruye la única evidencia de que se hizo antes');
END;

-- Una predicción POSTERIOR al primer lanzamiento no es una predicción.
CREATE TRIGGER IF NOT EXISTS trg_pred_antes_del_inicio
BEFORE INSERT ON prediccion
WHEN NEW.corte >= NEW.commence_time
BEGIN
    SELECT RAISE(ABORT, 'el corte es posterior al primer lanzamiento: eso no es una predicción');
END;
"""


class Prospectiva:
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

    def guardar(self, filas: Sequence[Dict[str, Any]]) -> int:
        """Inserta lo que no esté. Una llave repetida se ignora, no falla."""
        if not filas:
            return 0
        with self._conn() as conn:
            antes = conn.execute("SELECT COUNT(*) FROM prediccion").fetchone()[0]
            conn.executemany(
                """INSERT OR IGNORE INTO prediccion
                   (corte, game_pk, official_date, commence_time, home_team,
                    away_team, version, p_home, variables_json, cohorte, modelo_sha)
                   VALUES (:corte,:game_pk,:official_date,:commence_time,:home_team,
                           :away_team,:version,:p_home,:variables_json,:cohorte,:modelo_sha)""",
                filas)
            return conn.execute("SELECT COUNT(*) FROM prediccion").fetchone()[0] - antes

    def resumen(self) -> Dict[str, Any]:
        with self._conn() as conn:
            q = conn.execute
            return {
                "predicciones": q("SELECT COUNT(*) FROM prediccion").fetchone()[0],
                "juegos": q("SELECT COUNT(DISTINCT game_pk) FROM prediccion").fetchone()[0],
                "por_version": {r[0]: r[1] for r in q(
                    "SELECT version, COUNT(*) FROM prediccion GROUP BY version")},
                "por_cohorte": {r[0]: r[1] for r in q(
                    "SELECT cohorte, COUNT(DISTINCT game_pk) FROM prediccion GROUP BY cohorte")},
                "pareados": q(
                    "SELECT COUNT(*) FROM (SELECT game_pk, corte FROM prediccion "
                    "GROUP BY game_pk, corte HAVING COUNT(DISTINCT version)=2)").fetchone()[0],
                "primera": q("SELECT MIN(corte) FROM prediccion").fetchone()[0],
                "ultima": q("SELECT MAX(corte) FROM prediccion").fetchone()[0],
            }


# ── El ajuste congelado ──────────────────────────────────────────────────

def cargar_congelado(ruta: Path = CONGELADO) -> Dict[str, Any]:
    if not ruta.exists():
        raise FileNotFoundError(
            f"falta {ruta}: el ajuste tiene que congelarse ANTES de predecir "
            f"(`python3 -m fbq.model.congelar`)")
    return json.loads(ruta.read_text(encoding="utf-8"))


def aplicar(modelo: Dict[str, Any], x: Dict[str, float]) -> float:
    """Aplica un ajuste congelado. Falla si falta una variable: rellenarla con
    la media convertiría una predicción imposible en una plausible."""
    faltan = [n for n in modelo["variables"] if n not in x]
    if faltan:
        raise KeyError(f"faltan variables para {modelo['version']}: {faltan}")
    z = [(float(x[n]) - modelo["mu"][i]) / modelo["sd"][i]
         for i, n in enumerate(modelo["variables"])]
    lineal = modelo["beta"][0] + sum(b * zi for b, zi in zip(modelo["beta"][1:], z))
    return float(1.0 / (1.0 + np.exp(-lineal)))
