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
    -- Tres instantes distintos, y hacen falta los tres:
    --   corte          hasta dónde MIRÓ el modelo
    --   generado_utc   cuándo se CALCULÓ la probabilidad
    --   registrado_utc cuándo se ESCRIBIÓ en esta tabla
    -- Un corte anterior al partido no demuestra nada por sí solo: se puede
    -- declarar cualquier corte en cualquier momento. Lo que demuestra que la
    -- predicción existía antes del primer lanzamiento es `registrado_utc`.
    generado_utc   TEXT,
    registrado_utc TEXT,
    -- prospectiva_verificada | reconstruccion | no_verificable
    origen         TEXT    NOT NULL DEFAULT 'no_verificable',
    UNIQUE (game_pk, version, corte, modelo_sha)
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

-- Y una fila sólo puede llamarse prospectiva si se ESCRIBIÓ antes del inicio.
-- El corte lo declara quien escribe; `registrado_utc` lo pone el motor.
CREATE TRIGGER IF NOT EXISTS trg_pred_prospectiva_verificable
BEFORE INSERT ON prediccion
WHEN NEW.origen = 'prospectiva_verificada'
 AND (NEW.registrado_utc IS NULL OR NEW.registrado_utc >= NEW.commence_time)
BEGIN
    SELECT RAISE(ABORT, 'no se puede marcar prospectiva_verificada sin registrado_utc anterior al primer lanzamiento');
END;
"""

# Columnas agregadas el 2026-09-07. `ADD COLUMN` no reescribe filas y por eso no
# despierta el trigger de append-only; las filas viejas quedan con el default,
# que es exactamente lo que corresponde: su emisión ya no se puede demostrar.
_MIGRACIONES = (
    ("generado_utc", "TEXT"),
    ("registrado_utc", "TEXT"),
    ("origen", "TEXT NOT NULL DEFAULT 'no_verificable'"),
)

# La llave única pasó de (game_pk, version, corte) a incluir `modelo_sha`, y
# `CREATE TABLE IF NOT EXISTS` NO recrea una tabla existente: el constraint
# viejo seguía vigente y hacía que un ajuste nuevo colisionara con el anterior
# sobre el mismo juego y corte. Se detectó porque una regeneración con un sha
# distinto reportó "guardadas: 0".
#
# La migración reconstruye la tabla y COPIA todas las filas. No se pierde
# ninguna: el conteo se verifica antes de reemplazar, y si no coincide se
# aborta dejando el original intacto.
_LLAVE_NUEVA = "game_pk, version, corte, modelo_sha"


def _migrar_llave(conn) -> None:
    idx = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='prediccion'"
    ).fetchone()
    if idx is None or "modelo_sha" in (idx[0] or "").split("UNIQUE")[-1]:
        return
    # Respaldo ANTES de tocar nada. Una migración que sale mal sin copia previa
    # es exactamente cómo este proyecto perdió las emisiones originales.
    try:
        from fbq.respaldo import respaldar
        respaldar(["prospectiva.db"], motivo="antes_de_migracion")
    except Exception:                                     # noqa: BLE001
        pass
    n_antes = conn.execute("SELECT COUNT(*) FROM prediccion").fetchone()[0]
    cols = [r[1] for r in conn.execute("PRAGMA table_info(prediccion)")]
    lista = ", ".join(cols)
    conn.executescript(_SCHEMA.replace("prediccion", "prediccion_nueva")
                       .replace("trg_pred_", "trg_predn_"))
    conn.execute(f"INSERT INTO prediccion_nueva ({lista}) SELECT {lista} FROM prediccion")
    n_copiadas = conn.execute("SELECT COUNT(*) FROM prediccion_nueva").fetchone()[0]
    if n_copiadas != n_antes:
        conn.execute("DROP TABLE prediccion_nueva")
        raise RuntimeError(
            f"migración abortada: {n_antes} filas antes, {n_copiadas} copiadas")
    conn.executescript(
        "DROP TRIGGER IF EXISTS trg_pred_no_update;"
        "DROP TRIGGER IF EXISTS trg_pred_no_delete;"
        "DROP TABLE prediccion;"
        "ALTER TABLE prediccion_nueva RENAME TO prediccion;")
    conn.executescript(_SCHEMA)


class Prospectiva:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)
            existentes = {r[1] for r in c.execute("PRAGMA table_info(prediccion)")}
            for nombre, tipo in _MIGRACIONES:
                if nombre not in existentes:
                    c.execute(f"ALTER TABLE prediccion ADD COLUMN {nombre} {tipo}")
            _migrar_llave(c)

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
        """Inserta lo que no esté, sellando el instante de ESCRITURA.

        `registrado_utc` lo pone el motor, no quien llama: es lo único que
        demuestra que la fila existía antes del primer lanzamiento. El `origen`
        se deriva de ese sello, no se declara.
        """
        if not filas:
            return 0
        with self._conn() as conn:
            antes = conn.execute("SELECT COUNT(*) FROM prediccion").fetchone()[0]
            sellados = []
            for f in filas:
                ahora_utc = normalizar_utc(
                    datetime.now(timezone.utc).isoformat())
                origen = f.get("origen")
                if origen is None:
                    origen = ("prospectiva_verificada"
                              if ahora_utc < f["commence_time"] else "reconstruccion")
                sellados.append({**f, "registrado_utc": ahora_utc,
                                 "generado_utc": f.get("generado_utc"),
                                 "origen": origen})
            conn.executemany(
                """INSERT OR IGNORE INTO prediccion
                   (corte, game_pk, official_date, commence_time, home_team,
                    away_team, version, p_home, variables_json, cohorte,
                    modelo_sha, generado_utc, registrado_utc, origen)
                   VALUES (:corte,:game_pk,:official_date,:commence_time,:home_team,
                           :away_team,:version,:p_home,:variables_json,:cohorte,
                           :modelo_sha,:generado_utc,:registrado_utc,:origen)""",
                sellados)
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
                "por_origen": {r[0]: r[1] for r in q(
                    "SELECT origen, COUNT(*) FROM prediccion GROUP BY origen")},
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
