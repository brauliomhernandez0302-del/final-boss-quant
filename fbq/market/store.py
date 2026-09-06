"""market/store.py — esquema y acceso al almacén de precios.

Cinco decisiones de diseño, y las cinco vienen de un defecto real que este
proyecto ya pagó:

1. **Append-only, impuesto por la base de datos.** Dos triggers abortan
   cualquier UPDATE o DELETE sobre `odds_snapshot`. No es una convención que
   se respete por disciplina: `capture_closing_line()` en `track_record/db.py`
   hace doce UPDATE por día sobre la misma fila, cada uno borrando la captura
   anterior, y nadie lo notó por meses porque el docstring lo describía como
   una feature ("last pre-start capture wins"). Acá el motor rechaza el
   intento.

2. **Los dos lados de cada mercado, siempre.** Sin el par no se puede
   desvigorizar, y sin desvigorizar no hay ni precio justo ni CLV — solo el
   margen de la casa disfrazado de habilidad (medido: +1.514% de CLV medio
   pasó a −0.497% al desvigorizar). Guardar un lado suelto es guardar un dato
   que no se puede usar.

3. **El punto va FIRMADO y pegado a su lado.** Cada fila lleva el `point` del
   `side` de esa misma fila, tal como lo cotiza el libro. Es la defensa
   estructural contra PURP-1: emparejar la probabilidad de un evento con el
   precio de otro es imposible si el punto nunca se separa de su lado.

4. **`event_id` explícito, resuelto una vez.** El emparejamiento
   juego↔evento-de-odds es una identidad, no una búsqueda. Se resuelve una
   vez y se guarda en `event_link`; nunca se re-deriva por nombre de equipo en
   cada llamada, que es donde los doubleheaders fallan en silencio.

5. **El libro es parte del dato.** Un precio sin saber de qué casa salió no es
   comparable con nada (en el ledger actual, 324 de 464 picks no lo tienen).

El deduplicado por cambio de precio NO es una excepción a (1): una cotización
que no se movió entre dos barridas no es una observación nueva. Que la barrida
ocurrió y cubrió el evento queda registrado en `sweep`, así que "no hay fila"
siempre se distingue de "no miramos".
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

DB_PATH = Path(__file__).parent.parent.parent / "data" / "market.db"

# Mercados que se guardan, con el nombre que usa The Odds API. Se guarda el
# nombre crudo del proveedor a propósito: traducirlo acá obligaría a mantener
# un diccionario que se desincroniza, y el consumidor puede traducir.
MARKETS = ("h2h", "totals", "spreads")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshot (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at   TEXT    NOT NULL,   -- UTC ISO-8601: cuándo lo vimos NOSOTROS
    book_update   TEXT,               -- UTC ISO-8601: cuándo lo movió EL LIBRO
    sport_key     TEXT    NOT NULL,
    event_id      TEXT    NOT NULL,   -- id del proveedor, tal cual
    commence_time TEXT    NOT NULL,   -- inicio programado, UTC
    home_team     TEXT    NOT NULL,
    away_team     TEXT    NOT NULL,
    book          TEXT    NOT NULL,
    market        TEXT    NOT NULL,   -- h2h | totals | spreads
    side          TEXT    NOT NULL,   -- home | away | draw | over | under
    point         REAL,               -- FIRMADO desde la perspectiva de `side`
    price_dec     REAL    NOT NULL    -- decimal
);

CREATE INDEX IF NOT EXISTS ix_snap_lookup
    ON odds_snapshot (event_id, market, side, book, captured_at);
CREATE INDEX IF NOT EXISTS ix_snap_time
    ON odds_snapshot (captured_at);
CREATE INDEX IF NOT EXISTS ix_snap_commence
    ON odds_snapshot (sport_key, commence_time);
-- El evaluador y el modelo piden "el último precio de UN libro en UN mercado"
-- sobre toda la tabla. Sin este índice esa consulta barre las 143.899 filas y
-- tarda ~2 minutos, que es lo que hacía impracticable reconstruir el marco por
-- cada partido perturbado en el test de invariancia.
CREATE INDEX IF NOT EXISTS ix_snap_libro
    ON odds_snapshot (sport_key, market, book, event_id, side, captured_at);

-- Append-only impuesto por el motor, no por convención.
CREATE TRIGGER IF NOT EXISTS trg_snapshot_no_update
BEFORE UPDATE ON odds_snapshot
BEGIN
    SELECT RAISE(ABORT, 'odds_snapshot es append-only: un precio observado no se corrige, se agrega otro');
END;

CREATE TRIGGER IF NOT EXISTS trg_snapshot_no_delete
BEFORE DELETE ON odds_snapshot
BEGIN
    SELECT RAISE(ABORT, 'odds_snapshot es append-only: borrar una observación destruye la serie');
END;

-- Toda barrida queda registrada, se haya movido o no un solo precio. Es lo
-- que permite distinguir "el mercado no se movió" de "no miramos".
CREATE TABLE IF NOT EXISTS sweep (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at   TEXT    NOT NULL,
    sport_key     TEXT    NOT NULL,
    n_events      INTEGER NOT NULL,
    n_rows_new    INTEGER NOT NULL,   -- filas realmente insertadas
    n_unchanged   INTEGER NOT NULL,   -- cotizaciones vistas sin movimiento
    n_skipped     INTEGER NOT NULL,   -- outcomes que no se pudieron normalizar
    note          TEXT
);

CREATE INDEX IF NOT EXISTS ix_sweep_time ON sweep (captured_at);

-- Identidad juego↔evento: se resuelve una vez y se guarda.
CREATE TABLE IF NOT EXISTS event_link (
    event_id      TEXT    PRIMARY KEY,
    sport_key     TEXT    NOT NULL,
    game_pk       INTEGER,
    official_date TEXT,               -- día oficial de schedule, NO date(UTC)
    commence_time TEXT,
    home_team     TEXT,
    away_team     TEXT,
    method        TEXT,               -- cómo se resolvió (auditable)
    linked_at     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_link_game ON event_link (game_pk);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MarketStore:
    """Acceso al almacén de precios. Solo inserta y lee."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ── Escritura ────────────────────────────────────────────────────────

    def record_board(
        self,
        raw_events: Iterable[Dict[str, Any]],
        *,
        captured_at: Optional[str] = None,
        sport_keys: Optional[Iterable[str]] = None,
        dedupe: bool = True,
        note: Optional[str] = None,
    ) -> Dict[str, int]:
        """Guarda el board completo tal como lo devolvió el proveedor.

        `raw_events` son los eventos CRUDOS (con su lista `bookmakers`
        intacta), no la forma normalizada de la UI: normalizar antes de
        guardar es exactamente cómo se perdieron los precios de los derivados
        hasta el 2026-07-26 — el dato venía en la respuesta y se descartaba en
        el parseo.

        Con `dedupe=True` (default) solo se inserta la cotización que cambió
        respecto de la última guardada para esa misma llave
        (event_id, book, market, side). La barrida queda registrada en `sweep`
        de todos modos.
        """
        ts = captured_at or _utc_now()
        wanted = set(sport_keys) if sport_keys else None

        rows: List[Tuple] = []
        skipped = 0
        events_seen = 0
        by_sport: Dict[str, int] = {}

        for event in raw_events:
            sport_key = event.get("sport_key") or ""
            if wanted is not None and sport_key not in wanted:
                continue

            event_id = event.get("id")
            if not event_id:
                # Sin identidad del proveedor no se puede enlazar ni comparar
                # a lo largo del tiempo. Se cuenta como salto, no se inventa
                # una llave sintética a partir de los nombres de equipo.
                skipped += 1
                continue

            events_seen += 1
            by_sport[sport_key] = by_sport.get(sport_key, 0) + 1
            home = event.get("home_team") or ""
            away = event.get("away_team") or ""
            commence = event.get("commence_time") or ""

            for bk in event.get("bookmakers") or []:
                book = bk.get("key") or bk.get("title")
                if not book:
                    skipped += 1
                    continue
                for mkt in bk.get("markets") or []:
                    mkey = mkt.get("key")
                    if mkey not in MARKETS:
                        continue
                    book_update = mkt.get("last_update") or bk.get("last_update")
                    for out in mkt.get("outcomes") or []:
                        side = _normalize_side(mkey, out.get("name"), home, away)
                        price = out.get("price")
                        if side is None or not price or price <= 1.0:
                            skipped += 1
                            continue
                        rows.append((
                            ts, book_update, sport_key, event_id, commence,
                            home, away, book, mkey, side,
                            out.get("point"), float(price),
                        ))

        unchanged = 0
        if dedupe and rows:
            rows, unchanged = self._drop_unchanged(rows)

        with self._conn() as conn:
            if rows:
                conn.executemany(
                    """INSERT INTO odds_snapshot
                       (captured_at, book_update, sport_key, event_id,
                        commence_time, home_team, away_team, book, market,
                        side, point, price_dec)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    rows,
                )
            for sport_key, n_ev in (by_sport or {"": 0}).items():
                conn.execute(
                    """INSERT INTO sweep (captured_at, sport_key, n_events,
                                          n_rows_new, n_unchanged, n_skipped, note)
                       VALUES (?,?,?,?,?,?,?)""",
                    (ts, sport_key, n_ev, len(rows), unchanged, skipped, note),
                )

        return {
            "events": events_seen,
            "rows_new": len(rows),
            "unchanged": unchanged,
            "skipped": skipped,
        }

    def _drop_unchanged(self, rows: List[Tuple]) -> Tuple[List[Tuple], int]:
        """Quita las cotizaciones idénticas a la última guardada.

        Una llave es (event_id, book, market, side); "idéntica" es mismo
        `point` Y mismo `price_dec`. Un precio que se va y vuelve al mismo
        valor SÍ genera dos filas — es lo correcto, son dos observaciones
        distintas separadas por un movimiento.
        """
        keys = {(r[3], r[7], r[8], r[9]) for r in rows}
        last: Dict[Tuple, Tuple] = {}
        with self._conn() as conn:
            for key in keys:
                cur = conn.execute(
                    """SELECT point, price_dec FROM odds_snapshot
                       WHERE event_id=? AND book=? AND market=? AND side=?
                       ORDER BY id DESC LIMIT 1""",
                    key,
                )
                row = cur.fetchone()
                if row is not None:
                    last[key] = (row["point"], row["price_dec"])

        kept: List[Tuple] = []
        unchanged = 0
        for r in rows:
            key = (r[3], r[7], r[8], r[9])
            prev = last.get(key)
            if prev is not None and prev[0] == r[10] and prev[1] == r[11]:
                unchanged += 1
                continue
            kept.append(r)
            # Dentro de una misma barrida un libro puede aparecer dos veces
            # (mercados alternativos); la segunda ya se compara contra la
            # primera y no se duplica.
            last[key] = (r[10], r[11])
        return kept, unchanged

    # ── Lectura ──────────────────────────────────────────────────────────

    def latest_before(
        self,
        event_id: str,
        market: str,
        side: str,
        before: str,
        *,
        book: Optional[str] = None,
        solo_pregame: bool = True,
    ) -> Optional[sqlite3.Row]:
        """La última cotización observada estrictamente ANTES de `before`.

        `before` es un corte explícito y obligatorio: no existe "el precio de
        este evento" sin decir a qué momento. Es el mismo contrato de tiempo
        que el resto del proyecto aplica a los datos de juego.

        `solo_pregame=True` por defecto, y no es un detalle. El proveedor sigue
        devolviendo el evento después del primer lanzamiento, con precios EN
        VIVO: sobre la captura del 2026-08-04, el 31.5% de las cotizaciones de
        MLB eran post-inicio, con moneylines de hasta 150.0 a dos horas y media
        del comienzo. Mezclarlas con las pre-juego produce pares con overround
        negativo que parecen arbitraje y son sólo dos mercados distintos —
        pasó tres veces seguidas al analizar esta misma tabla.

        No se borran: un precio en vivo es un dato legítimo de otro producto.
        Se pide explícitamente con `solo_pregame=False`.
        """
        sql = ("SELECT * FROM odds_snapshot WHERE event_id=? AND market=? "
               "AND side=? AND captured_at < ?")
        params: List[Any] = [event_id, market, side, before]
        if solo_pregame:
            sql += " AND captured_at < commence_time"
        if book:
            sql += " AND book=?"
            params.append(book)
        sql += " ORDER BY captured_at DESC, id DESC LIMIT 1"
        with self._conn() as conn:
            return conn.execute(sql, params).fetchone()

    def pair_before(
        self,
        event_id: str,
        market: str,
        side: str,
        before: str,
        *,
        book: str = "pinnacle",
        solo_pregame: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """El par de precios de UN mismo libro para desvigorizar `side`.

        Devuelve None si falta cualquiera de los dos lados: medio par no
        sirve, y devolver el lado solo invitaría a compararlo contra un precio
        crudo, que es precisamente el error que hacía ver habilidad donde solo
        había margen de la casa.

        Verifica además que ambos lados estén cotizados en el MISMO punto
        (`point`), porque un total de 8.5 y uno de 9.0 no son dos lados del
        mismo mercado.
        """
        opposite = _opposite_side(market, side)
        if opposite is None:
            return None
        a = self.latest_before(event_id, market, side, before, book=book,
                               solo_pregame=solo_pregame)
        b = self.latest_before(event_id, market, opposite, before, book=book,
                               solo_pregame=solo_pregame)
        if a is None or b is None:
            return None
        if market in ("totals", "spreads"):
            pa, pb = a["point"], b["point"]
            if pa is None or pb is None or abs(abs(pa) - abs(pb)) > 1e-9:
                return None
        return {
            "book": book,
            "market": market,
            "side": side,
            "point": a["point"],
            "price_side": a["price_dec"],
            "price_opposite": b["price_dec"],
            "captured_at": max(a["captured_at"], b["captured_at"]),
        }

    def trajectory(
        self,
        event_id: str,
        market: str,
        side: str,
        *,
        book: Optional[str] = None,
        solo_pregame: bool = True,
    ) -> List[sqlite3.Row]:
        """Toda la serie observada de una cotización, en orden."""
        sql = ("SELECT * FROM odds_snapshot WHERE event_id=? AND market=? "
               "AND side=?")
        params: List[Any] = [event_id, market, side]
        if solo_pregame:
            sql += " AND captured_at < commence_time"
        if book:
            sql += " AND book=?"
            params.append(book)
        sql += " ORDER BY captured_at ASC, id ASC"
        with self._conn() as conn:
            return conn.execute(sql, params).fetchall()

    # ── Identidad juego ↔ evento ─────────────────────────────────────────

    def link_event(
        self,
        event_id: str,
        *,
        sport_key: str,
        game_pk: Optional[int],
        official_date: Optional[str],
        commence_time: Optional[str] = None,
        home_team: Optional[str] = None,
        away_team: Optional[str] = None,
        method: str = "manual",
    ) -> None:
        """Fija la identidad de un evento. Idempotente por `event_id`.

        `official_date` es el día oficial de schedule de MLB, no
        `date(commence_time)`: para cualquier juego nocturno de la costa oeste
        el timestamp UTC cae un día calendario adelante, y ese desfase ya costó
        un leak completo en el camino PIT (Fase 2B).
        """
        self.link_events([{
            "event_id": event_id, "sport_key": sport_key, "game_pk": game_pk,
            "official_date": official_date, "commence_time": commence_time,
            "home_team": home_team, "away_team": away_team, "method": method,
        }])

    def link_events(self, enlaces: Iterable[Dict[str, Any]]) -> int:
        """Fija muchas identidades en una sola transacción. Idempotente.

        Existe por una razón de costo, no de estilo: la importación del
        histórico enlaza ~6.000 eventos de una vez, y hacerlo con una conexión
        y un `COMMIT` por evento convierte segundos en minutos. El SQL vive acá
        una sola vez y `link_event()` delega, para que no haya dos sentencias
        que puedan divergir.
        """
        filas = [
            (e["event_id"], e["sport_key"], e.get("game_pk"),
             e.get("official_date"), e.get("commence_time"),
             e.get("home_team"), e.get("away_team"),
             e.get("method", "manual"), _utc_now())
            for e in enlaces
        ]
        if not filas:
            return 0
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO event_link
                   (event_id, sport_key, game_pk, official_date, commence_time,
                    home_team, away_team, method, linked_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(event_id) DO UPDATE SET
                       game_pk       = COALESCE(excluded.game_pk, event_link.game_pk),
                       official_date = COALESCE(excluded.official_date, event_link.official_date),
                       method        = excluded.method,
                       linked_at     = excluded.linked_at""",
                filas,
            )
        return len(filas)

    def event_for_game(self, game_pk: int) -> Optional[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM event_link WHERE game_pk=?", (game_pk,)
            ).fetchone()

    def unlinked_events(self, sport_key: Optional[str] = None) -> List[sqlite3.Row]:
        """Eventos con precios guardados que todavía no tienen `game_pk`."""
        sql = """SELECT DISTINCT s.event_id, s.sport_key, s.commence_time,
                        s.home_team, s.away_team
                 FROM odds_snapshot s
                 LEFT JOIN event_link l ON l.event_id = s.event_id
                 WHERE l.game_pk IS NULL"""
        params: List[Any] = []
        if sport_key:
            sql += " AND s.sport_key=?"
            params.append(sport_key)
        with self._conn() as conn:
            return conn.execute(sql, params).fetchall()

    # ── Diagnóstico ──────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        with self._conn() as conn:
            q = conn.execute
            return {
                "rows": q("SELECT COUNT(*) FROM odds_snapshot").fetchone()[0],
                "events": q("SELECT COUNT(DISTINCT event_id) FROM odds_snapshot").fetchone()[0],
                "books": q("SELECT COUNT(DISTINCT book) FROM odds_snapshot").fetchone()[0],
                "sweeps": q("SELECT COUNT(*) FROM sweep").fetchone()[0],
                "first": q("SELECT MIN(captured_at) FROM odds_snapshot").fetchone()[0],
                "last": q("SELECT MAX(captured_at) FROM odds_snapshot").fetchone()[0],
                "linked": q("SELECT COUNT(*) FROM event_link WHERE game_pk IS NOT NULL").fetchone()[0],
            }


# ── Normalización ────────────────────────────────────────────────────────


def _normalize_side(
    market: str, name: Optional[str], home: str, away: str
) -> Optional[str]:
    """Nombre del outcome → lado canónico.

    Compara contra los nombres de equipo del PROPIO evento, que es la única
    referencia correcta: no hay diccionario de alias ni fuzzy matching, así
    que no hay forma de que un equipo se resuelva mal en silencio. Un nombre
    que no coincide devuelve None y el llamador lo cuenta como salto.
    """
    if not name:
        return None
    if market == "totals":
        low = name.strip().lower()
        return {"over": "over", "under": "under"}.get(low)
    if name == home:
        return "home"
    if name == away:
        return "away"
    if market == "h2h" and name.strip().lower() == "draw":
        return "draw"
    return None


def _opposite_side(market: str, side: str) -> Optional[str]:
    if market == "totals":
        return {"over": "under", "under": "over"}.get(side)
    if market in ("h2h", "spreads"):
        return {"home": "away", "away": "home"}.get(side)
    return None
