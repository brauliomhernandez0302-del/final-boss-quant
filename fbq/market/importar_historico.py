"""Importa los precios históricos del sistema anterior al almacén propio.

Una sola vez, y repetible sin daño. Corta la última dependencia de datos con el
sistema anterior: después de esto, `evaluator/` y `features/` pueden leer sólo
de almacenes propios.

Lo que se importa son HECHOS —precios observados el día del juego— y por eso
sobreviven a la invalidación del sistema anterior. Lo que NO se importa es nada
derivado de un modelo.

## Las decisiones que quedan escritas acá

**El consenso entra con nombre `_consensus`.** No es una casa: es un agregado
calculado sobre ~28 libros al momento de la captura. El guion bajo marca que es
derivado, para que ninguna consulta lo confunda con un precio real cotizado.
Se importa porque es un dato medido que no se puede reconstruir —sólo tenemos
dos libros por juego del histórico, no el board completo.

**El mejor precio entra con el nombre REAL de su casa**, que el histórico
guarda por lado (`ml_home_best_bk`, `ml_away_best_bk`). Pueden ser dos casas
distintas para el mismo juego, y así queda.

**Todas las filas llevan `captured_at` = 17:00Z del día del juego**, que es lo
que el sistema anterior pidió al proveedor. No es el cierre ni una trayectoria:
es una foto diaria, y hay que tratarla como tal.

**La hora de inicio se VERIFICA contra el schedule oficial de MLB.** Es la
corrección del 2026-09-06 y el motivo de que este archivo se haya reescrito.
La versión anterior tomaba `commence_time` de `game_outcomes.game_date` con el
comentario "es el timestamp UTC de inicio". No lo es: las 5.762 filas de esa
columna miden 10 caracteres, o sea que son fechas sin hora, y además son
`date(UTC)`, que difiere del día oficial en los nocturnos del oeste. El filtro
pre-juego del almacén compara cadenas dentro de SQL, así que
`'2024-04-24T17:00:00Z' < '2024-04-24'` evaluaba a FALSO y **todo lo importado
habría quedado invisible para cualquier lectura por defecto** — el modo de
fallo silencioso, que es el peligroso. Nunca llegó a correrse.

**Nada se descarta en silencio.** Una fila sin hora verificable o sin juego
identificable no se importa —inventarle una hora sería fabricar el dato que
justamente falta— pero queda escrita en un informe de conciliación con su
motivo. "No se pudo" y "no miramos" tienen que ser distinguibles.

**Repetir la importación no duplica.** La llave de una observación histórica es
`(captured_at, event_id, book, market, side)`; lo que ya está no se vuelve a
insertar. Hace falta porque `odds_snapshot` es append-only por trigger: un
INSERT repetido no se puede corregir después con un DELETE.

**Los precios posteriores al primer lanzamiento SÍ se importan.** Sobre esta
foto son 126 de 5.429 juegos (2,32%), con mediana de 45 minutos después del
comienzo. Son cotizaciones reales de otro producto —el mercado en vivo— y
borrarlas destruiría un dato legítimo. Quien evalúa pre-juego las excluye con
el filtro del almacén, que ahora funciona porque `commence_time` es un instante
de verdad.

Uso:
    python3 -m fbq.market.importar_historico --dry-run
    python3 -m fbq.market.importar_historico
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fbq.core.clock import normalizar_utc, tiene_hora
from fbq.core.identity import ESTADOS_FINALES
from fbq.market.store import MarketStore
from fbq.sources import mlb_stats

log = logging.getLogger(__name__)

RAIZ = Path(__file__).parent.parent.parent
ORIGEN = RAIZ / "data" / "predictions_history.db"
CACHE_SCHEDULE = RAIZ / "data" / "schedule_inicios.json"
PENDIENTES = RAIZ / "data" / "importacion_historica_pendientes.csv"
DEPORTE = "baseball_mlb"

# Marca de agregado derivado. El guion bajo es deliberado: ninguna casa real
# empieza así, y una consulta que filtre por libro no lo va a confundir.
CONSENSO = "_consensus"

MOTIVOS = {
    "sin_event_id": "el histórico no guardó el id del proveedor: sin él no hay serie temporal posible",
    "sin_juego_en_schedule": "el game_pk no aparece en el schedule oficial de MLB del rango pedido",
    "sin_hora_verificada": "el schedule trae el juego pero sin hora de inicio legible",
    "fecha_ambigua": "el game_pk tiene varias horas de inicio en el schedule y ninguna regla las desempata (típicamente un juego suspendido y reanudado otro día: dos entradas terminadas con el mismo día oficial)",
}


# ── Horas de inicio verificadas ──────────────────────────────────────────

def _descargar_inicios(desde: str, hasta: str) -> List[Dict[str, Any]]:
    """Las horas de inicio del schedule oficial, del rango pedido.

    La API de MLB es gratis y sin clave, así que la hora real es un dato que se
    consigue, no que se estima. Se guarda cruda: `gameDate` es el instante UTC
    de inicio y `officialDate` el día de schedule, y son cosas distintas.
    """
    juegos = mlb_stats.schedule(desde=desde, hasta=hasta, hidratar="")
    return [
        {
            "game_pk": int(g["gamePk"]),
            "game_date": g.get("gameDate"),
            "official_date": g.get("officialDate"),
            "estado": (g.get("status") or {}).get("detailedState"),
        }
        for g in juegos
        if g.get("gamePk")
    ]


def inicios_verificados(
    desde: str, hasta: str, *, cache: Optional[Path] = CACHE_SCHEDULE,
    refrescar: bool = False,
) -> Dict[int, List[Dict[str, Any]]]:
    """`{game_pk: [entradas del schedule]}`, con caché en disco.

    Es una LISTA por `game_pk` y no un valor: un juego pospuesto y rejugado
    aparece en dos bloques de fecha con el mismo `gamePk`, y quedarse con "el
    último que pasó por el diccionario" elegiría la hora de un partido que no
    es el que se cotizó. La desambiguación la hace `resolver_inicio()` con el
    día de la propia cotización.
    """
    crudo: Optional[List[Dict[str, Any]]] = None
    if cache and cache.exists() and not refrescar:
        guardado = json.loads(cache.read_text(encoding="utf-8"))
        if guardado.get("desde") == desde and guardado.get("hasta") == hasta:
            crudo = guardado["juegos"]
            log.info("horas de inicio desde la caché (%s juegos, descargada %s)",
                     len(crudo), guardado.get("descargado_en"))
    if crudo is None:
        log.info("descargando horas de inicio del schedule oficial: %s → %s", desde, hasta)
        crudo = _descargar_inicios(desde, hasta)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({
                "fuente": "https://statsapi.mlb.com/api/v1/schedule",
                "desde": desde, "hasta": hasta,
                "descargado_en": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc).isoformat(),
                "juegos": crudo,
            }, indent=1), encoding="utf-8")

    indice: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for j in crudo:
        indice[int(j["game_pk"])].append(j)
    return dict(indice)


def resolver_inicio(
    game_pk: Optional[int],
    dia_cotizacion: Optional[str],
    indice: Dict[int, List[Dict[str, Any]]],
) -> Tuple[Optional[str], str]:
    """`(instante_utc, motivo)`. El instante es None cuando no se pudo verificar.

    Nunca devuelve una hora aproximada ni completa una fecha con medianoche: si
    la hora real no está, el motivo lo dice y la fila va al informe de
    conciliación.
    """
    entradas = indice.get(int(game_pk)) if game_pk is not None else None
    if not entradas:
        return None, "sin_juego_en_schedule"

    utiles = [e for e in entradas if e.get("game_date") and tiene_hora(e["game_date"])]
    if not utiles:
        return None, "sin_hora_verificada"
    if len(utiles) > 1:
        # Pospuesto y rejugado: el schedule devuelve dos entradas con el mismo
        # `gamePk`. Se desempata por reglas, en orden, y nunca por "el primero"
        # ni "el más cercano":
        #
        #   1. el día que la propia cotización dice estar cotizando, cuando la
        #      reprogramación cayó en otro día oficial;
        #   2. el estado: entre el cascarón `Postponed` y el partido que
        #      terminó, el partido es el que terminó. Medido sobre este
        #      histórico: 89 juegos llegan hasta acá y los 89 se resuelven con
        #      esta regla, porque las dos entradas comparten `officialDate`.
        #
        # Si después de las dos sigue habiendo empate, la fila va al informe de
        # conciliación. Elegir una al azar metería la hora de un partido que no
        # se jugó en la cotización de uno que sí.
        exactas = [e for e in utiles if e.get("official_date") == dia_cotizacion]
        if len(exactas) == 1:
            utiles = exactas
        else:
            candidatas = exactas or utiles
            jugadas = [e for e in candidatas if e.get("estado") in ESTADOS_FINALES]
            if len(jugadas) != 1:
                return None, "fecha_ambigua"
            utiles = jugadas
    return normalizar_utc(utiles[0]["game_date"]), "ok"


# ── Lectura del origen ───────────────────────────────────────────────────

def _filas(origen: Path) -> List[sqlite3.Row]:
    """Las cotizaciones históricas, tal como las guardó el sistema anterior.

    Ya NO se une con `game_outcomes`: esa unión existía sólo para sacar de ahí
    la hora de inicio —que esa tabla no tiene— y de paso perdía las 805 filas
    de `historical_odds` sin juego correspondiente. La hora ahora sale del
    schedule oficial, así que esas filas se recuperan.
    """
    con = sqlite3.connect(f"file:{origen}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return con.execute(
            "SELECT * FROM historical_odds ORDER BY snapshot_ts, game_pk"
        ).fetchall()
    finally:
        con.close()


def _cotizaciones(r: sqlite3.Row) -> List[Tuple[str, str, str, Optional[float], float]]:
    """(libro, mercado, lado, punto, precio) de una fila del histórico.

    El punto va FIRMADO y pegado a su lado, igual que en la captura en vivo:
    el visitante cotiza el complemento del punto del local.
    """
    out: List[Tuple[str, str, str, Optional[float], float]] = []

    def agregar(libro, mercado, lado, punto, precio):
        if libro and precio and precio > 1.0:
            out.append((str(libro), mercado, lado, punto, float(precio)))

    # Moneyline
    agregar("pinnacle", "h2h", "home", None, r["ml_home_pin"])
    agregar("pinnacle", "h2h", "away", None, r["ml_away_pin"])
    agregar(CONSENSO, "h2h", "home", None, r["ml_home_cons"])
    agregar(CONSENSO, "h2h", "away", None, r["ml_away_cons"])
    agregar(r["ml_home_best_bk"], "h2h", "home", None, r["ml_home_best"])
    agregar(r["ml_away_best_bk"], "h2h", "away", None, r["ml_away_best"])

    # Total — el punto es el mismo para los dos lados
    p = r["total_point_pin"]
    agregar("pinnacle", "totals", "over", p, r["total_over_pin"])
    agregar("pinnacle", "totals", "under", p, r["total_under_pin"])
    pb = r["total_point_best"]
    # El histórico no guarda QUÉ casa dio el mejor total, así que entra como
    # agregado derivado y no como precio cotizado por alguien.
    agregar(CONSENSO, "totals", "over", pb, r["total_over_best"])
    agregar(CONSENSO, "totals", "under", pb, r["total_under_best"])

    # Runline — el visitante lleva el complemento
    ph = r["rl_home_point_pin"]
    agregar("pinnacle", "spreads", "home", ph, r["rl_home_pin"])
    agregar("pinnacle", "spreads", "away", -ph if ph is not None else None, r["rl_away_pin"])
    phb = r["rl_home_point_best"]
    agregar(CONSENSO, "spreads", "home", phb, r["rl_home_best"])
    agregar(CONSENSO, "spreads", "away", -phb if phb is not None else None, r["rl_away_best"])

    return out


# ── Idempotencia ─────────────────────────────────────────────────────────

def _llaves_existentes(store: MarketStore, sport_key: str) -> set:
    """Las observaciones que el almacén ya tiene, por su llave histórica.

    `odds_snapshot` es append-only por trigger: un INSERT repetido NO se puede
    deshacer con un DELETE. La única defensa posible es no insertarlo, y para
    eso hay que saber qué hay. La llave es `(captured_at, event_id, book,
    market, side)`: dentro de una misma foto, un libro cotiza un lado una vez.
    """
    with store._conn() as conn:
        return {
            (r[0], r[1], r[2], r[3], r[4])
            for r in conn.execute(
                "SELECT captured_at, event_id, book, market, side "
                "FROM odds_snapshot WHERE sport_key = ?", (sport_key,))
        }


# ── La importación ───────────────────────────────────────────────────────

def importar(
    store: MarketStore,
    *,
    origen: Path = ORIGEN,
    dry_run: bool = False,
    cache_schedule: Optional[Path] = CACHE_SCHEDULE,
    refrescar_schedule: bool = False,
    pendientes: Optional[Path] = PENDIENTES,
    indice: Optional[Dict[int, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """`indice` se pasa ya resuelto sólo en los tests: así la importación se
    prueba sin red y sin depender de que el schedule oficial no cambie."""
    filas = _filas(origen)
    if not filas:
        return {"filas_origen": 0}

    if indice is None:
        dias = [r["game_date"] for r in filas if r["game_date"]]
        indice = inicios_verificados(min(dias), max(dias), cache=cache_schedule,
                                     refrescar=refrescar_schedule)

    existentes = _llaves_existentes(store, DEPORTE)
    lote: List[Tuple] = []
    enlaces: List[Dict[str, Any]] = []
    sin_resolver: List[Dict[str, Any]] = []
    resumen = {
        "filas_origen": len(filas),
        "juegos_resueltos": 0,
        "juegos_pendientes": 0,
        "cotizaciones_nuevas": 0,
        "cotizaciones_ya_estaban": 0,
        "cotizaciones_en_vivo": 0,
        "por_motivo": defaultdict(int),
    }

    for r in filas:
        ev = r["odds_api_id"]
        pk = r["game_pk"]
        dia = r["game_date"]

        motivo = "sin_event_id" if not ev else None
        inicio = None
        if motivo is None:
            inicio, motivo = resolver_inicio(pk, dia, indice)
            motivo = None if motivo == "ok" else motivo

        if motivo is not None:
            resumen["juegos_pendientes"] += 1
            resumen["por_motivo"][motivo] += 1
            sin_resolver.append({
                "game_pk": pk, "dia_cotizacion": dia, "odds_api_id": ev,
                "home_team": r["home_team"], "away_team": r["away_team"],
                "snapshot_ts": r["snapshot_ts"], "motivo": motivo,
                "explicacion": MOTIVOS[motivo],
                # Los candidatos que el schedule ofrecía. Sin esto, conciliar
                # una fila obliga a rehacer la consulta que ya se hizo acá.
                "candidatos_schedule": " | ".join(
                    f"{e.get('game_date')} official={e.get('official_date')} estado={e.get('estado')}"
                    for e in indice.get(int(pk), []) if pk is not None) or "(ninguno)",
            })
            continue

        capturado = normalizar_utc(r["snapshot_ts"])
        resumen["juegos_resueltos"] += 1
        if capturado >= inicio:
            resumen["cotizaciones_en_vivo"] += 1

        enlaces.append({
            "event_id": ev, "sport_key": DEPORTE, "game_pk": pk,
            "official_date": dia, "commence_time": inicio,
            "home_team": r["home_team"], "away_team": r["away_team"],
            "method": "import_historico",
        })

        for libro, mercado, lado, punto, precio in _cotizaciones(r):
            if (capturado, ev, libro, mercado, lado) in existentes:
                resumen["cotizaciones_ya_estaban"] += 1
                continue
            existentes.add((capturado, ev, libro, mercado, lado))
            lote.append((capturado, None, DEPORTE, ev, inicio,
                         r["home_team"], r["away_team"], libro, mercado, lado,
                         punto, precio))

    resumen["cotizaciones_nuevas"] = len(lote)
    resumen["por_motivo"] = dict(resumen["por_motivo"])

    if pendientes is not None:
        pendientes.parent.mkdir(parents=True, exist_ok=True)
        with pendientes.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "game_pk", "dia_cotizacion", "odds_api_id", "home_team",
                "away_team", "snapshot_ts", "motivo", "explicacion",
                "candidatos_schedule"])
            w.writeheader()
            w.writerows(sin_resolver)
        resumen["informe_pendientes"] = str(pendientes)

    if dry_run:
        resumen["dry_run"] = True
        return resumen

    if lote:
        with store._conn() as conn:
            conn.executemany(
                """INSERT INTO odds_snapshot
                   (captured_at, book_update, sport_key, event_id, commence_time,
                    home_team, away_team, book, market, side, point, price_dec)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                lote,
            )
            conn.execute(
                """INSERT INTO sweep (captured_at, sport_key, n_events, n_rows_new,
                                      n_unchanged, n_skipped, note)
                   VALUES (?,?,?,?,?,?,?)""",
                (normalizar_utc(filas[0]["snapshot_ts"]), DEPORTE,
                 resumen["juegos_resueltos"], len(lote),
                 resumen["cotizaciones_ya_estaban"], resumen["juegos_pendientes"],
                 "import histórico del sistema anterior — foto diaria 17:00Z, "
                 "hora de inicio verificada contra el schedule oficial"),
            )
    store.link_events(enlaces)
    return resumen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origen", type=Path, default=ORIGEN)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cache-schedule", type=Path, default=CACHE_SCHEDULE)
    ap.add_argument("--refrescar-schedule", action="store_true",
                    help="vuelve a pedir las horas de inicio al schedule oficial")
    ap.add_argument("--pendientes", type=Path, default=PENDIENTES,
                    help="dónde escribir el informe de filas sin resolver")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    store = MarketStore()
    antes = store.summary()
    r = importar(store, origen=args.origen, dry_run=args.dry_run,
                 cache_schedule=args.cache_schedule,
                 refrescar_schedule=args.refrescar_schedule,
                 pendientes=args.pendientes)
    for k, v in r.items():
        log.info("  %-24s %s", k, v)
    if not args.dry_run:
        despues = store.summary()
        log.info("almacén: %s → %s filas · %s → %s eventos enlazados",
                 antes["rows"], despues["rows"], antes["linked"], despues["linked"])


if __name__ == "__main__":
    main()
