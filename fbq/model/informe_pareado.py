"""Informe pareado v1.4 − v1.2 sobre los partidos ya terminados.

    python3 -m fbq.model.informe_pareado

**Descriptivo, no veredicto.** El umbral de decisión está fijado en
`docs/PROSPECTIVA_ACTIVADA_2026-09-06.md` §6: n ≥ 900 pares con resultado, que es
la potencia del 80 % para un ΔBrier de 0,001. Mirar la diferencia todos los días
y decidir el día que da positivo es cómo se fabrica un hallazgo.

## La regla de selección, declarada antes de conocer resultados

`docs/REGLA_SELECCION_PAREJA_2026-09-07.md`, commiteada con **0 partidos
evaluados**:

> **Para el informe principal se usa el PRIMER par completo y verificable de
> cada partido.**

Un partido con tres emisiones aporta **un** par, no tres. Contar las tres haría
caer el error estándar por √3 sin una sola observación nueva. Las emisiones
posteriores se conservan y se reportan aparte, en `emisiones_posteriores`.

Las poblaciones van **separadas y nunca se suman**:

| población | qué es |
|---|---|
| `prospectiva_verificada` | escrita ANTES del primer lanzamiento, con `registrado_utc` que lo demuestra |
| `reconstruccion` | calculada después, sobre un corte del pasado. Sirve para desarrollar, no para acreditar |
| `no_verificable` | su evidencia de emisión original no existe |
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fbq.evaluator.frame import DB_RESULTADOS
from fbq.model.prospectiva import DB_PATH as DB_PRED


def _shas_vigentes() -> Dict[str, str]:
    """El ajuste VIGENTE de cada versión. Un par tiene que salir de un solo
    ajuste por versión; mezclar dos shas compararía dos modelos distintos."""
    from fbq.model.prospectiva import CONGELADO
    if not CONGELADO.exists():
        return {}
    d = json.loads(CONGELADO.read_text(encoding="utf-8"))
    return {v: m["sha"] for v, m in d["modelos"].items()}


def _resultados(db: Path = DB_RESULTADOS) -> Dict[int, int]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {int(pk): int(y) for pk, y in con.execute(
            "SELECT game_pk, home_won FROM resultado")}
    finally:
        con.close()


DB_ANUNCIOS = Path(__file__).parent.parent.parent / "data" / "anuncios.db"
DB_RELEVISTAS = Path(__file__).parent.parent.parent / "data" / "relevistas.db"


def _contexto(pks: List[int], db_res: Path = DB_RESULTADOS) -> Dict[int, Dict[str, Any]]:
    """Marcador y equipos de cada partido. Sin esto la tabla no dice qué pasó."""
    if not pks:
        return {}
    con = sqlite3.connect(f"file:{db_res}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        # Se pide sólo lo que la tabla tiene. Un almacén más viejo o más
        # pequeño deja la columna en None y la tabla lo dice; lo que no puede
        # es tumbar el informe entero, que es la parte que sí acredita.
        hay = {r[1] for r in con.execute("PRAGMA table_info(resultado)")}
        quiero = ["game_pk", "official_date", "home_team", "away_team",
                  "home_runs", "away_runs", "home_won"]
        cols = [c for c in quiero if c in hay]
        marcas = ",".join("?" * len(pks))
        out = {}
        for r in con.execute(f"SELECT {', '.join(cols)} FROM resultado "
                             f"WHERE game_pk IN ({marcas})", pks):
            fila = {c: None for c in quiero}
            fila.update(dict(r))
            out[int(fila["game_pk"])] = fila
        return out
    finally:
        con.close()


def _abridores_reales(pks: List[int], db: Path = DB_RELEVISTAS) -> Dict[tuple, int]:
    """`{(game_pk, lado): pitcher_id}` de quien ABRIÓ de verdad.

    Sale de `data/relevistas.db`, que aplica la regla de rol ya verificada: el
    primero de la lista QUE LANZÓ. No es el probable del calendario — anunciar
    y abrir son cosas distintas, y ésa es justamente la pregunta.
    """
    if not pks or not Path(db).exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        marcas = ",".join("?" * len(pks))
        return {(int(pk), "home" if int(loc) else "away"): int(pid)
                for pk, loc, pid in con.execute(
                    f"""SELECT game_pk, es_local, pitcher_id FROM aparicion
                        WHERE rol='abridor' AND game_pk IN ({marcas})""", pks)}
    finally:
        con.close()


def _abridores_anunciados(pares: List[tuple], db: Path = DB_ANUNCIOS) -> Dict[tuple, Any]:
    """`{(game_pk, lado): fila}` del ÚLTIMO anuncio anterior al corte del par.

    `vigente_antes` es la única lectura permitida: devuelve lo que se sabía en
    ese instante, no lo que terminó pasando.
    """
    if not pares or not Path(db).exists():
        return {}
    from fbq.anuncios.store import AnunciosStore
    st = AnunciosStore(db)
    out = {}
    for pk, corte in pares:
        for lado in ("home", "away"):
            out[(int(pk), lado)] = st.vigente_antes(int(pk), lado, corte)
    return out


def _cotejo_abridores(pk: int, corte: str, anunciados, reales) -> Dict[str, Any]:
    """Por lado: qué se anunció antes del corte y quién abrió.

    Tres respuestas posibles, y ninguna se disfraza de otra: `coincidio`,
    `cambio`, o `sin_anuncio_al_corte` —que es información, no un hueco que
    haya que rellenar—. Si falta el dato del abridor real se dice `sin_dato`.
    """
    salida: Dict[str, Any] = {}
    for lado in ("home", "away"):
        a = anunciados.get((pk, lado))
        real = reales.get((pk, lado))
        anunciado = int(a["pitcher_id"]) if a and a["pitcher_id"] is not None else None
        if real is None:
            estado = "sin_dato"
        elif anunciado is None:
            estado = "sin_anuncio_al_corte"
        else:
            estado = "coincidio" if anunciado == real else "cambio"
        salida[lado] = {
            "anunciado_id": anunciado,
            "anunciado_nombre": (a["pitcher_nombre"] if a else None),
            "anunciado_observado_en": (a["observado_en"] if a else None),
            "abridor_real_id": real,
            "coincide": estado,
        }
    return salida


def informe(*, db_pred: Path = DB_PRED, db_res: Path = DB_RESULTADOS) -> Dict[str, Any]:
    # El instante EXACTO en que se leyó el estado. Sin él, «9 partidos» no se
    # puede reproducir ni contrastar: la muestra crece sola cada hora, y dos
    # lecturas del mismo informe con distinto número no son una contradicción
    # sino dos cortes distintos.
    generado_utc = datetime.now(timezone.utc).isoformat()
    y_real = _resultados(db_res)
    con = sqlite3.connect(f"file:{db_pred}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        # `calidad_datos` es append-only y SIN llave única a propósito: una
        # anotación equivocada se corrige agregando otra fila, así que vale la
        # ÚLTIMA (`MAX(id)`). Una base anterior a la tabla no es un error: se
        # reporta `sin_anotar`, que es la verdad — nadie miró esa calidad.
        hay_calidad = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='calidad_datos'"
        ).fetchone() is not None
        calidad_sql = ("""COALESCE(q.calidad, 'sin_anotar') AS calidad
               FROM prediccion p
               LEFT JOIN calidad_datos q
                 ON q.id = (SELECT MAX(id) FROM calidad_datos d
                            WHERE d.game_pk=p.game_pk AND d.version=p.version
                              AND d.corte=p.corte AND d.modelo_sha=p.modelo_sha)"""
                       if hay_calidad else "'sin_anotar' AS calidad FROM prediccion p")
        filas = con.execute(
            """SELECT p.game_pk, p.corte, p.version, p.p_home, p.cohorte, p.origen,
                      p.modelo_sha, p.commence_time, p.official_date,
                      """ + calidad_sql).fetchall()
    finally:
        con.close()

    # Un par es (juego, corte, cohorte) con las DOS versiones, tomando de cada
    # una el ajuste VIGENTE. Y su fuerza es la de su pierna MÁS DÉBIL: de nada
    # sirve una v1.4 escrita antes del partido si su v1.2 no lo está.
    vigentes = _shas_vigentes()
    FUERZA = {"no_verificable": 0, "reconstruccion": 1, "prospectiva_verificada": 2}
    por_llave: Dict[tuple, Dict[str, Any]] = defaultdict(dict)
    for r in filas:
        if vigentes and r["modelo_sha"] != vigentes.get(r["version"]):
            continue
        llave = (r["game_pk"], r["corte"], r["cohorte"])
        previo = por_llave[llave].get(r["version"])
        if previo is None or FUERZA.get(r["origen"], 0) > FUERZA.get(previo["origen"], 0):
            por_llave[llave][r["version"]] = r

    grupos: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    grupos_posteriores: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pendientes: Dict[str, int] = defaultdict(int)
    # ── La regla: UN par por partido, el primero completo y verificable ──
    #
    # Se aplica ANTES de mirar resultados y sin consultarlos: el criterio es el
    # `corte` más temprano, con el `id` como desempate. Un partido no puede
    # aportar más de un par al informe principal.
    completos: Dict[tuple, tuple] = {}
    for llave, v in sorted(por_llave.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if "v1.2" in v and "v1.4" in v:
            completos.setdefault((llave[0], llave[2]), (llave, v))
    elegidos = {llave for llave, _ in completos.values()}

    # Un CORTE rechazado no es un PARTIDO excluido.
    #
    # Se contaban juntos y eso exageraba el daño: de los cortes que no forman
    # par bajo el ajuste vigente, la mayoría pertenece a partidos que sí
    # aportaron un par en otro corte —la regla del primer par completo
    # funcionando— y no perdieron nada. Sólo son partidos excluidos los que no
    # tienen NINGÚN par válido.
    rechazados: List[Dict[str, Any]] = []
    sin_resultado: List[Dict[str, Any]] = []
    con_par = {pk for (pk, _) in completos}
    for (pk, corte, cohorte), v in por_llave.items():
        if "v1.2" not in v or "v1.4" not in v:
            presentes = sorted(v)
            rechazados.append({
                "game_pk": pk, "corte": corte, "cohorte": cohorte,
                "versiones_presentes": presentes,
                "falta": [x for x in ("v1.2", "v1.4") if x not in presentes],
                "el_partido_tiene_otro_par_valido": pk in con_par,
                "motivo": "sin_las_dos_versiones_bajo_el_ajuste_vigente"})
            continue
        origen = min((v["v1.2"]["origen"], v["v1.4"]["origen"]), key=lambda o: FUERZA.get(o, 0))
        # La CALIDAD es una dimensión aparte de la condición prospectiva: una
        # emisión puede ser impecablemente prospectiva y haber usado entradas
        # incompletas. Se reportan cruzadas, no fundidas.
        calidad = ("historial_incompleto"
                   if "historial_incompleto" in (v["v1.2"]["calidad"], v["v1.4"]["calidad"])
                   else v["v1.2"]["calidad"])
        clave_grupo = f"{origen}|{calidad}"
        if (pk, corte, cohorte) not in elegidos:
            # Emisión posterior del mismo partido: se conserva, no se cuenta.
            grupos_posteriores[clave_grupo].append({
                "game_pk": pk, "corte": corte,
                "y": y_real.get(pk),
                "p12": v["v1.2"]["p_home"], "p14": v["v1.4"]["p_home"]})
            continue
        if pk not in y_real:
            pendientes[f"{clave_grupo}:sin_resultado_todavia"] += 1
            sin_resultado.append({"game_pk": pk, "corte": corte,
                                  "official_date": v["v1.2"]["official_date"],
                                  "commence_time": v["v1.2"]["commence_time"],
                                  "poblacion": clave_grupo})
            continue
        y = y_real[pk]
        grupos[clave_grupo].append({
            "game_pk": pk, "corte": corte, "cohorte": cohorte, "y": y,
            "p12": v["v1.2"]["p_home"], "p14": v["v1.4"]["p_home"],
            "b12": (v["v1.2"]["p_home"] - y) ** 2,
            "b14": (v["v1.4"]["p_home"] - y) ** 2,
            "sha_v14": v["v1.4"]["modelo_sha"]})

    # ── El desglose por partido ──────────────────────────────────────
    #
    # Se publica en CADA corrida, no sólo cuando alguien lo pide a mano: el
    # objetivo declarado de este tramo es ver qué se pronosticó y qué ocurrió,
    # y un agregado de 9 pares no lo muestra. Los abridores se cotejan acá
    # porque un par cuyo abridor cambió entre el corte y el primer lanzamiento
    # se evaluó sobre una entrada que dejó de ser cierta — y eso hay que poder
    # verlo partido por partido, no deducirlo.
    todos = [x for xs in grupos.values() for x in xs]
    pks = [x["game_pk"] for x in todos]
    ctx = _contexto(pks, db_res)
    reales = _abridores_reales(pks)
    anunciados = _abridores_anunciados([(x["game_pk"], x["corte"]) for x in todos])
    for clave, xs in grupos.items():
        for x in xs:
            c = ctx.get(x["game_pk"], {})
            x["official_date"] = c.get("official_date")
            x["home_team"] = c.get("home_team")
            x["away_team"] = c.get("away_team")
            x["marcador"] = (None if c.get("home_runs") is None
                             else f"{c['home_runs']}-{c['away_runs']}")
            x["gano"] = (None if c.get("home_won") is None
                         else ("local" if c["home_won"] else "visita"))
            x["delta_brier"] = x["b14"] - x["b12"]
            x["poblacion"] = clave
            x["abridores"] = _cotejo_abridores(x["game_pk"], x["corte"],
                                               anunciados, reales)

    por_fecha: Dict[str, int] = defaultdict(int)
    for x in sin_resultado:
        por_fecha[x["official_date"] or "sin_fecha"] += 1

    salida: Dict[str, Any] = {
        "generado_utc": generado_utc,
        "advertencia": ("DESCRIPTIVO, no veredicto. Umbral de decisión: n ≥ 900 "
                        "pares con resultado (potencia 80% para ΔBrier=0,001)."),
        "regla_de_seleccion": (
            "primer par completo y verificable de cada partido — "
            "docs/REGLA_SELECCION_PAREJA_2026-09-07.md, declarada con 0 evaluados"),
        "poblaciones": {},
        "cortes_rechazados": {
            "nota": ("un CORTE rechazado no es un PARTIDO excluido: sólo lo es "
                     "el que no tiene ningún par válido en ningún corte"),
            "cortes": len(rechazados),
            "partidos_afectados": len({r["game_pk"] for r in rechazados}),
            "partidos_con_otro_par_valido": len(
                {r["game_pk"] for r in rechazados
                 if r["el_partido_tiene_otro_par_valido"]}),
            "partidos_excluidos_definitivamente": len(
                {r["game_pk"] for r in rechazados
                 if not r["el_partido_tiene_otro_par_valido"]}),
            "detalle": sorted(rechazados, key=lambda r: (r["corte"], r["game_pk"])),
        },
        "desglose_por_partido": sorted(
            (x for x in todos),
            key=lambda x: (x["poblacion"], x["official_date"] or "", x["game_pk"])),
        # `pendientes` queda sólo para lo que todavía puede resolverse solo:
        # partidos emitidos que aún no terminaron. Los cortes rechazados viven
        # en su propia sección porque no son lo mismo ni se arreglan esperando.
        "pendientes": dict(pendientes),
        "pendientes_por_fecha": dict(sorted(por_fecha.items())),
        "pendientes_detalle": sorted(
            sin_resultado, key=lambda x: (x["official_date"] or "", x["game_pk"])),
        "emisiones_posteriores": {
            origen: {
                "n": len(xs),
                "partidos": len({x["game_pk"] for x in xs}),
                "con_resultado": sum(1 for x in xs if x["y"] is not None),
                "nota": "conservadas y NO sumadas al informe principal",
            } for origen, xs in sorted(grupos_posteriores.items())},
    }
    for clave, xs in sorted(grupos.items()):
        n = len(xs)
        b12 = sum(x["b12"] for x in xs) / n
        b14 = sum(x["b14"] for x in xs) / n
        d = [x["b14"] - x["b12"] for x in xs]
        media = sum(d) / n
        if n > 1:
            var = sum((x - media) ** 2 for x in d) / (n - 1)
            se = math.sqrt(var / n)
        else:
            se = float("nan")
        salida["poblaciones"][clave] = {
            "pares": n,
            "brier_v1_2": b12, "brier_v1_4": b14,
            "diferencia_media_v14_menos_v12": media,
            "error_estandar": se,
            "ic95": [media - 1.96 * se, media + 1.96 * se] if n > 1 else None,
            "shas_v1_4": sorted({x["sha_v14"] for x in xs}),
            "origen": clave.split("|")[0], "calidad_datos": clave.split("|")[1],
            "juegos_donde_v1_4_mejora": sum(1 for x in d if x < 0),
            "potencia_alcanzada": f"{n}/900 del umbral de decisión",
            "partidos_unicos": len({x["game_pk"] for x in xs}),
            "juegos_donde_v1_4_empeora": sum(1 for x in d if x > 0),
            "juegos_sin_diferencia": sum(1 for x in d if x == 0),
            "perdida_media_v1_2": b12, "perdida_media_v1_4": b14,
            "abridores": {
                estado: sum(1 for x in xs for lado in ("home", "away")
                            if x["abridores"][lado]["coincide"] == estado)
                for estado in ("coincidio", "cambio", "sin_anuncio_al_corte",
                               "sin_dato")},
        }
        assert salida["poblaciones"][clave]["partidos_unicos"] == n, (
            "un partido no puede aportar más de un par al informe principal")
    return salida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", type=Path, default=None)
    args = ap.parse_args()
    r = informe()
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(json.dumps(r, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
