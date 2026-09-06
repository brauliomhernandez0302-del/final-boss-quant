"""Cuántos partidos tienen abridor conocido ANTES del corte de su cuota.

Es la pregunta que decide si el motor de abridores se puede evaluar. No es
"cuántos anuncios tenemos" —eso es fácil— sino cuántos estaban **observados
antes del instante en que se observó el precio de referencia**, que es el único
corte que vale.

    python3 -m fbq.anuncios.cobertura
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from fbq.anuncios.store import AnunciosStore
from fbq.evaluator.frame import DB_MERCADO, DB_RESULTADOS
from fbq.model.candidato import _precios_de_referencia


def _temporada_por_juego(db_resultados: Path, game_pks: Sequence[int]) -> Dict[int, int]:
    if not game_pks:
        return {}
    marcas = ",".join("?" * len(game_pks))
    con = sqlite3.connect(f"file:{db_resultados}?mode=ro", uri=True)
    try:
        return {int(pk): int(s) for pk, s in con.execute(
            f"SELECT game_pk, season FROM resultado WHERE game_pk IN ({marcas})",
            tuple(int(g) for g in game_pks))}
    finally:
        con.close()


def informe(
    *,
    seasons: Sequence[int] = (2024, 2025, 2026),
    store: Optional[AnunciosStore] = None,
    db_mercado: Path = DB_MERCADO,
    db_resultados: Path = DB_RESULTADOS,
    precios: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Cobertura por temporada, contra el marco evaluable de la v1.2."""
    store = store or AnunciosStore()
    precios = precios if precios is not None else _precios_de_referencia(
        seasons, db_mercado=db_mercado, db_resultados=db_resultados)

    with store._conn() as conn:
        filas = conn.execute(
            "SELECT game_pk, lado, pitcher_id, observado_en, commence_time "
            "FROM anuncio ORDER BY observado_en, id").fetchall()

    por_juego: Dict[int, Dict[str, List[sqlite3.Row]]] = defaultdict(
        lambda: {"home": [], "away": []})
    for r in filas:
        por_juego[int(r["game_pk"])][r["lado"]].append(r)

    temporadas = _temporada_por_juego(db_resultados, list(por_juego))
    marco_por_temp = Counter(temporadas.get(pk) for pk in precios
                             if temporadas.get(pk) is not None)
    # `precios` cubre sólo juegos con resultado; para el marco completo por
    # temporada hace falta consultar aparte.
    con = sqlite3.connect(f"file:{db_resultados}?mode=ro", uri=True)
    try:
        marco = Counter()
        for pk in precios:
            s = con.execute("SELECT season FROM resultado WHERE game_pk=?", (pk,)).fetchone()
            if s:
                marco[int(s[0])] += 1
    finally:
        con.close()

    por_temp: Dict[Any, Counter] = defaultdict(Counter)
    motivos: Counter = Counter()
    detalle: List[Dict[str, Any]] = []

    for pk, lados in por_juego.items():
        temp = temporadas.get(pk, "sin_resultado")
        por_temp[temp]["registrados"] += 1
        corte = (precios.get(pk) or {}).get("corte")
        if corte is None:
            por_temp[temp]["sin_precio_de_referencia"] += 1
            motivos["sin_precio_de_referencia"] += 1
            continue
        vig = {lado: store.vigente_antes(pk, lado, corte) for lado in ("home", "away")}
        if any(v is None for v in vig.values()):
            por_temp[temp]["sin_observacion_previa_al_corte"] += 1
            motivos["sin_observacion_previa_al_corte"] += 1
            continue
        if any(v["pitcher_id"] is None for v in vig.values()):
            por_temp[temp]["anunciado_pero_vacio"] += 1
            motivos["anunciado_pero_vacio"] += 1
            continue
        por_temp[temp]["ambos_conocidos_antes_del_corte"] += 1
        detalle.append({
            "game_pk": pk, "season": temp, "corte": corte,
            "home_pitcher_id": vig["home"]["pitcher_id"],
            "away_pitcher_id": vig["away"]["pitcher_id"],
            "observado_home": vig["home"]["observado_en"],
            "observado_away": vig["away"]["observado_en"],
        })

    return {
        "fuentes": [
            "track_record.pipeline_json (sistema anterior, commit 127bac6) — "
            "`published_at` es el instante real de observación",
            "fbq.anuncios.capturar — schedule de MLB con hydrate=probablePitcher, "
            "`observado_en` sellado al momento de la barrida",
        ],
        "almacen": store.resumen(),
        "marco_evaluable_por_temporada": dict(sorted(marco.items())),
        "por_temporada": {str(k): dict(v) for k, v in sorted(
            por_temp.items(), key=lambda kv: str(kv[0]))},
        "exclusiones": dict(motivos),
        "utilizables": len(detalle),
        "detalle": detalle,
    }


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
    r_corto = dict(r); r_corto["detalle"] = f"{len(r['detalle'])} juegos"
    print(json.dumps(r_corto, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
