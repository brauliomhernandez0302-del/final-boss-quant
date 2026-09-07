"""Informe pareado v1.4 − v1.2 sobre los partidos ya terminados.

    python3 -m fbq.model.informe_pareado

**Descriptivo, no veredicto.** El umbral de decisión está fijado en
`docs/PROSPECTIVA_ACTIVADA_2026-09-06.md` §6: n ≥ 900 pares con resultado, que es
la potencia del 80 % para un ΔBrier de 0,001. Mirar la diferencia todos los días
y decidir el día que da positivo es cómo se fabrica un hallazgo.

Las tres poblaciones van **separadas y nunca se suman**:

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


def informe(*, db_pred: Path = DB_PRED, db_res: Path = DB_RESULTADOS) -> Dict[str, Any]:
    y_real = _resultados(db_res)
    con = sqlite3.connect(f"file:{db_pred}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        filas = con.execute(
            "SELECT game_pk, corte, version, p_home, cohorte, origen, modelo_sha, "
            "commence_time FROM prediccion").fetchall()
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
    pendientes: Dict[str, int] = defaultdict(int)
    for (pk, corte, cohorte), v in por_llave.items():
        if "v1.2" not in v or "v1.4" not in v:
            pendientes["sin_las_dos_versiones"] += 1
            continue
        origen = min((v["v1.2"]["origen"], v["v1.4"]["origen"]), key=lambda o: FUERZA.get(o, 0))
        if pk not in y_real:
            pendientes[f"{origen}:sin_resultado_todavia"] += 1
            continue
        y = y_real[pk]
        grupos[origen].append({
            "game_pk": pk, "corte": corte, "cohorte": cohorte, "y": y,
            "p12": v["v1.2"]["p_home"], "p14": v["v1.4"]["p_home"],
            "b12": (v["v1.2"]["p_home"] - y) ** 2,
            "b14": (v["v1.4"]["p_home"] - y) ** 2,
            "sha_v14": v["v1.4"]["modelo_sha"]})

    salida: Dict[str, Any] = {
        "advertencia": ("DESCRIPTIVO, no veredicto. Umbral de decisión: n ≥ 900 "
                        "pares con resultado (potencia 80% para ΔBrier=0,001)."),
        "poblaciones": {}, "pendientes": dict(pendientes)}
    for origen, xs in sorted(grupos.items()):
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
        salida["poblaciones"][origen] = {
            "pares": n,
            "brier_v1_2": b12, "brier_v1_4": b14,
            "diferencia_media_v14_menos_v12": media,
            "error_estandar": se,
            "ic95": [media - 1.96 * se, media + 1.96 * se] if n > 1 else None,
            "shas_v1_4": sorted({x["sha_v14"] for x in xs}),
            "juegos_donde_v1_4_mejora": sum(1 for x in d if x < 0),
            "potencia_alcanzada": f"{n}/900 del umbral de decisión",
        }
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
