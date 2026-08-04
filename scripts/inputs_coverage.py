#!/usr/bin/env python3
"""Cobertura de la instantánea de entradas — detecta campos que nunca llegan.

Responde una sola pregunta, repetible y sobre datos reales: **de los picks ya
publicados, ¿qué fracción trae cada campo de `inputs_snapshot` con valor?**

POR QUÉ EXISTE
==============
El 2026-08-02 se descubrió que `weather_source` era `None` en 52 de 52 picks.
No porque el clima faltara —el motor lo recibía y movía λ, verificado en vivo
con `weather_mult` 1.018 sobre Truist Park— sino porque la instantánea leía
`game_data['weather']['source']`, una clave que el fetcher no emite: devuelve
catorce campos y ninguno se llama así.

Esa es la firma de toda una CLASE de fallo, y es visible sin saber nada del
dominio: **un campo en 0% de cobertura mientras sus vecinos están al 100%**.
No hace falta sospechar del clima ni leer el fetcher; basta con mirar la
columna. El análisis estático no lo ve —las claves se escriben con f-strings,
`.update()` y `results[...]`— pero la producción sí.

Un dato bueno con la etiqueta rota es peor que un dato ausente: nadie audita
lo que cree que nunca llegó. Este script hace que eso salte solo.

QUÉ NO HACE
===========
No juzga si un campo DEBE estar. `lineup_confirmed` al 3.8% es la realidad del
mercado —las alineaciones se publican después de que corre el cron— y no un
fallo. Por eso el umbral de alarma es 0%: la ausencia total es sospechosa; la
escasez puede ser honesta. Lee y reporta; ninguna decisión.

NOTA SOBRE LA VENTANA
=====================
El bloque `inputs` no existe en picks anteriores al 2026-08-01 (se añadió en
`127bac6`). Medir sobre todo el histórico daría ~12% en todo por edad del
esquema, no por fallo. Por defecto se acota a los picks que sí lo llevan.

Uso:
    python scripts/inputs_coverage.py
    python scripts/inputs_coverage.py --dias 7
    python scripts/inputs_coverage.py --db /ruta/a/track_record.db
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DB_POR_DEFECTO = RAIZ / "data" / "track_record.db"


def _picks_con_instantanea(db: Path, dias: int | None) -> list[dict]:
    if not db.exists():
        print(f"✗ no existe {db}", file=sys.stderr)
        raise SystemExit(2)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sql = "SELECT game_date, pipeline_json FROM picks WHERE pipeline_json IS NOT NULL"
    params: list = []
    if dias:
        sql += " AND game_date >= date('now', ?)"
        params.append(f"-{dias} day")
    sql += " ORDER BY game_date"
    filas = []
    for r in con.execute(sql, params):
        try:
            d = json.loads(r["pipeline_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if d.get("inputs"):
            filas.append({"fecha": r["game_date"], "inputs": d["inputs"]})
    con.close()
    return filas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path,
                    default=Path(os.environ.get("FBQ_TRACK_DB", DB_POR_DEFECTO)))
    ap.add_argument("--dias", type=int, default=None,
                    help="acotar a los últimos N días (default: todos los que tengan el bloque)")
    args = ap.parse_args()

    picks = _picks_con_instantanea(args.db, args.dias)
    if not picks:
        print("Ningún pick con bloque `inputs`. El bloque existe desde el 2026-08-01 "
              "(commit 127bac6); antes de esa fecha no hay nada que medir.")
        return 0

    campos: list[str] = []
    for p in picks:
        for k in p["inputs"]:
            if k not in campos:
                campos.append(k)

    n = len(picks)
    print(f"Instantánea de entradas — {n} picks  "
          f"({picks[0]['fecha']} a {picks[-1]['fecha']})\n")
    print(f"  {'campo':26s} {'con valor':>11s} {'%':>7s}  distribución")
    print(f"  {'-'*26} {'-'*11} {'-'*7}  {'-'*40}")

    mudos: list[str] = []
    for c in campos:
        vals = [p["inputs"].get(c) for p in picks]
        # False y 0 son valores legítimos: sólo None cuenta como ausencia.
        con_valor = sum(1 for v in vals if v is not None)
        pct = 100.0 * con_valor / n
        if con_valor == 0:
            mudos.append(c)
        distintos = Counter(str(v) for v in vals if v is not None).most_common(3)
        muestra = ", ".join(f"{k}×{v}" for k, v in distintos) if distintos else "—"
        marca = "  ⚠" if con_valor == 0 else ""
        print(f"  {c:26s} {con_valor:5d}/{n:<5d} {pct:6.1f}%  {muestra[:44]}{marca}")

    print()
    if mudos:
        print("⚠ CAMPOS SIEMPRE VACÍOS — revisar si la clave que se lee existe de verdad")
        for c in mudos:
            print(f"    {c}")
        print("\n  Un campo en 0% mientras sus vecinos están altos es la firma del bug de")
        print("  `weather_source` (2026-08-02): la instantánea leía una clave que el")
        print("  productor no emite. Comprobar el nombre contra lo que devuelve el fetcher")
        print("  antes de asumir que el dato falta.")
        return 1

    print("✓ Ningún campo en 0%. Cobertura baja no es lo mismo que ausencia total:")
    print("  `lineup_confirmed` bajo es el mercado (las alineaciones salen después de")
    print("  que corre el cron), no un fallo de plomería.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
