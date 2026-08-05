#!/usr/bin/env python3
"""A8 — ¿algún motor aporta sobre el precio FUERA DE MUESTRA? Solo lectura.

`a7_alfa_por_motor.py` ajusta y evalúa sobre los mismos datos. En muestra, un
coeficiente positivo con IC que no cruza cero es condición NECESARIA para que un
motor sirva, pero no suficiente: con ocho motores y dos temporadas, encontrar uno
que dé positivo es casi lo esperable por azar.

Éste es el paso 6 del recorrido desde cero: una feature entra al sistema sólo si
se gana su coeficiente contra el precio en datos que no vio. Se ajusta
`y ~ logit(Pinnacle) + motor` en una temporada y se aplica a la otra, comparando
el Brier resultante contra el de Pinnacle SOLO en esa misma temporada de prueba.

La estandarización del motor se calcula con la media y desvío del pliegue de
ENTRENAMIENTO únicamente. Estandarizar con la muestra completa mete el pliegue de
prueba en el ajuste por la puerta de al lado — es una fuga chica pero real, y en
señales de este tamaño (coeficientes de 0.05) importa.
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluator.score import _fit_logistica, brier

DB = Path(__file__).resolve().parents[2] / "data" / "predictions_history.db"
RUN = "2026-07-30"

SPEC = {
    "l0 (TTE ofensa)": ("l0_home_lambda", "l0_away_lambda"),
    "pitcher":         ("pitcher_on_home_lambda", "pitcher_on_away_lambda"),
    "bias (equipo)":   ("bias_on_home_lambda", "bias_on_away_lambda"),
    "bullpen":         ("bullpen_on_home_lambda", "bullpen_on_away_lambda"),
    "defense":         ("defense_on_home_lambda", "defense_on_away_lambda"),
    "context":         ("context_on_home_lambda", "context_on_away_lambda"),
    "hfa":             ("hfa_on_home_lambda", "hfa_on_away_lambda"),
    # `park` se omite: mueve idéntico ambos lados, así que su aporte al
    # moneyline es CERO por construcción (a7 ya lo señalaba).
}


def main() -> int:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    filas = con.execute(
        """SELECT season, backtest_stage_factors_json, ml_home_pin, ml_away_pin, home_won
           FROM game_outcomes
           WHERE source='backtest' AND season IN (2024,2025)
             AND substr(backtest_run_at,1,10)=?
             AND ml_home_pin IS NOT NULL AND backtest_stage_factors_json IS NOT NULL""",
        (RUN,),
    ).fetchall()
    con.close()

    season = np.array([r[0] for r in filas])
    d = [json.loads(r[1]) for r in filas]
    oh = np.array([r[2] for r in filas], float)
    oa = np.array([r[3] for r in filas], float)
    y = np.array([r[4] for r in filas], float)
    pin = (1 / oh) / ((1 / oh) + (1 / oa))
    Lp = np.log(pin / (1 - pin))

    print(f"n={len(filas)}  (2024: {(season==2024).sum()}, 2025: {(season==2025).sum()})")
    print("\nAjusta en una temporada, evalúa en la otra. `mejora` = Brier(Pinnacle) − Brier(mezcla)")
    print("en la temporada de PRUEBA: positivo = el motor ayudó donde no había visto los datos.\n")
    print(f"  {'motor':17} {'entrena→prueba':>16} {'Pinnacle':>9} {'con motor':>10} "
          f"{'mejora':>9} {'coef':>8}")
    print(f"  {'-'*17} {'-'*16} {'-'*9} {'-'*10} {'-'*9} {'-'*8}")

    resumen = {}
    for nombre, (kh, ka) in SPEC.items():
        fh = np.array([float(x.get(kh, 1.0) or 1.0) for x in d])
        fa = np.array([float(x.get(ka, 1.0) or 1.0) for x in d])
        senal_cruda = np.log(np.clip(fh, 1e-6, None) / np.clip(fa, 1e-6, None))
        if np.allclose(senal_cruda, senal_cruda[0]):
            print(f"  {nombre:17} {'(constante — sin variación, no evaluable)':>60}")
            continue

        mejoras = []
        for prueba in (2024, 2025):
            te = season == prueba
            tr = ~te
            # Estandarizar con estadísticos del ENTRENAMIENTO únicamente.
            mu, sd = senal_cruda[tr].mean(), senal_cruda[tr].std()
            if sd == 0:
                continue
            s = (senal_cruda - mu) / sd
            b = _fit_logistica(np.column_stack([Lp[tr], s[tr]]), y[tr])
            p_mix = 1 / (1 + np.exp(-(b[0] + b[1] * Lp[te] + b[2] * s[te])))
            b_pin = brier(pin[te], y[te])
            b_mix = brier(p_mix, y[te])
            mejoras.append(b_pin - b_mix)
            print(f"  {nombre:17} {f'{2024+2025-prueba}→{prueba}':>16} "
                  f"{b_pin:9.5f} {b_mix:10.5f} {b_pin-b_mix:+9.5f} {b[2]:+8.4f}")
        resumen[nombre] = mejoras

    print("\n  VEREDICTO (un motor sirve sólo si ayuda en LAS DOS direcciones):")
    alguno = False
    for nombre, m in resumen.items():
        if len(m) == 2 and all(x > 0 for x in m):
            print(f"    ✓ {nombre}: +{m[0]:.5f} y +{m[1]:.5f}")
            alguno = True
    if not alguno:
        print("    Ninguno. Ningún motor mejora a Pinnacle en las dos direcciones del corte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
