#!/usr/bin/env python3
"""A9 — `l0` (TTE de ofensa) evaluada como LA PRIMERA FEATURE. Solo lectura.

Éste es el paso 6 del recorrido desde cero hecho como build y no como auditoría.

Por qué `l0` es el equivalente, y no es una elección de gusto: los otros siete
motores son MULTIPLICADORES sobre `l0` (`pitcher`, `bullpen`, `defense`, `park`,
`hfa`, `context`, `bias`). Desde cero no se puede empezar por el ajuste de
abridor, porque no hay λ que ajustar todavía. `l0` es lo único que se sostiene
solo, y además es lo más sofisticado del proyecto: λ neutra de parque desde
xwOBA/barrel%/disciplina de Statcast, con contrato PIT y filtrada por alineación.

La λ se convierte a P(gana el local) con Skellam —diferencia de dos Poisson,
exacta para diferenciales enteros— sin pasar por el simulador, que no hace falta
para moneyline. Los empates de la Skellam representan juegos que irían a extra
innings y se reparten según la fuerza relativa, igual que hace el simulador
desde b3325a5.
"""

import json
import sqlite3
import sys
from pathlib import Path

from scipy.stats import skellam

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluator.frame import load_frame
from evaluator.score import evaluate

DB = Path(__file__).resolve().parents[2] / "data" / "predictions_history.db"
RUN = "2026-07-30"


def main() -> int:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    filas = con.execute(
        """SELECT game_pk, backtest_stage_factors_json FROM game_outcomes
           WHERE source='backtest' AND season IN (2024,2025)
             AND substr(backtest_run_at,1,10)=?
             AND backtest_stage_factors_json IS NOT NULL""",
        (RUN,),
    ).fetchall()
    con.close()

    cand = {}
    for pk, js in filas:
        d = json.loads(js)
        lh, la = d.get("l0_home_lambda"), d.get("l0_away_lambda")
        if not lh or not la:
            continue
        p_gana = float(1 - skellam.cdf(0, lh, la))
        p_emp = float(skellam.pmf(0, lh, la))
        cand[int(pk)] = p_gana + p_emp * (lh / (lh + la))

    rep = evaluate(load_frame([2024, 2025]), cand,
                   nombre="l0 SOLO (TTE ofensa)", n_bootstrap=400)

    print(f"n={rep.n}\n")
    print(f"  Brier l0 solo    {rep.brier_candidato:.5f}  (ventaja {rep.ventaja_candidato:+.3f}%)")
    print(f"  Brier mercado    {rep.brier_mercado:.5f}  (ventaja {rep.ventaja_mercado:+.3f}%)")
    print(f"  brecha           {rep.brecha:+.5f}")
    print(f"\n  b_candidato      {rep.b_candidato:+.4f}   "
          f"IC95 [{rep.ic_b_candidato[0]:+.4f}, {rep.ic_b_candidato[1]:+.4f}]   "
          f"P(b>0)={100*rep.p_b_candidato_positivo:.1f}%")
    if rep.ic_b_candidato[1] < 0:
        print("  → El IC excluye el cero POR ABAJO. No es 'no aporta': condicionado al")
        print("    precio, la predicción de l0 empeora. Donde l0 discrepa del mercado,")
        print("    el mercado tiene razón de forma sistemática — lo que queda de l0 tras")
        print("    descontar lo que el precio ya sabe es, sobre todo, su sesgo.")

    print("\n  ROI por umbral de edge:")
    for r in rep.roi:
        print(f"     {100*r['umbral']:2.0f}%   n={r['n']:5d}   ROI={r['roi_pct']:+.3f}%")

    print("\n  Mezcla fuera de muestra:")
    for r in rep.mezcla:
        gana = "gana la mezcla" if r["brier_mezcla"] < r["brier_v0"] else "gana v0 SOLO"
        print(f"     {r['pliegue']}: v0={r['brier_v0']:.5f}  mezcla={r['brier_mezcla']:.5f}  {gana}")

    print("\n  VEREDICTO: la primera feature no cruza el portón del paso 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
