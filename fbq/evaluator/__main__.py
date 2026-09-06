"""CLI del evaluador. Solo lectura.

    python3 -m fbq.evaluator --seasons 2024 2025 --candidato backtest
    python3 -m fbq.evaluator --seasons 2026 --candidato live
    python3 -m fbq.evaluator --seasons 2024 2025 --candidato mercado   # autoprueba
    python3 -m fbq.evaluator --seasons 2024 2025 2026 --almacen propio --candidato mercado

`--almacen` elige de dónde salen los hechos y los precios:

    propio  (default)  `market.db` + `results.db` — sólo almacenes de fbq
    legado             `predictions_history.db` — el sistema anterior

El legado se conserva para poder comparar los dos sobre las mismas claves. Los
candidatos guardados (`--candidato backtest|live`) sólo existen en el legado,
porque son salida de un modelo y el almacén propio no guarda opinión.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fbq.evaluator.frame import (load_candidate, load_frame,
                                 load_frame_propio)
from fbq.evaluator.score import Report, evaluate

# La corrida canónica del backtest. Sin esto se mezclarían tres corridas
# distintas, una de ellas anterior a la remediación de CHRON-001.
RUN_CANONICA = "2026-07-30"


def _imprimir(rep: Report) -> None:
    print(f"\n{'='*66}")
    print(f"  {rep.nombre}   n={rep.n}   tasa local {rep.tasa_base:.4f}")
    print(f"  overround medio de Pinnacle: {100*rep.overround_medio:.2f}%  "
          f"(breakeven por lado: {100*rep.overround_medio/(1+rep.overround_medio):.2f}%)")
    print("=" * 66)

    print("\n  ESCALERA (Brier, menor es mejor)")
    print(f"    moneda 0.5           {0.25:.5f}")
    print(f"    tasa base            {rep.tasa_base*(1-rep.tasa_base):.5f}")
    print(f"    {rep.nombre:<20} {rep.brier_candidato:.5f}   "
          f"(ventaja sobre azar {rep.ventaja_candidato:+.3f}%)")
    print(f"    MERCADO (nulo)       {rep.brier_mercado:.5f}   "
          f"(ventaja sobre azar {rep.ventaja_mercado:+.3f}%)")
    print(f"    brecha               {rep.brecha:+.5f}   "
          f"{'(el candidato es PEOR)' if rep.brecha > 0 else '(el candidato es mejor)'}")
    print(f"    log-loss             {rep.logloss_candidato:.5f} vs {rep.logloss_mercado:.5f}")

    print("\n  REGRESIÓN CONJUNTA  y ~ logit(mercado) + logit(candidato)")
    print(f"    b_mercado    {rep.b_mercado:+.4f}")
    print(f"    b_candidato  {rep.b_candidato:+.4f}   "
          f"IC95 agrupado [{rep.ic_b_candidato[0]:+.4f}, {rep.ic_b_candidato[1]:+.4f}]"
          f"   P(b>0)={100*rep.p_b_candidato_positivo:.1f}%   ({rep.n_bootstrap} remuestreos)")
    veredicto = ("APORTA sobre el precio" if rep.aporta_sobre_el_precio
                 else "NO aporta sobre el precio (el IC95 incluye el cero)")
    print(f"    → {veredicto}")

    print("\n  ROI POR UMBRAL DE EDGE (gate obligatorio — es la cola, no el promedio)")
    print(f"    {'umbral':>7} {'apuestas':>9} {'pnl (u)':>10} {'ROI':>9}")
    for r in rep.roi:
        roi = f"{r['roi_pct']:+.2f}%" if r["roi_pct"] is not None else "  n/a"
        print(f"    {100*r['umbral']:6.0f}% {r['n']:9d} {r['pnl_u']:10.2f} {roi:>9}")

    if rep.mezcla:
        print("\n  MEZCLA FUERA DE MUESTRA (¿le gana algo a v0 = el mercado?)")
        print(f"    {'pliegue':>8} {'n_test':>7} {'v0':>9} {'v0 recal':>9} {'mezcla':>9} {'b_cand':>8}")
        for r in rep.mezcla:
            print(f"    {str(r['pliegue']):>8} {r['n_test']:7d} {r['brier_v0']:9.5f} "
                  f"{r['brier_v0_recalibrado']:9.5f} {r['brier_mezcla']:9.5f} "
                  f"{r['b_candidato_ajustado']:+8.4f}"
                  f"   {'mezcla gana' if r['brier_mezcla'] < r['brier_v0'] else 'v0 SOLO gana'}")

    print("\n  CALIBRACIÓN POR DECIL")
    print(f"    {'n':>5} {'candidato: pred':>16} {'real':>7}   |{'mercado: pred':>16} {'real':>7}")
    for a, b in zip(rep.calib_candidato, rep.calib_mercado):
        print(f"    {a['n']:5d} {a['predicho']:16.4f} {a['real']:7.4f}   |"
              f"{b['predicho']:16.4f} {b['real']:7.4f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025])
    ap.add_argument("--candidato", default="backtest",
                    choices=["backtest", "live", "mercado"],
                    help="'mercado' puntúa el propio nulo: es la autoprueba del instrumento")
    ap.add_argument("--run-date", default=None,
                    help=f"filtra backtest_run_at (default {RUN_CANONICA} para --candidato backtest)")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--cluster-por", default="home_team",
                    choices=["home_team", "away_team", "season"])
    ap.add_argument("--almacen", default="propio", choices=["propio", "legado"],
                    help="de dónde salen hechos y precios (default: propio)")
    ap.add_argument("--incluir-en-vivo", action="store_true",
                    help="NO excluye las cotizaciones posteriores al primer "
                         "lanzamiento. Sólo para comparar contra el legado, que "
                         "las incluía sin saberlo.")
    args = ap.parse_args()

    if args.almacen == "propio":
        frame = load_frame_propio(args.seasons,
                                  solo_pregame=not args.incluir_en_vivo)
    else:
        if args.incluir_en_vivo:
            ap.error("--incluir-en-vivo sólo aplica al almacén propio: el "
                     "legado no sabe a qué hora empezó cada juego")
        frame = load_frame(args.seasons)

    if args.candidato == "mercado":
        rep = evaluate(frame, frame.p_market.copy(),
                       nombre=f"MERCADO (autoprueba, {args.almacen})",
                       n_bootstrap=args.bootstrap, cluster_por=args.cluster_por)
    elif args.almacen == "propio":
        ap.error(f"--candidato {args.candidato} vive en el almacén legado: es "
                 f"salida de un modelo, y el almacén propio no guarda opinión. "
                 f"Usá --almacen legado, o esperá al candidato de fbq/model/.")
    elif args.candidato == "live":
        rep = evaluate(frame, load_candidate("p_home", args.seasons),
                       nombre="MODELO (live)", n_bootstrap=args.bootstrap,
                       cluster_por=args.cluster_por)
    else:
        rep = evaluate(
            frame,
            load_candidate("backtest_p_home", args.seasons,
                           run_date=args.run_date or RUN_CANONICA),
            nombre="MODELO (backtest)", n_bootstrap=args.bootstrap,
            cluster_por=args.cluster_por,
        )
    _imprimir(rep)


if __name__ == "__main__":
    main()
