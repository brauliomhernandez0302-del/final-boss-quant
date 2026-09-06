"""Pasa una feature por el portón, o todas.

    python3 -m fbq.features                       # todas las registradas
    python3 -m fbq.features desacuerdo_cons_pin
"""

import argparse
import sys

from fbq.evaluator.frame import load_frame, load_frame_propio
from fbq.features import evaluar, imprimir, obtener, todas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feature", nargs="?", help="nombre; si se omite, todas")
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--almacen", default="propio", choices=["propio", "legado"],
                    help="de dónde sale el nulo (default: propio)")
    args = ap.parse_args()

    fs = [obtener(args.feature)] if args.feature else todas()
    # Un solo marco para todas: cargarlo por feature costaría una consulta por
    # cada una y, peor, permitiría que dos se evalúen sobre muestras distintas.
    frame = (load_frame_propio(args.seasons) if args.almacen == "propio"
             else load_frame(args.seasons))
    from fbq.evaluator.score import brier
    print(f"\n  nulo: almacén {args.almacen} · n={len(frame)} · "
          f"Brier del mercado {brier(frame.p_market, frame.y):.6f}")
    if args.almacen == "legado":
        print("  ⚠ --almacen legado cambia SÓLO el nulo. Las features leen del "
              "almacén propio\n    desde el 2026-09-06, así que esto NO "
              "reproduce el veredicto v1: para eso\n    hay que correr el "
              "código de su commit (ver `codigo` en cada veredicto).")
    for f in fs:
        r = evaluar(f, frame=frame)
        imprimir(r)
        for v in f.veredictos:
            print(f"  registrado {v.version} ({v.fecha}, {v.almacen}, "
                  f"{v.referencia}, código {v.codigo or '?'}): {v.resultado}")
        if f.veredicto and (f.veredicto == "CRUZA") != r.cruza:
            print(f"  ⚠ el veredicto vigente dice {f.veredicto!r} y la medición "
                  f"de hoy dice {'CRUZA' if r.cruza else 'NO CRUZA'} — "
                  f"algo cambió en los datos o en el código\n", file=sys.stderr)


if __name__ == "__main__":
    main()
