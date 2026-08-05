"""Pasa una feature por el portón, o todas.

    python3 -m fbq.features                       # todas las registradas
    python3 -m fbq.features desacuerdo_cons_pin
"""

import argparse
import sys

from fbq.evaluator.frame import load_frame
from fbq.features import evaluar, imprimir, obtener, todas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feature", nargs="?", help="nombre; si se omite, todas")
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    args = ap.parse_args()

    fs = [obtener(args.feature)] if args.feature else todas()
    # Un solo marco para todas: cargarlo por feature costaría una consulta por
    # cada una y, peor, permitiría que dos se evalúen sobre muestras distintas.
    frame = load_frame(args.seasons)
    for f in fs:
        r = evaluar(f, frame=frame)
        imprimir(r)
        if f.veredicto and (f.veredicto == "CRUZA") != r.cruza:
            print(f"  ⚠ el veredicto registrado dice {f.veredicto!r} y la medición "
                  f"de hoy dice {'CRUZA' if r.cruza else 'NO CRUZA'} — "
                  f"algo cambió en los datos o en el código\n", file=sys.stderr)


if __name__ == "__main__":
    main()
