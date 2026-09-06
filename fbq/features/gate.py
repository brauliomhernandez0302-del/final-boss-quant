"""fbq/features/gate.py — el portón. Una feature entra sólo si lo cruza.

Tres criterios, y hay que pasar los TRES. Cada uno existe porque una versión
anterior del proyecto se equivocó midiendo con los otros dos:

1. **Signo estable fuera de muestra.** Se ajusta en unas temporadas y se aplica
   a otra. En muestra, con siete señales y dos temporadas, encontrar una
   positiva es casi lo esperable por azar: el único motor que salía
   significativo en muestra (`defense`, P(b>0)=99.0%) resultó el PEOR de todos
   al probarlo fuera, con el coeficiente saltando de +0.1192 a +0.0218.

2. **Brier que mejora sobre v0.** Agregar un regresor ruidoso a un predictor
   casi óptimo cuesta Brier por error de estimación; si la señal no paga ese
   costo, no se paga a sí misma.

3. **ROI que no empeora al subir el umbral.** Es la cola, que es donde se
   apuesta. Brier y ROI pueden moverse en direcciones opuestas, y un ROI que
   empeora monótonamente cuando el edge declarado sube es la firma de una
   señal anti-predictiva — se vio exactamente así en el pipeline anterior
   (−0.42%, +0.21%, −2.87%, −3.27%, −4.04%).

El portón no borra nada. Una feature que no cruza se queda registrada con su
veredicto, para que nadie la reintente sin saber que ya se midió.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from fbq.evaluator.frame import EvalFrame, load_frame, load_frame_propio
from fbq.evaluator.score import _fit_logistica, _logit, brier
from fbq.features.base import Feature

UMBRALES = (0.005, 0.01, 0.02, 0.03)


@dataclass
class Resultado:
    feature: str
    n: int
    pliegues: List[dict]

    @property
    def signo_estable(self) -> bool:
        coefs = [p["coef"] for p in self.pliegues]
        return len(coefs) > 1 and (all(c > 0 for c in coefs) or all(c < 0 for c in coefs))

    @property
    def mejora_brier(self) -> bool:
        return bool(self.pliegues) and all(p["mejora"] > 0 for p in self.pliegues)

    # Mínimo de apuestas para que un ROI signifique algo. A n=500 el error
    # estándar del ROI ronda el 4.5% con cuotas cercanas a 2.0, o sea que un
    # +2% no distingue una señal de nada. Fijarlo alto es lo que impide que el
    # portón apruebe ruido por muestreo — se comprobó: con 1.500 juegos
    # sintéticos, una señal de ruido puro cruzaba.
    MIN_APUESTAS = 500

    @property
    def roi_no_empeora(self) -> bool:
        """Sobre los pliegues con muestra suficiente. Un ROI calculado sobre 13
        apuestas no es evidencia ni a favor ni en contra."""
        util = [p for p in self.pliegues
                if p["roi"] and p["roi"][0]["n"] >= self.MIN_APUESTAS]
        if not util:
            return False
        return all(r["roi_pct"] is not None and r["roi_pct"] > 0
                   for p in util for r in p["roi"] if r["n"] >= self.MIN_APUESTAS)

    @property
    def medible(self) -> bool:
        """¿Hubo pliegues con muestra suficiente para decir algo?

        "No cruza" y "no se pudo medir" son cosas distintas y confundirlas es
        cómo una feature queda archivada como refutada sin que nadie la haya
        refutado. `profundidad_mercado` cae acá desde que las features leen del
        almacén propio: `n_bookmakers` no existe ahí, y contar los libros
        guardados del histórico daría un artefacto de qué conservó el sistema
        anterior, no la profundidad del mercado.
        """
        return len(self.pliegues) > 1

    @property
    def cruza(self) -> bool:
        return (self.medible and self.signo_estable and self.mejora_brier
                and self.roi_no_empeora)


def evaluar(
    feature: Feature,
    *,
    seasons: Sequence[int] = (2024, 2025, 2026),
    frame: EvalFrame | None = None,
    almacen: str = "propio",
) -> Resultado:
    """`almacen` decide de dónde sale el nulo cuando no se pasa un `frame`.

    Default `propio` desde el 2026-09-06: `market.db` + `results.db`, con las
    cotizaciones posteriores al primer lanzamiento excluidas. `legado` se
    conserva para poder reproducir los veredictos v1 contra su referencia
    original — un veredicto viejo comparado contra un nulo nuevo no dice nada.
    """
    if frame is not None:
        f = frame
    elif almacen == "propio":
        f = load_frame_propio(seasons)
    else:
        f = load_frame(seasons)
    valores = feature.calcular([int(pk) for pk in f.game_pk])
    s = np.array([valores.get(int(pk), np.nan) for pk in f.game_pk], float)

    ok = np.isfinite(s) & np.isfinite(f.p_market) & np.isfinite(f.y)
    f, s = f.subset(ok), s[ok]
    Lp = _logit(f.p_market)

    pliegues: List[dict] = []
    for prueba in sorted(set(f.season.tolist())):
        te = f.season == prueba
        tr = ~te
        if te.sum() < 100 or tr.sum() < 200:
            continue
        # Estandarizar con estadísticos del ENTRENAMIENTO únicamente: hacerlo
        # con la muestra completa mete el pliegue de prueba en el ajuste por la
        # puerta de al lado, y en señales de coeficiente 0.05 eso decide.
        mu, sd = s[tr].mean(), s[tr].std()
        if sd == 0:
            continue
        z = (s - mu) / sd
        b = _fit_logistica(np.column_stack([Lp[tr], z[tr]]), f.y[tr])
        p = 1 / (1 + np.exp(-(b[0] + b[1] * Lp[te] + b[2] * z[te])))

        b_v0 = brier(f.p_market[te], f.y[te])
        pliegues.append({
            "pliegue": int(prueba),
            "n_test": int(te.sum()),
            "coef": float(b[2]),
            "brier_v0": b_v0,
            "brier_con": brier(p, f.y[te]),
            "mejora": b_v0 - brier(p, f.y[te]),
            "roi": _roi(p, f.subset(te)),
        })
    return Resultado(feature=feature.nombre, n=len(f), pliegues=pliegues)


def _roi(p: np.ndarray, f: EvalFrame) -> List[dict]:
    eh, ea = p - f.p_market, (1 - p) - (1 - f.p_market)
    salida = []
    for u in UMBRALES:
        pnl, n = 0.0, 0
        for edge, precio, gana in ((eh, f.best_home, f.y), (ea, f.best_away, 1 - f.y)):
            sel = (edge >= u) & np.isfinite(precio)
            if sel.any():
                pnl += float((gana[sel] * (precio[sel] - 1) - (1 - gana[sel])).sum())
                n += int(sel.sum())
        salida.append({"umbral": u, "n": n,
                       "roi_pct": round(100 * pnl / n, 2) if n else None})
    return salida


def imprimir(r: Resultado) -> None:
    print(f"\n{'='*72}\n  {r.feature}   n={r.n}\n{'='*72}")
    print(f"  {'prueba':>7} {'n':>6} {'coef':>9} {'Brier v0':>10} {'con señal':>11} {'mejora':>10}")
    for p in r.pliegues:
        print(f"  {p['pliegue']:>7} {p['n_test']:6d} {p['coef']:+9.4f} "
              f"{p['brier_v0']:10.5f} {p['brier_con']:11.5f} {p['mejora']:+10.5f}")
    print("\n  ROI por umbral de edge:")
    for p in r.pliegues:
        celdas = "  ".join(
            f"{100*x['umbral']:.1f}%:{x['roi_pct']:+7.2f}%({x['n']})"
            if x["roi_pct"] is not None else f"{100*x['umbral']:.1f}%:  n/a"
            for x in p["roi"])
        print(f"    {p['pliegue']}  {celdas}")
    if not r.medible:
        print(f"\n  → NO MEDIBLE — {len(r.pliegues)} pliegue(s) con muestra "
              f"suficiente sobre n={r.n}. No es un veredicto sobre la señal.\n")
        return
    print(f"\n  signo estable        {'sí' if r.signo_estable else 'NO'}")
    print(f"  mejora el Brier      {'sí' if r.mejora_brier else 'NO'}")
    print(f"  ROI positivo         {'sí' if r.roi_no_empeora else 'NO'}")
    print(f"  → {'CRUZA EL PORTÓN' if r.cruza else 'NO CRUZA'}\n")
