"""fbq/evaluator/derived.py — la balanza de los mercados derivados.

Total y runline. Mismo contrato que el moneyline: el candidato entra por
parámetro y el nulo es siempre el precio desvigorizado del mismo libro.

Por qué existe aparte y no como una opción del marco de moneyline: en un
derivado el evento depende de un PUNTO, y ese punto varía por juego. "El
mercado" del total 8.5 y el del 9.0 son eventos distintos, así que la muestra
no se puede mezclar sin decir cuál es cuál. El moneyline no tiene ese problema
y por eso su marco es más simple.

El punto del runline va FIRMADO y pegado a su lado. Emparejar la probabilidad
de un evento con el precio de otro es el error que en el sistema anterior
publicó 134 picks con EV inflado, 55 de ellos con edge fabricado, y se llevó el
61.8% del capital arriesgado. Acá el evento se DERIVA del punto, no al revés.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from fbq.evaluator.frame import _devig_multiplicativo

DB_PATH = Path(__file__).parent.parent.parent / "data" / "predictions_history.db"


@dataclass
class DerivedFrame:
    """Un juego por posición. `punto` es el del mercado que se está midiendo.

    `y` es el resultado del lado A: OVER para totales, el LOCAL CUBRE para
    runline. El lado B es su complemento salvo empuje, que se marca aparte.
    """

    game_pk: np.ndarray
    season: np.ndarray
    official_date: np.ndarray
    home_team: np.ndarray
    punto: np.ndarray          # FIRMADO en runline: −1.5 si el local es favorito
    y: np.ndarray              # 1 = ganó el lado A
    empuje: np.ndarray         # 1 = el resultado cayó exactamente en el punto
    p_market: np.ndarray       # Pinnacle desvigorizado, lado A
    best_a: np.ndarray
    best_b: np.ndarray
    home_runs: np.ndarray
    away_runs: np.ndarray

    def __len__(self) -> int:
        return len(self.game_pk)

    def subset(self, mask: np.ndarray) -> "DerivedFrame":
        return DerivedFrame(**{k: v[mask] for k, v in self.__dict__.items()})

    @property
    def sin_empuje(self) -> "DerivedFrame":
        """Los juegos que no cayeron en el punto.

        Un empuje devuelve la apuesta: no es ni acierto ni error, y meterlo en
        un Brier como 0 o como 1 inventa un resultado que no ocurrió. Para
        calibración se excluye; para ROI cuenta como 0 de ganancia, que es lo
        que realmente pasa.
        """
        return self.subset(self.empuje == 0)


def load_derived(
    mercado: str,
    seasons: Sequence[int] = (2024, 2025),
    *,
    db_path: Path = DB_PATH,
) -> DerivedFrame:
    """`mercado` es 'total' o 'runline'.

    El resultado sale de las carreras reales, no de una columna guardada: un
    total y un margen son funciones del marcador, y derivarlos en el momento
    evita que una columna se desincronice del marcador que la generó.
    """
    if mercado not in ("total", "runline"):
        raise ValueError(f"mercado no soportado: {mercado!r}")

    marcas = ",".join("?" * len(seasons))
    if mercado == "total":
        cols = ("h.total_point_pin AS punto, h.total_point_best AS punto_best, "
                "h.total_over_pin AS pin_a, h.total_under_pin AS pin_b, "
                "h.total_over_best AS best_a, h.total_under_best AS best_b")
    else:
        cols = ("h.rl_home_point_pin AS punto, h.rl_home_point_best AS punto_best, "
                "h.rl_home_pin AS pin_a, h.rl_away_pin AS pin_b, "
                "h.rl_home_best AS best_a, h.rl_away_best AS best_b")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        filas = con.execute(
            f"""SELECT g.game_pk, g.season, g.official_date, g.home_team,
                       g.actual_home_runs, g.actual_away_runs, {cols}
                FROM game_outcomes g JOIN historical_odds h ON h.game_pk = g.game_pk
                WHERE g.season IN ({marcas})
                  AND g.actual_home_runs IS NOT NULL
                ORDER BY g.official_date, g.game_pk""",
            tuple(seasons),
        ).fetchall()
    finally:
        con.close()

    # El precio del ROI tiene que estar cotizado en el MISMO punto que la
    # probabilidad. Sin esta condición se empareja el evento difícil con el
    # precio del fácil — es PURP-1 exactamente, y aparece en el 7.7% de los
    # runlines (343 juegos con Pinnacle en −1.5 y el mejor precio en +1.5) y en
    # el 12.8% de los totales. La regla ya existía en `market.pair_before`; hay
    # que aplicarla acá también.
    filas = [r for r in filas if r["punto"] is not None
             and r["pin_a"] and r["pin_b"] and r["best_a"] and r["best_b"]
             and r["punto_best"] is not None
             and abs(float(r["punto_best"]) - float(r["punto"])) < 1e-9]
    if not filas:
        raise ValueError(f"sin juegos con precio de {mercado} para {list(seasons)}")

    punto = np.array([r["punto"] for r in filas], float)
    hr = np.array([r["actual_home_runs"] for r in filas], float)
    ar = np.array([r["actual_away_runs"] for r in filas], float)

    if mercado == "total":
        real = hr + ar
        y = (real > punto).astype(float)          # OVER
        empuje = (real == punto).astype(float)
    else:
        # El local cubre si su margen supera el UMBRAL, que es −punto: con el
        # local favorito a −1.5 el umbral es +1.5 y hay que ganar por 2 o más;
        # con el local a +1.5 el umbral es −1.5 y alcanza con perder por 1.
        umbral = -punto
        real = hr - ar
        y = (real > umbral).astype(float)
        empuje = (real == umbral).astype(float)

    pin_a = np.array([r["pin_a"] for r in filas], float)
    pin_b = np.array([r["pin_b"] for r in filas], float)
    p_market, _ = _devig_multiplicativo(pin_a, pin_b)

    return DerivedFrame(
        game_pk=np.array([r["game_pk"] for r in filas]),
        season=np.array([r["season"] for r in filas]),
        official_date=np.array([r["official_date"] for r in filas]),
        home_team=np.array([r["home_team"] for r in filas]),
        punto=punto, y=y, empuje=empuje, p_market=p_market,
        best_a=np.array([r["best_a"] for r in filas], float),
        best_b=np.array([r["best_b"] for r in filas], float),
        home_runs=hr, away_runs=ar,
    )


def roi_derivado(
    p: np.ndarray, f: DerivedFrame, umbrales: Sequence[float] = (0.02, 0.04, 0.06, 0.08, 0.10),
) -> List[dict]:
    """ROI apostando 1u cuando el edge supera el umbral.

    Un empuje devuelve el stake: cuenta como apuesta con ganancia 0, no se
    excluye. Excluirlo inflaría el ROI, porque los empujes son sistemáticamente
    más frecuentes justo en las líneas enteras donde más se apuesta.
    """
    ea, eb = p - f.p_market, (1 - p) - (1 - f.p_market)
    salida = []
    for u in umbrales:
        pnl, n = 0.0, 0
        for edge, precio, gana in ((ea, f.best_a, f.y), (eb, f.best_b, 1 - f.y)):
            sel = (edge >= u) & np.isfinite(precio)
            if not sel.any():
                continue
            g, e = gana[sel], f.empuje[sel]
            # empuje → 0; acierto → precio−1; error → −1
            pnl += float(np.sum(np.where(e == 1, 0.0,
                                         np.where(g == 1, precio[sel] - 1, -1.0))))
            n += int(sel.sum())
        salida.append({"umbral": u, "n": n,
                       "roi_pct": round(100 * pnl / n, 3) if n else None})
    return salida
