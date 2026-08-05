"""evaluator/score.py — las métricas, y la única que decide.

Cuatro bloques, en orden de cuánto pesan para decidir si un candidato sirve:

1. **La escalera**: moneda, tasa base, candidato, mercado. Un candidato que no
   le gana a la tasa base no predice nada; uno que no le gana al mercado no
   aporta nada que el precio no tenga ya.
2. **La regresión conjunta**: `y ~ logit(mercado) + logit(candidato)`. Es LA
   prueba. Un Brier apenas peor que el del mercado puede seguir aportando
   información ortogonal; un coeficiente que no es positivo acá significa que,
   sabiendo el precio, el candidato no agrega nada. Con bootstrap AGRUPADO
   —los equipos se repiten cientos de veces y los errores iid inflan los t
   entre 5x y 21x sobre este tipo de dato, lección que costó tres falsos
   positivos en la auditoría del pipeline.
3. **El ROI por umbral de edge**: la cola, que es donde se apuesta. El paso 11
   de la auditoría del pipeline mostró que Brier y ROI pueden moverse en
   direcciones OPUESTAS —neutralizar el sesgo de equipo mejoraba el Brier y
   derrumbaba el ROI en los cinco umbrales—, así que este bloque es gate
   obligatorio y no métrica secundaria.
4. **La calibración por decil**: dónde está el error, no cuánto es.

Todo con `p_market` como referencia fija. Puntuar el propio mercado como
candidato tiene que reproducir la barra exactamente — es la autoprueba del
instrumento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from evaluator.frame import EvalFrame

Candidato = Union[Mapping[int, float], Callable[[int], float], np.ndarray]

# Umbrales de edge del gate de ROI. Los mismos que reporta
# backtest_and_retrain.py, para que los números sean comparables.
UMBRALES_EDGE = (0.02, 0.04, 0.06, 0.08, 0.10)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ventaja_sobre_azar(p: np.ndarray, y: np.ndarray) -> float:
    """Cuánto mejora el Brier sobre la moneda, en %.

    Toda la señal de este problema vive en ~3 milésimos sobre 0.25, así que
    "el Brier bajó 0.0005" es ilegible y "la ventaja subió 16% relativo" no.
    """
    return float((0.25 - brier(p, y)) / 0.25 * 100)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_logistica(X: np.ndarray, y: np.ndarray, iters: int = 300) -> np.ndarray:
    """IRLS con intercepto. Newton puro, sin dependencias nuevas."""
    X = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = p * (1 - p) + 1e-12
        g = X.T @ (y - p)
        H = (X * W[:, None]).T @ X
        b = b + np.linalg.solve(H + 1e-8 * np.eye(len(b)), g)
    return b


def calibracion(p: np.ndarray, y: np.ndarray, n_grupos: int = 10) -> List[dict]:
    """Predicho vs real por grupo de igual tamaño, ordenado por p."""
    orden = np.argsort(p)
    grupos = np.array_split(orden, n_grupos)
    return [
        {"n": int(len(g)), "predicho": float(p[g].mean()), "real": float(y[g].mean())}
        for g in grupos if len(g)
    ]


def roi_por_umbral(
    p: np.ndarray, frame: EvalFrame, umbrales: Sequence[float] = UMBRALES_EDGE,
) -> List[dict]:
    """ROI de apostar 1u cada vez que el edge supera el umbral.

    El edge es contra la probabilidad JUSTA (mercado desvigorizado), no contra
    la implícita cruda: contra el precio crudo cualquier modelo aparenta un edge
    del tamaño del vig sin tener ninguno.

    Se apuesta al MEJOR precio disponible, que es lo que un apostador haría —
    distinto del precio de Pinnacle con el que se calcula la línea justa.
    """
    edge_home = p - frame.p_market
    edge_away = (1 - p) - (1 - frame.p_market)
    out = []
    for u in umbrales:
        filas = []
        for lado, edge, precio, gana in (
            ("home", edge_home, frame.best_home, frame.y),
            ("away", edge_away, frame.best_away, 1 - frame.y),
        ):
            sel = (edge >= u) & np.isfinite(precio) & np.isfinite(edge)
            if sel.any():
                filas.append((gana[sel] * (precio[sel] - 1) - (1 - gana[sel])).sum())
        n = int(((edge_home >= u) & np.isfinite(frame.best_home)).sum()
                + ((edge_away >= u) & np.isfinite(frame.best_away)).sum())
        pnl = float(sum(filas))
        out.append({
            "umbral": u, "n": n,
            "pnl_u": round(pnl, 2),
            "roi_pct": round(100 * pnl / n, 3) if n else None,
        })
    return out


def mezcla_fuera_de_muestra(
    frame: EvalFrame,
    p: np.ndarray,
    *,
    particion: str = "season",
) -> List[dict]:
    """¿Alguna mezcla de v0 con el candidato le gana a v0 SOLO, fuera de muestra?

    Ésta es la pregunta del paso 4, y sólo tiene sentido fuera de muestra. Un
    barrido de pesos sobre los mismos datos siempre encuentra un peso que ayuda
    o, en el mejor caso, uno que no daña: no distingue señal de sobreajuste.

    Se ajusta `y ~ logit(v0) + logit(candidato)` en un pliegue y se aplica al
    otro. La comparación es contra `v0` evaluado en ESE MISMO pliegue de prueba,
    nunca contra el `v0` global — si no, se compararían dos muestras distintas.

    Los pliegues son temporadas enteras y no filas al azar, a propósito: un
    corte aleatorio pondría juegos del mismo día a ambos lados y el ajuste
    aprendería del futuro por la puerta de al lado.
    """
    Lm, Lc = _logit(frame.p_market), _logit(p)
    pliegues = np.unique(getattr(frame, particion))
    if len(pliegues) < 2:
        return []

    out = []
    for prueba in pliegues:
        test = getattr(frame, particion) == prueba
        train = ~test
        if train.sum() < 50 or test.sum() < 50:
            continue
        b = _fit_logistica(np.column_stack([Lm[train], Lc[train]]), frame.y[train])
        p_mezcla = 1 / (1 + np.exp(-(b[0] + b[1] * Lm[test] + b[2] * Lc[test])))
        b_v0 = _fit_logistica(Lm[train][:, None], frame.y[train])
        p_v0_cal = 1 / (1 + np.exp(-(b_v0[0] + b_v0[1] * Lm[test])))
        out.append({
            "pliegue": (prueba.item() if hasattr(prueba, "item") else prueba),
            "n_test": int(test.sum()),
            "brier_v0": brier(frame.p_market[test], frame.y[test]),
            "brier_v0_recalibrado": brier(p_v0_cal, frame.y[test]),
            "brier_mezcla": brier(p_mezcla, frame.y[test]),
            "b_candidato_ajustado": float(b[2]),
        })
    return out


@dataclass
class Report:
    nombre: str
    n: int
    tasa_base: float
    overround_medio: float
    brier_candidato: float
    brier_mercado: float
    logloss_candidato: float
    logloss_mercado: float
    ventaja_candidato: float
    ventaja_mercado: float
    b_mercado: float
    b_candidato: float
    ic_b_candidato: Tuple[float, float]
    p_b_candidato_positivo: float
    n_bootstrap: int
    roi: List[dict] = field(default_factory=list)
    mezcla: List[dict] = field(default_factory=list)
    calib_candidato: List[dict] = field(default_factory=list)
    calib_mercado: List[dict] = field(default_factory=list)

    @property
    def brecha(self) -> float:
        """Brier del candidato menos el del mercado. Positivo = peor."""
        return self.brier_candidato - self.brier_mercado

    @property
    def aporta_sobre_el_precio(self) -> bool:
        """El veredicto: ¿el IC95 del coeficiente conjunto excluye el cero?"""
        return self.ic_b_candidato[0] > 0


def evaluate(
    frame: EvalFrame,
    candidato: Candidato,
    *,
    nombre: str = "candidato",
    n_bootstrap: int = 400,
    cluster_por: str = "home_team",
    seed: int = 7,
) -> Report:
    """Puntúa un candidato contra el resultado real y contra el mercado.

    `candidato` puede ser un dict {game_pk: p_home}, un callable(game_pk) o un
    array ya alineado al marco. Los juegos sin valor se excluyen; nunca se
    rellenan con 0.5, que parecería una predicción.

    `cluster_por` gobierna el bootstrap: se remuestrean CLÚSTERES enteros, no
    filas. Con `"home_team"` son 30 unidades, que es poco — el IC resultante es
    ancho a propósito, y esa anchura es la información honesta sobre cuánta
    evidencia independiente hay realmente.
    """
    if isinstance(candidato, np.ndarray):
        p = candidato.astype(float)
    elif callable(candidato):
        p = np.array([candidato(int(pk)) for pk in frame.game_pk], float)
    else:
        p = frame.candidate_from_column(dict(candidato))

    valido = np.isfinite(p) & np.isfinite(frame.p_market) & np.isfinite(frame.y)
    if not valido.any():
        raise ValueError("el candidato no cubre ningún juego del marco")
    f = frame.subset(valido)
    p = p[valido]

    Lm, Lc = _logit(f.p_market), _logit(p)
    conjunta = _fit_logistica(np.column_stack([Lm, Lc]), f.y)

    grupos = getattr(f, cluster_por)
    unicos = np.unique(grupos)
    indices = {g: np.where(grupos == g)[0] for g in unicos}
    rng = np.random.default_rng(seed)
    coefs = []
    for _ in range(n_bootstrap):
        elegidos = rng.choice(unicos, len(unicos), replace=True)
        idx = np.concatenate([indices[g] for g in elegidos])
        try:
            coefs.append(_fit_logistica(np.column_stack([Lm[idx], Lc[idx]]), f.y[idx]))
        except np.linalg.LinAlgError:
            continue
    coefs = np.array(coefs)
    ic = tuple(np.percentile(coefs[:, 2], [2.5, 97.5])) if len(coefs) else (np.nan, np.nan)
    p_pos = float((coefs[:, 2] > 0).mean()) if len(coefs) else float("nan")

    return Report(
        nombre=nombre,
        n=len(f),
        tasa_base=f.base_rate,
        overround_medio=float(np.nanmean(f.overround)),
        brier_candidato=brier(p, f.y),
        brier_mercado=brier(f.p_market, f.y),
        logloss_candidato=log_loss(p, f.y),
        logloss_mercado=log_loss(f.p_market, f.y),
        ventaja_candidato=ventaja_sobre_azar(p, f.y),
        ventaja_mercado=ventaja_sobre_azar(f.p_market, f.y),
        b_mercado=float(conjunta[1]),
        b_candidato=float(conjunta[2]),
        ic_b_candidato=(float(ic[0]), float(ic[1])),
        p_b_candidato_positivo=p_pos,
        n_bootstrap=len(coefs),
        roi=roi_por_umbral(p, f),
        mezcla=mezcla_fuera_de_muestra(f, p),
        calib_candidato=calibracion(p, f.y),
        calib_mercado=calibracion(f.p_market, f.y),
    )
