"""fbq/model/logistica.py — regresión logística con ridge, y nada más.

Deliberadamente simple. El preregistro fija `LAMBDA_L2 = 1.0` a priori y
prohíbe ajustarlo por validación cruzada: cualquier búsqueda de hiperparámetro
sobre 2025 o 2026 sería tocar los años de evaluación.

La estandarización guarda media y desvío **del entrenamiento** y los aplica al
pliegue de evaluación. Estandarizar con la muestra completa mete el pliegue de
prueba en el ajuste por la puerta de al lado — fuga chica pero real, y el portón
de features del proyecto ya la documenta como tal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

LAMBDA_L2 = 1.0


@dataclass
class Logistica:
    """Coeficientes ajustados, con la estandarización que los acompaña.

    Van juntos a propósito: un coeficiente sobre variables estandarizadas y una
    estandarización distinta a la del ajuste dan una predicción silenciosamente
    equivocada.
    """

    beta: np.ndarray          # [intercepto, ...coeficientes]
    mu: np.ndarray
    sd: np.ndarray
    nombres: Tuple[str, ...]
    n_entrenamiento: int
    lambda_l2: float

    def predecir(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        lineal = self.beta[0] + Z @ self.beta[1:]
        return 1.0 / (1.0 + np.exp(-lineal))

    @property
    def coeficientes(self) -> dict:
        return {"intercepto": float(self.beta[0]),
                **{n: float(b) for n, b in zip(self.nombres, self.beta[1:])}}


def ajustar(
    X: np.ndarray, y: np.ndarray, nombres: Tuple[str, ...],
    *, lambda_l2: float = LAMBDA_L2, iteraciones: int = 200,
) -> Logistica:
    """IRLS con penalización L2. El intercepto NO se penaliza.

    Penalizar el intercepto encogería la tasa base hacia 0.5, que es
    precisamente el hecho que el intercepto tiene que capturar (la ventaja de
    local). Es un error clásico y silencioso.
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0                      # una constante no aporta, no rompe
    Z = np.column_stack([np.ones(len(y)), (X - mu) / sd])

    penal = np.full(Z.shape[1], float(lambda_l2))
    penal[0] = 0.0                          # el intercepto queda libre
    P = np.diag(penal)

    beta = np.zeros(Z.shape[1])
    for _ in range(iteraciones):
        p = 1.0 / (1.0 + np.exp(-(Z @ beta)))
        W = p * (1 - p) + 1e-12
        grad = Z.T @ (y - p) - penal * beta
        H = (Z * W[:, None]).T @ Z + P
        paso = np.linalg.solve(H + 1e-10 * np.eye(len(beta)), grad)
        beta = beta + paso
        if np.max(np.abs(paso)) < 1e-10:
            break
    return Logistica(beta=beta, mu=mu, sd=sd, nombres=tuple(nombres),
                     n_entrenamiento=len(y), lambda_l2=float(lambda_l2))
