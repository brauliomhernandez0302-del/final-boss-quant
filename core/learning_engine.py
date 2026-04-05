from __future__ import annotations
import numpy as np


class LearningEngine:
    """
    Motor de aprendizaje continuo.

    Combina predicciones de distintos submodelos (baseline, market, live)
    y ajusta sus pesos automáticamente según el desempeño real.
    """

    def __init__(self, names=("baseline", "market", "live"), lr: float = 0.05):
        """
        Args:
            names: Nombres de los submodelos que aportan predicciones.
            lr: Tasa de aprendizaje para el ajuste de pesos.
        """
        self.names = list(names)
        self.w = np.ones(len(self.names)) / len(self.names)  # pesos iniciales uniformes
        self.lr = lr
        self.n = 0  # número de observaciones vistas

    def predict(self, preds: dict[str, float]) -> float:
        """
        Calcula una probabilidad combinada ponderando las predicciones de cada fuente.
        """
        x = np.array([preds.get(k, 0.5) for k in self.names], dtype=float)
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return float(np.dot(self.w, x) / np.sum(self.w))

    def update(self, preds: dict[str, float], outcome: int) -> None:
        """
        Ajusta los pesos según el resultado real (outcome):
        outcome = 1 si el evento ocurrió, 0 si no.
        """
        p = self.predict(preds)
        grad = (p - outcome)  # gradiente del error
        x = np.array([preds.get(k, 0.5) for k in self.names], dtype=float)

        # Ajuste tipo gradiente descendente
        self.w -= self.lr * grad * x

        # Evitar pesos negativos y renormalizar
        self.w = np.clip(self.w, 1e-6, None)
        self.w /= np.sum(self.w)

        self.n += 1

    def get_weights(self) -> dict[str, float]:
        """
        Retorna los pesos actuales como diccionario.
        """
        return {k: float(v) for k, v in zip(self.names, self.w)}

    def reset(self) -> None:
        """
        Reinicia el calibrador a pesos uniformes.
        """
        self.w[:] = 1 / len(self.w)
        self.n = 0

