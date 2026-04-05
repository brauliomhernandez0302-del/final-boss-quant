from __future__ import annotations
import numpy as np


class AutoCalibrator:
    """
    Recalibra probabilidades en tiempo real usando un modelo Beta-Bernoulli.
    Este sistema aprende de los aciertos y fallos históricos.
    """

    def __init__(self, prior_alpha: float = 2.0, prior_beta: float = 2.0):
        """
        prior_alpha y prior_beta definen el "sesgo inicial" del modelo.
        Valores mayores => más suavizado, más lento para adaptarse.
        """
        self.alpha = prior_alpha
        self.beta = prior_beta
        self.n = 0  # número total de observaciones

    def update(self, y_true: int) -> None:
        """
        Actualiza el calibrador con un nuevo resultado binario.
        y_true = 1 si el evento ocurrió, 0 si no.
        """
        y_true = int(bool(y_true))
        self.alpha += y_true
        self.beta += 1 - y_true
        self.n += 1

    def transform(self, p: float) -> float:
        """
        Ajusta una probabilidad p (entre 0 y 1) usando el promedio posterior.
        Cuantos más datos haya, más precisa se vuelve la calibración.
        """
        p = np.clip(p, 1e-6, 1 - 1e-6)

        # La media posterior del modelo Beta es:
        posterior_mean = self.alpha / (self.alpha + self.beta)

        # Mezcla entre la predicción original y la calibrada según cantidad de datos
        weight = min(1.0, self.n / (self.n + 10))  # confianza gradual
        calibrated_p = (1 - weight) * p + weight * posterior_mean

        return float(calibrated_p)

