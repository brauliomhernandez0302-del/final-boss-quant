"""fbq/model/detector.py — los dos detectores de fuga.

Son dos porque el primero se puede esquivar, y el modo de fallo que importa no
es el error honesto sino el atajo que nadie revisó.

1. **Estructural** — vive en `pit.py`: todo hecho pasa por una compuerta que
   levanta `FugaDetectada` si al corte no estaba disponible. Impide la fuga por
   construcción... siempre que el constructor de features use la compuerta.

2. **Estadístico** — vive acá: mira el RESULTADO. Un candidato demasiado bueno
   para ser cierto se rechaza aunque su código parezca limpio. Es el que atrapa
   a quien esquivó la compuerta.

## De dónde sale el umbral

`BRIER_IMPLAUSIBLE = 0.22`. Pinnacle —el libro más afilado del mercado más
eficiente que existe— mide **0,2421** sobre 6.106 juegos
(`docs/REFERENCIA_MERCADO_2026-09.md`). Un candidato en 0,22 sería una mejora
relativa del 9% sobre Pinnacle; nada en la literatura reporta algo así sobre
moneyline de MLB, y el sistema anterior con Statcast, PIT y nueve motores quedó
en 0,24620.

En este problema **"demasiado bueno" es sinónimo de "fuga"**, y el umbral está
puesto donde la sorpresa deja de ser plausible y pasa a ser una alarma. Es
deliberadamente permisivo: prefiere dejar pasar una fuga chica antes que
rechazar un modelo bueno de verdad. Contra las fugas chicas está la compuerta.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from fbq.model.pit import FugaDetectada

BRIER_IMPLAUSIBLE = 0.22

# El nulo publicado, para poder decir "cuánto mejor que Pinnacle" en el mensaje.
BRIER_MERCADO_REFERENCIA = 0.242124067


def verificar_plausibilidad(
    p: np.ndarray, y: np.ndarray, *, nombre: str = "candidato",
    umbral: float = BRIER_IMPLAUSIBLE,
) -> float:
    """El Brier del candidato, o `FugaDetectada` si es implausiblemente bueno.

    Se llama SIEMPRE, incluso —sobre todo— cuando el resultado es el esperado:
    un detector que sólo se enciende cuando alguien sospecha no es un detector.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    brier = float(np.mean((p - y) ** 2))
    if brier < umbral:
        mejora = 100 * (BRIER_MERCADO_REFERENCIA - brier) / BRIER_MERCADO_REFERENCIA
        raise FugaDetectada(
            f"FUGA: {nombre} mide Brier {brier:.5f}, por debajo del umbral de "
            f"plausibilidad {umbral}. Serían {mejora:.1f}% mejor que Pinnacle "
            f"({BRIER_MERCADO_REFERENCIA:.6f}) sobre el mercado más eficiente "
            f"que existe. En este problema, demasiado bueno significa que un "
            f"hecho posterior al corte entró en las variables.")
    return brier
