"""fbq.model — pasos 7-8: probabilidad, y después detección de valor.

Depende de `core/`, `market/`, `results/` y `evaluator/`; de nada más. En
particular **no importa nada del sistema anterior**: ese árbol está declarado
inválido y puede borrarse en cualquier momento.

El candidato v1 está preregistrado en `docs/PREREGISTRO_MODELO_V1_2026-09-06.md`,
commiteado ANTES de medir. Lo que se espera de él está escrito ahí y no se
mueve: se espera que pierda contra Pinnacle y le gane a la tasa base.

    pit.py        la compuerta temporal: qué hecho estaba disponible y cuándo
    features.py   los hechos deportivos, calculados sólo con lo disponible
    logistica.py  regresión logística con ridge, ajustada sólo en entrenamiento
    detector.py   los dos detectores de fuga: estructural y estadístico
    candidato.py  arma las filas, entrena expansivo y predice
"""

from fbq.model.candidato import CONFIG, Candidato, construir, evaluar_expansivo
from fbq.model.detector import BRIER_IMPLAUSIBLE, FugaDetectada, verificar_plausibilidad

__all__ = ["CONFIG", "Candidato", "construir", "evaluar_expansivo",
           "FugaDetectada", "verificar_plausibilidad", "BRIER_IMPLAUSIBLE"]
