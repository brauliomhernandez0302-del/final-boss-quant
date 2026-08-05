"""fbq.features — pasos 5-6: las señales, y el portón que deciden si entran.

Una feature es una SEÑAL que se suma al precio, nunca una probabilidad que lo
reemplace. El precio es el punto de partida y la feature sólo tiene que aportar
lo que al precio le falta — una pregunta mucho más chica y mucho más
contestable que "predecir el juego".

Ninguna entra sin cruzar `gate.py`. Las que no cruzan se quedan registradas con
su veredicto: borrarlas garantizaría que alguien las reintente sin saber que ya
se midieron.
"""

from fbq.features.base import Feature, obtener, registrar, todas
from fbq.features.gate import Resultado, evaluar, imprimir
from fbq.features import market_shape  # noqa: F401  — registra sus features

__all__ = ["Feature", "obtener", "registrar", "todas",
           "Resultado", "evaluar", "imprimir"]
