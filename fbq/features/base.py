"""fbq/features/base.py — qué es una feature, y qué tiene que pasar para entrar.

Una feature es una **señal**, no una probabilidad: un número por juego que se
suma al precio, nunca lo reemplaza. Esa distinción es la lección más cara del
sistema anterior. Ahí cada motor producía su propio ajuste de λ, el conjunto
producía una probabilidad completa, y esa probabilidad competía contra el
mercado desde cero — perdiendo, porque re-derivaba peor lo que el precio ya
sabía. Medido: las nueve señales correlacionan 0.28-0.65 con el mercado y sólo
0.05-0.10 con el resultado.

Acá el precio es el punto de partida y la feature sólo tiene que aportar lo que
al precio le falta. Es una pregunta mucho más chica y mucho más contestable.

El portón está en `gate.py` y es obligatorio. Una feature que no lo cruza no se
borra: se registra con su veredicto, para que nadie la vuelva a intentar sin
saber que ya se midió.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

# Una feature calcula {game_pk: valor} para los juegos que se le pidan. Devuelve
# sólo los que puede: un juego ausente se excluye de la evaluación, nunca se
# rellena con la media, que parecería una observación y no lo es.
Calculo = Callable[[Sequence[int]], Dict[int, float]]


@dataclass(frozen=True)
class Veredicto:
    """Un veredicto medido, con la referencia contra la que se midió.

    La referencia es parte del veredicto, no un pie de página. Un "NO CRUZA"
    sin decir sobre qué muestra y contra qué nulo se midió no se puede ni
    reproducir ni comparar con el siguiente — y este proyecto ya tiró siete
    baselines por comparar números de muestras distintas creyendo que eran la
    misma.

    Los veredictos NO se pisan: se apilan. Cuando el nulo cambia, la medición
    vieja sigue siendo cierta sobre su propia referencia.
    """

    version: str          # v1, v2, ...
    fecha: str            # cuándo se midió
    almacen: str          # "legado" | "propio"
    referencia: str       # la barra exacta: n y Brier del nulo
    resultado: str        # "CRUZA" | "NO CRUZA" | "NO MEDIBLE"
    nota: str = ""
    # El commit del que salió la medición. Sin esto, "reproducir el v1" es una
    # invitación a correr el código de hoy y creer que se reprodujo el de ayer.
    codigo: str = ""


@dataclass
class Feature:
    nombre: str
    descripcion: str
    calcular: Calculo
    # Por qué se espera que aporte ALGO QUE EL PRECIO NO TIENE. Escribirlo antes
    # de medir es la parte que evita el sobreajuste narrativo: si la única razón
    # que se puede dar es "salió significativo", no hay razón.
    hipotesis: str = ""
    # Historial de veredictos, del más viejo al más nuevo. Nunca se reemplaza
    # uno: se agrega el siguiente.
    veredictos: List[Veredicto] = field(default_factory=list)

    @property
    def veredicto(self) -> Optional[str]:
        """El resultado vigente: el del último veredicto medido."""
        return self.veredictos[-1].resultado if self.veredictos else None

    @property
    def vigente(self) -> Optional[Veredicto]:
        return self.veredictos[-1] if self.veredictos else None


_REGISTRO: Dict[str, Feature] = {}


def registrar(f: Feature) -> Feature:
    if f.nombre in _REGISTRO:
        raise ValueError(f"feature duplicada: {f.nombre!r}")
    _REGISTRO[f.nombre] = f
    return f


def obtener(nombre: str) -> Feature:
    if nombre not in _REGISTRO:
        raise KeyError(f"feature desconocida: {nombre!r}. Registradas: "
                       f"{sorted(_REGISTRO)}")
    return _REGISTRO[nombre]


def todas() -> List[Feature]:
    return [_REGISTRO[k] for k in sorted(_REGISTRO)]
