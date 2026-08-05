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


@dataclass
class Feature:
    nombre: str
    descripcion: str
    calcular: Calculo
    # Por qué se espera que aporte ALGO QUE EL PRECIO NO TIENE. Escribirlo antes
    # de medir es la parte que evita el sobreajuste narrativo: si la única razón
    # que se puede dar es "salió significativo", no hay razón.
    hipotesis: str = ""
    # Veredicto medido, cuando ya pasó por el portón.
    veredicto: Optional[str] = None
    nota: str = ""


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
