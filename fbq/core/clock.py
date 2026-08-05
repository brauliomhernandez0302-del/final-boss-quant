"""fbq/core/clock.py — el contrato de tiempo.

Tres reglas, y las tres son funciones para que ningún llamador tenga que
acordarse de aplicarlas:

1. **El día de un juego es el día OFICIAL de schedule**, nunca `date()` del
   timestamp UTC de inicio. Para cualquier nocturno de la costa oeste el UTC
   cae un día calendario adelante. Sobre datos reales de 2024-2026 eso pasa en
   el 22-24% de los juegos, y usarlo como día del juego adelanta un día todos
   los cortes derivados.

2. **Predecir un juego del día D usa datos hasta el FIN de D−1.** Un solo sitio
   calcula ese corte. Antes vivía repetido en cinco funciones que restaban un
   segundo a mano, y una de las cinco no lo restaba.

3. **La comparación contra un corte es ESTRICTA.** `<`, nunca `<=`. Un `<=`
   sobre un corte que cae en el propio día del juego incluye el juego en su
   propia instantánea, que es la definición exacta de un leak.

`AsOf` no es un alias cosmético de `str`: es el recordatorio en la firma de que
ese parámetro no es una fecha cualquiera sino un corte, y que quien lo construye
tiene que hacerlo con `cutoff_para_dia`.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence, Tuple, TypeVar

# Un corte temporal, en ISO-8601 UTC. Ver la nota del módulo sobre por qué
# tiene nombre propio.
AsOf = str

T = TypeVar("T")


def _a_fecha(valor: str | date | datetime) -> date:
    if isinstance(valor, datetime):
        return valor.date()
    if isinstance(valor, date):
        return valor
    return date.fromisoformat(str(valor)[:10])


def dia_oficial(juego: dict) -> str:
    """El día de schedule al que pertenece el juego.

    Prefiere `official_date`/`officialDate` y sólo cae al timestamp de inicio si
    el schedule no lo trajo — caso que hay que tratar como excepcional y
    ruidoso, no como equivalente.
    """
    for clave in ("official_date", "officialDate"):
        valor = juego.get(clave)
        if valor:
            return str(valor)[:10]
    for clave in ("game_date", "gameDate", "commence_time"):
        valor = juego.get(clave)
        if valor:
            return str(valor)[:10]
    raise KeyError("el juego no trae ni día oficial ni timestamp de inicio")


def cutoff_para_dia(dia: str | date | datetime) -> AsOf:
    """El corte para predecir un juego de ese día: el fin del día ANTERIOR.

    Único sitio donde se calcula. Cualquier consumidor que necesite "hasta
    cuándo puedo mirar" llama acá y no hace aritmética de fechas por su cuenta.
    """
    d = _a_fecha(dia) - timedelta(days=1)
    return f"{d.isoformat()}T23:59:59+00:00"


def es_anterior(momento: str, corte: AsOf) -> bool:
    """`momento < corte`, ESTRICTO, tolerando formatos ISO mezclados.

    La comparación se hace sobre datetimes y no sobre cadenas: `"2024-05-01"` y
    `"2024-05-01T00:00:00+00:00"` son el mismo instante y como texto ordenan
    distinto. Ese detalle es real — el cache del proyecto anterior guardaba dos
    convenciones a la vez (`T00:00:00` y `T23:59:59`) y sólo funcionaba por
    aritmética afortunada.
    """
    return _a_instante(momento) < _a_instante(corte)


def _a_instante(valor: str) -> datetime:
    texto = str(valor).replace("Z", "+00:00")
    if len(texto) == 10:
        texto += "T00:00:00+00:00"
    dt = datetime.fromisoformat(texto)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def instantanea_vigente(
    candidatas: Iterable[Tuple[str, T]], corte: AsOf,
) -> Optional[T]:
    """La instantánea más reciente ESTRICTAMENTE anterior al corte.

    `candidatas` son pares `(momento, valor)`. Devuelve `None` si ninguna
    califica — nunca la más cercana, nunca la primera disponible. Que no haya
    dato es un estado legítimo y el llamador tiene que verlo; devolver un dato
    posterior al corte para "no quedarse sin nada" es el leak.
    """
    validas = [(m, v) for m, v in candidatas if es_anterior(m, corte)]
    if not validas:
        return None
    return max(validas, key=lambda par: _a_instante(par[0]))[1]


def ahora() -> str:
    """El instante actual en UTC ISO-8601. Un solo sitio, para poder fijarlo
    en los tests sin parchear `datetime` en cada módulo."""
    return datetime.now(timezone.utc).isoformat()
