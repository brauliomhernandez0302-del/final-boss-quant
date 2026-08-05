"""fbq/core/identity.py — cuándo dos referencias son el mismo juego.

El emparejamiento juego↔evento-de-odds es una IDENTIDAD, no una búsqueda: se
resuelve una vez y se guarda. Pero esa primera vez hay que resolverla, y la
regla que la gobierna vive acá para que sea una sola en todo el sistema.

La regla central es la ABSTENCIÓN. Ante dos candidatos casi igual de
plausibles, no se elige el más cercano: no se elige ninguno. El caso real que
la fuerza es el doubleheader tradicional, cuyos dos juegos el schedule de MLB
separa por 5 minutos — quedarse con "el que estaba un minuto más cerca" es
quedarse con el precio del otro partido, y eso no deja rastro.

El umbral sale de medir la separación REAL entre los dos juegos de un
doubleheader sobre las 431 fechas con datos (2024-2026). La distribución es
bimodal y el hueco entre modos está vacío:

    partido      (doubleHeader='S'): 275 a 405 min   (mediana 330)
    tradicional  (doubleHeader='Y'): 5 min, sin excepción

90 minutos es el único umbral que sirve para los dos regímenes: queda a 3x del
modo de arriba y a 18x del de abajo.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")

# Cuán lejos puede estar un candidato del inicio buscado para seguir siendo
# candidato. Amplia a propósito: filtrar de más acá esconde el problema en vez
# de resolverlo, y la desambiguación real la hace MARGEN_MINIMO.
VENTANA_EMPAREJAMIENTO = timedelta(hours=6)

# Por cuánto tiene que ganarle el mejor candidato al segundo para considerarse
# identificado. Ver la nota del módulo sobre de dónde sale el número.
MARGEN_MINIMO = timedelta(minutes=90)

# Alias de equipos → nombre canónico, aplicado a AMBOS lados de cualquier
# comparación. Un mapa de un solo sentido lleva un nombre LEJOS del otro en
# cuanto una de las dos fuentes cambia: los Athletics cambiaron de ciudad dos
# veces y eso costó 276 juegos sin precio, atribuidos a "team name mismatch"
# sin que nadie mirara cuál.
ALIAS_EQUIPOS = {
    "oakland athletics":    "athletics",
    "sacramento athletics": "athletics",
    "las vegas athletics":  "athletics",
}


def canonico(nombre: str) -> str:
    limpio = (nombre or "").strip().lower()
    return ALIAS_EQUIPOS.get(limpio, limpio)


def mismo_equipo(a: str, b: str) -> bool:
    """Contención de cadenas sobre los nombres canónicos.

    No es fuzzy matching: no hay distancia de edición ni umbral de parecido, así
    que dos equipos distintos no pueden acercarse lo suficiente por accidente.
    Resuelve "Athletics" vs "Oakland Athletics" y nada más.
    """
    x, y = canonico(a), canonico(b)
    return bool(x) and bool(y) and (x in y or y in x)


def _instante(valor: str) -> Optional[datetime]:
    if not valor:
        return None
    try:
        dt = datetime.fromisoformat(str(valor).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def elegir_unico(
    candidatos: Sequence[T],
    objetivo: str,
    *,
    inicio_de: Callable[[T], str],
    ventana: timedelta = VENTANA_EMPAREJAMIENTO,
    margen: timedelta = MARGEN_MINIMO,
) -> Tuple[Optional[T], str]:
    """El candidato inequívocamente más cercano al inicio buscado.

    Devuelve `(elegido, motivo)`. `elegido` es None en tres casos y el motivo
    los distingue, porque significan cosas distintas para quien llama:

        "sin_inicio"   el objetivo no trae un instante legible
        "sin_candidato" ninguno cae dentro de la ventana
        "ambiguo"      los dos mejores están más cerca entre sí que el margen

    Nunca devuelve "el mejor disponible" cuando hay empate práctico. Un enlace
    equivocado mete los datos de un juego en otro y no deja rastro; un enlace
    ausente se vuelve a intentar mañana.
    """
    t = _instante(objetivo)
    if t is None:
        return None, "sin_inicio"

    cercanos: List[Tuple[timedelta, T]] = []
    for c in candidatos:
        ti = _instante(inicio_de(c))
        if ti is None:
            continue
        delta = abs(ti - t)
        if delta <= ventana:
            cercanos.append((delta, c))

    if not cercanos:
        return None, "sin_candidato"

    cercanos.sort(key=lambda par: par[0])
    if len(cercanos) > 1 and (cercanos[1][0] - cercanos[0][0]) < margen:
        return None, "ambiguo"
    return cercanos[0][1], "ok"
