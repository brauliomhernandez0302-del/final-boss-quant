"""fbq.anuncios — lo que se ANUNCIÓ, y cuándo se supo.

Un paquete aparte de `market/` y de `results/` porque guarda una tercera clase
de dato, y confundirla con las otras dos es exactamente cómo se fabrica una
fuga:

    market/     un PRECIO observado — un hecho sobre el presente
    results/    un MARCADOR — un hecho sobre el pasado
    anuncios/   una AFIRMACIÓN SOBRE EL FUTURO: quién va a abrir mañana

Lo que hace útil a un anuncio no es su contenido sino **el instante en que se
lo observó**. El contenido se puede volver a pedir en cualquier momento; el
instante, no. Y sin el instante el dato es inservible para predecir, porque no
se puede afirmar que estuviera disponible antes del corte.

**Por qué esto no se puede rellenar hacia atrás.** Medido el 2026-09-06 sobre
400 juegos al azar (800 equipo-juego): el `probablePitcher` que la API publica
hoy para un juego ya jugado coincide con el abridor real en **799 de 800** y
**difiere en 0**. Los abridores se cancelan por lesión, enfermedad o lluvia
varias veces por temporada, así que una discrepancia de 0,00% no dice que
nunca fallen: dice que el campo se rellena con lo que pasó. Reconstruir el
anuncio del pasado es imposible; capturarlo desde hoy, trivial y gratis.
"""

from fbq.anuncios.store import AnunciosStore

__all__ = ["AnunciosStore"]
