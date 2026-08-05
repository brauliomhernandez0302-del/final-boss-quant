"""fbq — el sistema, construido en orden de dependencias.

Cada paquete es un paso del build y sólo puede depender de los anteriores. Esa
regla es la arquitectura: si `features/` necesitara algo de `model/`, el orden
está mal y el error aparece como un import circular, no como una sorpresa seis
meses después.

    core/       paso 0    el contrato de tiempo e identidad
    sources/    —         los fetchers: hablan con el exterior, devuelven crudo
    market/     paso 1    precios, append-only
    results/    paso 2    hechos: quién ganó
    evaluator/  pasos 3-4 la balanza, y v0 = el mercado
    features/   pasos 5-6 señales con contrato as-of
    model/      pasos 7-8 probabilidad y detección de valor
    stake/      paso 9    sizing
    ledger/     pasos 10-11 publicación y reconciliación
    app/        paso 12   la UI

`sources/` no tiene número porque no es un paso: es la frontera con el mundo.
Vive aparte por una razón concreta — traer el dato, persistirlo y derivarlo son
tres trabajos distintos, y mezclarlos es cómo se pierde un precio al
normalizarlo antes de guardarlo, o cómo se lee una clave de clima que el
fetcher nunca emitió.
"""

__all__ = []
