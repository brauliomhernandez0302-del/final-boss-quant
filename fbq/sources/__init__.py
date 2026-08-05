"""fbq.sources — la frontera con el mundo exterior.

No es un paso del build: es dónde vive todo lo que habla con una API ajena.

Vive aparte por una regla que el proyecto anterior no tenía: **traer el dato,
persistirlo y derivarlo son tres trabajos distintos.** Mezclarlos produce dos
fallos concretos que ya ocurrieron:

- Normalizar antes de guardar pierde lo que la normalización no contempló. Los
  precios de Pinnacle para total y runline llegaban en cada respuesta desde
  siempre y se descartaban en el parseo; cuando hicieron falta, no existía ni un
  histórico.
- Leer del dato derivado en vez del crudo hace posible leer una clave que nadie
  emite. La telemetría de clima registró `None` en 52 de 52 picks porque leía
  `game_data['weather']['source']`, que el fetcher no devuelve — mientras el
  motor SÍ recibía el clima y SÍ movía λ.

Por eso cada módulo de acá devuelve estructuras crudas del proveedor, sin
interpretarlas. Quien las interpreta es el `store` del paso correspondiente.
"""

__all__ = []
