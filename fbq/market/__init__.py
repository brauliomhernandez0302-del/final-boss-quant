"""market/ — el almacén de precios de mercado.

Escrito desde cero el 2026-08-04 con una sola regla de diseño: **este módulo
solo inserta, nunca actualiza ni borra**. El precio de un mercado en un
instante es un hecho histórico; una vez ocurrido no cambia, y sobreescribirlo
destruye la única serie que no se puede volver a comprar.

No sabe nada de picks, de modelos ni de λ. Captura el board completo esté o
no esté apostado ese juego, porque la decisión de qué juegos importan se toma
después y no puede reescribir el pasado.
"""

from fbq.market.store import MarketStore, DB_PATH

__all__ = ["MarketStore", "DB_PATH"]
