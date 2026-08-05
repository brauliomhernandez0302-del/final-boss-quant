"""fbq.results — paso 2: los hechos.

Un juego, una fila, sólo lo que ocurrió. Ninguna probabilidad, ninguna λ,
ninguna columna de modelo.

Esa separación no es estética. En el proyecto anterior el marcador real vivía
en la misma tabla que las predicciones, y una corrida rutinaria de backtest
sobrescribió 563 filas de predicciones en vivo sin dejar rastro ni backup. Un
hecho y una opinión tienen ciclos de vida distintos: el hecho se escribe una vez
cuando el juego termina, la opinión se escribe cada vez que se corre un modelo.
Compartir tabla es garantizar que tarde o temprano una pise a la otra.
"""

from fbq.results.store import ResultsStore, DB_PATH

__all__ = ["ResultsStore", "DB_PATH"]
