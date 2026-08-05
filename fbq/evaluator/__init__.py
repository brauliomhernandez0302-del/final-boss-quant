"""evaluator/ — la balanza del proyecto.

Puntúa **cualquier** función `juego → probabilidad` contra el resultado real, y
siempre contra el mismo nulo: el precio de mercado desvigorizado.

La regla de diseño es una sola y es la razón de que este módulo exista separado:
**el candidato entra por parámetro**. El evaluador no sabe nada del pipeline, no
importa ningún engine y no puede favorecer al modelo de la casa. El mercado no
es el eje del gráfico, es un candidato más — el que arranca arriba.

Por qué importa el orden: en este proyecto la balanza llegó décima. El hallazgo
de que el modelo no le gana al precio (0.24620 contra 0.24052, coeficiente
NEGATIVO condicionado al precio) salió recién el 2026-07-30, de un script suelto
clavado a una corrida y a dos temporadas. Mientras la comparación correcta no
estuvo disponible, "mejor" significó "mejor que mi versión anterior", y así se
sostuvieron siete baselines que fueron cayendo uno tras otro por defectos de
medición.

Solo lectura: nada de este módulo escribe en ninguna base.
"""

from fbq.evaluator.frame import EvalFrame, load_frame
from fbq.evaluator.score import evaluate, Report

__all__ = ["EvalFrame", "load_frame", "evaluate", "Report"]
