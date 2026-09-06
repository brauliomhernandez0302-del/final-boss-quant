"""fbq/model/recuperado.py — el primer componente deportivo traído del sistema anterior.

Origen: `modules/baseball_module/context_engine/contextual_engine.py`, clase
`ContextualEngine`, método `_rest_days()`, expuesto por `adjust_for_context()`.
Preregistro: `docs/PREREGISTRO_COMPONENTE_RECUPERADO_2026-09-06.md`.

## Qué hacía allá

Multiplicaba la λ de un equipo en *back-to-back*:

    _B2B_MULT_AWAY = 0.960   # −4 % al visitante
    _B2B_MULT_HOME = 1.000   # local neutralizado

La asimetría **no es una hipótesis**: el sistema anterior la midió y dejó la
medición escrita en el propio código — visitante en b2b **−4,41 % real** contra
−4 % modelado, y local en b2b **+7,87 % real**, o sea el signo OPUESTO al que el
modelo suponía. Por eso el lado local quedó neutralizado a 1.000 en vez de
borrarse: el efecto existe, pero va para el otro lado y se lo atribuyó a un
confundido de gira en casa.

## Qué se reutiliza acá, y qué no

**Se reutiliza el cálculo**, que es verificable y no aprendido: un equipo está en
back-to-back si entre el inicio de su partido anterior y el de éste pasaron
**menos de 30 horas**. Es la definición literal de `data_fetchers.py:1275`
(`hours_between = (game_dt - prev_dt).total_seconds()/3600`;
`back_to_back = hours_between < 30`). Se copia el umbral tal cual.

**Se reutiliza la asimetría**: sólo el lado visitante entra al modelo.

**NO se reutiliza la magnitud.** El `0.960` se calibró sobre datos que incluyen
2024 y 2025 —los años de evaluación—, así que traerlo sería meter el futuro por
la puerta de al lado. La logística aprende el peso de la variable **sólo con el
pliegue de entrenamiento**.

## Contrato temporal

El "partido anterior" es el último **disponible al corte** de la fila, no el
último del calendario. Si el de ayer todavía no había terminado cuando se
observó el precio de referencia, no cuenta. Es lo que el modelo sabía, no lo que
pasó.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from fbq.model.pit import Partido

# Umbral del sistema anterior, copiado sin cambios desde `data_fetchers.py:1275`.
# No es un parámetro aprendido: es la definición de "back to back" que ese
# sistema usaba, y cambiarla sería medir otra cosa.
HORAS_B2B = 30.0


def _instante(valor: str) -> datetime:
    return datetime.fromisoformat(str(valor).replace("Z", "+00:00"))


def en_back_to_back(
    previos: Sequence[Partido],
    inicio_juego: Optional[str],
    inicios: dict,
) -> Optional[float]:
    """1.0 si el equipo llega en back-to-back; 0.0 si no; None si no se puede saber.

    `previos` son los partidos del equipo YA DISPONIBLES al corte, en orden.
    `inicios` mapea `game_pk` → instante de inicio.

    Devuelve None —y el llamador excluye la fila— cuando falta el inicio de este
    partido o el del anterior. No se rellena con 0: "no sé si venía en b2b" y
    "no venía en b2b" son cosas distintas, y el sistema anterior ya pagó caro
    confundir un dato ausente con un dato neutro.
    """
    if not inicio_juego or not previos:
        return None
    anterior = inicios.get(previos[-1].game_pk)
    if not anterior:
        return None
    horas = (_instante(inicio_juego) - _instante(anterior)).total_seconds() / 3600.0
    return 1.0 if horas < HORAS_B2B else 0.0
