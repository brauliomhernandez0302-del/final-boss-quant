# Primer componente recuperado — resultado (v1.3)

> 📌 **CERRADO** el 2026-09-06 como experimento sin mejora demostrada. La v1.2
> sigue siendo la referencia. Ver `docs/CIERRE_V1_3_2026-09-06.md`, que además
> separa la asociación observada de la explicación causal por viaje o fatiga:
> el 91,7 % de los visitantes "con día libre" cambiaron de rival, o sea que su
> día libre fue el día de viaje.

**Preregistro**: `docs/PREREGISTRO_COMPONENTE_RECUPERADO_2026-09-06.md`, commit
`5684d32`, escrito **antes** de medir. Nada de lo preregistrado se cambió
después de ver los números.

**La v1.2 queda congelada como referencia.** Esto no la reemplaza: la mide.

> ⚠️ **Evaluación HISTÓRICA, no fuera de muestra.** 2025 y 2026 ya fueron
> explorados por este proyecto.

## El componente

| | |
|---|---|
| archivo | `modules/baseball_module/context_engine/contextual_engine.py` |
| función | `ContextualEngine._rest_days()`, expuesta por `adjust_for_context()` |
| qué hacía | `_B2B_MULT_AWAY = 0.960` sobre λ del visitante en back-to-back; `_B2B_MULT_HOME = 1.000` |

**Motivo deportivo**: un visitante sin día libre llega de un nocturno **más un
viaje**; el local en la misma situación duerme en su casa. La asimetría no es
hipótesis — el sistema anterior la midió y dejó la medición en el código:
visitante en b2b **−4,41 % real**, local en b2b **+7,87 % real**, o sea el signo
opuesto al que su propio modelo suponía, y por eso neutralizó el lado local.

**Entradas**: sólo fechas y horas de inicio. Es el único de los nueve motores
cuyas entradas se reconstruyen enteras con los almacenes propios.

| se reutiliza | se re-ajusta |
|---|---|
| el cálculo: `hours_between < 30`, literal de `data_fetchers.py:1275` | la **magnitud**. El `0.960` se calibró sobre datos que incluyen 2024-2025 — los años de evaluación |
| la **asimetría medida**: sólo el lado visitante entra al modelo | el peso lo aprende la logística **sólo con entrenamiento** |

Un test verifica por AST que ni `0.96` ni `_B2B_MULT` aparecen en el código
nuevo (la nota del módulo sí los cita, para explicar por qué **no** se reutilizan).

## Contrato temporal

El "partido anterior" es el último **disponible al corte**, no el último del
calendario. La fuente de horas de inicio es la caché de fines (7.664 partidos);
la del schedule sola dejaba fuera los 25 juegos de la captura propia en vivo,
porque se armó con el rango de `historical_odds`, que termina el 2026-08-03.
Corregido: **cobertura idéntica a la v1.2**, 5.736 filas y las mismas
exclusiones.

`b2b_visita` vale 1 en el **78,8 %** de las filas — jugar días consecutivos es
la norma en MLB, así que el caso informativo es el contrario: el visitante
**con** día libre.

## Resultado: el componente NO aporta

Mismas filas, mismos cortes, mismas exclusiones.

### 2025 — n = 2.191

| candidato | Brier | log-loss |
|---|---|---|
| tasa base del entrenamiento | 0,247959 | 0,689060 |
| **v1.3** (con el componente) | **0,243279** | 0,679483 |
| **v1.2** (referencia congelada) | **0,243238** | 0,679403 |
| Pinnacle (nulo, sobre la intersección) | 0,241417 | 0,675417 |

### 2026 — n = 1.557

| candidato | Brier | log-loss |
|---|---|---|
| tasa base del entrenamiento | 0,249470 | 0,692088 |
| **v1.3** (con el componente) | **0,248368** | 0,689989 |
| **v1.2** (referencia congelada) | **0,248193** | 0,689616 |
| Pinnacle (nulo, sobre la intersección) | 0,246411 | 0,685983 |

### El aporte, aislado

| temporada | ΔBrier (v1.3 − v1.2) | Δlog-loss | coef `b2b_visita` |
|---|---|---|---|
| 2025 | **+0,000041** | +0,000080 | +0,0470 |
| 2026 | **+0,000175** | +0,000373 | +0,0318 |

**Empeora en las dos temporadas.** Es exactamente lo que el preregistro declaró
que se esperaba, y coincide con el portón fuera de muestra del sistema anterior,
donde los nueve motores dieron las 14 celdas negativas.

**Lo que sí se reproduce es el signo.** El coeficiente de `b2b_visita` sale
**positivo y estable** en los dos ajustes (+0,0470 y +0,0318): un visitante en
back-to-back sube la probabilidad de victoria local, que es la dirección que el
componente afirmaba. El efecto deportivo aparece; lo que no aparece es que
**pague el costo de estimarlo**. Agregar un regresor a un predictor ya casi
óptimo cuesta Brier por error de estimación, y este no cubre ese costo.

### Veredicto del evaluador

| temporada | b_candidato | IC95 agrupado por equipo | veredicto |
|---|---|---|---|
| 2025 | +0,2020 | [−0,2391, +0,6241] | **NO aporta sobre el precio** |
| 2026 | +0,0476 | [−0,5375, +0,5314] | **NO aporta sobre el precio** |

ROI con bootstrap agrupado: los **diez intervalos cruzan el cero** (el mejor,
2026 con edge ≥6 %: +6,96 %, IC95 [−3,22 %, +17,37 %]). Ninguno es evidencia.

### Diagnóstico: ¿se sostiene la neutralización del lado local?

Ajustado **sólo con entrenamiento** y **fuera del modelo**:

| entrena | coef `b2b_visita` | coef `b2b_local` |
|---|---|---|
| 2024 | +0,0774 | −0,0416 |
| 2024 + 2025 | +0,0390 | −0,0099 |

El lado local **no reproduce** el +7,87 % que el sistema anterior midió: acá
sale negativo, y su magnitud se encoge 4x entre los dos pliegues — la firma de
ruido, no de señal. La decisión de neutralizarlo se sostiene, aunque por una
razón distinta de la registrada: allá porque el efecto medido iba al revés del
modelado; acá porque no es estable.

## Diferencias por juego

`docs/candidato_v1_3_2026-09-06.csv` — 3.748 filas con `p_v12`, `p_v13`,
`delta_p`, `brier_v12`, `brier_v13` y `delta_brier`, más las cuatro variables
(incluido el diagnóstico `b2b_local`).

| | |
|---|---|
| \|Δp\| medio | 0,008011 |
| \|Δp\| máximo | 0,034313 |
| juegos donde v1.3 **mejora** | 1.957 |
| juegos donde v1.3 **empeora** | 1.791 |
| ΔBrier medio | **+0,0000967** |

Mejora en más juegos de los que empeora (52,2 %) y aun así el Brier total sube:
cuando se equivoca, se equivoca más caro.

## Controles de invariancia, conservados

| prueba | v1.3 |
|---|---|
| 1 · el objetivo no se mueve | 14 comprobables, **0 movidas** → PASA |
| 2 · el complemento reacciona | 2.834 de 2.835, 1 exento verificado → PASA |
| 3 · fuga chica **inyectada en la variable nueva** | 0,05 en `b2b_visita` esquivando la compuerta: el detector estadístico **no salta** (Brier 0,239375 en 2025, que le gana a Pinnacle) y la **invariancia la RECHAZA**, 14 de 14 → PASA |

La prueba 3 se hizo esta vez sobre el **canal del componente recuperado**, no
sobre el de la v1.2: una pieza nueva abre una vía nueva, y hay que probarla.

## Congelado

`VENTANA=162`, `K_REGRESION=67`, `LAMBDA_L2=1.0` y las dos variables de la v1.2,
sin tocar. Lo único que cambia es que se suma una tercera columna al ajuste.

## Conclusión

**El primer componente recuperado no se paga a sí mismo.** Su efecto deportivo
aparece con el signo correcto y estable, pero el modelo con él es peor que sin
él en las dos temporadas, y el evaluador dice que no aporta sobre el precio.

**La v1.2 sigue siendo la referencia.** El valor de este paso no es haber
mejorado el modelo —no mejoró— sino que ahora existe una **barra y un
procedimiento** para medir una pieza recuperada: preregistro, contrato temporal,
comparación sobre las mismas filas, controles de invariancia y diferencia por
juego. El siguiente motor se mide igual, y ya sin discutir el método.
