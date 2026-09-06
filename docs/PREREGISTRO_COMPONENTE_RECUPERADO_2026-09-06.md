# Preregistro — primer componente deportivo recuperado (v1.3)

**Commiteado antes de medir.** Un preregistro escrito después de ver el
resultado es una racionalización. La v1.2 queda **congelada como referencia**;
esto se compara contra ella, no la reemplaza.

## 1. El componente elegido

| | |
|---|---|
| **archivo** | `modules/baseball_module/context_engine/contextual_engine.py` |
| **clase / función** | `ContextualEngine._rest_days()`, expuesta por `adjust_for_context()` |
| **posición original** | PASO 3 del pipeline, después del motor de abridores |
| **qué hace** | multiplica λ del equipo en *back-to-back*: `_B2B_MULT_AWAY = 0.960`, `_B2B_MULT_HOME = 1.000` |

### Por qué éste y no otro

De los nueve motores del sistema anterior, es **el único cuyas entradas se
reconstruyen enteras con los almacenes propios**: sólo necesita **fechas y horas
de inicio de partidos**, que es exactamente lo que el contrato temporal ya
valida (`fbq/results/fines.py`, `fbq/model/pit.py`). Los demás necesitan
Statcast por jugador, líneas de abridor, uso de bullpen, OAA o coordenadas de
estadio — nada de eso vive hoy en los almacenes propios.

Descartado explícitamente el **factor de parque** (`hfa/park_weather_engine.py`),
que también sería reconstruible: el propio proyecto ya midió que *"park se omite
del análisis: mueve idéntico ambos lados, así que su aporte al moneyline es cero
por construcción"*. Volver a medirlo sería pagar dos veces la misma respuesta.

### El motivo deportivo

Un visitante que juega sin día libre llega de un partido nocturno **más un
viaje**. El local en la misma situación duerme en su casa. La asimetría no es
una hipótesis: el sistema anterior la **midió** y dejó registrada la medición en
el propio código —visitante en b2b **−4,41 % real** contra −4 % modelado, y
local en b2b **+7,87 % real** contra −4 % modelado, o sea el signo opuesto al
que el modelo suponía— y por eso neutralizó el lado local a `1.000` en vez de
borrarlo. Esa asimetría medida es justo lo que un modelo de ganador puede usar;
un efecto simétrico no movería la probabilidad de victoria.

## 2. Entradas, y cómo se reconstruyen

| entrada del componente | reconstrucción propia |
|---|---|
| `back_to_back` del equipo | **`horas_entre_inicios < 30`**, la definición literal de `data_fetchers.py:1275` (`hours_between = (game_dt − prev_dt)/3600`) |
| inicio del partido anterior | `commence_time` del schedule oficial, ya validado |
| qué partido es "el anterior" | el último **disponible al corte** según el contrato de la v1.2 (fin medido + margen de 20 min) |

**El corte manda sobre el calendario**: si el partido de ayer todavía no estaba
disponible al corte de la fila, no se usa — se usa el anterior que sí lo
estuviera. Es lo que el modelo sabía, no lo que pasó.

Se reutiliza el **umbral de 30 horas tal cual**: es un cálculo verificable, no
un parámetro aprendido. Sobre datos propios dispara en ~83 % de los equipo-juego
—jugar días consecutivos es la norma en MLB, no la excepción— así que el caso
raro es el contrario, el visitante **con** día libre.

## 3. Qué se reutiliza y qué se re-ajusta

| | |
|---|---|
| **se reutiliza** | la determinación de b2b (`hours_between < 30`), y la **asimetría medida**: sólo el lado visitante entra al modelo |
| **se RE-AJUSTA** | la magnitud. `_B2B_MULT_AWAY = 0.960` se calibró sobre datos que **incluyen 2024-2025**, o sea los años de evaluación. Reusar esa constante sería meter el futuro por la puerta de al lado |

El coeficiente de la variable nueva lo aprende la logística **sólo con el
pliegue de entrenamiento**, igual que los otros dos.

## 4. La variable nueva

```
b2b_visita = 1  si  (inicio_juego − inicio_del_partido_anterior_del_visitante) < 30 h
             0  en otro caso
```

**Sólo el lado visitante**, que es lo que el componente dice tras su propia
medición. Como diagnóstico —**fuera del modelo**— se reporta el coeficiente del
mismo indicador para el local, ajustado también sólo con entrenamiento, para ver
si la neutralización se sostiene sobre datos propios. No entra a la predicción.

## 5. Lo que queda congelado

`VENTANA=162`, `K_REGRESION=67`, `LAMBDA_L2=1.0`, `EXPONENTE_PITAGORICO=1.83`,
`MIN_JUEGOS_PREVIOS=30`, `TOPE_DESCANSO=5`, y las **dos variables de la v1.2**
(`dif_pitagorica`, `dif_descanso`) tal cual. Lo único que cambia es que se suma
una tercera variable.

## 6. Expectativa, declarada de antemano

**Se espera que NO aporte.** El propio portón fuera de muestra del sistema
anterior pasó los nueve motores —`context` incluido— y **las 14 celdas dieron
negativas**. Además `dif_descanso` ya lleva parte de esta información: la
pregunta real es si la **no linealidad asimétrica en cero** agrega algo sobre la
diferencia lineal de descanso.

Un resultado positivo grande sería motivo de **sospecha**, no de celebración, y
tendría que pasar primero por los controles de invariancia.

## 7. Comparación

Sobre **exactamente los mismos juegos y los mismos cortes**:

| candidato | qué es |
|---|---|
| **v1.3** | v1.2 + `b2b_visita` |
| **v1.2** | la referencia congelada |
| **tasa base** | frecuencia de victoria local del entrenamiento |
| **Pinnacle** | el nulo, desvigorizado, recalculado sobre la intersección |

Se conservan los tres controles de invariancia y se exporta la **diferencia por
juego** entre v1.3 y v1.2.
