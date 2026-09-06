# Control de fuga por invariancia, y el fin de la aproximación de 8 h

> 📌 **La v1.1 quedó superada por la v1.2** el mismo día: la validación del
> contenido de las horas de fin encontró que el instante medido puede quedarse
> corto hasta 15,7 minutos, y se le agregó un margen de 20. Ver
> `docs/INFORME_FINAL_CANDIDATO_V1_2_2026-09-06.md`, que es el informe vigente.
> Este documento se conserva entero: la comparación v1 ↔ v1.1 y el detalle de
> las tres pruebas siguen siendo válidos.

Completa el control de fuga del candidato v1
(`docs/CANDIDATO_V1_RESULTADO_2026-09-06.md`). **Ventana (162), `k` (67) y
regularización (`λ=1.0`) quedaron congeladas durante toda esta corrección**, tal
como se pidió: nada de lo que se toca acá es un hiperparámetro.

## 1. Por qué los dos detectores anteriores no alcanzaban

| detector | qué impide | su hueco |
|---|---|---|
| **estructural** (`model/pit.py`) | que un hecho posterior al corte entre por la compuerta | quien **esquiva** la compuerta no lo despierta |
| **estadístico** (`model/detector.py`) | un candidato implausiblemente bueno (Brier < 0,22) | una fuga **chica** deja el Brier en un rango normal y pasa |

Los dos huecos se solapan justo donde vive el peligro real: una fuga pequeña
introducida sin pasar por la compuerta. Está **demostrado abajo** que esa
combinación atraviesa ambos.

## 2. El detector nuevo: invariancia del proceso completo

`fbq/model/invariancia.py`. No mira el código ni la calidad del modelo: mira si
**la predicción pre-juego depende del resultado del propio partido**. Si cambiar
ese resultado en los datos de origen mueve la predicción, hay fuga — de
cualquier tamaño, esquive o no la compuerta.

**Cómo se perturba**: insertando una **corrección** en `results.observacion`,
que es el mecanismo que el propio almacén tiene para eso (es append-only por
trigger: un UPDATE aborta, y la vista `resultado` resuelve al último registro).
La perturbación entra por la puerta legítima del almacén, no por un atajo.

El nuevo marcador **siempre invierte al ganador y siempre cambia el total**:
`(0, CF+CC+7)` si ganaba el local, `(CF+CC+7, 0)` si ganaba el visitante. Nunca
empata — en MLB no existe un final empatado y `Final` lo rechaza.

> ⚠️ La primera versión sumaba 7 e invertía los lados, y con margen original
> grande **dejaba al mismo ganador**: 3 de 14 partidos no se perturbaban de
> verdad. Un test de fuga que a veces no perturba nada es peor que no tenerlo.
> Corregido y fijado con un test.

Se perturban sólo partidos de la temporada de **evaluación**, así el modelo se
entrena sobre temporadas intactas y queda fijo: si las variables no se mueven,
la predicción tampoco, y la igualdad es **exacta** en vez de aproximada.

## 3. Las tres pruebas

Perturbación: los **27** partidos evaluables del 2025-06-10 y 2025-06-11.

### Prueba 1 — el objetivo no se mueve

| | |
|---|---|
| partidos perturbados | 27 |
| excluidos por contaminación (otro perturbado estaba disponible en su corte) | 13 |
| **comprobables** | **14** |
| con variables movidas | **0** |
| con predicción movida | **0** |
| | **PASA** |

### Prueba 2 — el complemento: los posteriores sí reaccionan

Sin ésta, una construcción que **ignore los datos** —una constante— pasaría la
prueba 1 con nota perfecta.

| | |
|---|---|
| partidos posteriores que usan alguno de los perturbados | **2.835** |
| de esos, con variables movidas | **2.834** |
| exentos por simetría (verificado) | **1** |
| | **PASA** |

**El exento, y por qué no es una falla.** `game_pk` 823037 (2026-06-28, Miami
Marlins @ St. Louis Cardinals) tiene `x = (0,0 · 0,0)` antes y después, aunque
la media de carreras de la liga sí se movió (4,532427 → 4,545607). La razón está
verificada: **los dos equipos tienen perfiles idénticos** —162 partidos, 680
carreras a favor, 739 en contra, sobre listas distintas con sólo 12 partidos en
común— y con perfiles idénticos `pitagorica(local) − pitagorica(visita)` vale
**exactamente 0 para cualquier media de liga**, porque la media entra igual en
los dos lados. La fila es insensible **por construcción**, no por ignorar los
datos. Es 1 de 5.736 filas y el detector la separa y la reporta en vez de
contarla como defecto.

### Prueba 3 — la fuga chica que atraviesa los otros dos detectores

Se inyecta `dif_pitagorica += 0,005 × signo(resultado del propio partido)`,
leyendo el marcador directo del objeto y **esquivando la compuerta**.

| detector | resultado |
|---|---|
| **estructural** | **NO se despierta** — se construyen las 5.736 filas sin una sola excepción |
| **estadístico** | **NO salta** — Brier 2025 **0,237449** y 2026 **0,242894**, ambos muy por encima del umbral 0,22 |
| **invariancia** | **RECHAZA** — 14 de 14 comprobables mueven variables y predicción; ejemplo `game_pk` 777557: `+0,024066 → +0,034066`, exactamente `2 × 0,005` |

**Lo que hace peligrosa a esta fuga**: con ella el modelo mide **0,237449** en
2025, que le gana a Pinnacle (0,241417). Sin el detector de invariancia, ese
número se habría publicado como el primer candidato que le gana al mercado.

## 4. La aproximación de 8 h: qué evidencia la respalda

El preregistro fijó `inicio + 8 h` como cota del fin de un partido, con este
argumento: *"la mediana de un partido de MLB ronda las 3 horas y los extremos de
entradas extra no llegan a 7; ninguna fuente propia guarda la hora de FIN"*.

**Esa evidencia sí existe** y no se había buscado. El feed en vivo publica el
instante de la **última jugada** (`liveData.plays.currentPlay.about.endTime`), y
la API acepta un parámetro `fields` que recorta la respuesta a ~1,3 KB. Se
fecharon los **7.664** partidos de los almacenes, **sin un solo fallo**.

### Lo medido — reloj de pared, inicio → última jugada

| | |
|---|---|
| mediana | **2,69 h** |
| p90 | 3,22 h |
| p99 | 4,96 h |
| p99,9 | 6,73 h |
| **máximo** | **9,04 h** |

| exceden | n | % |
|---|---|---|
| 6 h | 24 | 0,313% |
| 7 h | 3 | 0,039% |
| **8 h** | **2** | **0,026%** |
| 9 h | 1 | 0,013% |
| 10 h | 0 | 0% |

**La cota era mala en las dos direcciones.** Retrasaba de más la disponibilidad
de casi todos los partidos —eso cuesta cobertura— y **no era una cota superior**:
dos partidos la exceden, con un máximo de 9,04 h. Se reemplaza por el fin
medido; la cota queda **sólo como respaldo**, marcada como tal en
`Partido.procedencia`.

### Retrasos

El instante guardado es el de la última jugada, o sea **reloj de pared**, así
que una demora por lluvia ya está adentro. El boxscore lo confirma por separado
en su campo `T`. Los casos extremos son todos demoras:

| game_pk | elapsed | `T` |
|---|---|---|
| 745169 | 9,04 h | `2:59` |
| 747039 | 8,35 h | `2:59 (1:41 delay)` |
| 747059 | 7,58 h | `2:35 (5:00 delay)` |
| 823566 | 6,99 h | `3:24 (3:35 delay)` |

Dos horas y media de juego dentro de siete y media de reloj. **Para un corte
temporal importa el reloj, no el tiempo de juego.**

### Casos inciertos

| situación | tratamiento |
|---|---|
| fin medido | se usa; `procedencia = "medido"` |
| sin fin medido, con inicio | cota `inicio + 8 h`; `procedencia = "cota"`, para poder separarlos en cualquier análisis |
| sin fin ni inicio | **nunca disponible**; `procedencia = "desconocido"` |
| suspendido y reanudado | el fin medido ya corresponde a la reanudación; como respaldo se usa el **inicio más tardío** |

En la descarga del 2026-09-06 los 7.664 partidos trajeron fin, así que hoy no
hay ninguno en los dos últimos estados — pero el camino existe porque un partido
en curso, o uno que la API todavía no publicó, sí lo estará.

Reproducible con `python3 -m fbq.results.fines`.

## 5. Versión v1.1 del resultado

Cambiar la cota por el fin medido cambia las variables, así que el resultado se
versiona. **La población NO cambió** — se verificó, no se supuso:

| | v1 (cota 8 h) | v1.1 (fin medido) |
|---|---|---|
| filas predecibles | 5.736 | **5.736** |
| `sin_precio_pinnacle_pre_juego` | 1.558 | **1.558** |
| `historial_insuficiente` | 370 | **370** |
| evaluables 2025 / 2026 | 2.191 / 1.557 | **2.191 / 1.557** |

534 de 5.736 filas (9,3%) mueven alguna variable —`|Δ dif_pitagorica|` medio
0,00040 y máximo 0,00804— pero ninguna entra ni sale del conjunto evaluable.
Como la intersección es la misma, **los comparadores dan exactamente lo mismo**,
que es la comprobación que corresponde:

### 2025 — n = 2.191

| candidato | v1 | **v1.1** |
|---|---|---|
| v1 (modelo) | 0,243241 | **0,243238** |
| tasa base del entrenamiento | 0,247959 | **0,247959** |
| Pinnacle (nulo, recalculado sobre la intersección) | 0,241417 | **0,241417** |

### 2026 — n = 1.557

| candidato | v1 | **v1.1** |
|---|---|---|
| v1 (modelo) | 0,248113 | **0,248193** |
| tasa base del entrenamiento | 0,249470 | **0,249470** |
| Pinnacle (nulo, recalculado sobre la intersección) | 0,246411 | **0,246411** |

Veredicto del evaluador, sin cambios: **NO aporta sobre el precio** en las dos
temporadas (2025 `b=+0,1929`, IC95 `[−0,2368, +0,5724]`; 2026 `b=+0,0774`, IC95
`[−0,5480, +0,5670]`).

La escalera se mantiene: `tasa base > v1 > Pinnacle`. **La conclusión del
candidato v1 no se mueve**; lo que se movió es la calidad de su cimiento
temporal.

Entregables: `docs/candidato_v1_1_2026-09-06.csv` y `.json`. Los de la v1 se
**conservan** para poder comparar.

## 6. Qué quedó congelado

`VENTANA=162`, `K_REGRESION=67`, `LAMBDA_L2=1.0`, `EXPONENTE_PITAGORICO=1.83`,
`MIN_JUEGOS_PREVIOS=30`, `TOPE_DESCANSO=5` — **ninguno se tocó**. Un test lo
fija (`test_las_constantes_son_las_del_preregistro`).

## 7. Reproducir

```bash
python3 -m fbq.results.fines                       # fecha el fin de cada partido
python3 -m fbq.model --salida docs/candidato_v1_1_2026-09-06
python3 -m pytest tests/test_fbq_model.py -q       # 18 tests
```

Las tres pruebas de invariancia sobre los datos reales corren con
`fbq.model.invariancia.verificar(dias=[...], temporada=...)`; los tests de la
suite las ejercitan sobre almacenes sintéticos, sin red y sin las bases reales.
