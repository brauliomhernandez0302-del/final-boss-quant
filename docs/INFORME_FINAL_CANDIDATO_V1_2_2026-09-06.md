# Informe final — candidato v1.2 de P(gana el local)

Cierra la línea que empezó con el preregistro `59a39b5`. Reemplaza como
**vigente** a la v1 y la v1.1, que se conservan enteras para poder comparar.

| documento | qué aporta |
|---|---|
| `PREREGISTRO_MODELO_V1_2026-09-06.md` | la configuración, commiteada **antes** de medir |
| `CANDIDATO_V1_RESULTADO_2026-09-06.md` | la v1, con la cota de 8 h |
| `CONTROL_FUGA_INVARIANCIA_2026-09-06.md` | el detector de invariancia y la v1.1 |
| **este** | la validación de las horas de fin y la **v1.2**, vigente |

> ⚠️ **Evaluación HISTÓRICA, no fuera de muestra.** 2025 y 2026 ya fueron
> explorados por este proyecto. El diseño temporal impide que el modelo vea el
> futuro; no impide que lo haya visto quien eligió las variables.

---

## 1. Validación de las horas de finalización

### Campo utilizado

`liveData.plays.currentPlay.about.endTime` del feed en vivo
(`/api/v1.1/game/{game_pk}/feed/live`) — el instante de la **última jugada**. Se
pide con el parámetro `fields`, que recorta la respuesta a ~1,3 KB. Queda
registrado en la caché junto a la fuente y la fecha de descarga.

### Cobertura

| | |
|---|---|
| partidos en `results.db` | **7.664** |
| con fin medido | **7.664 (100%)** |
| fallos de descarga | **0** |
| `procedencia = "medido"` | **7.664** · `"cota"` 0 · `"desconocido"` 0 |

### Auditoría del contenido — no sólo de la cobertura

Reportar cobertura no es validar. Se auditaron los 7.664 registros:

| comprobación | resultado |
|---|---|
| sin fin | **0** |
| sin inicio | **0** |
| fin anterior al inicio | **0** |
| fin anterior a la reanudación (suspendidos) | **0** |
| elapsed < 1 h | **1** — `game_pk` 745180, y es correcto (ver suspendidos) |
| estado del feed fuera de `ESTADOS_FINALES` | **24** — explicado abajo, sin pérdida de datos |
| **elapsed menor que la duración `T`** | **51** — la imprecisión real, cuantificada abajo |

### Suspendidos

8 partidos traen `resumeDateTime`. En los 8 el fin medido corresponde a la
**reanudación**, que es lo correcto: un partido suspendido el día D y terminado
el D+1 no estaba disponible el D.

| game_pk | inicio (reanudación) | fin | `T` |
|---|---|---|---|
| 745180 | 2024-05-22T16:15Z | 2024-05-22T17:01Z | 2:08 (1:31 delay) |
| 746942 | 2024-08-26T18:05Z | 2024-08-26T20:24Z | 2:36 (1:48 delay) |
| 746755 | 2024-08-28T21:10Z | 2024-08-28T23:42Z | 2:31 (1:29 delay) |
| 777861 | 2025-05-21T17:10Z | 2025-05-21T19:16Z | 2:49 (1:42 delay) |
| 777623 | 2025-06-07T18:10Z | 2025-06-07T19:17Z | 2:56 (:48 delay) |
| 777294 | 2025-07-02T18:30Z | 2025-07-02T19:52Z | 2:09 (1:20 delay) |

En ellos `T` es el tiempo de juego de **los dos días**, así que la comparación
`inicio + T` contra el fin **no aplica** — y por eso se excluyen del contraste
de precisión de abajo. Confundirlos habría dado una imprecisión aparente de 108
minutos que no existe.

### Casos sin resolver

| situación | tratamiento | cuántos hoy |
|---|---|---|
| fin medido | se usa, con margen | 7.664 |
| sin fin, con inicio | cota `inicio + 8 h`, marcada `procedencia="cota"` | 0 |
| sin fin ni inicio | **nunca disponible**, `"desconocido"` | 0 |

Hoy no hay ninguno en los dos últimos estados, pero el camino existe: un partido
en curso, o uno que la API todavía no publicó, sí lo estará.

### La imprecisión real, y el margen que la cubre

El instante publicado es el de la última **jugada**, que no es exactamente el
cierre oficial, y `T` está redondeado al minuto. Contrastando el fin contra la
estimación **independiente** `inicio + T` sobre los **7.656 no suspendidos**:

| | |
|---|---|
| mediana de `(inicio+T) − fin` | −1,3 min |
| p99 | +0,8 min |
| casos en que la estimación supera al fin | **302** |
| **máximo** | **+15,7 min** (`game_pk` 778780) |

Se adopta **`MARGEN_FIN = 20 min`** sobre el fin medido. Cubre el máximo
observado con holgura, y **sólo puede costar cobertura, nunca causar una fuga**.

### Sustitución de la cota de 8 h — registro de exclusiones

| | cota 8 h | fin medido + 20 min |
|---|---|---|
| filas predecibles | 5.736 | **5.736** |
| `sin_precio_pinnacle_pre_juego` | 1.558 | **1.558** |
| `historial_insuficiente` | 370 | **370** |

**Ninguna exclusión nueva.** El cambio de criterio no sacó ni metió un solo
juego del conjunto evaluable.

Lo que sí movió, exactamente: **una fila de 3.748**. `game_pk` 824806
(2026-08-05, Angels @ Orioles, corte `21:51:02Z` — una de las 25 capturas
propias en vivo). El partido `824322` terminó a las `21:49:16Z`, **1 minuto y 46
segundos** antes de ese corte: dentro de la banda de incertidumbre de 15,7
minutos, así que no se puede afirmar que estuviera terminado. Con el margen deja
de contar. `dif_pitagorica` pasa de `0,034911758` a `0,034911034` y `p_v1` de
`0,576859893` a `0,576859166`.

El margen no es decorativo y tampoco es caro: movió un caso genuinamente
incierto, en la dirección de la cautela, y nada más.

### Hallazgo lateral: dos granularidades del mismo estado

24 partidos tienen en el feed un estado que `ESTADOS_FINALES` no acepta:
`Completed Early: Rain` (17), `Completed Early: Mercy` (5), `Completed Early:
Wet Grounds` (1), `Final: Tied (won in tiebreaker)` (1). En `results.db` los 24
están guardados como `Completed Early` (23) o `Final` (1), porque **el schedule
y el feed publican el mismo estado con distinta granularidad**.

**No se perdió ningún partido** y nada valida hoy estados del feed contra esa
lista. Queda fijado con un test para que quien conecte esa validación se entere
antes, no después.

### Error corregido en el camino de fallos

`fines.descargar()` guardaba un error con `list.__setitem__(len(lista), r)`, que
levanta `IndexError`: **un solo partido caído tiraba la descarga entera**. No se
disparó porque los 7.664 respondieron. Corregido y fijado con un test que fuerza
el fallo del proveedor.

---

## 2. Control de invariancia

Perturbando los **27** partidos evaluables del 2025-06-10 y 2025-06-11, con el
criterio de disponibilidad de la v1.2:

| prueba | resultado |
|---|---|
| **1 · el objetivo no se mueve** | 14 comprobables (13 excluidos por contaminación cruzada) · **0 variables movidas · 0 predicciones movidas** → **PASA** |
| **2 · el complemento reacciona** | 2.835 posteriores usan alguno · **2.834 se movieron** · 1 exento verificado → **PASA** |
| **3 · fuga chica que esquiva la compuerta** | inyección de **0,005**: la compuerta **no se despierta**, el detector estadístico **no salta** (Brier 0,237449 en 2025, umbral 0,22) y la **invariancia la RECHAZA** — 14 de 14 mueven, Δ = exactamente 2×0,005 → **PASA** |

El exento de la prueba 2 es `game_pk` 823037: sus dos equipos tienen perfiles
idénticos (162 partidos, 680 CF, 739 CC, sobre listas con sólo 12 en común), y
ahí `dif_pitagorica` vale 0 para **cualquier** media de liga. Insensible por
construcción, no por ignorar los datos.

**Lo que hace peligrosa a la fuga chica**: con ella el modelo mide 0,237449 en
2025, que le **gana** a Pinnacle (0,241417). Sin el detector de invariancia ese
número se habría publicado como el primer candidato que le gana al mercado.

---

## 3. Constantes congeladas

`VENTANA=162`, `K_REGRESION=67`, `LAMBDA_L2=1.0`, `EXPONENTE_PITAGORICO=1.83`,
`MIN_JUEGOS_PREVIOS=30`, `TOPE_DESCANSO=5`. **Ninguna se tocó** en ninguna de
las tres versiones; un test lo fija. Lo único que cambió entre versiones es el
**criterio de disponibilidad**, y por eso cada cambio lleva versión propia.

---

## 4. Evaluación temporal, y qué cambió entre versiones

Expansiva: 2024 → 2025, y 2024+2025 → 2026. Población idéntica en las tres
versiones (2.191 y 1.557 evaluables), así que los comparadores están sobre
**exactamente los mismos juegos**.

### 2025 — n = 2.191

| candidato | v1 (cota 8 h) | v1.1 (fin medido) | **v1.2 (fin + 20 min)** |
|---|---|---|---|
| **v1 (modelo)** | 0,243241 | 0,243238 | **0,243238** |
| tasa base del entrenamiento (0,5307) | 0,247959 | 0,247959 | **0,247959** |
| **Pinnacle** (nulo, sobre la intersección) | 0,241417 | 0,241417 | **0,241417** |
| log-loss del modelo | 0,679410 | 0,679403 | **0,679403** |

### 2026 — n = 1.557

| candidato | v1 (cota 8 h) | v1.1 (fin medido) | **v1.2 (fin + 20 min)** |
|---|---|---|---|
| **v1 (modelo)** | 0,248113 | 0,248193 | **0,248193** |
| tasa base del entrenamiento (0,5401) | 0,249470 | 0,249470 | **0,249470** |
| **Pinnacle** (nulo, sobre la intersección) | 0,246411 | 0,246411 | **0,246411** |
| log-loss del modelo | 0,689455 | 0,689616 | **0,689616** |

### Veredicto del evaluador

| temporada | b_candidato | IC95 agrupado por equipo | veredicto |
|---|---|---|---|
| 2025 | +0,1929 | [−0,2368, +0,5724] | **NO aporta sobre el precio** |
| 2026 | +0,0774 | [−0,5480, +0,5670] | **NO aporta sobre el precio** |

### Qué cambió, en una línea

**Nada que mueva una conclusión.** El Brier del modelo se movió en la quinta
cifra decimal; la tasa base y Pinnacle no se movieron en absoluto; el veredicto
es el mismo en las tres versiones. La escalera se mantiene:

> **tasa base > v1 > Pinnacle**

que es exactamente lo que el preregistro declaró que se esperaba, antes de
medir.

Lo que sí mejoró es el **cimiento temporal**: se pasó de una cota inventada que
fallaba en 2 de 7.664 partidos, a una finalización medida en los 7.664 con un
margen derivado de su imprecisión real.

---

## 5. Entregables y versiones

| | |
|---|---|
| detalle por juego | `docs/candidato_v1_2_2026-09-06.csv` — **3.748 filas** |
| resumen | `docs/candidato_v1_2_2026-09-06.json` |
| versiones anteriores, conservadas | `candidato_v1_2026-09-06.*`, `candidato_v1_1_2026-09-06.*` |
| caché de fines | `data/fines_de_juego.json` — 7.664 partidos, 0 fallos |

El CSV lleva por juego: `game_pk`, fecha, temporada, equipos, corte,
`inicio_utc`, las dos variables, los partidos previos de cada equipo, `p_v1`,
`p_tasa_base`, `p_mercado`, el resultado y las tres pérdidas de Brier. El JSON
lleva cobertura, exclusiones por motivo, métricas, veredicto del evaluador con
ROI e intervalos, y las versiones de código y datos.

### Reproducir

```bash
python3 -m fbq.results.fines                                  # fecha el fin de cada partido
python3 -m fbq.model --salida docs/candidato_v1_2_2026-09-06  # entrena, evalúa y exporta
python3 -m pytest tests/test_fbq_model.py -q                  # 22 tests
```

Las tres pruebas de invariancia sobre datos reales:
`fbq.model.invariancia.verificar(dias=[...], temporada=...)`.

---

## Lo que sigue abierto, nombrado

1. **BaseRuns** en vez de la pitagórica — el propio curso del proyecto lo señala
   como mejor estimador de talento. Faltan hits, bases por bolas y bases totales
   en los almacenes propios.
2. **El `k` de regresión para carreras** — se usa 67, que es la constante para
   *victorias*. Sobre-encoge a propósito: medir el correcto exigiría ajustarlo
   sobre los años que se evalúan.
3. **Las 370 exclusiones por historial** desaparecen con una temporada más de
   almacén.
4. **La prueba limpia es 2027**, hacia adelante. Todo lo de acá es histórico.
