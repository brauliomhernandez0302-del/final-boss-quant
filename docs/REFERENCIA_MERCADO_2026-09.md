# Referencia del mercado — almacén propio, 2026-09-06

La barra contra la que se mide cualquier candidato de moneyline. Es un **hecho
medido**, no una conclusión sobre ningún modelo, así que sobrevive a la
invalidación del sistema anterior (`CLAUDE.md` §MODO RECONSTRUCCIÓN).

## La referencia

```bash
python3 -m fbq.evaluator --seasons 2024 2025 2026 --candidato mercado
```

| | |
|---|---|
| **n** | **6.106 juegos** |
| **Brier del mercado (Pinnacle desvigorizado)** | **0,242124** |
| ventaja sobre el azar | +3,150% |
| log-loss | 0,67705 |
| tasa de victoria local | 0,5341 |
| overround medio de Pinnacle | 2,05% (breakeven por lado 2,01%) |
| almacén | `data/market.db` + `data/results.db` — **ningún dato del sistema anterior** |
| autoprueba | brecha candidato − mercado = **+0,00000**, deciles idénticos |

Por temporada:

| temporada | n | Brier | tasa local |
|---|---|---|---|
| 2024 | 2.358 | 0,239950 | 0,5254 |
| 2025 | 2.191 | 0,241417 | 0,5486 |
| 2026 | 1.557 | 0,246411 | 0,5267 |

Mezcla fuera de muestra (pliegues = temporadas enteras): **v0 solo gana en las
tres**. Es lo esperado cuando el candidato ES el mercado; sirve como control de
que el pliegue temporal está bien armado.

## La referencia anterior, conservada aparte

No se reemplaza en silencio. La medición del 2026-08-04, sobre el almacén del
sistema anterior, sigue siendo reproducible:

```bash
python3 -m fbq.evaluator --seasons 2024 2025 2026 --candidato mercado --almacen legado
```

| marco | n | Brier | tasa local | overround |
|---|---|---|---|---|
| legado (`predictions_history.db`) | 5.429 | 0,241560 | 0,5349 | 2,098% |
| propio, sin excluir en vivo | 6.245 | 0,241333 | 0,5331 | 2,103% |
| **propio, excluyendo en vivo** | **6.106** | **0,242124** | **0,5341** | **2,048%** |

El Brier **sube** 0,00056 al pasar del legado a la referencia nueva. No es un
empeoramiento del mercado: son dos correcciones que iban en esa dirección y una
muestra más grande. Ver abajo.

## Control positivo — los dos almacenes dan el MISMO número

Antes de aceptar la referencia nueva hay que probar que el instrumento no
cambió de opinión al cambiar de fuente. Sobre las **5.423 claves comunes**:

| comprobación | resultado |
|---|---|
| max \|Δ p_market\| | **0,000e+00** |
| max \|Δ best_home\| | **0,000e+00** |
| max \|Δ best_away\| | **0,000e+00** |
| resultados `y` idénticos | **sí** |
| Brier legado | **0,241540** |
| Brier propio | **0,241540** |

Mismas claves, mismas cuotas, mismas reglas de desvigorización, mismo número.
El almacén propio reproduce el legado bit a bit.

Las diferencias de tamaño están todas explicadas:

| | n | por qué |
|---|---|---|
| sólo en el legado | **6** | los 6 juegos suspendidos sin hora de inicio desempatable (ver abajo) |
| sólo en el propio | **822** | juegos que `game_outcomes` no tenía: la importación ya no depende de esa tabla para fechar, así que se recuperan |

## Exclusión de precios en vivo

La foto histórica se pidió siempre a las **17:00Z del día del juego**, y para
los partidos que empiezan antes de esa hora la "cotización" es en realidad un
precio **en vivo**.

| temporada | juegos excluidos |
|---|---|
| 2024 | 48 |
| 2025 | 53 |
| 2026 | 38 |
| **total** | **139** — 2,23% de 6.245 |

**Por qué importa, medido**: el Brier del mercado sobre los 139 excluidos es
**0,227976**, contra 0,242124 sobre los que quedan. Un precio en vivo sabe
cosas que un modelo pre-juego no puede saber; dejarlo dentro hacía parecer al
nulo mejor de lo que es, y cualquier candidato se habría medido contra un rival
que ya había visto parte del partido.

No se borran del almacén: un precio en vivo es un dato legítimo de otro
producto. Se excluyen al leer, con `solo_pregame=True`, que es el default.

## Conteos de la importación

```bash
python3 -m fbq.market.importar_historico
```

| | |
|---|---|
| filas de origen (`historical_odds`) | 6.278 |
| **juegos resueltos** | **6.272 (99,90%)** |
| juegos pendientes de conciliar | 6 (0,10%) |
| cotizaciones insertadas | 74.778 |
| cotizaciones ya presentes (2ª y 3ª corrida) | 74.778 |
| cotizaciones posteriores al primer lanzamiento | 161 |
| almacén | 69.121 → **143.899 filas**, estable en tres corridas |
| eventos enlazados a `game_pk` | 25 → **6.295** |

Pares de Pinnacle visibles con el filtro pre-juego, por mercado:

| mercado | par visible pre-juego | sin filtro | excluidos por en vivo |
|---|---|---|---|
| `h2h` | 6.087 | 6.226 | 139 |
| `totals` | 4.550 | 4.653 | 103 |
| `spreads` | 4.550 | 4.653 | 103 |

## Las 6 filas pendientes de conciliar

Juegos **suspendidos y reanudados al día siguiente**: el schedule devuelve dos
entradas terminadas con el mismo día oficial, y ninguna regla las desempata sin
adivinar. Se dejan fuera a propósito — elegir una al azar metería la hora de un
partido que no es el que se cotizó.

| game_pk | día cotizado | partido | horas candidatas (ambas `Final`) |
|---|---|---|---|
| 745180 | 2024-05-22 | Baltimore @ St. Louis | 2024-05-21T23:45Z · 2024-05-22T16:15Z |
| 777861 | 2025-05-21 | Cleveland @ Minnesota | 2025-05-19T23:40Z · 2025-05-21T17:10Z |
| 777623 | 2025-06-07 | Arizona @ Cincinnati | 2025-06-06T23:10Z · 2025-06-07T18:10Z |
| 777294 | 2025-07-02 | Cincinnati @ Boston | 2025-07-01T23:10Z · 2025-07-02T18:30Z |
| 776907 | 2025-08-03 | Atlanta @ Cincinnati | 2025-08-02T23:15Z · 2025-08-03T17:05Z |
| 824912 | 2026-06-17 | San Francisco @ Atlanta | 2026-06-16T23:15Z · 2026-06-17T18:00Z |

**Decisión abierta para el dueño**: si la hora que vale es la del comienzo
original —la defendible, porque es cuando cerró la apuesta— entonces las seis
cotizaciones de las 17:00Z son EN VIVO y quedarían excluidas igual. Confirmarlo
mueve la referencia en, como máximo, 6 juegos sobre 6.106.

## Qué queda listo con esto

El nulo vive sobre almacenes propios, está medido, es reproducible y no depende
de ninguna tabla del sistema anterior. Es lo que hacía falta antes de escribir
el primer candidato de `P(gana el local)` en `fbq/model/`.
