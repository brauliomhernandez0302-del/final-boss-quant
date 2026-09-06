# Resumen técnico — proyecto apartado en `~/anterior`

**Generado**: 2026-09-06 · **Alcance**: inspección y documentación, sin modificar
código, datos ni automatizaciones.
**Objeto**: `/home/raulio/anterior` (FINAL BOSS QUANT G8+ y su reconstrucción `fbq/`).
**Estado del árbol inspeccionado**: HEAD `e71f8aa` (2026-08-07), sin cambios de
contenido pendientes (ver §6).

> ⚠️ **Advertencia que gobierna todo lo que sigue.** Desde el 2026-08-05 el
> proyecto está en **MODO RECONSTRUCCIÓN** (`CLAUDE.md` §1, líneas 5-80). Esa
> sección declara sin excepciones que **ningún backtest, ningún motor, ninguna
> calibración y ninguna conclusión de rentabilidad del sistema anterior es
> estado vigente**. Sólo sobreviven (i) los hechos —marcadores y precios
> observados— y (ii) el catálogo de modos de falla. Este informe respeta esa
> declaración: las cifras del sistema viejo aparecen etiquetadas como
> **INVALIDADAS** y se citan como historia, nunca como estado.

---

## 0. Cómo leer este informe

Cada afirmación lleva una de dos marcas:

| marca | significa |
|---|---|
| ✅ **COMPROBADO** | Verificado en esta sesión ejecutando código o consultando los datos. Se indica el comando o la ruta. |
| ⚠️ **PENDIENTE** | Tomado de la documentación del proyecto sin re-verificarlo, o hipótesis no medida. Se indica de dónde sale. |

Todas las rutas son relativas a `/home/raulio/anterior` salvo que se diga otra cosa.
El intérprete usado en todas las corridas es `/home/raulio/mi_entorno/bin/python`
(Python 3.12.3, numpy 2.3.3); el `python3` del sistema **no** tiene pytest ni numpy.

---

## 1. Objetivo: deportes, mercados y funcionalidades que realmente existen

### 1.1 Qué es el proyecto

Un sistema cuantitativo de predicción y detección de valor en apuestas
deportivas. Conviven **dos sistemas en el mismo árbol**, y confundirlos es el
error más fácil de cometer al leer este repositorio:

| | sistema | ubicación | estado declarado |
|---|---|---|---|
| **A** | **FINAL BOSS QUANT G8+** — el sistema original | raíz del proyecto: `app.py`, `modules/`, `core/`, `backtest_and_retrain.py`, `track_record/`, `ui/`, `api/` | **INVALIDADO** por `CLAUDE.md` §1. El código corre; sus conclusiones no valen. |
| **B** | **`fbq/`** — la reconstrucción | `fbq/` | En construcción. Pasos 0-6 escritos; pasos 7-12 vacíos. |

### 1.2 Deportes ✅ COMPROBADO

| deporte | módulo | líneas | desarrollo |
|---|---|---|---|
| **MLB** | `modules/baseball_module/` (47 archivos) | 14.723 | Único pipeline completo: 9 etapas, PIT, calibración, backtest, ledger |
| NBA | `modules/basketball_module.py` | 2.411 | Analizador simple, sin PIT ni backtest |
| UFC | `modules/ufc_module.py` | 626 | Analizador simple, sin PIT ni backtest |

`fbq/` es **sólo MLB**. La captura de precios de `fbq/market` sí barre 9 deportes
(ver §3.3), pero nada los consume.

### 1.3 Mercados ✅ COMPROBADO

| mercado | sistema A (motor viejo) | sistema B (`fbq/`) | histórico de precios |
|---|---|---|---|
| Moneyline (`h2h`) | sí — pipeline completo | evaluador + nulo desvigorizado | 5.429 juegos con par de Pinnacle |
| Total (`totals`) | sí — evaluador de derivados | `evaluator/derived.py` | 4.659 juegos |
| Runline (`spreads`) | sí — evaluador de derivados | `evaluator/derived.py` | 4.658 juegos |
| F5 (primeras 5 entradas) | sí (`tests/test_f5_lambda.py`) | no | no |

### 1.4 Funcionalidades que existen y corren ✅ COMPROBADO

| funcionalidad | entrada | verificación |
|---|---|---|
| UI Streamlit | `streamlit run app.py` (344 líneas) + `ui/mlb.py` | ⚠️ PENDIENTE — no se levantó en esta sesión |
| API HTTP | `api/server.py`, `api/mlb_presentation.py` (402 líneas) | ⚠️ PENDIENTE |
| Publicación diaria de picks | `run_daily_picks.py` (147 líneas) | ⚠️ PENDIENTE — cron desactivado (§6.4) |
| Backtest + reentrenamiento | `backtest_and_retrain.py` (3.485 líneas) | ⚠️ PENDIENTE — no se corrió (coste alto, y su salida está invalidada) |
| Descarga de cuotas históricas | `fetch_historical_odds.py` (791 líneas) | ⚠️ PENDIENTE — consume cuota de The Odds API |
| Track record + CLV | `track_record/` (8 archivos, 2.480 líneas) | ✅ base con 603 picks (§3.5) |
| **Evaluador `fbq`** | `python -m fbq.evaluator` | ✅ **corrido de punta a punta**, ver §4.4 |
| **Captura de precios `fbq`** | `python -m fbq.market.capture` | ✅ 69.121 filas en la base; proceso detenido (§6.4) |
| **Almacén de hechos `fbq`** | `python -m fbq.results.fetch` | ✅ 7.664 finales en la base |
| Portón de features | `python -m fbq.features` | ✅ 3 features medidas, 0 cruzan (§4.5) |

---

## 2. Arquitectura

### 2.1 `fbq/` — la reconstrucción (3.125 líneas, 26 archivos) ✅ COMPROBADO

La regla estructural es que **cada paquete sólo puede depender de los
anteriores**; una violación aparece como import circular, no como sorpresa
tardía (`fbq/__init__.py`).

| paso | paquete | responsabilidad | líneas | almacenamiento | estado |
|---|---|---|---|---|---|
| 0 | `core/` | Contrato de tiempo (`clock.py`) e identidad (`identity.py`) | 254 | — (funciones puras) | ✅ escrito, 16 tests |
| — | `sources/` | Frontera con el exterior: `odds_api.py`, `statcast.py`, `mlb_stats.py`. Devuelven crudo, no interpretan | 390 | — | ✅ escrito, 13 tests |
| 1 | `market/` | Precios append-only con trayectoria: `store.py`, `capture.py`, `link_events.py`, `importar_historico.py` | 976 | `data/market.db` | ✅ escrito, 19 tests |
| 2 | `results/` | Los hechos: quién ganó. `store.py`, `fetch.py` | 327 | `data/results.db` | ✅ escrito, 11 tests |
| 3-4 | `evaluator/` | La balanza. `frame.py`, `score.py`, `derived.py`, `__main__.py` | 751 | sólo lectura | ✅ escrito, 18 tests |
| 5-6 | `features/` | Señales + portón. `base.py`, `gate.py`, `market_shape.py` | 401 | sólo lectura | ✅ escrito, 6 tests |
| 7-8 | `model/` | Probabilidad y detección de valor | **0** | — | ❌ **directorio vacío, sin un solo archivo** |
| 9 | `stake/` | Sizing | **0** | — | ❌ **vacío** |
| 10-11 | `ledger/` | Publicación y reconciliación | **0** | — | ❌ **vacío** |
| 12 | `app/` | UI | **0** | — | ❌ **vacío** |

Verificación de los cuatro vacíos: `ls -A fbq/model fbq/stake fbq/ledger fbq/app`
devuelve nada en los cuatro casos.

#### Las reglas impuestas por el motor (no por disciplina) ✅ COMPROBADO en el esquema

| regla | dónde | mecanismo |
|---|---|---|
| `odds_snapshot` es append-only | `fbq/market/store.py:79-90` | dos triggers `RAISE(ABORT)` sobre UPDATE y DELETE |
| `observacion` es append-only | `fbq/results/store.py:59-69` | idem |
| Un `Final` no se puede construir con estado no-final ni marcador empatado | `fbq/results/store.py:98-113` | `__post_init__` que lanza `ValueError` |
| `pair_before()` devuelve `None` con medio par o puntos que no coinciden | `fbq/market/store.py:329-360` | comprobación explícita antes de devolver |
| `elegir_unico()` se abstiene ante dos candidatos casi empatados | `fbq/core/identity.py:78-118` | `MARGEN_MINIMO = 90 min`, motivo `"ambiguo"` |
| El corte temporal es estricto (`<`, nunca `<=`) | `fbq/core/clock.py:73-83` | `es_anterior()`, comparando instantes y no cadenas |
| Toda barrida queda registrada aunque no cambie ningún precio | `fbq/market/store.py:93-103` | tabla `sweep` |

### 2.2 Sistema A — el original (≈36.000 líneas) ✅ COMPROBADO (tamaños)

| componente | archivos | líneas | responsabilidad |
|---|---|---|---|
| `modules/` | 49 | 17.760 | pipelines por deporte; `baseball_module/` son 47 archivos / 14.723 líneas |
| raíz (`backtest_and_retrain.py`, `data_fetchers.py`, `odds_fetcher.py`, `fetch_historical_odds.py`, `app.py`, `config.py`, `run_daily_picks.py`) | 7 | 9.123 | backtest, fetchers, UI, CLI |
| `scripts/` | 20 | 4.484 | tooling, builders PIT, promoción de calibración |
| `track_record/` | 8 | 2.480 | picks en vivo, captura de cierre, CLV |
| `core/` | 2 | 1.446 | `value_detector.py` — detección de valor compartida entre deportes |
| `ui/` | 5 | 1.065 | `mlb.py` (analizador + render de MLB, importado por `app.py`) |
| `api/` | 3 | 402 | servidor HTTP y capa de presentación |
| `db/` | 2 | 263 | wrappers de SQLite |

#### Interfaz de los módulos de deporte ⚠️ PENDIENTE (según `CLAUDE.md`, no re-verificado por ejecución)

Cada deporte expone `run_module() -> Dict[str, Any]` con las claves `status`,
`game_info`/`fight_info`, `probabilities`, `lambdas_history`, `best_bets`,
`metadata`.

⚠️ **Trampa operativa documentada** (`CLAUDE.md` línea 207): cada llamada a
`run_module()` escribe una fila permanente en `game_outcomes(source='live')`
salvo que se pase `persist=False` o se exporte `FBQ_NO_PERSIST=1`. Se descubrió
en vivo el 2026-07-19 tras escribir 46 filas reales en una tarde de llamadas de
diagnóstico (`audit_20260714/verificacion_operativa/nota_46_rows.md`).

### 2.3 Almacenamiento — mapa completo ✅ COMPROBADO

| base | tamaño | tablas | quién escribe |
|---|---|---|---|
| `data/market.db` | 27 MB | `odds_snapshot`, `sweep`, `event_link` | `fbq/market/` |
| `data/results.db` | 1,1 MB | `observacion` (+ vista `resultado`) | `fbq/results/` |
| `data/predictions_history.db` | 14 MB | `game_outcomes`, `historical_odds`, `predictions`, `results`, `kalman_state`, `ml_state`, `sport_stats` | sistema A (vivo **y** backtest) |
| `data/track_record.db` | 3,3 MB | `picks`, `bankroll`, `daily_snapshots` | `track_record/` |
| `data/pit_cache_merged.db` | 340 MB | `pit_metric_cache` (240.307 filas) | builders PIT |
| `data/pit_cache_pitcher.db` | 378 MB | caché PIT de pitchers | builders PIT |
| `data/pit_cache_2024.db` / `2025.db` | 64 / 62 MB | `pit_metric_cache` (13.044 filas en 2025) | builders PIT |
| `data/pit_raw/raw_savant_*.db` | 8,4 GB | `raw_savant_events` | `scripts/build_raw_savant_events.py` |

**Total de `data/`: 9,8 GB.** Hay además copias de respaldo fechadas
(`*_backup_pre_math002_*`, `*_backup_pre_fase2b_*`, `*_backup_pre_chron002_*`,
`*_backup_pre_outcome_repair_*`) que no se listan una por una.

---

## 3. Datos disponibles

### 3.1 Hechos: resultados de juegos

**`data/results.db`** (almacén de `fbq`, append-only) ✅ COMPROBADO

| temporada | juegos con resultado vigente |
|---|---|
| 2024 | 2.558 |
| 2025 | 2.911 |
| 2026 | 2.195 |
| **total** | **7.664** — cobertura 2024-03-20 → 2026-08-06 |

Observaciones crudas: 7.664 filas en `observacion` (una corrección agregaría una
fila nueva, no pisaría). Fuente: MLB Stats API (`statsapi.mlb.com/api/v1`,
gratis, sin clave).

**`data/predictions_history.db::game_outcomes`** (almacén del sistema A) ✅ COMPROBADO

| temporada | filas | con `home_won` | rango de `official_date` |
|---|---|---|---|
| 2024 | 2.429 | 2.429 | 2024-03-20 → 2024-09-30 |
| 2025 | 2.430 | 2.430 | 2025-03-18 → 2025-09-28 |
| 2026 | 903 | 872 | 2026-03-25 → 2026-08-08 |
| **total** | **5.762** | **5.731** | |

Por `source`: 5.422 filas `'backtest'` y 340 `'live'`. **Esta tabla mezcla hechos
con salida de modelo** (`p_home`, `lambda_*`, `backtest_p_*`,
`stage_factors_json`), que es la estructura exacta que produjo CHRON-001 (§5.2).

### 3.2 Cuotas históricas — **sí las hay, y son el activo principal**

**`data/predictions_history.db::historical_odds`**: 6.278 filas ✅ COMPROBADO

| | 2024 | 2025 | 2026 | total |
|---|---|---|---|---|
| juegos con resultado | 2.429 | 2.430 | 872 | 5.731 |
| **con par de Pinnacle (`ml_*_pin`)** | **2.407** | **2.247** | **775** | **5.429** |
| con mejor precio (`ml_home_best`) | 2.426 | 2.268 | 779 | 5.473 |

Cobertura de derivados sobre las 6.278 filas: **total 4.659**, **runline 4.658**.

Columnas por fila: `ml_{home,away}_{pin,best,cons,open}`, `ml_*_best_bk` (la casa
real del mejor precio, por lado), `n_bookmakers`, `fair_prob_*`,
`total_{point,over,under}_{pin,best}`, `rl_{home_point,home,away}_{pin,best}`.

**Política de captura** ✅ COMPROBADO: `snapshot_ts` es **siempre 17:00:00Z del
día del juego**, sin excepción (6.278 de 6.278 no nulos, rango 2024-03-20T17:00Z
→ 2026-08-03T17:00Z). Es una **foto diaria, no una trayectoria**: no hay línea de
apertura ni de cierre para estos juegos. Fuente: The Odds API (endpoint
histórico, de pago), vía `fetch_historical_odds.py`.

⚠️ **Defecto medido en esta foto — ver §5.4**: 126 de esos 5.429 juegos (2,32%)
ya habían empezado a las 17:00Z, así que su "precio de mercado" es una cotización
**en vivo**.

### 3.3 Precios propios de `fbq` ✅ COMPROBADO

**`data/market.db`**: 69.121 filas de `odds_snapshot`, 113 barridas.

| | valor |
|---|---|
| cobertura temporal | **2026-08-04T19:48Z → 2026-08-07T23:37Z — 4 días** |
| casas distintas | 34 (top: onexbet 3.500, pinnacle 3.200, mybookieag 3.086, bovada 3.011, betrivers 2.956, draftkings 2.800) |
| mercados | `h2h` 33.323 · `totals` 18.904 · `spreads` 16.894 |
| deportes | `baseball_mlb` 48.624 · MLS 4.676 · La Liga 3.090 · EPL 2.878 · Serie A 2.803 · Ligue 1 2.565 · Bundesliga 2.481 · MMA 2.004 |
| pre-juego vs en vivo | 55.869 pre-juego · **13.252 post-inicio (19,2%)** |
| `event_link` | 25 eventos enlazados a `game_pk` |

Esta base es **trayectoria real** (múltiples capturas por evento, deduplicadas
por cambio de precio) — cualitativamente mejor que la foto histórica, pero con
sólo 4 días de historia. **La captura está detenida desde el 2026-08-07** (§6.4).

### 3.4 Statcast crudo ✅ COMPROBADO

**`data/pit_raw/`** — 8,4 GB, sin agregación, una fila por lanzamiento:

| temporada | lanzamientos | tamaño |
|---|---|---|
| 2023 | 720.984 | 2,9 GB |
| 2024 | 711.455 | 2,8 GB |
| 2025 | 712.192 | 2,8 GB |
| **total** | **2.144.631** | **8,4 GB** |

Cada base lleva su `*.manifest.json` con la lista de fechas completadas (la de
2025 arranca en 2025-03-27). Fuente: Baseball Savant (gratis, sin clave, con tope
duro de filas por consulta que `fbq/sources/statcast.py` detecta y hace fallar
ruidoso). **No hay datos de 2026 en `pit_raw`.**

### 3.5 Picks publicados ✅ COMPROBADO

**`data/track_record.db::picks`**: 603 picks, del **2026-07-20 al 2026-08-08**.

| resultado | n | | mercado | n |
|---|---|---|---|---|
| WIN | 268 | | RL_AWAY | 158 |
| LOSS | 267 | | ML_HOME | 115 |
| VOID | 7 | | UNDER | 100 |
| PUSH | 1 | | OVER | 94 |
| sin resolver | 60 | | RL_HOME | 71 |
| | | | ML_AWAY | 65 |

Telemetría: 179/603 con `odds_book`, 403/603 con `clv_pct`, 520/603 con
`closing_pin_home`. **Estos picks NO constituyen evidencia de habilidad** — ver
§5.3.

### 3.6 Resumen de fuentes

| fuente | clave | costo | qué aporta | estado |
|---|---|---|---|---|
| MLB Stats API | no | gratis | schedule, resultados, linescore | ✅ funciona (usada en esta sesión) |
| Baseball Savant | no | gratis | Statcast crudo | ✅ 2,1 M de lanzamientos en disco |
| The Odds API | sí (`.env`) | de pago | cuotas en vivo e históricas | clave presente en `anterior/.env`; cuota compartida con GANICUS |
| OpenWeather | sí | opcional | clima | ⚠️ el backtest corre 100% ciego al clima (`04_leakage_audit.md`) |
| FanGraphs | no | gratis | proyecciones, splits | usado por builders PIT |

---

## 4. Modelos

### 4.1 Qué predice el sistema A ⚠️ PENDIENTE (documentación, no re-ejecutado)

Predice **λ de carreras esperadas por equipo** y de ahí, por simulación, la
probabilidad de victoria local, el total y la runline. El pipeline
(`modules/baseball_module/core/run_module.py`) es secuencial: cada etapa ajusta λ
recibida de la anterior.

| paso | etapa | archivo | qué aporta |
|---|---|---|---|
| — | Datos | `data_fetchers.py` | juego, equipos, pitchers |
| 1 | True Talent Engine | `offense/true_talent_engine.py` | λ base neutral de parque desde xwOBA / barrel% / disciplina |
| 2 | Kalman + sesgo multidimensional | inline + `calibration/learning_engine.py` | tira λ hacia la tasa observada del equipo |
| 3 | Pitchers | `context_engine/pitcher_engine.py` | SIERA/xFIP/xERA/FIP/ERA, forma, historial, platoon, fatiga |
| 4 | Contexto | `context_engine/contextual_engine.py` | descanso, back-to-back, asimetría local/visitante |
| 5 | Bullpen | `context_engine/bullpen_engine.py` | calidad y carga, ponderado por entradas esperadas del abridor |
| 6 | Parque y clima | `hfa/park_weather_engine.py` | factor de carreras del parque, temperatura, viento, lluvia, techo |
| 7 | Defensa | `context_engine/defensive_efficiency_engine.py` | DER (1−BABIP) y OAA |
| 8 | HFA | `hfa/hfa_engine.py` | fatiga de viaje del visitante + corrección uniforme de prob. local |
| 9 | Monte Carlo | `montecarlo/simulator.py` | hasta 5.000.000 de simulaciones Poisson/Binomial Negativa, parada temprana con SE<0.003 |
| 10 | Detección de valor | `core/value_detector.py` | Platt-2D contra el mercado, EV, Kelly, confianza, tiers |

**Entrenamiento y calibración** ⚠️ PENDIENTE: `backtest_and_retrain.py` recorre
temporadas y ajusta; el estado aprendido vive en `ml_state` y `kalman_state`,
separados por `state_source` ('live' vs 'backtest') desde CHRON-002. Un cambio
validado en backtest **no llega a producción** hasta correr
`scripts/promote_calibration.py --confirm`.

**Sizing**: Kelly con `KELLY_FRACTION=0.25` fijo.

### 4.2 Qué predice `fbq` — **nada todavía** ✅ COMPROBADO

`fbq/model/` está vacío. `fbq/` hoy **mide**, no predice. Su único "candidato"
ejecutable es el propio mercado, usado como autoprueba del instrumento.

### 4.3 Cómo se evalúa — el instrumento de `fbq` ✅ COMPROBADO

`fbq/evaluator` puntúa **cualquier** función `juego → probabilidad` contra el
resultado real, siempre contra el mismo nulo: **el precio de Pinnacle
desvigorizado**. El candidato entra por parámetro; el evaluador no importa ningún
engine. Cuatro bloques, en orden de peso para decidir:

1. **La escalera**: moneda → tasa base → candidato → mercado.
2. **La regresión conjunta** `y ~ logit(mercado) + logit(candidato)`, con
   **bootstrap agrupado por equipo** (los errores iid inflan los t entre 5x y 21x
   sobre este dato). Es *la* prueba: si el IC95 del coeficiente del candidato no
   excluye el cero por abajo, el candidato no aporta nada dado el precio.
3. **ROI por umbral de edge** — gate obligatorio, porque Brier y ROI pueden
   moverse en direcciones opuestas.
4. **Calibración por decil**.

### 4.4 Autoprueba del evaluador — **corrida en esta sesión** ✅ COMPROBADO

```
$ /home/raulio/mi_entorno/bin/python -m fbq.evaluator --seasons 2024 2025 2026 --candidato mercado
```

| | valor |
|---|---|
| n | **5.429 juegos** |
| tasa de victoria local | 0,5349 |
| overround medio de Pinnacle | **2,10%** (breakeven por lado: 2,05%) |
| Brier de la moneda (0,5) | 0,25000 |
| Brier de la tasa base | 0,24878 |
| **Brier del mercado** | **0,24156** — ventaja sobre azar **+3,376%** |
| brecha candidato − mercado | **+0,00000** (autoprueba exacta) |
| log-loss | 0,67582 vs 0,67582 |
| coeficiente conjunto | b=+0,4750, IC95 agrupado [+0,4144, +0,5380], P(b>0)=100,0% |

**La autoprueba reproduce la barra exactamente** (brecha 0,00000, deciles
idénticos): el instrumento no se favorece a sí mismo. Éste es el resultado más
sólido que conserva el proyecto.

Mezcla fuera de muestra, por temporada:

| pliegue | n_test | Brier v0 | v0 recalibrado | mezcla | b_cand | veredicto |
|---|---|---|---|---|---|---|
| 2024 | 2.407 | 0,24037 | 0,24104 | 0,24104 | +0,4299 | v0 solo gana |
| 2025 | 2.247 | 0,24065 | 0,24085 | 0,24085 | +0,4804 | v0 solo gana |
| 2026 | 775 | 0,24789 | 0,24788 | 0,24788 | +0,5000 | mezcla gana |

Calibración por decil del mercado (n≈543 por decil):

| predicho | real | | predicho | real |
|---|---|---|---|---|
| 0,3722 | 0,3812 | | 0,5439 | 0,5506 |
| 0,4381 | 0,4567 | | 0,5663 | 0,5506 |
| 0,4728 | 0,4825 | | 0,5914 | 0,5709 |
| 0,4998 | 0,5193 | | 0,6241 | 0,6169 |
| 0,5227 | 0,5193 | | 0,6893 | 0,7011 |

### 4.5 El portón de features y su veredicto ✅ COMPROBADO (en el código y sus notas)

Tres criterios, hay que pasar los **tres**: signo estable fuera de muestra, Brier
que mejora sobre v0, y ROI que no empeora al subir el umbral (sobre pliegues con
`MIN_APUESTAS = 500`).

Medición del 2026-08-04 sobre 5.429 juegos — **ninguna de las tres features
cruza**:

| feature | hipótesis | coeficientes por temporada | veredicto |
|---|---|---|---|
| `desacuerdo_cons_pin` | cuando el consenso se aparta del sharp, el sharp tiene razón | +0,0387 / +0,0755 / +0,0554 — **estable** | **NO CRUZA**: Brier empeora en 2024 y 2025; ROI negativo en los dos años con muestra grande. La señal existe; su magnitud no cubre el vig. |
| `prima_mejor_precio` | una prima grande sobre el sharp es rezago, no oportunidad | −0,059 / −0,055 / −0,043 — **estable, signo correcto** | **NO CRUZA**: no alcanza para el vig |
| `profundidad_mercado` | menos casas = mercado menos eficiente | −0,0662 / +0,0267 / +0,0210 — **cambia de signo** | **NO CRUZA**: firma de señal inexistente |

Las tres quedan registradas con su veredicto, a propósito: borrarlas garantizaría
que alguien las reintente sin saber que ya se midieron.

---

## 5. Auditoría anterior: qué quedó invalidado, por qué y dónde está

### 5.1 Dónde está la documentación ✅ COMPROBADO (existencia de rutas)

| documento | qué contiene |
|---|---|
| `CLAUDE.md` (85 KB) | Guía del repo. §1 = MODO RECONSTRUCCIÓN, que gobierna sobre el resto. El resto es historia. |
| `audit_20260714/` (17 informes numerados + subcarpetas) | Auditoría formal: fugas, matemática, doble conteo, fallbacks, cronología, calibración, mercado, evaluación estadística, operativa, deuda técnica, hoja de remediación y su cierre |
| `docs/AUDITORIA_MLB_2026-07.md` | Auditoría previa del módulo MLB |
| `docs/AUDIT_FINDINGS.md` | Hallazgos consolidados |
| `docs/PROTOCOLO_CLV_V1.md` | Protocolo de medición de CLV y su registro de estado |
| `docs/INVESTIGACION_2026-08-05.md` | Investigación con fuentes que fundamenta la reconstrucción |
| `docs/CURSO_INGENIERIA.md` | Las 8 materias, en orden, cada una con **el fallo real que el proyecto tuvo por no saberla** |
| `docs/FBQ_MASTER_BLUEPRINT.md` | Plano del sistema A |
| `audit_20260714/val_audit/derived_eval/` | Único instrumento que midió calibración de runline/total |

### 5.2 Métricas y componentes INVALIDADOS

**Todo lo de esta tabla está invalidado por declaración explícita**
(`CLAUDE.md` §1). No es que estén en duda: no son estado.

| qué | valor que llegó a declararse | por qué no vale |
|---|---|---|
| **Baseline del motor** | **Brier 0,24624 / accuracy 54,84%** sobre 4.825 juegos, `--season 2024,2025 --use-full-pit`, reporte canónico `audit_20260714/paso10/backtest_canonico_kalman0.json`, fijado el 2026-07-28 | Invalidado en bloque. Además su propia cadena histórica muestra **siete baselines caídos en tres meses**: `0,24242 → 0,24525 → 0,24482 → 0,24483 → 0,24650 → 0,24675 → 0,24624`, cada eslabón derribado por un defecto de **medición**, no de modelo |
| Los 9 motores de λ | — | Invalidados. Medido: las nueve señales correlacionan **0,28-0,65 con el mercado** y sólo **0,05-0,10 con el resultado**. Al pasarlas por un portón el 2026-08-04: **las 14 celdas negativas** |
| `l0` (True Talent Engine de Statcast) | — | Coeficiente **negativo** condicionado al precio; explicaba el **42,5%** de la opinión del mercado. Era un mal duplicado del precio |
| Toda calibración guardada | Platt, Kalman, sesgo de equipo, descenso de gradiente | Ningún estado guardado vale |
| Toda conclusión de rentabilidad | en cualquier mercado | Invalidada |

**Modos de falla concretos que explican por qué** (del catálogo que sí sobrevive):

| id | defecto | magnitud medida | dónde |
|---|---|---|---|
| **CHRON-001** | Hechos y predicciones en la misma tabla; una corrida de backtest sobrescribió filas de predicciones en vivo | **563 filas**, cero recuperables | `audit_20260714/08_chronology_audit.md`, `chron001_forensics_report.md` |
| **Leak V4** | El día del juego salía del timestamp UTC truncado en vez de `officialDate` | **22-24% de los juegos** caían un día adelante | `CLAUDE.md` §Fase 2B |
| **PURP-1** | EV inflado por emparejar la probabilidad de un evento con el precio de otro; el EV inflado agrandaba las apuestas vía Kelly | **508 de 822 unidades arriesgadas (61,8% del capital)** en picks con edge fabricado; stake medio 13,04u vs 3,82u del resto | `docs/CURSO_INGENIERIA.md` §6 |
| Captura de cierre | 12 capturas de precio por día, **una guardada**; el docstring lo describía como feature | — | `fbq/market/store.py:5-12` (la nota que lo documenta) |
| Caché de ofensa congelada | Una caché congelada a mitad de temporada, usada en silencio | **59%** de los juegos del backtest insignia | `docs/CURSO_INGENIERIA.md` §3 |
| `batted_ball_count` | Inflado por contar fouls no terminales, en **dos implementaciones independientes** | **1,905x** | `project_math002` / `05_math_audit.md` |
| Falsos positivos por errores iid | Tres hallazgos "significativos" que desaparecieron al agrupar los errores por equipo | QS% t=+4,79 y "suerte" t=−21 se evaporaron | `docs/CURSO_INGENIERIA.md` §2 |
| Fugas PIT | TTE, pitcher, bullpen y defensa son PIT-safe **sólo** con sus flags `--use-*-pit`; sin ellas el backtest usa Statcast y rosters de **hoy** para juegos del pasado | Alcance limitado a la invocación que omita las flags; el baseline citado sí las pasaba | `audit_20260714/04_leakage_audit.md` |
| Parity clima | El backtest corre **100% ciego al clima** (instanciación comentada) mientras el motor vivo sí lo usa | magnitud **no medida** | `audit_20260714/04_leakage_audit.md` |

### 5.3 El caso especial del CLV — un positivo que era un artefacto

| | |
|---|---|
| Qué se creyó | CLV medio **+1,514%**, 61,4% de picks positivos — el único indicador de habilidad positivo del proyecto |
| Qué era | El CLV se medía contra el **precio crudo**, regalando el vig entero como si fuera habilidad |
| Qué quedó al desvigorizar | **−0,497%**, 40,9% positivos, **t = −0,85** |
| Dónde | `CLAUDE.md` líneas 697-698 |

**Estado del protocolo de CLV** ✅ COMPROBADO en `docs/PROTOCOLO_CLV_V1.md`:

- Congelamiento del motor: **LEVANTADO** desde 2026-07-27.
- **D0 = 2026-07-26, SUPERSEDED el mismo día** (día 1), por dos causas: la
  muestra primaria ML-only se moría de hambre y hubo fuga de captura de
  derivados.
- **No hay muestra primaria abierta.** Los 603 picks de `track_record.db` son
  **shakedown**, descriptivos, no evidencia. Una muestra nueva requiere un
  `PROTOCOLO_CLV_V2` commiteado antes de aplicarse.

### 5.4 🔴 Hallazgo nuevo de esta auditoría: la foto de 17:00Z contiene precios en vivo

✅ **COMPROBADO** — medido en esta sesión cruzando `historical_odds.snapshot_ts`
contra la hora real de primer lanzamiento del schedule oficial de MLB (7.791
juegos descargados de `statsapi.mlb.com`, cero juegos sin hora).

| temporada | juegos con foto **posterior** al primer lanzamiento | de | % |
|---|---|---|---|
| 2024 | 49 | 2.407 | 2,04% |
| 2025 | 53 | 2.247 | 2,36% |
| 2026 | 24 | 775 | 3,10% |
| **total** | **126** | **5.429** | **2,32%** |

Mediana: **45 minutos** después del primer lanzamiento. Máximo: **415 minutos**
(el juego de apertura en Seúl, `game_pk` 745444, empezó 10:05Z y la foto es de
17:00Z — el partido ya había terminado).

**Magnitud del sesgo, medida y honesta**: el Brier del nulo pasa de 0,24156 (todo)
a 0,24168 (limpio); las 126 filas sucias puntúan 0,23651. **El sesgo es
−0,00012** — real, direccionalmente el esperado (un precio en vivo sabe más), y
**pequeño**. No invalida el marco; sí hay que corregirlo, y es una línea de SQL.

⚠️ **PENDIENTE**: no está medido si estas 126 filas afectan las conclusiones del
portón de features (§4.5). Dado el tamaño del sesgo, es improbable que cambien un
veredicto, pero no se comprobó.

### 5.5 🔴 Hallazgo nuevo: `importar_historico.py` fallaría en silencio si se corriera

✅ **COMPROBADO** — reproducido contra la clase `MarketStore` real.

`fbq/market/importar_historico.py` es el script que corta la última dependencia
de datos con el sistema A: importa las 6.278 filas de `historical_odds` al
almacén propio. **Nunca se corrió** (`market.db` sólo tiene capturas del
2026-08-04 al 08-07). Si se corriera hoy, fallaría así:

1. Toma `commence_time` de `game_outcomes.game_date` (línea 62, con el comentario
   *"es el timestamp UTC de inicio"*).
2. **`game_date` NO es un timestamp**: las 5.762 filas tienen longitud 10, o sea
   `'2024-04-24'`. Además es `date(UTC)`, que difiere de `official_date` en los
   nocturnos del oeste (ej. `game_pk` 778213: `game_date`=2025-04-23,
   `official_date`=2025-04-22).
3. `latest_before(..., solo_pregame=True)` filtra con
   `captured_at < commence_time` como **comparación de cadenas en SQLite**:
   `'2024-04-24T17:00:00Z' < '2024-04-24'` evalúa a **0**.
4. Resultado reproducido en una base temporal con dos filas reales:

   ```
   con solo_pregame=True (el default):
      latest_before -> None
      pair_before   -> None
   con solo_pregame=False:
      latest_before -> {'captured_at': '2024-04-24T17:00:00Z', 'commence_time': '2024-04-24', ...}
   ```

**Consecuencia**: las 6.278 filas entrarían a `market.db` y quedarían
**invisibles para toda lectura por defecto**. Es exactamente el modo de falla
silencioso que el proyecto documenta como el peligroso.

5. Además, **805 de 6.278 filas** de `historical_odds` no tienen juego en
   `game_outcomes`, así que `inicio_utc` sería NULL y el script las descartaría
   como `"sin_inicio"` — un 12,8% del histórico perdido sin que nada falle.

**Arreglo**: tomar la hora real del schedule de MLB (gratis, ya demostrado en
esta sesión) en vez de `game_date`. El mismo dato arregla §5.4.

### 5.6 Lo que SÍ conserva evidencia válida

Separado a propósito de todo lo anterior:

| resultado | por qué sobrevive | evidencia |
|---|---|---|
| **Los hechos** — 7.664 finales y 5.429 pares de precios observados | Un hecho no deja de ser cierto porque el modelo que lo rodeaba fuera malo (`CLAUDE.md` §1) | `data/results.db`, `historical_odds` |
| **El mercado tiene Brier 0,24156** sobre 5.429 juegos, con overround 2,10% | Medido en esta sesión con un instrumento que pasa su propia autoprueba | §4.4 |
| **El modelo no le gana al precio** | Es una comparación entre dos números medidos con el mismo instrumento y la misma muestra | 0,24620 (modelo) vs 0,24052 (mercado); coeficiente conjunto **negativo** |
| **Las tres features de forma de mercado no cruzan el portón** | Medición fuera de muestra, con bootstrap agrupado y ROI como gate | §4.5 |
| **El CLV desvigorizado es −0,497%** | La corrección del artefacto es en sí un resultado válido | §5.3 |
| **El catálogo de modos de falla** | No dice si el sistema predice; dice cómo esta clase de sistema se rompe | §5.2 y `docs/CURSO_INGENIERIA.md` |
| **El edge de totales no sobrevive el corte por temporada** | Commit `b274037`, evaluador de derivados | `fbq/evaluator/derived.py` |

---

## 6. Estado operativo

### 6.1 Git ✅ COMPROBADO

| | |
|---|---|
| Raíz del repositorio | **`/home/raulio`** (no `/home/raulio/anterior`) |
| Rama | `feature/point-in-time-rebuild` |
| Último commit | **`e71f8aa`** — 2026-08-07 16:23:17 -0700 — *"docs: curso de ingeniería del proyecto, por materia, con los errores propios"* |
| Ramas | `main`, `feature/point-in-time-rebuild`, `backup-before-secret-cleanup`, + 2 remotas |
| Commits totales | 483 |

**Los últimos 9 commits, en orden inverso:**

| sha | fecha | asunto |
|---|---|---|
| `e71f8aa` | 2026-08-07 | docs: curso de ingeniería del proyecto, por materia, con los errores propios |
| `eb0f797` | 2026-08-05 | docs: constancia de investigación previa a definir las secciones |
| `3c2013d` | 2026-08-05 | docs: MODO RECONSTRUCCIÓN — nada del estado anterior es válido |
| `2dfac46` | 2026-08-04 | fix(fbq/market): las cotizaciones EN VIVO no se mezclan con las pre-juego |
| `b274037` | 2026-08-04 | feat(fbq/evaluator): balanza de derivados — el edge de totales no sobrevive el corte por temporada |
| `9b7c33a` | 2026-08-04 | feat(fbq/features): el portón, y las tres primeras señales medidas |
| `58e3f78` | 2026-08-04 | feat(fbq/sources): statcast — filas crudas, y falla ruidoso al truncar |
| `0208fa6` | 2026-08-04 | feat(fbq/sources): odds_api propio — fbq/ queda autónomo del sistema anterior |
| `66496cb` | 2026-08-04 | refactor(fbq): el sistema en un solo árbol, ordenado por dependencias |

### 6.2 🔴 Cambios pendientes: el proyecto se movió y el movimiento no está commiteado ✅ COMPROBADO

`git status` muestra **440 archivos borrados** y `anterior/` como **no rastreado**.
La causa: el proyecto se movió físicamente de `/home/raulio/` a
`/home/raulio/anterior/` **sin commitear el movimiento**. HEAD sigue teniendo los
442 archivos en la raíz del repo.

**No se perdió trabajo** — verificado comparando hash a hash el árbol de HEAD
contra los archivos en disco:

| | n |
|---|---|
| archivos de HEAD presentes en `anterior/` e **idénticos** | **440** |
| presentes pero **modificados** | **0** |
| **ausentes** en `anterior/` | **2** — `.github/workflows/ci.yml` y `.gitignore`, que quedaron en `/home/raulio/` |

**Riesgos concretos de dejarlo así:**

1. Un `git restore .` o `git checkout .` desde `/home/raulio` **resucitaría el
   proyecto entero en la raíz**, duplicándolo junto a `anterior/`, `beisbol/` y
   `ganicus-datos/`.
2. Un `git add -A` desde la raíz **agregaría 446 archivos bajo `anterior/`** (y
   también `beisbol/` y `ganicus-datos/`, que están igual de no rastreados).
   Las bases `.db` siguen ignoradas por el patrón `*.db`, así que los 8,4 GB de
   Statcast **no** entrarían; sí entrarían los `*.manifest.json` de `pit_raw/`.
3. `.github/workflows/ci.yml` quedó en la raíz apuntando a rutas que ahí ya no
   existen: **el CI corre contra un árbol vacío**.

### 6.3 Pruebas ✅ COMPROBADO — suite completa corrida en esta sesión

```
$ /home/raulio/mi_entorno/bin/python -m pytest -q
1 failed, 1037 passed, 2 skipped, 132 warnings in 151.22s
```

**102 archivos de test, 1.040 tests.** Desglose de los que cubren `fbq/`
(83 tests, **todos pasan**):

| archivo | tests |
|---|---|
| `tests/test_fbq_core.py` | 16 |
| `tests/test_fbq_results.py` | 11 |
| `tests/test_fbq_sources.py` | 13 |
| `tests/test_fbq_features.py` | 6 |
| `tests/test_fbq_derived.py` | 5 |
| `tests/test_market_store.py` | 19 |
| `tests/test_evaluator.py` | 13 |

**El único fallo**:
`tests/test_build_raw_savant_events_script.py::test_pit_raw_directory_and_database_are_gitignored`
(`assert 1 == 0`, línea 66). **Causa comprobada**: el test verifica que
`data/pit_raw/` esté git-ignorado. El `.gitignore` (que quedó en `/home/raulio/`)
tiene el patrón `data/pit_raw/`, que ya no coincide con la ruta nueva
`anterior/data/pit_raw/`. Los `.db` siguen cubiertos por `*.db`; **los
`*.manifest.json` quedaron expuestos**. Es decir: **el único test que falla está
haciendo exactamente su trabajo** y señala la consecuencia real de §6.2.

### 6.4 Automatizaciones — todas desactivadas ✅ COMPROBADO

El `crontab` activo del usuario apunta **exclusivamente a `/home/raulio/beisbol`**
(GANICUS): 10 entradas de captura de props, resultados, prospectiva, simulación,
supervisión y respaldo. **Ninguna toca `~/anterior`.**

Las automatizaciones del proyecto apartado están archivadas, inactivas, en
`crontab.anterior.txt`:

| horario | comando | qué hacía |
|---|---|---|
| `0 7 * * *` | `run_daily_picks.py` | publicación diaria de picks |
| `0 13 * * *` | `run_daily_picks.py --publish-only` | segunda publicación |
| `15 7`, `30 19` | `scripts/cron_health_check.py` | chequeo de salud |
| `50 8-19 * * *` | `track_record.capture_closing_lines` | captura de cierre (sistema A) |
| `51 8-19 * * *` | **`fbq.market.capture`** | captura de precios propia |
| `5 20 * * *` | **`fbq.market.link_events`** | identidad evento↔juego |
| `30 23 * * *` | **`fbq.results.fetch`** | hechos del día |

**Consecuencia medida**: la última captura de `fbq/market` es del
**2026-08-07T23:37Z** y el último hecho de `fbq/results` es del **2026-08-06**.
Hay **un mes de hueco** en la serie de precios propia, y sigue creciendo.

⚠️ **Costo de reactivar**: `fbq/` es autónomo y no comparte caché con el fetcher
viejo, así que sus llamadas se pagan aparte: **6 créditos por barrida × 12
barridas = 72 créditos/día** sólo con MLB (`fbq/README.md` §Cuota). Esa cuota es
**la misma** que hoy consume la captura de props de GANICUS.

### 6.5 Errores conocidos, consolidados

| # | error | severidad | verificación |
|---|---|---|---|
| 1 | El movimiento del proyecto no está commiteado; CI apunta a la raíz vacía | alta (operativa) | ✅ §6.2 |
| 2 | `importar_historico.py` haría invisible todo lo que importe | alta | ✅ §5.5 |
| 3 | 126 juegos (2,32%) del marco de evaluación llevan precio en vivo | media | ✅ §5.4 |
| 4 | `evaluator/` y `features/` leen del almacén del sistema A, no del propio | media (dependencia) | ✅ §7.2 |
| 5 | 805 filas de `historical_odds` sin juego asociado | media | ✅ §5.5 |
| 6 | El test de `.gitignore` falla | baja (síntoma de #1) | ✅ §6.3 |
| 7 | `run_module()` escribe en el ledger salvo `persist=False` | alta si se toca | ⚠️ documentado, no re-probado |
| 8 | El backtest corre ciego al clima; el motor vivo no | media | ⚠️ documentado, magnitud no medida |
| 9 | `engine_commit` `b3325a5` del protocolo CLV no es alcanzable desde ninguna rama; el test de congelamiento hace **skip** en vez de fallar | alta si se re-arma el freeze | ⚠️ documentado y verificado el 2026-08-02, no re-verificado hoy |

---

## 7. Recuperación

### 7.1 Aprovechable tal como está

| activo | por qué | evidencia |
|---|---|---|
| **`fbq/core/`** — contrato de tiempo e identidad | 254 líneas, 16 tests, reglas impuestas por funciones; el umbral de 90 min sale de medir 431 fechas reales de doubleheaders | ✅ tests pasan |
| **`fbq/market/store.py`** — almacén append-only | Triggers, punto firmado pegado a su lado, `pair_before` que se niega ante medio par, filtro pre-juego | ✅ 19 tests pasan |
| **`fbq/results/`** — almacén de hechos | `Final` no se puede construir mal; 7.664 finales cargados | ✅ 11 tests pasan |
| **`fbq/evaluator/`** — la balanza | **Pasa su propia autoprueba con brecha exacta 0,00000** | ✅ §4.4 |
| **`fbq/features/gate.py`** — el portón | `MIN_APUESTAS=500` derivado de que ruido puro cruzaba con ~580 apuestas | ✅ 6 tests pasan |
| **`fbq/sources/`** — los tres fetchers | Devuelven crudo; Savant falla ruidoso al truncar | ✅ 13 tests pasan |
| **Los datos de §3** | Hechos y precios observados: 7.664 finales, 5.429 pares de precios, 2,1 M de lanzamientos | ✅ contados |
| **`docs/CURSO_INGENIERIA.md` + `INVESTIGACION_2026-08-05.md`** | El registro de los fallos propios por materia. Un curso genérico está en cualquier lado; éste no | ✅ leídos |

### 7.2 Necesita reconstrucción o arreglo

| qué | por qué | tamaño estimado |
|---|---|---|
| `fbq/model/`, `stake/`, `ledger/`, `app/` | **Vacíos.** No existe ningún modelo en la reconstrucción | el grueso del trabajo |
| `evaluator/frame.py` y `features/market_shape.py` | Leen de `data/predictions_history.db` (sistema A), la tabla que mezcla hechos con salida de modelo. La autonomía que declara `fbq/README.md` **es cierta para `sources/`, no para los datos de `evaluator/` y `features/`** | 1 función nueva de carga |
| `importar_historico.py` | Bug §5.5: `commence_time` inservible | ~10 líneas |
| Filtro de precios en vivo en el histórico | §5.4 | 1 cláusula WHERE, una vez que haya `commence_time` real |
| El movimiento sin commitear + CI | §6.2 | 1 commit y mover 2 archivos |
| Todo el sistema A (≈36.000 líneas) | Invalidado por declaración. Se conserva como catálogo de modos de falla | no se reconstruye: se consulta |

### 7.3 Primer objetivo verificable propuesto

**Objetivo**: *un candidato `juego → P(gana el local)` para partidos completos,
construido sólo con hechos ya en disco, puntuado por el evaluador contra el
mercado desvigorizado, y pasado por el portón.*

Es deliberadamente modesto. **El criterio de éxito no es ganarle al mercado**
—casi con certeza va a perder— sino tener por primera vez en el proyecto la
cadena completa `datos propios → modelo → medición honesta` cerrada y auditable,
con un número reproducible. La lección más cara del proyecto es que la balanza
llegó décima; ésta es la oportunidad de que llegue primera.

**Por qué esta elección**:

- **No cuesta cuota**: `results.db` ya tiene 7.664 finales con fecha oficial. No
  hace falta ninguna llamada de pago.
- **Es PIT-safe por construcción**: un modelo de fuerza de equipo caminando hacia
  adelante sólo puede mirar juegos anteriores al corte, y `core/clock.py` ya
  impone ese corte.
- **Es el control positivo que falta**: si el evaluador no detecta señal en un
  modelo de fuerza de equipo —que sabemos que tiene *algo*, aunque menos que el
  precio— el instrumento está roto.

**Secuencia, con criterio de salida en cada paso:**

| # | paso | criterio de salida |
|---|---|---|
| 0 | Commitear el movimiento del proyecto y devolver `.gitignore`/CI a su sitio | `git status` limpio; los 1.040 tests pasan, incluido el de `.gitignore` |
| 1 | Arreglar `importar_historico.py`: tomar la hora de inicio del schedule de MLB en vez de `game_outcomes.game_date` | Un test que inserte una fila histórica y la recupere con `pair_before(solo_pregame=True)` |
| 2 | Correr la importación | `market.db` pasa de 4 días a 2024-2026; `pair_before` devuelve par para ≥5.400 juegos |
| 3 | Excluir del marco los juegos cuya foto es posterior al primer lanzamiento | El marco baja de 5.429 a ~5.303 juegos, y queda registrado por qué |
| 4 | Nueva carga de `EvalFrame` desde `fbq.market` + `fbq.results`, sin tocar `predictions_history.db` | **Control positivo**: la autoprueba del mercado sobre el almacén propio reproduce n y Brier del §4.4 dentro de la tolerancia explicada por el paso 3 |
| 5 | `fbq/model/` — el candidato más simple que puede funcionar: fuerza de equipo caminando hacia adelante (Bradley-Terry o Poisson de carreras), ajustado sólo con resultados anteriores al corte | Produce `{game_pk: p_home}` para ≥90% del marco, sin mirar un solo juego posterior al corte |
| 6 | Puntuarlo con `fbq.evaluator` y pasarlo por `features.gate` | **Un número reproducible y un veredicto escrito**, gane o pierda |

**Qué NO hacer en esta etapa**, por lección pagada: no reactivar la captura de
`fbq/market` mientras GANICUS necesite la cuota (§6.4); no reconstruir motores de
λ antes de que la balanza esté sobre almacén propio; y no declarar ningún
baseline hasta que el paso 4 haya pasado su control positivo.

---

## Anexo: comandos de verificación usados

```bash
cd /home/raulio/anterior
PY=/home/raulio/mi_entorno/bin/python      # el python3 del sistema no tiene numpy ni pytest

# Suite completa
$PY -m pytest -q

# Autoprueba del evaluador (§4.4)
$PY -m fbq.evaluator --seasons 2024 2025 2026 --candidato mercado

# Inventario de datos (§3)
$PY -c "import sqlite3;c=sqlite3.connect('file:data/market.db?mode=ro',uri=True);print(c.execute('select count(*),min(captured_at),max(captured_at) from odds_snapshot').fetchone())"
$PY -c "import sqlite3;c=sqlite3.connect('file:data/results.db?mode=ro',uri=True);print(c.execute('select season,count(*) from resultado group by season').fetchall())"

# Estado de git (§6.2) — ojo: la raíz del repo es /home/raulio, no anterior/
cd /home/raulio && git status --porcelain | awk '{print $1}' | sort | uniq -c
```

---

*Informe de inspección. No se modificó ningún archivo de código, ninguna base de
datos ni ninguna automatización de `~/anterior` ni de `~/beisbol`.*

---

# Addendum — 2026-09-06, después de la recuperación

Este addendum no reescribe nada de arriba: corrige lo que quedó incompleto y
registra lo que cambió. El informe original se conserva tal como se emitió.

## Corrección a §6.2 — la comparación era en una sola dirección

§6.2 verificó que los 442 archivos de HEAD estuvieran íntegros en `anterior/`
(440 idénticos, 0 modificados, 2 no movidos) y concluyó **"no se perdió
trabajo"**. Esa comparación iba de HEAD hacia el disco y por eso no podía ver lo
contrario: archivos que estaban en el disco y **nunca habían entrado a git**.

Los había:

| archivo | qué era |
|---|---|
| `fbq/market/importar_historico.py` | 179 líneas — el script que corta la dependencia de datos con el sistema anterior |
| `crontab.anterior.txt` | las ocho automatizaciones archivadas |
| `RESUMEN_PROYECTO_ANTERIOR.md` | este informe |
| `tatus` | basura: la salida de un `git status` mal tipeado. No se recupera |

Un clon limpio no los habría tenido. Recuperados a git en el commit `d2966e4`.

## Estado de los errores de §6.5

| # | error | estado |
|---|---|---|
| 1 | El movimiento no está commiteado; CI apunta a la raíz vacía | **resuelto en la rama `recuperacion/fbq`**: el worktree tiene el proyecto en la raíz, con `.gitignore` y `ci.yml` recuperados desde git. El árbol de `~/anterior` queda como estaba |
| 2 | `importar_historico.py` haría invisible todo lo que importe | **corregido** (`83824f2`) y verificado con la importación real |
| 3 | 126 juegos con precio en vivo en el marco | **corregido** (`15e1d81`). El conteo definitivo sobre el marco propio es **139 de 6.245** |
| 4 | `evaluator/` lee del almacén del sistema anterior | **corregido** (`15e1d81`): `--almacen propio` es el default; el legado se conserva para comparar |
| 5 | 805 filas de `historical_odds` sin juego asociado | **recuperadas**: la importación ya no depende de `game_outcomes` para fechar |
| 6 | El test de `.gitignore` falla | **pasa** en la rama de recuperación |
| 7, 8, 9 | `persist=False`, parity de clima, `engine_commit` inalcanzable | **sin tocar** — son del sistema A, invalidado |

## Hallazgo nuevo durante la recuperación

`mlb_stats.schedule()` con un rango de más de un año **se truncaba en
silencio**: 2024-03-20 → 2026-08-03 devolvía 3.023 juegos, cortados en
2025-03-20, con HTTP 200 y sin aviso. En tramos devuelve 7.816. Corregido en
`df3aefd`. Es el mismo modo de fallo que el tope de filas de Savant, y aparece
en §5 del curso como "el proveedor entrega las primeras N y calla".

## Referencia vigente

La barra del mercado ya no sale del sistema anterior:
**n=6.106, Brier 0,242124**, sobre `market.db` + `results.db`.
Detalle completo, con la referencia anterior al lado y el control positivo que
autoriza el cambio, en `docs/REFERENCIA_MERCADO_2026-09.md`.
