# AUDITORÍA PURP — ¿qué pregunta responde cada paso, y la responde bien?

**Fecha**: 2026-07-26 · **Motor auditado**: `b3325a5` (congelado) · **Modo**: 100% read-only
**Única escritura**: este directorio.

El VAL verificó **aritmética**: cada paso ejecuta bien la cuenta que dice ejecutar. Esta auditoría
verifica **propósito**: qué pregunta *debería* responder cada paso, cuál responde su código, y si
coinciden. Donde el VAL ya cerró la aritmética se cita y no se re-verifica.

**Caso que motivó la auditoría** (over/under): el paso integra la distribución correctamente
(VAL-2: CORRECTO) y responde "¿cuál es P(total > línea) **según la forma de mi distribución**?",
cuando debería responder "¿cuál es P(total > línea)?". El evaluador de derivados del 2026-07-26 ya
probó que esa forma es conocida-incorrecta (σ del margen 3.87 vs 4.50 real). Aritmética perfecta,
propósito desalineado. Esa es la clase de hallazgo.

---

## Tabla resumen — por severidad

| # | Paso | Veredicto | Una línea |
|---|---|---|---|
| **PURP-1** | 8 · derivación de runline | **DESALINEADO — CRÍTICA** | Cuando el visitante es favorito, el código precia el evento del *underdog* con el precio del *favorito*: +17.6pp sobre 11 picks reales |
| **PURP-2** | 7→8→10 · forma de la distribución | **DESALINEADO — ALTA** | La forma es −9%/−14% angosta y no hay capa que lo reconozca; envenena total, runline, EV y tier a la vez |
| **PURP-3** | 4 · sesgo multidimensional | **DESALINEADO — ALTA** | Responde "¿cuánto erró el pipeline completo?" y se aplica como si respondiera "¿cuánto vale esta ofensa?" — medido: **cuesta 0.0018 de Brier** |
| **PURP-4** | 10 · EV / Kelly / tier | **DESALINEADO — ALTA** | "ULTRA VALUE" responde "¿qué tan lejos estoy del mercado?", que con un número malo es lo contrario de valor |
| **PURP-5** | 9 · Platt-1D / 2D | **DESALINEADO — MEDIA-ALTA** | La única capa de humildad ante el mercado existe sólo donde el modelo ya no aporta (ML), y falta donde manda el número crudo |
| **PURP-6** | 3+4 · Kalman y sesgo | **DESALINEADO — MEDIA** | El sesgo se estima sobre λ_final y se inyecta en la etapa 2, donde 6 motores lo vuelven a escalar |
| **PURP-7** | 6 · pesos del pipeline | **DESALINEADO — MEDIA, rebajado** | Los pesos live de 2024/2025 son de un motor superado (`pitcher` 1.23), pero **el que corre hoy es el de season=2026: 0.9406** — dentro del rango del backtest. Lo que queda: se entrenó sobre el pool contaminado (n=683) |
| **PURP-8** | 5 · defensa vs parque | **DESALINEADO — MEDIA** | DER/BABIP no está ajustado por parque; el mismo efecto se cobra dos veces |
| **PURP-9** | 1-2 · lineups al momento del pick | **DESALINEADO — MEDIA** | 27 de 28 análisis de hoy corrieron sin lineup confirmado; el mercado sí los precia y el modelo nunca re-precia |
| **PURP-10** | 3 · Kalman vs sesgo (solapamiento) | **PARCIAL — BAJA** | Solapamiento real, ya amortiguado a propósito y empíricamente load-bearing; se documenta, no se toca |
| **PURP-11** | 1 · features de pitcheo | **DESALINEADO — BAJA** | La foul-inflation de barrel% sigue sin corregir del lado de pitchers (residual documentado de MATH-002) |
| — | 5 · pitcher, bullpen, park, HFA, descanso | **ALINEADOS** | Cada uno responde su pregunta con su propio dato; la coordinación pitcher↔bullpen es explícita (`innings_weight`) |

> **Complemento (2026-07-26, segunda pasada)**: `mapa_pasos.md` recorre la cadena real de
> ejecución paso por paso desde el disparador del cron (paso 0) hasta la reconciliación, sin
> agrupar por capa conceptual. Añade 11 **huecos** (`H-1`…`H-11`) que no son desalineaciones de un
> paso existente sino cosas que **ningún** paso cubre — entre ellas la ausencia total de monitoreo
> de calibración en producción (H-11), un camino en el publisher que fabrica picks sin pasar por
> `value_detector` (H-8), y que nada compara λ contra el mercado antes de publicar (H-6).

---

## Los pasos, uno por uno

### Paso 1 — Ingesta de datos / features

**(P)** ¿Qué tan buena es esta ofensa / este pitcheo, en unidades comparables entre equipos?
**(C)** Casi siempre eso. Dos excepciones medidas.
**(¿=?)** ALINEADO con dos huecos.

- El caso `batted_ball_count` (foul-inflation, factor 1.905x) fue **corregido del lado de ofensa**
  en MATH-002. **Sigue sin corregir del lado de pitchers** (`savant_daily_aggregator.py`,
  `pitcher_prior_baseline.py`) — CLAUDE.md lo registra como decisión pendiente del dueño.
  → **PURP-11 (BAJA)**: el barrel% de pitchers responde "¿cuántos batazos duros por *pitch
  trackeado*?" en vez de "por *batted ball terminal*". Misma clase de error que ya se pagó una vez.
- `wrc_plus_approx` se calcula y **el propio código dice** que es "computed for UI display only"
  (`true_talent_engine.py:34`). No es un error: es un feature que no responde ninguna pregunta del
  motor. Va al mapa de datos.

### Paso 2 — λ de talento base (TTE)

**(P)** ¿Cuántas carreras anota este equipo contra un rival neutral, en un parque neutral?
**(C)** Eso mismo, con la salvedad de PURP-9: el λ que llega al pick es el del **roster**, no el
del lineup que va a jugar.
**(¿=?)** ALINEADO en definición, DESALINEADO en operación.

**PURP-9 (MEDIA)** — La capacidad existe: `get_lambda(..., lineup_ids=...)` y `run_module.py:456`
la pasa cuando hay ≥9 bateadores. Pero el cron publica a las **07:00 PDT** y el log de hoy dice 27
veces `"Lineup not yet posted — TTE will use gameday roster"` sobre 28 análisis. O sea: **el 96% de
los picks se toman con λ de roster**, y no hay re-precio cuando el lineup sale. El mercado sí se
mueve con esa información. La pregunta que el paso responde en producción es "¿cuánto anota este
equipo con su plantel?", no "¿cuánto anota el equipo que hoy juega".

*Cambio conceptual*: separar "generar la opinión" de "publicar el pick" — la opinión puede
formarse temprano, pero el precio y la probabilidad deberían recalcularse cuando entra información
que el mercado ya incorporó (lineup, y por extensión clima de última hora). No es un cambio de
motor: es un cambio de *cuándo* se le pregunta.

### Paso 3 — Kalman offense

**(P)** ¿Este equipo está anotando más o menos de lo que su talento implica, recientemente?
**(C)** Eso, sobre `game_outcomes` de la temporada.
**(¿=?)** ALINEADO, con solapamiento conocido — **PURP-10 (BAJA)**.

Kalman y el sesgo leen la misma señal (rendimiento reciente vs esperado). El código **sabe** que se
solapan: `compute_team_bias_kalman_adjusted` amortigua con la fórmula
`adjusted = raw/(1−blend+blend×raw)` precisamente para descontar lo que Kalman ya movió. El
postmortem del 2026-07-11/12 documenta que su justificación teórica declarada es falsa pero que el
mecanismo es empíricamente load-bearing (dos intentos de "arreglarlo" regresaron el Brier:
0.24479→0.24550→0.24636). **Se documenta y no se toca**: cualquier rediseño del sesgo (PURP-3) debe
tratar Kalman+sesgo como una sola capa, no dos.

### Paso 4 — Sesgo multidimensional · **PURP-3 (ALTA)**

**(P)** ¿Cuánto más/menos anota este equipo de lo que mi λ de ofensa predice, con suficiente
muestra para que la corrección sea señal y no ruido?
**(C)** ¿Cuánto se desvió el **resultado real** del **λ FINAL** — el que ya pasó por Kalman, sesgo
previo, pitcher, bullpen, parque, defensa y HFA? (`_l0_ratio` divide por `lambda_final`; el
docstring lo dice explícitamente).
**(¿=?)** **DESALINEADO en tres ejes a la vez.**

1. **Qué mide** (PURP-6, MEDIA): mide el residual del **pipeline completo** y se aplica en la
   **etapa 2**, donde los 6 motores de abajo lo vuelven a multiplicar. Un error de parque termina
   atribuido a la ofensa del equipo, y luego re-escalado por el parque.
2. **Con qué datos** (diagnóstico 2026-07-26): en live 2026 el pool es **563 filas `source='backtest'`
   (82.4%) + 120 live (17.6%)**, y las dos mitades apuntan en direcciones opuestas (backtest ×1.014,
   live ×0.978). Las 563 son el daño irrecuperable de CHRON-001 (`lambda_home == backtest_lambda_home`
   en 563/563) escrito por un motor anterior a cinco rondas de arreglos.
3. **Con qué resolución**: `_BIAS_CLAMP=0.30` y `min_samples=8` permiten un multiplicador de ±30%
   desde 8 juegos. En el canónico hay **93 valores pegados al clamp** y el 2.4% mueve λ más de 20%.

**Lo nuevo de esta auditoría: está medido, y es negativo.** Tres corridas completas del backtest
insignia (4,825 juegos, mismos flags que el canónico, cada una sobre su copia de la DB — evidencia
en `variante_*.json`, driver en `bias_variant_run.py`):

| variante | Brier | ΔBrier | accuracy | logloss |
|---|---:|---:|---:|---:|
| canónico (en disco) | 0.24675 | — | 55.05% | 0.68681 |
| **base (re-corrida, ancla de identidad)** | **0.24675** | **+0.00000** | 55.05% | 0.68681 |
| **(a) sesgo apagado** ≡ `source='live'` | **0.24494** | **−0.00181** | 55.05% | 0.68284 |
| **(b) clamp ±10% + n≥30** | **0.24550** | **−0.00125** | 55.05% | 0.68402 |

El ancla reproduce el canónico **exacto**, así que los deltas son del cambio y no del harness.
**Las dos variantes le ganan al canónico**, y monótonamente: cuanto menos sesgo, mejor Brier. La
accuracy no se mueve en ninguna (el sesgo cambia probabilidades, no el signo de la llamada). Para
calibrar la magnitud: el rebaseline VAL entero movió 0.00025 y el fix del leak de Fase 2B movió
0.00167 — esto es más grande que ambos.

*(Nota sobre (a)≡`source='live'`: 2024 y 2025 tienen **cero** filas live, así que el filtro deja el
pool vacío, cae bajo `min_samples` y el propio código retorna 1.0. Por eso (a) es exactamente
"sesgo apagado", y la variante (c) del pedido —source + clamp— es idéntica a (a) en estas
temporadas: el filtro ya vacía el pool y el clamp no tiene sobre qué actuar.)*

*(ROI: (a) empeora, (b) queda mixto. No lo pondero — a esta n el ROI es ruido por doctrina del
propio proyecto, y además los buckets de edge contienen juegos distintos en cada variante, así que
no son comparables columna a columna.)*

**Pregunta que DEBERÍA responder**: "dado mi λ de ofensa (etapa 2), ¿este equipo anota
sistemáticamente distinto?" — estimada sobre el residual de **esa etapa**, no del pipeline entero,
con n mínimo que soporte el tamaño de corrección permitido. *Cambio conceptual*: una sola capa de
ajuste de ofensa (fusionando Kalman y sesgo, ver PURP-10), estimada y aplicada en el mismo punto de
la cadena, con clamp derivado del error estándar de su propia muestra en vez de una constante fija.
Los números de arriba dicen que, mientras eso no exista, **apagarla es mejor que dejarla como está**.

### Paso 5 — Engines de contexto

**(P)** ¿Cuánto modifica ESTE factor las carreras esperadas, independientemente de los demás?
**(C)** Cada uno responde su pregunta con su propio dato. Un solapamiento real.

- **pitcher ↔ bullpen**: coordinados explícitamente — el bullpen pesa por `innings_weight` derivada
  de las entradas esperadas del abridor. No hay doble conteo: se reparten el juego. **ALINEADO.**
- **pitcher ↔ defensa**: SIERA/xFIP son *defense-independent* por construcción. **ALINEADO.**
- **defensa ↔ parque** — **PURP-8 (MEDIA)**: `defensive_efficiency_engine.py` no menciona parque en
  ninguna línea. Su componente principal es `der_factor = _LG_DER / der_regressed`, y DER = 1−BABIP
  es fuertemente dependiente del parque (jardines grandes → más hits en bolas en juego, sin que el
  fildeo sea peor). El componente OAA sí viene ajustado por dificultad desde Statcast. Resultado: la
  parte DER del multiplicador cobra un efecto de parque que el motor de parque vuelve a cobrar.
  *Debería responder*: "¿qué tan bien fildea esta defensa, neto del parque donde le tocó fildear?"
  *Cambio conceptual*: DER normalizado por parque, o subir el peso de OAA (que ya lo está) frente al
  de DER.
- **park/weather, HFA, descanso**: cada uno con su dato propio y sin lectura cruzada. **ALINEADOS.**

### Paso 6 — Composición λ_pre → λ_final · **PURP-7 (MEDIA)**

**(P)** ¿Cuánto confío en cada motor?
**(C)** Eso, por descenso de gradiente — pero los pesos que **corren en vivo** no son los del motor
que corre en vivo.
**(¿=?)** DESALINEADO por procedencia, igual que PURP-3.

| stage | live (2024) | backtest (2024) | live (2025) | backtest (2025) |
|---|---:|---:|---:|---:|
| pitcher | **1.2297** | 1.0559 | **1.1794** | 0.9593 |
| bullpen | 1.0532 | 0.9394 | 1.0840 | 0.9641 |
| defense | 1.0647 | 1.0443 | 1.1251 | 1.1042 |

Los pesos live se escribieron el **2026-07-17** — antes de Fase 2B, MATH-002/003 y el rebaseline
VAL. El backtest del motor actual dice que `pitcher` debería pesar ~0.96-1.06; en vivo pesa **1.23**.
Y los de 2026 live salen del mismo pool contaminado (n=683 = 563+120).
*Cambio conceptual*: los pesos son parte del motor, no del ledger — deben re-derivarse con cada
re-baseline y promoverse explícitamente, igual que Platt.

### Paso 7 — Simulador → distribución de carreras · **PURP-2 (ALTA)**

**(P)** ¿Cómo se distribuyen realmente las carreras de este juego?
**(C)** ¿Cómo se distribuyen bajo NB(r=6.0) con walk-off truncado y ρ compartido?
**(¿=?)** DESALINEADO en la **forma**, alineado en el **centro**.

VAL cerró la aritmética. Lo que el evaluador de derivados midió sobre 4,825 juegos:

| | simulado | real |
|---|---:|---:|
| σ del total | 4.04 | **4.46** |
| σ del margen | 3.87 | **4.50** |
| PIT del margen | U (colas sobre-pobladas) | uniforme si estuviera calibrado |

El centro quedó bien (total medio 8.850 vs 8.837 real). La **forma** está −9%/−14% angosta, y eso
viene de `NB_DISPERSION=6.0`, elegido **a propósito** por métricas de moneyline pese a un peor
ajuste de cola (está documentado en el docstring del simulador). Es decir: el paso responde
correctamente la pregunta que se le hizo cuando se lo calibró — "¿qué maximiza el moneyline?" — y
esa no es la pregunta que le hacen los mercados derivados.

*Cambio conceptual*: la dispersión no puede ser un solo escalar sirviendo a dos objetivos. O se
calibra la forma contra la realidad y el moneyline se corrige aparte (que es lo que Platt ya hace),
o los derivados llevan su propia capa de corrección (PURP-5).

### Paso 8 — Derivación de mercados

#### 8a. Moneyline — **ALINEADO** (con la salvedad de PURP-5)
**(P)** ¿Cuál es P(gana el local)? **(C)** Eso, y encima una capa que lo reconcilia con el mercado.

#### 8b. Runline — **PURP-1 (CRÍTICA)**

**(P)** ¿Cuál es la probabilidad de que **el lado que estoy preciando** cubra **su** línea?
**(C)** `analyze_runline` (`core/value_detector.py:591-594`):
```python
p_home_cover = np.mean(diff > runline_line)   # local gana por 2+
p_away_cover = np.mean(diff < runline_line)   # local NO gana por 2+
```
`runline_line` llega como **magnitud** — `odds_fetcher.py` la construye con `abs(...)` y su propio
comentario lo dice: *"kept for the live EV path (core/value_detector.py::analyze_runline, frozen),
which still assumes 'home is always the favorite' and doesn't consume a sign"*.

**(¿=?) DESALINEADO cuando el visitante es favorito**: ahí `RL_AWAY` no es "+1.5" sino "−1.5"
(ganar por 2+), pero el código sigue devolviendo `P(diff < 1.5)`, que es el evento del underdog
(~65-70%), y lo multiplica por el precio del favorito. De ahí salen los EV de tres dígitos.

Evidencia sobre los picks reales del ledger:

| caso | evento que calcula el código | ¿es el del pick? | modelo | real | Δ | EV medio |
|---|---|---|---:|---:|---:|---:|
| RL_AWAY, visitante **favorito** (−1.5) | P(diff<1.5) = "no pierde por 2+" | **NO** | 0.6306 | 0.4545 | **+17.6pp** | **58.1%** |
| RL_AWAY, visitante underdog (+1.5) | P(diff<1.5) | sí | 0.6909 | 0.7333 | −4.2pp | 11.3% |
| RL_HOME, local favorito (−1.5) | P(diff>1.5) = "gana por 2+" | sí | 0.4476 | 0.3077 | +14.0pp | 13.2% |

n de 11 a 15 por celda: el ±17.6pp tiene un SE de ~15pp, así que **la evidencia dura es el código**,
y el ledger es corroboración direccional. Dos lecturas adicionales importantes:

- El grupo del evento equivocado **es el que va ganando dinero** (+38.88u). Es la ilustración más
  limpia de por qué el PnL no sirve para auditar propósito: una probabilidad rota puede ser
  rentable por 11 juegos.
- El sesgo también **suprime** picks: `RL_HOME` con local underdog debería calcularse como
  P(diff>−1.5)≈65% y se calcula como P(diff>1.5)≈35% → nunca da EV → **cero picks publicados** de
  esa categoría en 141. La distorsión no sólo infla, también censura.
- La reconciliación **ya fue arreglada** para esto (`f474111`, "grading de RL_HOME/RL_AWAY asumía
  home siempre favorito"). Se arregló cómo se *gradúa* el pick y se dejó cómo se *precia*, porque
  eso vive en el archivo congelado.

*Cambio conceptual*: la probabilidad debe derivarse del **punto firmado del lado que se precia**
(`runline_home_point` / `runline_away_point`, que el fetcher **ya expone**), no de una magnitud con
una convención implícita sobre quién es favorito.

#### 8c. Total — **PURP-2**
**(P)** ¿Cuál es P(total > línea)? **(C)** P(total > línea) **bajo mi forma, que sé angosta**.
Sin capa de humildad, a diferencia del ML. Sesgo medido: +2.46pp en la línea 6.5 y −1.36pp en 11.5
— la firma exacta de una distribución demasiado angosta con el centro bien puesto.

### Paso 9 — Platt-1D y Platt-2D · **PURP-5 (MEDIA-ALTA)**

**(P)** 1D: "¿mis probabilidades están bien calibradas contra la realidad?" · 2D: "¿cuánto vale la
opinión del modelo **una vez que conozco el precio del mercado**?"
**(C)** Exactamente eso, y la respuesta que da es honesta e incómoda. Refit propio sobre los mismos
4,654 juegos con los que está ajustada la que corre en vivo:

| coeficiente | valor | SE | z |
|---|---:|---:|---:|
| b (probabilidad del MODELO) | **−0.097** | 0.120 | **−0.80** |
| c (probabilidad del MERCADO) | **+1.051** | 0.104 | **+10.11** |

b no es distinguible de cero: **dado el precio, la opinión del modelo no aporta información medible
en moneyline**. Con el mercado en 0.55, mover el modelo 30 puntos (0.40→0.70) mueve la decisión
3 puntos, y hacia el otro lado.

**(¿=?)** El paso está ALINEADO — responde bien su pregunta. **La desalineación es la asimetría**:
la única capa de humildad del sistema existe justo donde el modelo ya no aporta (ML), y **no existe**
donde su número crudo manda sin contrapeso (total, runline). El sistema está construido para no
equivocarse donde no puede ganar, y para no protegerse donde apuesta fuerte.

*Cambio conceptual*: una capa análoga por mercado derivado — el mismo tipo de reconciliación
(modelo, precio) → decisión. Con la advertencia de que **debe medirse antes de asumirla útil**: es
perfectamente posible que en runline/total el `b` salga significativamente positivo (el modelo sí
aporta ahí) o también ~0. Ese experimento es barato y no existe hoy porque hasta ayer no se
guardaba el cierre de esos mercados.

### Paso 10 — EV / Kelly / tier · **PURP-4 (ALTA)**

**(P)** ¿Cuánto valor real tiene esta apuesta?
**(C)** ¿Cuánta distancia hay entre mi número y el precio del mercado? — que es lo mismo **sólo si
mi número es bueno**.
**(¿=?)** DESALINEADO por herencia. El EV es aritméticamente correcto (VAL) sobre insumos que los
pasos 4, 7 y 8 entregan sesgados; el tier ordena por esa misma distancia.

Consecuencia observable: `RUNLINE` tiene EV medio **25.2%** y máximo **105.09%**, contra 6.5% de
moneyline — no porque el runline sea más explotable, sino porque es el mercado donde ninguna capa
contradice al modelo (PURP-5) y donde además el evento está mal identificado (PURP-1). "🔥 ULTRA
VALUE" se está asignando, en la práctica, a **los juegos donde el modelo más se aparta del mercado**,
que con un número mal calibrado es un ranking de errores, no de oportunidades.

*Cambio conceptual*: el tier debería incorporar cuánta confianza merece el número en ese mercado
específico (una calibración por mercado, PURP-5), no sólo la magnitud del desacuerdo. Un desacuerdo
grande en un mercado mal calibrado debería bajar de tier, no subir.

---

## Mapa de datos

### 1. Recolectado-pero-nunca-usado (cero consumidores, con evidencia)

| dato | evidencia | nota |
|---|---|---|
| **`savant.batter.rolling`** | `grep -rl` → 2 archivos, ambos **escritores** (`tte_daily_snapshot_builder.py`, `build_offense_savant_rolling_incremental.py`). Cero lectores. | Se pobló por primera vez en julio 2026 "para trabajo futuro de lineup confirmado". Es exactamente el insumo que PURP-9 necesitaría. |
| `wrc_plus_approx` | El propio `true_talent_engine.py:34`: *"is still computed for UI display only"* | Autodocumentado |
| `n_reliever_ids` | 1 archivo (productor) | Exposición pura, por diseño (VAL-7.2) |
| `kelly_unfractional`, `kelly_floor_applied` | 1 archivo (productor) | Exposición por diseño (VAL-4.4) — y hoy causaron el crash del publisher por tipo numpy |
| `pin_vig_pct` | 1 archivo (productor) | Exposición |
| `postponement_risk` | 2 archivos, ninguno lo lee para decidir | Se calcula y no gatea nada |
| Mercados F5 completos | 0 picks F5 en los 141 del ledger | El path se ejecuta cada corrida y nunca produce un pick publicable |

### 2. Debería-medirse-y-no-se-mide (candidatos, por dónde entrarían)

| dato | paso donde entra | por qué movería la aguja |
|---|---|---|
| **Lineup confirmado al momento de publicar** | 1-2 (y re-precio) | El más barato y el más grande: el insumo ya existe (`lineup_ids` funciona), lo que falta es *preguntar más tarde*. Hoy 27/28 picks salen sin él (PURP-9) |
| **Métricas de bateador individual** | 2 (TTE) | `savant.batter.rolling` ya está poblada y sin usar; es lo que convierte "λ de plantel" en "λ de los 9 que juegan" |
| Splits L/R reales | 2 y 5 (platoon) | Hoy el platoon del pitcher usa handedness del lineup rival, no splits medidos |
| Dispersión por juego (no un `r` global) | 7 | La raíz de PURP-2: un duelo de ases y un slugfest no tienen la misma forma |
| Park factor por tipo de batazo | 5 | Separaría el efecto de parque del de defensa (PURP-8) |
| Umpire, catcher framing | 5 | Mencionados en el blueprint; efecto real pero chico comparado con lo de arriba |

### 3. Medido-mal (el proxy no responde la pregunta)

| dato | qué mide | qué debería medir |
|---|---|---|
| **`runline_line`** | magnitud, con "el local siempre es favorito" implícito | el punto **firmado** del lado preciado — **PURP-1** |
| **barrel% de pitchers** | por pitch trackeado (foul-inflado) | por batted ball terminal — residual de MATH-002, sin corregir |
| **`_l0_ratio`** | residual del pipeline **completo** | residual de la **etapa de ofensa** — PURP-3/6 |
| **DER como "defensa pura"** | fildeo + parque mezclados | fildeo neto de parque — PURP-8 |
| **NB_DISPERSION=6.0** | lo que optimiza el moneyline | la forma real de la distribución — PURP-2 |

---

## Síntesis — cuántos problemas de raíz hay realmente

De las 11 desalineaciones, **tres raíces** explican casi todo:

**RAÍZ A — El sistema no tiene una noción de "cuánto vale mi número en ESTE mercado".**
Explica PURP-2, PURP-4, PURP-5 y la mitad de PURP-1. Hay exactamente una capa de humildad
(Platt-2D) y está donde el modelo ya no aporta. Los mercados donde el número crudo manda sin
contrapeso son justo donde se generan los EV de tres dígitos y donde la forma de la distribución
—conocida-angosta— entra sin corrección. Arreglar esto (una calibración por mercado, medida antes
de asumirla) desactiva de golpe el 105% de runline, el sesgo de totales en las líneas extremas y el
tier que premia el desacuerdo.

**RAÍZ B — La capa de aprendizaje responde una pregunta distinta a la que su posición implica.**
Explica PURP-3, PURP-6, PURP-7 y PURP-10. El sesgo mide el error del pipeline completo y se aplica
en la etapa 2; los pesos y el sesgo live se entrenan con procedencias mezcladas; Kalman y sesgo se
solapan y se compensan con una fórmula cuya justificación está documentada como falsa. Y ahora está
medido: **apagar esa capa mejora el Brier 0.0018**, más que cualquier fix de este año. Esta raíz es
la que tiene el número más accionable del reporte.

**RAÍZ C — El pick se toma antes de que exista la información que el mercado ya precia.**
Explica PURP-9 y le pone techo a todo lo demás: por bueno que sea el motor, publicar a las 07:00
con λ de plantel contra un mercado que a las 16:00 ya vio el lineup es competir con información
vieja. Es la única raíz que no requiere tocar el motor.

Un cuarto ítem no es raíz sino **bug de propósito aislado**: PURP-1 (el signo del runline). No
depende de las otras tres, es el más severo por magnitud de error, y es el más acotado de arreglar —
el dato firmado ya está expuesto por el fetcher y sólo hay que consumirlo.

**Orden que sugiere la evidencia** (no es una decisión, es lo que dicen los números): PURP-1 primero
por severidad/costo; RAÍZ B segunda porque ya tiene su medición hecha y su ganancia cuantificada;
RAÍZ A tercera porque es rediseño y necesita el instrumento de cierres de derivados que empezó a
capturar ayer; RAÍZ C en paralelo, porque es operación y no motor.

---

## Qué NO cubre esta auditoría

- No re-verifica la aritmética que el VAL cerró (simulador, EV, devig, composite, fórmula del
  pitcher). Donde aplica, se cita.
- El path F5 se auditó sólo hasta constatar que no produce picks; su lógica interna no se revisó.
- Los deltas del paso 4 son de tres corridas completas y reproducibles, pero las variantes (a) y (b)
  son *puntos*, no una optimización: no se buscó el mejor clamp ni el mejor n.
- Las Δ del ledger (PURP-1) tienen n de 11-15 y SE de ~15pp. La evidencia dura de PURP-1 es el
  código; el ledger corrobora la dirección.
- Nada de esto se ejecutó contra producción: las tres corridas usaron copias de la DB
  (`--db-path`), y el motor congelado no se tocó — `tests/test_engine_freeze.py` sigue en verde.

---

## Revisión independiente (Fable, 2026-07-26) — correcciones aceptadas y hallazgos añadidos

Una segunda pasada independiente verificó los cuatro hallazgos más fuertes contra el código y las
DBs (read-only, sin correr el backtest). Resultado: **PURP-1, el mecanismo de PURP-3, el sesgo
train/serve del clima y el refit de Platt-2D quedaron confirmados**, reproduciendo los números a
3-4 cifras. Corrigió dos cosas y agregó dos hallazgos. Todo lo de abajo lo re-verifiqué yo antes de
incorporarlo.

**Correcciones aceptadas (yo estaba mal):**

1. **PURP-7 / paso 18 — el peso que corre hoy no es 1.23, es 0.9406.** `run_module.py:369` resuelve
   `_season = _current_mlb_season()` y la línea 561 pide `get_pipeline_weights(_season)`; para un
   pick de hoy eso es **season=2026**, cuyo peso live de `pitcher` es **0.9406** (n=683, actualizado
   2026-07-26T14:00) — dentro del rango que el backtest actual respalda (0.92-1.06). El 1.23 que
   citaba es el de season=2024, que solo aplicaría a una predicción en vivo de un juego de 2024.
   **La severidad de PURP-7 baja**: lo que sobrevive del hallazgo no es "los pesos son de otro
   motor" sino "los pesos de 2026 se entrenaron sobre el pool contaminado de PURP-3".
2. **H-8 / paso 32 — el fallback usa probabilidad post-Platt-1D, no cruda.**
   `run_module.py:929-931` sobrescribe `mc_results['p_home']` con el valor calibrado antes de
   guardarlo. Lo que le falta al pick sintetizado sigue siendo Platt-2D, devig-Pinnacle, confianza y
   composite — la severidad no cambia, la redacción sí.

**Hallazgos añadidos:**

3. **PURP-3b (ALTA) — no es solo daño histórico, es un filtro ausente que volverá a contaminar.**
   `_compute_multidim` (`learning_engine.py:1018`) arma su `WHERE` con `season`, `actual_home_runs
   IS NOT NULL` y el equipo — **nunca con `source`**. `prediction_source` solo decide *qué columna*
   de λ leer, no *qué filas* incluir. Consecuencia hacia adelante: cualquier corrida futura de
   backtest que toque la temporada en curso vuelve a contaminar el sesgo live, en silencio. Mi
   reporte lo describía como una característica del dato ("las 563 filas son daño de CHRON-001");
   la caracterización correcta es **un bug de filtro en la función que CHRON-001 tocó y no cerró**.
4. **Hueco nuevo (MEDIA) — el congelamiento del motor no tiene ningún control técnico.**
   `promote_calibration.py` no está en cron y nadie lo invoca, pero **nada impide ejecutarlo**: el
   freeze es enteramente documental y disciplinar. Es exactamente el modo de falla que produjo
   CHRON-001. `tests/test_engine_freeze.py` detecta un cambio de motor *después* de que ocurra; no
   hay nada que lo prevenga.

**Lo que la revisión no pudo verificar** (queda marcado como tal, con su procedencia): el sesgo por
línea de total (+2.46pp en 6.5, −1.36pp en 11.5) sale de mi propia corrida de `derived_eval`
(`results/metrics_20260726_1308.json`, trazable); el `r=0.82` entre sharpe y EV es **una cita del
comentario de `value_detector.py`**, no una medición mía ni suya.

**Coincidencia sobre la prioridad**: preguntada por "una sola cosa antes de volver a publicar", la
revisión independiente eligió **PURP-1**, con el mismo razonamiento y un argumento extra — es el
único de los candidatos que no depende de una estimación estadística con error grande, que ya tiene
un consumidor correcto copiable en el mismo repo (`reconciler.py:148-164`), y que no abre ninguna
pregunta de diseño nueva.
