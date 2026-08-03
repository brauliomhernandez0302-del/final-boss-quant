# MAPA DETALLADO DE PASOS — del cron al pick reconciliado

Complemento de `reporte.md`. Aquí no se agrupa por "capa conceptual" sino por **la cadena real de
ejecución**, leída del código de orquestación (`run_daily_picks.py` → `track_record/publisher.py`
→ `modules/baseball_module/core/run_module.py` → `core/value_detector.py` → `track_record/`), no de
memoria. Cada paso lleva la tripleta **(P)** pregunta que debería responder · **(C)** la que su
código responde · **(=)** veredicto.

Numeración: **A** antes del modelo · **B** datos del juego · **C** construcción de λ · **D**
distribución · **E** mercados · **F** decisión · **G** publicación · **H** post-juego.

Los hallazgos ya reportados se citan como **PURP-n**. Los nuevos de esta pasada son **H-n**
(huecos) y están consolidados al final.

---

## FASE A — Antes de que exista una opinión

### A0. El disparador · **el verdadero paso 0**
**(P)** ¿Cuándo debo formarme una opinión sobre los juegos de hoy?
**(C)** "A las 07:00 PDT, siempre" (`crontab`: `0 7` publica, `0 13` re-publica, y tres barridas de
cierre a 09:15/15:30/18:30).
**(=) DESALINEADO — es la raíz C del reporte.** La hora no se eligió por disponibilidad de
información sino por conveniencia operativa. A las 07:00 no hay lineups (27/28 hoy), y las barridas
de cierre de 15:30 y 18:30 caen **después** del primer pitcheo de la mayoría de la cartelera.
Ningún paso pregunta "¿ya existe la información que necesito?" antes de opinar.
→ **H-1**: no hay disparador por evento (lineup publicado, línea movida), sólo por reloj.

### A1. Ventana de juegos
**(P)** ¿De qué juegos me toca opinar?
**(C)** `get_todays_games(hoy)` + `get_todays_games(mañana)` (`publisher.py:189-190`).
**(=) ALINEADO con un efecto colateral medido.** Incluir mañana duplica la oportunidad de publicar,
pero los juegos de mañana casi nunca tienen mercado posteado — de ahí que el 57% de las filas live
salgan sin pins (medido el 2026-07-26 al certificar V1). No es un bug: es que el paso responde
"¿qué juegos existen?" y no "¿de qué juegos puedo opinar con precio?".

### A2. Filtro de anticipación (`MIN_LEAD_MINUTES`)
**(P)** ¿Este pick puede demostrarse pre-juego?
**(C)** Exactamente eso, y desde el fix del 2026-07-19 (antes leía una clave que MLB nunca
puebla y el gate **nunca disparaba**) sí funciona.
**(=) ALINEADO.**

### A3. Resolución de identidad juego ↔ evento de odds
**(P)** ¿Cuál evento del mercado corresponde a ESTE juego?
**(C)** Coincidencia por **substring de nombres** de ambos equipos + ventana de **±6h** contra
`commence_time`, con desempate por cercanía; `{}` si hay empate exacto (`odds_fetcher.py:588-598`).
**(=) DESALINEADO — riesgo estructural, no observado aún.**
- En un **doubleheader** los dos juegos caen dentro de ±6h del mismo par de nombres. El desempate
  por cercanía elige uno; si la diferencia es chica, es una moneda al aire. → **H-2**.
- El fallo es **silencioso**: devuelve `{}` y el pipeline sigue sin odds. Hoy lo provoqué sin
  querer pasando `sport="MLB"` en vez de `baseball_mlb` y obtuve `{}` para los 15 juegos sin una
  sola alerta. En producción la misma firma (`{}`) puede significar "no hay mercado todavía" o
  "no supe emparejar", y **nada distingue los dos casos**. → **H-3**.

---

## FASE B — Datos del juego (PASO 0 del código)

### B1. Datos base del juego
**(P)** ¿Quiénes juegan, dónde y cuándo? **(C)** Eso. **(=) ALINEADO.**

### B2. Abridores probables + enriquecimiento (SIERA/xFIP/xERA/FIP/ERA, forma, fatiga)
**(P)** ¿Qué tan bien suprime carreras el abridor de hoy? **(C)** Eso, con jerarquía de fallback
explícita. **(=) ALINEADO** (VAL-6 verificó la aritmética).
**Hueco**: nada distingue "abridor confirmado" de "abridor probable"; si cambia, no hay re-precio.

### B3. Lineup del día
**(P)** ¿Qué nueve batean hoy?
**(C)** Se pide antes del TTE y se usa si hay ≥9 (`run_module.py:432-441`). En la práctica, a las
07:00, **no está**.
**(=) DESALINEADO en operación — PURP-9.**

### B4. Mezcla de manos del lineup (`*_lineup_lhb_pct`)
**(P)** ¿Contra qué mezcla de bateadores zurdos/derechos lanza este abridor?
**(C)** Si el lineup no llegó: **`0.45` fijo** (`run_module.py:365-366`, vía `setdefault`).
**(=) DESALINEADO — nuevo, H-4.** El sub-factor `platoon` del Pitcher Engine (peso 0.103) se
calcula contra una mezcla **asumida** en la gran mayoría de los picks. No es neutro: hoy produjo
aportes de +0.0072 y +0.0063 en el duelo real. El paso responde "¿cómo le va a este pitcher contra
un lineup promedio de la liga?" y se presenta como si respondiera "contra ESTE lineup".

### B5. Forma reciente (runs/game)
**(P)** ¿Cuánto viene anotando este equipo? **(C)** Eso. **(=) ALINEADO** — pero es el mismo insumo
que consumen Kalman y el sesgo (ver PURP-10).

---

## FASE C — Construcción de λ

### C1. λ de talento (TTE) — PASO 1
Ver **paso 2** del reporte. **ALINEADO** en definición, comprometido por B3/B4.
**Hueco**: no hay **gate de suficiencia de datos**. Hoy el TTE reportó `n_statcast_players=9` para
ambos equipos, pero nada impide que corra con 3 y produzca un λ con la misma cara de confianza. →
**H-5**.

### C2. Kalman de ofensa · C3. Sesgo multidimensional
Ver **pasos 3 y 4** del reporte (PURP-3, PURP-6, PURP-10). **DESALINEADO (ALTA)** — con la medición
de las tres corridas: apagar el sesgo mejora el Brier 0.00181.

### C4. Pitcher Engine (PASO 2)
**(P)** ¿Cuánto suprime carreras el abridor rival, hoy?
**(C)** Eso — combinación lineal de 5 sub-factores (VAL-6 verificó la fórmula a 6 cifras).
**(=) ALINEADO**, con dos matices propios de este paso:
- Es **el motor con más señal real** del pipeline: 1,882 valores distintos en 4,825 juegos,
  p05-p95 = 0.922-1.099. Es el único que mueve λ en el rango de ±10%.
- Corre sobre λ **park-neutral** a propósito (`run_module.py:577`), y el parque escala después.
  Es el contraejemplo del sistema: cuando alguien pensó el solapamiento, lo resolvió bien.
- **Hueco heredado**: su sub-factor `platoon` (peso 0.103) se calcula contra el `lhb_pct = 0.45`
  asumido de **B4** en el 96% de los picks.

### C5. Contextual Engine — descanso / B2B (PASO 3)
**(P)** ¿Qué factores binarios de esta noche mueven las carreras (descanso, fatiga, umpire)?
**(C)** **Una sola pregunta: "¿el visitante jugó ayer?"** — y nada más.

Medido sobre los 4,825 juegos del canónico:

| lado | valores observados |
|---|---|
| `context_on_home_lambda` | **1.0 en 4,825 de 4,825** |
| `context_on_away_lambda` | 0.96 en 621 (12.9%), 1.0 en 4,204 |

**(=) DESALINEADO por alcance — H-12 (BAJA-MEDIA).** El nombre y el docstring prometen "factores
de contexto de la noche"; lo que queda vivo es **una constante (`_B2B_MULT_AWAY = 0.960`) aplicada
al 12.9% de los juegos, de un solo lado**. El resto se apagó por buenas razones documentadas
(umpire: `umpire_stats` nunca se puebla → siempre 1.0; *rust* ≥3 días: 0 activaciones en 5,422
juegos → borrado). Lo llamativo es el lado local: la propia auditoría que fijó estas constantes
midió **+7.87% real** para el local en B2B y decidió **neutralizarlo a 1.000** por no creerle. Es
una decisión defendible, pero significa que el paso **declina responder** para el 50% de los casos
que su nombre cubre. Y sobre esa etapa —neutra en el 87% de los juegos— se aprende igual un peso
de pipeline (`context`), lo que le da resolución aparente a algo que casi nunca actúa.

### C6. Bullpen Engine (PASO 4)
**(P)** ¿Cuánto suprime carreras el bullpen rival, ponderado por cuántas entradas le tocarán?
**(C)** Eso, y la ponderación por `innings_weight` derivada de las entradas esperadas del abridor
es explícita — se reparten el juego con C4 sin doble conteo.
**(=) ALINEADO.** 973 valores distintos, p05-p95 = 0.973-1.043: señal real, la mitad de ancha que
la del abridor, coherente con que cubre menos entradas.
**Hueco**: los tres conteos de relevistas (VAL-7.2) ya se etiquetaron en la UI hoy, pero el engine
sigue promediando sobre **quien tenga cobertura de datos**, no sobre quien vaya a lanzar — no existe
noción de disponibilidad (relevista que lanzó 3 días seguidos y hoy no está).

### C7. Park + Weather Engine (PASO 5) · **H-13 (ALTA) — sesgo train/serve**
**(P)** ¿Cuánto infla o desinfla las carreras este parque **con las condiciones de hoy**?
**(C)** En vivo: eso. **En el backtest: sólo el parque** — el clima no existe.

`backtest_and_retrain.py:63-64` lleva el fetcher histórico de clima **comentado**
(`# DEFERRED F7: import kept for reactivation post-Sprint 3`), así que `weather_fetcher is None` y
`game_data["weather"]` nunca se puebla. Medido:

| entorno | n | valores distintos de `park_on_home_lambda` | rango |
|---|---:|---:|---|
| backtest (canónico) | 4,825 | **12** | 0.9400 – 1.1300 |
| live | 140 | **102** | 0.9331 – 1.1586 |

**(=) DESALINEADO — y contamina hacia arriba.** El modelo que corre en vivo **no es el modelo que
el backtest certifica**: en vivo el paso tiene un componente continuo que en la calibración no
existe, y sale del rango que el backtest llegó a ver. Todo lo que se ajusta contra el backtest
—Platt-1D, Platt-2D, pesos del pipeline, sesgo— está ajustado sobre la versión **sin clima**. El
peso `park` (~1.05-1.10) se aprendió para un factor de parque puro y en vivo multiplica un factor
parque×clima.
*Cambio conceptual*: o el backtest incorpora clima histórico (el fetcher existe, está desactivado),
o el clima se declara explícitamente como una capa **no calibrada** y se acota su rango. Hoy es lo
peor de ambos: activo en producción, invisible para toda la calibración.

### C8. Defensive Efficiency Engine (PASO 6)
**(P)** ¿Qué tan bien fildea esta defensa, neto del parque donde le tocó fildear?
**(C)** DER (1−BABIP) regresado + OAA, **sin ninguna mención de parque** en el archivo.
**(=) DESALINEADO — PURP-8 (MEDIA).** 559 valores distintos pero rango angosto (p05-p95 =
0.982-1.020): aporta poco y parte de ese poco es efecto de parque que C7 vuelve a cobrar. El
componente OAA sí viene ajustado por dificultad desde Statcast; el DER no.

### C9. HFA Engine (PASO 7) · **H-14 (ALTA) — corrección de probabilidad inyectada como λ**
**(P)** ¿Cuánto ayuda jugar en casa y cuánto penaliza viajar?
**(C)** Dos cosas muy distintas metidas en un multiplicador:

| lado | valores distintos en 4,825 juegos |
|---|---|
| `hfa_on_home_lambda` | **1** — exactamente `1.0280`, en todos los juegos |
| `hfa_on_away_lambda` | 5 niveles discretos (fatiga de viaje), 0.9867-1.0000 |

El lado local **no es un motor: es un intercepto global**. Y su origen, según su propio docstring,
es un residual de **probabilidad**: "el modelo sub-predice la probabilidad de victoria local por
~1.6-1.7pp". **(=) DESALINEADO en dos ejes:**

1. **Se diagnosticó en probabilidad y se aplica en λ.** No son el mismo objeto: +2.8% sobre λ_home
   mueve **toda la distribución de carreras**, no sólo el signo de quién gana. Sobre un λ_home
   típico de ~4.5 eso son **+0.13 carreras en el total de cada juego** — un empujón sistemático a
   los mercados de total y runline que **nunca formaron parte del diagnóstico que lo justificó**.
   Para dimensionarlo: la sobre-predicción de carreras que medí en las filas live es de +0.28
   carreras; este término solo explica cerca de la mitad.
2. **El diagnóstico dice que el error NO es uniforme y la corrección sí lo es.** Su propio
   docstring: *"la firma es asimétrica por lado favorito: los picks con favorito local están
   calibrados casi exacto (−0.4pp/+0.8pp), mientras que **los picks con favorito visitante
   sobre-valoran al visitante en 3-4pp**"*. Se aplicó una constante plana a un efecto que la misma
   medición describe como concentrado en un subconjunto. MATH-003 lo volvió a medir en julio y
   confirmó que no cierra parejo: 2024 queda en +0.08pp, 2025 en +1.47pp.

**Conexión que vale más que el hallazgo suelto**: el subconjunto donde este diagnóstico dice que el
modelo se equivoca —**juegos con favorito visitante**— es exactamente el mismo donde vive
**PURP-1** (el runline precia el evento del underdog con el precio del favorito). Dos defectos
independientes, hallados por caminos distintos, concentrados en la misma población. Es plausible
—no probado— que parte del "1.6pp que falta en probabilidad local" sea la sombra de un defecto de
identificación de favorito, y que el `_UNIFORM_HOME_MULT` esté tapando síntoma en vez de causa.
*Cambio conceptual*: una corrección de probabilidad pertenece a la capa de probabilidad (Platt), no
a λ; y antes de calibrar nada plano, separar la muestra por lado favorito.

### C10. Pesos del pipeline

Ver **paso 6** del reporte (**PURP-7**): los pesos live son de un motor superado.

### C11. Clip pre-Monte Carlo
**(P)** ¿Este λ es físicamente plausible? **(C)** Lo recorta a un rango duro y registra
`lambdas_history.final`.
**(=) ALINEADO como red de seguridad** — pero es el **único** control de plausibilidad de toda la
cadena, y sólo mira el rango absoluto de λ, nunca su relación con el mercado. → **H-6**.

---

## FASE D — Distribución (PASO 8 del código)

### D1. Monte Carlo
Ver **paso 7** del reporte (**PURP-2**): centro bien, forma −9%/−14% angosta.

### D2. Sub-simulación F5
**(P)** ¿Cómo se distribuyen las carreras de las primeras 5 entradas?
**(C)** Escala λ por `F5_SCALE=0.575` y re-simula.
**(=) Indeterminado — y es un paso que se ejecuta siempre y no produce nada.** **0 picks F5 en los
141** del ledger. O el gate nunca pasa o el mapeo de mercado no llega al publisher. Se paga el
cómputo en cada corrida sin salida observable. → **H-7**.

---

## FASE E — Mercados (PASO 9, dentro de `value_detector`)

### E1. Devig del precio de Pinnacle
**(P)** ¿Cuál es la probabilidad justa que implica el mercado? **(C)** Eso
(`remove_vig_multiplicative`, VAL verificó). **(=) ALINEADO.**

### E2. Moneyline + Platt-2D · E3. Total · E4. Runline
Ver **paso 8** del reporte: **PURP-1 (CRÍTICA)** en runline, **PURP-2** en total, ML alineado.

**Precisión nueva de esta pasada**, que cierra el argumento de PURP-1: el **reconciliador** grada el
runline con el punto **firmado** —`cover = diff + runline_point`, y devuelve `VOID` en vez de
adivinar cuando el signo falta (`reconciler.py:148-151`)— mientras el **detector de valor** lo
precia con la magnitud sin signo. **El mismo sistema tiene dos definiciones distintas del mismo
mercado**: la de graduar se arregló (`f474111`), la de preciar no, porque vive en el archivo
congelado. El pick con EV 105.09% del ledger es exactamente el caso `runline_point IS NULL` → se
publicó con una probabilidad inventada y se graduó `VOID`.

### E5. Confianza epistémica
**(P)** ¿Qué tan confiable es este número? **(C)** Un score de calidad de datos.
**(=) ALINEADO pero desconectado**: alimenta el `composite_score` (ranking) y **no** la
probabilidad. Un número con confianza baja no se encoge hacia el mercado, sólo baja de puesto en la
lista. → parte de la RAÍZ A.

---

## FASE F — Decisión

### F1. EV
**(P)** ¿Cuánto valor real tiene esta apuesta?
**(C)** `p × (odds−1) − (1−p)` — impecable (VAL) sobre la `p` que le entreguen.
**(=) DESALINEADO por herencia (PURP-4).** No tiene defecto propio: es el amplificador. Con la `p`
del evento equivocado (PURP-1) o de una distribución angosta (PURP-2), devuelve el número
equivocado con total precisión.

### F2. Kelly
**(P)** ¿Cuánto apostar dado mi borde y mi incertidumbre?
**(C)** Cuarto de Kelly con piso `MIN_KELLY=1%` y techo 15%.
**(=) ALINEADO con una salvedad ya expuesta (VAL-4.4)**: el piso puede convertir un borde marginal
en un stake 3.6x mayor al que la fórmula pide. Está documentado y desde b3325a5 se expone
(`kelly_floor_applied`). **Hueco propio**: Kelly asume que `p` es correcta; no existe ningún
encogimiento por incertidumbre del modelo — la confianza epistémica (E5) no entra acá tampoco.

### F3. Tier · **H-15 (ALTA) — un umbral de moneyline aplicado a todos los mercados**
**(P)** ¿Qué tan buena es esta apuesta?
**(C)** ¿En qué percentil del EV cae? — **y el percentil se calibró sobre otra población.**

El comentario de `ValueTier` lo dice con todas las letras: los cortes salen de "los percentiles de
la distribución de EV corregido de este backtest fresco (**4,627 juegos, moneyline, post-Platt-2D**)
… ≈ top 5% / 20% / 50%", con `p95 = 7.25%` → `ULTRA`. Es decir: **ULTRA significa "top 5% del EV de
moneyline después de la corrección de mercado"**.

Se aplica tal cual a runline y totales, que (i) nunca pasan por Platt-2D y (ii) tienen EV
sistemáticamente inflado. Resultado medido sobre los 141 picks:

| familia | picks | ULTRA | % |
|---|---:|---:|---:|
| MONEYLINE | 36 | 11 | 30.6% |
| RUNLINE | 53 | 35 | **66.0%** |
| TOTAL | 52 | 30 | **57.7%** |
| **todos** | **141** | **76** | **53.9%** |

**Una etiqueta definida como "top 5%" se está poniendo en el 54% de los picks**, y en dos de cada
tres del runline. El tier no está midiendo calidad: está midiendo en qué mercado se generó el pick.
*Cambio conceptual*: los cortes deben ser **por mercado** y calibrados sobre la misma población a la
que se aplican; y mientras un mercado no tenga capa de corrección (E2), su EV no es comparable con
el de uno que sí la tiene.

### F4. Composite / ranking
**(P)** ¿En qué orden debería mirar estas oportunidades?
**(C)** `composite = f(EV, confianza, sharpe, …)`, con gates de percentil re-escalados varias veces
(los comentarios documentan un ULTRA gate=67 que era matemáticamente inalcanzable, techo ~65.1).
**(=) DESALINEADO por herencia + una observación propia**: es el único lugar donde la confianza
epistémica entra, y entra a **ordenar**, no a **corregir**. Un número poco confiable no se encoge
hacia el mercado; sólo baja de puesto en una lista que igual se publica entera (en cuarentena, sin
filtro de tier).
**Nuevo**: el `⚫ NEUTRAL` también se publica (6 picks) porque la cuarentena salta el filtro de tier
a propósito. Correcto por diseño, pero significa que **la futura muestra primaria incluye picks que
el propio sistema clasifica como sin valor**, algo que V2 debería decidir explícitamente.

---

## FASE G — Publicación

### G1. Filtro de tier
**(C)** En cuarentena se salta entero (por diseño, Fase 2A). **(=) ALINEADO con el protocolo.**

### G2. Fallback sintetizado de ML_HOME
**(P)** — (ninguna; es una red de seguridad)
**(C)** Si `best_bets` viene vacío y `p_home > 0.5` y `EV > 2%`, **fabrica** un pick `ML_HOME` con
tier `"SLIGHT"` hardcodeado (`publisher.py:~281-300`).
**(=) DESALINEADO — nuevo, H-8.** Ese pick **no pasa por `value_detector`**: sin Platt-2D, sin
devig de Pinnacle, sin confianza, sin composite. Usa la probabilidad cruda del modelo contra el
precio de mercado — justo lo que la capa de humildad existe para impedir. Es raro pero real: **1
pick en el ledger** tiene `confidence_tier = 'SLIGHT'` (sin emoji), la firma inconfundible de esta
rama.

### G3. `pick_uid` e idempotencia
**(P)** ¿Este pick ya existe? **(C)** `MLB:{game_pk}:{market}:{game_date}` + `INSERT OR IGNORE`.
**(=) ALINEADO** (game_pk distingue doubleheaders; el bug de colisión F5/OVER se arregló en 2026-07-06).
**Hueco**: al ser idempotente por día, **el sistema no puede cambiar de opinión**. Si a las 13:00 el
modelo ve el lineup y opina distinto, el `INSERT OR IGNORE` descarta la opinión nueva. La
inmutabilidad es correcta para el registro, pero no hay un camino para "segunda opinión con más
información". → **H-9**, contracara operativa de PURP-9.

### G4. Persistencia de la predicción (`record_prediction`)
**(P)** ¿Qué predije y con qué? **(C)** Eso, en `game_outcomes(source='live')`.
**(=) ALINEADO** desde CHRON-001 — pero es la entrada del bucle de aprendizaje contaminado (PURP-3).

### G5. Registro de `model_prob` / `decision_prob`
**(=) DESALINEADO — H-10.** Ambas columnas se estampan del **mismo** campo post-Platt-2D: en las 141
filas son idénticas. El ledger **no guarda** qué pensaba el modelo antes de la corrección; eso sólo
vive en `game_outcomes.p_home_raw`. Un análisis de CLV que quiera separar "opinión del modelo" de
"decisión final" no puede hacerlo desde `picks`.

---

## FASE H — Después del juego

### H1. Captura de cierre
**(P)** ¿A qué precio cerró el mercado del lado que aposté?
**(C)** Desde ayer (`b629f55`), para las tres familias. Antes, sólo moneyline.
**(=) ALINEADO desde 2026-07-26**; todo lo anterior a esa fecha es pérdida permanente en derivados.

### H2. Reconciliación
**(P)** ¿Ganó o perdió? **(C)** Eso, con el punto firmado y `VOID` explícito ante ambigüedad.
**(=) ALINEADO** — y es el paso mejor construido de la cadena en cuanto a honestidad ante lo
desconocido. **Hueco**: no gradúa F5 ni mercados alternativos (consistente con H-7).

### H3. Realimentación al aprendizaje
**(P)** ¿Qué aprendo de lo que pasó?
**(C)** Kalman + sesgo + Platt se re-ajustan desde `game_outcomes`.
**(=) DESALINEADO — PURP-3.** Es el único paso de la cadena que **cierra el lazo**, y lo cierra
sobre un pool con 82% de filas de otra procedencia.

### H4. Monitoreo de calibración en producción
**(P)** ¿Mis probabilidades siguen siendo buenas, por mercado, en vivo?
**(C)** — **este paso no existe.** → **H-11.**
Hay `calibration_health()` (moneyline, y su alerta llevaba semanas sin explicación hasta la
certificación V1 de ayer) y el evaluador de derivados creado el 2026-07-26, que corre **offline y
sobre backtest**. Nada compara, en producción y por mercado, la probabilidad publicada contra la
frecuencia observada. Los +17.6pp de PURP-1 llevaban 7 días en el ledger y **el sistema no tenía
cómo notarlo**: sólo aparecieron cuando alguien leyó el código.

---

## HUECOS — cosas que ningún paso cubre

| # | Hueco | Evidencia | Severidad |
|---|---|---|---|
| **H-11** | **No existe monitoreo de calibración por mercado en producción** | Ningún consumidor; PURP-1 vivió 7 días sin detección | **ALTA** |
| **H-8** | Pick sintetizado en el publisher que **evita `value_detector`** (sin Platt-2D ni devig) | 1 pick con tier `SLIGHT` sin emoji en el ledger | **ALTA** |
| **H-6** | Ningún paso compara λ contra el mercado antes de publicar | CHC@PIT hoy: modelo 11.55 vs línea 9.0, publicado sin alerta | **ALTA** |
| **H-4** | `lhb_pct = 0.45` asumido alimenta el factor platoon cuando no hay lineup | `run_module.py:365-366`; 27/28 picks de hoy | MEDIA |
| **H-9** | El sistema no puede cambiar de opinión el mismo día (`INSERT OR IGNORE` por día) | `pick_uid` incluye fecha, no hora ni versión | MEDIA |
| **H-3** | Fallo de emparejamiento de odds indistinguible de "no hay mercado" | Ambos devuelven `{}`; reproducido hoy con 15 juegos | MEDIA |
| **H-5** | No hay gate de suficiencia de datos: un λ con 3 jugadores cubiertos sale igual que uno con 9 | La confianza epistémica existe pero no gatea nada | MEDIA |
| **H-2** | Doubleheader: emparejamiento por nombres + ±6h puede tomar el precio del juego equivocado | `odds_fetcher.py:588-598` | MEDIA |
| **H-10** | `picks.model_prob` y `decision_prob` son el mismo número; se pierde la opinión pre-corrección | 141/141 filas idénticas | MEDIA |
| **H-7** | F5 se computa siempre y nunca produce un pick | 0 de 141 | BAJA |
| **H-1** | No hay disparador por evento, sólo por reloj | crontab | (parte de RAÍZ C) |

---

## Cómo encaja con las tres raíces del reporte

- **RAÍZ A** (no hay noción de "cuánto vale mi número en este mercado") se extiende con **E5**
  (la confianza existe y no toca la probabilidad), **H-11** (nadie mide calibración en vivo) y
  **H-8** (hay una puerta trasera que ni siquiera pasa por la capa que sí existe).
- **RAÍZ B** (la capa de aprendizaje responde otra pregunta) suma **H3** como su único punto de
  cierre de lazo, y **C11** como el único control de plausibilidad —absoluto, nunca relativo al
  mercado (**H-6**).
- **RAÍZ C** (se opina antes de tener la información) suma **A0** (disparador por reloj), **B3/B4**
  (lineup ausente y handedness asumida) y **H-9** (no hay forma de cambiar de opinión cuando la
  información llega).

**PURP-1 sigue siendo el hallazgo aislado y más severo**, y esta pasada lo refuerza: el sistema ya
contiene la definición correcta del mercado —en el reconciliador— y la ignora al preciar.

---

## ÍNDICE DE TRIPLETAS — todos los pasos, sin agrupar

Una línea por paso: **qué pregunta hace** vs **qué responde realmente**. Es el resumen ejecutable
del mapa: si un paso aparece con dos frases distintas, ahí hay una desalineación.

### Fase A — antes de opinar
| paso | PREGUNTA que hace | LO QUE RESPONDE | = |
|---|---|---|---|
| A0 disparador | ¿Cuándo debo opinar? | "A las 07:00, siempre" | ✗ H-1 |
| A1 ventana | ¿De qué juegos opino? | ¿Qué juegos existen hoy y mañana? | ~ |
| A2 lead mínimo | ¿Es demostrable pre-juego? | Eso exactamente | ✓ |
| A3 identidad odds | ¿Qué evento de mercado es ESTE juego? | ¿Qué evento tiene nombres parecidos dentro de ±6h? | ✗ H-2/H-3 |

### Fase B — datos del juego
| paso | PREGUNTA | RESPONDE | = |
|---|---|---|---|
| B1 datos base | ¿Quiénes, dónde, cuándo? | Eso | ✓ |
| B2 abridores | ¿Qué tan bien suprime el abridor de hoy? | Eso (probable, no confirmado; sin re-precio si cambia) | ~ |
| B3 lineup | ¿Qué nueve batean hoy? | "Todavía no se publicó" en 27/28 | ✗ PURP-9 |
| B4 manos del lineup | ¿Contra qué mezcla L/R lanza? | ¿Contra una mezcla de 45% asumida? | ✗ H-4 |
| B5 forma reciente | ¿Cuánto viene anotando? | Eso | ✓ |

### Fase C — construcción de λ
| paso | PREGUNTA | RESPONDE | = |
|---|---|---|---|
| C1 TTE | ¿Cuánto anota este equipo en neutral? | ¿Cuánto anota este **plantel** en neutral? | ~ PURP-9 |
| C2 Kalman | ¿Anota distinto de lo que su talento implica? | Eso | ✓ (solapa con C3) |
| C3 sesgo | ¿Cuánto se desvía la ofensa de mi λ de ofensa? | ¿Cuánto se desvió el resultado del **λ FINAL** del pipeline entero? | ✗ PURP-3/6 |
| C4 pitcher | ¿Cuánto suprime el abridor rival? | Eso | ✓ |
| C5 contexto | ¿Qué factores binarios de la noche mueven carreras? | ¿El visitante jugó ayer? (y nada más) | ✗ H-12 |
| C6 bullpen | ¿Cuánto suprime el bullpen rival, ponderado por entradas? | ¿Cuánto suprimen **los relevistas con cobertura de datos**? | ~ |
| C7 parque+clima | ¿Cuánto infla este parque **con las condiciones de hoy**? | En vivo: eso. En backtest: **sólo el parque** | ✗ H-13 |
| C8 defensa | ¿Qué tan bien fildea, neto del parque? | ¿Qué tan bien fildea, **parque incluido**? | ✗ PURP-8 |
| C9 HFA | ¿Cuánto ayuda jugar en casa? | ¿Cuánto me falta de probabilidad local en promedio? (aplicado como λ) | ✗ H-14 |
| C10 pesos | ¿Cuánto confío en cada motor? | ¿Qué pesos minimizaban el log-likelihood de Poisson **en otro motor**? | ✗ PURP-7 (corregido: ver nota) |
| C11 clip | ¿Es físicamente plausible este λ? | ¿Está dentro de un rango absoluto? (nunca vs el mercado) | ~ H-6 |

### Fase D — distribución
| paso | PREGUNTA | RESPONDE | = |
|---|---|---|---|
| D1 Monte Carlo | ¿Cómo se distribuyen realmente las carreras? | ¿Cómo se distribuyen bajo NB(r=6.0), un r elegido por métricas de moneyline? | ✗ PURP-2 |
| D2 F5 | ¿Cómo se distribuyen las carreras de 5 entradas? | Lo mismo escalado por 0.575 — y **nadie consume la respuesta** | ✗ H-7 |

### Fase E — mercados
| paso | PREGUNTA | RESPONDE | = |
|---|---|---|---|
| E1 devig | ¿Qué probabilidad implica el precio? | Eso | ✓ |
| E2 ML + Platt-2D | ¿Cuánto vale mi opinión dado el precio? | "Prácticamente nada" (b=−0.097±0.120) — respuesta honesta | ✓ |
| E3 total | ¿Cuál es P(total > línea)? | ¿Cuál es P(total > línea) **bajo mi forma, que sé angosta**? | ✗ PURP-2 |
| E4 runline | ¿Cuál es P(que cubra el lado que precio)? | ¿Cuál es P(diff<1.5), **sin mirar de qué lado es la línea**? | ✗ PURP-1 |
| E5 confianza | ¿Es correcta mi probabilidad? | **¿Tengo datos reales para este juego?** — y su docstring aclara que NO responde lo primero | ✗ (por uso) |

### Fase F — decisión
| paso | PREGUNTA | RESPONDE | = |
|---|---|---|---|
| F1 EV | ¿Cuánto valor real hay? | ¿Cuánto valor habría **si mi p fuera correcta**? | ✗ PURP-4 |
| F2 Kelly | ¿Cuánto apostar dada mi incertidumbre? | ¿Cuánto apostar **asumiendo p exacta**? (la confianza no entra) | ~ |
| F3 tier | ¿Qué tan buena es esta apuesta? | ¿En qué percentil del EV **de moneyline** cae? | ✗ H-15 |
| F4 composite | ¿En qué orden mirar? | ~75% del score es EV con otro sombrero (ev .40 + edge .15 + kelly .10 + sharpe .10; sharpe r=0.82 con EV) | ✗ |

### Fase G — publicación
| paso | PREGUNTA | RESPONDE | = |
|---|---|---|---|
| G1 filtro tier | ¿Merece publicarse? | En cuarentena: "todo se publica" (por diseño) | ✓ |
| G2 fallback ML | — | Fabrica un pick **sin pasar por `value_detector`** | ✗ H-8 |
| G3 pick_uid | ¿Ya existe este pick? | ¿Ya existe **hoy**? → no se puede cambiar de opinión | ✗ H-9 |
| G4 persistencia | ¿Qué predije? | Eso | ✓ |
| G5 model/decision_prob | ¿Qué pensaba el modelo y qué decidí? | El mismo número dos veces | ✗ H-10 |

### Fase H — después del juego
| paso | PREGUNTA | RESPONDE | = |
|---|---|---|---|
| H1 cierre | ¿A qué precio cerró mi lado? | Desde 2026-07-26, eso (antes: sólo ML) | ✓ |
| H2 reconciliación | ¿Ganó o perdió? | Eso, con punto firmado y VOID explícito | ✓ |
| H3 aprendizaje | ¿Qué aprendo de lo que pasó? | ¿Qué aprendo de un pool con 82% de otra procedencia? | ✗ PURP-3 |
| H4 monitoreo | ¿Siguen siendo buenas mis probabilidades? | **nadie responde** | ✗ H-11 |
