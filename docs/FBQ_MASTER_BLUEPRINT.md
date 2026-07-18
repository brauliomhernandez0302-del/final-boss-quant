# FINAL BOSS QUANT — MASTER BLUEPRINT G∞
## De sistema casero a operación cuantitativa de nivel institucional

**Versión:** 1.7 — Julio 2026 (fallbacks honestos FALL-001 + FALL-002, roadmap paso 4, sesión 2026-07-18)
**Punto de partida:** MLB pipeline con Brier honesto **0.24482 / accuracy 55.34%** (sin cambio — ver nota v1.7, ambos fixes son señalización de procedencia pura, confirmada no-op por su propio gate), 504 tests, remediación look-ahead sustancialmente completada (ver 0.1)
**Principio rector:** Un cambio a la vez. Backtest después de cada uno. Nada entra sin validación PIT.

**Nota de esta versión (v1.7)**: roadmap paso 4 — eleva a principio de proyecto un patrón que
ya existía en el mejor código del repo (`ui/odds_loader.py`/REG-028, `get_platt_2d_params()`):
*un fallback jamás produce un valor indistinguible de una medición real; lo faltante se
señala con procedencia explícita, nunca se fabrica*. **FALL-001**: `adjust_for_park_and_weather()`
gana `weather_source: "live"|"missing"` en su metadata — puramente aditivo, el multiplicador
numérico no cambió ni un bit (verificado: `weather_mult` idéntico al valor neutral pre-fix
para un venue no mapeado). Efecto colateral correcto: el backtest, weather-blind por diseño,
ahora reporta `"missing"` en el 100% de los juegos — verdad auto-documentada, no un bug.
**FALL-002**: `get_travel_fatigue()` dejó de fabricar `1000mi/1tz` para un venue sin
coordenadas — el modo de fallo exacto que dejó a REG-015 invisible semanas para 4 estadios
renombrados. Ahora regresa `0`s honestos + `travel_source: "live"|"missing"`; `hfa_engine.py`
aplica penalización neutral (0.0) cuando ve `"missing"` en vez de adivinar. Cambio de
comportamiento real, pero solo en el caso de fallo — con los 30 estadios activos ya mapeados
post-REG-015, cero ocurrencias en producción hoy; la diferencia solo se manifestará en el
próximo rename, degradando con honestidad en vez de fabricar. **Test nuevo que cierra el gap
que el propio audit dejó anotado** (`findings.csv` bajo REG-015): cada venue de
`STADIUM_DATABASE` debe resolver en los otros 2 diccionarios de coordenadas — la invariante
exacta que REG-015 violó, ahora en CI. Ninguno de los dos fixes se serializa en el reporte del
backtest ni en `stage_factors_json` (verificado: `_park_meta`/la metadata de HFA se descartan
en `backtest_and_retrain.py`, nunca llegan a `_sf`) — gate estándar, diff byte a byte contra
el reporte del paso 3, única diferencia el timestamp `run_at`. 9 tests nuevos
(`tests/test_paso4_honest_fallbacks.py`), 504/504 en verde. Regla de promoción de
calibración (ride-along del paso 3): confirmado por grep que ya vive en `CONTRACTS.md` desde
el paso 3 — no se agregó a `CLAUDE.md` porque la condición ("si no está en ninguno") no se
cumplió. `MANIFIESTO.md` sigue sin crearse, decisión deliberada — consolidar en los tres docs
existentes en vez de una cuarta autoridad driftable, la lección directa de CLAUDE2.md/REG-033.

**Nota de esta versión (v1.6)**: roadmap paso 3 — el bundle de deduplicación que el propio audit
(`audit_20260714/`) marcó como "provablemente no-op", en un solo commit atómico con un solo
gate. **ODDS-001**: `backtest_and_retrain.py` mantenía su propia `_devig()`, copia algebraica
de `core/value_detector.py::remove_vig_multiplicative` nunca importada de ahí — la 5ª
instancia confirmada del patrón de drift-por-duplicación de este codebase (`LG_XWOBA`,
diccionarios de estadios, F5 naming, fórmula TTE). Reemplazada por el import compartido;
verificado bit-idéntico (no solo aproximado) contra 8 pares de odds diversos antes del swap.
**MATH-001**: `_l0_ratio()` en `learning_engine.py` aceptaba `stage_factors_json` sin leerlo
nunca — vestigio de los dos intentos revertidos de REG-006 (Brier 0.24479→0.24550→0.24636).
Removido de la firma y de sus 3 call sites; verificado que el acceso a rows en los tres es
por nombre (`sqlite3.Row`), nunca posicional, así que quitar la columna del `SELECT` no corrió
ningún índice. Los postmortems de `_l0_ratio` y `compute_team_bias_kalman_adjusted` se
preservaron verbatim — son memoria institucional que ya evitó un tercer intento fallido.
12 tests nuevos (`tests/test_paso3_dedup.py`), 495/495 en verde. Gate: `--season 2024,2025
--use-full-pit` re-corrido en tmux, diff contra el reporte validado del paso 2
(`chron002_gate_validation/backtest_report_20260718_0543.json`) — única diferencia, el
timestamp `run_at`, confirmando que ninguno de los dos cambios movió el modelo. **Regla
operativa nueva, documentada en `CONTRACTS.md`** (ride-along del paso 2, no un cambio de este
paso): desde CHRON-002, la calibración Platt (1D) en vivo NO hereda automáticamente los
refits de un backtest validado — requiere `scripts/promote_calibration.py --confirm`
explícito. Kalman y team bias sí se auto-mantienen en vivo. `MANIFIESTO.md`, citado en los
tres prompts de esta sesión como el lugar para esta regla, **no existe en el repo** (confirmado
por `find` + `git log --all`) — la regla quedó solo en `CONTRACTS.md` hasta que se decida la
forma de ese documento.

**Nota de esta versión (v1.5)**: sesión de dos fases. **Fase 1 (2026-07-14)**: auditoría
completa de solo-lectura de todo el proyecto (13 secciones — leakage, cronología,
calibración, mercado de odds, double-counting, fallbacks, matemática, operacional — reporte
en `audit_20260714/`, veredicto general PARTIAL). Encontró **CHRON-001** (Alto): `game_outcomes`
compartida, sin protección, entre escrituras live y sobrescrituras del backtest — verificado
empíricamente que 563 de 615 rows de season 2026 ya habían sido sobrescritas por una sola
corrida del 2026-06-28, 0 recuperables de ningún backup en disco. **Fase 2 (2026-07-17/18,
roadmap paso 1-2)**: remediación en 4 commits atómicos secuenciales, cada uno con su propio
gate de identidad. **CHRON-001** (`175f497`): `source`/columnas `backtest_*` en
`game_outcomes`, escritura del backtest aislada de las columnas live, 10 funciones de
`learning_engine.py` parametrizadas con `prediction_source`. **CHRON-002** (`f369f12`,
roadmap paso 2 commit A): mismo problema pero en `ml_state`/`kalman_state` — el residuo que
CHRON-001 dejó explícitamente abierto (el "step 4" del backtest todavía pisaba la caché que
lee producción); cerrado con `state_source` en ambas tablas + `scripts/promote_calibration.py`
como único camino sancionado (manual, `--confirm` obligatorio) para promover calibración de
backtest a vivo. **Fuente de entrenamiento cross-season** (roadmap paso 2 commit B): única
lectora cross-season de columnas de predicción en modo vivo (`recalibrate_platt_2d`, enumerada
exhaustivamente) pasa a preferir `COALESCE(backtest_p_home, p_home)` — las columnas live de
seasons pasadas quedan congeladas por diseño desde CHRON-001, las `backtest_*` sí se
refrescan con cada backtest validado. **LEARN-002** (roadmap paso 2 commit C): monitor
`calibration_health()` — % de predicciones vivas con Platt activo + % con Pinnacle presente,
alerta si cualquiera cae bajo 5% con n≥20, línea nueva en el status bar de la UI. Smoke test
contra la DB real (2026-07-18) encontró `pct_pinnacle_present=0%` (n=23, 14 días) — real,
reportado tal cual, consistente con `ODDS_API_KEY` desactivada esta sesión, no un bug nuevo.
**Las cuatro corridas de identidad** (`--season 2024,2025 --use-full-pit`, 4,830 juegos cada
una: FASE 6 de CHRON-001, gate de CHRON-002, mas las dos re-verificaciones cruzadas)
produjeron JSON idéntico byte a byte entre sí salvo el timestamp `run_at` — ninguno de los
cuatro commits movió un solo decimal del modelo, como exigía cada gate. Disciplina: 483/483
tests en verde en todo momento, un commit por fix, tmux para las corridas largas. Corregido
de pasada: el baseline "0.24486/55.42%" que v1.3/v1.4 citaban no coincidía con el último
reporte real en disco (`reports/round10_composite_weight_nudge/`) — era prosa desactualizada,
no el modelo; corregido en `CLAUDE.md` y arriba. Errata en `audit_20260714/
01_known_issues_register.md`: REG-021 afirmaba que `recalibrate_platt_2d` corría en el
backtest (heredado de `AUDITORIA_MLB_2026-07.md`, nunca re-verificado) — confirmado por grep
repetido que tiene cero call sites ahí; corregido.

**Nota de esta versión (v1.4)**: por primera vez desde v1.1, el número de backtest NO cambia — porque los bugs de esta sesión vivían aguas abajo del modelo, en la capa que convierte probabilidades en apuestas reales. Disparador: los picks en vivo del día mostraban EVs absurdos en runline/totales (+24% a +82%) y 6 de 7 moneylines en tier ULTRA. La auditoría completa de `odds_fetcher.py`/`value_detector.py`/`ui/`/`track_record/` (10 commits, `da721c1`→`0fd674e`, cada hallazgo de Fable re-verificado línea por línea antes de confiarlo) encontró, entre otros: **(1)** Pinnacle estructuralmente excluido del fetch en vivo — `REGION="us"` lo excluye (no es book US-licensed) y la key `"pinnaclesports"` era simplemente incorrecta (`"pinnacle"`); verificado vía `ml_home_pin` NULL en el 100% de filas live desde al menos 2026-07-04. Consecuencia real: la corrección Platt-2D de `value_detector.py`, gateada en tener línea justa de Pinnacle, estuvo silenciosamente apagada en producción TODA la temporada — el backtest no se ve afectado (usa el script histórico separado, que siempre tuvo la key correcta), pero backtest y producción nunca fueron comparables. **(2)** El origen directo de los EVs absurdos: el "best price" shopping tomaba `max(price)` ignorando el campo `point` — podía casar la probabilidad calculada para Over 9.5 con el precio de un Over 10.5 de otro book. Corregido con línea de consenso por punto (preferir el punto de Pinnacle, si no mayoría) — y, en un follow-up (`0fd674e`), se encontró que la matemática de cobertura del runline estaba además hardcodeada a ±1.5 sin importar qué línea real llegara, así que conectar el dato real sin arreglar esto habría solo cambiado la etiqueta, no el cálculo; ambos corregidos. **(3)** Push en líneas enteras ("Over 9.0") contado como pérdida completa en el EV — corregido con `push_prob` real. **(4)** El fallback pick de `track_record/publisher.py` pasaba cuotas ya-decimales por una conversión que asume americanas (~2830% de EV falso, stake al tope de Kelly). **(5)** `ui/odds_loader.py` fabricaba precios even-money (2.0) que `ui/mlb.py` trataba como mercado real. **(6)** La señal de "calidad de bullpen" se calculaba sobre el roster COMPLETO (abridores incluidos) — premisa documentada ("relievers dominan por volumen") verificada como falsa; corregido con clasificación real de rol por jugador. Disciplina refinada respecto al principio rector: fixes que tocan probabilidad/λ recibieron backtest before/after real (el fix de ruido epistémico: bit-for-bit idéntico en 4,830 juegos; el de bullpen: 286 juegos por la ruta legacy, Brier +0.00024 = piso de ruido; el de runline: identidad matemática probada para la línea estándar 1.5, no hizo falta backtest); fixes de solo clasificación de tier, display de EV o plumbing se validaron con tests dirigidos — backtestear lo que provablemente no puede mover el número da falsa confianza en la dirección equivocada. 462 tests en verde en cada commit; nada pusheado aún. Lección de proceso: varios commits tempranos mezclaron accidentalmente trabajo previo sin commitear del usuario (fix de truncamiento walk-off en `learning_engine.py`/`run_module.py`) — detectado, informado, decidido dejarlos como están; los commits posteriores se hicieron con cuidado quirúrgico de exclusión.

**Nota de esta versión (v1.3)**: la v1.2 citaba 0.24592/54.35% — ese número, aunque libre del leak de team-bias, estaba a su vez confundido por un segundo problema: un lanzamiento duplicado accidental de un proceso de backtest (matado a los segundos) alcanzó a resetear `ml_state.platt_params` a identidad antes de que el proceso real arrancara, dejando la corrida entera de v1.2 sin calibración Platt aplicada — confirmado empíricamente (`p_home == p_home_raw` en las 4,830 filas). No es un bug del pipeline en sí, sino un artefacto de lanzamiento; pero reveló que el mecanismo de reset+warm-start no tiene protección contra esta clase de corrupción silenciosa (mismo patrón que el leak de team-bias: falla sin error visible). Corregido relanzando limpio: **accuracy 54.35%→55.30%, Brier 0.24592→0.24525**. El ROI mejora pero sigue delgado e inconsistente (edge≥8% +0.20%, edge≥10% -2.97%) — ni la conclusión de rentabilidad de v1.0/v1.1 ni la de "claramente no rentable" de v1.2 se sostienen; sigue siendo una pregunta abierta con el número más honesto disponible hasta ahora. Nota: la temporada 2024 de este backtest sigue en calibración identidad por diseño (no hay datos de temporada 2023 para warm-start) — limitación aceptada, no nueva.

**Nota de la v1.2 (conservada por continuidad)**: la v1.1 citaba Brier 0.24242/accuracy 56.87% como "honesto" — no lo era. Construyendo el test anti-leakage que esta misma FASE 0 pedía (0.1), se encontró que `LearningEngine.compute_team_bias`/`compute_multidim_bias` no tenía ningún corte de fecha: en el backtest, el bias que se multiplica a λ para un juego de marzo veía los resultados reales de esa misma temporada completa, incluyendo septiembre — usando el futuro para corregir el pasado. Corregido con un parámetro `before_date` walk-forward (sin cambiar el comportamiento en vivo, donde nunca hubo leak real). Nuevo test de regresión permanente: `tests/test_anti_leakage_opening_day_2024.py` — cierra buena parte del ítem 0.1 pendiente (ver abajo).

**Nota de la v1.1 (conservada por continuidad)**: la v1.0 quedó desactualizada casi de inmediato. Una revisión exhaustiva motor-por-motor y archivo-por-archivo (los 7 motores del pipeline MLB, `value_detector.py`, `data_fetchers.py`, `track_record/`, `ui/mlb.py`, `odds_fetcher.py`) encontró y corrigió una serie de bugs reales de producción que la v1.0 no conocía, y confirmó que varios ítems de FASE 0 ya estaban sustancialmente resueltos por el rebuild PIT de esa sesión.

## VISIÓN FINAL
Una plataforma cuantitativa completa: Predice, Detecta, Optimiza, Ejecuta, Monetiza, Aprende.
Stack de capas: Producto (Streamlit/Telegram/API) → Decisión (Portfolio Kelly/Correlaciones) → Mercado (Odds multi-book/Devig/Steam/CLV) → Modelos (MLB/NBA/UFC/Ensemble/Calibración) → Datos PIT (Feature Store/Snapshots/Savant/Odds/Weather) → Infra (PostgreSQL/Airflow/Tests/Monitoring).

## FASE 0 — CERRAR DEUDA TÉCNICA (2-4 semanas)

**0.1 Completar remediación look-ahead** — MAYORMENTE COMPLETADO vía el rebuild PIT de esta sesión (TTE/Bullpen/Defense/Pitcher ahora point-in-time correctos). Estado por sub-ítem:
- El leak de `_team_dict()` (campos `team_era`/`team_whip`/`runs_allowed_per_game` alimentándose sin gate desde un `get_team_pitching_stats()` con agregado de temporada completa) — encontrado y corregido el 2026-07-06.
- **[NUEVO, el hallazgo más severo de todo FASE 0] El leak de `compute_team_bias`/`compute_multidim_bias`** (`calibration/learning_engine.py`) — sin corte de fecha, contaminaba el bias multiplicado a λ en cada backtest con resultados reales de toda la temporada, incluyendo juegos futuros respecto al que se predecía. Encontrado y corregido el 2026-07-08 (parámetro `before_date` walk-forward). Este era el leak que más importaba de los encontrados hasta ahora — su remoción movió accuracy -1.92pp y volvió el ROI negativo para edge<10%.
- **✅ Test anti-leakage automatizado en Opening Day 2024** — construido 2026-07-08, `tests/test_anti_leakage_opening_day_2024.py` (5 tests, metodología poison-test). Cubre el leak de team-bias (arriba) y la capa de eventos raw de Savant (representativa de los 4 dominios PIT).
- La capa `advanced_pit_enrichment/` (26 archivos) nunca tuvo una auditoría formal archivo-por-archivo — se tocó extensivamente construyendo Pitcher PIT, pero "tocado en el camino" no es "auditado". Sigue pendiente.

Sigue pendiente, sin cambios de la v1.0/v1.1: guards `start_dt`/`end_dt` en el resto de queries SQL de backtest más allá de las ya revisadas (`recalibrate_platt`/`recalibrate_platt_2d` ya verificadas como seguras — corren después del loop principal, no contaminan el backtest), eliminar `datetime.utcnow()` en contexto histórico, reset de bias state entre corridas (parcialmente resuelto: el bypass de cache de `before_date` ya evita que un `ml_state` viejo contamine corridas nuevas), invalidación de cache por fecha.

**0.2** ~~Resolver tests de defense multipliers que fallan~~ — **COMPLETADO.** 8/8 tests pasan, confirmado en vivo el 2026-07-06. Sin acción pendiente.

**0.3 Activar cron de producción** — REENCUADRADO. El cron/publisher ya existe (`track_record/publisher.py` + `reconciler.py`: publicación pre-partido + reconciliación post-partido). No es "activar", es "corregir 3 bugs reales" encontrados el 2026-07-06:
- `TrackRecordDB.resolve_pick()`/`upsert_daily_snapshot()`: el running total del bankroll se calculaba con `MAX(running_total)` en vez de "más reciente por orden de inserción" — se rompe en cuanto hay una pérdida, corrompiendo el gráfico de equity y el `cumulative_pnl` de los snapshots diarios.
- `_market_label()`: colisión de substring hacía que "F5 OVER 4.5" se guardara como "OVER", chocando literalmente con un over de partido completo real y resolviéndose contra la línea equivocada; "F5 ML HOME"/"F5 ML AWAY" nunca hacían match con ninguna rama, resolviendo siempre como VOID.
- El esquema `all_opportunities` (fuente de `best_bets`) no tenía campos `odds`/`odds_decimal` ni `model_prob`, sin fallback viable — cada pick resuelto calculaba profit_loss_units con un payout hardcodeado de 1.909 en vez de las cuotas reales, corrompiendo el propio `profit_loss_units`, no solo el bankroll ledger derivado de él.

Los 3 corregidos el 2026-07-06.

**0.4 Completar rolling data** — MAYORMENTE COMPLETADO vía el mismo rebuild PIT (TTE rolling, Pitcher Savant rolling + FanGraphs diario, construidos desde cero esta sesión). "Validar multiplicadores" se cumplió vía la revisión motor-por-motor del 2026-07-06 (TTE, Pitcher, Bullpen, Defense — cada uno con hallazgos reales corregidos, ver 0.5).

**0.5 [NUEVO] Cerrar hallazgos de la revisión exhaustiva 2026-07-06 que no encajan en los 4 puntos originales:**
- Bugs de escala en `calculate_composite_score()`: `kelly_score` tenía un techo estructuralmente inalcanzable (asumía Kelly hasta 0.5, pero `CONFIG.MAX_KELLY=0.15` lo topa en 30/100 real); `ev_score`/`sharpe_score` tenían compresión severa/saturación contra la distribución real post-Platt-2D. El tier ULTRA era literalmente inalcanzable por cualquier apuesta, independiente del problema de varianza de confianza que motivó encontrarlo. Corregido: kelly con hard-cap contra `MAX_KELLY`; ev/sharpe con anclaje por percentil contra la distribución real. Umbrales de tier recalibrados por percentil sobre la distribución corregida.
- Mercados F5 nunca funcionaron end-to-end por ninguna de las dos rutas de producción reales: (a) ruta auto-fetch — `get_best_odds_for_teams()` devolvía `f5_home`/`f5_over` en vez de la convención `f5_ml_home`/`f5_total_over` usada en el resto del código; (b) ruta UI-selector — `GameData` no tenía campos F5 en absoluto, `build_game_selector()` nunca los leía, `MLBAnalyzer.analyze()` nunca los construía. Ambas corregidas el 2026-07-06 (3 archivos en la ruta UI-selector).
- Inversión de naming en `stage_factors_json` (pitcher/defense/bullpen): las claves `home_X`/`away_X` están nombradas por qué lambda afectan, no por de qué equipo es el motor — documentado vía TODOs en los 3 archivos, el renombrado real aún no ejecutado (no afecta la corrección del pipeline, gradient descent es auto-consistente respecto a esto, pero sí puede confundir cualquier análisis externo, como ya ocurrió una vez esta sesión).

**0.6 [NUEVO, v1.4] Cerrar hallazgos de la auditoría odds/valor 2026-07-12/14** — la capa hermana de 0.5, pero aguas abajo del modelo. Los 13 bugs principales ya corregidos (ver nota v1.4); dos pasadas independientes de re-verificación (`4c0e672`) confirmaron cada fix y encontraron 3 issues menores nuevos, también corregidos (incluido el patrón "partial fetch envenena el caché" repetido en el código NUEVO de esta misma sesión — los patrones de bug reinciden incluso dentro de una sesión que los está corrigiendo, argumento adicional para 1.1). Gaps conocidos, deliberadamente no corregidos, en orden de prioridad:
- `evaluate_value_ultra()`: edge/Kelly aún no push-adjusted como sí lo está el EV (`94dc635`) — subestima ambos levemente en totales de línea entera; dirección conservadora, menor.
- Mercados F5 estructuralmente muertos: el endpoint bulk de The Odds API responde 422 si se piden keys F5 — requiere el endpoint per-event, no cableado. Predata esta sesión; los fixes F5 de esta sesión (flag de consistencia MC↔detector, push-prob en el path duplicado) son "proactivamente correctos" para cuando se cablee.
- `bullpen_engine.py`: la clasificación de relevistas usa split de un solo equipo — un pitcher traspasado a mitad de temporada puede clasificarse mal (edge case raro).
- `learning_engine.py`: el chequeo nuevo de `abstractGameState=="Final"` toma el primer juego de la respuesta sin match explícito de gamePk (la query ya está scoped server-side; riesgo bajo, endurecer algún día).

**Criterio salida** (actualizado 2026-07-09): ✅ test anti-leakage automatizado en verde (`tests/test_anti_leakage_opening_day_2024.py`); ✅ `stage_factors_naming` renombrado; ✅ backtest re-corrido (tres veces — seed determinístico, fix del leak de team-bias, fix de la corrupción de Platt); ✅ auditoría formal de `advanced_pit_enrichment/` completada (26 archivos, 1 gap real encontrado y corregido en `tte_prior_baseline_builder.py`). **Deliberadamente pospuesto por decisión del usuario**: cron de producción corriendo 7 días — no tiene sentido dejarlo correr sin intervención hasta no confirmar que el sistema es rentable (el ROI actual es delgado e inconsistente, no una conclusión clara en ningún sentido). FASE 0 está funcionalmente cerrada salvo ese último punto, que ahora depende de FASE 1+ (mejorar el modelo) antes de retomarse, no de más trabajo de infraestructura.

## FASE 1 — MLB DE GRADO INDUSTRIAL (4-8 semanas)

1.1 Feature Store con snapshots PIT formales (tabla `features(entity_id, feature_name, value, as_of_date, computed_at)`, backtest y producción leen de la misma tabla). **[NOTA 2026-07-06]**: la revisión de hoy encontró, repetidamente, la misma clase de bug que este ítem previene estructuralmente — constantes duplicadas que se desalinean silenciosamente (`LG_XWOBA` hardcodeado de forma idéntica-pero-propensa-a-desactualizarse en 4 archivos distintos, `LG_DER` con el mismo patrón, el propio esquema de `all_opportunities` necesitando el mismo campo parchado dos veces por dos consumidores independientes). No es un "nice to have" arquitectónico — es la corrección directa de un patrón de bug que costó tiempo real de revisión tres veces en una sola sesión.

1.2 Migrar SQLite → PostgreSQL (concurrencia, índices por fecha, particionado). **[NOTA]**: no agendar por calendario. SQLite manejó sin esfuerzo todo lo que esta sesión le exigió — corridas de Monte Carlo de millones de muestras, un backtest de 4,830 juegos, varios cachés PIT concurrentes de 60-130MB. Gatear esta migración a un incidente real de lock-contention observado, o al inicio real de la complejidad de scheduling multi-deporte de FASE 5 — no antes.

1.3 Orquestación Airflow/Prefect (DAG odds→stats→weather→lineups→features→picks→snapshot, retry automático). **[NOTA]**: misma lógica que 1.2. Airflow está diseñado para orquestar muchos pipelines interdependientes entre equipos; esto es un DAG lineal corrido por una persona, que un cron + wrapper de reintentos ya cubre — y que, de hecho, `track_record/publisher.py`+`reconciler.py` ya implementan arquitectónicamente. Adoptarlo ahora es overhead operacional real (stack nuevo que aprender y mantener, nueva superficie de fallo) para un problema de escala que aún no existe. Misma re-gatilla que 1.2.

1.4 CLV Tracking — "la métrica que separa sharps de squares": guardar línea de entrada + línea de cierre Pinnacle, CLV = closing vs entrada. "Si tu CLV promedio es positivo de forma sostenida, tienes edge real aunque tengas racha perdedora. Si es negativo con 57% accuracy, la accuracy es ruido." **[NOTA 2026-07-14]**: el instrumento ya existe (`track_record/capture_closing_lines.py`, sesión anterior) pero estuvo inerte por dos dependencias resueltas esta sesión: el feed live de Pinnacle estaba estructuralmente roto (`da721c1`, ver nota v1.4) y el precio de Pinnacle nunca se persistía a la DB de entrenamiento (`9f64463` — sin ese fix, la calibración quedaba congelada en temporadas backfilled para siempre, sin importar cuánto corriera la app). Ambos cerrados; los datos live recién empiezan a fluir desde 2026-07-12.

1.5 Monitoring en producción (Brier rolling 30 días real, alerta de degradación, calibration plot mensual, drift detection).

1.6 Simulación PA-por-PA (matchup L/R, orden lineup, TTE penalty explícito, bullpen chains — desbloquea props nativamente). **[NOTA]**: complejidad real equivalente a FASE 3 (es dependencia directa de 3.3, el motor de props), no un par de tamaño comparable a los otros 5 ítems de esta fase. Es probable que domine el timeline real de FASE 1 completa por sí solo — planificarlo como tal, no como "un ítem más entre seis".

**Criterio salida**: 30 días de picks en producción con CLV positivo documentado.

## FASE 2 — CAPA DE MERCADO AVANZADA (4-6 semanas)

2.1 Odds screen multi-book con historia (`odds_history` table, guardar cada X min). **[NOTA]**: probablemente requiere acceso pago a datos multi-book en tiempo real (feeds cruzados entre sharp books rara vez son gratuitos), y este proyecto ya tuvo el fetch de odds históricas bloqueado por cuota de API al menos una vez antes. Confirmar que el acceso a datos existe antes de comprometer tiempo de ingeniería aquí, no después.

2.2 Steam detection (movimientos sincronizados Pinnacle/Circa vs soft books, alerta Telegram).

2.3 Devig avanzado (power method, Shin's method, comparar cuál da mejor CLV).

2.4 Modelo de line movement (predecir cierre dado apertura+movimiento temprano+volumen).

2.5 Limit/availability tracking (cuándo te limitan books, estrategia de preservación de cuentas).

**Criterio salida**: sistema de alertas de steam funcionando + entrada mejor que naive (medido en CLV).

## FASE 3 — MODELADO DE ÉLITE (8-12 semanas)

3.1 Ensemble de modelos (Modelo A=NegBin actual, B=XGBoost/LightGBM sobre feature store, C=Sim PA-por-PA; stacking con pesos walk-forward, nunca aleatorio).

3.2 Calibración de 2da generación (Isotonic Regression, por régimen día/noche/dome/mes, validado out-of-sample — "ya te quemaste con la circularidad de Platt una vez"). **[NOTA 2026-07-14]**: ahora hay TRES clases confirmadas de corrupción/apagado silencioso de calibración (leak de team-bias, reset de Platt, y el gate de Pinnacle apagando Platt-2D en vivo toda la temporada) — cualquier calibración de 2da generación necesita, de diseño, un monitor de "¿está la corrección efectivamente aplicándose?" en producción, no solo estar bien fitteada.

3.3 Props Engine formal (distribuciones nativas de Ks/hits/HRs/TB por jugador desde sim PA-por-PA).

3.4 Bayesian updating intra-temporada (priors tipo ZiPS/Steamer actualizados con evidencia de temporada, shrinkage explícito). **[NOTA]**: puede que esto ya esté sustancialmente construido — el blend de temporada-previa de TTE (`prior_w = 1000/(1000+PA_current)`, revisado en profundidad esta sesión) es funcionalmente este mismo patrón. Verificar qué existe antes de planificarlo como trabajo desde cero.

3.5 Datos nueva generación (Statcast pitch-level, umpire tendencies, lineup confirmations en tiempo real → repricing automático).

**Criterio salida**: Brier del ensemble < modelo actual en walk-forward + props engine con CLV positivo en ≥1 mercado.

## FASE 4 — PORTFOLIO Y RIESGO (3-4 semanas)

4.1 Kelly de portafolio con correlaciones (optimización conjunta maximizando E[log(bankroll)] sobre vector completo de bets con matriz de correlación por simulación).

4.2 Risk limits duros en código (max % bankroll por día/juego/mercado, fractional Kelly 0.25x-0.5x, circuit breaker por drawdown).

4.3 Bankroll multi-book (tracking por book, optimizador de dónde colocar cada bet).

4.4 Simulador de bankroll (Monte Carlo de trayectorias a 1 año, probabilidad de ruina).

**Criterio salida**: todo bet pasa por el optimizador de portafolio, cero bets manuales fuera del sistema.

## FASE 5 — EXPANSIÓN MULTI-DEPORTE (8-16 semanas, solo cuando MLB imprima CLV positivo consistente)

5.1 NBA rebuild sobre infra nueva (migrar el G10+ Ultra Pro actual al feature store PIT).

5.2 UFC rebuild honesto ("el módulo actual genera datos ficticios con hashes de nombres — se demuele completo").

5.3 Soccer opcional (evaluar ROI de esfuerzo, ligas secundarias si se hace).

Regla de oro: cada deporte nuevo hereda TODA la infra, solo cambia la capa de modelo.

## FASE 6 — PRODUCTO Y MONETIZACIÓN (4-6 semanas, en paralelo con FASE 5)

6.1 Track record verificable criptográficamente (timestamp antes del juego, hash inmutable, dashboard público auditable).

6.2 Bot Telegram/Discord (tiers free/premium, Stripe).

6.3 Dashboard Streamlit Pro para clientes (explicación de cada pick, educación bankroll).

6.4 Separación estricta (bet personal entra antes de publicar, documentado con transparencia).

**Criterio salida**: primeros 10 suscriptores de pago + track record público 90+ días.

## FASE 7 — FRONTERA (sin fecha, solo cuando FASES 0-4 impriman)

7.1 LLM como capa de información (noticias/beat reporters/lesiones → señales estructuradas, Claude API batch).

7.2 Modelo de comportamiento de books (modelar el book, no solo el juego).

7.3 Mercados de derivados y live betting (repricing en vivo con sim PA-por-PA).

7.4 Investigación continua (framework walk-forward → shadow mode → producción reducida → producción completa).

## STACK OBJETIVO

DB: SQLite→PostgreSQL. Orquestación: cron→Airflow/Prefect. Features: ad-hoc→Feature Store PIT versionado. Modelos: NegBin+mult→Ensemble. Calibración: Platt→Isotonic por régimen. Sizing: Kelly individual→Portfolio Kelly. Frontend: Streamlit→+Bot+API. Monitoring: ninguno→Brier rolling+CLV+drift. Tests: 448→600+ incl. anti-leakage.

## MÉTRICAS NORTE (orden de importancia)

1. CLV promedio 2. Brier de producción rolling 30d 3. ROI en unidades a 500+ bets 4. Calibración por bucket 5. Uptime del pipeline

## LO QUE ESTE PLAN NO ES

No es permiso para saltar FASE 0. No es garantía de rentabilidad (57% accuracy backtest es prometedor pero mercado es adversarial, CLV de producción es el único juez). No es un sprint (9-15 meses hasta FASE 6, comprimirlo reintroduce bugs).

## PRÓXIMOS 3 PASOS (actualizado 2026-07-14)

1. **Acumular datos live de Pinnacle y, con fecha, recalibrar los tiers.** El síntoma que disparó esta sesión ("6 de 7 picks en ULTRA") tenía DOS causas independientes: el bug de mezcla de puntos/líneas (corregido, `44c81d8`) y un desajuste estructural aún abierto — los umbrales de ValueTier/composite-score se fittearon contra una población de backtest solo-Pinnacle, pero el EV en vivo se calcula contra el mejor precio shoppeado entre books (estructuralmente más alto). No se puede recalibrar hoy: requiere semanas de datos live de Pinnacle, que recién empezaron a fluir el 2026-07-12 (`da721c1` + persistencia en `9f64463` + `capture_closing_lines.py` corriendo). **Acción concreta: dejar acumular ~4-6 semanas; luego refittear umbrales de tier y percentiles de composite contra la población live real referenciada a Pinnacle.** Esto absorbe y supersede parcialmente el paso 1 de v1.3 (decisión de rentabilidad): con Platt-2D apagado en vivo toda la temporada, backtest y producción nunca fueron comparables — la pregunta de rentabilidad solo tiene respuesta con CLV de producción real, y por primera vez el instrumento completo existe y está encendido.
2. **La verificación de circularidad Platt/Platt-2D** (heredada de v1.3, aún no ejecutada): que ningún input a `recalibrate_platt`/`recalibrate_platt_2d`/`_gradient_step` derive circularmente de un output ya calibrado. Sube otra vez de urgencia: ya son tres clases confirmadas de falla silenciosa de calibración en este código (ver nota en 3.2), y el refit de tiers del paso 1 va a apoyarse directamente sobre esta maquinaria.
3. **Monitor mínimo de "la calibración está viva"** — evolución del paso 3 de v1.3: el endurecimiento de `ml_state` contra escritura concurrente ya se hizo (lockfile, sesión 2026-07-11/12, ver CLAUDE.md), pero el hallazgo de Pinnacle demuestra que la falla más cara no fue corrupción de estado sino un gate apagando la corrección sin ningún error visible durante meses. Un chequeo barato (¿% de predicciones live con `p_home != p_home_raw`? ¿% de filas con `ml_home_pin` no-NULL?) alertando cuando cae a cero habría detectado esto el día uno. Es el precursor mínimo del 1.5 (monitoring) que estos incidentes ya justifican adelantar.
