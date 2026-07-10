# FINAL BOSS QUANT — MASTER BLUEPRINT G∞
## De sistema casero a operación cuantitativa de nivel institucional

**Versión:** 1.3 — Julio 2026 (segunda corrección de baseline tras hallazgo de corrupción de calibración Platt, sesión 2026-07-09)
**Punto de partida:** MLB pipeline con Brier honesto **0.24525 / accuracy 55.30%** (backtest limpio 2026-07-09, `--season 2024,2025 --use-full-pit`, seed determinístico por `game_pk`, fix del leak de team-bias, calibración Platt correctamente aplicada), 453 tests, remediación look-ahead sustancialmente completada (ver 0.1)
**Principio rector:** Un cambio a la vez. Backtest después de cada uno. Nada entra sin validación PIT.

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

**Criterio salida** (actualizado 2026-07-09): ✅ test anti-leakage automatizado en verde (`tests/test_anti_leakage_opening_day_2024.py`); ✅ `stage_factors_naming` renombrado; ✅ backtest re-corrido (tres veces — seed determinístico, fix del leak de team-bias, fix de la corrupción de Platt); ✅ auditoría formal de `advanced_pit_enrichment/` completada (26 archivos, 1 gap real encontrado y corregido en `tte_prior_baseline_builder.py`). **Deliberadamente pospuesto por decisión del usuario**: cron de producción corriendo 7 días — no tiene sentido dejarlo correr sin intervención hasta no confirmar que el sistema es rentable (el ROI actual es delgado e inconsistente, no una conclusión clara en ningún sentido). FASE 0 está funcionalmente cerrada salvo ese último punto, que ahora depende de FASE 1+ (mejorar el modelo) antes de retomarse, no de más trabajo de infraestructura.

## FASE 1 — MLB DE GRADO INDUSTRIAL (4-8 semanas)

1.1 Feature Store con snapshots PIT formales (tabla `features(entity_id, feature_name, value, as_of_date, computed_at)`, backtest y producción leen de la misma tabla). **[NOTA 2026-07-06]**: la revisión de hoy encontró, repetidamente, la misma clase de bug que este ítem previene estructuralmente — constantes duplicadas que se desalinean silenciosamente (`LG_XWOBA` hardcodeado de forma idéntica-pero-propensa-a-desactualizarse en 4 archivos distintos, `LG_DER` con el mismo patrón, el propio esquema de `all_opportunities` necesitando el mismo campo parchado dos veces por dos consumidores independientes). No es un "nice to have" arquitectónico — es la corrección directa de un patrón de bug que costó tiempo real de revisión tres veces en una sola sesión.

1.2 Migrar SQLite → PostgreSQL (concurrencia, índices por fecha, particionado). **[NOTA]**: no agendar por calendario. SQLite manejó sin esfuerzo todo lo que esta sesión le exigió — corridas de Monte Carlo de millones de muestras, un backtest de 4,830 juegos, varios cachés PIT concurrentes de 60-130MB. Gatear esta migración a un incidente real de lock-contention observado, o al inicio real de la complejidad de scheduling multi-deporte de FASE 5 — no antes.

1.3 Orquestación Airflow/Prefect (DAG odds→stats→weather→lineups→features→picks→snapshot, retry automático). **[NOTA]**: misma lógica que 1.2. Airflow está diseñado para orquestar muchos pipelines interdependientes entre equipos; esto es un DAG lineal corrido por una persona, que un cron + wrapper de reintentos ya cubre — y que, de hecho, `track_record/publisher.py`+`reconciler.py` ya implementan arquitectónicamente. Adoptarlo ahora es overhead operacional real (stack nuevo que aprender y mantener, nueva superficie de fallo) para un problema de escala que aún no existe. Misma re-gatilla que 1.2.

1.4 CLV Tracking — "la métrica que separa sharps de squares": guardar línea de entrada + línea de cierre Pinnacle, CLV = closing vs entrada. "Si tu CLV promedio es positivo de forma sostenida, tienes edge real aunque tengas racha perdedora. Si es negativo con 57% accuracy, la accuracy es ruido."

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

3.2 Calibración de 2da generación (Isotonic Regression, por régimen día/noche/dome/mes, validado out-of-sample — "ya te quemaste con la circularidad de Platt una vez").

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

## PRÓXIMOS 3 PASOS (actualizado 2026-07-09)

1. **Decidir qué hacer con la conclusión de rentabilidad.** Con el número más honesto hasta ahora (Brier 0.24525, accuracy 55.30%, ROI delgado e inconsistente: +0.13% a edge≥2%, +0.20% a edge≥8%, -2.97% a edge≥10%), ni "el sistema imprime" ni "el sistema no sirve" están validados — es una zona gris real. Antes de seguir construyendo (feature store, ensemble, etc.), vale la pena decidir: ¿se investiga la calibración por bucket más de cerca (el bucket `<40%` sigue con +6.5pp de sub-predicción), o se avanza asumiendo que solo el CLV de producción real puede resolver la ambigüedad (como dice el propio blueprint en "MÉTRICAS NORTE")?
2. La verificación de circularidad Platt/Platt-2D que Fable había sugerido (que ningún input a `recalibrate_platt`/`recalibrate_platt_2d`/`_gradient_step` derive circularmente de un output ya calibrado por el mismo mecanismo) — especificada, no ejecutada. Cobra más urgencia ahora que se confirmaron DOS clases distintas de corrupción silenciosa de calibración en este código (el leak de team-bias y la corrupción de Platt por reset), no solo un riesgo teórico.
3. Endurecer el mecanismo de reset+warm-start de `ml_state` (Platt, bias, pipeline weights) contra corrupción por procesos concurrentes/interrumpidos — hoy falla en silencio (ver hallazgo de la corrupción de Platt, sesión 2026-07-09). No urgente si se evita lanzar procesos duplicados, pero es la misma clase de fragilidad que ya causó un número incorrecto una vez.
