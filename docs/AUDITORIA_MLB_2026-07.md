# AUDITORÍA MLB — FINAL BOSS QUANT G8+

**Fecha:** 2026-07-06
**Alcance:** Motor de predicción MLB completo (modelo, mercado, tracking) y su infraestructura de datos conectada. Excluye NBA/UFC por instrucción explícita.
**Modo:** Solo lectura y documentación — ningún archivo de código fue modificado durante esta auditoría.
**Baseline de referencia:** Brier 0.24242 / accuracy 56.87% (backtest limpio 2026-07-06, `--season 2024,2025 --use-full-pit`, 448 tests). Este número en sí es **provisional** — ver hallazgo #2.

> **⚠️ ACTUALIZACIÓN 2026-07-09 — el baseline de este documento está SUPERSEDED, no solo desactualizado.**
> El hallazgo #2 de abajo ("el backtest no incorpora los fixes de motor") resultó ser el síntoma menor de algo más grave: `LearningEngine.compute_team_bias`/`compute_multidim_bias` tenía un look-ahead leak real (sin corte de fecha — veía resultados de toda la temporada, incluyendo juegos futuros respecto al que se predecía). Corregido con un parámetro `before_date` walk-forward, verificado empíricamente y cubierto por `tests/test_anti_leakage_opening_day_2024.py`. Un segundo problema (corrupción de la calibración Platt por un lanzamiento duplicado accidental) se encontró y corrigió justo después. **Baseline final, triple-limpio (reproducible + sin leak + calibrado): Brier 0.24525 / accuracy 55.30%** — bajó respecto al 0.24242 original, porque ese número estaba inflado por el leak; subió respecto al 0.24592 intermedio, porque ese estaba sub-calibrado por la corrupción de Platt. **ROI quedó delgado e inconsistente** (ni claramente rentable ni claramente no rentable — ver `docs/FBQ_MASTER_BLUEPRINT.md` v1.3). Cualquier cifra de este documento (0.24242, y las conclusiones de rentabilidad que dependen de ella) debe leerse como histórica, no como estado actual. Ver `CLAUDE.md` → "Estado actual" y `docs/FBQ_MASTER_BLUEPRINT.md` v1.3 para el número vigente.

---

## 1. RESUMEN EJECUTIVO

El pipeline de lambda (TTE → Kalman → Pitcher → Contextual → Bullpen → Park+Weather → Defense → HFA → Monte Carlo) está, motor por motor, en el mejor estado de toda la historia del proyecto — cada uno recibió una revisión profunda esta sesión con hallazgos reales corregidos y verificados en vivo. El riesgo real ya no está concentrado ahí. Está en tres lugares que una auditoría centrada solo en "el modelo" pasaría por alto: (1) **`CONTRACTS.md` describe un sistema que ya no existe** — cita archivos borrados, líneas incorrectas, y un componente (`AutoCalibrator`) eliminado hace tiempo, lo que lo vuelve activamente engañoso como documentación de referencia; (2) **el backtest que produce el Brier "honesto" citado en todo este proyecto todavía no incorpora los fixes de motor de esta sesión** — el re-backtest sigue diferido, así que el número 0.24242 mismo es una instantánea desactualizada del día en que se corrió, no una validación de dónde está el pipeline hoy; (3) **la capa de tracking de picks (`track_record/`) tuvo, hasta hoy, un bug que hacía que el P&L de cada pick resuelto se calculara con una cuota falsa fija en vez de la real** — no es un problema de modelo, es un problema de medición, pero como la métrica norte del proyecto es CLV/ROI real, esto invalidaba silenciosamente la única prueba que el propio blueprint exige antes de confiar en el sistema.

El componente más sólido: el **Monte Carlo Negative Binomial** (`montecarlo/simulator.py`) — ya calificado como el mejor implementado en auditorías previas, confirmado de nuevo esta sesión (parada temprana, correlación bivariada, F5_SCALE validado empíricamente contra datos reales de 60 juegos).

---

## 2. TABLA MAESTRA

| Componente | Archivo:función | Validación | PIT-safety | Riesgo | Mejora propuesta | Fase blueprint |
|---|---|---|---|---|---|---|
| Distribución base (NB) | `montecarlo/simulator.py::monte_carlo_advanced`, `NB_DISPERSION=6.0` (L.29) | VALIDADO | SEGURO | Bajo | Ninguna urgente; extender early-stopping a O/U y F5, no solo p_home | 1.5 |
| TTE (ofensiva) | `offense/true_talent_engine.py::get_lambda()` | VALIDADO | SEGURO (PIT 99.4%) | Bajo | Ninguna pendiente crítica | — |
| Kalman + multidim_bias | `calibration/learning_engine.py` | VALIDADO | SEGURO | Bajo | Grid search de `_KALMAN_BLEND=0.35` (sin validar, no urgente) | 3.2 |
| Pitcher Engine | `context_engine/pitcher_engine.py` | VALIDADO | SEGURO (PIT 96.5%+3.1%+0.9%) | Bajo | Ninguna pendiente crítica | — |
| Contextual Engine | `context_engine/contextual_engine.py` | VALIDADO | SEGURO | Bajo | Ninguna — revisado dos veces, "fine as-is" | — |
| Bullpen Engine | `context_engine/bullpen_engine.py` | VALIDADO | SEGURO (PIT 100%) | Bajo | Derivar `_K_BARREL_BP` real (constante interina) | 1.1 |
| Park + Weather Engine | `hfa/park_weather_engine.py` | VALIDADO | SEGURO (solo ruta viva; backtest deliberadamente ciego, ver ficha) | Medio | Revisar clamp combinado `[0.90,1.12]` bajo datos reales de parques ventosos | — |
| Defensive Efficiency Engine | `context_engine/defensive_efficiency_engine.py` | VALIDADO | SEGURO (PIT 100%) | Bajo | Re-derivar split de peso DER/OAA tras el fix de denominador | — |
| HFA Engine | `hfa/hfa_engine.py` | VALIDADO | SEGURO | Bajo | Ninguna — "fine as-is" | — |
| Devig (3 métodos) | `core/value_detector.py::remove_vig_multiplicative/power/shin` (L.107-127) | PARCIAL | N/A (no es dato histórico) | Bajo | Comparar cuál método da mejor CLV real (requiere 1.4) | 2.3 |
| EV / Composite Score / Tier | `core/value_detector.py::calculate_composite_score/classify_value_tier` | VALIDADO (recién recalibrado) | N/A | Medio (recién cambiado, sin ciclo completo de validación en producción) | Confirmar bajo el próximo re-backtest | — |
| Kelly | `core/value_detector.py::kelly_criterion` | VALIDADO | N/A | Bajo | Ninguna | — |
| Platt / Platt-2D | `calibration/learning_engine.py` | VALIDADO | SEGURO (walk-forward, expanding window) | Bajo | Ninguna crítica; ver nota de circularidad en Fase C | 3.2 |
| Pesos de pipeline (GD) | `calibration/learning_engine.py::get_pipeline_weights` | VALIDADO | SEGURO | Bajo | Ninguna | — |
| Capa de ingesta de datos | `data_fetchers.py` | VALIDADO | SEGURO (ver 3.2, corrección post-Fable) | Bajo | Ninguna crítica pendiente | — |
| Capa de datos PIT | `advanced_pit_enrichment/` (26 archivos) | SIN VALIDAR (nunca auditado formalmente) | PROBABLEMENTE SEGURO (construido con ese fin) | Medio | Auditoría formal archivo-por-archivo | 0.1 |
| Backtest/training path | `backtest_and_retrain.py` | VALIDADO (corregido post-Fable, ver 3.1) | SEGURO (gate `use_team_full_season_pitching_base` confirmado en código) | Bajo-Medio (fidelidad, no leak) | Re-correr backtest batched con fixes de motor de hoy (sigue diferido) | 1.1 |
| Adquisición de odds | `odds_fetcher.py`, `ui/odds_loader.py` | VALIDADO (recién corregido) | N/A | Bajo (post-fix) | Ninguna crítica pendiente | — |
| Tracking de picks | `track_record/publisher.py/reconciler.py/db.py` | VALIDADO (recién corregido) | N/A | Bajo (post-fix) | Correr en producción real 30+ días para primera medición honesta | 1.4 |
| `CONTRACTS.md` | — | **ROTO como documentación** | N/A | Alto (induce a error a cualquiera que lo use como mapa) | Reescribir completo | — |

---

## 3. FICHAS DETALLADAS (componentes con hallazgos, no repetidos de la tabla si ya están "sin riesgo")

### 3.1 `backtest_and_retrain.py` — CORREGIDO tras revisión de Fable, ver nota

**Corrección post-revisión (Fable, 2026-07-06):** la versión anterior de esta ficha afirmaba que el leak de `team_era`/`team_whip`/`runs_allowed_per_game` en `_team_dict()` seguía abierto. Verificado directamente contra el código (`backtest_and_retrain.py` líneas 446-542, 2741-2747) — **esto es falso, el gap ya está cerrado**: el flag `use_team_full_season_pitching_base` (default `True`, puesto en `False` cuando se pasa `--use-defense-pit` vía `use_team_full_season_pitching_base=not args.use_defense_pit`, línea 2747) gatea exactamente estos 3 campos para caer a `LEAGUE_AVG_ERA`/`LEAGUE_AVG_WHIP`/`LEAGUE_AVG_RUNS` en modo PIT, en vez de leakear el agregado de temporada completa — el mismo patrón que `use_team_full_season_defense` ya aplicaba a DER/OAA. El comentario en el propio código (línea 529-534) documenta esto explícitamente como un fix ya hecho, no como un TODO. Este hallazgo fue un error de esta auditoría, no un hallazgo real del sistema — corregido antes de tratarlo como autoritativo.

- **Estado de validación real:** VALIDADO. El gate existe y funciona como se espera.
- **PIT-safety real:** SEGURO en modo `--use-defense-pit` (cae a promedios de liga, no a leak). Nota de diseño, no un bug: en modo PIT, `team_era`/`team_whip`/`runs_allowed_per_game` se degradan a constantes de liga en vez de calcularse a partir de datos reales point-in-time del equipo — es una simplificación que preserva la corrección temporal (no hay leak) pero sacrifica precisión (un valor real point-in-time sería mejor que el promedio de liga). No es el mismo tipo de problema que un leak; es una pregunta de fidelidad, no de validez.
- **Riesgo:** Bajo-Medio (por la pérdida de fidelidad, no por leak).
- **Mejora atómica propuesta (revisada):** Si se busca más precisión que la constante de liga, construir una versión point-in-time de `get_team_pitching_stats()` (mismo patrón que los 4 dominios PIT ya existentes) en vez de degradar a promedio — pero esto es una mejora de fidelidad, no un fix de corrección urgente. Sin acción crítica pendiente aquí.

**Punto que SÍ sigue siendo válido y no fue afectado por esta corrección:** el re-backtest batched que incorpora los fixes de motor de esta sesión (bullpen SIERA/IP, defense OAA/games-played, roof_open, LG_XWOBA, LG_DER) sigue diferido — el Brier 0.24242 citado en este documento es real pero no incorpora esos cambios. Ver punto 2 del resumen ejecutivo, que se mantiene sin cambios.

### 3.2 Capa de ingesta de datos (`data_fetchers.py`) — hoy con hallazgos reales de código muerto

- **Estado de validación:** PARCIAL. La sesión de hoy encontró y eliminó 5 fetchers completamente muertos (`get_head_to_head`, `get_standings_status`, `get_team_offensive_stats`, `get_team_recent_form`, `get_umpire_historical_stats`) — costo real de red en cada predicción en vivo, cero valor downstream. También se simplificó el diccionario `home_team`/`away_team` de `run_module.py` de 13 campos a 2 (solo `rest_days` se consumía realmente).
- **PIT-safety:** SEGURO — el gate en `_team_dict()` (`backtest_and_retrain.py`, ver 3.1 corregida) confirma que el consumo de `get_team_pitching_stats()` para el backtest ya está protegido correctamente en modo PIT.
- **Riesgo:** Bajo — el código muerto no distorsiona probabilidades (nunca llegaba a nada); el punto de PIT que originalmente se marcó como riesgo aquí resultó ser un error de esta auditoría (ver corrección en 3.1).
- **Mejora atómica propuesta:** Ya ejecutada hoy (eliminación de código muerto). Sin acción crítica pendiente.

### 3.3 `CONTRACTS.md` — documentación activamente engañosa

- **Estado de validación:** ROTO como documento de referencia. Generado 2026-05-13, antes del rebuild PIT completo, antes de la corrección de confianza, antes de Platt-2D, antes de toda la sesión de hoy.
- **Evidencia concreta, verificada línea por línea hoy:**
  - Describe `odds_api.py` como un archivo real de 93 líneas con `get_best_odds_for_teams()` — **este archivo no existe**, la función siempre vivió en `odds_fetcher.py`.
  - Describe `storage.py` como activo, usado por "app.py sidebar 'Save Picks' button" — **confirmado hoy con cero llamadores en todo el repo**, eliminado en esta sesión.
  - Describe `app.py` como 1,591 líneas con `MLBAnalyzer`/`render_mlb_results` definidos directamente — **`app.py` real son 305 líneas**; esa lógica vive en `ui/mlb.py`, no mencionado en absoluto en `CONTRACTS.md`.
  - Cita `ablation_calibrator.py`, `post_game.py`, `train_historical.py` como archivos activos — **ninguno existe en el repo actual**.
  - Cita `config.py` con `MAX_RISK_PCT=0.05` — **esta constante no existe** en el `config.py` real (54 líneas, verificado completo).
  - El flujo de datos MLB "canónico" (sección 4 de `CONTRACTS.md`, no reproducido aquí por brevedad pero verificado) presumiblemente sigue describiendo `AutoCalibrator` como paso activo del pipeline — componente que ya no existe, superseded por Kalman+multidim_bias.
- **Riesgo:** Alto — no porque afecte al pipeline (no lo toca), sino porque cualquier persona (o agente) que use `CONTRACTS.md` como mapa del sistema va a operar sobre información falsa con alta confianza, exactamente el mismo patrón que ya causó que el `FBQ_MASTER_BLUEPRINT.md` original quedara desactualizado casi de inmediato.
- **Mejora atómica propuesta:** Reescritura completa de `CONTRACTS.md` contra el estado real verificado hoy — no un parche, dado el volumen de secciones afectadas. Sugerido como acción de FASE 0 aunque no estaba en el blueprint original (ver sección 6).

### 3.4 `track_record/` — corregido hoy, aún sin ciclo de validación en producción

- **Estado de validación:** VALIDADO a nivel de código (6 bugs reales corregidos y probados hoy: bankroll `MAX()`, colisión de `_market_label()` para F5, esquema `all_opportunities` sin odds/model_prob/line causando payout hardcodeado de 1.909, λ final mostrando la etapa equivocada, confidence siempre 0.5 en `save_value_picks()`, crash de `json.dumps` por arrays de NumPy). Pero la tabla `picks` de `track_record.db` está **vacía** — cero picks reales han pasado por el sistema corregido todavía.
- **Riesgo:** Bajo ahora que está corregido, pero el criterio de salida real (blueprint 1.4: "30 días de picks en producción con CLV positivo documentado") no puede empezar a medirse hasta que corra en vivo.
- **Mejora atómica propuesta:** Ninguna de código — correr en producción real y dejar acumular datos.

---

## 4. DIAGNÓSTICO: MÉTODO DE COMBINACIÓN REAL (no es media geométrica)

Verificado directamente en el código de los 7 motores del pipeline lambda: **ningún punto de combinación usa media geométrica.** El patrón real, consistente en TODOS los puntos donde `run_module.py` aplica el peso aprendido de un motor sobre la lambda corriente, es un **blend delta-ponderado**:

```python
# run_module.py, patrón repetido idéntico en cada PASO (ej. L.492-496, Pitcher):
_raw_h = lh_post_engine / lh_pre_engine
_w = _weights.get(stage_name, 1.0)          # peso aprendido por gradient descent
lh = lh_pre_engine * (1.0 + _w * (_raw_h - 1.0))
```

Es decir: `λ_out = λ_in × (1 + w×(ratio_crudo − 1))` — el motor produce un ratio crudo (cuánto cambiaría la lambda si su peso fuera 1.0), y el peso aprendido escala cuánto de ese cambio se aplica realmente. Con `w=1.0` esto es multiplicación directa; con `w<1.0` amortigua el efecto del motor; con `w>1.0` lo amplifica.

**Dentro de cada motor individual**, el método varía y es internamente consistente pero NO geométrico:
- Pitcher Engine: producto directo de sub-factores (`skill × woba_factor × brl_factor × kbb_mult × ...`).
- TTE, Bullpen, Defense: suma ponderada aditiva de factores normalizados a 1.0 (ej. Bullpen: `xwoba_factor×0.55 + era_factor×0.35 + barrel_factor×0.10`).
- Contextual, HFA: multiplicadores discretos aplicados directamente (no hay "combinación" de múltiples sub-factores, son factores únicos por condición).

**Conclusión:** no hay inconsistencia que corregir — el diseño real (delta-ponderado entre motores, suma/producto según corresponda dentro de cada motor) es coherente y deliberado. La premisa original de "verificar consistencia de media geométrica" simplemente no aplicaba a este código.

---

## 5. DIAGNÓSTICO DE DEFENSE MULTIPLIERS

**Ya no fallan.** Confirmado en vivo el 2026-07-06: `pytest tests/test_defense_multiplier.py -v` → 8/8 PASSED. La causa raíz histórica (antes de esta sesión) fue una desalineación de `_LG_DER` (constante de liga usada para regresión bayesiana hacia la media) — el valor hardcodeado de 0.715 no coincidía con el promedio real medido vía MLB Stats API (0.7097 tanto en 2024 como 2025), encontrado y corregido en una sesión previa a la de hoy. Sin acción pendiente sobre esto específicamente.

---

## 6. LISTA PRIORIZADA DE ACCIONES (impacto en Brier × riesgo actual)

1. **[Alto impacto documental / riesgo de proceso] Reescribir `CONTRACTS.md` completo.** No afecta al pipeline, pero cualquier decisión futura (humana o de un agente) que se apoye en él como mapa del sistema va a estar mal informada. Criterio de aceptación: cada afirmación verificable contra el código real, sin excepciones.
2. **[Medio impacto] Re-correr el backtest batched incorporando TODOS los fixes de motor de esta sesión** (bullpen SIERA/IP, defense OAA/games-played, roof_open, LG_XWOBA, LG_DER) — de una sola vez, no incrementalmente. Este es el único punto pendiente real sobre la validez del Brier 0.24242 citado en este documento (el gap de `get_team_pitching_stats()` originalmente listado aquí como acción #1 resultó ser un error de esta auditoría — ver corrección en 3.1 — el gate ya existe y funciona).
3. **[Medio impacto] Auditoría formal archivo-por-archivo de `advanced_pit_enrichment/`** (26 archivos, fase 0.1). Se tocó extensivamente construyendo Pitcher PIT pero nunca se revisó con el mismo rigor que los 7 motores del pipeline lambda.
4. **[Bajo impacto inmediato / alto valor a 90 días] Dejar correr `track_record/` en producción real** 30+ días para la primera medición honesta de CLV — es el criterio de salida explícito de la fase 1.4 del blueprint y la única prueba que el propio proyecto define como decisiva.
5. **[Bajo impacto, bajo esfuerzo] Renombrar `stage_factors_json`** (inversión de naming home/away en pitcher/defense/bullpen, ya documentada vía TODOs) — no afecta la corrección del gradient descent (que es auto-consistente), pero ya causó una confusión real en análisis externo esta sesión.
6. **[Bajo impacto, opcional] Construir versión point-in-time real de `get_team_pitching_stats()`** para el backtest, en vez de degradar a promedios de liga en modo PIT (ver 3.1) — mejora de fidelidad, no fix de corrección; el estado actual ya es seguro, esto solo lo haría más preciso.

---

## 7. ACTUALIZACIÓN SUGERIDA DE CONTRACTS.md (solo sugerencia — no se editó el archivo)

Dado el volumen de drift encontrado (sección 3.3), la sugerencia no es un parche sino una regeneración completa, con estas correcciones mínimas no negociables:
- Eliminar toda referencia a `odds_api.py`, `storage.py`, `ablation_calibrator.py`, `post_game.py`, `train_historical.py` (no existen).
- Corregir `app.py` a su tamaño y responsabilidad real (305 líneas, glue únicamente; MLB vive en `ui/mlb.py`).
- Eliminar toda referencia a `AutoCalibrator` como componente activo; documentar el reemplazo real (Kalman + multidim_bias en `calibration/learning_engine.py`).
- Añadir `track_record/` (publisher.py/reconciler.py/db.py/stats.py/ui.py), `ui/odds_loader.py`, `db/predictions_db.py`, y `advanced_pit_enrichment/` como secciones propias — ninguno aparece en la versión actual.
- Añadir el orden PASO 0-9 real del pipeline (documentado con números de línea en este mismo informe, sección FASE A del proceso de auditoría).

---

## 8. FASE C — CAPA DE PICKS (rastreo manual de picks reales)

`track_record.db::picks` está vacía (0 filas) — el sistema corregido hoy aún no ha procesado ningún pick real en producción. Se usó en su lugar `data/predictions_history.db::predictions` (59 filas históricas, todas de 2026-04-12, previas al rebuild PIT completo y a todos los fixes de esta sesión).

**Hallazgo real, no esperado:** en las 3 filas más recientes rastreadas a mano (ids 57, 58, 59), el campo `confidence` almacenado es **idéntico** al `p_home` o `p_away` del lado elegido, no a ninguna señal de confianza epistémica real:

| id | pick | p_home | p_away | confidence guardado |
|---|---|---|---|---|
| 59 | Dodgers (home) ML | 0.7302 | 0.2698 | 0.7302 — igual a p_home |
| 58 | Diamondbacks (away) ML | 0.4389 | 0.5611 | 0.5611 — igual a p_away |
| 57 | Athletics (home) ML | 0.5261 | 0.4739 | 0.5261 — igual a p_home |

Esto es consistente con (aunque no idéntico a) el bug de `confidence` encontrado y corregido hoy en `ui/mlb.py::save_value_picks()` (`bet.get("confidence", 0.5)` sin fallback viable) — pero estos valores NO son 0.5 constante, son el model_prob del lado. **No puedo verificar con certeza absoluta la ruta de código exacta de abril 2026** que produjo este patrón específico (posiblemente una versión anterior de `save_value_picks()` con una asignación distinta, o un mix-up de variable `confidence`/`p_home` en ese momento) — marcado como PREGUNTA ABIERTA. Lo que sí es cierto, verificado hoy: con el código actual, este campo ya recibe la confianza epistémica real (`compute_data_quality_confidence`) vía el fix de propagación completa de `all_opportunities` implementado esta sesión — cualquier pick nuevo que se guarde de aquí en adelante no debería reproducir este patrón. **Nota explícita (revisión de Fable):** este patrón histórico NO refleja el comportamiento actual del pipeline — verificado múltiples veces hoy que `compute_data_quality_confidence()` calcula una señal real e independiente de `model_prob`. Un lector futuro de esta sección de preguntas abiertas no debe interpretar esto como algo que sigue ocurriendo. Recomendado: guardar 3 picks nuevos reales tras el fix y confirmar que `confidence` ya no coincide trivialmente con `model_prob`.

**EV/Devig:** confirmado, `ev = model_prob × (odds−1) − (1−model_prob)`, decimal odds, coincide con la fórmula documentada en `CLAUDE.md` y con `core/value_detector.py::calculate_ev`.

**Kelly:** confirmado, `kelly_criterion()` es la única fuente de verdad (`core/value_detector.py`), clip a `[MIN_KELLY=0.01, MAX_KELLY=0.15]`; `save_value_picks()` en `ui/mlb.py` reescala por `kelly_factor/DEFAULT_KELLY_FACTOR` pero nunca bypasea el clip. Ningún camino de código encontrado hoy genera un pick saltándose `classify_value_tier`'s gate de EV/confidence/edge negativos.

**Circularidad del Learning Engine:** sin evidencia encontrada en la revisión de hoy — Platt/Platt-2D calibran contra `home_won` (resultado real independiente), Kalman y gradient descent entrenan contra `actual_home_runs`/`actual_away_runs`. No fue el foco específico de la revisión de hoy, así que se deja como PREGUNTA ABIERTA para verificación dedicada, no como cerrada. **Chequeo concreto recomendado para cerrar esta pregunta** (Fable): verificar que ningún input a `recalibrate_platt`/`recalibrate_platt_2d`/`_gradient_step` deriva circularmente de un output ya calibrado por el mismo mecanismo, en vez de un outcome real independiente — esto le da a quien retome esta pregunta un paso accionable concreto, no solo un gap reconocido.

---

## 9. PREGUNTAS ABIERTAS

1. ¿Por qué el `confidence` histórico de `predictions_history.db` (abril 2026) coincide exactamente con el model_prob del lado elegido? (sección 8, no se pudo verificar la ruta de código exacta de esa fecha).
2. ¿Sigue existiendo circularidad en algún punto del Learning Engine no cubierto por la revisión de hoy? (sin evidencia encontrada, pero no fue el foco específico de ninguna revisión).
3. La referencia original a "4 fuentes de look-ahead identificadas" en la v1.0 del blueprint — ¿corresponde exactamente a los 4 dominios PIT (TTE/Bullpen/Defense/Pitcher), o hay una quinta fuente documentada en `docs/AUDIT_FINDINGS.md` que esta auditoría no confirmó directamente? Requiere lectura cruzada de `docs/AUDIT_FINDINGS.md` línea por línea, no hecha en esta pasada.
