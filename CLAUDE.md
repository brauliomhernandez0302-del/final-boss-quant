# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**FINAL BOSS QUANT G8+** — a quantitative sports betting prediction system. It analyzes MLB, NBA, and UFC games and identifies positive-EV betting opportunities using Monte Carlo simulation, Poisson modeling, Kelly criterion sizing, and odds API integration.

## ⚠️ Motor de ML congelado — `docs/PROTOCOLO_CLV_V1.md`

Desde que el protocolo de evaluación de CLV se commiteó, **el motor de predicción de moneyline MLB está congelado en el `engine_commit` registrado en ese documento** hasta que la ventana de evaluación cierre con un veredicto. Esto significa, mientras dure la ventana:
- **No correr `scripts/promote_calibration.py`** — una promoción cambia las probabilidades live, y el protocolo exige que cualquier cambio de motor a media ventana reinicie la muestra primaria.
- No tocar ningún engine de predicción (`tte_*`, `hfa_engine.py`, `park_weather_engine.py`, `pitcher_engine.py`, `bullpen_engine.py`, `defensive_efficiency_engine.py`, `learning_engine.py`, `montecarlo/simulator.py`, `core/value_detector.py`) ni `backtest_and_retrain.py` ni los builders PIT.
- Trabajo en paralelo permitido SOLO fuera del camino de predicción de ML: mercados derivados (F5, totals), higiene de backlog, tooling, track_record (siempre que no toque probabilidades).
- Verifica `docs/PROTOCOLO_CLV_V1.md`'s sección "Registro" antes de asumir que el congelamiento sigue vigente — ahí vive la fecha de D0 y cualquier cierre de ventana.

## Running the app

```bash
# Activate the virtual environment first
source mi_entorno/bin/activate

# Launch the Streamlit app
streamlit run app.py
```

## Environment setup

Create a `.env` file in the project root:

```
ODDS_API_KEY=your_key_from_the_odds_api
OPENWEATHER_API_KEY=your_openweather_key   # optional, for weather adjustments
```

- **MLB Stats API** (`statsapi.mlb.com/api/v1`): free, no key needed
- **The Odds API**: required for live odds; loaded via `odds_fetcher.py`
- **OpenWeather**: optional; used in `data_fetchers.py` for park weather factors

## Architecture

### Entry point

`app.py` is the Streamlit entry point. It directly defines `AppConfig`/`SportConfig` (global constants and per-sport settings), `PredictionsDB` (SQLite wrapper for `data/predictions_history.db`), the NBA/UFC analyzers and renderers, and `main()` (Streamlit page layout). MLB's analyzer and renderer (`MLBAnalyzer`, `render_mlb_results`) live in `ui/mlb.py`, imported into `app.py` via `from ui.mlb import MLBAnalyzer, render_mlb_results` — not defined directly in `app.py`.

### Sport modules

Each sport has a `run_module()` function that returns `Dict[str, Any]` with keys: `status`, `game_info`/`fight_info`, `probabilities`, `lambdas_history`, `best_bets`, `metadata`.

| Sport | Module | Location |
|-------|--------|----------|
| MLB | `run_module()` | `modules/baseball_module/core/run_module.py` |
| NBA | `run_module()` | `modules/basketball_module.py` |
| UFC | `run_module()` | `modules/ufc_module.py` |
| Soccer | `run_module()` | `modules/football_module.py` (disabled) |

### MLB pipeline (most complex)

`run_module.py` orchestrates a sequential pipeline of engines, each adjusting Poisson λ (expected runs) values. `AutoCalibrator` (`calibration/auto_calibrator.py`) no longer exists — it was superseded by the Kalman + multidim-bias system below. `context_engine/pitchers_regression.py` was also deleted (disconnected from the pipeline to avoid double-counting luck-correction with SIERA/xFIP; only a stale `.pyc` remains on disk). The real, current sequence:

1. **`MLBStatsAPI`/`MLBDataIntegrator`** (`data_fetchers.py`) — fetches game data, team stats, pitchers from MLB Stats API
2. **True Talent Engine** (`offense/true_talent_engine.py`) — computes park-neutral λ_base per team from Statcast xwOBA/barrel%/plate discipline (PIT-aware, lineup-filtered)
3. **Kalman offense + team bias + pipeline weights** (inline in `run_module.py`, backed by `calibration/learning_engine.py`) — pulls λ toward each team's observed run-scoring rate and applies a learned multidimensional bias
4. **`adjust_for_pitchers`** (`context_engine/pitcher_engine.py`, PASO 2) — adjusts λ for starter SIERA/xFIP/xERA/FIP/ERA (fallback hierarchy), recent form, matchup history, platoon splits, fatigue
5. **`adjust_for_context`** (`context_engine/contextual_engine.py`, PASO 3) — rest/back-to-back fatigue, asymmetric home/away
6. **`adjust_for_bullpen`** (`context_engine/bullpen_engine.py`, PASO 4) — bullpen quality (SIERA/ERA) and workload, tier-weighted by starter's expected innings
7. **`get_adjusted_lambdas`** (`hfa/park_weather_engine.py`, PASO 5) — park run-environment factor, weather (temp/wind/rain), retractable-roof status
8. **`adjust_for_defense`** (`context_engine/defensive_efficiency_engine.py`, PASO 6) — fielding-pure adjustment via DER (1 − BABIP) and OAA, independent of pitching
9. **`get_adjusted_lambdas`** (`hfa/hfa_engine.py`, PASO 7) — away-team travel fatigue, plus a small uniform (not per-park) home-win-probability correction (`_UNIFORM_HOME_MULT`, added 2026-07-11 from a calibration diagnostic; the old per-park crowd-boost lookup stays removed, confirmed noise)
10. **`monte_carlo_advanced`** (`montecarlo/simulator.py`, PASO 8) — runs up to 5,000,000 Poisson/Negative-Binomial simulations with early stopping at `SE < 0.003`; outputs win/total/run-line probabilities
11. **`evaluate_value_ultra`** (`core/value_detector.py`, repo root — shared across sports, not under `modules/baseball_module/`, PASO 9) — Platt-2D corrects the model probability against the market before comparing to odds; calculates EV, Kelly fraction, confidence (real epistemic data-quality score, not MC sampling precision), composite score and tiers (ULTRA/HIGH/MEDIUM/SLIGHT)

### Supporting modules

- `odds_fetcher.py` — `get_odds_data()` (UI dropdown) and `get_best_odds_for_teams()` (fuzzy team name matching) both live here, using The Odds API; caches results
- `data_fetchers.py` — `MLBStatsAPI`, `MLBDataIntegrator`, weather fetcher

### Data persistence

- `data/predictions_history.db` — SQLite; predictions saved by `PredictionsDB.save()`
- `data/track_record.db` — SQLite; live pre-game picks + auto-reconciled results (`track_record/`)
- `data/mlb_complete_data.json` — cached MLB data
- `.cache/` — JSON caches for bullpen data, pitcher/team stats, odds

## Key design patterns

- **Lambda (λ) pipeline**: MLB scoring is modeled as Poisson processes. Each engine modifies λ_home and λ_away multiplicatively; the final adjusted λ values feed into Monte Carlo.
- **Fractional Kelly**: Bet sizing uses `KELLY_FRACTION=0.25` by default (quarter Kelly). `core.value_detector.kelly_criterion()` is the single source of truth — it clips the result to `[MIN_KELLY, MAX_KELLY]` = `[1%, 15%]` of bankroll (`config.py`). Every caller (pipeline `best_bets`, UI display cards, `track_record` fallback picks) delegates to this function so the cap can't drift between call sites.
- **EV calculation**: `EV = (prob × (odds - 1)) - (1 - prob)` in decimal odds. Positive EV + minimum rating threshold = actionable pick.
- **Safe module loading**: `safe_import()` in `app.py` gracefully handles missing modules; analyzers fall back gracefully when sport modules are unavailable.
- **Streamlit + terminal dual mode**: `run_module.py` detects Streamlit availability with a try/except around `import streamlit as st`; falls back to selecting the first game in terminal mode.
- **Backtest-validated calibration does not reach live automatically**: since CHRON-002, `ml_state`/`kalman_state` are split by `state_source` (`'live'` vs `'backtest'`). A model/pipeline change validated via a backtest run must be followed by `python3 scripts/promote_calibration.py --season <N> --mechanism <...> --confirm` before it's trusted in production — see `CONTRACTS.md`'s `ml_state` entry and the operational-rule callout right above the `calibration/learning_engine.py` row for the full mechanism list and TTL/promotion details.
- **The live ledger is production — testing `run_module()` needs `persist=False`**: every call writes a permanent row to `game_outcomes(source='live')` via `LearningEngine.record_prediction()`, indistinguishable from a real pick, unless you pass `persist=False` (or set `FBQ_NO_PERSIST=1` in the environment, which forces it regardless of the argument). Found live 2026-07-19 after a session of diagnostic calls wrote 46 real rows in one afternoon — see `audit_20260714/verificacion_operativa/nota_46_rows.md`.

## Estado actual

**Baseline vigente (2026-07-09): Brier 0.24525 / accuracy 55.30%** (`--season 2024,2025 --use-full-pit`, 4,830 juegos, 453 tests). Este es el primer número simultáneamente reproducible (seed determinístico), libre del leak de team-bias, y correctamente calibrado con Platt — reemplaza cualquier cifra anterior (0.24242, 0.24274, 0.24272, 0.24592).

Tres bugs de medición reales, independientes, se encontraron y corrigieron en sucesión, cada uno moviendo el número 1-2pp de accuracy:
1. **Look-ahead leak en `LearningEngine.compute_team_bias`/`compute_multidim_bias`** (`calibration/learning_engine.py`) — sin corte de fecha (`WHERE season = ?` sin `game_date <`), el bias que se multiplica a λ veía resultados reales de *toda la temporada*, incluyendo juegos posteriores al que se predecía. Corregido con un parámetro `before_date` walk-forward. El modo live nunca tuvo este leak (no existen resultados futuros en producción). Test de regresión: `tests/test_anti_leakage_opening_day_2024.py`.
2. **No-determinismo del Monte Carlo** — corregido con seed determinístico por `game_pk`.
3. **Corrupción de la calibración Platt** — un lanzamiento duplicado accidental de un proceso de backtest (matado a los segundos) alcanzó a resetear `ml_state.platt_params` a identidad antes de que el proceso real arrancara, dejando el backtest entero corriendo sin calibración Platt aplicada. No es un bug del pipeline — fue un artefacto de lanzamiento de esa sesión — pero reveló que el mecanismo de reset+warm-start no tiene protección alguna contra un proceso concurrente/interrumpido corrompiendo el cache compartido en silencio (misma clase de falla que el leak de arriba). Corregido relanzando limpio.

**Importante**: "confiable" acá significa "los bugs de medición conocidos están corregidos", no "el sistema es rentable" — el ROI de esta corrida es delgado e inconsistente (edge≥8% apenas +0.20%, edge≥10% en -2.97%), muy lejos de la conclusión de rentabilidad que sostenían los números anteriores (todos contaminados). Sigue siendo una pregunta abierta.

Nota adicional: la temporada 2024 de este backtest corre con calibración Platt en identidad por diseño (no existen datos de la temporada 2023 en esta DB para hacer warm-start) — limitación conocida y aceptada, no un bug nuevo.

**Consecuencia real, no solo cosmética**: al remover el leak, accuracy cayó de 56.27% a 54.35% y el ROI se volvió negativo para edge<10% (antes positivo desde edge≥8%). La conclusión "el sistema es rentable" que sostenían los números anteriores ya no está validada — sigue siendo una pregunta abierta con el número honesto actual.

Auditoría completa (solo lectura, componente por componente) realizada el 2026-07-06 — reporte completo en `docs/AUDITORIA_MLB_2026-07.md` (su baseline citado, 0.24242, está superseded por lo de arriba, ver nota al inicio del documento). Resumen aún válido: el pipeline de lambda (los 7 motores) está en su mejor estado histórico tras la revisión exhaustiva de esa sesión. `CONTRACTS.md` ya fue reescrito (2026-07-06, ya no es un riesgo pendiente). Ver `docs/FBQ_MASTER_BLUEPRINT.md` (v1.2) para la hoja de ruta completa.

### Actualización 2026-07-11/12 — higiene de motores + descubrimiento de caché de TTE incompleta

**Baseline vigente: Brier 0.24482 / accuracy 55.34%** (mismo comando, mismos 4,830 juegos, 453 tests) — supersede el 0.24525/55.30% de arriba. **Corrección 2026-07-18** (roadmap Step 2, FASE 6 de CHRON-001/CHRON-002): el número "0.24486/55.42%" que este documento citaba desde el 2026-07-11/12 no coincide con el último reporte real en disco de esa misma corrida (`reports/round10_composite_weight_nudge/backtest_report_20260712_0827.json`, comparado byte a byte contra dos re-corridas independientes de identidad del `--season 2024,2025 --use-full-pit`, 2026-07-17 y 2026-07-18 — las tres coinciden exactamente salvo el timestamp `run_at`). El canónico es el JSON del reporte, no la prosa — corregido aquí; ver `audit_20260714/chron001_fase6_validation_report.md` para la comparación exacta. Nueve bugs reales encontrados y corregidos, cada uno validado con una corrida de backtest completa antes/después:

1. Proxy de defensa PIT (`team_defense_pit_builder.py`): `force_out` mal clasificado como "batter safe" — la defensa sí convierte el out (sobre el corredor forzado). Explicaba casi todo un sesgo de -3pp a nivel liga.
2. Tres constantes de bullpen PIT sin calibrar contra datos reales (`_PIT_LG_K_BB`, `_PIT_LG_BARREL_PC`, `_PIT_NORMAL_IP_3D`) + un bono de fatiga por días consecutivos que saturaba en 23% de los casos (casi constante, no señal real).
3. Constante de barrel% en `tte_pit_adapter.py` (`LG_BARREL_PA`) — mismo bug ya arreglado en el motor en vivo, pero esta copia independiente del path PIT nunca recibió el fix.
4. Sub-predicción sistemática de la probabilidad de victoria de local (~1.6-1.7pp en ambas temporadas) — nuevo `_UNIFORM_HOME_MULT=0.028` en `hfa_engine.py` (uniforme, no por parque — distinto del crowd-boost por parque que sigue eliminado).
5. Sesgo de truncamiento en la capa de aprendizaje (`learning_engine.py`): el Monte Carlo no modela el walk-off del 9no inning, pero los box scores reales sí están truncados — los learners (Kalman/bias/gradient-descent) entrenaban contra la observación truncada y peleaban contra el fix #4.
6. Histéresis de warm-start de Platt entre corridas de backtest — cada corrida calibraba con el ajuste de la corrida ANTERIOR en vez del propio.
7. `ml_state` sin protección contra escritura concurrente — nuevo lockfile de archivo (mismo tipo de incidente que corrompió Platt en una sesión anterior).
8. Caché de Platt-2D en producción (`ml_state.platt2d_params`, temporada 2026) estaba ajustada sobre datos contaminados por los bugs de esta sesión — refrescada.
9. **La más grande**: `savant.team_offense.rolling` (caché PIT de ofensa por equipo) nunca terminó de construirse — se detuvo a mitad de temporada (2024-06-27, 2025-05-25) mientras defensa/bullpen sí tienen cobertura completa. Sin límite de antigüedad en la búsqueda de "última instantánea válida", el 59% de los juegos del backtest insignia usó silenciosamente datos de ofensa congelados a mitad de temporada, sin que ningún reporte de cobertura lo detectara. Arreglado con un nuevo builder incremental (`scripts/build_offense_savant_rolling_incremental.py`, O(días) en vez de O(días²)) y reconstruyendo ambas temporadas completas — también se pobló por primera vez `savant.batter.rolling` (0 filas antes), necesaria para cualquier trabajo futuro de lineup confirmado.

**Un fix diagnosticado pero revertido dos veces**: la fórmula de "dampening exacto" de `compute_team_bias_kalman_adjusted` tiene una premisa documentada como falsa, pero dos intentos de arreglarla en línea con esa premisa causaron regresiones reales del backtest (0.24479→0.24550→0.24636). Revertida a la fórmula original, que funciona como un estimador de retroalimentación cerrada con shrinkage implícito — no reintentar sin nueva información (ver postmortem completo en el docstring de la función).

### Actualización 2026-07-17 — auditoría completa (`audit_20260714/`) + remediación CHRON-001

Auditoría de solo-lectura de todo el proyecto (13 secciones: leakage, cronología, calibración,
mercado de odds, double-counting, fallbacks, matemática, operacional) — reporte completo en
`audit_20260714/`, `00_executive_summary.md` como punto de entrada. Veredicto general:
**PARTIAL** — sin leakage activo en la configuración citada como baseline, suite de 462 tests
en verde, pero con hallazgos reales pendientes (`findings.csv`), el más serio de ellos
**CHRON-001** (Alto): `game_outcomes` era compartida, sin protección, entre escrituras de
producción en vivo y sobrescrituras del backtest — un backtest de rutina que tocara un juego
en vivo ya reconciliado destruía la predicción real sin ningún rastro. Confirmado
empíricamente: 563 de 615 rows de season 2026 ya habían sido sobrescritas por una sola
corrida del 2026-06-28, ninguna recuperable de ningún backup en disco (`audit_20260714/chron001_forensics_report.md`).

**CHRON-001 remediado el mismo día** (roadmap Paso 1, `audit_20260714/14_remediation_roadmap.md`):
- Nuevas columnas en `game_outcomes`: `source` (`'live'`\|`'backtest'`\|`'import'`, fijada una
  sola vez, nunca volteada) + columnas espejo `backtest_lambda_home/away`,
  `backtest_p_home/away`, `backtest_p_home_raw/away_raw`, `backtest_stage_factors_json`.
- `backtest_and_retrain.py::update_game_outcomes()` ahora escribe SOLO en las columnas
  `backtest_*` — nunca más toca las columnas live de ningún row, sea cual sea su procedencia.
- Las 10 funciones de `learning_engine.py` que leen columnas de predicción de `game_outcomes`
  (team bias, multidim bias, Kalman-adjusted bias, gradient descent, Platt 1D y 2D, y sus
  wrappers) ahora toman `prediction_source: str = "live"` y resuelven el nombre de columna
  real vía `_pred_col()`. Cada invocación del backtest (loop principal, refits de frontera de
  temporada, refit final) pasa explícitamente `prediction_source="backtest"` — auditado call
  site por call site.
- **Validado por identidad**: re-corrida completa `--season 2024,2025 --use-full-pit` (4,830
  juegos) produjo un JSON de reporte idéntico byte a byte al último reporte real pre-cambio
  en disco (`reports/round10_composite_weight_nudge/`, 2026-07-12) — única diferencia, el
  timestamp `run_at`. El fix no movió ni un decimal del modelo. Nota honesta (**corregida
  2026-07-18**, ver la nota de baseline arriba): el resultado real (0.24482 Brier / 55.34%
  accuracy) difería de "0.24486/55.42%", que este documento citaba entonces — era una
  discrepancia preexistente entre esa prosa y el último reporte real en disco, no introducida
  por este cambio, y ya está corregida en la nota de baseline al inicio de esta sección (ver
  `audit_20260714/chron001_fase6_validation_report.md` para la comparación exacta).
- 7 tests nuevos de regresión (`tests/test_chron001_provenance.py`), 469/469 en verde.
- **0 rows recuperables** de los 563 ya sobrescritos antes de este fix — los backups
  disponibles empiezan una semana después del evento (`audit_20260714/chron001_forensics_report.md`).
  El daño histórico es permanente; el fix previene que se repita.
- Residual conocido, fuera de alcance de este paso: la caché `ml_state` de team-bias/Platt no
  está separada por procedencia — el refresco final del backtest ("step 4") sigue escribiendo
  en la misma cache key que leería una llamada en vivo, igual que antes de este fix (no es una
  regresión, ver comentario en el código).

Memoria de la sesión: `project_mlb_engine_hygiene_20260711.md`.

### Actualización 2026-07-18 — MATH-002/MATH-003 (roadmap audit_20260714, Paso 5)

**Baseline vigente: Brier 0.24483 / accuracy 55.51%** (mismo comando `--season 2024,2025
--use-full-pit`, 4,830 juegos) — supersede el 0.24482/55.34% de arriba. **Reporte canónico**:
`audit_20260714/paso5c_gate/backtest_report_20260718_1452.json` — esta ruta, no la prosa, es
la fuente de verdad; ver la nota de baseline al inicio de esta sección de CLAUDE.md sobre por
qué esa distinción importa.

Dos bugs reales de conteo (mismo mecanismo, dos implementaciones independientes) y un fix de
paridad de métrica, cada uno validado con backtest completo antes/después:

1. **`batted_ball_count` inflado por fouls** (`savant_offense_daily_aggregator.py`): contaba
   cualquier pitch con `launch_speed` trackeado, incluyendo fouls no terminales de PA —
   inflación real 1.905x sobre el conteo verdadero (16/16 muestras validadas offline,
   `audit_20260714/math002_diagnostico/`). Encontrado en DOS implementaciones independientes
   (`_aggregate_common()` y `_RawTeamAccumulator`, esta última alimentando `prior_baseline` y
   una vía alterna de `team_offense.rolling` que el diagnóstico inicial no había detectado).
   Ambas corregidas; los tres namespaces afectados (`savant.batter.rolling`,
   `savant.team_offense.rolling`, `savant.team_offense.prior_baseline`) reconstruidos 100%
   offline para ambas temporadas. Fixture permanente de regresión:
   `tests/test_math002_batted_ball_count_fixture.py`. Commit identity (nada consumía el campo
   todavía): backtest byte-idéntico salvo timestamp.
2. **Switch de MATH-002** (`tte_pit_adapter.py`): el adaptador PIT regresionaba `barrel%` como
   una métrica per-PA de punta a punta (numerador `barrel_count/plate_appearances`, prior
   `0.054` per-PA, n=`pa`) — internamente consistente pero distinta a la definición del motor
   en vivo (`barrels_sum/attempts_sum`, prior `0.088`, n=`attempts`). Corregido para igualar
   exactamente al motor en vivo (numerador, prior, Y n cambiados juntos — cambiar solo la n
   hubiera sido una quimera dimensional). `K_BARREL=120` sin tocar, ya coincidía. Delta
   acotado y aprobado por el dueño: +0.17pp accuracy, Brier esencialmente plano (+0.00001),
   solo 1 de 4,859 juegos movió su `p_home` más de 1pp (tabla completa en
   `audit_20260714/math002_diagnostico/paso5c4_delta_gate.md`).

**Residual pendiente, fuera de alcance de este paso**: el mismo bug de conteo (foul-inflation)
existe de forma independiente en el lado de PITCHERS (`savant_daily_aggregator.py`,
`pitcher_prior_baseline.py`) — no es código compartido con el lado de ofensa, pero es la misma
clase de bug. No tocado; reportado como decisión pendiente del dueño en
`audit_20260714/math002_diagnostico/paso5b1_precondiciones.md` §4.

3. **MATH-003 — análisis residual, sin implementar** (`scripts/math003_home_win_residual.py`,
   repetible): re-corrida la misma metodología que originalmente derivó
   `_UNIFORM_HOME_MULT=0.028`, sobre el baseline post-5c. Resultado: el término cierra
   consistentemente ~1.6pp de sub-predicción en ambas temporadas (como fue diseñado), pero el
   gap subyacente real NO es igual entre temporadas — 2024 queda con residual +0.08pp
   (esencialmente cerrado), 2025 retiene +1.47pp sin corregir (su gap real, ~3.10pp, es casi el
   doble que el de 2024, ~1.70pp). **Decisión abierta del dueño, no resuelta aquí** (por regla
   explícita del paso, un residual que implica otro valor se reporta, no se cambia): subir el
   valor cerraría 2025 pero sobre-corregiría 2024; ver
   `audit_20260714/math002_diagnostico/paso5d_math003_residual.md` para las opciones
   consideradas. `_UNIFORM_HOME_MULT` se queda en `0.028` sin cambios.

Ride-alongs: `run_module.py` ahora emite `logger.warning` (modo live únicamente — el backtest
nunca pasa por este entrypoint) cuando `weather_source`/`travel_source` llegan `"missing"`,
para que la próxima falla clase REG-015 aparezca en logs y no solo en metadata sin abrir.

Memoria de la sesión: `project_math002_math003_20260718.md`.

### Actualización 2026-07-19 — Fase 2B: remediación del leak V4 (día oficial en el camino PIT)

**Baseline vigente: Brier 0.24650 / accuracy 54.80%** (mismo comando `--season 2024,2025
--use-full-pit`, 4,825 juegos — 5 menos que antes, ver nota abajo) — **supersede el
0.24483/55.51% de arriba, que queda re-etiquetado como "baseline con leak V4"**, no como
número honesto del modelo. **Reporte canónico**:
`audit_20260714/fase2b/gate_delta/backtest_report_20260719_1420.json`.

**El hallazgo**: `game_outcomes.game_date` es el timestamp UTC crudo de inicio del juego
truncado a fecha — para cualquier juego nocturno que cruce medianoche UTC (la norma para
equipos de la costa oeste), eso es un día de calendario ADELANTE del día oficial real de
schedule (`officialDate` de la API de MLB). Cada uno de los 5 cutoffs PIT walk-forward de
`backtest_and_retrain.py` (Pitcher, TTE, Defense, Bullpen, más el corte general) y el corte de
entrenamiento de team-bias/Kalman (`_bias_before_date`, el mecanismo anti-leak de CHRON-001)
derivaban de ese campo contaminado — con `PITCache.get_latest()` usando `<=` inclusivo, el
snapshot resuelto terminaba incluyendo el propio día del juego que se estaba prediciendo. Leak
real, confirmado empíricamente (`audit_20260714/verificacion_operativa/reporte.md` V4,
`audit_20260714/fase2b/b0_verificacion_previa.md`), caso de prueba: `game_pk=745199`
(oficial=2024-09-18, contaminado a 2024-09-19).

**Remediación en 2 commits** (B0 de verificación read-only decidió el alcance: los 3 builders
PIT —defense, bullpen, TTE ofensa— agrupan limpio por dentro usando el `game_date` propio de
Statcast, sin componente de hora — cero rebuilds de cache necesarios; solo schema + derivación
de cutoffs):
- **B1** (identidad, 7ª de este proyecto): nueva columna `official_date`, backfileada al 100%
  desde el endpoint de schedule de la API de MLB (5520/5520 filas, 0 sin resolver; 22.2% con
  `official_date != date(game_date)` — sustancial, no ~0%, confirmando la tesis). Nada leía la
  columna todavía.
- **B2** (delta, aprobado por el dueño): los 5 cutoffs + el corte de bias/Kalman pasan a derivar
  de `official_date`. Resultado: **peor en absolutamente todas las métricas** — accuracy
  55.51%→54.80% (-0.71pp), Brier 0.24483→0.24650 (+0.00167), ROI empeora en los 5 buckets de
  edge (varios pasan de positivo a negativo). 57.2% de los juegos movieron su `p_home` más de
  0.5pp (mucho más que el 22.2% de filas nocturnas — el leak se propagaba en cascada por el
  entrenamiento walk-forward de team-bias, no solo al juego puntual contaminado). Tabla completa
  y aprobación en `audit_20260714/fase2b/b2_gate_delta_reporte.md`. **Esto es éxito, no
  fracaso** — un baseline más bajo pero honesto es exactamente lo que este roadmap se propuso
  medir; el ROI positivo que el baseline anterior mostraba en varios buckets de edge dependía en
  parte de este leak.

**Nota de conteo de juegos**: 4830→4825. Un puñado de juegos de inicio de temporada que antes
"colaban" porque el día extra del leak alcanzaba a incluir la primera snapshot disponible, ahora
correctamente no encuentran cobertura PIT previa a su propio primer juego y se excluyen —
comportamiento honesto esperado, no un bug nuevo.

Ver `audit_20260714/fase2b/` para B0/B1/B2 completos.

### Actualización 2026-07-25 — auditoría VAL + rebaseline del simulador

**Baseline vigente: Brier 0.24675 / accuracy 55.05%** (mismo comando `--season 2024,2025
--use-full-pit`, 4,825 juegos) — supersede el 0.24650/54.80% de arriba. **Reporte canónico**:
`audit_20260714/val_audit/rebaseline/backtest_report_20260725_0644.json`; delta completo contra
el canónico anterior en `audit_20260714/val_audit/rebaseline/reporte_delta.md`.

**⚠️ Framing obligatorio de este número — leer antes de citarlo en cualquier prosa futura**:
esto es un **baseline del motor corregido, neutro en moneyline — la justificación del fix vive
en runline/total, que este backtest no mide**. `backtest_and_retrain.py` evalúa
*exclusivamente* moneyline (Brier/accuracy/ROI sobre `home_won`), y el sesgo que estos fixes
corrigen vive en la **cola** de la distribución de carreras (margen y total), no en el signo de
quién gana. El movimiento observado en moneyline es pequeño y sin dirección clara: Brier +0.00025
(peor), accuracy +0.25pp (mejor), ningún juego movió su `p_home` más de 1.82pp. **Ninguna prosa
futura debe presentar este rebaseline como una mejora del modelo** — este instrumento no puede
mostrar una mejora aunque exista, y tampoco mostró un deterioro. La evidencia de que el fix
corrige un sesgo real es la del reporte VAL (`audit_20260714/val_audit/reporte.md` VAL-1.3): sobre
881 juegos reales, la brecha real-vs-modelo en P(margen local ≥2 | ganó) bajó de ~8.7pp a ~0.9pp
— y eso es sobre mercados que este backtest no evalúa.

Cambios del motor incluidos en el rebaseline (todos derivados de la auditoría VAL,
`audit_20260714/val_audit/reporte.md`):
1. **Truncamiento de walk-off** (`montecarlo/simulator.py`, VAL-1.3): el simulador no modelaba
   que la baja del 9no no se juega si el local ya va arriba. Implementado como binomial thinning
   con `WALKOFF_9TH_SHARE=1/9` — **constante asumida, no ajustada empíricamente** (no existe un
   split real por inning en este repo todavía); nombrada explícitamente para que una calibración
   futura tenga dónde enchufar el número real. Flag `model_walkoff=True` por default; `False`
   reproduce el comportamiento viejo para comparaciones analíticas.
2. **`rho_game` ahora sí llega a las carreras** (mismo archivo, hallazgo bonus de VAL-1): la
   versión anterior aplicaba la correlación solo al ruido de λ (~5% de λ), y quedaba tragada por
   la varianza condicional del NB — ρ=-0.008 de input daba ρ≈-0.0007 observado (~11x de
   atenuación), y hasta ρ=-0.5 solo daba -0.0024. Reimplementado compartiendo una unidad de la
   mezcla Gamma del NB con probabilidad derivada en forma cerrada. La marginal NB de cada lado se
   preserva EXACTAMENTE. Una versión con cópula gaussiana exacta acierta el target mejor pero
   depende de `nbinom.ppf`/`gamma.ppf` de scipy, ~22x más lento (6s → 34s en una corrida de 5M) —
   descartada por costo; la actual queda dentro de ~10-15% del target.
3. **Resolución de empates proporcional** (mismo archivo): un empate en la simulación representa
   un juego real que iría a extra innings. Antes 50/50 implícito; ahora el crédito se reparte
   según el λ_noise de esa misma simulación, así que un empate entre desiguales se inclina hacia
   el favorito. `p_home + p_away` sigue sumando exactamente 1.0.
4. **CI de bootstrap al n real** (`core/value_detector.py`, VAL-1.4): submuestreaba a 10,000
   draws fijos sin importar cuántas simulaciones corrieron (500K-5M live), reportando un CI
   hasta ~7-22x más ancho que el real — y eso alimentaba `ev_std`, `sharpe` y el
   `sharpe_component` del `composite_score`. Para arrays binarios (todos los callers actuales)
   ahora usa la forma cerrada Binomial(n, p̂)/n al n verdadero.
5. **Transparencia de Kelly** (mismo archivo, VAL-4.4): el piso `MIN_KELLY=1%` puede inflar un
   edge marginal (caso real: quarter-Kelly 0.28% → 1.00%, 3.6x). Es diseño deliberado, no bug;
   ahora `analyze_market_generic` expone además `kelly_unfractional` y `kelly_floor_applied`
   para que se vea cuándo el piso cambió el stake. `kelly_criterion()` sigue siendo la única
   fuente de verdad para sizing.
6. **Tres conteos de relevistas etiquetados** (`context_engine/bullpen_engine.py` +
   `api/mlb_presentation.py`, VAL-7.2): `n_pitchers` (cobertura Savant), `n_siera_pitchers`
   (cobertura FanGraphs) y el roster que ve la UI son tres números distintos por construcción,
   ninguno afecta λ directamente, y se mostraban sin etiqueta. Documentados en el sitio; además
   `fetch_bullpen_roster` ahora reusa `_fetch_reliever_ids` del propio engine en vez de un filtro
   "posición == P" que colaba abridores de rotación (confirmado live: Buehler, King, Márquez,
   Sears).

**Tests permanentes**: `tests/verification/test_val_audit_invariants.py` (17 tests) — incluye dos
que fueron escritos rojos a propósito contra los bugs de VAL-1.3 y VAL-1.4 y que estos fixes
ponen en verde.

**`promote_calibration.py`: NO se corrió — decisión tomada, no pendiente.** Razón operativa: lo
que promovería no es lo que hace falta. La Platt-1D live vigente es `season=2026`,
`state_source='live'`; esta corrida solo re-ajustó Platt de `season=2024/2025`,
`state_source='backtest'`, y `promote_calibration.py --season N` mueve backtest→live *de esa
misma temporada* — no existe una Platt de backtest 2026 fresca que promover. Ver
`audit_20260714/val_audit/rebaseline/reporte_delta.md` §8 para las otras dos razones (protocolo y
magnitud) y para qué sí correspondería hacer.

**Protocolo CLV**: `engine_commit` de `docs/PROTOCOLO_CLV_V1.md` re-apuntado a este rebaseline.
D0 sigue **pendiente**, así que la ventana no había arrancado y este cambio de motor no reinicia
ninguna muestra primaria — pero fijar D0 debe ocurrir *después* de este motor, nunca antes.

### Actualización 2026-07-26 — evaluador distribucional de runline/total (`derived_eval`)

**Baseline sin cambios** (Brier 0.24675 / accuracy 55.05%, reporte canónico
`audit_20260714/val_audit/rebaseline/backtest_report_20260725_0644.json`). Este trabajo no toca
ningún engine ni ninguna λ: es el instrumento que faltaba para medir el fix del simulador sobre
los mercados que sí corrige.

**`audit_20260714/val_audit/derived_eval/evaluate_derived_markets.py`** (repetible, solo lectura)
re-simula las `game_outcomes.backtest_lambda_*` guardadas de la corrida canónica —4,825 juegos,
2024/2025— con el simulador **pre-fix** (`b3325a5^`, vendorizado y verificado contra git en cada
corrida) y con el **actual**, mismo seed por `game_pk`, y compara ambas distribuciones contra
`actual_home_runs`/`actual_away_runs`. Las λ son entrada del simulador, así que esto aísla los
tres cambios de b3325a5 de forma exacta. Reporte: `audit_20260714/val_audit/derived_eval/reporte.md`.

**CALIBRACIÓN, NO ROI** — no existen líneas históricas de runline/total en esta DB (`game_outcomes`
solo guarda moneyline), así que la rentabilidad de estos mercados sigue sin medirse y ningún
número de ese reporte debe citarse como tal.

Resultados (los tres primeros, evidencia positiva del fix; el cuarto, el residual que queda):
1. **VAL-1.3 replicado a escala completa**: P(margen local ≥2 | ganó el local) pasa de +11.58pp a
   +3.05pp de sobre-predicción. **+3.05pp es el número canónico** — el ~0.9pp que cita el reporte
   del fix mide otro estimando (la ventana de λ del hallazgo, `5.0≤λ_h≤7.5`/`2.5≤λ_a≤4.5`, con NB
   puro y 300-500 sims). El evaluador reproduce ese par en esa misma ventana (+8.90pp → +1.23pp,
   N=813); la brecha poblacional es mayor porque la ventana selecciona juegos donde el local es
   muy favorito, justo donde el truncamiento más muerde. Ver §2.1 del reporte.
2. **Totales**: sesgo de la línea 7.5 +5.72pp → +3.22pp, 8.5 +2.80pp → **+0.44pp**, 9.5 +3.27pp →
   **+0.57pp**; total medio 9.112 → 8.850 contra 8.837 real. Brier pareado significativo en las
   cuatro líneas centrales; ECE mejora en 4 de 6.
3. **Moneyline (control)**: −0.00037 de Brier, minúsculo — reproduce la neutralidad que ya
   reportaba el rebaseline, confirmando que el instrumento mide lo que dice medir.
4. **El residual dominante ya no es el walk-off: es sub-dispersión.** El margen simulado tiene
   desviación típica 3.87 vs 4.50 real (−14%); el total, 4.04 vs 4.46 (−9%), con PIT en U. Se
   origina en `NB_DISPERSION=6.0`, elegido en su momento *a propósito* por métricas de moneyline
   pese a un peor ajuste de cola (está documentado en el docstring del simulador).

**Dos constantes DECIDIDAS el 2026-07-26 (§7 del reporte) — no son pendientes**:
- **`WALKOFF_9TH_SHARE` se queda en `1/9`**, decidida con la evidencia del barrido
  (`--walkoff-sweep`): ningún valor único satisface los cuatro criterios —RL_HOME pide ≈0.033,
  margen medio ≈0.093, total medio ≈0.118 (donde ya está), VAL-1.3 ≈0.165—, y 1/9 centra el total
  medio, el objetivo con más datos y sin condicionar. Residuales aceptados: RL_HOME −2.65pp,
  VAL-1.3 +3.05pp. **Disparador de re-visita: cuando se toque la dispersión** (esos residuales se
  estiman sobre una forma que está −9%/−14% angosta; re-correr el barrido es obligatorio después).
- **`NB_DISPERSION` se queda en `6.0` durante la ventana de CLV de moneyline** — está en el camino
  congelado y su valor se eligió justo por las métricas que la ventana está midiendo.
  **Disparador de re-visita: el arranque del motor de derivados**, donde la balanza cambia de lado
  y este evaluador es la balanza (el costo distribucional de r=6.0 ya está medido).

Memoria de la sesión: `project_derived_eval_20260726.md`.
