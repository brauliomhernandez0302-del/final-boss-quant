# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🔴 MODO RECONSTRUCCIÓN — vigente desde el 2026-08-05

**Esta sección gobierna sobre TODO lo demás de este archivo. Ante cualquier
contradicción con lo que sigue más abajo, manda esta sección.**

El proyecto entra en reconstrucción y reordenamiento **estricto, paso a paso, en
el orden en que se debió construir desde el principio**. No es una refactorización
ni una auditoría: es empezar de cero con la libertad de borrar, mover o reordenar
cualquier cosa que estorbe.

### Qué NO tiene validez

Nada de lo anterior es estado vigente. En concreto, y sin excepciones:

- **Ningún backtest.** Ninguna cifra de Brier, accuracy o ROI de ninguna corrida.
  El "baseline vigente" que este documento declara más abajo **no es vigente**.
- **Ningún motor de predicción.** Ni el pipeline de λ, ni sus nueve etapas, ni el
  simulador, ni la detección de valor, ni el sizing tal como están construidos.
- **Ninguna calibración.** Platt, Kalman, sesgo de equipo, descenso de gradiente:
  ninguno de sus estados guardados vale.
- **Ninguna conclusión sobre rentabilidad**, en ningún mercado.

Todo eso se conserva en el disco y en este archivo **como referencia histórica y
como catálogo de modos de falla**, no como estado. Se puede borrar en cualquier
momento sin pedir permiso.

### Qué sí sobrevive, y por qué

Sólo dos cosas, y ninguna es una afirmación sobre el modelo:

1. **Los HECHOS**: marcadores reales verificados contra el schedule oficial, y
   precios de mercado observados. Un hecho no deja de ser cierto porque el
   modelo que lo rodeaba fuera malo.
2. **El catálogo de modos de falla**: los bugs concretos, con su mecanismo. No
   dicen nada sobre si el sistema predice; dicen cómo esta clase de sistema se
   rompe. Repetirlos sería el único desperdicio real de lo hecho hasta acá.

### Qué se está construyendo

Un **software de predicciones deportivas con inteligencia artificial**. En una
segunda fase, un **proceso agéntico** que tome decisiones sobre esas
predicciones y opere el negocio.

Es un proyecto ambicioso y del que depende el futuro de su dueño. Eso no cambia
los criterios técnicos —al contrario, los endurece— pero sí cambia el estándar:
nada entra "por ahora", nada queda "para arreglar después".

### Cómo se trabaja

- **Por secciones, en orden.** Una sección se cierra sólo cuando **deja el camino
  listo para la siguiente**. Antes de cerrar hay que poder decir qué necesita la
  sección siguiente y verificar que ya está disponible.
- **Cada paso se evalúa minuciosamente en busca de fallas** antes de darse por
  cerrado. Un paso "terminado" sin haber buscado activamente cómo está mal es un
  paso no terminado.
- **Se lleva constancia escrita** de cada sección: qué se construyó, qué se
  midió, qué se decidió y qué queda abierto. Esa constancia es lo que permite
  retomar en otra sesión sin reconstruir el contexto de memoria.
- **Mejores prácticas, sin atajos.** Las reglas que importan las impone el
  mecanismo (esquema, tipos, tests), nunca la disciplina de quien llama.
- **Sin barreras de datos.** Cualquier dato necesario se consigue, sea gratuito
  o de pago. Si una fuente de pago es la correcta, se dice y se justifica en vez
  de conformarse con una peor porque es gratis.
- **Toda práctica necesaria se implementa.** Si hace falta infraestructura,
  versionado de datos, CI o un instrumento de medición, se construye; no se
  omite por costo de tiempo.

### Estado de la reconstrucción

| Sección | Estado |
|---|---|
| — | sin arrancar |

La primera tarea es **definir las secciones y su orden** antes de escribir
código. Nada se construye hasta que exista ese plan y esté acordado.

---

## What this project is

**FINAL BOSS QUANT G8+** — a quantitative sports betting prediction system. It analyzes MLB, NBA, and UFC games and identifies positive-EV betting opportunities using Monte Carlo simulation, Poisson modeling, Kelly criterion sizing, and odds API integration.

## 🔓 Congelamiento del motor LEVANTADO desde el 2026-07-27

**El estado vive en `docs/PROTOCOLO_CLV_V1.md` §Registro, campo `Estado del
congelamiento`, y `tests/test_engine_freeze.py` LO LEE de ahí.** Este archivo es
una copia informativa: ante cualquier duda manda el protocolo, no esta sección.

El dueño levantó el congelamiento **hasta que termine la auditoría paso-a-paso**
del pipeline (pasos 0..39, en curso desde el 2026-07-27), porque empezó a
encontrar defectos que sí mueven λ en vivo y que no se podían arreglar con el
motor congelado. Ninguna corrida de este período es muestra primaria — ya no lo
era desde que D0 quedó SUPERSEDED, y ahora además el motor se mueve.

Para re-armarlo hay que hacer **las dos cosas**: volver el campo del protocolo a
`VIGENTE` **y** re-apuntar el `engine_commit` al HEAD de ese momento.

> ⚠️ **Al re-armar, el `engine_commit` DEBE apuntar a un SHA de esta historia.**
> El Registro ancla hoy `b3325a52680e44ea75e1b1dcc26f9d9af6369393`. Ese commit
> existe en el almacén de objetos local, pero **no es alcanzable desde ninguna
> rama** —ni desde `HEAD` ni desde lo publicado en GitHub— porque el repo
> remoto nació como *clean upload* y reescribió los SHA del repositorio viejo.
> Un `git gc` puede podarlo, y un clon limpio nunca lo tuvo.
>
> El modo de fallo es silencioso, que es lo peligroso:
> `tests/test_engine_freeze.py` hace `git cat-file -e` antes de diffear y, si el
> commit no resuelve, hace **skip** en vez de fallar. Así que un freeze marcado
> `VIGENTE` contra un SHA inalcanzable **se ve armado y no verifica nada** —
> peor que un fallo ruidoso. Verificado el 2026-08-02: el objeto existe local,
> `git merge-base --is-ancestor` da falso contra `HEAD` y contra
> `origin/feature/point-in-time-rebuild`.

⚠️ Esta sección estuvo desactualizada del 2026-07-27 al 2026-07-28 (decía
"congelado" cuando ya no lo estaba) y alcanzó a hacer que una consulta externa
diera una advertencia operativa equivocada. Si cambia el estado del freeze, se
actualizan los DOS documentos.

<details><summary>Reglas que aplican cuando el congelamiento está VIGENTE</summary>

Mientras dure la ventana:
- **No correr `scripts/promote_calibration.py`** — una promoción cambia las probabilidades live, y el protocolo exige que cualquier cambio de motor a media ventana reinicie la muestra primaria.
- No tocar ningún engine de predicción (`tte_*`, `hfa_engine.py`, `park_weather_engine.py`, `pitcher_engine.py`, `bullpen_engine.py`, `defensive_efficiency_engine.py`, `learning_engine.py`, `montecarlo/simulator.py`, `core/value_detector.py`) ni `backtest_and_retrain.py` ni los builders PIT.
- Trabajo en paralelo permitido SOLO fuera del camino de predicción de ML: mercados derivados (F5, totals), higiene de backlog, tooling, track_record (siempre que no toque probabilidades).
- Verifica `docs/PROTOCOLO_CLV_V1.md`'s sección "Registro" antes de asumir que el congelamiento sigue vigente — ahí vive la fecha de D0 y cualquier cierre de ventana.

</details>

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

> ### ⬛ BASELINE VIGENTE — declarado UNA sola vez, acá
>
> **Brier 0.24624 / accuracy 54.84%**
> `--season 2024,2025 --use-full-pit`, 4.825 juegos.
> **Reporte canónico:** `audit_20260714/paso10/backtest_canonico_kalman0.json`
> (la ruta del JSON es la fuente de verdad, no la prosa — ver por qué más abajo).
> Fijado el 2026-07-28 por el paso 10 de la auditoría (Kalman de ofensa a 0.0).
>
> Cualquier cifra distinta que aparezca más abajo en este documento es
> **historia**, no el estado actual. La cadena completa se conserva a propósito
> —cada eslabón explica por qué cayó el anterior— pero al citar el baseline en
> cualquier sitio nuevo, se cita éste.
>
> Cadena histórica, de la más vieja a la más nueva:
> `0.24242 → 0.24525 → 0.24482 → 0.24483 → 0.24650 → 0.24675 → **0.24624**`

**Nota de coherencia (2026-08-02)**: esta sección declaraba como vigente el
baseline del 2026-07-09 (0.24525 / 55.30%), que llevaba superseded desde el
2026-07-11 y contradecía a la sección del 2026-07-28 más abajo. Se corrige acá
sin borrar nada de lo que sigue: el relato de los tres bugs de medición del
párrafo siguiente conserva su valor aunque su número ya no sea el vigente.

### Histórico — 2026-07-09 (superseded)

**Brier 0.24525 / accuracy 55.30%** (4.830 juegos, 453 tests). Fue el primer
número simultáneamente reproducible (seed determinístico), libre del leak de
team-bias, y correctamente calibrado con Platt — reemplazó a 0.24242, 0.24274,
0.24272 y 0.24592.

Tres bugs de medición reales, independientes, se encontraron y corrigieron en sucesión, cada uno moviendo el número 1-2pp de accuracy:
1. **Look-ahead leak en `LearningEngine.compute_team_bias`/`compute_multidim_bias`** (`calibration/learning_engine.py`) — sin corte de fecha (`WHERE season = ?` sin `game_date <`), el bias que se multiplica a λ veía resultados reales de *toda la temporada*, incluyendo juegos posteriores al que se predecía. Corregido con un parámetro `before_date` walk-forward. El modo live nunca tuvo este leak (no existen resultados futuros en producción). Test de regresión: `tests/test_anti_leakage_opening_day_2024.py`.
2. **No-determinismo del Monte Carlo** — corregido con seed determinístico por `game_pk`.
3. **Corrupción de la calibración Platt** — un lanzamiento duplicado accidental de un proceso de backtest (matado a los segundos) alcanzó a resetear `ml_state.platt_params` a identidad antes de que el proceso real arrancara, dejando el backtest entero corriendo sin calibración Platt aplicada. No es un bug del pipeline — fue un artefacto de lanzamiento de esa sesión — pero reveló que el mecanismo de reset+warm-start no tiene protección alguna contra un proceso concurrente/interrumpido corrompiendo el cache compartido en silencio (misma clase de falla que el leak de arriba). Corregido relanzando limpio.

**Importante**: "confiable" acá significa "los bugs de medición conocidos están corregidos", no "el sistema es rentable" — el ROI de esta corrida es delgado e inconsistente (edge≥8% apenas +0.20%, edge≥10% en -2.97%), muy lejos de la conclusión de rentabilidad que sostenían los números anteriores (todos contaminados). Sigue siendo una pregunta abierta.

Nota adicional: la temporada 2024 de este backtest corre con calibración Platt en identidad por diseño (no existen datos de la temporada 2023 en esta DB para hacer warm-start) — limitación conocida y aceptada, no un bug nuevo.

**Consecuencia real, no solo cosmética**: al remover el leak, accuracy cayó de 56.27% a 54.35% y el ROI se volvió negativo para edge<10% (antes positivo desde edge≥8%). La conclusión "el sistema es rentable" que sostenían los números anteriores ya no está validada — sigue siendo una pregunta abierta con el número honesto actual.

Auditoría completa (solo lectura, componente por componente) realizada el 2026-07-06 — reporte completo en `docs/AUDITORIA_MLB_2026-07.md` (su baseline citado, 0.24242, está superseded por lo de arriba, ver nota al inicio del documento). Resumen aún válido: el pipeline de lambda (los 7 motores) está en su mejor estado histórico tras la revisión exhaustiva de esa sesión. `CONTRACTS.md` ya fue reescrito (2026-07-06, ya no es un riesgo pendiente). Ver `docs/FBQ_MASTER_BLUEPRINT.md` (v1.8) para la hoja de ruta completa.

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

**UI (VAL-6)**: el duelo de abridores (`frontend/src/components/matchup/StartingPitchers.tsx`)
ahora grafica **aportes = peso × (crudo − 1)**, que suman al delta total, en vez de los cinco
multiplicadores crudos — la combinación del pitcher engine es lineal, nunca un producto. Los pesos
los expone el API desde `config.py::PITCHER_ENGINE_WEIGHTS` (`engine_weights.pitcher` en
`api/server.py`), no se hardcodean en el frontend; el crudo y el peso de cada factor están en el
detalle de cada fila, la suma se escribe explícita bajo las barras, y el recorte a [0.65, 1.45] se
marca cuando aplica. Tests: `frontend/src/test/pitcherBreakdown.test.ts` (reproduce los casos
Sugano/Drohan calculados a mano en VAL-6).

Memoria de la sesión: `project_derived_eval_20260726.md`.

### Nota operativa 2026-07-26 — D0 fijado, y el crash que lo tenía bloqueado

**D0 del protocolo de CLV = 2026-07-26 — y marcado SUPERSEDED ese mismo día** (ver el Registro de
`docs/PROTOCOLO_CLV_V1.md`, y la nota del 2026-07-26 más abajo). **No hay ventana primaria abierta
hoy**: los picks que se sigan publicando son shakedown hasta que `PROTOCOLO_CLV_V2` fije un D0
nuevo. **El congelamiento del motor SÍ sigue vigente** — no se levantó nada. V1 quedó cerrado como **(a)** con evidencia fresca de la
cadena de cron (25/25 juegos analizados el mismo día tienen ambos pins de Pinnacle en
`game_outcomes`; la cobertura baja a 1-2 días vista es horizonte de mercado, no falla de captura)
y el keepalive de Windows está instalado y verificado (5 tareas, `Last Result: 0`).

**Qué certifica el congelamiento**: NO el stamp `picks.engine_commit` — ese es el HEAD del repo y
avanza con cualquier commit de UI o docs. Lo que certifica es que el diff de los paths del motor
contra el `engine_commit` registrado esté **vacío**; ahora es un test
(`tests/test_engine_freeze.py`), que también falla con cambios sin commitear en esos paths porque
el cron corre desde el working tree.

**El bloqueador que lo impedía** (`f4a4bf4`): desde que b3325a5 agregó `kelly_floor_applied`
(VAL-4.4), ese flag llega como `np.bool_` al `json.dumps` de `pipeline_json` en
`track_record/publisher.py`. Bajo numpy 2.x la clase se llama literalmente `bool`, así que el
traceback decía "Object of type bool is not JSON serializable" — imposible de creer y fácil de
pasar por alto. Las corridas de cron del 2026-07-24 (07:00 y 13:00) y del 2026-07-25 (07:00)
corrieron el pipeline entero y murieron al guardar: **cero picks publicados esos días**, y por eso
no existía ningún pick post-b3325a5 con el cual fijar D0. Lección transferible: cuando un cambio
del motor agrega un campo nuevo a `bet`, ese campo viaja hasta `pipeline_json` — cualquier tipo de
numpy ahí rompe la publicación **después** de que el pipeline ya hizo todo el trabajo.

### Actualización 2026-07-26 (tarde) — D0 SUPERSEDED en día 1 + fuga de captura de derivados

**No hay ventana de CLV corriendo.** D0 se fijó y se marcó SUPERSEDED el mismo día, con dos
razones medidas (detalle completo en el Registro de `docs/PROTOCOLO_CLV_V1.md`):

1. **La muestra primaria ML-only no da el número.** La métrica primaria se define sobre el devig
   de Pinnacle de ambos lados, que solo existe para h2h: la primaria ES el subconjunto moneyline.
   En D0 fueron 3 de 25 picks (12%); sobre los 141 de cuarentena, 25.5%. Proyección a 6 semanas:
   ~257 picks ML usables al ritmo medio, ~107 al mediano, contra el objetivo pre-registrado de
   n≥300. Atrición de ML medida: 24.2%, sobre el 15% que el propio protocolo llama "problema del
   instrumento".
2. **Los derivados no tenían cierre capturado en absoluto.** `capture_closing_lines.py` solo tenía
   ramas ML: RUNLINE 53 picks → 0 con precio de cierre, TOTAL 52 → 0, y el cierre es irrecuperable
   una vez que empieza el juego.

**Arreglado el mismo día (`b629f55`), fuera del camino congelado**: `odds_fetcher.py` ya pedía
`totals`/`spreads` y extraía solo el PUNTO de Pinnacle, nunca su precio — ahora guarda el par de
precios de Pinnacle para total y runline (aditivo, cero quota extra, ninguna clave existente
cambia). `capture_closing_lines.py` tiene ramas OVER/UNDER/RL_HOME/RL_AWAY, y `picks` tiene
`closing_pin_side`/`closing_pin_opposite`/`closing_point`/`closing_point_moved` — el par del
mercado del PROPIO pick (lo que un devig necesita) y la marca de movimiento de línea, porque un
total que cerró en 9.0 no gradúa un pick tomado a 8.5. `clv_pct` sigue ML-only a propósito: qué
es CLV para un derivado lo define V2; lo que cambió es que ya no falta el dato para computarlo
después. Nada histórico se recomputó — los derivados previos son pérdida documentada.

**Estado tras la primera barrida real con el arreglo** (2026-07-26, 24/24 picks pendientes):
RUNLINE 12 y TOTAL 9 con precio de cierre y par de Pinnacle devig-able, donde antes había 0 y 0;
2 de esos 9 totales ya cerraron en un punto distinto al que se tomaron. Instrumento repetible:
`scripts/closing_capture_coverage.py` (solo lectura, `--live` para proyectar la barrida actual).

**Pendiente, del dueño con Fable**: escribir `PROTOCOLO_CLV_V2` y fijar D0 nuevo, después de ~2
días de captura de derivados corriendo. Hasta entonces no se cita ninguna ventana como abierta.

### Actualización 2026-07-28 — auditoría paso a paso, pasos 0..10

**Baseline vigente: Brier 0.24624 / accuracy 54.84%** (mismo comando
`--season 2024,2025 --use-full-pit`, 4.825 juegos) — supersede el 0.24675/55.05% del
rebaseline VAL. **Reporte canónico**: `audit_20260714/paso10/backtest_canonico_kalman0.json`.

⚠️ **Framing obligatorio**: este número NO es "el modelo mejoró 0.0005 de Brier". La ventaja
del modelo sobre el azar pasó de 1.300% a 1.510% — **+16.2% relativo**, que es la forma honesta
de dimensionarlo, porque toda la señal vive en esos ~3 milésimos sobre 0.25. Y viene con una
contrapartida: **la accuracy BAJÓ 0.21pp** (55.05% → 54.84%), consistente en ambas temporadas.
Se eligió el Brier porque acá no se apuesta a quién gana sino a que `p × cuota > 1`.

⚠️ **Ese criterio quedó INCOMPLETO y el paso 11 lo corrigió**: ignoraba el ROI por
umbral de edge, que es el proxy más directo de para qué existe el sistema. Al medir el
paso 11 (neutralizar el sesgo de equipo) el Brier mejoraba mucho —ventaja sobre el azar
de 1.51% a 2.00%— mientras el ROI se derrumbaba en los cinco umbrales (edge≥10%: −6.69%
a −19.89%) y el conteo de apuestas caía a la mitad. El Brier mide la calidad PROMEDIO;
el ROI mide la COLA, que es donde se apuesta. El paso 10 no queda invalidado (su ROI
alternaba de signo, o sea ruido), pero **de acá en adelante el ROI por umbral es un gate
obligatorio para cualquier cambio de λ**, no una métrica secundaria. Ver
`audit_20260714/paso11/reporte.md`.

**Salvedad que no se borra**: la ganancia de Brier está concentrada en 2024 (−0.00095) con 2025
casi plano (−0.00008). Una temporada aporta ~92%. Si un tercer año no la reproduce, el cambio
del paso 10 es el primer candidato a revisarse.

**El congelamiento del motor está LEVANTADO** desde el 2026-07-27 mientras dure esta auditoría
(ver la sección al inicio de este archivo y el Registro del protocolo). Ninguna corrida de este
período es muestra primaria.

Cambios de los pasos 0..10, en orden. Los pasos 0-3 son del camino de publicación (no tocan λ);
del 4 en adelante sí:

- **0-3 (publicación)**: disparador con chequeo de salud propio; lista de estados publicables
  en vez de deny-list; garantía pre-juego que falla cerrada y se mide contra el reloj real;
  identidad juego↔evento de odds que se abstiene ante dos candidatos igual de plausibles
  (umbral derivado de la separación real medida entre juegos de doubleheader: 5 min los
  tradicionales, 275-405 min los partidos).
- **4 (datos base)**: el día del juego sale de `official_date` y no del timestamp UTC truncado
  — 30% de los juegos difieren, y `days_rest` salía +1. Ventana de viaje anclada al día
  oficial. Doubleheader conservado. Fallo de enriquecimiento marcado en vez de invisible.
- **5 (abridores)**: `ip_mlb_equivalent` — cuánto se le cree a un abridor depende de la
  PROCEDENCIA del dato (MLB actual 1.0, temporada anterior 0.50, AAA 0.25, AA 0.15), no sólo
  del tamaño. Un split de lado con 2 innings ya no desplaza a la línea consolidada de MLB.
- **6 (lineup)**: sin alineación confirmada (83% de las corridas) se usa la mezcla medida
  50/50 de última alineación real + roster, en vez de una constante que además tapaba el dato
  real. Ambidiestros resueltos por la mano del abridor rival.
- **7 (platoon)**: regresión por tamaño de muestra con prior poblacional por mano (RHP 1.144,
  LHP 0.852, medidos) — antes los abridores con 8-13 innings de split se iban al tope del
  recorte. Notación de innings de béisbol convertida en los 9 sitios (era cruda en 8).
- **8 (forma reciente)**: NEUTRALIZADO. Pesaba 0.256 y era anti-predictivo — sus tres señales
  salían de la misma lista de 5 arranques y captaban regresión a la media, leída como
  persistencia. Ninguna sobrevive a control por nivel con errores agrupados.
- **9 (TTE ofensiva)**: sin cambios de comportamiento. La λ ofensiva está bien; su
  documentación describía dos estados ya superados.
- **11 (sesgo de equipo)**: SIN CAMBIOS, y es el hallazgo. Neutralizarlo mejora el
  Brier y destruye el ROI (ver la advertencia de arriba). El sesgo se queda; quien
  quiera tocarlo tiene que pasar el gate de ROI en los umbrales altos.
- **10 (Kalman)**: `_KALMAN_BLEND` 0.35 → **0.0**. El Kalman observa carreras REALES y tiraba
  hacia ellas una λ MERECIDA (xwOBA) que por construcción filtra esa suerte; además inyectaba
  parque en una λ neutra de parque que vuelve a recibir el factor en el PASO 5.

**Advertencia metodológica que costó cara y vale para todo trabajo futuro**: los snapshots PIT
diarios y los arranques de un mismo pitcher NO son observaciones independientes. Usar errores
estándar iid sobre ellos infla los t entre 5x y 21x. Todo análisis sobre esas fuentes necesita
errores agrupados por entidad (pitcher, equipo). Tres falsos positivos de esta auditoría —QS%
con t=+4.79, "suerte" con t=−21, y la ventaja del compuesto TTE— desaparecieron al agrupar.

Memoria de la sesión: `project_auditoria_pasos_2026_07.md`.

### Actualización 2026-08-02 — trabajo posterior al paso 11, ya en la rama

**Baseline sin cambios** (Brier 0.24624 / accuracy 54.84%, reporte canónico
`audit_20260714/paso10/backtest_canonico_kalman0.json`). Nada de lo que sigue toca un motor de
predicción ni ninguna λ: es precio de mercado, medición y telemetría. Se documenta acá porque
estaba en la rama sin quedar registrado en ningún sitio.

**Runline preciado con el punto FIRMADO, de punta a punta** (`c6bb482`, `ffb2c93`). `analyze_runline`
usaba la MAGNITUD de la línea y calculaba siempre `P(diff > 1.5)` —"el local gana por 2 o más"—
sin mirar quién era el favorito. Cuando el local era el NO-favorito, el mercado traía los precios
de HOME +1.5 y AWAY −1.5 pero se apareaban con las probabilidades de HOME −1.5 y AWAY +1.5: la
probabilidad del evento FÁCIL con el precio del evento DIFÍCIL. Es PURP-1, arreglada el 07-31 con
`runline_home_point`. El 08-02 se cerró el segundo agujero: el camino del selector de la UI perdía
el punto en DOS sitios independientes —`build_game_selector()` no lo copiaba a `GameData`, y una
vez copiado `_safe_float()` lo anulaba por descartar todo `<= 0`, justo cuando el local favorito
cotiza −1.5, el caso mayoritario—. Nuevo `_safe_signed_float()` para magnitudes con signo. El
camino de cron nunca estuvo afectado.

**CLV medido contra el cierre JUSTO, no contra el crudo** (`2a40a04`). `clv_pct` era
`odds_tomadas / precio_crudo − 1`, y el precio crudo lleva el margen de la casa adentro: regalaba
el vig entero como si fuera habilidad. Sobre los picks reales, media +1.514% y 61.4% positivos
pasaron a **−0.497% y 40.9%** (t=−0.85) al desvigorizar. La diferencia es exactamente el overround
de cierre de Pinnacle. El signo se invierte con los cuatro métodos habituales, con magnitudes casi
idénticas: el método no importa, medir contra crudo sí. **El único indicador de habilidad positivo
que tenía el proyecto era un artefacto del vig.**

**CLV extendido a derivados cuando el punto no se movió** (`83299f3`). La objeción del 07-26 era
correcta pero demasiado amplia: un total tomado a 8.5 que cierra en 9.0 no se puede comparar por
precio, pero eso descarta el caso MOVIDO, no el mercado entero. Con `closing_point_moved == 0` un
runline o un total se compara igual que un moneyline. Duplicó la muestra del ledger: 105 → 253
picks con CLV. `None` también excluye — preferible sin CLV que con uno que compara dos líneas
distintas sin saberlo. Motivación medida: elegir el filtro del producto por ROI pediría ~14.800
picks (740 días al ritmo actual); por CLV alcanzan ~117 (6 días).

**El track record guarda lo que el modelo VIO** (`127bac6`). `pipeline_json` guardaba `lambdas`,
`mc_probs` y `bet` — los RESULTADOS, nunca las ENTRADAS. Se descubrió al intentar medir cuántas
veces el abridor analizado no es el que termina lanzando: con 26.6 horas medianas de anticipación
tiene que pasar seguido, y resultó IMPOSIBLE de medir porque la API sólo devuelve el probable
ACTUAL. Ahora se guarda identidad y PROCEDENCIA de ambos abridores, `lineup_confirmed`,
`home_lhb_source`, descanso y las banderas de calidad de dato.

**Lesiones integradas al módulo de béisbol** (`4f6b4c6`). `app.py` tenía `home_injuries: []`
hardcodeado en vacío y nada en el módulo miraba lesiones. Importa porque la ofensa sale del
Statcast ACUMULADO, que incluye entera la producción de quien hoy está en lista de lesionados: los
Yankees del 08-01 tenían 19.2% de sus turnos ofensivos en jugadores que no iban a jugar. Sobre la
cartelera real la dispersión va de 1.6% a 22.1% entre equipos. **λ todavía NO lo descuenta** — se
registra como `injured_pa_share` para medir cuánto importa antes de decidir cómo usarlo.

**Telemetría de clima honesta** (`092cefd`). La instantánea de entradas leía
`game_data['weather']['source']`, una clave que NO existe: el fetcher devuelve catorce campos y
ninguno se llama así. Resultado, 0 de 52 picks registraban procedencia mientras el motor SÍ recibía
el clima y SÍ movía λ (verificado en vivo, `weather_mult` 1.018 sobre Truist Park). Ahora se estampa
desde `park_meta`, la fuente que el propio motor calcula. Era un fallo de telemetría, no de captura
— y un dato bueno con la etiqueta rota es peor que un dato ausente, porque nadie audita lo que cree
que nunca llegó.

**Guard de implausibilidad del edge** (`d885267`). Contra un precio YA desvigorizado, un edge enorme
no es ventaja: es la firma de que modelo y mercado hablan de eventos distintos. Vive en
`analyze_market_generic`, el embudo único de los tres mercados, y bloquea forzando el tier a
NEGATIVE. Umbrales calibrados contra la muestra real de PURP-1, no elegidos a ojo: 20pp bloquea el
62% de aquellos falsos tocando 1 de 71 picks posteriores sanos; 15pp subiría a 77% pero clipearía 5
sanos. Zona de aviso 12-20pp que marca sin bloquear. **Es un detector de humo, no la solución**: el
38% de aquellos falsos tenía inflación menor a 20pp. La defensa real es que probabilidad y precio
salgan de la misma identidad de mercado.

**Instrumentos de estrategia** (`3120226`). Nueve scripts repetibles de solo lectura en
`audit_20260714/estrategia/` (a1..c1) que atacan la pregunta abierta del proyecto —¿hay alfa real o
sólo Brier?— más las salidas de sus corridas. No tocan el motor ni ninguna λ.

**Dimensión del daño de PURP-1, medida el 2026-08-02**: re-calificados los 134 picks de runline
previos al arreglo, re-simulando las λ finales guardadas en cada `pipeline_json`, **55 (41%) se
publicaron con EV positivo cuando el real era ≤ 0**. EV medio publicado +34.47% contra −2.27% real;
probabilidad media 0.6419 contra 0.4871. Y vía Kelly el EV inflado agrandaba las apuestas: 508 de
las 822 unidades arriesgadas —el 61.8% del capital— estaban en picks con edge fabricado, con stake
medio 13.04 u contra 3.82 u del resto. **El track record en vivo anterior al 07-31 no mide el
modelo**: mide el modelo más un error de emparejamiento que sobredimensionaba las apuestas justo
donde más se equivocaba.

### Actualización 2026-08-04 — recorrido desde cero (pasos 0..2 de 12)

**Baseline sin cambios** (Brier 0.24624 / accuracy 54.84%, reporte canónico
`audit_20260714/paso10/backtest_canonico_kalman0.json`). Nada de lo que sigue toca un motor de
predicción ni ninguna λ: es identidad, precio de mercado y verdad de terreno.

**Qué es este recorrido, y en qué se diferencia de la auditoría paso-a-paso.** La auditoría de
`audit_20260714/` recorre el PIPELINE (pasos 0..39: datos base, abridores, lineup, platoon...).
Esto es otra cosa y corre en paralelo: recorrer el proyecto **en el orden en que se construiría
desde cero**, capa por capa, preguntando en cada una "¿qué exigiría este paso si lo escribiera
hoy?" ANTES de mirar el código —si no, el código define el criterio y siempre aprueba— y después
midiendo la brecha. Doce pasos:

| # | Paso | # | Paso |
|---|---|---|---|
| 0 | Identidad y tiempo | 6 | La primera feature |
| 1 | Precios | 7 | Distribución / simulador |
| 2 | Resultados | 8 | Detección de valor |
| 3 | Evaluador | 9 | Sizing |
| 4 | Modelo v0 = el mercado | 10 | Publicación |
| 5 | Datos de entrada PIT | 11 | Reconciliación y CLV |
| | | 12 | Producto |

**El hallazgo del mapa, antes de tocar nada**: el proyecto está construido **invertido**. Los pasos
5, 7, 8, 9, 10 y 12 —la maquinaria— están maduros. Los pasos 2, 3, 4 y 6 —la columna de medición—
son los débiles. Eso explica estructuralmente los siete baselines en tres meses: todos cayeron por
defectos de MEDICIÓN, no de modelado. Y explica que el hallazgo más caro del proyecto (el modelo no
le gana al precio: 0.24620 contra 0.24052 de Pinnacle, coeficiente NEGATIVO en la regresión
conjunta, mezcla óptima w=0.00) apareciera recién el 2026-07-30 en un script suelto de 130 líneas
—`audit_20260714/estrategia/a1_descomposicion.py`, que ES el paso 3— y llegara décimo.

#### Paso 0 — Identidad y tiempo: 4 de 6

Los DATOS están bien (`official_date` 5.721/5.721, `game_pk` estable, doubleheaders distinguibles).
Lo que falta es que las reglas las imponga el MECANISMO y no la disciplina del llamador.

- **Borrada `_prediction_cutoff_for_row()`** de `backtest_and_retrain.py`. Devolvía el fin del día
  DEL JUEGO (`{official_day}T23:59:59Z`) mientras sus cuatro hermanas devuelven día_del_juego − 1s.
  Cero llamadores de producción; sólo 3 tests que la ejercitaban contra sí misma, borrados con ella.
  Era un arma cargada: combinada con el `<=` inclusivo de `PITCache.get_latest()` reintroduce el
  leak de la Fase 2B apenas alguien la conecte.
- **Residual anotado**: `PITCache.get_latest()` sigue comparando con `<=`. Hoy es seguro sólo
  porque los cuatro cutoffs restan un segundo a mano.
- **Residual anotado**: dos convenciones de etiqueta conviven en `pit_metric_cache` —
  `savant.batter.rolling` y `savant.team_offense.rolling` guardan `DT00:00:00`, defensa/bullpen y
  los tres `prior_baseline` guardan `DT23:59:59`— y el builder de ofensa etiqueta `T00:00:00Z` un
  snapshot cuyo `source_window_end_date` es el FIN de ese día. Hoy ambas resuelven al mismo
  snapshot de D−1: es correcto **por aritmética, no por convención compartida**. Mordería con
  cualquier corte a media jornada; está dormido porque el camino en vivo no usa el cache PIT
  (`modules/baseball_module/core/` no tiene una sola referencia a él).

#### Paso 1 — Precios: `market/`, almacén append-only

**Nuevo módulo `market/`** (`data/market.db`, independiente de `picks` y `game_outcomes`). Cinco
reglas, cada una nacida de un defecto que este proyecto ya pagó, y **ninguna impuesta por
convención**:

| Regla | Quién la impone |
|---|---|
| Append-only | Dos triggers de SQLite abortan `UPDATE` y `DELETE` |
| Los dos lados, siempre | `pair_before()` devuelve `None` con medio par |
| Punto FIRMADO pegado a su lado | Una fila por lado, con su propio `point` |
| `event_id` explícito | `NOT NULL`; sin id del proveedor no se guarda |
| El libro es parte del dato | `book NOT NULL` |

El motivo: `track_record/capture_closing_lines.py` pide el board **doce veces por día** y hace
`UPDATE picks SET closing_* = ...` en cada barrida — se pagan doce capturas y se guarda una. El
docstring lo describía como feature ("last pre-start capture wins"), que es exactamente por qué
nadie lo cuestionó. `historical_odds` tiene el mismo problema por esquema (`game_pk UNIQUE`, una
fila por juego, siempre a las 17:00Z: no es el cierre, es una foto de la mañana).

Cron a los `:51` de 8 a 19h, **un minuto después** de la barrida de cierre: esa barrida deja el
board en la caché de 10 min de `odds_fetcher`, así que es cache hit y **cuesta CERO llamadas
nuevas**. `market.link_events` una vez al día (el enlace es recuperable, el precio no). Primera
barrida real: 156 eventos, 34 libros, 12.535 cotizaciones, 0 saltadas; segunda barrida 0 filas
nuevas y 12.535 sin movimiento, con la barrida igual registrada en `sweep` para que "el mercado no
se movió" se distinga de "no miramos". 25/25 juegos de MLB enlazados, 0 ambiguos, 6/25 (24%) con
`official_date != date(commence_time)` — consistente con el 22,2% de la Fase 2B.

Tres arreglos sobre lo que ya existía:

1. **Cobertura 2026: 0% → 90.4%** (779/862). Antes la temporada en curso no tenía UN SOLO precio
   histórico: era immedible contra el mercado. Costo 5.160 créditos, de los cuales **2.580 se
   desperdiciaron**: `--dry-run` en `fetch_historical_odds.py` significa "fetch and parse, do not
   write" — SÍ llama a la API.
2. **Los "team name mismatch" eran todos Athletics.** `_MLB_TO_ODDS_NAME` traducía en un solo
   sentido (`"athletics" → "oakland athletics"`) y sólo del lado de MLB; cuando la Odds API pasó a
   decir "Athletics", la traducción llevaba el nombre LEJOS del de la API en vez de acercarlo.
   Ahora `_TEAM_ALIASES` canoniza AMBOS lados. Costo medido del bug: 162 juegos de 2025 y 114 de
   2026 sin precio, atribuidos a "mismatch" sin que nadie mirara cuál.
3. **La línea justa de los derivados salía de un par sintético.** Runline y total desvigorizaban
   `{mejor over, mejor under}` — dos máximos a través de 32 casas, potencialmente de casas
   distintas, un par que no cotiza nadie — mientras el moneyline sí usaba el par propio de
   Pinnacle. Y `true_implied` no es cosmético: alimenta `edge`, que alimenta `composite_score`,
   `classify_value_tier` y el guard de implausibilidad. Nuevo `_fair_two_way()`, que además
   **descarta el par de Pinnacle si cotiza otro punto**. Efecto medido sobre el board real,
   controlando por punto firmado: media −0.06pp en totales y −0.17pp en runline, máximo ~1pp — es
   una asimetría real y gratuita de corregir, **no una fuente grande de error**, y no debe citarse
   como si lo fuera.

   **Lo que enseñó el arreglo**: `tests/test_odds_seam_contract.py` falló y tenía razón. Arreglarlo
   sólo en el camino de cron habría dejado la UI preciando contra el par sintético — la forma
   EXACTA del segundo agujero de PURP-1, un arreglo que alcanza una sola de las dos rutas hacia
   `GameOdds`. Fue en las cinco capas: `_normalize_event` (ya calculaba los valores y no los
   emitía), `build_game_selector` (con `_safe_signed_float` para los puntos), `MARKET_ODDS_FIELD_MAP`,
   `GameOdds` y los dos analizadores.

#### Paso 2 — Resultados: 2024/2025 limpios, tres defectos en vivo

Barrido completo de las 5.721 filas de `game_outcomes` contra el schedule real de MLB (API gratis).

**2024 y 2025 salieron 100% limpias** — 4.859 juegos, 0 marcadores discrepantes, 0 fechas
discrepantes. El dato con el que se entrena y se hace backtest es sólido; los defectos están
confinados al camino en vivo.

Tres defectos en 2026, con una causa común que los vuelve caros: `update_outcome()` es idempotente
a propósito (`WHERE actual_home_runs IS NULL`), así que **la guarda que evita re-aprender el mismo
juego vuelve PERMANENTE cualquier marcador equivocado que entre una vez**.

1. **Un pospuesto guardado como empate.** `game_pk=824490` (Guardians @ Reds): DB `5-5` con
   `home_won=0`, final real `6-5`. El guard de `c527594` no lo atrapó por dos agujeros simultáneos:
   `abstractGameState` vale **"Final" también para un juego POSPUESTO** (el que distingue es
   `detailedState`), y `/schedule?gamePk=` devuelve **DOS bloques de fecha con el mismo gamePk**
   cuando un juego se pospone y se rejuega — leer `dates[0]` agarraba el cascarón pospuesto.
2. **Un marcador congelado en pleno juego.** `game_pk=822958`: DB `3-1`, final real `5-1`, 9
   innings, sin postergación. Predata el guard.
3. **`home_won = 1 if home > away else 0`** convertía un empate en derrota del local, en silencio.

Corregidos los tres, más una carrera que quedaba: el estado Y el marcador salen ahora de la MISMA
respuesta (antes el marcador venía de una llamada previa a `/linescore`; si el juego terminaba entre
las dos, se escribía el parcial). De paso, una llamada menos por juego. Un empate se rechaza y se
loguea, dejando la fila pendiente. Tests: `tests/test_outcome_finality.py` (4), uno reproduciendo la
forma exacta de la respuesta de 824490.

**Datos reparados** (backup previo, valores verificados contra la API antes de escribir): 5 filas
(2 marcadores + 4 fechas oficiales, con solape) y **34 juegos ya jugados sin marcador desde mayo**,
que quedaban fuera de la ventana de 7 días de `fetch_pending_outcomes` y nunca se iban a recoger.
Post-reparación: 2026 con 0 discrepancias, 832/862 con resultado (los 30 restantes son de hoy y
mañana).

**Salvedad que no se borra**: los dos marcadores equivocados ya habían alimentado Kalman y descenso
de gradiente cuando se escribieron. Se corrigió la verdad de terreno, **no se rebobinó el
aprendizaje** — fabricar una historia de aprendizaje que no ocurrió sería peor que documentarlo. Los
34 rellenados tampoco dispararon aprendizaje, por la misma razón: aprenderlos hoy en lote no es lo
mismo que haberlos aprendido en orden.

**Veredicto del paso**: requisitos 1 y 4 (cobertura, finalidad) cerrados. Los requisitos 2 y 3
quedan abiertos y son cambios de esquema: `game_outcomes` **mezcla el hecho con la opinión** (el
marcador real vive junto a λ, `p_home` y las columnas `backtest_*`, que es la misma mezcla que
permitió CHRON-001), y hay **dos reconciliadores independientes** — la ironía útil es que esa
duplicación fue lo que DELATÓ el error, porque `picks` tenía el `6-5` correcto mientras
`game_outcomes` tenía el `5-5`.

#### Residuales de estos tres pasos, por decisión y no por olvido

| # | Residual | Por qué se dejó |
|---|---|---|
| 0 | `PITCache.get_latest()` con `<=` inclusivo | La seguridad vive en 4 llamadores, no en el mecanismo |
| 0 | Dos convenciones de etiqueta en `pit_metric_cache` | Hoy inocuo por aritmética, no por convención |
| 1 | 276 juegos (Athletics) sin precio | Ya se emparejarían bien; recuperarlos cuesta ~2.580 (2026) y ~7.380 (2025) créditos. Quedaban 4.486 |
| 2 | `game_outcomes` mezcla hechos con predicciones | Cambio de esquema en la tabla que usa el backtest |

Ninguno bloquea el paso 3, que es el evaluador — donde el mercado deja de ser un dato y pasa a ser
la barra contra la que compite todo lo demás.

#### Paso 3 — El evaluador (`evaluator/`, 2026-08-04)

**Nuevo módulo `evaluator/`**, solo lectura. Regla de diseño única: **el candidato entra por
parámetro**. No importa ningún engine, no puede favorecer al modelo de la casa, y el mercado
desvigorizado es siempre el nulo. Reemplaza conceptualmente a
`audit_20260714/estrategia/a1_descomposicion.py`, que era una FOTO (clavada a
`backtest_run_at='2026-07-30'` y a dos temporadas, un solo candidato, y produciendo un `_data.npz`
del que dependen otros cinco scripts).

```
python3 -m evaluator --seasons 2024 2025 --candidato backtest
python3 -m evaluator --seasons 2026 --candidato live
python3 -m evaluator --seasons 2024 2025 --candidato mercado   # autoprueba
```

**Autoprueba**: puntuar el propio mercado reproduce la barra con brecha `+0.00000`. **Validación
cruzada**: sobre 2024/2025 reproduce a1 al último decimal (0.24620 vs 0.24052, `b_cand=−0.1294`,
IC95 `[−0.3774, +0.1049]`, P(b>0)=14.2%).

**Lo que agrega — el gate de ROI por umbral**, sobre la corrida canónica:

```
 2%: n=3494  −0.42%   |   4%: n=2518  +0.21%   |   6%: n=1677  −2.87%
 8%: n=1053  −3.27%   |  10%: n= 617  −4.04%
```

El ROI empeora casi monótonamente al SUBIR el umbral. Si hubiera edge real, más edge declarado
debería dar más ROI; que ocurra lo contrario dice que la señal de edge es anti-predictiva — los
picks donde el modelo está más seguro son sus peores picks.

**Primera medición de 2026 contra el mercado, y lo que destapó.** El primer resultado fue el modelo
GANÁNDOLE al mercado (Brier 0.24581 vs 0.24789, b=+0.6061, ROI +10.72%). Era falso, y la señal de
alarma fue el NULO, no el candidato: un mercado con Brier 0.24789 contra tasa base 0.24900 es
Pinnacle casi sin habilidad, lo cual no ocurre. Al tirar del hilo: **578 de 832 filas de 2026 (69%)
tienen su `p_home` escrito DESPUÉS del juego, y 563 son `source='backtest'`** — el daño de
CHRON-001, la corrida del 2026-06-28, con un motor anterior al arreglo del leak V4. La columna
"live" de 2026 es 68% predicción contaminada.

Sobre las 209 predicciones genuinamente pre-juego (168 con precio): Brier 0.24775 vs 0.25201 del
mercado, `b=+0.7539` pero **IC95 `[−0.6469, +1.9049]`, P(b>0)=86.5% — el intervalo cruza el cero**.
**NO es medible todavía**, y los ROI de +8% a +30% se calculan sobre 25-132 apuestas. Citarlos como
evidencia sería exactamente el error que este paso existe para impedir.

**Tests**: `tests/test_evaluator.py` (10), incluido un CONTROL POSITIVO — un candidato que hace
trampa mirando el resultado TIENE que detectarse, porque un instrumento que no detecta una señal
plantada tampoco detectaría una real.

**Residuales del paso 3**:
- El veredicto "aporta sobre el precio" no distingue información nueva de una transformación
  determinística del precio: con colinealidad perfecta el coeficiente se reparte y el IC excluye el
  cero igual (se ve en la autoprueba, `+0.4999/+0.5000`). Importa al evaluar un candidato
  construido a partir del propio precio.
- **Sólo puntúa moneyline.** Runline y total no tienen líneas históricas en esta DB, así que su ROI
  sigue sin medirse — la única herramienta para esos mercados sigue siendo
  `audit_20260714/val_audit/derived_eval/`, y mide CALIBRACIÓN, no rentabilidad.
- Los cinco scripts que consumen `_data.npz` de a1 no se migraron; a1 sigue en pie.

#### Paso 4 — El modelo v0 = el mercado (2026-08-04)

v0 = `devig(par de Pinnacle)`, ya ejecutable como candidato de primera clase del evaluador. Lo que
tenía contenido en este paso era la pregunta que v0 obliga a hacer: **¿existe alguna mezcla de v0
con el modelo que le gane a v0 SOLO?** `a1_descomposicion.py` la respondió sólo EN MUESTRA (barrido
de pesos sobre los mismos datos, óptimo w=0.00). Fuera de muestra es otra pregunta.

Nuevo `mezcla_fuera_de_muestra()`: ajusta `y ~ logit(v0) + logit(candidato)` en un pliegue y lo
aplica al otro, comparando contra el v0 evaluado en ESE MISMO pliegue de prueba. Los pliegues son
temporadas enteras y no filas al azar a propósito — un corte aleatorio pondría juegos del mismo día
a ambos lados y el ajuste aprendería del futuro por la puerta de al lado.

```
 pliegue  n_test         v0   v0 recal     mezcla    b_cand    veredicto
    2024    2391    0.24042    0.24102    0.24098   -0.0650    v0 SOLO gana
    2025    2232    0.24063    0.24106    0.24131   -0.2129    v0 SOLO gana
```

**Ninguna mezcla le gana a v0, en ninguna dirección del corte**, y el coeficiente ajustado del
candidato es NEGATIVO en los dos pliegues — consistente con la regresión conjunta.

**El detalle que dice más que el veredicto**: `v0 recalibrado` (0.24102 / 0.24106) es PEOR que `v0`
crudo (0.24042 / 0.24063) en los dos pliegues. Ajustarle una logística al precio desvigorizado de
Pinnacle lo empeora: está tan bien calibrado que cualquier corrección le agrega ruido. Eso acota
cuánto se puede esperar de recalibrar contra el mercado — la respuesta es nada.

**Lectura honesta del estado, que no debe suavizarse**: sobre moneyline, con 4.623 juegos de
2024-2025, el modelo no aporta información sobre el precio (IC95 del coeficiente conjunto incluye
el cero, P(b>0)=14.2%), ninguna mezcla lo rescata fuera de muestra, y el ROI empeora al subir el
umbral de edge. Eso NO significa "usar el mercado para apostar" —un modelo que ES el mercado tiene
edge cero por construcción— sino que **hoy no hay apuesta demostrada que hacer en moneyline**. La
pregunta abierta del proyecto sigue abierta, y ahora con instrumento para responderla.

Tests: 3 más en `tests/test_evaluator.py` (13 en total), con control positivo — si una señal
plantada no aparece en la mezcla fuera de muestra, el instrumento no serviría para aceptar ninguna.

#### Paso 5 — Datos de entrada PIT (2026-08-04)

Dos preguntas de CONEXIÓN que el paso 0 había dejado sospechadas, y las dos salieron mejor de lo
temido:

**1. ¿Vivo y backtest corren motores distintos? NO — los motores son compartidos.** Ambos importan
y llaman las MISMAS funciones (`adjust_for_pitchers`, `adjust_for_bullpen`, `adjust_for_defense`,
`adjust_for_context`, `get_adjusted_lambdas`, `monte_carlo_advanced`). Lo que difiere es sólo la
FUENTE de datos: el vivo arma `game_data` con `MLBDataIntegrator` (acumulado de temporada al
momento de correr, que en producción es point-in-time por construcción — no existe el futuro), y
el backtest superpone las instantáneas PIT sobre el mismo dict vía `_merge_pit_*`. `PITCache` no
tiene una sola referencia en `modules/baseball_module/core|offense|context_engine|hfa` — es
backtest-only, confirmado.

Eso NO es un leak, pero sí acota qué transfiere el backtest a producción: transfiere en la medida
en que las dos fuentes llenen las mismas claves. El riesgo real es de FORMA, no de tiempo — si el
camino PIT deja vacía una clave que el vivo puebla, el motor compartido cae a su fallback y el
backtest mide el fallback. Es exactamente la clase de deriva que ya ocurrió una vez (la constante
de barrel% arreglada en el motor vivo y no en la copia PIT, 2026-07-11) y que `tte_formula.py`
unificó para ofensa.

**2. ¿La cobertura PIT se volvió a cortar a mitad de temporada? NO.** Los seis namespaces rolling
tienen 186-187 días por temporada, sin huecos, cubriendo marzo→septiembre completo en 2024 y 2025.
El bug #9 del 2026-07-11 —`savant.team_offense.rolling` congelado a mitad de temporada, usado en
silencio por el 59% de los juegos del backtest insignia— no volvió.

**Lo que sí se encontró, en el instrumento de cobertura**: `scripts/inputs_coverage.py` promediaba
sobre toda la historia, y la ventana cruza los arreglos. `weather_source` salía **55.9%** y
`home_injured_pa_share` **69.5%**, y los dos estaban al **100% desde el día en que aterrizó su
arreglo** (`092cefd` y `4f6b4c6`) — el agregado mezclaba el antes y el después.

Cosmético en un campo ya arreglado; **grave al revés**: un campo que se rompiera ayer aparecería al
90% y pasaría limpio el chequeo de "ningún campo en 0%", que es justamente la alarma que el script
existe para dar. Ahora compara el último día contra los previos y reporta el rumbo. El umbral de
alarma es "venía ≥50% y hoy no llegó ni una vez" — no basta con "vacío hoy", porque
`lineup_confirmed` al 3.8% es la realidad del mercado y está vacío muchos días sueltos. Ese matiz
lo impuso un test existente que el primer intento rompió.

Tests: 2 más en `tests/test_inputs_coverage_detector.py` (8 en total), uno por cada dirección —
el campo que dejó de llegar (alarma) y el campo ya arreglado (se reporta, no alarma).

**Residual del paso 5**: no existe una verificación automática de que el camino PIT y el vivo
llenen las mismas claves de `game_data` para el mismo juego. Hoy la única defensa es que los
motores sean compartidos, que detecta una divergencia de CÓDIGO pero no una de DATOS.

#### Paso 6 — La primera feature (2026-08-04)

En un build desde cero, acá se agrega UNA cosa a v0 y tiene que ganarse su coeficiente contra el
precio, fuera de muestra. La traducción honesta al proyecto real: **¿cuál de los nueve motores
aporta algo, individualmente, sobre el precio?**

`a7_alfa_por_motor.py` ya atacaba esto pero EN MUESTRA, y su resultado era: sólo `defense` con
coeficiente positivo e IC que no cruza cero (+0.0680, [+0.0071, +0.1299], P=99.0%). Con ocho
motores y dos temporadas, encontrar uno positivo es casi lo esperable por azar — condición
necesaria, no suficiente.

**Nuevo `a8_motores_fuera_de_muestra.py`** (solo lectura, repetible): ajusta
`y ~ logit(Pinnacle) + motor` en una temporada y lo aplica a la otra, comparando contra el Brier de
Pinnacle en esa MISMA temporada de prueba. La estandarización del motor usa media y desvío del
pliegue de ENTRENAMIENTO únicamente — estandarizar con la muestra completa mete el pliegue de
prueba en el ajuste por la puerta de al lado, fuga chica pero real en señales de coeficiente 0.05.

**Resultado: las 14 celdas son negativas. Ningún motor mejora a Pinnacle fuera de muestra, en
ninguna dirección del corte.**

```
  motor               entrena→prueba  Pinnacle  con motor    mejora     coef
  l0 (TTE ofensa)          2025→2024   0.24042    0.24105  -0.00063  -0.0943
  pitcher                  2025→2024   0.24042    0.24095  -0.00054  -0.0508
  defense                  2025→2024   0.24042    0.24145  -0.00103  +0.1192
  defense                  2024→2025   0.24063    0.24083  -0.00020  +0.0218
  hfa                      2024→2025   0.24063    0.24076  -0.00013  -0.0561
```

**`defense` es el hallazgo del paso**: el único que a7 marcaba como significativo en muestra es el
PEOR de los siete al probarlo en 2024 (−0.00103). Su coeficiente ajustado en 2025 vale +0.1192 y el
ajustado en 2024 vale +0.0218 — no transfiere. Era sobreajuste, y era justo el que alguien habría
tomado por bueno.

**Lo que este resultado NO dice**: que los motores sean dañinos. Agregar cualquier regresor ruidoso
a un predictor ya casi óptimo cuesta un poco de Brier por error de estimación, y las magnitudes
(−0.0001 a −0.0012) son de ese orden. Lo que dice es que **ninguno se paga a sí mismo** sobre
moneyline.

`park` se omite del análisis: mueve idéntico ambos lados, así que su aporte al moneyline es cero
por construcción.

**Consecuencia para el recorrido**: los pasos 7 a 12 (simulador, detección de valor, sizing,
publicación, CLV, producto) son maquinaria construida sobre la salida de estos motores. Auditarlos
encuentra defectos de plomería reales —el recorrido ya encontró varios— pero no cambia este
veredicto. La única puerta que sigue abierta con evidencia a favor son los mercados DERIVADOS,
donde el fix del simulador sí corrigió un sesgo medido (VAL-1.3: 11.58pp → 3.05pp) y donde nunca se
midió rentabilidad por falta de líneas históricas.

##### Paso 6 hecho como BUILD, no como auditoría: `l0` es la primera feature

`a8` pasó los nueve motores YA CONSTRUIDOS por el portón del paso 6 — eso es auditoría, no build.
El paso 6 de verdad construye UNA feature y la somete al portón. De lo que existe, el equivalente
es **`l0` (la TTE de ofensa)**, y no por gusto: los otros siete motores son MULTIPLICADORES sobre
`l0` (`pitcher`, `bullpen`, `defense`, `park`, `hfa`, `context`, `bias`). Desde cero no se puede
empezar por el ajuste de abridor porque no hay λ que ajustar todavía. `l0` es lo único que se
sostiene solo, y es además lo más sofisticado del proyecto.

`a9_l0_como_primera_feature.py` (solo lectura, repetible) la convierte a P(gana el local) con
Skellam —sin simulador, que no hace falta para moneyline— y la pasa por el evaluador completo:

```
  Brier l0 solo    0.24755  (ventaja sobre azar +0.979%)
  Brier mercado    0.24052  (ventaja sobre azar +3.792%)
  b_candidato      -0.4067   IC95 [-0.8576, -0.0035]   P(b>0)=2.0%
  ROI  2%: -0.29%   4%: -1.41%   6%: -1.73%   8%: -3.19%   10%: -4.69%
  mezcla fuera de muestra: gana v0 SOLO en los dos pliegues
```

**El IC excluye el cero POR ABAJO**, y eso es más fuerte que "no aporta": condicionado al precio,
la predicción de `l0` EMPEORA. Donde `l0` discrepa del mercado, el mercado tiene razón de forma
sistemática — lo que queda de `l0` tras descontar lo que el precio ya sabe es, sobre todo, su
sesgo.

**El matiz que corresponde a los otros ocho motores**: `l0` sola da 0.24755 y el pipeline completo
da 0.24620, contra 0.24052 del mercado. O sea que **los ocho ajustes SÍ mejoran sobre su propia
base — cierran el 19% de la distancia entre `l0` y el mercado — y ahí se detienen.** No es que los
motores no hagan nada; es que la base desde la que parten está demasiado lejos.

**Convergencia de los dos marcos**: un build desde cero, aplicando su propia regla en el paso 6, no
habría conservado ninguna de las nueve señales que este proyecto construyó. No porque estén mal
programadas —varias están muy bien hechas— sino porque ninguna se midió contra el precio mientras
se construía. Eso no lo descubrió el paso 6: lo causó que la balanza (paso 3) llegara décima en vez
de tercera.
