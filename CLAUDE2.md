# CLAUDE2.md — Estado actual del sistema (G10 Ultra Pro)

> Documento de referencia actualizado. Reemplaza la descripción del CLAUDE.md original (G8+).
> Compara con CLAUDE.md para ver qué cambió y qué queda pendiente.

---

## Qué es este proyecto

**FINAL BOSS QUANT G10 Ultra Pro** — sistema cuantitativo de predicción deportiva.
Analiza partidos de MLB (principal), NBA y UFC. Identifica oportunidades de apuesta
con EV positivo usando simulación Monte Carlo, modelado Poisson, Kelly criterion y
datos de Statcast/FanGraphs/The Odds API.

---

## Cómo correr

```bash
source mi_entorno/bin/activate
streamlit run app.py
```

Variables de entorno en `.env`:
```
ODDS_API_KEY=...
OPENWEATHER_API_KEY=...   # opcional
```

---

## Arquitectura — app.py y UI

`app.py` es el entry point de Streamlit. **Ya no es monolito** — fue descompuesto en:

| Módulo | Responsabilidad |
|--------|----------------|
| `app.py` | Orquestador: page config, tabs, sport routing |
| `ui/components.py` | Temas, header, footer, SportConfig |
| `ui/mlb.py` | MLBAnalyzer + render_mlb_results |
| `ui/sidebar.py` | Sidebar + historial |
| `ui/odds_loader.py` | Carga y filtro de odds para el selector |
| `db/predictions_db.py` | PredictionsDB (SQLite wrapper) |

**Diferencia vs CLAUDE.md original**: en G8+ todo vivía en `app.py` (~1000 líneas),
incluyendo las clases Analyzer, la DB, y todo el render. Ahora son capas separadas.

---

## Pipeline MLB — 9 pasos (la parte que más cambió)

`run_module.py` orquesta una cadena secuencial de engines. Cada engine recibe
`(λ_home, λ_away, game_data)` y devuelve `(λ_home_adj, λ_away_adj, metadata)`.

### λ_base — True Talent Engine

**Antes (G8+):** `get_team_lambda()` usaba runs/game histórico + fallback a LEAGUE_AVG.

**Ahora:** `TrueTalentOffenseEngine` (`offense/true_talent_engine.py`):
- Statcast xwOBA por jugador → agrega a nivel de equipo (PA-weighted)
- barrel%, BB%-K%, wRC+ aproximado desde wOBA
- Regresión Bayesiana a la media por PA (early-season safe)
- Blend temporada actual + temporada anterior (prior_w = 1000/(1000+PA))
- Caches en instancia: Savant actual y Savant prior ambos en memoria
  (segunda llamada para el away team no re-fetchea nada)

### PASO 1 — AutoCalibrator

**Antes (G8+):** hacía offense_mult + defense_mult + rest + forma en uno solo
→ triple-counting con los engines de pitcher, bullpen y defensa.

**Ahora** (`calibration/auto_calibrator.py`):
- Solo aplica **forma reciente** (last_10, streak) y **contexto de temporada tardía**
- `tte_active=True` → skip offense_mult (TTE ya lo cubre con Statcast)
- Cap ajustado a ±8% (antes ±15%, con 5 factores; ahora solo 2)
- `_defense_multiplier` **eliminado** → triple-counting removido

### PASO 2 — Park + Weather Engine

**Antes (G8+):** HFA mezclaba parque + crowd boost + altitude en un solo factor.

**Ahora** (`hfa/park_weather_engine.py`):
- Park factor (simétrico: afecta a ambos equipos igual)
- Weather mult (temperatura, viento, humedad)
- Altitude (Coors Field, Chase Field)
- **Separado de HFA** porque el parque no es ventaja de local — es el ambiente

### PASO 3 — HFA Engine

**Antes (G8+):** mezclaba park factor + crowd + back-to-back.

**Ahora** (`hfa/hfa_engine.py`) — solo efectos asimétricos:
- **Crowd boost** → aumenta λ_home únicamente
- **Travel fatigue** → reduce λ_away únicamente (millas + zonas horarias)
- back-to-back **eliminado** de aquí (vive en Contextual Engine)
- Metadata: `hfa_boost_runs`, `hfa_mult`, `travel_penalty`, `park_name`

### PASO 4 — Defensive Efficiency Engine

**Nuevo en G10** (`context_engine/defensive_efficiency_engine.py`):
- DER = 1 − BABIP_allowed (fielding puro, sin pitching)
- OAA (Outs Above Average) de Savant cuando disponible
- Regresión Bayesiana por BIP (k=500, estabilización a ~19 juegos)
- Convención correcta: defensa home → reduce λ_away; defensa away → reduce λ_home
- Cap ±5%

### PASO 5 — Pitcher Engine

**Antes (G8+):** usaba ERA directamente. AutoCalibrator también aplicaba ERA
vía defense_mult → doble conteo.

**Ahora** (`context_engine/pitcher_engine.py`):
- Jerarquía de métricas: SIERA > xFIP > FIP > ERA (con Savant/FanGraphs enrichment)
- Regresión Bayesiana por IP (k_TBF=350; a IP=0 → colapsa a liga promedio)
- est_wOBA penalty, barrel% penalty
- Form mult (era_last_5 vs season ERA), fatigue (días de descanso + pitch count)
- Platoon splits, matchup vs oponente
- Delta formula (no producto): `total = 1 + Σ((factor_i − 1) × weight_i)`

### PASO 6 — Bullpen Engine

**Nuevo en G10** (`context_engine/bullpen_engine.py`):
- 4 señales: xwOBA against (0.40), K%-BB% (0.30), ERA (0.20), barrel% (0.10)
- Todos los datos de relievers específicamente (endpoint `pitcherType=R`)
- Regresión Bayesiana por TBF
- Innings weighting: `(9 − avg_ips) / 9` — el bullpen solo mueve la aguja para
  los innings que realmente pitchea
- `effective_era = quality_mult × LG_BP_ERA` disponible en metadata

### PASO 7 — Contextual Engine

**Nuevo en G10** (`context_engine/contextual_engine.py`):
- **B2B**: ×0.960 al equipo que jugó ayer
- **Rest ≥ 3 días**: ×0.980 (óxido)
- **Umpire zone factor**: simétrico, clip [0.96, 1.04], mínimo 4 juegos de muestra
- Consolida lo que antes estaba disperso en HFA + AutoCalibrator

### PASO 8 — Monte Carlo

**Antes (G8+):**
- Clip duro `[3.0, 7.0]` antes de entrar al MC — sobreescribía todo el trabajo del pipeline
- `lambda_noise=0.05` hardcodeado
- No pasaba `total_line` del mercado
- No computaba run-line probabilities

**Ahora** (`montecarlo/simulator.py` + `run_module.py`):
- Clip ampliado a `[1.5, 12.0]` — solo protección contra valores absurdos
- `lambda_noise` dinámico: 0.04 (TTE+Savant+FIP), 0.06 (parcial), 0.08 (legacy)
- `total_line` del mercado se fetchea ANTES del MC y se pasa al simulador
- `p_rl_home` / `p_rl_away` (run-line ±1.5) calculados desde muestras, cero sims extra
- F5 lambda = `λ_post_pitcher × F5_SCALE` (derivado del pipeline, no de ERA cruda)
- Early stopping: SE < 0.0005 después de 500K sims mínimo

### PASO 9 — Value Detection

**Antes (G8+):** solo moneyline. Bootstrap con for-loop Python sobre 500K elementos.

**Ahora** (`core/value_detector.py`):
- Todos los mercados: moneyline, totals O/U, run-line ±1.5, F5 ML + totals
- Pinnacle como referencia fair-line (devig multiplicativo)
- Bootstrap vectorizado con subsample 10K: ~50x más rápido
- `GameOdds` recibe `total_line`, `total_over/under`, `runline_home/away`

---

## Odds Fetcher

**Antes (G8+):** `odds_api.py` y `odds_fetcher.py` eran dos archivos con lógica
duplicada. `get_best_odds_for_teams` solo extraía moneyline.

**Ahora** (`odds_fetcher.py`):
- Archivo único — `odds_api.py` eliminado
- `get_best_odds_for_teams` extrae los 3 mercados que ya fetcheaba la API:
  `total_line/over/under` (totals), `runline_home/away` (spreads)
- Best price across all bookmakers para cada mercado

---

## Aprendizaje continuo

(`calibration/learning_engine.py`) — existía en G8+, no modificado en esta sesión:
- **Kalman filter**: ajuste bayesiano del λ_base por equipo/rol/temporada
- **Platt calibration**: shrinkage de las probabilidades del MC (refitted semanalmente)
- **Gradient descent**: ajusta los pipeline weights (calibration, hfa, pitcher)
  basado en errores de predicción históricos
- `record_prediction()` persiste cada análisis para aprendizaje futuro

---

## Datos de persistencia

| Archivo | Qué guarda |
|---------|-----------|
| `data/predictions_history.db` | Predicciones, outcomes, Platt params, pipeline weights |
| `data/bets_log.csv` | Log de apuestas (storage.py) |
| `.cache/odds_last.json` | Odds cacheadas (TTL configurable) |
| `.cache/tte_savant_*.json` | Statcast batters (6h TTL) |
| `.cache/bp_savant_*.json` | Statcast pitchers/bullpen (6h TTL) |
| `.cache/tte_roster_*.json` | Rosters (24h TTL) |
| `.cache/tte_hitting_*.json` | Team hitting stats (6h TTL) |

---

## Lo que QUEDA PENDIENTE

### Alta prioridad

1. **Backtest completo** — correr `backtest_and_retrain.py` con el pipeline nuevo.
   El accuracy previo era 57.84% / ROI +9.36% (edge ≥ 5%). Hay que validar que
   los cambios de arquitectura no lo degradaron.

2. **`pitchers_regression.py`** — está desconectado del pipeline (ver comentario
   en el archivo: "NOT CONNECTED TO THE PIPELINE"). Requiere BABIP/LOB%/HR-FB%
   por start, que ninguna API gratuita provee a esa granularidad. Decisión pendiente:
   eliminarlo o conectarlo cuando haya fuente de datos.

3. **F5 en Value Detection** — `evaluate_value_ultra` se llama con `analyze_f5=False`
   en run_module. El engine de F5 existe y funciona pero no está activado para
   producción. Requiere odds F5 del fetcher (actualmente no se extraen).

### Media prioridad

4. **NBA y UFC** — los módulos existen pero no tienen la arquitectura de pipeline
   por engines. Son monolitos. No son el foco actual pero eventualmente necesitan
   el mismo refactor.

5. **DER desde data_fetchers** — el campo `defense_home/away` se construye en
   `get_complete_game_data` desde los stats de pitching (hits, AB, K, HR, SF).
   OAA viene de Savant. Verificar que en producción los campos llegan correctamente
   al Defensive Efficiency Engine.

6. **F5 odds en el fetcher** — `get_best_odds_for_teams` no extrae odds de F5
   (f5_ml_home, f5_total_line, etc.). The Odds API sí las provee en algunos
   mercados pero no están siendo capturadas.

### Baja prioridad / deuda técnica

7. **Soccer module** — deshabilitado. `football_module.py` existe pero no está
   conectado al pipeline de engines.

8. **`weight_optimizer.py`** — archivo sin callers conocidos. Posible código
   de exploración del gradient descent. Revisar si es útil o eliminar.

9. **Track record system** — `run_daily_picks.py` + `track_record/` añadidos
   anteriormente. Verificar que funcionan con el nuevo pipeline.

---

## Invariantes del pipeline (no romper)

```
Away pitcher  → reduce λ_home  (no λ_away)
Home pitcher  → reduce λ_away  (no λ_home)
Home defense  → reduce λ_away  (no λ_home)
Away defense  → reduce λ_home  (no λ_away)
HFA crowd     → aumenta λ_home únicamente
Travel        → reduce λ_away únicamente
Park factor   → simétrico (ambos igual)
Umpire        → simétrico (ambos igual)
```

## Convenciones de código

- Cada engine expone una función pública `adjust_for_X(lh, la, game_data) → (lh, la, meta)`
- Singleton a nivel módulo para engines con caché pesada (TTE, BullpenEngine)
- Regresión Bayesiana: `(obs × n + prior × k) / (n + k)`; a n=0 → prior; a n=k → 50/50
- Delta formula en PitcherEngine: `total = 1 + Σ((fi − 1) × wi)`; pesos suman 1.0
- Tests en `tests/`: 142 tests, todos deben pasar antes de cualquier merge
