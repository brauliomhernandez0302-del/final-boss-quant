# ESTADO DEL SISTEMA — FINAL BOSS QUANT G8+
**Fecha:** 2026-05-06  
**Versión:** 2.1  
**Autor del reporte:** Claude Sonnet 4.6 (generado automáticamente)

---

## 1. RESUMEN EJECUTIVO

**FINAL BOSS QUANT G8+** es un sistema cuantitativo de predicción de apuestas deportivas enfocado principalmente en MLB, con soporte secundario para NBA, UFC y boxeo. El sistema combina modelado estadístico (Poisson/Monte Carlo), aprendizaje histórico, detección de valor en mercados y gestión de bankroll con Kelly fraccional.

**Estado general:** Operacional. Fases 1–6 completadas. El motor de aprendizaje histórico está inicializado con datos 2024-2025.

| Métrica | Valor |
|---------|-------|
| LOC estimadas (Python) | ~11,500 líneas |
| Tablas SQLite activas | 6 |
| Registros en `game_outcomes` | 4,859 |
| Registros en `historical_odds` | 4,695 |
| Correcciones de sesgo cacheadas | 60 (equipo × temporada) |
| Predicciones históricas en BD | 59 |
| Simulaciones MLB por ejecución | 5,000,000 |

---

## 2. ARQUITECTURA GENERAL

### 2.1 Punto de entrada

`app.py` (1,585 líneas) es el único entry point Streamlit. Define:

- **`AppConfig`** / **`SportConfig`** — configuración global desde `config.py`
- **`PredictionsDB`** — wrapper SQLite (`data/predictions_history.db`)
- **`BaseAnalyzer`** / **`MLBAnalyzer`** / **`NBAAnalyzer`** / **`UFCAnalyzer`** — clases que llaman a los módulos de cada deporte
- Funciones de renderizado UI (`render_mlb_results`, `render_nba_results`, `render_ufc_results`)
- `main()` — layout de página Streamlit con selector de deporte

**Tema visual:**
```python
PRIMARY   = "#00C2FF"   # Azul eléctrico
SUCCESS   = "#00C853"   # Verde confirmación
DANGER    = "#FF5252"   # Rojo alerta
```

### 2.2 Constantes globales (config.py — 54 líneas)

```python
APP_VERSION        = "2.1"
MLB_SIMULATIONS    = 5_000_000
NBA_SIMULATIONS    = 50_000
UFC_SIMULATIONS    = 100_000

KELLY_FRACTION     = 0.25          # Quarter-Kelly
MAX_RISK_PCT       = 0.05          # 5% máximo por apuesta
MIN_KELLY          = 0.01
MAX_KELLY          = 0.15
MIN_CONFIDENCE     = 0.65
MIN_EDGE           = 0.005         # 0.5% de ventaja mínima

LEAGUE_AVG_RUNS    = 4.5
LEAGUE_AVG_OPS     = 0.735
LEAGUE_AVG_ERA     = 4.15
LEAGUE_AVG_WHIP    = 1.30

ODDS_CACHE_TTL     = 300           # segundos (Streamlit cache)
ODDS_FILE_CACHE_TTL = 600          # segundos (archivo JSON)
MLB_FALLBACK_GAME_ID = 746_929     # Serie Mundial 2024
```

---

## 3. ESTRUCTURA DE ARCHIVOS

```
/home/raulio/
├── app.py                          (1,585 líneas) — Entry point Streamlit
├── config.py                       (54 líneas)    — Constantes globales
├── data_fetchers.py                (1,209 líneas) — MLB Stats API
├── odds_fetcher.py                 (383 líneas)   — The Odds API
├── odds_api.py                     (77 líneas)    — Fuzzy matching de equipos
├── storage.py                      (46 líneas)    — CSV bet logger
├── nba_stats_fetcher.py            (783 líneas)   — ESPN / Basketball Reference
├── injuries_fetcher.py             (892 líneas)   — Estado de lesiones
├── ufc_data_fetcher.py             (524 líneas)   — Stats UFC
├── train_historical.py             (359 líneas)   — Bootstrap learning engine
├── fetch_historical_odds.py        (571 líneas)   — Descarga odds históricas
│
├── modules/
│   ├── baseball_module/
│   │   ├── core/
│   │   │   └── run_module.py       (401 líneas)   — Orquestador pipeline MLB
│   │   ├── calibration/
│   │   │   ├── auto_calibrator.py                 — Calibración λ histórica
│   │   │   └── learning_engine.py  (318 líneas)   — Loop de aprendizaje
│   │   ├── hfa/
│   │   │   └── hfa_engine.py       (390 líneas)   — Home Field Advantage
│   │   ├── context_engine/
│   │   │   ├── pitcher_engine.py   (483 líneas)   — Ajuste por pitcher
│   │   │   └── pitchers_regression.py (283 líneas) — Regresión ERA
│   │   ├── montecarlo/
│   │   │   └── simulator.py                       — Simulador Poisson 5M
│   │   └── utils/helpers.py
│   │
│   ├── basketball_module.py        (~91K bytes)   — Predictor NBA G10+
│   ├── ufc_module.py               (21K bytes)    — Predictor UFC
│   ├── football_module.py          (26K bytes)    — Predictor Soccer (DESHABILITADO)
│   └── boxing_module.py            (8.2K bytes)   — Predictor Boxeo
│
├── data/
│   ├── predictions_history.db      (2.7MB)        — SQLite principal
│   ├── mlb_complete_data.json      (2.2KB)        — Cache MLB
│   └── .g8_autorun_once                           — Marker de ejecución única
│
└── .cache/                                        — JSONs cacheados (odds, bullpen, H2H)
```

---

## 4. PIPELINE MLB (módulo principal)

El módulo MLB es el más desarrollado y complejo del sistema. Implementa un pipeline secuencial de 7 etapas donde cada motor ajusta multiplicativamente los valores λ (lambda) de Poisson que representan las carreras esperadas.

### 4.1 Diagrama del pipeline

```
[MLBStatsAPI]
     ↓  game_pk, equipos, pitchers, stats de temporada
     ↓  λ_home = RPG_home,  λ_away = RPG_away
     ↓
[AutoCalibrator]
     ↓  ajusta por últimos 30 juegos, splits home/away,
     ↓  rachas recientes, fase de temporada
     ↓
[HFA Engine — get_adjusted_lambdas]
     ↓  factores de estadio (park factor, altitud, clima,
     ↓  domo vs exterior), splits bateador/pitcher
     ↓
[Pitcher Engine — adjust_for_pitchers]
     ↓  calidad ERA/FIP/WHIP, forma reciente (últimas 5),
     ↓  historial vs rival, días de descanso, bullpen
     ↓
[Pitcher Regression — calculate_pitcher_regression]
     ↓  regresión BABIP / LOB% / HR-FB% por suerte
     ↓  corrección ponderada por confianza (IP-based)
     ↓
[Monte Carlo — monte_carlo_advanced]
     ↓  5M muestras Poisson en bloques de 200K
     ↓  parada temprana si SE < 0.003
     ↓  p_home, p_away, p_over, p_under
     ↓
[Value Detector — evaluate_value_ultra]
     ↓  compara probabilidades del modelo vs odds de mercado
     ↓  EV%, Kelly fraccional, tier de confianza
     ↓
SALIDA: game_info, probabilities, lambdas_history, best_bets
```

### 4.2 AutoCalibrator

Distribución de pesos internos:

| Factor | Peso |
|--------|------|
| Ofensa | 0.25 |
| Defensa | 0.20 |
| Forma reciente | 0.20 |
| Splits home/away | 0.15 |
| Descanso/viaje | 0.10 |
| Contexto de temporada | 0.10 |

Ventana de análisis: 30 juegos; racha: últimos 10.

### 4.3 HFA Engine (390 líneas)

- `StadiumFactors` dataclass: `runs_factor`, `hr_factor`, `hits_factor`, `altitude`
- `WeatherConditions`: temp_F, wind_mph, humidity, precipitation, roof_closed
- Base de datos de ~30 estadios MLB con factores individuales
- Ajuste por altitud (Coors Field, etc.)
- Multiplicación diferenciada bateador/pitcher

### 4.4 Pitcher Engine (483 líneas)

| Factor | Peso |
|--------|------|
| Calidad del pitcher | 0.30 |
| Forma reciente | 0.20 |
| Historial vs rival | 0.15 |
| Fatiga/bullpen | 0.10 |
| Factor de parque | 0.10 |
| Viaje | 0.05 |
| Bullpen | 0.10 |

Métricas usadas: ERA, FIP, xFIP, SIERA, K/9, BB/9, ERA últimas 5, días descanso, pitch_count.

### 4.5 Pitcher Regression (283 líneas)

Indicadores de suerte para regresión:

| Indicador | Media de liga |
|-----------|---------------|
| BABIP | 0.300 |
| LOB% | 0.720 |
| HR/FB% | 0.125 |

Confianza basada en tamaño de muestra (IP). Devuelve tupla `(factor, confidence)`.

### 4.6 Monte Carlo Simulator

```python
MonteCarloLimits:
    λ_range      = [0.1, 20.0]
    sims_range   = [10_000, 10_000_000]
    default_sims = 5_000_000
    block_size   = 200_000
    lambda_noise = ±5% (varianza dinámica)
    se_threshold = 0.003  # parada temprana

Algoritmo:
    rng = numpy.random.default_rng()
    Por bloque de 200K:
        home_scores ~ Poisson(λ_home + noise)
        away_scores ~ Poisson(λ_away + noise)
        acumula: wins_home, wins_away, ties
    Parar cuando SE(p_home) < 0.003
```

### 4.7 Value Detector

**Fórmula EV:**
```
EV% = (prob × decimal_odds - 1) × 100
```

**Tiers de valor:**

| Tier | EV mínimo | Rating |
|------|-----------|--------|
| ULTRA | ≥ 15.0% | S |
| HIGH | ≥ 8.0% | A |
| MEDIUM | ≥ 4.0% | B |
| SLIGHT | ≥ 1.0% | C |
| NEUTRAL | 0.0% | D |
| NEGATIVE | < 0% | F |

**Mercados soportados:** Moneyline, Over/Under totales, Run Line ±1.5, First 5 Innings.

**Remoción de vig:** Métodos `multiplicative`, `power`, `shin` — probabilidades justas normalizadas a 1.0.

---

## 5. MOTOR DE APRENDIZAJE HISTÓRICO

### 5.1 Learning Engine (318 líneas)

El sistema cierra el loop de retroalimentación comparando predicciones con resultados reales.

**Tablas involucradas:**

- `game_outcomes` (4,859 filas): almacena λ predicho y carreras reales
- `ml_state` (60 filas): cache de correcciones de sesgo por equipo/temporada

**Flujo:**
1. `record_prediction()` — guarda λ pre-partido
2. `fetch_pending_outcomes()` — obtiene marcador final post-partido (MLB Stats API)
3. `compute_team_bias()` — calcula `mean(actual_runs / predicted_λ)` con mínimo 10 muestras
4. Sesgo clampeado a ±20%; cache válido 6 horas

**Corrección de sesgo:** Los equipos de alto punteo acumulan bias > 1.0 y los de bajo punteo < 1.0. Este prior ajusta predicciones futuras automáticamente.

### 5.2 train_historical.py (359 líneas)

Script de bootstrap que inicializa el motor con datos 2024-2025.

**Workflow:**
1. Obtiene todos los juegos `Final` de MLB Stats API (una llamada por temporada)
2. Calcula RPG por equipo y temporada
3. Popula `game_outcomes` con λ predicho = `LEAGUE_AVG_RUNS` (4.5) como prior inicial
4. Inserta marcadores reales de juegos completados
5. Calcula y cachea correcciones de sesgo en `ml_state`

**Resultado post-entrenamiento:**
- `game_outcomes`: 4,859 filas (temporadas 2024 y 2025)
- `ml_state`: 60 filas (30 equipos × 2 temporadas)

**Flags CLI:** `--seasons`, `--min-samples`, `--dry-run`

### 5.3 fetch_historical_odds.py (571 líneas)

Descarga moneylines pre-partido históricos desde The Odds API.

**Estrategia de snapshot:**
- Una llamada bulk por día-juego a las 17:00 UTC (mediodía ET, antes del partido)
- Regiones: `us` + `eu` (27+ casas de apuestas incluyendo Pinnacle)
- Mercado: `h2h` (moneyline únicamente)

**Costo estimado:** ~369 días × 20 requests = ~7,380 llamadas API totales

**Datos extraídos por juego:**

| Campo | Descripción |
|-------|-------------|
| `ml_home_best` / `ml_away_best` | Mejor línea disponible |
| `ml_home_best_bk` | Casa con mejor línea |
| `ml_home_cons` / `ml_away_cons` | Consenso entre casas |
| `ml_home_pin` / `ml_away_pin` | Línea Pinnacle (sharps) |
| `fair_prob_home` / `fair_prob_away` | Probabilidad justa (sin vig) |
| `n_bookmakers` | Número de casas disponibles |

**Estado actual:** 4,695 filas en `historical_odds`; las 162 partidas de Athletics 2025 están pendientes de re-descarga cuando se renueve la cuota de API.

**Flags CLI:** `--seasons`, `--dry-run`, `--enrich-only`

---

## 6. BASE DE DATOS SQLite

**Archivo:** `data/predictions_history.db` (2.7 MB)

### Esquema completo

#### Tabla: `game_outcomes` (4,859 filas)
```sql
game_pk          INTEGER PRIMARY KEY
game_date        TEXT
season           INTEGER
home_team        TEXT
away_team        TEXT
lambda_home      REAL        -- λ predicho home
lambda_away      REAL        -- λ predicho away
p_home           REAL        -- prob. de victoria home
p_away           REAL        -- prob. de victoria away
actual_home_runs INTEGER
actual_away_runs INTEGER
home_won         INTEGER     -- 1/0/-1 (empate)
created_at       TEXT
ml_home_open     REAL        -- moneyline apertura home
ml_away_open     REAL
ml_home_cons     REAL        -- moneyline consenso
ml_away_cons     REAL
ml_home_pin      REAL        -- Pinnacle sharps
ml_away_pin      REAL
market_prob_home REAL        -- prob. justa de mercado
market_prob_away REAL
```

#### Tabla: `ml_state` (60 filas)
```sql
key         TEXT
scope       TEXT        -- equipo o 'global'
season      INTEGER
value_json  TEXT        -- JSON con bias y metadata
sample_count INTEGER
updated_at  TEXT
PRIMARY KEY (key, scope, season)
```

#### Tabla: `historical_odds` (4,695 filas)
```sql
game_pk         INTEGER UNIQUE
game_date       TEXT
season          INTEGER
odds_api_id     TEXT
home_team       TEXT
away_team       TEXT
snapshot_ts     TEXT
ml_home_best    REAL
ml_away_best    REAL
ml_home_best_bk TEXT
ml_away_best_bk TEXT
ml_home_cons    REAL
ml_away_cons    REAL
ml_home_pin     REAL
ml_away_pin     REAL
fair_prob_home  REAL
fair_prob_away  REAL
n_bookmakers    INTEGER
created_at      TEXT
```

#### Tabla: `predictions` (59 filas)
Predicciones en vivo con EV, Kelly, confianza, rating, resultado y P&L.

#### Tabla: `results` (0 filas — sin datos aún)
Resultados finales enlazados a predicciones.

#### Tabla: `sport_stats` (0 filas — sin datos aún)
Tasa de aciertos, ROI, EV promedio por deporte.

---

## 7. MÓDULOS DE OTROS DEPORTES

### 7.1 NBA (basketball_module.py — ~91K bytes)

NBA G10+ ULTRA PRO V2 con 12 motores:

| Motor | Peso |
|-------|------|
| Injuries | 0.20 |
| Context (descanso, back-to-backs, viaje) | 0.18 |
| Monte Carlo | 0.16 |
| Pace (posesiones/48min) | 0.14 |
| Sharp Money (movimientos de steam) | 0.12 |
| Matchups (10+ factores) | 0.10 |
| Shooting Luck | 0.08 |
| HFA | 0.02 |
| Blowout | 0.02 |
| Risk | 0.02 |
| Stability | 0.02 |
| Trends | 0.02 |

### 7.2 UFC (ufc_module.py — 21K bytes)

- Matriz de ventaja por estilo (striker vs grappler, etc.)
- Stats: wins, KO rate, submission rate, striking accuracy, takedown rate
- Modificadores por título en juego
- 100,000 simulaciones Monte Carlo

### 7.3 Soccer (football_module.py — 26K bytes)

**Estado: DESHABILITADO.** Soporta EPL, La Liga, Bundesliga, Serie A. Cargado con `safe_import()` pero no renderizado en el dashboard actual.

### 7.4 Boxeo (boxing_module.py — 8.2K bytes)

Motor básico de predicción de peleas de boxeo.

---

## 8. INTEGRACIÓN DE APIs EXTERNAS

### 8.1 MLB Stats API
- **URL base:** `statsapi.mlb.com/api/v1`
- **Autenticación:** Ninguna (gratuita y pública)
- **Endpoints usados:** juegos del día, datos por `game_pk`, stats de equipo, stats de pitcher
- **Cacheo:** JSON en `.cache/` con TTL configurable

### 8.2 The Odds API
- **URL base:** `api.the-odds-api.com/v4`
- **Key:** `ODDS_API_KEY` en `.env`
- **Regiones:** `us`, `uk`, `eu` (27+ casas)
- **Mercados:** `h2h`, `totals`, `spreads`
- **Throttle:** 0.25s entre llamadas (cortesía)
- **Cacheo:** 600 segundos en disco

### 8.3 OpenWeather (opcional)
- **Key:** `OPENWEATHER_API_KEY` en `.env`
- Usado para ajustes climáticos en el HFA engine

### 8.4 ESPN / Basketball Reference
- Scraping sin autenticación en `nba_stats_fetcher.py` e `injuries_fetcher.py`

### 8.5 Fuzzy matching de equipos (odds_api.py)
- `get_best_odds_for_teams(home, away, sport)` normaliza nombres de equipos (abreviaciones, variantes por ciudad)

---

## 9. GESTIÓN DE BANKROLL

**Fórmula Kelly fraccional:**
```
kelly_full  = (prob × decimal_odds - 1) / (decimal_odds - 1)
kelly_frac  = kelly_full × KELLY_FRACTION (0.25)
stake       = min(kelly_frac × bankroll, MAX_RISK_PCT × bankroll)
stake       = max(stake, MIN_STAKE)
```

- `MIN_KELLY = 0.01` (mínimo 1% para filtrar señales débiles)
- `MAX_KELLY = 0.15` (tope del 15% para Kelly completo antes de fracción)
- `MAX_RISK_PCT = 0.05` (tope absoluto: 5% del bankroll por apuesta)

**Log de apuestas:** `data/bets_log.csv` — historial CSV gestionado por `storage.py`.

---

## 10. PATRONES DE DISEÑO

| Patrón | Descripción |
|--------|-------------|
| **Lambda pipeline** | Cada motor ajusta λ_home y λ_away multiplicativamente. El λ final alimenta Monte Carlo. |
| **Safe module loading** | `safe_import()` captura excepciones; los analizadores degradan graciosamente si un módulo falta. |
| **Dual mode (Streamlit + CLI)** | `run_module.py` detecta Streamlit con try/except; en terminal selecciona el primer juego. |
| **Idempotent DB writes** | Restricciones `UNIQUE` en `game_pk` evitan duplicados; re-ejecuciones son seguras. |
| **Feedback loop** | Learning engine compara λ predicho con carreras reales; acumula sesgo por equipo para corregir futuras predicciones. |
| **Bootstrap prior** | `train_historical.py` usa `LEAGUE_AVG_RUNS` como λ inicial neutral; cada equipo construye su sesgo propio desde ahí. |

---

## 11. HISTORIAL DE COMMITS RECIENTE

```
ad35c70  feat(phase-6): fetch_historical_odds.py — 2024-2025 pre-game moneylines
6bb750f  feat(phase-5): train_historical.py — bootstrap learning engine
4eadd92  feat(phase-4): SQLite persistence for calibrator & learning engine
02ab6b3  fix: game dict key mismatch + MC probability log keys in run_module
dd0a2d0  refactor(phase-3): fix EC-01/02/03 math bugs + PM-02/03 lambda pipeline bugs
503faaf  refactor(phase-2): wire config.py constants into all modules
7603435  refactor(phase-1): consolidate structure, create config.py, rename LambdaCalibrator
2cd671f  fix: cache format mismatch in odds_fetcher load_cache
bb3e8b0  fix: standings float crash + odds API 404 in value detector
615e9b4  docs: add CLAUDE.md architecture guide
64d2dba  feat: dynamic per-team lambdas with automatic seasonal weights
15263de  feat: game log endpoint + era_last_5/days_rest in pitcher engine
81faa7b  fix: probabilities keys p_home/p_away + value detection wired
94d0423  fix: moneyline odds API integration + standings + full pipeline
bf984f5  security: add sensitive files to .gitignore
```

Las fases 1–6 conforman un ciclo de refactoring estructurado:

| Fase | Contenido |
|------|-----------|
| 1 | Consolidación de estructura, creación de `config.py`, renombrado de `LambdaCalibrator` |
| 2 | Wiring de constantes de `config.py` en todos los módulos |
| 3 | Corrección de bugs matemáticos en motores (EC-01/02/03, PM-02/03) |
| 4 | Persistencia SQLite para calibrador y learning engine |
| 5 | Bootstrap del motor de aprendizaje con datos históricos 2024-2025 |
| 6 | Descarga de odds históricas pre-partido desde The Odds API |

---

## 12. DEPENDENCIAS (requirements.txt)

```
requests
pandas
numpy
matplotlib
flask
websocket-client
scikit-learn
beautifulsoup4
streamlit
```

**Entorno virtual:** `mi_entorno/` — activar con `source mi_entorno/bin/activate`

---

## 13. ESTADO ACTUAL POR COMPONENTE

| Componente | Estado | Notas |
|------------|--------|-------|
| Pipeline MLB (7 motores) | ✅ Operacional | Fases 1-6 completadas; bugs matemáticos corregidos |
| Monte Carlo 5M | ✅ Operacional | Early stopping SE < 0.003; bloques de 200K |
| HFA Engine | ✅ Operacional | ~30 estadios MLB; integración climática opcional |
| Pitcher Engine | ✅ Operacional | 7 factores ponderados; regresión BABIP/LOB%/HR-FB% |
| Learning Engine | ✅ Inicializado | 4,859 juegos; 60 correcciones de sesgo cacheadas |
| Historical Odds | ⚠️ Parcial | 4,695 filas; 162 juegos Athletics 2025 pendientes (cuota API) |
| Value Detector | ✅ Operacional | 4 tiers (ULTRA/HIGH/MEDIUM/SLIGHT); Kelly fraccional |
| NBA Module | ✅ Operacional | 12 motores; 50K simulaciones |
| UFC Module | ✅ Operacional | 100K simulaciones; matriz de estilos |
| Soccer Module | ❌ Deshabilitado | Código presente pero no renderizado en UI |
| Tabla `results` | ⚠️ Vacía | 0 filas — sin resultados post-predicción registrados |
| Tabla `sport_stats` | ⚠️ Vacía | 0 filas — sin métricas de ROI/aciertos acumuladas |

---

## 14. DEUDA TÉCNICA Y OPORTUNIDADES DE MEJORA

### Crítico
- **`results` y `sport_stats` vacías:** El loop de evaluación de rendimiento (ROI, tasa de aciertos) no está cerrando. Hay que implementar la lógica de post-procesamiento que lea `predictions`, compare con `game_outcomes` y escriba en `results` y `sport_stats`.
- **162 juegos Athletics 2025 faltantes:** Re-ejecutar `fetch_historical_odds.py --enrich-only` cuando se renueve la cuota de The Odds API.

### Importante
- **Sin tests unitarios:** Los motores (especialmente `pitchers_regression.py` y `auto_calibrator.py`) no tienen tests. Un cambio pequeño puede propagar errores silenciosos a través del pipeline λ.
- **Sin versionado de schema DB:** Las migraciones de esquema no están rastreadas. Un `ALTER TABLE` manual puede romper instancias existentes. Considerar Alembic o un sistema de migraciones propio.
- **Llamadas API síncronas:** `data_fetchers.py` y `odds_fetcher.py` son síncronos. Para múltiples juegos simultáneos, `asyncio` + `httpx` reduciría la latencia de carga.

### Menor
- **`results` y `sport_stats`:** Poblar estos datos habilitaría el dashboard de rendimiento del modelo.
- **Soccer Module:** Re-habilitar si se decide expandir cobertura.
- **Capa de caché centralizada:** Actualmente hay dos sistemas de caché (Streamlit `@st.cache_data` y archivos JSON manuales). Una capa Redis unificada simplificaría la lógica.
- **Logging estructurado:** Pasar a JSON-logging permitiría ingesta en herramientas de observabilidad.

---

## 15. CÓMO EJECUTAR EL SISTEMA

```bash
# Activar entorno virtual
source mi_entorno/bin/activate

# (Opcional) Re-bootstrap si se cambia la BD
python train_historical.py --seasons 2024 2025

# (Opcional) Actualizar odds históricas cuando se renueve cuota API
python fetch_historical_odds.py --seasons 2024 2025

# Lanzar la app
streamlit run app.py
```

**Variables de entorno requeridas en `.env`:**
```
ODDS_API_KEY=<clave de The Odds API>
OPENWEATHER_API_KEY=<opcional, para ajustes climáticos>
```

---

*Reporte generado automáticamente el 2026-05-06 por Claude Sonnet 4.6.*  
*Basado en inspección completa del código fuente, esquemas de BD y historial de git.*
