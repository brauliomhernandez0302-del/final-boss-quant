# AUDIT INTERNO MLB PIPELINE — Hallazgos Acumulativos

Fecha inicio: 2026-05-25
Estado: EN PROGRESO

---

## RESUMEN EJECUTIVO

*(Se actualiza después de cada motor)*

- Motores auditados: 9/10
- Bugs CRÍTICOS: 6
- Bugs MEDIOS: 14
- Bugs BAJOS: 18
- Áreas oscuras: 8
- Magic numbers sin justificación: 58

---

## MOTOR #1 — TRUE TALENT ENGINE (TTE)

Estado: AUDITADO ✓
Archivo: `modules/baseball_module/offense/true_talent_engine.py`

### A. Interface Contract

- **Entrada:** team, season, game_data (PA, stats de FanGraphs/Savant)
- **Salida:** λ_talent ∈ [3.0, 7.0] (clamp explícito)
- **Sin dependencias upstream** (primer motor del pipeline)

### B. Fórmulas internas

```
composite = f_xwoba×0.50 + f_barrel×0.30 + f_plate×0.20
λ_talent = prior_w × λ_prior + current_w × λ_cur
prior_w   = 1000 / (1000 + pa_cur)
plate_reg_val = max(0.85, min(1.15, 1.0 + disc_cur × 3.5))
```

### C. Bugs encontrados

**BUG #1 (BAJO): Docstring desincronizado con código real**
- El docstring menciona wRC+ pero el código usa xwOBA como señal principal.
- `_K_WRC = 400` permanece como dead code (constante sin uso).

**BUG #2 (MEDIO): LG_XWOBA=0.312 inconsistente con Pitcher Engine (0.320)**
- TTE normaliza ofensa con `LG_XWOBA=0.312`.
- Pitcher Engine normaliza xwOBA permitida con `_LG_XWOBA_ALLOWED=0.320`.
- Son valores distintos para la misma liga/época. Si la ofensa promedio batea 0.312 xwOBA pero el pitcher promedio permite 0.320 xwOBA, existe un gap de 0.008 que se convierte en sesgo sistemático: el modelo sobreestima pitchers vs bateadores.

**BUG #3 (MEDIO): Doble shrinkage — 15% confianza en datos actuales a 30 juegos**
- `prior_w = 1000 / (1000 + pa_cur)` con _PRIOR_PA_EQUIVALENT=1000.
- A 30 juegos (~110 PA), prior_w ≈ 90% → solo 10% del dato actual.
- A 50 juegos (~183 PA), prior_w ≈ 85% → 15% del dato actual.
- El TTE ya shrinkea fuertemente. Luego el Kalman blend (35%) shrinkea de nuevo.
- Triple shrinkage total cuando se agrega el bias dampening.

### D. Áreas oscuras

**ÁREA OSCURA #1: Multiplier 3.5 en plate discipline sin justificación empírica**
- `plate_reg_val = 1.0 + disc_cur × 3.5`
- No hay cita de investigación ni derivación del backtest.

### E. Evidencia empírica (5,422 juegos)

| Métrica | Valor |
|---------|-------|
| λ_home post-TTE: mean / p5 / p95 | 4.41 / 3.57 / 5.38 |
| λ_away post-TTE: mean / p5 / p95 | 4.34 / 3.43 / 5.37 |
| λ_home min / max | 2.77 / 6.59 |
| λ_away min / max | 2.74 / 7.77 |
| λ_home fuera de clamp [3.0, 7.0] | 203 juegos (3.7%) |
| λ_away fuera de clamp [3.0, 7.0] | 386 juegos (7.1%) |

### F. Acoplamiento

- Output → Kalman Adjustment (Motor #2)
- Sin dependencias upstream

### G. Magic numbers

- `3.5` (plate discipline multiplier)
- `1000` (prior PA equivalent — _PRIOR_PA_EQUIVALENT)
- `0.50 / 0.30 / 0.20` (pesos composite xwOBA/barrel/plate)

---

## MOTOR #2 — KALMAN ADJUSTMENT

Estado: AUDITADO ✓
Archivo: `modules/baseball_module/calibration/learning_engine.py`

### A. Interface Contract

- **Entradas:** team, context ∈ {offense_home, offense_away, defense_home, defense_away}, season, model_lambda
- **Salidas:** λ_adjusted ∈ [1.0, 12.0] (clamp muy permisivo)
- **Llamadas en pipeline (run_module.py ~líneas 434-456):**
  1. `lh = kalman(home_team, "offense_home", season, lh)`
  2. `la = kalman(away_team, "offense_away", season, la)`
  3. `la = kalman(home_team, "defense_home", season, la)` ← Fase 2.1

### B. Fórmulas internas

```
Blend:     result = 0.65 × model_λ + 0.35 × kalman_x_est   → clamp [1.0, 12.0]
Update:    K = (p_est + Q) / (p_est + Q + R)               Q=0.025, R=9.0
           x_est ← x_est + K × (observed - x_est)
Bias raw:  ratio = actual_runs / lambda_final
Dampened:  dampened = 1.0 + 0.65 × (raw_bias - 1.0)        → clamp [0.70, 1.30]
```

Steady-state K ≈ 5% por observación. Con n=162 juegos/season, x_est converge lentamente.

### C. Bugs encontrados

**BUG #1 (CRÍTICO): Gradient descent NO converge — pipeline weights = 1.000**
- Tras 5,422 juegos y 3 temporadas, TODOS los pesos permanecen exactamente en 1.000.
- Causa raíz: los stage_factors de defensa siempre llegan como 1.0 (ver BUG #2), y los
  factores de contexto también son siempre 1.0 (motor muerto). Gradiente = 0 → no hay
  actualización. El sistema de pesos del pipeline es decorativo.

**BUG #2 (CRÍTICO): defense_home invisible a stage_factors — ordering bug**
- `la = kalman(home_team, "defense_home", season, la)` ocurre ~línea 448.
- `_stage_factors = {}` se inicializa ~línea 468 (DESPUÉS del ajuste Kalman).
- Resultado: el ajuste Kalman de defensa nunca se registra en `stage_factors_json`.
- El gradient descent NUNCA puede aprender el peso correcto para la defensa.

**BUG #3 (MEDIO): compute_team_bias usa lambda_final como denominador**
- `ratio = actual_runs / lambda_final` donde lambda_final incluye todos los ajustes.
- El dampening `0.65 × (raw_bias - 1)` fue diseñado asumiendo denominador = lambda_base
  (pre-Kalman). Con lambda_final como denominador, el cálculo matemático es inconsistente.
- Evidencia: `mean(actual_home / lambda_home) = 1.0088` — el modelo subestima globalmente.

**BUG #4 (MEDIO): Bias único home+away — contextos muy diferentes ignorados**
- Mismo raw_bias se aplica a todos los contextos del equipo.
- Caso extremo: Milwaukee tiene residual home=+3.2pp, away=+11.65pp.
- El bias resultante es un promedio que subestima el problema away.

**BUG #5 (BAJO): Clamp Kalman [1.0, 12.0] inconsistente con TTE [3.0, 7.0]**
- TTE garantiza λ ∈ [3.0, 7.0]. Kalman puede devolver hasta 12.0.
- Empíricamente no se observan valores extremos, pero no hay garantía estructural.

### D. Áreas oscuras

**ÁREA OSCURA #1: ¿Por qué stage_factors de HFA y Context son siempre 0.0 std?**
- `away_hfa` std=0.0000, `home_context` y `away_context` std=0.0000.
- Pre-señal de motores HFA (Motor #7) y Contextual (Motor #8) que parecen muertos.
- Pendiente de confirmación en audits correspondientes.

### E. Evidencia empírica

| Métrica | Valor |
|---------|-------|
| Pipeline weights (todos, todas las temporadas) | **1.000 (invariante)** |
| mean(actual_home / lambda_home) | 1.0088 |
| Kalman offense_home x_est: p5/p95 | 3.37 / 5.64 |
| Teams con bias clampeado (±30%) | 0 de 31 |
| Teams con bias > ±20% | 0 de 31 |
| away_hfa stage factor std | 0.0000 |
| context stage factors std | 0.0000 |

### F. Acoplamiento

- Lee: λ_base del TTE (Motor #1)
- Escribe: λ post-Kalman → input a Team Bias → Pitcher Engine
- Crítico: defense_home nunca registrado en stage_factors → gradient blind

### G. Magic numbers

- `_KALMAN_BLEND = 0.35`
- `_KF_Q = 0.025`
- `_KF_R = 9.0`
- `_BIAS_CLAMP = 0.30`
- `_LR = 0.01`
- `_MIN_WEIGHT = 0.30`
- `_MAX_WEIGHT = 1.50`
- `_MIN_SAMPLES = 10`

---

## MOTOR #3 — PITCHER ENGINE

Estado: AUDITADO ✓
Archivo: `modules/baseball_module/context_engine/pitcher_engine.py`

### A. Interface Contract

- **Entradas:** lh, la (floats), game_data (pitcher_home, pitcher_away dicts)
- **Salidas:** (lh_new, la_new, metadata) donde cada λ ∈ [clamp_interno]
- **Clamp de salida:** total_multiplier ∈ [0.65, 1.45] por pitcher
- **Llamada en pipeline:** `lh, la, meta = adjust_for_pitchers(lh, la, game_data)`

**Nota de orientación (INVERSIÓN SEMÁNTICA):**
- Away pitcher (`pitcher_away`, `is_home=False`) → ajusta **λ_home** (home lineup)
- Home pitcher (`pitcher_home`, `is_home=True`) → ajusta **λ_away** (away lineup)
- En `stage_factors_json`: "home_pitcher" = adj_home (home pitcher → λ_away); "away_pitcher" = adj_away (away pitcher → λ_home). **El nombre en stage_factors es el PITCHER, no el team que recibe el ajuste.**

### B. Fórmulas internas

**Sub-factores (todos combinados vía delta-weighted):**
```
total = 1.0 + (quality-1)×0.321 + (form-1)×0.256 + (matchup-1)×0.192
             + (fatigue-1)×0.128 + (platoon-1)×0.103
→ clamp [0.65, 1.45]
```

**Quality (más complejo):**
```
primary = SIERA ?? xFIP ?? xERA ?? FIP ?? ERA
shrink_w = 350 / (350 + IP×4.3)          # Bayesian hacia LG_ERA
primary_reg = primary×(1-shrink_w) + LG_ERA×shrink_w
if not is_home: primary_reg += 0.15      # away ERA penalty
skill_mult = primary_reg / LG_ERA        # 4.15
woba_mult = 1.0 + (est_woba - 0.320) × 3.0
brl_mult  = 1.0 + max(0, brl_pct - 8.0) × 0.012
kbb_mult  = clamp(1.0 - k_bb_diff × 1.5, [0.88, 1.12])
quality_mult = clamp(skill_mult × woba_mult × brl_mult × kbb_mult, [0.70, 1.35])
```

**Form:** level_adj × trend_adj × qs_adj → clamp [0.85, 1.15]

**Matchup:** ERA vs equipo shrinkada a < 15 IP → clamp [0.85, 1.15]

**Platoon:** lineup_whip / overall_whip → clamp [0.93, 1.07]

**Fatigue:** days_rest gradient + pitch_count penalty → clamp [0.95, 1.12]

### C. Bugs encontrados

**BUG #1 (MEDIO): Pitcher engine tiene poder predictivo casi nulo**
- Pearson(away_pitcher_mult, home_actual_error) = **−0.0065**
- Pearson(home_pitcher_mult, away_actual_error) = **−0.0121**
- Ambas correlaciones son estadísticamente indistinguibles de cero sobre n=5,422 juegos.
- El motor produce señal, pero esa señal no predice el resultado real.

**BUG #2 (MEDIO): Dirección invertida en casos extremos (anti-predicción)**
- Cuando home pitcher es BUENO (away_pitcher mult < 0.95, modelo reduce λ_away):
  → actual_away − lambda_away = **+0.1803** (modelo subestima λ_away en 0.18 runs)
- Cuando home pitcher es MALO (away_pitcher mult > 1.05, modelo aumenta λ_away):
  → actual_away − lambda_away = **+0.0148** (modelo sobreestima λ_away levemente)
- Resultado: el motor sobre-corrige hacia "buenos pitchers dan menos runs", pero la
  realidad no lo confirma empíricamente. El efecto es el opuesto al esperado.

**BUG #3 (MEDIO): LG_XWOBA_ALLOWED=0.320 inconsistente con TTE LG_XWOBA=0.312**
- Mismo bug identificado en Motor #1. El gap de 0.008 entre ofensa (0.312) y pitching (0.320)
  introduce un sesgo sistemático que favorece pitchers sobre bateadores.
- A efectos del Pitcher Engine específicamente: `woba_mult = 1 + (est_woba − 0.320) × 3.0`.
  Si un pitcher tiene xwOBA=0.312 (league avg del TTE), el motor lo penaliza con woba_mult=0.976,
  tratándolo como un pitcher MEJOR que la media cuando en realidad es promedio.

**BUG #4 (MEDIO): Sesgo sistemático upward — modelo predice más runs que la media**
- 59.5% de juegos tienen home_pitcher mult > 1.0 (aumentan λ_away).
- 54.8% de juegos tienen away_pitcher mult > 1.0 (aumentan λ_home).
- Ambos multipliers tienen media > 1.0 (home_pitcher=1.0155, away_pitcher=1.0090).
- Significa que el pitcher promedio en el dataset tiene ERA > LG_ERA (4.15), o que
  el _AWAY_ERA_PENALTY=0.15 introduce sesgo positivo en away pitchers.
- El motor predice sistemáticamente más runs que el promedio de la liga en ambos lados.

**BUG #5 (BAJO): Pesos sub-factores sin justificación empírica**
- `PITCHER_ENGINE_WEIGHTS = {quality:0.321, form:0.256, matchup:0.192, fatigue:0.128, platoon:0.103}`
- Suman 1.0 y parecen calibrados, pero no hay evidencia de que estos pesos minimicen
  error en el backtest.
- Los 5 sub-factores están en config.py pero no están sujetos al gradient descent del pipeline.

**BUG #6 (BAJO): stage_factors naming es semánticamente invertido**
- `stage_factors["home_pitcher"]` = adj_home["total_multiplier"] = HOME pitcher que ajusta λ_away.
- `stage_factors["away_pitcher"]` = adj_away["total_multiplier"] = AWAY pitcher que ajusta λ_home.
- El nombre sugiere "ajuste al home side" pero almacena "ajuste DEL home pitcher".
- Confusión potencial al interpretar análisis de gradient descent.

### D. Áreas oscuras

**ÁREA OSCURA #1: _AWAY_ERA_PENALTY=0.15 — ¿justificado por el backtest?**
- El código cita "Baseball Prospectus, FG" como fuente pero sin enlace específico.
- Empíricamente: away_pitcher mult tiene media 1.0090 y home_pitcher 1.0155. Si el
  penalty fuera el factor dominante, away_pitcher debería ser consistentemente mayor.
  El hecho de que sea MENOR sugiere que away pitchers son en promedio mejores que
  home pitchers en el dataset, o que el penalty de 0.15 ERA ≈ 0.036 mult es absorbido
  por la calidad real de los pitchers.
- No se puede validar sin comparar ERA real de home vs away pitchers en el dataset.

### E. Evidencia empírica

| Métrica | Valor |
|---------|-------|
| home_pitcher mult: mean / std / p5 / p95 | 1.0155 / 0.0680 / 0.9003 / 1.1255 |
| away_pitcher mult: mean / std / p5 / p95 | 1.0090 / 0.0662 / 0.9028 / 1.1214 |
| home_pitcher min / max | 0.8684 / 1.1572 |
| away_pitcher min / max | 0.8674 / 1.1557 |
| Games at clamp [0.65, 1.45] | 0 (clamp nunca activa) |
| Games neutral (mult near 1.0) | 6.5% home / 5.8% away |
| Pearson(home_pitcher, away_error) | **−0.0121** |
| Pearson(away_pitcher, home_error) | **−0.0065** |
| home_pitcher > 1.0 | 59.5% de juegos |
| away_pitcher > 1.0 | 54.8% de juegos |

### F. Acoplamiento

- Lee: lh/la post-Kalman + Team Bias (Motor #2)
- Escribe: lh/la para Bullpen Engine (Motor #4)
- Interacción conocida: defense_away Kalman tenía Pearson r=0.25 con away_pitcher_mult
  (correlación parcial, identificada en análisis Fase 2)
- LG_XWOBA inconsistencia con Motor #1 — riesgo de double-counting en calibración

### G. Magic numbers

- `0.321 / 0.256 / 0.192 / 0.128 / 0.103` (pesos sub-factores en config.py)
- `_K_TBF_ERA = 350` (constante Bayesian de estabilización)
- `4.3` (TBF por IP para starters)
- `_AWAY_ERA_PENALTY = 0.15`
- `3.0` (woba_mult scaling — cada 0.010 xwOBA ≈ 3% runs)
- `0.012` (brl_pct scaling)
- `1.5` (K%-BB% scaling, clamped [0.88, 1.12])
- `0.06` (form level/matchup ERA-to-mult scaling)
- `0.03` (era_trend scaling)

---

## MOTOR #4 — BULLPEN ENGINE

Estado: AUDITADO ✓
Archivo: `modules/baseball_module/context_engine/bullpen_engine.py`

### A. Interface Contract

- **Entradas:** lh, la (float), game_data (bullpen_away/home, pitcher_away/home, team_ids)
- **Salidas:** (lh_new, la_new, metadata) con total_mult ∈ [0.90, 1.10]
- **Orientación (igual que Pitcher Engine):**
  - Away bullpen → ajusta λ_home
  - Home bullpen → ajusta λ_away
- **Posición en pipeline:** PASO 6, después del Pitcher Engine, antes del Contextual

### B. Fórmulas internas

**Tier ERA adjustment (por avg_ips del starter):**
```
if avg_ips < 5.5:  tier_era = era + 0.80 × blend   (long relief, peor ERA)
if avg_ips > 6.5:  tier_era = era - 0.70 × blend   (high leverage, mejor ERA)
else:              tier_era = era                    (average)
blend = linear, 0 a 1 en ventana de 1.0-1.5 IP
```

**Quality composite (4 señales):**
```
era_reg   = regress(tier_era, LG_BP_ERA=4.10, tbf, K=250)
xwoba_reg = regress(xwoba_ag, LG_XWOBA_AG=0.312, savant_pa, K=200)
k_bb_reg  = regress(k_bb, LG_K_BB=0.162, tbf, K=180)
barrel_reg = regress(barrel, LG_BARREL=0.088, attempts, K=200)

quality_raw = xwoba_factor×0.40 + k_bb_factor×0.30 + era_factor×0.20 + barrel_factor×0.10
quality_mult = clamp(quality_raw, [0.75, 1.30])
```

**Innings weighting:**
```
innings_weight = (9.0 - avg_ips) / 9.0          ← clamped avg_ips ∈ [4.0, 8.0]
total_mult = 1.0 + innings_weight × (raw_mult - 1.0)
total_mult = clamp(total_mult, [0.90, 1.10])
```

**Workload fatigue:**
```
delta = ip_3d - 9.0
if delta > 0: workload = min(1.0 + delta×0.012, 1.10)   # tired: +1.2%/IP
else:         workload = max(1.0 + delta×0.005, 0.97)    # rested: -0.5%/IP (cap -3%)
```

### C. Bugs encontrados

**BUG MEDIO #1 — Double counting con Pitcher Engine: Pearson r = 0.41**
- Pearson(home_pitcher_mult, home_bullpen_mult) = **0.4118** — sobre umbral de correlación
- Pearson(away_pitcher_mult, away_bullpen_mult) = **0.3751** — cercano al umbral
- Los equipos con starters de alta ERA también tienen bullpens de alta ERA (correlación de
  calidad de pitching global del equipo). Ambos motores amplifican en la misma dirección
  el mismo signal subyacente: "este equipo tiene pitching malo".
- Efecto concreto: cuando home_pitcher mult = 1.10 (mal starter), home_bullpen también
  será ~1.03-1.06 (mal bullpen). El λ_away recibe doble penalización por la misma causa raíz.

**BUG MEDIO #2 — season = datetime.now().year hardcoded**
- `season = datetime.now().year` en `adjust_for_bullpen()` — línea 337.
- Durante backtest de juegos 2024 y 2025, la función carga datos Savant de 2026.
- El Pitcher Engine y TTE reciben el season correcto por parámetro; el Bullpen Engine no.
- Impacto en backtest: los 2429 juegos de 2024 y 2430 de 2025 usaron stats de bullpen
  del año equivocado. Los resultados del backtest están contaminados para esas temporadas.

**BUG BAJO #3 — Thresholds de tier (5.5, 6.5 IP) son magic numbers**
- `_LONG_RELIEF_IPS_THRESHOLD = 5.5`, `_HIGH_LEVERAGE_IPS_THRESHOLD = 6.5`
- Comentado como "empirical MLB averages" pero sin fuente ni backtest.
- Los ERA delta (0.80 y 0.70) tampoco tienen cita.

**BUG BAJO #4 — Workload asimetría sin justificación empírica**
- Tired: +1.2% por IP sobre la norma (max +10%)
- Rested: −0.5% por IP bajo la norma (max −3%)
- El penalty por sobre-uso es 2.4× más fuerte que el bonus por descanso.
- Sin evidencia empírica de esta asimetría.

**BUG BAJO #5 — Clamp inferior [0.90] nunca activa**
- El workload bonus máximo es −3% → quality_mult mínimo es 0.75 → raw = 0.75×0.97 = 0.728.
- Con innings_weight ≤ 0.556 (avg_ips ≥ 4.0): total_mult ≥ 1 + 0.556×(0.728−1) = 0.849.
- Pero el clamp es [0.90, 1.10], y el mínimo empírico observado es 0.9602.
- La mitad inferior del clamp ([0.90, 0.96)) nunca se alcanza: 0 juegos.
- El clamp inferior es efectivamente inasequible dado los inputs normales.

**BUG BAJO #6 — LG_XWOBA_AG=0.312 crea inconsistencia cross-motor**
- Bullpen Engine usa `_LG_XWOBA_AG = 0.312` (consistente con TTE).
- Pitcher Engine usa `_LG_XWOBA_ALLOWED = 0.320` (diferente).
- Un bullpen con xwOBA=0.316 sería castigado por el Bullpen Engine (+1.3% runs)
  pero un starter con el mismo xwOBA sería considerado mejor que la media por el Pitcher Engine.
- La inconsistencia crea un sesgo sistemático contra bullpens comparado con starters.

### D. Áreas oscuras

**ÁREA OSCURA #1 — Impacto real del tier adjustment en el backtest**
- El tier ERA (long relief vs high leverage) se basa en avg_ips del starter del día.
- avg_ips es promedio de la temporada, no de partidos recientes. Un pitcher que tiene
  avg_ips=6.0 pero está en racha de 8 innings recibirá tier "average" aunque su bullpen
  no haya lanzado mucho.
- ¿El avg_ips estacional es una proxy válida para el dia del juego? No hay análisis.

### E. Evidencia empírica

| Métrica | Valor |
|---------|-------|
| home_bullpen: mean / std | 1.0097 / 0.0251 |
| away_bullpen: mean / std | 1.0095 / 0.0250 |
| Rango empírico (ambos) | [0.960, 1.100] |
| Juegos al clamp máximo (1.10) | **25 home / 25 away (0.5%)** |
| Juegos al clamp mínimo (0.90) | **0** |
| home_bullpen > 1.0 | 59.6% |
| Pearson(home_bullpen, home_error) | **+0.0469** (positivo, débil) |
| Pearson(away_bullpen, away_error) | **+0.0704** (positivo, débil) |
| Pearson(home_pitcher, home_bullpen) | **+0.4118** ← double counting |
| Pearson(away_pitcher, away_bullpen) | **+0.3751** ← near threshold |
| home_bullpen >1.05 → actual_home_err | **+0.342** (dirección CORRECTA) |
| home_bullpen <0.98 → actual_home_err | **−0.258** (dirección CORRECTA) |
| away_bullpen >1.05 → actual_away_err | **+0.669** (dirección CORRECTA) |

### F. Acoplamiento

- Lee: lh/la post-Pitcher Engine (Motor #3); avg_ips del starter del Pitcher Engine
- Escribe: lh/la para Contextual Engine (Motor #8)
- **Doble-counting confirmado**: Pearson > 0.37 con Pitcher Engine en ambas direcciones

### G. Magic numbers

- `_LG_BP_ERA = 4.10`
- `_LG_BP_K_PCT = 0.248`, `_LG_BP_BB_PCT = 0.086`
- `_K_ERA_BP = 250`, `_K_XWOBA_BP = 200`, `_K_K_BB_BP = 180`
- `_NORMAL_IP_3D = 9.0`
- `_LONG_RELIEF_ERA_DELTA = 0.80`, `_HIGH_LEVERAGE_ERA_DELTA = 0.70`
- `_LONG_RELIEF_IPS_THRESHOLD = 5.5`, `_HIGH_LEVERAGE_IPS_THRESHOLD = 6.5`
- `0.40 / 0.30 / 0.20 / 0.10` (pesos composite)
- `1.5` (k_bb_factor scaling)
- `0.012` (workload tired scaling)
- `0.005` (workload rested scaling)

---

## MOTOR #5 — PARK + WEATHER ENGINE

Estado: AUDITADO ✓
Archivo: `modules/baseball_module/hfa/park_weather_engine.py`

### A. Interface Contract

- **Entradas:** lh, la (float), game_data (park.name, weather dict, roof_closed/open, lineup handedness)
- **Salidas:** (lh_new, la_new, metadata) con total_mult = park_factor × weather_mult
- **Diseño declarado:** SIMÉTRICO — mismo multiplicador a λ_home y λ_away
- **Sin season logic:** NO tiene `datetime.now().year`. Park factors son estáticos en el archivo.
- **Posición pipeline:** PASO 2, después de AutoCalibrator

### B. Fórmulas internas

```
park_mult    = STADIUM_DATABASE[park_name].runs_factor   ← estático, 30 parques
weather_mult = temp_mult × wind_mult × rain_mult          → clamp [0.90, 1.12]
total_mult   = park_mult × weather_mult
lh_new = lh × total_mult
la_new = la × total_mult                                  ← IDÉNTICO a lh_new

# Única parte ASIMÉTRICA: wind × handedness
hnd_home = signed_cross × (home_lhb - 0.45) × speed × 0.040  → clamp [−0.015, +0.015]
hnd_away = signed_cross × (away_lhb - 0.45) × speed × 0.040
lh_new *= (1 + hnd_home)
la_new *= (1 + hnd_away)
```

**Temp:** +0.5% per 5°F sobre 72°F → clamp [0.94, 1.06]
**Wind:** out=+2.0%/5mph, in=−1.5%/5mph, cross=+0.3%/5mph → clamp [0.94, 1.10]
**Rain:** drizzle=0.99, light=0.97, moderate=0.95 (probability-weighted)
**Roof default:** `has_roof=True AND NOT roof_open` → weather_mult=1.0

### C. Bugs encontrados

**BUG CRÍTICO #1 — Weather component completamente inerte en producción**
- Evidencia: `home_park std = 0.000000` en TODOS los 12 valores únicos de parque.
- Solo hay 12 valores posibles (los runs_factor del STADIUM_DATABASE). Dentro de cada parque,
  el multiplicador es EXACTAMENTE igual en todos los juegos de ese parque.
- Conclusión: `weather_mult = 1.0` en el 100% de los juegos del backtest.
- Causa probable: (a) `game_data.get("weather", {})` siempre devuelve {} (sin datos de weather);
  o (b) los parques con techo (`has_roof=True`) son tratados como closed y suprimen weather.
- El motor se llama "Park + WEATHER Engine" pero solo aplica factor de parque.
- **La mitad del motor es decorativa.**

**BUG MEDIO #2 — Overconfidence en park factor vs runs reales**
- `Pearson(home_park, predicted_total_runs) = 0.4891` — el modelo es muy confiado.
- `Pearson(home_park, actual_total_runs) = 0.0998` — la realidad tiene correlación ~5× menor.
- El modelo predice que Coors (1.13) tendrá 13% más runs que un parque neutral, pero los datos
  muestran solo ~10% de correlación real entre park factor y runs actuales.
- Consecuencia: el park factor amplifica los λs más de lo que los datos justifican.
- **Caso extremo Coors:** park=[1.07,1.14] → mean_total_runs=11.12, mean_err=+0.72
  El modelo aún subestima Coors en 0.72 runs/juego después del factor 1.13.

**BUG BAJO #3 — Park factors estáticos de 2024, no actualizados**
- STADIUM_DATABASE hardcodeado en el archivo con "FanGraphs 2024 five-year weighted" factors.
- Usado para 2024, 2025 y 2026 sin actualización.
- `Sutter Health Park` (A's, desde 2025) tiene factor 1.00 por defecto — sin datos reales.

**BUG BAJO #4 — Roof default-to-closed probablemente causa Bug #1**
- `if stadium.has_roof and not game_data.get("roof_open", False): roof_closed = True`
- 8 parques tienen `has_roof=True`: Tropicana, Rogers Centre, Minute Maid, Globe Life,
  T-Mobile, American Family, loanDepot, Chase Field.
- En ausencia de `roof_open` explícito en game_data, TODOS se tratan como closed.
- American Family Field, Chase Field, Globe Life Field frecuentemente juegan con techo abierto.
- Si `roof_open` nunca llega en game_data, el sistema siempre asume techo cerrado.

**BUG BAJO #5 — STADIUM_DATABASE keyed por nombre de estadio**
- Si game_data provee nombre distinto (typo, abreviación, nombre viejo), factor = 1.00.
- No hay lookup alternativo por team_id o park_id de MLB API.

### D. Área oscura

**ÁREA OSCURA #1 — Wind × handedness: ¿alguna vez activa en producción?**
- Requiere: wind_speed_mph ≥ 10 + roof open + `home_lhb_pct`/`away_lhb_pct` en game_data.
- Si handedness defaults a `_AVG_LHB_PCT = 0.45` para ambos: delta = 0 → hnd = 0.
- Max efecto si activa: ±1.5% por lineup (±3.0% diferencial).
- En el backtest, stage_factors "home_park" == "away_park" en 100% de juegos → handedness
  tampoco está activando (porque store stage_factors guarda el total_mult pre-handedness, o
  handedness es siempre 0.0).
- **Imposible confirmar sin agregar wind_handedness al stage_factors log.**

### E. Evidencia empírica

| Métrica | Valor |
|---------|-------|
| Valores únicos de home_park | **12** (igual que STADIUM_DATABASE entries) |
| Std dentro de cada valor de parque | **0.000000** (weather 100% inerte) |
| home_park == away_park | **100% de 5,422 juegos** |
| Pearson(home_park, home_actual_error) | −0.0065 |
| Pearson(home_park, away_actual_error) | +0.0252 |
| Pearson(home_park, total_runs_error) | +0.0138 |
| Pearson(home_park, actual_total_runs) | **+0.0998** |
| Pearson(home_park, predicted_total)  | **+0.4891** ← model overconfident 5× |
| Coors Field: mean_err total runs | +0.72 (subestima incluso con factor 1.13) |

### F. Acoplamiento

- Lee: game_data.park.name, game_data.weather (siempre vacío en la práctica)
- Escribe: lh/la × park_mult (weather inerte)
- **Importante:** el park factor amplifica los λs ANTES de HFA, Pitcher, Bullpen.
  Si park=1.10, el HFA engine opera sobre λs ya inflados. Esto amplifica todos los
  efectos downstream en proporción al park factor — potencial de over-amplification en Coors.

### G. Magic numbers

- `_NEUTRAL_TEMP_F = 72.0`
- `_TEMP_RATE = 0.005` ("FanGraphs / Codify research" sin URL)
- Wind rates: `0.020 (out), 0.015 (in), 0.003 (cross)` per 5 mph unit
- Wind thresholds: `60° (out zone), 120° (in zone)`
- `_HANDEDNESS_WIND_SCALE = 0.040`
- `_HANDEDNESS_WIND_MIN_MPH = 10.0`
- `_AVG_LHB_PCT = 0.45`
- Rain tiers: `1.0mm, 5.0mm, 10.0mm`; multipliers: `0.99, 0.97, 0.95`

---

## MOTOR #6 — DEFENSIVE EFFICIENCY ENGINE

Estado: AUDITADO ✓
Archivo: `modules/baseball_module/context_engine/defensive_efficiency_engine.py`

### A. Interface Contract

- **Entradas requeridas:** `game_data['defense_home']` y `game_data['defense_away']` — dicts con
  claves `der` (float), `bip` (int), `oaa` (float, opcional)
- **Salidas:** (lh_new, la_new, metadata) con multiplicador ∈ [0.95, 1.05]
- **Orientación:** home defence → λ_away; away defence → λ_home (idéntico a Bullpen Engine)
- **Señal de datos:** DER (Defensive Efficiency Ratio = 1−BABIP_allowed) + OAA (Outs Above Average)

### B. Fórmulas internas

```
# DER factor (fielding puro):
shrink_w = 500 / (500 + bip)                    # Bayesian hacia LG_DER=0.715
der_reg  = der × (1-shrink_w) + 0.715 × shrink_w
der_factor = 0.715 / der_reg                     # <1 = buen campo → menos runs

# OAA factor (opcional, rango puro):
oaa_run_pg = oaa × 0.80 / 162                   # runs_saved per game
oaa_factor  = clamp(1 - oaa_run_pg / LEAGUE_AVG_RUNS, [0.92, 1.08])

# Combinación:
if oaa available: raw = der_factor×0.55 + oaa_factor×0.45
else:             raw = der_factor
mult = clamp(raw, [0.95, 1.05])
```

### C. Bugs encontrados

**BUG CRÍTICO #1 — Motor completamente muerto: datos nunca llegan a game_data**
- Evidencia: `home_defense = away_defense = 1.000000` en 5,422/5,422 juegos (100%).
- Causa raíz: ningún fetcher en `data_fetchers.py` ni en el pipeline popula
  `game_data['defense_home']` o `game_data['defense_away']` con datos reales.
- grep en todo el codebase: solo `run_module.py` (que los lee) y
  `defensive_efficiency_engine.py` (que los consume) referencian estas claves.
  `data_fetchers.py` no tiene ninguna referencia a estas claves.
- Bypass explícito en run_module.py línea 628:
  `if _def_home or _def_away:` → siempre False → engine nunca ejecuta.
- **El motor fue diseñado y escrito completamente (código correcto, fórmulas válidas)
  pero la fuente de datos nunca fue implementada.**

**BUG BAJO #2 — `calculate_der()` en el mismo archivo es dead code**
- La función `calculate_der(hits, at_bats, strikeouts, home_runs, sac_flies)` está
  implementada en el mismo archivo (línea 212) pero no es importada por ningún otro módulo.
- No aparece en data_fetchers.py, enrichment, ni backtest.
- Fue diseñada para ser usada en el enriquecimiento de datos — ese paso nunca se construyó.

**BUG BAJO #3 — Naming collision conceptual con Kalman "defense_home"**
- Kalman usa el string `"defense_home"` como contexto (clave en kalman_state DB).
- DEE usa `game_data['defense_home']` como dict de datos de campo (DER/OAA).
- Mismo término, dos significados completamente distintos.
- No hay colisión en ejecución (son namespaces distintos), pero confunde el análisis.

### D. Área oscura

**ÁREA OSCURA #1 — ¿Triple counting si DEE se activara?**
Si el motor se activara con datos reales, habría correlación parcial con:
- **Pitcher Engine:** K-heavy pitchers tienen menor BABIP → mayor DER → DEE les da crédito.
  Overlap con era_factor y kbb_factor del Pitcher Engine.
- **Kalman defense_home:** rastrea runs_allowed históricos (que incluyen fielding + pitching).
  Overlap con DER, que también correlaciona con runs_allowed.

Las señales NO son triples-counted exactamente (son ortogonales en teoría), pero tienen
correlación parcial no-nula. El impacto real requeriría correr backtest con DEE activo.

### E. Evidencia empírica

| Métrica | Valor |
|---------|-------|
| home_defense == 1.0 exactamente | **5422/5422 (100%)** |
| away_defense == 1.0 exactamente | **5422/5422 (100%)** |
| std dentro de todos los juegos | **0.000000** |
| Valores únicos | **1** (solo: 1.000000) |
| Referencias a defense_home en data_fetchers.py | **0** |

### F. Acoplamiento

- Motor correctamente posicionado en pipeline (PASO 6, después de HFA, antes de Pitcher)
- No acopla con nada porque NUNCA SE EJECUTA
- Kalman defense_home (Motor #2) es la ÚNICA señal defensiva que actualmente llega al pipeline
- La diferencia: Kalman defense = histórico de runs allowed; DEE = fielding puro (DER/OAA)
  Son señales ortogonales que deberían complementarse, no redundantes.

### G. Magic numbers

- `_LG_DER = 0.715` (citado como "2024 MLB average")
- `_LG_OAA_RUN = 0.80` ("Dewan/StatCast research" sin cita)
- `_K_BIP_DER = 500` (citado como "estabiliza ≈500 BIP" — cercano a investigación real)
- `_DER_WEIGHT = 0.55`, `_OAA_WEIGHT = 0.45` (sin justificación)
- `_MAX_DEF_ADJ = 0.05` (sin justificación)

---

## MOTOR #7 — HFA ENGINE

Estado: AUDITADO ✓
Archivo: `modules/baseball_module/hfa/hfa_engine.py`

### A. Interface Contract

- **Entradas:** lh, la (float), game_data (park.name, miles_traveled_away, time_zones_crossed_away)
- **Salidas:** (lh_new, la_new, metadata) — dos ajustes asimétricos independientes
- **Componente 1 (ACTIVO):** Crowd boost → sube λ_home únicamente
- **Componente 2 (MUERTO):** Travel fatigue → debería bajar λ_away, nunca lo hace
- **Posición pipeline:** PASO 7 (después de Park+Weather, DEE, antes de Pitcher Engine)

### B. Fórmulas internas

**Crowd boost (activo):**
```
hfa_boost = hfa_base[park_name]   ← hardcoded por parque, rango [0.0250, 0.0450]
hfa_mult  = 1.0 + hfa_boost / LEAGUE_AVG_RUNS   (= 1.0 + boost / 4.5)
lh_new    = lh × hfa_mult
la_new    = la   (sin cambio)
```

**Travel fatigue (dead):**
```
time_zones = game_data.get("time_zones_crossed_away", 0)   ← siempre 0
miles      = game_data.get("miles_traveled_away", 0)        ← siempre 0
penalty = 0.0   ← siempre, porque time_zones=0 y miles=0 fallan todas las condiciones
la_new *= 1.0 - 0.0 / 4.5   = la_new × 1.0   (sin cambio)
```

### C. Bugs encontrados

**BUG CRÍTICO #1 — Componente travel fatigue muerto: datos hardcodeados a 0**

En `run_module.py` líneas 263-264:
```python
game_data.setdefault('miles_traveled_away', 0)        # ← SIEMPRE 0
game_data.setdefault('time_zones_crossed_away', 0)    # ← SIEMPRE 0
```

Ningún fetcher del pipeline calcula o asigna millas o timezone delta para los partidos.
El `setdefault` garantiza que ambas claves existan con valor 0 antes de que el HFA Engine
las lea. La función `_calculate_travel_fatigue()` recibe miles=0, time_zones=0, produce
penalty=0.0, y la_new permanece inalterada.

Resultado: `away_hfa = 1.000000` en 5,422/5,422 juegos (std=0.0, unique=1).

El propio docstring lo admite: "In practice travel fields are rarely populated by the
free MLB API; the penalty fires mainly when game_data is enriched externally."

**BUG BAJO #2 — home_hfa signal estadísticamente indetectable**
- home_hfa varía entre 1.0056 y 1.0100 (rango = 0.0044, sólo 0.44% de λ_home)
- Pearson(home_hfa, home_actual_error) = **−0.0049** ≈ 0
- La crowd boost más alta (Yankee: 0.045 runs) añade 1.0% a λ_home.
- La diferencia entre el parque más hostil (Yankee 1.0100) y el más neutral
  (Tropicana 1.0056) es 0.44% de λ — por debajo del umbral detectable en el backtest.

**BUG BAJO #3 — hfa_boost per-park son magic numbers sin validación**
- 32 entradas hardcodeadas por parque en `self.hfa_base`, rango [0.0250, 0.0450].
- Comentario: "Empirically recalibrated to match +0.034 run/game home scoring advantage."
  Pero la recalibración es global (÷4.0 del valor original) — no hay evidencia de que
  los valores RELATIVOS entre parques sean correctos.
- Yankee (0.045) tiene 1.8× el crowd boost de Tropicana (0.025). ¿Es eso real o subjetivo?

**BUG BAJO #4 — Gradient descent ciego a travel component**
- `away_hfa` en stage_factors = `la_hfa / la_pre = 1.0` siempre.
- Gradient descent nunca puede aprender el peso correcto del componente travel.
- Si se activa el travel pipeline, el gradient descent podría aprender, pero actualmente
  también está roto (Bug #1 Motor #2). Problema compuesto.

### D. Área oscura

**ÁREA OSCURA #1 — ¿Los hfa_boost relativos entre parques tienen soporte empírico?**
- Los 32 valores distintos fueron reducidos ÷4.0 de estimaciones previas para calibrar
  el promedio a +0.034 runs. Pero la distribución relativa (cuánto más hostil es Yankee
  vs Tropicana) es completamente subjetiva.
- Sin backtest por parque desagregado, es imposible saber si los pesos relativos ayudan
  o añaden ruido. El Pearson ≈ 0 sugiere que la variación entre parques no produce señal.

### E. Evidencia empírica

| Métrica | Componente | Valor |
|---------|-----------|-------|
| home_hfa unique values | Crowd (activo) | **9** (por parque, estático) |
| home_hfa rango | Crowd | [1.0056, 1.0100] |
| home_hfa std | Crowd | 0.001076 |
| Pearson(home_hfa, home_error) | Crowd | **−0.0049** ≈ 0 |
| away_hfa unique values | Travel (muerto) | **1** (siempre 1.0) |
| away_hfa std | Travel | **0.000000** |
| away_hfa = 1.0 exactly | Travel | **5422/5422 (100%)** |

### F. Acoplamiento

- Lee: park name (disponible siempre), miles/time_zones (siempre 0)
- Escribe: lh × crowd_mult (activo), la inalterada (travel muerto)
- La crowd boost amplifica los λ_home antes de que Pitcher/Bullpen operen sobre ellos.
  Al igual que park_factor, la magnitud (≤1%) es demasiado pequeña para observarse.

### G. Magic numbers

- 32 valores `hfa_boost` por parque: [0.0250 – 0.0450]
- `_default_hfa = 0.0325` (parques no encontrados)
- Travel thresholds: `time_zones ≥3 → 0.06`, `≥2 → 0.04`, `≥1 → 0.02`
- Distance thresholds: `miles > 2000 → 0.05`, `miles > 1000 → 0.03`
- Travel cap: `0.10` runs máximo

---

## MOTOR #8 — CONTEXTUAL ENGINE

Estado: AUDITADO ✓
Archivo: `modules/baseball_module/context_engine/contextual_engine.py`

### A. Interface Contract

- **Entradas:** lh, la (float), game_data (home_team.rest_days, away_team.rest_days,
  back_to_back_home/away, umpire_stats, hp_umpire_name)
- **Salidas:** (lh_new, la_new, metadata) — ajustes asimétricos (rest) + simétrico (umpire)
- **Componentes declarados:**
  1. Rest/B2B — asimétrico, por equipo
  2. Umpire zone factor — simétrico, afecta ambos λ igual
- **Posición pipeline real:** PASO 3 en run_module.py (antes de Park+Weather, HFA, Pitcher, Bullpen)
- **Posición pipeline en docstring:** "PASO 7 (after Bullpen, before Monte Carlo)" ← INCORRECTO

### B. Fórmulas internas

**Rest (asimétrico, por equipo):**
```
rest_days = team.get("rest_days")    # default = 1 (vía setdefault)
b2b_flag  = game_data.get("back_to_back_X", False)  # siempre False

if rest == 0:   mult = 0.960   # B2B:  −4%
if rest >= 3:   mult = 0.980   # Rust: −2%
else:           mult = 1.000   # Optimal (1-2 days)
```

**Umpire (simétrico):**
```
games_worked = umpire_stats.get("games_worked")
if games_worked < 4: factor = 1.0   # skip — muestra pequeña
factor = clamp(zone_factor, [0.96, 1.04])
lh_new *= factor; la_new *= factor   # idéntico a home y away
```

### C. Bugs encontrados

**BUG MEDIO #1 — Motor funcionalmente muerto: 99.74% games = 1.0**
- home_context = 1.000000 en 5,443/5,451 juegos (99.85%)
- away_context = 1.000000 en 5,440/5,451 juegos (99.80%)
- Causa combinada: B2B muerto + rust raro + umpire raro = 14 juegos activos en 5,451

| Componente | Games activos | % del total |
|-----------|--------------|------------|
| B2B (mult=0.960) | **0** | 0.00% |
| Rust (mult=0.980) | **11** | 0.20% |
| Umpire (factor≠1) | **4** | 0.07% |

**BUG MEDIO #2 — B2B completamente muerto: tres defaults superpuestos lo deshabilitan**

Cadena de cortocircuitos:
1. `game_data.setdefault('back_to_back_away', False)` → b2b_flag = False siempre
2. `rest_days` en team dict viene de `game_data.get('away_days_rest', 1)` → 1 (optimal) por default
3. Ningún fetcher en data_fetchers.py popula `back_to_back_home/away` ni calcula días de descanso ≤ 0

El engine tiene código correcto para detectar B2B, pero las tres vías de entrada
están bloqueadas simultáneamente. Un equipo que juega back-to-back en realidad
NUNCA recibe el −4% de penalización.

**BUG BAJO #3 — Docstring declara posición errónea en el pipeline**
- Docstring: "Pipeline position: PASO 7 (after Bullpen, before Monte Carlo)"
- Realidad: PASO 3 en run_module.py — ANTES de Park+Weather, HFA, Pitcher y Bullpen
- El comentario en run_module.py explica la razón: "Posición intencional: ANTES del
  Bullpen Engine. El F5 snapshot se toma aquí: incluye pitcher + rest/umpire (ambos
  aplican al F5) y excluye bullpen (starters lanzan F5)."
- La posición es INTENCIONAL y matemáticamente correcta (multiplicación es conmutativa),
  pero la documentación en el engine file nunca se actualizó.

**BUG BAJO #4 — Umpire data activa en apenas 4 games, sin validación de dirección**
- 4 juegos con factor umpire ≠ 1.0 (0.9954, 1.0021, 1.0039, 1.0078)
- n=4 es demasiado pequeño para cualquier análisis estadístico.
- No hay evidencia de que el zone_factor de umpire prediga runs reales en este sistema.

### D. Diferencia clave con motores muertos anteriores

| Motor | Categoría | Causa raíz |
|-------|----------|-----------|
| DEE (Motor #6) | **Completamente muerto** | Data pipeline nunca construido |
| Weather (Motor #5) | **Completamente muerto** | game_data.weather siempre {} |
| HFA Travel (Motor #7) | **Completamente muerto** | setdefault a 0 |
| Contextual (Motor #8) | **Funcionalmente muerto** | Defaults bloquean B2B; rust y umpire rarísimos |

El Contextual Engine no está "muerto por datos faltantes" — tiene acceso a `rest_days`
ocasionalmente (11 juegos de rust real). Está casi muerto porque:
1. El caso principal (B2B) está explícitamente bloqueado por 3 defaults
2. Los casos que sí activan (rust, umpire) son estadísticamente raros

### E. Evidencia empírica

| Métrica | Valor |
|---------|-------|
| home_context = 1.0 exactamente | 5443/5451 (99.85%) |
| away_context = 1.0 exactamente | 5440/5451 (99.80%) |
| Valores únicos (home_context) | 6 |
| Rango de valores | [0.9800, 1.0078] |
| B2B (mult=0.960) games | **0** |
| Rust (mult=0.980) games | 11 (0.20%) |
| Umpire (factor≠1.0) games | 4 (0.07%) |
| Pearson(home_context, home_err) | **0.0000** (n=14 insuficiente) |

### F. Acoplamiento

- B2B tiene overlap potencial con Pitcher Engine (días de descanso del starter ya considerados).
  El docstring del HFA Engine lo nota: "Back-to-back is intentionally excluded [del HFA] —
  owned by ContextualEngine." Coordinación correcta en diseño, pero muerta en práctica.
- Umpire factor SIMÉTRICO no afecta win probability (igual razón que park factor simétrico).
  Solo afecta expected totals.

### G. Magic numbers

- `_B2B_MULT = 0.960` ("empirical MLB: ~3–5%", rango citado pero no cita específica)
- `_RUST_MULT = 0.980` (ninguna cita)
- `_UMP_CLIP_LOW / HIGH = 0.960 / 1.040` (ninguna cita)
- `_UMP_MIN_GAMES = 4` (ninguna cita)

---

## MOTOR #9 — MONTE CARLO SIMULATOR

Estado: AUDITADO ✓
Archivo: `modules/baseball_module/montecarlo/simulator.py`

### A. Interface Contract

- **Entradas:** `lh`, `la` (λ_home/away finales del pipeline), `total_line` (opcional),
  `lambda_noise=0.05`, `rho_game=-0.06`, `early_stop_se=0.0005`, `n_max=5_000_000`
- **Salidas:** `p_home`, `p_away`, `p_rl_home/away`, `p_over/under`, `mean_total`, `std_total`,
  `converged_early`, `f5_home/away/draw` (si `analyze_f5=True`), percentiles de distribución
- **Posición en pipeline:** PASO 8 — recibe λ calibradas y produce probabilidades puras (pre-Platt)

### B. Fórmulas internas

```
# Bivariate Cholesky decomposition (epistemic noise)
_sigma_h = lambda_noise * max(lh, 0.5)
_sigma_a = lambda_noise * max(la, 0.5)
_rho_sqrt_comp = sqrt(1 - rho_game²)

z1 = Normal(0,1)  # shared shock
z2 = Normal(0,1)  # independent residual

lh_noise = clip(lh + sigma_h * z1, MIN_LAMBDA, MAX_LAMBDA)
la_noise = clip(la + sigma_a * (rho_game * z1 + rho_sqrt_comp * z2), MIN_LAMBDA, MAX_LAMBDA)

home_runs ~ Poisson(lh_noise)
away_runs ~ Poisson(la_noise)

# Win probability (ties split 50/50)
p_home = (wins_home + 0.5 * ties) / sims_done

# Early stopping (minimum 500K warm-up)
SE = sqrt(p_home * (1 - p_home) / sims_done)
break if sims_done >= 500_000 and SE < 0.0005

# F5 scale (if actual F5 lambdas not provided)
F5_SCALE = 0.575
lh_f5_noise = clip(lh_noise * F5_SCALE, MIN, MAX)
```

### C. Bugs encontrados

**BUG #1 (MEDIO): `lambda_noise=0.05` hardcodeado — NO adaptativo**
- El usuario preguntó si el ruido epistémico se ajusta dinámicamente (0.04–0.08 según calidad de datos).
- La respuesta: **NO**. Es un parámetro fijo que se pasa desde `run_module.py` como valor por defecto.
- No hay lógica de `if data_quality == 'sparse': noise = 0.08`. El mismo 5% se aplica a un pitcher
  con 200 IP como a uno con 5 IP en su rookie year.
- **Impacto:** El modelo tiene la misma confianza en estimaciones bien respaldadas y en estimaciones
  con escasez de datos. La incertidumbre epistémica no refleja la realidad.

**BUG #2 (CRÍTICO): `rho_game=-0.06` no derivado del sistema — 8× más negativo que realidad**
- El docstring afirma "matches empirical MLB data" sin fuente.
- Medición empírica en 5,422 juegos del backtest:
  - `rho(actual_home_runs, actual_away_runs) = -0.0078`
  - `rho asumido por el modelo = -0.06`
  - **Discrepancia: 7.7×** — el modelo asume correlación negativa 8 veces más fuerte que la real.
- Consecuencia matemática: la varianza del total (`Var(home+away)`) se comprime artificialmente.
  `Var(total) = Var(home) + Var(away) + 2*Cov(home,away)`
  Con rho=-0.06 vs rho=-0.008: el total tiene ~5% menos varianza de la correcta.
- Efecto secundario: probabilidades de over/under ligeramente sesgadas; impacto en win prob es marginal
  (la win prob depende principalmente de las marginales, no de la correlación conjunta).

**BUG #3 (MEDIO): `F5_SCALE=0.575` — magic number sin validación empírica**
- Comentario en código: "55-58%; 57.5% is the calibrated midpoint" — calibrado a mano, no de datos.
- El sistema tiene en su BD los resultados de F5 innings reales (para los juegos donde aplica).
- La relación real entre λ_full_game y λ_first_5 puede variar por:
  - Pitcher quality (starters vs bullpen ERA gap)
  - Season phase (starters van más innings en septiembre)
  - Team strategy (bullpen teams vs traditional rotations)
- **Área oscura:** No se puede verificar la bondad de 0.575 sin datos F5 explícitos en la BD.
  La BD solo almacena `lambda_home` / `lambda_away` (full game), no las versiones F5.

**BUG #4 (BAJO): 2024 sin calibración Platt — p_home = p_home_raw para 2,429 juegos**
- El pipeline usa `_platt(p_mc, a, b)` donde `a,b` se obtienen de `_learning.get_platt_params(season)`.
- Para season=2024, el log muestra: "no prior-season params (n=0) — using identity defaults".
- Verificación empírica: `p_home == p_home_raw` en 100% de los 2,429 juegos de 2024 (diff máx = 0.0).
- Toda la temporada 2024 opera con Platt identidad → sin corrección de calibración.
- Implicación: el backtest de 2024 mide el modelo SIN ningún ajuste de calibración aprendida.
  Los resultados de 2024 son los más "puros" pero también los menos representativos de producción.

### D. Lo que funciona correctamente

**CORRECTO #1: Cholesky decomposition matemáticamente exacta**
- `la_noise = la + sigma_a * (rho_game * z1 + sqrt(1-rho²) * z2)` — implementación correcta.
- `Cov(lh_noise, la_noise) = sigma_h * sigma_a * rho_game` ✓ (exacto por construcción)
- Marginals preservadas: cada noise term tiene varianza sigma² exactamente. ✓

**CORRECTO #2: Early stopping funciona y ahorra ~80% del cómputo**
- Para p≈0.50: SE < 0.0005 requiere n > 1,000,000 sims → dispara en bloque 5 de 25 máximos.
- Para p≈0.70: SE < 0.0005 requiere n > 840,000 → dispara en bloque 5 (sims=1M).
- **Ahorro efectivo: ~80% del cómputo** vs correr las 5M sims completas.
- Verificación indirecta: `p_home_raw` tiene 93.9% valores únicos a 5dp → consistente con ~1M sims/juego.
- El comentario en código documenta la fix de un bug anterior (modulo silencioso).

**CORRECTO #3: Ties split 50/50 es estadísticamente correcto**
- `p_home = (wins_home + 0.5 * ties) / sims` — en MLB los juegos van a extra innings.
- El modelo no diferencia ML vs extra innings pero la aproximación 50/50 es neutral.

**CORRECTO #4: Platt scaling preserva estructura aprendida**
- La decisión de aplicar Platt solo a p_home (p_away = 1 - p_home_cal) es intencional y documentada.
- Preserva el intercept `b` que codifica HFA estructural que Poisson no puede capturar.
- Platt 2025: a=0.7277, b=0.1459 → comprensión + sesgo upward (+0.009 promedio).

### E. Análisis empírico

**Distribución `p_home_raw` (5,422 juegos, 3 seasons):**

```
min=0.1551   mean=0.5096   max=0.8398   std=0.1093
```

- Rango [0.155, 0.840] — bien distribuido, no sobre-concentrado
- 18.0% de juegos con p_home_raw < 0.35 o > 0.65 (tails existen)
- 93.9% únicos a 5dp — consistente con early stopping a ~1M sims

**Correlaciones predictivas:**

| Señal | Pearson(·, home_won) |
|-------|----------------------|
| p_home_raw (pre-Platt) | **0.1604** |
| p_home (post-Platt) | **0.1612** |
| market_prob_home (Pinnacle) | (proxy estimado ~0.19) |

- Platt mejora Pearson solo marginalmente (+0.0008).
- Pearson=0.16 es consistente con el Brier=0.243 observado.
- `Pearson(p_home_raw, lambda_home) = 0.711` — el simulador transforma λ correctamente.
- `Pearson(p_home_raw, market_prob_home) = 0.7613` — alta correlación con Pinnacle,
  pero **NO** por anchoring. Ambos sistemas responden al mismo conjunto de factores del juego.

**Efecto Platt por temporada:**

| Season | p_home == p_home_raw | Interpretación |
|--------|---------------------|----------------|
| 2024 | 100.0% (2,429/2,429) | Identity Platt — sin calibración |
| 2025 | 0.0% (todos diferentes) | Platt a=0.727, b=0.146 — calibrado |
| 2026 | 0.0% (todos diferentes) | Platt a=0.567, b=0.099 — recalibrado |

- Mean Platt shift = +0.0092 (upward hacia home team)
- Max Platt shift = 0.097 (casi 10 puntos porcentuales en casos extremos)
- 43.2% upward, 12.2% downward, 44.6% sin cambio (2024)

**Validación rho_game:**

| | Valor |
|--|-------|
| rho asumido por modelo | -0.060 |
| rho empírico (5,422 juegos) | **-0.0078** |
| Discrepancia | 7.7× |

El modelo sobre-estima la correlación negativa entre carreras de equipos locales y visitantes.

**Pinnacle anchor:**
- **NO existe Pinnacle anchor en el simulador.** La señal pura de Monte Carlo es independiente
  de las odds de Pinnacle. El blend con mercado (si existe) no está en este archivo.
- 14.0% de juegos (757/5,422) sin datos de Pinnacle — todos tratados con mismo código path.

### F. Interacciones con otros motores

- Recibe λ ya modificadas por todos los motores #1-#8.
- Ruido epistémico `lambda_noise=0.05` añade ~0.21 unidades de std adicional a λ≈4.3.
  Esta incertidumbre aplana las probabilidades extremas — reduce over-confidence.
- p_home_raw es el input de Platt (Motor #2's recalibrate_platt), que aprende sobre estas
  probabilidades sin-anclar para producir p_home final.
- Correlación con motores muertos: dado que 4-5 motores no aportan señal, las λ que recibe
  el simulador son esencialmente función de TTE + Pitcher (parcial) + Bullpen (contaminado).
  El simulador amplifica fielmente lo bueno y lo malo de las λ previas.

### G. Resumen de severidad

| # | Severidad | Descripción |
|---|-----------|-------------|
| BUG #1 | MEDIO | lambda_noise hardcodeado a 0.05, no adaptativo a calidad de datos |
| BUG #2 | CRÍTICO | rho_game=-0.06 vs empírico -0.008; over-estima correlación 8× |
| BUG #3 | MEDIO | F5_SCALE=0.575 sin validación empírica |
| BUG #4 | BAJO | 2024 sin Platt — calibración aprendida no aplica al primer año |

**Diagnóstico global:** Motor #9 es el mejor implementado del pipeline. El núcleo matemático
(Cholesky, Poisson, early stopping) es correcto. Los problemas son paramétricos, no algorítmicos.
El simulador es un buen transductor de λ→probabilidad; la calidad del output depende casi
enteramente de la calidad de las λ upstream (donde están los problemas reales).

---

## MOTOR #10 — VALUE DETECTOR

Estado: PENDIENTE

---

## CROSS-CUTTING ISSUES

*(Sección viva — se actualiza con cada motor auditado)*

---

### HIPÓTESIS H1 — CADENA CAUSAL PRINCIPAL (LG_XWOBA → sobre-confianza → bucket <40%)

**Estado:** Activa. Verificar al terminar todos los motores.

**Cadena causal documentada:**

```
LG_XWOBA inconsistente (TTE=0.312 vs Pitcher=0.320)
        ↓
woba_mult sesgado upward en Pitcher Engine
  · pitcher con xwOBA=0.312 (media TTE) → woba_mult=0.976 (Pitcher lo trata como bueno)
  · pitcher con xwOBA=0.320 (media Pitcher) → woba_mult=1.000 (neutral)
  · Gap de 0.008 × 3.0 = 0.024 de sesgo en todo pitcher near-league-avg
        ↓
Pitcher Engine sobre-castiga calidad en extremos
  · Cuando identifica pitcher "bueno" (mult <0.95): reduce λ más de lo justificado
  · Pearson r ≈ 0 con runs reales — señal no predictiva
  · home_pitcher: 59.5% de mults > 1.0, mean=1.016 (sesgo upward sistemático)
        ↓
Bullpen Engine amplifica en la misma dirección (Pearson r=0.41 con Pitcher)
  · Doble penalización del mismo signal subyacente: "equipo con pitching malo"
  · Pero el Bullpen Engine SÍ tiene señal válida (Pearson r=+0.07 con errores reales)
        ↓
λ en extremos (juegos muy desiguales) sobre-estimados hacia un lado
  · El sistema es excesivamente confiado en matchups asimétricos
  · El bucket <40% tiene predicted=35.4% pero actual=38.4% (+3.0pp)
  · La sobre-confianza en los extremos es la causa del error sistemático
```

**Hipótesis de fix:**
Si se unifica LG_XWOBA a un valor único (0.315 o derivado de datos reales), y se re-calibra
el Pitcher Engine, la sobre-confianza en extremos debería reducirse sin Platt 2D.

**Nota:** "Si H1 es correcta, los fixes en cadena podrían reducir la sobre-confianza
estructural y eliminar el comportamiento errático del bucket <40% sin necesidad de Platt 2D."

---

### Issues pre-identificados (evidencia parcial)

1. **Triple shrinkage TTE→Kalman→Bias**: TTE shrinkea 85-90% hacia prior, Kalman blend 35% más,
   Bias dampening 65%. El dato real tiene influencia mínima.

2. **Gradient descent roto**: 5,422 juegos, 0 aprendizaje. Todo el sistema de pesos del pipeline
   es decorativo. Los fixes individuales de cada motor no serán descubiertos por el sistema de
   aprendizaje.

3. **LG_XWOBA inconsistente entre TTE (0.312) y Pitcher Engine (0.320)**: Sesgo sistemático
   cruzando dos motores. Núcleo de H1.

4. **HFA y Context aparentemente muertos** (stage_factors std=0.0000 en ambos): Si confirmado
   en audits, 2 de 6 stage_factors del gradient descent son constantes → otra razón por la que
   los pesos no aprenden.

5. **Double counting Pitcher + Bullpen** (Pearson r=0.41): ambos motores capturan el mismo
   signal de calidad de pitching del equipo. Amplificación artificial en matchups desiguales.

---

### OBSERVACIÓN O1 — CAPACIDAD LATENTE NO ACTIVADA

**Patrón identificado:** motores correctamente implementados (código válido, fórmulas correctas,
cierta justificación documentada) están inertes porque el data pipeline no les suministra inputs.
No son errores de diseño — son cables sin conectar.

**Motores en esta categoría (confirmados):**

| Motor | Componente muerto | Data pipeline faltante |
|-------|-------------------|------------------------|
| Park+Weather (#5) | Weather (temp/viento/lluvia) | `game_data['weather']` siempre {} |
| Defensive Efficiency (#6) | Motor completo | `game_data['defense_home/away']` nunca poblado |
| HFA Engine (#7) | Travel fatigue (away_hfa) | `miles_traveled_away` y `time_zones_crossed_away` hardcodeados a 0 en run_module.py |

**Motores sospechados en esta categoría (pendiente audit):**
- Contextual Engine (#8) — context stage_factors std=0.0, a verificar

**Implicación estratégica:**
Activar motores muertos vía data pipeline tiene cost-benefit superior a reescribir motores
activos con señal débil. El esfuerzo es "conectar cables", no "rediseñar arquitectura".
El Defensive Efficiency Engine, por ejemplo, tiene código correcto con señal potencialmente
ortogonal a todos los motores activos. Conectar su data pipeline podría mejorar Brier sin
tocar ninguna fórmula existente.

---

### VALOR LATENTE ESTIMADO

Componentes con código correcto esperando data pipeline (confirmados + sospechados):

| Motor | Tipo de señal | Ortogonalidad a motores activos | Esfuerzo activación |
|-------|--------------|--------------------------------|---------------------|
| DEE (Motor #6) | Fielding puro (DER/OAA) | **Alta** — ortogonal a ERA/pitching | Bajo — fetcher MLB API + Savant |
| Weather (Motor #5) | Clima/viento/lluvia | Media — correlaciona con park factor | Bajo — fetcher OpenWeather ya existe |
| HFA Travel (Motor #7) | Fatiga visitante por viaje | **Alta** — ortogonal a todo | Bajo — cálculo geodésico por ciudad |
| Contextual B2B (Motor #8) | Fatiga back-to-back | Media — correlaciona con Pitcher Engine | Bajo — schedule API tiene fechas |
| Contextual Umpire (Motor #8) | Zone factor HP umpire | **Alta** — completamente independiente | Medio — requiere fuente externa |

**Estimación de impacto agregado en Brier si todos se activan correctamente:**
−0.0013 a −0.0028 (de 0.24305 → 0.2403–0.2418)
Pinnacle Brier = 0.24051 — este rango acercaría el sistema a Pinnacle.

**Implicación:** Activar motores muertos puede igualar o superar Pinnacle sin tocar
fórmulas de motores activos. Los "cables sin conectar" valen más que los ajustes de
magic numbers en motores activos con señal débil.

---

### HIPÓTESIS H3 — MOTORES MUERTOS / DECORATIVOS

**Estado:** Activa. Verificar con cada motor restante.

**Patrón detectado:** múltiples componentes del pipeline producen std=0.000000 en
stage_factors o Pearson ≈ 0 con outcomes reales, indicando que no producen señal.

**Lista de "motores muertos" (confirmados y sospechados):**

| Motor | Estado | Causa |
|-------|--------|-------|
| Weather (Motor #5) | **CONFIRMADO MUERTO** | game_data.weather siempre {} o roof_closed |
| Defensive Efficiency Engine (Motor #6) | **CONFIRMADO MUERTO** | defense_home/away nunca poblados en game_data — data pipeline faltante |
| Kalman defense en stage_factors (Motor #2) | **CONFIRMADO MUERTO** | Se aplica ANTES de inicializar stage_factors → gradient blind |
| HFA away_hfa factor (Motor #7) | **CONFIRMADO MUERTO** | miles_traveled_away y time_zones_crossed_away hardcodeados a 0 en run_module.py |
| Contextual Engine (Motor #8) | **FUNCIONALMENTE MUERTO** | B2B bloqueado por 3 defaults; rust/umpire activan en 0.26% de juegos |

**Implicación:** Si H3 se confirma completamente, el sistema declara hacer cosas que
no hace. Revivir motores muertos tiene mayor impacto esperado que ajustar magic numbers
de motores que sí funcionan. Un motor dead-but-correct (DEE) es más valioso que un
motor activo-pero-sin-señal (Pitcher Engine Pearson ≈ 0).

---

### HIPÓTESIS H2 — BACKTEST CONTAMINADO POR LOOK-AHEAD (Bullpen Engine)

**Estado:** Confirmada. Severidad ALTA.

**Motor afectado:** Bullpen Engine — `season = datetime.now().year` en línea 337.

**Alcance del daño:**
- 2,429 juegos de 2024 usaron estadísticas Savant de bullpen de **2026**
- 2,430 juegos de 2025 usaron estadísticas Savant de bullpen de **2026**
- Solo 563 juegos de 2026 usaron datos del año correcto

**Impacto potencial doble:**
1. **LOOK-AHEAD BIAS:** el motor "ve" información futura (stats de 2026 para predecir 2024).
   Si el bullpen de un equipo mejoró entre 2024 y 2026, el sistema sabrá eso de antemano.
   Esto podría inflar artificialmente la accuracy del backtest de 2024/2025.
2. **STALE DATA BIAS:** al momento de ejecutar el backtest (Mayo 2026), los datos de 2026
   solo tienen ~50 juegos jugados. Los juegos de 2024 reciben datos de temporada parcial de 2026.

**Consecuencia:** El Brier=0.24305 que se ha tomado como baseline puede estar inflado.
El verdadero Brier en producción (donde el motor usaría datos del año correcto) es desconocido.

**Prioridad:** ALTA — afecta toda interpretación de métricas del backtest. Cuando se arregle
este bug, re-correr backtest completo para calibrar expectativas reales.

---

## PRIORIZACIÓN DE FIXES

*(Se completa al final del audit)*

---

## MAGIC NUMBERS ACUMULADOS (todos los motores)

| Motor | Constante | Valor | Justificación |
|-------|-----------|-------|---------------|
| TTE | plate_disc_mult | 3.5 | Ninguna |
| TTE | _PRIOR_PA_EQUIVALENT | 1000 | Ninguna |
| TTE | composite weights | 0.50/0.30/0.20 | Ninguna |
| Kalman | _KALMAN_BLEND | 0.35 | Ninguna |
| Kalman | _KF_Q | 0.025 | Ninguna |
| Kalman | _KF_R | 9.0 | Ninguna |
| Kalman | _BIAS_CLAMP | 0.30 | Ninguna |
| Kalman | _LR | 0.01 | Ninguna |
| Kalman | _MIN_WEIGHT / _MAX_WEIGHT | 0.30 / 1.50 | Ninguna |
| Kalman | _MIN_SAMPLES | 10 | Ninguna |
| Pitcher | sub-factor weights | 0.321/0.256/0.192/0.128/0.103 | Ninguna |
| Pitcher | _K_TBF_ERA | 350 | "Research-based" sin cita |
| Pitcher | TBF/IP ratio | 4.3 | Sin cita |
| Pitcher | _AWAY_ERA_PENALTY | 0.15 | Cita vaga (BP/FG) sin URL |
| Pitcher | woba_mult scaling | 3.0 | Ninguna |
| Pitcher | brl_pct scaling | 0.012 | Ninguna |
| Pitcher | kbb_mult scaling | 1.5 | Ninguna |
| Pitcher | form/matchup ERA scaling | 0.06 | Ninguna |
| Pitcher | era_trend scaling | 0.03 | Ninguna |
| Bullpen | _LG_BP_ERA | 4.10 | Ninguna |
| Bullpen | _LG_BP_K_PCT / _BB_PCT | 0.248 / 0.086 | Ninguna |
| Bullpen | _K_ERA_BP / _K_XWOBA_BP / _K_K_BB_BP | 250 / 200 / 180 | Ninguna |
| Bullpen | _NORMAL_IP_3D | 9.0 | Ninguna |
| Bullpen | _LONG_RELIEF_ERA_DELTA | 0.80 | "empirical" sin cita |
| Bullpen | _HIGH_LEVERAGE_ERA_DELTA | 0.70 | "empirical" sin cita |
| Bullpen | _LONG_RELIEF_IPS_THRESHOLD | 5.5 | Ninguna |
| Bullpen | _HIGH_LEVERAGE_IPS_THRESHOLD | 6.5 | Ninguna |
| Bullpen | composite weights | 0.40/0.30/0.20/0.10 | Ninguna |
| Bullpen | k_bb scaling | 1.5 | Ninguna |
| Bullpen | workload tired/rested | 0.012 / 0.005 | Ninguna |
| Park | _NEUTRAL_TEMP_F | 72.0 | Ninguna |
| Park | _TEMP_RATE | 0.005 | "FanGraphs/Codify" sin URL |
| Park | wind rates (out/in/cross) | 0.020 / 0.015 / 0.003 | Ninguna |
| Park | wind angle zones | 60° / 120° | Ninguna |
| Park | _HANDEDNESS_WIND_SCALE | 0.040 | Ninguna |
| Park | _HANDEDNESS_WIND_MIN_MPH | 10.0 | Ninguna |
| Park | _AVG_LHB_PCT | 0.45 | Ninguna |
| Park | rain tiers (mm) | 1.0 / 5.0 / 10.0 | Ninguna |
| Park | rain multipliers | 0.99 / 0.97 / 0.95 | Ninguna |

**Total magic numbers hasta Motor #5: 34**
| DEE | _LG_DER | 0.715 | "2024 MLB average" sin cita |
| DEE | _LG_OAA_RUN | 0.80 | "Dewan/StatCast" sin URL |
| DEE | _K_BIP_DER | 500 | Cercano a investigación real |
| DEE | _DER_WEIGHT / _OAA_WEIGHT | 0.55 / 0.45 | Ninguna |
| DEE | _MAX_DEF_ADJ | 0.05 | Ninguna |

**Total magic numbers hasta Motor #6: 39**
| HFA | hfa_boost per-park (32 valores) | 0.025–0.045 | "Recalibrado" — relativo subjetivo |
| HFA | _default_hfa | 0.0325 | Ninguna |
| HFA | travel time_zones thresholds | 3/2/1 | Ninguna |
| HFA | travel miles thresholds | 2000 / 1000 | Ninguna |
| HFA | travel penalty values | 0.06/0.04/0.02/0.05/0.03 | Ninguna |
| HFA | travel cap | 0.10 | Ninguna |

**Total magic numbers hasta Motor #7: 47**
| Context | _B2B_MULT | 0.960 | "empirical MLB ~3-5%" sin cita |
| Context | _RUST_MULT | 0.980 | Ninguna |
| Context | _UMP_CLIP range | 0.960/1.040 | Ninguna |
| Context | _UMP_MIN_GAMES | 4 | Ninguna |

**Total magic numbers hasta Motor #8: 51**
| MC | rho_game | -0.06 | "empirical MLB data" — empírico real: -0.0078 (7.7×) |
| MC | lambda_noise | 0.05 | Ninguna — no adaptativo a calidad de datos |
| MC | early_stop_se | 0.0005 | Sin justificación del threshold específico |
| MC | F5_SCALE | 0.575 | "calibrated midpoint 55-58%" — sin datos |
| MC | n_max | 5_000_000 | Ninguna — nunca se alcanza (early stop a ~1M) |
| MC | block | 200_000 | Ninguna |
| MC | n_min (early stop) | 500_000 | Ninguna |

**Total magic numbers hasta Motor #9: 58**

**Total magic numbers hasta Motor #3: 19**
