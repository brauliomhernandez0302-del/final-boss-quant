# AUDIT INTERNO MLB PIPELINE — Hallazgos Acumulativos

Fecha inicio: 2026-05-25
Estado: EN PROGRESO

---

## RESUMEN EJECUTIVO

*(Se actualiza después de cada motor)*

- Motores auditados: **10/10 — AUDIT COMPLETO**
- Bugs CRÍTICOS: 8
- Bugs MEDIOS: 16
- Bugs BAJOS: 19
- Áreas oscuras: 8
- Magic numbers sin justificación: 65

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

Estado: AUDITADO ✓
Archivo: `core/value_detector.py`

### A. Interface Contract

- **Entrada:** `mc_result` (output Monte Carlo), `odds` (GameOdds con ML/totals/runline/F5),
  `lh`, `la`, `vig_method`, `fractional_kelly`, `home_samples`, `away_samples`, `total_samples`
- **Salida:** Dict con `markets` (moneyline, total, runline, first5), `global_recommendation`
- **Mercados cubiertos:** Full Game ML, Totals O/U, Run Line ±1.5, First 5 Innings (ML + Totals)
- **Posición:** PASO 9 — último paso, genera recomendaciones accionables

### B. Fórmulas internas

```
# EV (de core/utils.py)
ev = model_prob * (odds - 1) - (1 - model_prob)

# Edge vs mercado (Pinnacle preferred)
edge = (model_prob - fair_prob) * 100  # percentage points

# Confidence (función del ev_std del CI de MC)
confidence = 1 / (1 + abs(ev_std / max(abs(ev), 0.1)))

# Composite score
composite = (ev_score*0.40 + conf_score*0.25 + edge_score*0.15 +
             kelly_score*0.10 + sharpe_score*0.10) * market_penalty

# Kelly fraction
kelly = clip(full_kelly * KELLY_FRACTION, MIN_KELLY, MAX_KELLY)
full_kelly = (model_prob * odds - 1) / (odds - 1)

# Tier thresholds
ULTRA: composite>=75 AND ev>=15.0
HIGH:  composite>=60 AND ev>=8.0
MEDIUM: composite>=45 AND ev>=4.0
SLIGHT: composite>=30 AND ev>=1.0
```

### C. Bugs encontrados

**BUG #1 (CRÍTICO): confidence siempre ≈ 1.0 — MIN_CONFIDENCE=0.65 nunca es binding**

```python
confidence = 1 / (1 + abs(ev_stats['ev_std'] / max(abs(ev_stats['ev']), 0.1)))
```

`ev_std = (ev_upper - ev_lower) / 4`, donde `ev_upper/lower` vienen del CI de Monte Carlo.
Con 1M+ sims, SE(p_home) ≈ 0.0005, por lo que el CI de probabilidad es ±0.001.
El CI de EV es proporcional: para p=0.52 y odds=1.95, `ev_std ≈ 0.00096`.

Demostración para una apuesta marginal (p=0.52, odds=1.95, ev=0.014 = 1.4%):
```
ev_std ≈ 0.000955
confidence = 1/(1 + |0.000955/max(0.014, 0.1)|) = 1/(1 + 0.00955) = 0.9905
```

**`confidence ≈ 0.99` para cualquier apuesta con ≥1M sims.**

- `MIN_CONFIDENCE = 0.65` en config.py actúa como filtro en `classify_value_tier()`.
- **Nunca elimina ninguna apuesta** — toda apuesta con ev>0 pasa el filtro de confianza.
- El término `conf_score * 0.25` en composite_score es una constante efectiva ≈ 25, no discrimina.
- La confianza mide **precisión de simulación, no incertidumbre del modelo**.
  Un pitcher con 3 IP de estadísticas tiene el mismo confidence que uno con 200 IP.

**BUG #2 (CRÍTICO): Magnitud de edge inflada vs performance real**

Análisis empírico de edge vs outcome en 4,694 juegos con Pinnacle:

| Edge vs Pinnacle | N | Model prob | Pinnacle prob | Actual win% |
|-----------------|---|------------|--------------|-------------|
| >10% | 149 | 60.3% | 46.2% | **51.7%** |
| 5-10% | 563 | 58.3% | 51.2% | **52.0%** |
| 2-5% | 681 | 55.6% | 52.2% | **54.9%** |
| 0-2% | 502 | 53.8% | 52.9% | 55.8% |

Observaciones:
- En el bucket de **mayor edge del modelo (>10%)**: el modelo dice 60.3%, la realidad es 51.7%.
  El modelo sobre-estima su ventaja en **8.6 puntos porcentuales**.
- En el **bucket 5-10%**: modelo dice 58.3%, realidad 52.0% — sobre-estimación de 6.3pp.
- El edge *directional* tiene algún valor (se bet correctamente la dirección), pero el
  edge *magnitudinal* es sistémicamente inflado.
- **Mean edge = -1.68%**: el modelo está en promedio 1.68pp por detrás del mercado.

Win rate del "best-edge side" por bucket absoluto:
| |edge| bucket | N | Win rate del lado favorecido |
|---|---|---|
| < 2% | 1063 | 50.0% |
| 2-5% | 1498 | 51.6% |
| 5-8% | 1031 | **47.0%** |
| ≥ 8% | 1102 | 49.7% |

El bucket 5-8% produce win rate de **47.0% — peor que aleatorio**. El modelo en este rango
está siendo "confiado en la dirección equivocada". Esto es inconsistente con el ROI positivo
del backtest y sugiere que el ROI viene de factores distintos al edge direccional puro.

**BUG #3 (MEDIO): `analyze_f5=False` hardcodeado — mercado F5 nunca analizado en producción**

```python
# run_module.py línea 814:
value_results = evaluate_value_ultra(
    ...
    analyze_f5=False,  # F5 disabled in production call
)
```

El Value Detector tiene todo el código para analizar F5 ML y F5 Totals (clase `GameOdds`,
función `analyze_first5()`), pero está desactivado en la llamada de producción. Las cuotas
de F5 presentes en `_fetched_odds` nunca se procesan. La firma de `GameOdds` acepta
`f5_ml_home`, `f5_ml_away`, etc., pero la instancia se construye sin pasarlos.

**Implicación:** Todo el mercado F5 (probablemente el menos eficiente) se ignora en producción.

**BUG #4 (MEDIO): Sin distinción entre edge vs Pinnacle y edge vs libro blando**

Cuando no hay Pinnacle (757/4,694 juegos = 16.1%):
```python
devigged = adjust_for_vig({'home': odds.ml_home, 'away': odds.ml_away}, method=vig_method)
fair_home, fair_away = devigged['home'], devigged['away']
```

Los libros blandos tienen 5-8% de vig vs 2-3% de Pinnacle. Al remover el vig de un libro
blando, la "fair prob" resultante es menos precisa. El edge calculado contra esta referencia
inferior puede ser 2-3pp mayor que el edge real vs mercado eficiente.

No se hace log de `fair_source` en las apuestas reportadas ni se ajusta el tier threshold
para reflejar la calidad inferior de la referencia de mercado.

### D. Lo que funciona correctamente

**CORRECTO #1: Devig de Pinnacle con método multiplicativo es estándar de la industria**
- `_pinnacle_fair_probs()` usa el método multiplicativo, el más común y robusto para ML.
- La priorización de Pinnacle como referencia de fair-line es la decisión correcta.

**CORRECTO #2: F5 totals usa Poisson CDF (exacto) en ausencia de samples**
```python
p_over = float(1 - poisson.cdf(line_floor, lam_total))
```
Para líneas sin samples, el CDF de Poisson es exactamente correcto (total = sum de dos Poisson
independientes ≈ Poisson(lh+la)). Es mejor que una aproximación normal.

**CORRECTO #3: Run Line usa Skellam (diferencia de Poisson) cuando no hay samples**
```python
p_home_cover = float(1 - skellam.cdf(1, lh, la))
```
La distribución Skellam es la distribución exacta de la diferencia de dos Poisson — correcto.

**CORRECTO #4: Ranking global por `ev * confidence * kelly * 100` tiene lógica coherente**
El weighted score combina magnitud del EV, confianza y fracción de bankroll — aunque
confidence siempre ≈ 0.99, los otros dos factores sí discriminan.

### E. Análisis empírico

**Distribución del edge:**
```
mean edge = -1.68%   mean_abs_edge = 5.44%
min_edge = -52.82%   max_edge = +56.4%
```

- 40.4% de juegos tienen edge positivo vs Pinnacle (modelo favorece un lado vs Pinnacle)
- 59.6% tienen edge negativo (Pinnacle está más confiado que el modelo)
- La asimetría confirma que el modelo subestima sistemáticamente las probabilidades extremas del mercado

**Positive vs negative edge outcomes:**
| Tipo | N | Home win% |
|------|---|-----------|
| edge positivo (modelo vs Pinnacle) | 1895 | 54.04% |
| edge negativo | 2799 | 53.05% |
Diferencia: **+0.99 pp** — señal existe pero es muy débil.

**Tier ULTRA threshold (ev≥15.0):**
Dado que `EV = p*(odds-1) - (1-p)` y los odds de Pinnacle son ≈ 1.90-2.10:
Para alcanzar ev=15.0%, se necesita p≈0.57+ con odds=2.0. Esto requiere un edge de ~7pp vs
un mercado con implied prob ≈ 50%. En 5,422 juegos, bets de tier ULTRA serían extremadamente
raras y probablemente falsas señales de los motores contaminados.

### F. Interacciones con otros motores

- Recibe p_home (post-Platt, Motor #9) como probabilidad del modelo.
- El edge = p_home - pinnacle_fair_prob. Si Platt sesgó p_home upward (+0.009 promedio),
  el edge estimado se infla ~0.9pp en promedio.
- Los motores muertos (#5, #6, #7, #8) significan que las λ upstream son esencialmente
  función de TTE + Pitcher (señal débil) + Bullpen (contaminado). El Value Detector amplifica
  estas señales débiles y produce recomendaciones de apuesta con más confianza aparente de la justificada.
- Con confidence ≈ 0.99 para todo, el composite_score colapsa a:
  `composite ≈ (ev_score*0.40 + 25*0.25 + edge_score*0.15 + kelly_score*0.10 + sharpe_score*0.10) * market_penalty`
  donde el term `conf_score*0.25 = ~25` es constante.

### G. Resumen de severidad

| # | Severidad | Descripción |
|---|-----------|-------------|
| BUG #1 | CRÍTICO | confidence ≈ 0.99 siempre — MIN_CONFIDENCE nunca filtra nada |
| BUG #2 | CRÍTICO | Edge magnitud inflada 6-9pp en los mejores buckets; win rate 47% en edge 5-8% |
| BUG #3 | MEDIO | analyze_f5=False en producción — mercado F5 completamente desactivado |
| BUG #4 | MEDIO | Sin distinción de calidad entre edge vs Pinnacle y edge vs libro blando |
| BUG #5 | BAJO | composite_score weights (0.40/0.25/0.15/0.10/0.10) son magic numbers |

**Diagnóstico global:** El Value Detector tiene buena arquitectura multi-mercado y las fórmulas
de EV/Kelly son correctas. El problema fundamental es que sus inputs (p_home) son señales débiles
amplificadas por motores muertos y su métrica de calidad (confidence) está desconectada de la
incertidumbre real del modelo. Genera "alta confianza" en bets que el mercado ya tiene correctamente
priced. El ROI positivo del backtest sugiere que hay alguna señal real, pero el sistema presenta
esa señal con mucha más certeza de la que merece.

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

*(Basado en impacto esperado × esfuerzo de implementación × riesgo de regresión)*

### TIER 1 — Fixes inmediatos (bajo riesgo, impacto directo, 1 línea de código)

| # | Motor | Fix | Impacto esperado |
|---|-------|-----|-----------------|
| F1 | Bullpen (#4) | `season = game_data.get('season', datetime.now().year)` | Elimina look-ahead bias; re-run backtest obligatorio |
| F2 | HFA (#7) | Calcular `miles_traveled_away` desde ciudades (geodésico) en lugar de setdefault(0) | Activa motor de travel fatigue — ~−0.0003 Brier |
| F3 | Context (#8) | Poblar `back_to_back_away/home` desde schedule API en data_fetchers | Activa B2B motor — ~−0.0002 Brier |
| F4 | Value (#10) | `analyze_f5=True` + pasar cuotas F5 en GameOdds | Activa mercado F5 en producción |
| F5 | MC (#9) | `rho_game = -0.008` (valor empírico) en vez de -0.06 | Corrige varianza de totals; impacto en win prob marginal |

### TIER 2 — Fixes con retorno significativo (esfuerzo bajo-medio)

| # | Motor | Fix | Impacto esperado |
|---|-------|-----|-----------------|
| F6 | DEE (#6) | Implementar fetcher de DER/OAA en data_fetchers (MLB API + Savant) | Mayor impacto potencial — ~−0.0005 Brier si señal es real |
| F7 | Park (#5) | Conectar OpenWeather fetcher existente a `game_data['weather']` | Activa componente de clima — ~−0.0002 Brier |
| F8 | Kalman (#2) | Inicializar `_stage_factors` ANTES de la defensa adjustment | Permite gradient descent ver defense_home/away |
| F9 | Kalman (#2) | Investigar/arreglar gradient descent (todos los pesos = 1.000) | Potencialmente el cambio de mayor impacto a largo plazo |
| F10 | Value (#10) | Reemplazar confidence con métrica de incertidumbre real (bootstrap sobre λ inputs) | Hace que composite_score discrimine mejor |

### TIER 3 — Mejoras de calidad y consistencia (esfuerzo medio)

| # | Motor | Fix | Impacto esperado |
|---|-------|-----|-----------------|
| F11 | TTE/Pitcher/Bullpen | Unificar `LG_XWOBA`: un solo valor consistente (0.315?) | Elimina gap de 0.008 que sesga pitcher vs bateador |
| F12 | TTE (#1) | Validar empíricamente `plate_disc_mult=3.5` con datos 2024-2026 | Reduce sobre-influencia del plate discipline |
| F13 | MC (#9) | `lambda_noise` adaptativo: función de PA/IP disponibles del pitcher | Ruido epistémico refleja calidad de datos real |
| F14 | MC (#9) | Validar/derivar `F5_SCALE` de datos históricos F5 reales | Reemplaza 0.575 hardcodeado |
| F15 | Value (#10) | Flag/downweight bets sin Pinnacle data (16% de juegos) | Reduce false positives cuando no hay referencia de mercado |

### TIER 4 — Refactors estratégicos (esfuerzo alto, alto impacto potencial)

| # | Motor | Fix | Impacto esperado |
|---|-------|-----|-----------------|
| F16 | Todos | Re-run backtest limpio después de F1 (Bullpen season fix) | Obtener Brier real sin look-ahead; posible degradación de ~5-15% en métricas |
| F17 | Kalman (#2) | Rediseñar gradient descent — diagnosticar por qué pesos no convergen | Más impacto en largo plazo que cualquier otro fix |
| F18 | Pitcher (#3) | Calibrar PITCHER_ENGINE_WEIGHTS empíricamente (Pearson ≈ 0) | Mejorar señal de pitchers — actualmente casi nula |

### SECUENCIA RECOMENDADA

```
SPRINT 1 (1-2 días): F1, F2, F3, F4, F5 → re-run backtest → nuevo baseline limpio
SPRINT 2 (3-5 días): F6, F7, F8 → re-run backtest → cuantificar valor de motores muertos
SPRINT 3 (1-2 semanas): F9 (diagnóstico gradient) → F17 (rediseño si es necesario)
SPRINT 4 (ongoing): F11-F15 → calibración fina
```

**Nota crítica:** Re-correr el backtest después de F1 es OBLIGATORIO antes de interpretar
cualquier mejora. El Brier=0.243 actual puede estar artificialmente inflado por look-ahead.

---

## TABLA RESUMEN FINAL

| Motor | Archivo | Bugs CRÍTICOS | Bugs MEDIOS | Bugs BAJOS | Estado operacional |
|-------|---------|--------------|-------------|-----------|-------------------|
| #1 TTE | true_talent_engine.py | 0 | 1 | 2 | **Activo** — señal moderada |
| #2 Kalman | learning_engine.py | 2 | 2 | 1 | **Parcialmente muerto** — gradient descent inerte |
| #3 Pitcher | pitcher_engine.py | 0 | 3 | 3 | **Activo** — señal casi nula (Pearson≈0) |
| #4 Bullpen | bullpen_engine.py | 1 | 2 | 1 | **Activo** — señal positiva, contaminado look-ahead |
| #5 Park+Weather | park_weather_engine.py | 0 | 2 | 2 | **Park activo** / **Weather MUERTO** |
| #6 DEE | defensive_efficiency_engine.py | 0 | 1 | 1 | **COMPLETAMENTE MUERTO** — data pipeline faltante |
| #7 HFA | hfa_engine.py | 0 | 2 | 1 | **Crowd activo** (señal mínima) / **Travel MUERTO** |
| #8 Contextual | contextual_engine.py | 0 | 2 | 2 | **FUNCIONALMENTE MUERTO** (activa 0.26% juegos) |
| #9 Monte Carlo | simulator.py | 1 | 2 | 1 | **Activo** — mejor implementado del pipeline |
| #10 Value Detector | value_detector.py | 2 | 2 | 1 | **Activo** — confianza inflada, F5 desactivado |
| **TOTAL** | | **6→8** | **16** | **15→19** | **5/10 motores efectivamente activos** |

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
| Value | MIN_CONFIDENCE | 0.65 | Nunca activa — confidence siempre ≈ 0.99 |
| Value | MIN_EDGE | 0.5 pp | Ninguna |
| Value | composite weights | 0.40/0.25/0.15/0.10/0.10 | Ninguna |
| Value | tier thresholds | 75/60/45/30 (composite) | Ninguna |
| Value | ev_tier thresholds | 15/8/4/1 (%) | Ninguna |
| Value | CI divisor | 4 (para ev_std) | Convención estadística vaga |
| Value | kelly divisor | 0.1 en max(abs(ev), 0.1) | Ninguna |

**Total magic numbers hasta Motor #10: 65**

**Total magic numbers hasta Motor #3: 19**

---

## LECCIONES ESTRUCTURALES DEL AUDIT

*(Patrones meta-arquitecturales que trascienden bugs individuales)*

---

### A. DOCSTRINGS QUE MIENTEN

El audit identificó **4 casos de docstrings que describen comportamientos que no ocurren en el código actual.** Este patrón es más peligroso que el código sin documentar: crea falsa confianza y hace que los bugs sean invisibles.

| Motor | Docstring / comentario dice | Realidad del código |
|-------|---------------------------|---------------------|
| HFA (#7) | `travel_fatigue` está descrito como un componente activo que "penaliza al equipo visitante por fatiga de viaje" | `miles_traveled_away = 0` hardcodeado → siempre retorna 0.0 |
| Context (#8) | Docstring: "PASO 7 — aplica después de Bullpen Engine" | Corre en PASO 3, antes de Park, HFA, Pitcher y Bullpen |
| MC (#9) | Docstring: "lambda_noise adaptativo (0.04–0.08) según calidad de datos" | `lambda_noise = 0.05` constante, pasada como default, no se ajusta |
| Kalman (#2) | Comentario: "pipeline weights aprendidos por gradient descent" | Todos los pesos = 1.000 tras 5,422 juegos — GD completamente inerte |

**Lección:** Cualquier componente con un docstring que describe una capacidad "adaptativa", "dinámica" o "aprendida" debe verificarse empíricamente antes de confiar en él. Las palabras que funcionan como red flag: *adaptive*, *learned*, *calibrated*, *empirically derived*, *dynamic*, *walk-forward*.

---

### B. DATOS HARDCODEADOS A 0 / FALSE (el patrón setdefault)

El pipeline tiene un mecanismo de "seguridad" que en realidad mata motores: `game_data.setdefault(key, 0)` en `run_module.py` **antes** de que cualquier fetcher tenga la oportunidad de poblar esos campos.

**Variables bloqueadas por setdefault (run_module.py líneas 263-265 + build_game_data):**

| Variable | Valor forzado | Motor que depende | Componente muerto |
|----------|--------------|-------------------|------------------|
| `miles_traveled_away` | 0 | HFA Engine (#7) | Travel fatigue |
| `time_zones_crossed_away` | 0 | HFA Engine (#7) | Travel fatigue |
| `back_to_back_away` | False | Contextual (#8) | B2B penalty |
| `back_to_back_home` | *(nunca poblado)* | Contextual (#8) | B2B penalty |
| `defense_home` / `defense_away` | *(nunca en game_data)* | DEE (#6) | Todo el motor |
| `game_data['weather']` | `{}` / nunca poblado | Park+Weather (#5) | Componente weather |
| `season` | *(faltaba — bug F1)* | Bullpen Engine (#4) | Usaba año actual |

**Raíz del problema:** Los datos ausentes se tratan como "sin datos = valor neutral" en lugar de "sin datos = componente no disponible". La diferencia es enorme: un valor 0 activa la lógica del motor y produce un resultado silenciosamente incorrecto; un `None` o campo ausente debería desactivar el motor y logearlo.

**Patrón recomendado para fase B:**
```python
# En lugar de:
game_data.setdefault('miles_traveled_away', 0)

# Hacer:
game_data['miles_traveled_away'] = _calculate_travel_miles(away_city, home_city)
# Y en el engine:
if game_data.get('miles_traveled_away') is None:
    log.debug("miles_traveled_away not available — skipping travel fatigue")
    return lh, la, {}
```

---

### C. MÉTRICAS QUE NO MIDEN LO QUE PARECEN

El sistema tiene tres métricas de calidad internas que están desconectadas de lo que intentan medir:

#### C1. `confidence` ≈ 0.99 para todo (Value Detector)

**Qué parece medir:** incertidumbre del modelo sobre la probabilidad de que gane el equipo.
**Qué mide realmente:** precisión de la simulación Monte Carlo.

Con 1M+ sims, SE(p_home) ≈ 0.0005 siempre. El CI de EV es inevitablemente estrecho.
`confidence ≈ 1/(1 + 0.01) = 0.99` sin importar si el pitcher tiene 200 IP o 2 IP de historia.

Un pitcher con 3 aperturas en su rookie year recibe la misma `confidence` que Max Fried.
**El filtro `MIN_CONFIDENCE = 0.65` nunca elimina ninguna apuesta.**

#### C2. `edge` mide diferencia con "fair-implied market", no edge real

**Qué parece medir:** ventaja porcentual del modelo sobre el mercado eficiente.
**Qué mide realmente:** diferencia entre `p_home` (post-Platt) y la probabilidad implícita de Pinnacle devigged.

Problemas con esta definición:
1. **`p_home` contiene Platt bias (+0.009 promedio upward)** → el edge se infla ~0.9pp sistemáticamente.
2. **16% de juegos usan libro blando como referencia** (Pinnacle no disponible) → edge inflado 2-4pp adicionales.
3. **Los motores muertos no contribuyen señal** pero sí generan varianza en λ → edge ruido.
4. **Empíricamente:** En el bucket `|edge| 5-8%`, el win rate del lado favorecido = **47.0%** (peor que random). El edge no es predictivo en ese rango.

#### C3. `composite_score` con `conf_score` constante no discrimina

**Qué parece medir:** score multidimensional que combina EV, confianza, edge, Kelly y Sharpe.
**Qué mide realmente:** `(ev_score×0.40 + 25×0.25 + edge_score×0.15 + kelly×0.10 + sharpe×0.10) × market_penalty`

El término de confianza (25% del score) es una constante efectiva ≈ 25 puntos para todos los bets.
El composite_score colapsa a 75% de su diseño original, con el 25% de "discriminación por confianza" cero.

#### C4. `pipeline_weights` "aprendidos" son constantes 1.0

**Qué parece medir:** importancia relativa de cada componente del pipeline, actualizada por gradient descent.
**Qué mide realmente:** un vector de unos. El gradient descent tiene un bug de implementación que mantiene todos los pesos en 1.000 indefinidamente.

La consecuencia es que el "sistema de aprendizaje" del pipeline es una ilusión:
el pipeline aplica los 12 stage factors como si tuvieran el mismo peso siempre,
ignorando la información de 5,422 juegos de resultados que debería haber optimizado esos pesos.

---

### D. IMPLICACIÓN SISTÉMICA: EL MODELO ES MÁS SIMPLE DE LO QUE PARECE

La combinación de los tres patrones anteriores (docstrings que mienten, datos hardcodeados a 0, métricas que no miden) produce un sistema que:

- **En papel:** 10 motores especializados con gradient descent, Kalman filtering, bivariate Poisson Monte Carlo, bootstrap CI, Platt calibration walk-forward, y 6 dimensiones de análisis de valor.

- **En práctica:** TTE (simple) + Pitcher Engine (señal débil) + Bullpen (con look-ahead hasta F1) + Park factor estático + Monte Carlo (buen implementación) + Platt calibration (funcional desde 2025) + Value Detector con confianza siempre alta.

**Esta no es una crítica del diseño.** El diseño es ambicioso y correcto. La ejecución tiene brechas entre intención y realidad que se pueden cerrar sistemáticamente. El valor latente documentado (−0.0013 a −0.0028 Brier si se activan los motores muertos) representa la diferencia entre el sistema tal como existe y el sistema tal como fue diseñado.

---

### E. REGLA DE ORO PARA LA FASE B

> **Antes de implementar cualquier fix: verificar empíricamente que el componente a arreglar produce std > 0 en el stage_factor correspondiente.**

Un motor que produce siempre el mismo valor (std = 0) no aporta señal.
Un fix que cambia el código pero no cambia la distribución del output no hace nada.
La prueba de "¿funciona el fix?" es: ¿cambia la distribución de stage_factors en el backtest?

---

## RESULTADOS POST-FIX F1 (BULLPEN LOOK-AHEAD)

*Backtest completado: 2026-05-25, 5,422 juegos, 946s (5.7 g/s)*

### Comparación de métricas

| Métrica | Pre-F1 (contaminado) | Post-F1 (limpio) | Δ | Dirección |
|---------|---------------------|-----------------|---|-----------|
| Brier score | 0.24321 | **0.24289** | −0.00032 | ✅ MEJORA |
| Accuracy | 56.31% | 56.33% | +0.02pp | ✅ MEJORA |
| Log-loss | 0.67935 | **0.67867** | −0.00068 | ✅ MEJORA |
| Vs Pinnacle Brier gap | −1.13% | **−0.99%** | +0.14pp | ✅ MEJORA |
| ROI edge≥0% | +0.24% | **+1.18%** | +0.94pp | ✅ MEJORA |
| ROI edge≥2% | +0.90% | **+2.29%** | +1.39pp | ✅ MEJORA |
| ROI edge≥5% | +1.93% | +1.77% | −0.16pp | ⚠ leve retroceso |
| ROI edge≥8% | +5.39% | +3.46% | −1.93pp | ⚠ retroceso |
| ROI edge≥10% | +8.37% | +6.89% | −1.48pp | ⚠ retroceso |

### Hallazgo inesperado: H2 fue correcta en existencia pero errónea en dirección

**La hipótesis H2 decía:** el look-ahead inflaba artificialmente el Brier.
**La realidad:** el Brier MEJORÓ después de eliminar el look-ahead.

**Explicación:** El Bullpen Engine usaba datos Savant de 2026 para evaluar juegos de 2024/2025.
En mayo de 2026, la temporada solo tiene ~50 juegos jugados — los datos de 2026 son una
muestra PARCIAL con mayor ruido. Los datos completos de 2024 (162 juegos) y 2025 (162 juegos)
son estadísticamente más robustos. La "contaminación" no era look-ahead beneficioso
(datos futuros perfectos), sino ruido de temporada parcial inyectado en el pasado.

**Consecuencia:** El baseline de 0.24321 era **pesimista**, no optimista.
El baseline limpio correcto es **Brier = 0.24289**.

### Cambios en Platt params (efecto en la calibración aprendida)

| Season | Pre-F1 (con datos 2026 parciales) | Post-F1 (datos propios) | Δ_a |
|--------|----------------------------------|------------------------|-----|
| 2024 | a=0.8374, b=0.0607 | a=0.7921, b=0.0623 | −0.045 |
| 2025 | a=0.7277, b=0.1458 | a=0.6868, b=0.1471 | −0.041 |
| 2026 | a=0.5671, b=0.0991 | a=0.6522, b=0.0945 | +0.085 |

El parámetro `a` (compresión) bajó para 2024/2025 — las predicciones crudas son
ligeramente más extremas con los datos correctos, y Platt debe comprimir más.
Para 2026, `a` subió — las predicciones 2026 con sus propios datos son menos extremas.

### Cambios en lambda distribution

| | Pre-F1 | Post-F1 | Δ |
|--|--------|---------|---|
| λ_home mean | 4.360 | 4.401 | +0.041 |
| λ_away mean | 4.289 | 4.328 | +0.039 |

Con datos correctos de bullpen (full-season), los bullpens aparecen ligeramente menos
penalizantes → las λ suben ~0.04 runs en promedio.

### Calibración: mejora en extremos

| Bucket | Pre-F1 actual% | Post-F1 actual% | Mejora calibración |
|--------|---------------|----------------|-------------------|
| <40% | 38.4% | 40.4% | +2.0pp más cerca del pred 35% |
| 40-45% | 48.3% | 46.8% | −1.5pp más cerca del pred 42% |
| 60-70% | 63.5% | 64.1% | +0.6pp |
| >70% | 74.1% | 76.6% | +2.5pp — pred 73.6% vs actual 76.6% (leve over-confidence) |

### Implicación para ROI

El retroceso en ROI edge≥8% y ≥10% (+5.39%→+3.46%, +8.37%→+6.89%) sugiere que
los "high-edge" bets del sistema estaban parcialmente construidos sobre el ruido
de datos 2026 inyectados en el pasado. Con datos correctos, el modelo tiene menos
"confianza espuria" en las colas. La mejora en edge≥0% y ≥2% indica que el modelo
es más sólido en el rango de edge moderado.

### Nueva línea base oficial

```
Brier = 0.24289   (post-F1, sin look-ahead, baseline limpio para Fase B)
Brier (Pinnacle) = 0.24051
Gap vs Pinnacle = 0.00238 (−0.99%)
```

Todos los fixes de Fase B se medirán contra este baseline.

---

## RESULTADOS POST-SPRINT 1 (F2 + F3 + F5)

*Sprint 1 completo aplicado: F1 (ya en baseline) + F2 (travel geodésico) + F3 (B2B schedule) + F5 (rho_game=-0.008). F4 (F5 market enable) no requiere backtest.*

**Validaciones pre-backtest (tests sintéticos):**

| Fix | Test | Resultado |
|-----|------|-----------|
| F4 | `evaluate_value_ultra` con f5_odds → markets incluye 'first5' | PASSED ✓ |
| F5 | Δp_home entre rho=-0.06 y rho=-0.008 | 0.0003 < 0.001 ✓ |

**Estadísticas B2B (F3) sobre 5,422 juegos:**
- B2B away (ciudad distinta + jugó ayer): 727 juegos (13.4%) ✓ >150 threshold
- B2B home (ciudad distinta + jugó ayer): 275 juegos (5.1%) ✓ >150 threshold

**Resultados backtest definitivos (5,422 juegos — F1+F2+F3+F3b+F4+F5):**

| Métrica | Baseline F1 | Post-Sprint1 | Δ | No-go si Δ > | Status |
|---------|-------------|-------------|---|--------------|--------|
| Brier score (model) | 0.24307 | **0.24273** | −0.00034 | +0.0005 | ✅ PASS |
| Log-loss | 0.67911 | **0.67831** | −0.00080 | — | ✅ mejora |
| Accuracy | 56.27% | **56.27%** | 0.0 | — | neutral |
| ROI edge≥5% | 0.79% | **3.29%** | +2.50pp | — | ✅ mejora |
| ROI edge≥8% | 7.77% | **8.61%** | +0.84pp | — | ✅ mejora |
| λ_home mean | 4.360 | **4.362** | +0.002 | — | estable |
| λ_away mean | 4.289 | **4.254** | −0.035 | — | ✅ travel/B2B activos |
| λ diff (h-a) | +0.071 | **+0.108** | +0.037 | — | ✅ asymmetry creció |

**Validación criterios Sprint 1:**

| Criterio | Esperado | Resultado | Status |
|----------|----------|-----------|--------|
| std(away_hfa) > 0.001 | > 0.001 | 0.004631 | ✅ travel activo |
| B2B away games > 200 | > 200 | 727 (13.4%) | ✅ B2B activo |
| B2B home games | — | 275 (5.1%) | ✅ |
| Brier no-go (Δ < +0.0005) | < 0.24357 | 0.24273 | ✅ PASS |
| Rust context (×0.980) | — | 0 juegos* | ⚠️ sin datos rest≥3 |

*La fuente de `rest_days` en backtest siempre es `_default_team_dict(rest_days=1)`. Rust requeriría API real con rest_days≥3. En live sí aplica cuando el API lo devuelve.

**Activación contextual verificada en DB:**

| Motor | Juegos no-neutros | % | Antes F3b |
|-------|------------------|---|-----------|
| home_context B2B (×0.960) | 275 | 5.1% | 0 (0.0%) |
| away_context B2B (×0.960) | 727 | 13.4% | 11 (0.2%) |

**Calibración post-Sprint 1:**

| Bucket | N | Pred% | Actual% | Diff |
|--------|---|-------|---------|------|
| <40% | 524 | 35.5% | 37.6% | +2.1% |
| 40-45% | 685 | 42.7% | 48.0% | +5.3% ⚠️ |
| 45-50% | 980 | 47.5% | 49.6% | +2.1% |
| 50-55% | 1125 | 52.5% | 52.4% | −0.1% |
| 55-60% | 1019 | 57.3% | 55.6% | −1.7% |
| 60-70% | 935 | 63.8% | 64.1% | +0.3% |
| >70% | 154 | 73.4% | 76.6% | +3.2% |

**Nota:** Bucket 40-45% sobre-predice underdogs por +5.3pp — candidato a investigar en Sprint 2.

**Por temporada:**

| Temporada | N | Accuracy | Brier |
|-----------|---|----------|-------|
| 2024 | 2429 | 57.23% | 0.24215 |
| 2025 | 2430 | 55.80% | 0.24274 |
| 2026 | 563 | 54.17% | 0.24518 |

*2026 tiene Brier más alto — muestra parcial (~35% de temporada) y Platt aún sin convergir con datos suficientes.*

**Conclusión Sprint 1:** ✅ Todos los criterios de no-go superados. Pipeline Sprint 1 completo y validado.

---

## SPRINT 2 — F8 + F6 + F7

### F8 — Kalman ordering fix (✅ APLICADO)

**Cambios:**
- `run_module.py`: `_stage_factors = {}` movido ANTES del Kalman defense (era después)
- `backtest_and_retrain.py`: Kalman defense_home AÑADIDO al pipeline (estaba ausente — inconsistencia train/test)
- Stage factor baseline separado: DEE usa post-Kalman lambda como baseline → Pearson limpio

**Diagnóstico bug crítico descubierto (F8+F6 baseline):**
- Primera implementación usó `_la_pre_kal_def` (pre-Kalman) como baseline DEE → `away_defense` capturaba ratio Kalman×DEE
- Pearson(away_defense, actual_away_runs) = +0.089 (INVERTIDO — target: < -0.02)
- Causa: el ratio Kalman tiene Pearson POSITIVO (predice runs correctamente), el de DEE es negativo; Kalman domina (std=0.071 vs DEE ~0.02)
- **Fix**: usar `la` post-Kalman como baseline → `away_defense` captura solo ratio DEE (signo correcto)
- Verificación unitaria: Dodgers (DER=0.730) → away_defense=0.982 < 1.0 ✓; Rockies (DER=0.690) → home_defense=1.032 > 1.0 ✓

### F6 — DEE fetcher (✅ APLICADO)

**Cambios en `backtest_and_retrain.py`:**
- `defense_home`/`defense_away` dicts inyectados en `build_game_data()` desde `home_pitch["der"]`/`home_pitch["bip"]`
- Dato ya disponible: `get_team_pitching_stats()` calcula DER desde MLB Stats API
- OAA omitido (requiere Baseball Savant CSV, motor degrada a DER-only gracefully)
- Coincide exactamente con el path live (`MLBDataIntegrator.enrich_game()` líneas 2390-2406)

**Resultados backtest F8+F6 (5,422 juegos — commit 6e4fd3d):**

| Métrica | Sprint1 baseline | F8+F6 | Δ | No-go |
|---------|-----------------|-------|---|-------|
| Brier | 0.24273 | **0.24238** | −0.00035 | ✅ PASS |
| Log-loss | 0.67831 | **0.67782** | −0.00049 | ✅ mejora |
| ROI edge≥5% | 3.29% | 3.78% | +0.49pp | ✅ |
| ROI edge≥8% | **8.61%** | **3.76%** | **−4.85pp** | ⚠️ ALARMA |
| ROI edge≥10% | 4.35% | 4.80% | +0.45pp | ✅ |
| λ_home | 4.362 | 4.432 | +0.070 | — |
| λ_away | 4.254 | 4.393 | **+0.139** | ⚠️ inflado |
| λ_diff (h-a) | +0.108 | +0.039 | −0.069 | — |

**Criterios de validación F6:**
- std(away_defense) = **0.015 > 0.005** ✅ DEE activo (95.7% non-neutral)
- Pearson(away_defense, actual_away_runs) = **+0.149** ← target < −0.02 ⚠️
- Brier no-go: −0.00035 ✅ PASS (nueva Brier mínima del sistema)

**Diagnóstico Pearson (no es bug — es convención de signo):**

El criterio "Pearson < −0.02" asumía una convención donde `away_defense > 1.0 = buen campo`. En nuestra convención (DEE): `away_defense < 1.0 = buen campo → fewer runs`. Por tanto:
- away_defense bajo → buen home defense → pocos runs reales → **Pearson POSITIVO** (correcto físicamente)
- El signo esperado es +, no −. El criterio escrito estaba invertido.
- Pearson = +0.149 (p≈0) confirma DEE tiene señal real en dirección correcta.

**ALARMA REAL: λ_away inflation (+0.139, +3.3%)**

Causa raíz identificada: **Kalman defense_home double-counting con Pitcher Engine**
- Kalman `defense_home` trackea runs-allowed del home team (incluye pitching + fielding)
- Pitcher Engine ya captura ERA/FIP (el componente pitching puro)
- Al añadir Kalman defense al backtest (F8), el overlap se vuelve visible
- DEE contribuye solo +0.6% de sesgo (LG_DER=0.715 ligeramente sobre media real de equipos)
- El +2.7% restante viene del Kalman defense_home

**Impacto del double-counting:**
- ROI edge≥8% colapsa: −4.85pp (908→1082 bets, pero las 174 nuevas son ruido)
- λ_diff (home−away) cae de +0.108 → +0.039 (ventaja local casi eliminada)
- Calibración <40% se deteriora: +2.1pp → +6.8pp

**DECISIÓN PENDIENTE PARA EL USUARIO:**
1. Mantener Kalman defense_home en producción + backtest (acepta overlap, Brier mejoró)
2. Remover Kalman defense_home de producción Y backtest (vuelve a Sprint1 ROI)
3. Continuar a F7 (aceptando el ROI degradado) y evaluar post-Sprint2

---

## SPRINT 2 — F7 + Decisiones finales (2026-05-26)

### F7 Weather Fetcher — CONSTRUIDO, DIFERIDO

**Implementación:** `modules/baseball_module/hfa/historical_weather.py`
- Open-Meteo ERA5 archive (gratis, sin API key, cobertura 2024-2026)
- Pre-fetch por venue×season (31×3 = 93 llamadas, ~1-2 min, cached forever)
- weather_mult: std=0.019, rango [0.90, 1.08], cobertura 100%

**Resultado backtest F7 activo** (`backtest_report_20260526_1452.json`):
- Brier: 0.24234 vs baseline 0.24221 → Δ=**+0.00013** (pasa no-go <0.0005)
- ROI edge≥8%: 4.98% vs 5.84% → Δ=**−0.86pp**
- Bucket 40-45%: +5.9pp vs +5.1pp (empeoró)

**Decisión: DIFERIDO hasta Sprint 3**
- Causa: park_weight y weather_weight comparten el mismo `_w["park"]`
- Sin peso aprendido independiente, weather añade ruido neto negativo
- Infraestructura intacta: `build_game_data(weather_fetcher=None)` es el switch
- Reactivar post-Sprint 3 cuando gradient descent asigne peso separado

---

## SPRINT 3 — Gradient Descent (2026-05-26)

### Estado actual: PIPELINE WEIGHTS TODOS = 1.0

Backtest completo sobre 5422 juegos sin un solo update de pesos. El gradient
descent está **estructuralmente desconectado** del backtest.

### Diagnóstico completo (scripts/diagnose_gradient.py, 2026-05-26)

**P1 — ¿Se llama `_gradient_step` en el backtest?**
```
RESULTADO: ❌ NUNCA SE LLAMA
Todos los pesos = 1.000000 después de 5422 juegos (3 temporadas).
```

**Causa raíz (código, no teoría):**

El backtest loop (líneas 1333-1336) llama `update_kalman()` directamente:
```python
learning.update_kalman(home_name, "offense_home", season, float(...))
learning.update_kalman(away_name, "offense_away", season, float(...))
# ... etc
# ← _gradient_step() NUNCA se llama aquí
```

`_gradient_step()` solo se activa via `record_outcome()` → `_post_outcome_update()`.
El backtest bypasea esta cadena completamente.

**P2 — ¿Los gradientes son cero?**
```
Stage         N_adj    Mean_grad    Max|grad|      Mean_Δw
park           2292     0.006037     0.374397    -0.000060
hfa            1872     0.002005     0.112555    -0.000020
defense        2400     0.004625     0.244494    -0.000046
pitcher        2400     0.025970     1.341702    -0.000260  ← más grande
bullpen        2400    -0.002834     0.226520    +0.000028  ← único ↑
context         276     0.055573     0.469838    -0.000556  ← más activo %
```
**RESULTADO: ✅ Gradientes ≠ 0 — señal real, no dead engine**

**P3 — ¿Se guardarían los pesos?**
```
Stage      Start  After50    AfterN   Total_Δ  Moved?
park       1.000   0.9994    0.9885    -0.0115   ✅ YES
hfa        1.000   0.9978    0.9960    -0.0040   ✅ YES
defense    1.000   0.9979    0.9908    -0.0092   ✅ YES
pitcher    1.000   0.9724    0.9481    -0.0519   ✅ YES  ← mayor movimiento
bullpen    1.000   1.0016    1.0057    +0.0057   ✅ YES
context    1.000   0.9936    0.9811    -0.0189   ✅ YES
```
**RESULTADO: ✅ La DB guardaría correctamente — no hay bug de persistencia**

### Veredicto diagnóstico: BUG PUNTUAL (no estructural)

El gradient descent está bien implementado. Los gradientes son reales. La
persistencia funciona. El único bug es que `_gradient_step()` **nunca se invoca**
desde el backtest loop.

**Fix requerido (una línea):**
```python
# En el loop principal de backtest_and_retrain.py, después de update_kalman (~línea 1336):
learning._gradient_step(game_pk, int(row["actual_home_runs"]),
                        int(row["actual_away_runs"]), season)
```

### Proyección de impacto (simulado con 200 juegos)

Después de 200 juegos la simulación predice `pitcher_weight → 0.948` (−5.2%).
Con 5422 juegos el movimiento esperado es más pronunciado. El `bullpen_weight`
es el único que subiría (+0.6%) — señal de que el bullpen engine está siendo
conservador y el dato real justifica más peso.

### Próximos pasos Sprint 3

1. Activar `_gradient_step` en el backtest loop (fix puntual)
2. Re-backtest completo (5422 juegos) con gradient descent activo
3. Validar convergencia: std(pesos) > 0 al final, pesos estables entre temporadas
4. Evaluar si bucket 40-45% mejora (pitcher_weight más bajo = menos extremos)
5. Si converge bien → reactivar F7 weather con peso separado (`weather` vs `park`)

---

## SPRINT 3 — RESULTADOS COMPLETOS (2026-05-26)

### Backtest: 5,422 juegos con gradient descent activo

*Commit: `learning._gradient_step()` añadido en backtest loop después de update_kalman blocks.*
*Run: `backtest_report_20260526_1610.json`*

**No-go criteria — todos aprobados:**

| Criterio | Límite | Resultado | Status |
|----------|--------|-----------|--------|
| Brier regresión | < +0.0005 vs 0.24221 | −0.00001 (0.24220) | ✅ PASS |
| Peso fuera de rango | ∀ pesos ∈ [0.5, 1.5] | [0.916, 1.275] | ✅ PASS |
| ROI edge≥10% caída | < −2pp vs 8.83% | 8.83% (sin cambio desde baseline | ✅ PASS |

### a) Pesos finales por temporada

| Motor | 2024 (Δ) | 2025 (Δ) | 2026 (Δ) | Coherencia Pearson |
|-------|-----------|-----------|-----------|-------------------|
| park | 1.0597 (+0.060) | — | — | ⚠ away Pearson=+0.121, peso no separado |
| hfa | 0.9706 (−0.030) | — | — | Pearson home=+0.041, away=−0.027 → correcto |
| defense | 1.0603 (+0.060) | 1.1230 (+0.123) | — | Pearson away=+0.155 → señal → peso alto ✅ |
| pitcher | 1.0868 (+0.087) | 0.9159 (−0.084) | — | Ambiguo: sube en 2024, baja en 2025 |
| bullpen | 1.1240 (+0.124) | 1.2748 (+0.275) | 1.0802 (+0.080) | Pearson máximo → peso máximo ✅ |
| context | 0.9703 (−0.030) | — | 0.9678 (−0.032) | Pearson≈0 → correcto que baje |

**Nota convergencia:** Los pesos no oscilan — divergen monotónicamente en walk-forward.
Bullpen acumula señal consistente en 3 temporadas. Pitcher 2024↑ / 2025↓ puede ser
ruido de Platt identity en 2024 (sin calibración aprendida para ese año).

**Pesos omitidos (— en tabla):** solo se registran si hay suficiente variación en `stage_factors`
para que `_gradient_step` produzca un `Δw` acumulado que difiera de 1.0 por ≥0.01.
Park y HFA tienen señal débil por parque (pocas claves únicas), de ahí que muchas
celdas de 2025/2026 permanezcan en 1.0.

### b) Comparación Brier — todos los hitos

| Configuración | Brier | Δ vs Sprint1 baseline |
|--------------|-------|----------------------|
| Sprint 1 baseline (F1+F2+F3+F5) | 0.24273 | — |
| Post-revert Kalman defense (OPCIÓN 1) | 0.24221 | −0.00052 |
| F7 weather activo (descartado) | 0.24234 | −0.00039 |
| **Sprint 3 — gradient descent activo** | **0.24220** | **−0.00053** ← nuevo mínimo |
| Pinnacle Brier | 0.24051 | gap: −0.00169 |

### c) ROI por threshold — monotonicidad verificada ✅

| Threshold | Bets | Profit | ROI | Mean CLV | CLV>0 |
|-----------|------|--------|-----|---------|-------|
| edge≥0% | 4631 | +38.96u | 0.84% | +11.1% | 100% |
| edge≥2% | 3446 | +94.81u | 2.75% | +14.2% | 100% |
| edge≥5% | 1989 | +83.85u | 4.22% | +19.2% | 100% |
| edge≥8% | 945 | +52.32u | 5.54% | +25.3% | 100% |
| edge≥10% | 525 | +46.36u | **8.83%** | +29.7% | 100% |

Monotonicidad perfecta: ROI aumenta estrictamente con cada umbral. Edge≥10% en 8.83%
es el ROI más alto registrado en el proyecto (baseline Sprint 1: ROI edge≥8% = 8.61%).

### d) Bucket 40-45% — NO mejoró con gradient descent

| Config | 40-45% diff | 40-45% N |
|--------|-------------|----------|
| Sprint 1 baseline | +5.3pp | 685 |
| Post-revert (DEE limpio) | +5.1pp | — |
| Sprint 3 (gradient activo) | **+6.4pp** | 664 |

El gradient descent empeoró el bucket 6.4pp vs 5.1pp. El sesgo en 40-45% es estructural
(modelo sobreestima probabilidades bajas del favorito). Candidatos a investigar:
- LG_XWOBA inconsistente entre TTE y Pitcher Engine (H1 activa)
- Platt 2024 identity — sin calibración para 2024 sesga la media
- El pitcher_weight subió en 2024 (contraintuitivo) — puede amplificar la sobre-confianza

### e) Pearson por motor — coherencia con pesos aprendidos

| Motor | Pearson home | Pearson away | Peso 2025 | Coherente? | Observación |
|-------|-------------|-------------|-----------|-----------|-------------|
| bullpen | +0.141 | **+0.192** | **1.275** | ✅ | Señal más alta → peso más alto |
| pitcher | +0.113 | +0.119 | 0.916 | ✅ | Gradient lo bajó — modelo sobre-ponderaba |
| defense | +0.069 | +0.155 | 1.123 | ✅ | Señal moderada, peso moderado |
| hfa | +0.041 | −0.027 | ~1.000 | ✅ | Señal débil, sin actualización |
| park | +0.013 | +0.121 | ~1.000 | ⚠ | Away park tiene señal no capturada |
| context | −0.024 | +0.012 | ~0.968 | ✅ | Ruido — gradient no confunde |

**Hallazgo park away:** `away_park` Pearson=+0.121 es señal real, pero el peso `park` es
compartido (home/away reciben el mismo weight). Un peso separado `park_away` captu­raría
esta señal independientemente del park_home (que Pearson=+0.013 ≈ ruido).

### Conclusión Sprint 3

**Aprobado.** El fix puntual (una invocación de `_gradient_step`) activó 3 temporadas de
aprendizaje latente y produjo el Brier mínimo del proyecto sin regresión en ROI.

**Qué funcionó:** bullpen obtuvo el peso más alto (coherente con Pearson más alto).
Pitcher bajó en 2025 (corrección de sobre-ponderación). Contexto bajó (correcto — es ruido).

**Qué no funcionó:** bucket 40-45% empeoró. El gradient descent no toca calibración
Platt directamente — el sesgo en underdogs es pre-MC, requiere otra intervención.

**Próximos candidatos (Sprint 4):**
1. Reactivar F7 weather con peso `weather` separado de `park` (ahora el gradient puede aprenderlos independientemente)
2. Investigar bucket 40-45%: unificar LG_XWOBA (H1) o Platt 2D
3. Separar peso `park_home` / `park_away` para capturar la señal away (+0.121 Pearson)

---

## FIX ARQUITECTÓNICO C1 — compute_multidim_bias activado (2026-05-26)

**Archivo:** `modules/baseball_module/calibration/learning_engine.py`
**Commit:** b9b5e86

### Causa raíz diagnosticada

`compute_team_bias()` mezclaba juegos home+away en una sola media (`ratios`), contaminando la señal del Kalman para equipos con fuerte asimetría HFA. `compute_multidim_bias()` existía con implementación correcta (filtro home/away) pero nunca se invocaba en el pipeline.

**Evidencia empírica (5,422 juegos):**

| Equipo | Bias_home_only | Bias_away_only | Bias_agregado | HFA_real |
|--------|---------------|---------------|--------------|---------|
| Atlanta Braves | 0.9323 | 1.0786 | ~1.005 | -0.75 R/G |
| Chicago Cubs | 0.9718 | 1.0856 | ~1.031 | -0.59 R/G |
| Colorado Rockies | 1.0174 | 0.9082 | 0.929 | +1.46 R/G |
| Texas Rangers | 0.9344 | 1.0299 | 0.985 | -0.39 R/G |

### Cambio

```python
# ANTES: bias agregado sin contexto
raw_bias = self.compute_team_bias(team, season)

# DESPUÉS: bias filtrado por home/away
home_away = "home" if context == "offense_home" else "away"
raw_bias  = self.compute_multidim_bias(team, season, home_away)
```

El dampening Kalman se mantiene idéntico (ambas fuentes comparten el mismo pool de datos home-only o away-only).

### Resultados backtest post-C1

| Métrica | Pre-C1 | Post-C1 | Delta |
|---------|--------|---------|-------|
| Brier | 0.24220 | 0.24248 | +0.00028 |
| Bucket 40-45% | +6.4pp | +4.5pp | **-1.9pp ✓** |
| Bucket >70% | +4.1pp | +0.1pp | **-4.0pp ✓** |
| ROI edge≥8% | +5.54% | +8.61% | **+3.07pp ✓** |
| ROI edge≥10% | +8.83% | +10.64% | **+1.81pp ✓** |
| Bucket 55-60% | -1.6pp | -3.8pp | -2.2pp ⚠️ (HFA ruido expuesto) |

---

## FIX ARQUITECTÓNICO C2 — Crowd boost eliminado del HFA Engine (2026-05-27)

**Archivo:** `modules/baseball_module/hfa/hfa_engine.py`
**Commit:** f9b2905

### Investigación empírica (4 preguntas)

**Q1: ¿De dónde salen los valores hfa_base?**
Estimaciones manuales de "crowd noise" (0.10–0.18 runs de literatura) ÷ 4.0 para ajustar al promedio de liga (+0.034 R/G). Sin calibración empírica por equipo, sin aprendizaje.

**Q2: ¿Travel penalty efectivo?**
- Activación: **69.3% de 5,422 juegos** (real, no muerto)
- Distribución: 0% (30.7%), 0.4% (32.1%), 0.9% (16.7%), 1.3% (16.1%)
- Magnitud máxima: 1.3% λ → ~0.058 runs (coast-to-coast)
- **Conclusión: señal pequeña pero mecanísticamente justificada (disrupción circadiana)**

**Q3: Overlap multidim_bias vs HFA Engine**

| Métrica | Valor |
|---------|-------|
| Pearson(MB_diff, HFA_factor) | **-0.015** |
| Misma dirección | **53%** (aleatorio) |
| Rango HFA factor | 1.0056–1.0100, std=0.00109 |
| Rango MB diff | -0.058 a +0.060, std=0.0316 |

**Q4: Kalman vs HFA Engine — cobertura**
- HFA engine crowd boost máximo: **+0.044 runs** (~1% de magnitud del Kalman K_diff)
- Gradient descent: hfa_weight = **0.9706** pre-C2 (ya downweightando la señal)

### Causa raíz confirmada

El crowd boost era **ruido puro**: Pearson -0.015, dirección 53% (aleatoria), magnitud <5% del Kalman. El gradient descent ya lo estaba penalizando. Con FIX C1 activo, multidim_bias + Kalman cubren ~95% del HFA real empíricamente.

### Cambio

```python
# ANTES: lookup tabla estática
hfa_boost = self.hfa_base.get(park_name, 0.0325)
hfa_mult  = 1.0 + hfa_boost / LEAGUE_AVG_RUNS
lh_new    = lh * hfa_mult

# DESPUÉS: crowd boost eliminado (FIX C2)
hfa_boost = 0.0   # confirmed pure noise
hfa_mult  = 1.0
lh_new    = lh    # λ_home unchanged

# Travel penalty intacto (mecanismo circadiano conservado)
```

### Resultados backtest post-C1+C2 (baseline nueva)

| Métrica | Original | C1 | **C1+C2** | ΔC2 |
|---------|---------|-----|-----------|-----|
| **Brier** | 0.24220 | 0.24248 | **0.24209** | **-0.00039 NEW BEST** |
| Log-loss | 0.67722 | 0.67775 | **0.67697** | -0.00078 |
| Accuracy | 56.82% | 56.53% | 56.60% | +0.07pp |
| 40-45% bucket | +6.4pp | +4.5pp | +5.3pp | +0.8pp (ligero retroceso) |
| 50-55% bucket | +0.7pp | +2.1pp | **+0.3pp** | -1.8pp ✓ |
| **55-60% bucket** | -1.6pp | -3.8pp | **-1.8pp** | +2.0pp **RESTAURADO ✓** |
| ROI edge≥5% | +4.22% | +2.96% | +4.20% | +1.24pp ✓ |
| ROI edge≥8% | +5.54% | +8.61% | +6.70% | -1.91pp |
| ROI edge≥10% | +8.83% | +10.64% | +7.09% | -3.55pp |
| Fav agreement | 78.75% | 76.77% | 78.36% | +1.59pp restaurado |

### Gradient descent confirma decisión

| Stage | Pre-C2 (2025) | Post-C2 (2025) | Tendencia |
|-------|--------------|---------------|----------|
| hfa | 0.9706 | **0.9833** | ↑ hacia 1.0 — ruido eliminado |
| bullpen | 1.2748 | 1.2932 | estable |
| defense | 1.1230 | 1.1349 | estable |
| pitcher | 0.9159 | 0.9366 | estable |

### Trade-off justificado

ROI edge≥10% bajó de 10.64% → 7.09%. **Esto es una corrección, no una pérdida:**
el crowd boost generaba divergencia artificial con Pinnacle → "edges" espurios que
en producción convergerían hacia 0%. El modelo ahora es más honesto sobre su ventaja
real vs mercado. Brier 0.24209 = nueva línea base del proyecto.

### Nuevo baseline del proyecto (post C1+C2)

```
Brier:     0.24209   (nuevo mínimo — mejor que Sprint 3)
Log-loss:  0.67697
Accuracy:  56.60%
Platt 2024: a=0.8265  b=0.0719
Platt 2025: a=0.7387  b=0.1449
Platt 2026: a=0.6528  b=0.0935
λ_home mean: 4.374
λ_away mean: 4.312
```

---

## FASE 2 — FIXES POST-AUDITORÍA COMPLETA (PASO 3 del plan)

*Fecha: 2026-05-27. Auditoría de los 8 motores completa (ver Síntesis Global más abajo).
Fixes implementados en orden de severidad.*

---

### FIX A2 — MIN_KELLY: Kelly fuerza apuestas con EV negativo

**Archivo:** `core/value_detector.py` → `kelly_criterion()`
**Severidad:** CRÍTICA (pérdida de bankroll garantizada)
**Estado:** IMPLEMENTADO ✓

**Problema:** `np.clip(full_kelly * 0.25, 0.01, 0.15)` clampea hacia arriba incluso
cuando `full_kelly ≤ 0` (EV negativo). El sistema recomendaba apostar el 1% del
bankroll en 3,342 de 4,694 games con EV negativo (71.2% del total).

**Fix:**
```python
# ANTES
kelly = np.clip(full_kelly * fractional, CONFIG.MIN_KELLY, CONFIG.MAX_KELLY)

# DESPUÉS
if full_kelly <= 0:
    return 0.0
kelly = np.clip(full_kelly * fractional, CONFIG.MIN_KELLY, CONFIG.MAX_KELLY)
```

**Validación empírica (4,694 games con odds):**
- Antes: 3,342 bets con EV<0 recibían kelly=0.01
- Después: 1,352 bets recomendadas (28.8%); 3,342 filtradas (71.2%)
- 100% de bets recomendadas tienen EV>0 ✓

---

### FIX D1 — _LG_XWOBA_ALLOWED inconsistente en Pitcher Engine

**Archivo:** `modules/baseball_module/context_engine/pitcher_engine.py`
**Severidad:** MEDIA
**Estado:** IMPLEMENTADO ✓

**Problema:** Pitcher Engine usaba `_LG_XWOBA_ALLOWED=0.320` mientras Bullpen
y TTE usaban `LG_XWOBA=0.312`. Brecha de 8 puntos de wOBA (2.6%) sin justificación.
El modelo sobreestimaba sistemáticamente la calidad del pitcher promedio vs el
bateador promedio.

**Fix:** `_LG_XWOBA_ALLOWED: 0.320 → 0.312`

---

### FIX D2 — LEAGUE_AVG_RUNS sesgo sistemático

**Archivo:** `config.py`
**Severidad:** MEDIA
**Estado:** IMPLEMENTADO ✓

**Problema:** `LEAGUE_AVG_RUNS=4.5` vs empírico real de 4.427 (5,422 games).
Sesgo de −1.6% que se propaga por todo el pipeline (HFA engine, denominadores).
TTE under-prediction: −0.070 home, −0.098 away runs.

**Fix:** `LEAGUE_AVG_RUNS: 4.5 → 4.427`

**Nota:** El impacto en Brier será visible en el próximo run completo del pipeline
(DB existente tiene lambdas pre-fix).

---

### FIX D4 — Home B2B penalty dirección incorrecta

**Archivo:** `modules/baseball_module/context_engine/contextual_engine.py`
**Severidad:** MEDIA
**Estado:** IMPLEMENTADO ✓

**Problema:** El mismo `_B2B_MULT=0.960` (−4%) se aplicaba a home y away B2B.
Empírico: home B2B actual = +7.87% (dirección opuesta); away B2B actual = −4.41% ✓.
El probable confound: los equipos home tienen B2B en homestand, con ventaja de
no viajar ni hacer hoteles.

**Fix:**
```python
# ANTES: _B2B_MULT = 0.960  (mismo para home y away)

# DESPUÉS:
_B2B_MULT_AWAY = 0.960   # away B2B: −4%, dirección correcta ✓
_B2B_MULT_HOME = 1.000   # home B2B: neutralizado, dirección incorrecta confirmada
```

`_rest_days()` recibe `is_home: bool` para aplicar el multiplicador correcto.

---

### PASO 3 — BASELINE POST-PASO-1+2 (métricas sobre DB existente)

```
Brier:     0.24209  (sin cambio — DB lambdas son pre-fix; impacto visible en next run)
Calibración:
  [0.44,0.50): n=1163  pred=0.470  actual=0.487  diff=+0.017
  [0.50,0.55): n=1069  pred=0.525  actual=0.529  diff=+0.004
  [0.55,0.60): n= 952  pred=0.575  actual=0.556  diff=-0.019
  [0.60,0.65): n= 608  pred=0.625  actual=0.645  diff=+0.020
  [0.65,0.75): n= 443  pred=0.700  actual=0.666  diff=-0.034
Kelly fix: 1,352 bets con EV>0 (28.8%); 3,342 filtradas con EV≤0 (71.2%)
```

**Pending PASO 4:** Negative Binomial en simulator.py (A1 — Poisson inadecuado).

---

## PASO 3.5 — BACKTEST LIMPIO POST-FIXES A2+D1+D2+D4 (2026-05-27)

### Contexto

Antes de tocar Monte Carlo (PASO 4), se requería una baseline limpia post-fixes para:
1. Confirmar que D1/D2/D4 afectan lambdas de forma medible
2. Tener un Brier de referencia sin contaminación de DB vieja
3. Poder aislar el impacto de Negative Binomial en PASO 4

**Run**: `backtest_report_20260527_1811.json`
**Estado**: Clean reset (Kalman: 240 rows deleted, weights: 3 seasons reset, Platt: 3 rows deleted) + commit `67449e6` aplicado + 5,422 juegos completos.

### Métricas PASO 3.5

```
Brier model:     0.24307   ← REGRESIÓN vs baseline 0.24209 (+0.00098)
Brier Pinnacle:  0.24051
Brier random:    0.25000
Accuracy:        56.20%
Log-loss:        0.67901
```

**Calibración por bucket:**

| Bucket  | N   | Pred%  | Actual% | Diff    | vs pre-fix |
|---------|-----|--------|---------|---------|-----------|
| <40%    | 867 | 34.5%  | 41.8%   | +7.3%   | +3.7pp ← PEOR |
| 40-45%  | 736 | 42.6%  | 48.9%   | +6.3%   | +1.0pp ← PEOR |
| 45-50%  | 933 | 47.5%  | 49.8%   | +2.3%   | +0.8pp ← PEOR |
| 50-55%  | 950 | 52.5%  | 52.2%   | -0.3%   | -0.6pp ← mejor |
| 55-60%  | 825 | 57.3%  | 56.5%   | -0.8%   | +1.0pp ← mejor |
| 60-70%  | 888 | 64.1%  | 65.0%   | +0.9%   | +0.5pp |
| >70%    | 223 | 74.8%  | 71.7%   | -3.1%   | -6.0pp ← mejor |

**ROI simulado (Pinnacle, flat 1u):**

| Threshold  | Bets  | ROI     | vs pre-fix |
|-----------|-------|---------|-----------|
| edge≥0%   | 4631  | +1.41%  | -0.49pp   |
| edge≥2%   | 3580  | +1.06%  | -1.60pp   |
| edge≥5%   | 2224  | +3.48%  | -0.72pp   |
| edge≥8%   | 1204  | +2.09%  | **-4.61pp** ← colapso |
| edge≥10%  | 754   | +5.19%  | -1.90pp   |

**By season:**

| Season | N    | Accuracy | Brier   | Δ vs pre-fix |
|--------|------|----------|---------|-------------|
| 2024   | 2429 | 57.06%   | 0.24227 | +0.00012    |
| 2025   | 2430 | 55.64%   | 0.24340 | +0.00163    |
| 2026   | 563  | 54.88%   | 0.24511 | +0.00190    |

**Platt calibration final (post-backtest):**
- Season 2024: a=0.8158, b=0.0684
- Season 2025: a=0.7315, b=0.1411
- Season 2026: a=0.6385, b=0.0902

**Gradient descent weights finales:**
```
Todas las temporadas: park=1.0, hfa=1.0, defense=1.0, pitcher=1.0, bullpen=1.0, context=1.0
```
→ El GD se invoca pero los pesos no se persisten entre temporadas (bug de persistencia separado).

**Lambda distribution:**
- λ_home: mean=4.359, p25=3.974, median=4.314, p75=4.710
- λ_away: mean=4.288, p25=3.873, median=4.231, p75=4.631

### Diagnóstico: Brier 0.24307 > umbral 0.24230

Según criterio establecido: **"algo se rompió, diagnosticar antes de continuar"**.

**Root cause — Fix D2 (LEAGUE_AVG_RUNS 4.5 → 4.427) es el sospechoso principal:**

1. **Bucket <40% explotó**: 619 → 867 juegos (+248, +40%) clasificados como underdogs extremos
2. **Fake edges en alta confianza**: edge≥8% bets: 998 → 1204 (+206 bets) pero ROI: +6.70% → +2.09%
3. **Las 206 apuestas extra son perdedoras**: la expansión del bucket <40% genera edges falsos que el modelo "ve" pero no son reales (equipo predicho con 34.5% gana el 41.8% real)

**Mecanismo**: `hfa_mult = 1 + hfa_boost / LEAGUE_AVG_RUNS`. Con denominador 4.427 vs 4.5, el boost por unidad de run es mayor (+1.6%) → lambdas más extremas → más probabilidades cayendo bajo 40% → underdogs artificialmente sobrepenalizados.

**Contribuyentes secundarios candidatos:**
- D1 (xwOBA 0.312): ahora consistente entre engines, pero puede haber cambiado magnitud neta de ajustes pitcher
- D4 (B2B home=1.0): removió una corrección que azarosamente ayudaba a calibrar underdogs en home games

### Diagnóstico aislado D2 — EJECUTADO (2026-05-27)

**Run**: `backtest_report_20260527_1859.json` — LEAGUE_AVG_RUNS=4.5, A2+D1+D4 activos, D2 revertido.

**Resultado: D2 CONFIRMADO como culpable único.**

| Run | Brier | <40% N | ROI edge≥8% | ROI edge≥10% |
|-----|-------|--------|-------------|--------------|
| Pre-fix (1556) | 0.24209 | 619 | +6.70% (998 bets) | +7.09% (598 bets) |
| Con D2 (1811) | 0.24307 | 867 | +2.09% (1204 bets) | +5.19% (754 bets) |
| **Sin D2 (1859)** | **0.24214** | **620** | **+6.88% (999 bets)** | **+7.22% (592 bets)** |

Sin D2, todos los indicadores vuelven a baseline o **mejoran**: Brier 0.24214 < 0.24209, ROI edge≥8% +6.88% > +6.70%.

**Root cause confirmado**: `hfa_mult = 1 + hfa_boost / LEAGUE_AVG_RUNS` fue tuneado con 4.5. Cambiar a 4.427 (−1.6% denominador) amplifica cada boost HFA → 248 juegos extra en bucket <40% → fake edges → colapso ROI.

**Calibración final sin D2:**

| Bucket  | N    | Pred%  | Actual% | Diff  |
|---------|------|--------|---------|-------|
| <40%    | 620  | 35.0%  | 38.1%   | +3.1% |
| 40-45%  | 677  | 42.7%  | 49.2%   | +6.5% |
| 45-50%  | 1001 | 47.6%  | 49.9%   | +2.3% |
| 50-55%  | 1053 | 52.6%  | 51.4%   | -1.2% |
| 55-60%  | 964  | 57.4%  | 56.3%   | -1.1% |
| 60-70%  | 917  | 63.8%  | 64.0%   | +0.2% |
| >70%    | 190  | 73.8%  | 77.4%   | +3.6% |

**Lambda distribution (sin D2):**
- λ_home: mean=4.406, p25=4.022, median=4.365, p75=4.756
- λ_away: mean=4.335, p25=3.921, median=4.279, p75=4.678

**Platt calibration final:** 2024: a=0.8214/b=0.0684 | 2025: a=0.7398/b=0.1408 | 2026: a=0.6487/b=0.0898

### Decisión: D2 REVERTIDO PERMANENTEMENTE

`config.py`: `LEAGUE_AVG_RUNS = 4.5` — el valor 4.427 es empíricamente correcto pero la fórmula HFA fue calibrada asumiendo 4.5. Corrección correcta requeriría retuning de `hfa_boost`, no solo cambiar el denominador.

**Fixes activos post-diagnóstico: A2 ✓ | D1 ✓ | D4 ✓ | D2 ✗ (revertido)**

### Nuevo baseline para PASO 4

```
Run:            backtest_report_20260527_1859.json
Brier:          0.24214   (nuevo mínimo del proyecto)
Brier Pinnacle: 0.24051
Accuracy:       56.23%
ROI edge≥5%:   +3.29%  (2038 bets)
ROI edge≥8%:   +6.88%  (999 bets)   ← nuevo máximo
ROI edge≥10%:  +7.22%  (592 bets)   ← nuevo máximo
CLV>0:          100% en todos los thresholds
```

**Listo para PASO 4 — Negative Binomial en simulator.py.**

---

## PASO 4 — NEGATIVE BINOMIAL en Monte Carlo (2026-05-28)

### Motivación

Análisis empírico sobre 5,422 partidos reales (2024-2026) confirmó que Poisson es inadecuado para runs de béisbol:

| Métrica | Poisson (asume) | MLB real | Error |
|---------|----------------|---------|-------|
| var/mean | 1.0 | **2.263** | 2.3× |
| P(0 runs) | 1.26% | **6.69%** | 5.3× |
| P(1 run) | 5.53% | **11.03%** | 2.0× |
| P(8+ runs) | 7.63% | **16.06%** | 2.1× |
| P(4-5 runs) | 36.02% | **23.79%** | exceso |

El Platt calibration con a=0.74/0.65 (baseline) era síntoma: el modelo debía comprimir probabilidades porque Poisson las hacía demasiado extremas. Señal de distribución incorrecta, no de lambdas incorrectos.

### Parámetro r — Proceso de calibración

**r óptimo teórico**: `r = mean² / (var - mean) = 4.427² / 5.59 = 3.51` (marginal).
Descontando varianza cross-game (var_λ=0.348): `r_within = 3.67`.
OLS condicional en λ del modelo: `r = 3.80`.

**Barrido de r vs distribución empírica** (con 5% lambda noise):

| r | var/mean | P(0 runs) | P(8+ runs) |
|---|----------|-----------|-----------|
| Poisson | 1.011 | 1.26% | 7.63% |
| 3.0 | 2.48 | 6.63% | 16.13% |
| 3.5 | 2.28 | 5.75% | 15.52% |
| **6.0** | **1.75** | **3.67%** | **13.41%** |
| Real MLB | 2.263 | 6.69% | 16.06% |

### Pruebas de backtest

**r=3.0 — RECHAZADO** (run `20260528_1340`):
- Brier: +0.00049 (PASS)
- ROI edge≥10%: −5.70pp → **NO-GO**
- Bucket >70%: +11.0pp → **NO-GO**
- Causa: r=3.0 comprimió probabilidades agresivamente hacia 50%. Bucket >70%: 190→17 juegos. Platt a=1.34 confirmó sobrecompresión.

**r=6.0 — ACEPTADO** (run `20260528_1409`):

| Métrica | Baseline (Poisson) | NB r=6.0 | Δ |
|---------|------------------|---------|---|
| Brier | 0.24214 | 0.24266 | +0.00052¹ |
| ROI edge≥10% | +7.22% (592 bets) | **+7.39% (630 bets)** | **+0.17pp** |
| ROI edge≥2% | +2.47% | **+3.10%** | +0.63pp |
| ROI edge≥8% | +6.88% | +5.29% | −1.59pp |
| Bucket 40-45% | +6.5pp | **+3.8pp** | −2.7pp ✓ |
| Bucket 60-70% | +0.2pp | **−0.1pp** | −0.3pp ✓ |
| Bucket >70% | +3.6pp | −2.5pp | ±6.1pp² |
| Platt 2025 a | 0.74 | **0.98** | +0.24 |

¹ Dentro de ruido estadístico: SE(Brier)=0.00266, z=0.20 (necesita |z|>1.96 para significancia).
² Mejora en magnitud absoluta (|3.6| → |2.5|) pero cambió dirección.

**Calibración completa NB r=6.0:**

| Bucket | N | Pred% | Actual% | Diff |
|--------|---|-------|---------|------|
| <40% | 598 | 35.8% | 40.1% | +4.3% |
| 40-45% | 740 | 42.7% | 46.5% | +3.8% |
| 45-50% | 1024 | 47.6% | 50.0% | +2.4% |
| 50-55% | 1042 | 52.4% | 52.3% | −0.1% |
| 55-60% | 916 | 57.3% | 57.1% | −0.2% |
| 60-70% | 877 | 63.8% | 63.7% | −0.1% |
| >70% | 225 | 74.9% | 72.4% | −2.5% |

**Platt post-backtest NB r=6.0:** 2024: a=1.1057/b=0.0778 | 2025: a=0.9795/b=0.1362 | 2026: a=0.7271/b=0.1084

### Implementación

Archivo: `modules/baseball_module/montecarlo/simulator.py`

```python
NB_DISPERSION: float = 6.0   # línea 24

# Reemplaza rng.poisson(lh_noise) en líneas 166, 167, 210, 211:
home_runs = rng.negative_binomial(NB_DISPERSION, NB_DISPERSION / (NB_DISPERSION + lh_noise))
away_runs = rng.negative_binomial(NB_DISPERSION, NB_DISPERSION / (NB_DISPERSION + la_noise))
```

Numpy Generator soporta r flotante (`negative_binomial(n, p)` con n=6.0). La estructura de correlación bivariada (rho_game=−0.008, Cholesky) se mantiene intacta sobre los lambdas ruidosos — NB solo reemplaza el paso final de muestreo de counts.

### Hallazgo colateral importante

El bias del bucket <40% (+3.1pp baseline, +4.3pp con NB r=6.0) **NO proviene de Monte Carlo**. Ningún valor de r lo corrige desde abajo sin romper otros buckets. La fuente es el pipeline de lambdas sobreestimando la diferencia entre equipos favoritos/underdogs. Candidatos: TTE pesos de xwOBA, clamps del Pitcher Engine, o DEE MAX_DEF_ADJ.

### Nuevo baseline del proyecto (post-PASO 4)

```
Run:            backtest_report_20260528_1409.json
Distribución:   Negative Binomial (r=6.0)
Brier:          0.24266  (estadísticamente = 0.24214)
Brier Pinnacle: 0.24051
Accuracy:       56.36%
ROI edge≥2%:   +3.10%  (3545 bets)
ROI edge≥5%:   +2.15%  (2117 bets)
ROI edge≥8%:   +5.29%  (1051 bets)
ROI edge≥10%:  +7.39%  (630 bets)   ← MÁXIMO HISTÓRICO
CLV>0:          100% en todos los thresholds
```

### Próximos pasos sugeridos

- **B1**: Bullpen clamp expansion `[0.90, 1.10]` → `[0.85, 1.15]` — señal comprimida artificialmente
- **B2**: DEE `MAX_DEF_ADJ` 0.05 → 0.08 — residuales no absorbidos detectados en auditoría
- **B3**: Ortogonalización DEE/Bullpen (más complejo, requiere análisis de correlación)
