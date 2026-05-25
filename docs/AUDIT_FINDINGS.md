# AUDIT INTERNO MLB PIPELINE — Hallazgos Acumulativos

Fecha inicio: 2026-05-25
Estado: EN PROGRESO

---

## RESUMEN EJECUTIVO

*(Se actualiza después de cada motor)*

- Motores auditados: 4/10
- Bugs CRÍTICOS: 2
- Bugs MEDIOS: 10
- Bugs BAJOS: 8
- Áreas oscuras: 4
- Magic numbers sin justificación: 25

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

Estado: PENDIENTE

---

## MOTOR #6 — DEFENSIVE EFFICIENCY ENGINE

Estado: PENDIENTE

---

## MOTOR #7 — HFA ENGINE

Estado: PENDIENTE
Pre-señal: `away_hfa` stage factor siempre = 1.000 (std=0.0) → posible motor parcialmente muerto

---

## MOTOR #8 — CONTEXTUAL ENGINE

Estado: PENDIENTE
Pre-señal: `home_context` y `away_context` stage factors siempre = 1.000 (std=0.0) → motor posiblemente muerto

---

## MOTOR #9 — MONTE CARLO SIMULATOR

Estado: PENDIENTE

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

**Total magic numbers hasta Motor #4: 25**

**Total magic numbers hasta Motor #3: 19**
