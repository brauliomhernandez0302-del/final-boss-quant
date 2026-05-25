# AUDIT INTERNO MLB PIPELINE — Hallazgos Acumulativos

Fecha inicio: 2026-05-25
Estado: EN PROGRESO

---

## RESUMEN EJECUTIVO

*(Se actualiza después de cada motor)*

- Motores auditados: 3/10
- Bugs CRÍTICOS: 2
- Bugs MEDIOS: 9
- Bugs BAJOS: 5
- Áreas oscuras: 3
- Magic numbers sin justificación: 13

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

Estado: PENDIENTE

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

*(Se completa al final del audit)*

### Issues pre-identificados (evidencia parcial)

1. **Triple shrinkage TTE→Kalman→Bias**: TTE shrinkea 85-90% hacia prior, Kalman blend 35% más,
   Bias dampening 65%. El dato real tiene influencia mínima.

2. **Gradient descent roto**: 5,422 juegos, 0 aprendizaje. Todo el sistema de pesos del pipeline
   es decorativo. Los fixes individuales de cada motor no serán descubiertos por el sistema de
   aprendizaje.

3. **LG_XWOBA inconsistente entre TTE (0.312) y Pitcher Engine (0.320)**: Sesgo sistemático
   cruzando dos motores.

4. **HFA y Context aparentemente muertos** (stage_factors std=0.0000 en ambos): Si confirmado
   en audits, 2 de 6 stage_factors del gradient descent son constantes → otra razón por la que
   los pesos no aprenden.

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

**Total magic numbers hasta Motor #3: 19**
