# AUDITORÍA VAL — El camino del valor: de λ a pick

**Fecha**: 2026-07-23. **Alcance**: `montecarlo/simulator.py` + `core/value_detector.py` (nunca auditados
end-to-end antes; el audit 2026-07-14 y su remediación cubrieron PIT/backtest/learning-loop/devig-single-source/
TTE, todo upstream de λ). **Commit auditado**: `4e95aac` (working tree, rama `feature/point-in-time-rebuild`).
`simulator.py`/`value_detector.py` no tienen cambios sin commitear — última modificación real `0fd674e`.

**Régimen de congelamiento vigente** (`docs/PROTOCOLO_CLV_V1.md`, sección Registro, leída en esta sesión):
`engine_commit` congelado = `80d0faea8fae96a1bb11fd18d53cafcb382a1682`. **D0: pendiente** — el protocolo
está comprometido y el motor ya congelado *desde ese commit*, pero la ventana de 6 semanas todavía no arrancó
(falta veredicto V1(a) + keepalive de Windows). Implicación práctica citada por el dueño en este mismo hilo:
un fix de motor ANTES de que D0 arranque no resetea un reloj que aún no corre — después de D0 sí lo haría.
Esto es contexto para priorizar, no una autorización para tocar nada; ver STOP al final.

**Método**: (1) respuesta conocida — computar la misma cantidad por dos caminos independientes y exigir
coincidencia; (2) invariantes — propiedades que deben cumplirse siempre; (3) transparencia forzada —
reproducir a mano cada número interno contra el output real. Caso real usado en todo el reporte: Colorado
Rockies @ Milwaukee Brewers, hoy (2026-07-23), Sugano (away) vs Drohan (home), λ_h=6.115 λ_a=3.558 —
`persist=False` en todas las corridas, cero escrituras a `game_outcomes`. Scripts de verificación ad-hoc en
`/tmp/claude-*/scratchpad/val1_*.py` (no committeados — desechables); lo permanente vive en
`tests/verification/test_val_audit_invariants.py` (17 tests: 15 verdes, 2 rojos intencionales, ver VAL-8).

---

## Tabla resumen

| Item | Veredicto | Severidad | Nota |
|---|---|---|---|
| VAL-1.1 Distribución exacta | CORRECTO | — | NB(r=6.0) con λ ruidoso bivariado, documentado con cita |
| VAL-1.2 Empates | CORRECTO (documentado) | BAJA | split 50/50 fijo, ignora la fuerza relativa de los equipos |
| VAL-1.3 Walk-off | **HALLAZGO** | **ALTA** | sobreestima P(margen≥2\|ganó) +8.8pp, P(over) +5.1pp — exactamente donde salen los ULTRA |
| VAL-1.4 Convergencia / CI | **HALLAZGO** | **MEDIA-ALTA** | bootstrap CI subsamplea a 10K fijo, ~7-9x más ancho que la precisión real |
| VAL-1.5 Respuesta conocida | CORRECTO | — | coincide con NB independiente analítica dentro de 5·SE |
| VAL-1 bonus: correlación ρ_game | **HALLAZGO** | MEDIA | ρ de entrada casi no sobrevive al draw final (atenuación >10x) |
| VAL-2 Mercados derivados | CORRECTO | — | conteo exacto verificado, push manejado, invariantes exactos |
| VAL-3 Devig / fair line | CORRECTO | — | single-source confirmado, fallback honesto si falta un lado |
| VAL-4.1-3 EV/edge/Kelly | CORRECTO | — | fórmulas reproducidas a mano, antisimetría exacta |
| VAL-4.4 Piso de Kelly | OBSERVACIÓN | BAJA-MEDIA | floor 1% infla 3.6x un stake marginal (diseño existente, no bug) |
| VAL-5 Tiers / composite | CORRECTO | — | 6/6 tiers reales reproducidos, composite_score=58.5 reproducido a mano |
| VAL-5.3 confidence constante | CORRECTO (documentado) | — | es una métrica de juego, no de mercado — por diseño |
| VAL-6 total_multiplier pitcher | CORRECTO | — | fórmula lineal ponderada, reproducida a 6+ cifras significativas |
| VAL-7.3 total_mult bullpen | CORRECTO | — | reproducido exacto para ambos bullpens del caso real |
| VAL-7.2 Conteos de relievers | **HALLAZGO** | BAJA | 3 números sin coordinar (n_pitchers=2 vs n_siera_pitchers=7 vs roster UI) |

**5 CORRECTO limpios, 6 HALLAZGO/OBSERVACIÓN reales, 0 NO VERIFICABLE.**

---

## VAL-1 — El simulador

### 1.1 Distribución exacta (CORRECTO)

`monte_carlo_advanced()` (`montecarlo/simulator.py:66-331`): cada lado samplea
`NegativeBinomial(NB_DISPERSION=6.0, p=NB_DISPERSION/(NB_DISPERSION+λ_noise))` (líneas 177-178), donde
`λ_noise` es la λ de entrada perturbada por ruido gaussiano bivariado con correlación `rho_game=-0.008`
por defecto (líneas 121-175, descomposición de Cholesky explícita: `la_noise = la + σ_a·(ρ·z1 + √(1-ρ²)·z2)`).
`NB_DISPERSION=6.0` está documentado y justificado contra un backtest real (líneas 14-29: r=3.0 daba mejor
ajuste de cola pero peor ROI; r=6.0 fue la elección final). No hay independencia total entre home/away — están
correlacionados *a través de* λ_noise, no directamente entre los draws NB (ver hallazgo de correlación abajo).

### 1.2 Empates — la pregunta central (CORRECTO, documentado; observación BAJA)

Línea 261-262:
```python
p_home = (wins_home_total + 0.5 * ties_total) / sims_done
p_away = (wins_away_total + 0.5 * ties_total) / sims_done
```
No hay re-sampleo, no hay modelo de entradas extra — cada empate cuenta como **medio punto para cada lado**,
determinísticamente, sin importar cuán desbalanceadas sean las λ de entrada. Cuantificado en una corrida
instrumentada del caso real (λ_h=6.115, λ_a=3.558, n=2,000,000, sin ruido para aislar el mecanismo puro):

- **8.85% de las simulaciones empatan** antes de la resolución (`ties_pct_raw_nb_no_noise=0.0885235`).
- Ese 8.85% coincide, por curiosidad, con la tasa real aproximada de juegos de MLB que van a entradas extra
  (~8-9%) — la FRECUENCIA de empates está bien calibrada.
- Pero el REPARTO no lo está: en este juego, home gana el 68.2% de las simulaciones NO empatadas
  (`p_home_from_wins_only=0.6818`). Un modelo que continuara sampleando con las mismas tasas relativas en una
  entrada extra le daría a home ≈63.2% de los empates (proporcional a λ_h/(λ_h+λ_a)), no 50%. El split fijo
  le resta a home ~1.2pp de p_home en este caso específico — pequeño, pero sistemático a favor del equipo
  más débil en cada empate. Severidad BAJA porque el efecto neto es de fracciones de punto porcentual, y
  cualquier fix tocaría probabilidad final (archivo congelado, ver STOP).

### 1.3 Walk-off — HALLAZGO, severidad ALTA

El simulador **no modela la regla de walk-off en absoluto**: ambos lados samplean su distribución NB completa
sin truncar cuando el home ya va ganando entrando a la baja de la 9ª. Verificado empíricamente contra
`game_outcomes` real (`source='backtest'`, 4,825-juego baseline vigente), condicionando a juegos con λ
cercanas al caso (`5.0≤λ_h≤7.5`, `2.5≤λ_a≤4.5`, **N=881 juegos reales**), comparando contra el modelo mismo
sampleado con las λ propias de cada juego (NB puro, 300-500 sims/juego, sin ruido ni Platt — "qué cree el
modelo" vs "qué pasó de verdad"):

| Métrica | Real | Modelo (NB puro) | Δ |
|---|---|---|---|
| home runs \| home ganó | 6.413 | 6.982 | **−0.568 runs** |
| P(margen local ≥ 2 \| ganó) | 73.76% | 82.48% | **+8.72pp** |
| P(total > 9) [línea común] | 39.39% | 44.50% | **+5.11pp** |
| Home win rate (bruta, sin Platt/mercado) | 64.02% | 59.27% | +4.75pp |

La fila más limpia (misma condición, mismas λ, aislada del resto del pipeline): cuando el home gana, el
modelo le asigna en promedio **0.57 carreras de más** que las que realmente anota — exactamente lo esperable
si el modelo sigue "bateando" la baja de la 9ª que en la realidad nunca se juega. Esto se propaga directo a
**P(margen≥2\|ganó)**, +8.7pp, y a **P(total>9)**, +5.1pp — **los dos mercados exactos** (`RUNLINE HOME -1.5`,
`OVER`) donde el sistema reporta sus EVs "ULTRA" más altos hoy (+15.99% y +14.83% en el caso real capturado,
ver VAL-4). No es coincidencia: es la misma fuente de sesgo. Test permanente:
`tests/verification/test_val_audit_invariants.py::test_walkoff_truncation_bias_is_absent` — **rojo hoy**,
intencionalmente.

### 1.4 Convergencia — HALLAZGO, severidad MEDIA-ALTA

**Dispersión real (5 seeds, mismo juego, n=800,000 sims cada corrida, `store_samples=True`):**

| Métrica | std entre seeds | rango |
|---|---|---|
| p_home | 0.000179 | [0.72536, 0.72586] |
| p_over | 0.000459 | [0.56811, 0.56935] |
| p_rl_home | 0.000150 | [0.5826, 0.5830] — **ancho total 0.0004** |

Extremadamente estable — el motor de simulación en sí converge bien a n=800K.

**Pero el CI que se muestra al usuario (`prob_ci`, y por extensión `ev_ci`/`ev_std`/`sharpe`) no refleja
esa precisión.** `bootstrap_confidence_interval()` (`core/value_detector.py:163-186`) subsamplea a
`n_sub = min(n, 10_000)` **antes** de bootstrapear (línea 175), sin importar que la corrida real tuvo
500,000-5,000,000 sims. Verificado: para `p_rl_home` en una corrida de 800K, el CI bootstrap reportado
tiene **ancho 0.0193** — **48x más ancho** que la dispersión seed-a-seed real medida arriba (0.0004). Test
permanente reproduce esto de forma determinista a n=500,000: ancho reportado 0.01971 vs ancho honesto a esa n
0.00273 (**ratio 7.2x**) — y el ratio empeora con n, porque el subsample cap es fijo mientras n crece
(`√(n_sims/10,000)`: a n=5,000,000 típico en vivo, el ratio esperado es ~22x).

**Impacto real, no solo cosmético**: `ev_std` (derivado del ancho del CI) alimenta `sharpe_ratio()`
(`ev/ev_std`), que a su vez pesa 10% en `calculate_composite_score()`. Un CI artificialmente ancho da un
`ev_std` artificialmente grande, lo que **subestima** `sharpe` y por tanto el `sharpe_component` del
composite score — el sesgo va en la dirección conservadora (castiga el score, no lo infla), pero sigue
siendo un número mostrado directamente al usuario (`prob_ci` en cada mercado) que no significa lo que dice
significar. Nota: esto es un mecanismo **distinto** del ya documentado en `_SHARPE_SCORE_ANCHOR`
(`value_detector.py:293-312`, la discrepancia backtest-n_max=50K vs live-n_max=5M) — ese ajuste ya corrige la
escala del ancla; el bug de aquí es que el ANCHO DEL CI EN SÍ ni siquiera usa el n real de la corrida. Test
permanente: `test_bootstrap_ci_width_reflects_actual_sample_size` — **rojo hoy**, intencionalmente.

### 1.5 Respuesta conocida (CORRECTO)

Con `lambda_noise=0.0, rho_game=0.0` el simulador se reduce a dos `NegativeBinomial(6.0, ·)` independientes
(sin ruido de λ ni correlación). Comparado contra la distribución analítica exacta (convolución directa de
las dos pmf NB, mismo split de empates 50/50 que usa el propio simulador):

- Analítico: p_home = 0.725939 (n=infinito)
- Simulado: p_home = 0.7253 (n=3,000,000, seed=99)
- Diferencia: 0.00064 — dentro de 2.5·SE de muestreo (SE≈0.00026 a esa n)

**El mecanismo de sampleo del núcleo del simulador es correcto** — no hay bug en el generador NB en sí. Test
permanente: `test_known_answer_independent_nb_matches_analytic` (verde).

Referencia adicional (no un hallazgo, contexto): la misma comparación contra Skellam-Poisson puro da
p_home=0.7465 — la sobredispersión NB (elegida para replicar var/mean≈2.26 real vs 1.0 de Poisson) por sí
sola mueve p_home casi 2.5pp hacia abajo relativo a un modelo Poisson ingenuo, y sube la masa de empates de
8.85%(NB) a 9.39%(Poisson) muy poco — la elección NB está bien fundamentada para lo que se propone.

### Bonus: correlación ρ_game casi no sobrevive al draw final — HALLAZGO, severidad MEDIA

El docstring de `monte_carlo_advanced` (líneas 92-95) afirma que `rho_game=-0.008` modela "pitcher duels keep
both teams down". Verificado que el efecto real es casi nulo:

| ρ_game de entrada | correlación observada en (home_runs, away_runs) |
|---|---|
| −0.008 (default) | **−0.00074** |
| −0.5 (exagerado a propósito) | **−0.0024** |

Incluso forzando una correlación de entrada 62x más fuerte que el default, la correlación observada en los
runs finales apenas se mueve (de -0.0007 a -0.0024, un factor ~3x, no ~62x). Mecanismo: la correlación se
inyecta en `λ_noise` (`σ = lambda_noise(0.05) × λ`, i.e. ruido ≈5% de λ), pero la varianza condicional del
muestreo NB dado λ (`Var(NB|λ) = λ(1+λ/r)`, con r=6 eso es ~2x λ) domina la varianza total por un factor de
~30-60x sobre la varianza del ruido de λ. El resultado es que, aunque el código está bien construido
matemáticamente (descomposición de Cholesky correcta, verificable), el parámetro `rho_game` es, en la
práctica, **casi decorativo** — el modelo simula esencialmente dos equipos independientes pese a la
documentación y al parámetro dedicado. Severidad MEDIA porque no introduce un sesgo direccional (el efecto es
solo "casi cero" en vez de "el valor pretendido"), pero contradice lo que el docstring afirma que el código
hace, y cualquier análisis que asuma que el modelo captura anti-correlación de duelos de pitcheo (p.ej. para
combos de mercados / parlays) estaría mal fundamentado.

---

## VAL-2 — Mercados derivados del sim (CORRECTO en todo)

Verificado por conteo directo sobre una corrida instrumentada (n=1,000,000, seed=55, mismo caso real):

- `p_rl_home` reportado = 0.5833 == `mean(diff >= 2)` recontado a mano = 0.58331 → **coinciden exacto**.
- `p_rl_away` reportado = 0.4167 == `mean(diff <= 1)` = 0.41669 → **coinciden exacto**. Suma = 1.0 exacto.
- `p_over`/`p_under` (línea 8.5, semi-entera) recontados a mano desde `total_samples`: coinciden a 6 cifras
  decimales. `p_push=0` siempre en línea semi-entera (total de carreras es siempre entero, nunca puede
  empatar 8.5) — confirmado.
- **Línea entera (9.0)**: `p_push=0.0964` (¡9.6% de las simulaciones empatan exactamente en 9!, no trivial) —
  correctamente separado de over/under, y `calculate_ev_stats()` (`value_detector.py:192-222`) compensa el
  push sumando `push_prob*100` al EV (devuelve el stake, no lo cuenta como pérdida). `p_over+p_under+p_push`
  = 1.0 exacto en ambos casos.
- Todos los invariantes pedidos (`p_home+p_away=1`, `p_over+p_under+p_push=1`,
  `p_rl_home+p_rl_away=1`) se cumplen **exactos**, no aproximados, por construcción de las condiciones
  (`>` / `<` / `>=` / `<=`) usadas — no hay solapamiento ni hueco posible dado que los runs son enteros.

Tests permanentes: `test_runline_cover_probabilities_sum_to_one`, `test_total_ou_sums_to_one_half_integer_line`,
`test_total_ou_push_handled_on_integer_line`, `test_moneyline_probabilities_sum_to_one` — 4/4 verdes.

---

## VAL-3 — Devig y fair line (CORRECTO)

`_pinnacle_fair_probs()` (`value_detector.py:129-137`) llama `remove_vig_multiplicative([pin_home, pin_away])`
— confirmado single-source, ambos lados del mismo par. Reproducido a mano:

- **Caso real** (pin 1.44/3.02): `implied=[0.694444, 0.331126]`, `total=1.025571` →
  `fair_home=0.677130, fair_away=0.322870` — **coincide exacto** con `pin_fair_home=0.6771,
  pin_fair_away=0.3229` del render real. `pin_vig_pct` recalculado = 2.557% ≈ 2.56% reportado.
- **Caso extremo** (favorito fuerte, 1.10/8.00): `implied=[0.909091, 0.125]` → `fair_home=0.879121,
  fair_away=0.120879`, suma exacta 1.0.

**Si falta un lado** (`odds.pin_home` o `odds.pin_away` es `None`/0): la guardia
`if odds.pin_home and odds.pin_away:` (línea 829) falla y el código cae a
`adjust_for_vig({'home': odds.ml_home, 'away': odds.ml_away}, ...)` — devigea las cuotas del **propio libro
que se está apostando**, no fabrica un número. `fair_source` queda etiquetado con el método usado (no
`"pinnacle"`), así que un consumidor downstream puede distinguir "fair Pinnacle" de "fair del libro propio".
Si además faltan `ml_home`/`ml_away`, todo el bloque de moneyline (línea 817) se salta — no aparece la key
`moneyline` en `all_markets`, no hay número fabricado. **Comportamiento honesto confirmado en los tres
escalones de degradación.**

Tests permanentes: `test_devig_multiplicative_sums_to_one` (3 casos parametrizados, incluye 1.91/1.91
simétrico) — 3/3 verdes.

---

## VAL-4 — EV, edge y Kelly (CORRECTO en fórmula; una observación de diseño)

### 4.1 Reproducción a mano (caso real, 3 mercados; corrida fresca — los números difieren ligeramente de
los citados en la tarea original por ruido MC entre corridas, ver VAL-1.4 arriba sobre por qué eso es
esperable y cuánto debería moverse)

| Mercado | p (display, 4 dec) | odds | EV mano | EV real | edge mano | edge real | Kelly mano | Kelly real |
|---|---|---|---|---|---|---|---|---|
| MONEYLINE HOME | 0.6884 | 1.46 | 0.5064 | 0.500 | 1.130 | 1.122 | 0.0100 | 0.0100 |
| OVER 8.5 | 0.5685 | 2.02 | 14.8370 | 14.828 | 7.990 | 7.985 | 0.0364 | 0.0363 |
| RUNLINE HOME −1.5 | 0.5829 | 1.99 | 15.9971 | 15.988 | 8.800 | 8.793 | 0.0404 | 0.0404 |

Fórmulas usadas (`core/utils.py:calculate_ev`, `value_detector.py:224-238 kelly_criterion`):
`EV=(p·odds−1)×100`, `edge=(p−implied_true)×100`, `kelly=clip(((p·odds−1)/(odds−1))×0.25, [0.01,0.15])`.
Las diferencias residuales (≤0.019) son enteramente atribuibles al redondeo a 4 decimales del campo
`probability` mostrado (d(EV)/dp = odds×100 ≈ 150-200, así que un error de 0.00005 en p produce ~0.01 de
error en EV) — no hay discrepancia real de fórmula.

**Confirmado edge antisimétrico exacto**: `edge_home + edge_away = 0.0` en los tres pares (ML, OVER/UNDER,
RUNLINE), por construcción (`p_home+p_away=1` y `implied_true_home+implied_true_away=1` ⟹ la suma de edges
es idénticamente 0). Test permanente `test_edge_antisymmetric_real_market` (verde).

### 4.2 Dónde se aplica `fractional_kelly=0.25`

Dentro de `kelly_criterion()` mismo (línea 237): `kelly = clip(full_kelly * fractional, MIN_KELLY, MAX_KELLY)`.
El valor que se **muestra y se persiste** (`analyze_market_generic` → `'kelly': kelly`) **ya es el
fraccional, recortado** — no hay un segundo lugar donde se vuelva a fraccionar. Consistente en las tres
capas (persistido = mostrado = documentado en `CLAUDE.md`: "`core.value_detector.kelly_criterion()` es la
única fuente de verdad").

### 4.3 Piso de Kelly — OBSERVACIÓN, severidad BAJA-MEDIA (diseño existente, no un bug nuevo)

`CONFIG.MIN_KELLY=0.01` (`config.py:28`). Para MONEYLINE HOME: `full_kelly=(0.6884×1.46−1)/(1.46−1)=0.01101`
(1.10% del bankroll sin fraccionar). Cuarto-Kelly: `0.01101×0.25=0.00275` (0.275%) — **por debajo del piso**,
así que el valor final mostrado/apostado es `0.01` (1.00%), no `0.00275`. **El piso convierte un stake
fraccional-Kelly "correcto" de 0.275% en un stake real de 1.00% — 3.6x más grande** — justo en el mercado con
el EV más marginal del caso (+0.5%, apenas por encima de `MIN_EDGE=0.5`). Esto es un diseño intencional
documentado (`CLAUDE.md`: "clips the result to [MIN_KELLY, MAX_KELLY] = [1%, 15%]"), no un bug — pero vale
la pena que el dueño lo tenga presente como un riesgo real: bets con edge apenas por encima del umbral mínimo
reciben un tamaño de apuesta que NO refleja proporcionalmente qué tan marginal es la edge. Test permanente
`test_kelly_clipped_to_configured_bounds` documenta el comportamiento actual (verde — no es un hallazgo que
falle, es una confirmación de que el piso existe y se aplica donde se espera).

---

## VAL-5 — Tiers y composite score (CORRECTO)

### 5.1 Umbrales del enum — umbral de **EV%** (no de edge ni de score)

`ValueTier` (`value_detector.py:31-49`): `ULTRA=(…, 7.25, "S")`, el segundo campo de la tupla es el EV%
mínimo. `classify_value_tier()` (línea 378) exige AMBOS `composite_score >= gate` Y `ev >= umbral` para cada
tier (AND, no OR) — verificado contra los 6 mercados reales del caso:

| Mercado | EV | composite | Tier esperado | Tier real |
|---|---|---|---|---|
| RUNLINE HOME −1.5 | 15.99 | 58.5 | comp≥50 & ev≥7.25 → ULTRA | 🔥 ULTRA ✓ |
| OVER 8.5 | 14.83 | 57.2 | ULTRA | 🔥 ULTRA ✓ |
| MONEYLINE HOME | 0.50 | 20.7 | comp≥17 & ev≥0.5 → SLIGHT | ⚪ SLIGHT ✓ |
| MONEYLINE/OVER/RUNLINE AWAY (los 3) | <0 | — | ev<0 → NEGATIVE (guardia de línea 385) | 🔴 NEGATIVE ✓ |

6/6 coinciden.

### 5.2 `composite_score` — fórmula y reproducción a mano de RUNLINE HOME (58.5)

```
ev_score     = clip(ev/8.86×100, 0, 100)        = clip(15.988/8.86×100) = clip(180.4) = 100.0
conf_score   = confidence×100                    = 0.9284×100 ≈ 92.84
edge_score   = clip(edge×10, 0, 100)             = clip(87.93) = 87.93
kelly_score  = clip(kelly/0.15×100, 0, 100)      = clip(26.93) = 26.93
sharpe_score = clip(sharpe/42.35×100, 0, 100)    = clip(39.33) = 39.33

market_efficiency = 1 − overround/100 = 1 − 1.53/100 = 0.9847
market_penalty     = 1 − (0.9847×0.3) = 0.7046   [reportado: 0.705, coincide]

pre_penalty = 100×0.40 + 92.84×0.25 + 87.93×0.15 + 26.93×0.10 + 39.33×0.10
            = 40.00 + 23.21 + 13.19 + 2.69 + 3.93 = 83.02
composite   = 83.02 × 0.7046 = 58.49 ≈ 58.5   ← coincide con el real
```

Los 5 componentes redondeados (`ev_component=40.0, conf_component=23.21, edge_component=13.19,
kelly_component=2.69, sharpe_component=3.93`) coinciden exactos con `score_breakdown` del render real.

`market_penalty≈0.70` — **qué es y por qué**: un descuento multiplicativo sobre TODO el composite,
proporcional al vig del libro que se está apostando (`market_efficiency=1−overround/100`). Observación menor
(no un hallazgo): dado que `overround` real rara vez supera ~3-5% en libros líquidos, `market_penalty` vive
casi siempre en la banda estrecha 0.685-0.715 — funciona más como un multiplicador casi constante que como
una señal diferenciadora real entre mercados de distinto vig. No es incorrecto, solo tiene rango dinámico
práctico limitado.

### 5.3 `confidence: 0.929` idéntico en todos los mercados — CORRECTO, por diseño documentado

`compute_data_quality_confidence()` (`value_detector.py:403-451`) es una métrica **de juego**, no de mercado:
combina `sp_ip` (40%, del abridor más débil de los dos), `1−prior_weight` del TTE (35%, más débil), y
`kalman_n_obs` (25%, más débil) — inputs de `game_meta`, calculados **una sola vez** por
`evaluate_value_ultra()` (línea 801) antes de iterar mercados, y pasados idénticos a cada
`analyze_market_generic()`. Mide "¿el modelo tiene información real de calidad para ESTE juego?", no
"¿qué tan seguro está el modelo de ESTE mercado?" — están respondiendo preguntas distintas por diseño, no es
un bug. El propio docstring lo aclara explícitamente (líneas 427-430): no cierra el problema separado y
todavía abierto de sobreconfianza en el bucket de edge alto.

### 5.4 Ranking `weighted`/`weighted_score`

`weighted = ev × confidence × kelly × 100` (líneas 885, 959-961, 978, etc. — mismo patrón en las 4 secciones
de mercado). Reproducido a mano para RUNLINE HOME: `15.988×0.929×0.0404×100=60.0055` — coincide exacto con
`weighted=60.00552208` del render real (14 dígitos, no solo 4 decimales — coincidencia total, no solo dentro
de tolerancia de redondeo). Ordena `all_opportunities` descendente por este valor; `rank`/`score`/
`weighted_score` son alias del mismo par (`composite_score`, `weighted`) añadidos en el nivel de agregación
de `run_module.py` — sin lógica nueva, solo renombrado para el consumo del frontend.

---

## VAL-6 — Pitcher Engine: cómo se combina `total_multiplier`

**Fórmula real** (`context_engine/pitcher_engine.py:130-138`): **combinación lineal ponderada por deltas**,
NO un producto:
```
total = 1 + Σ w_i × (factor_i − 1),   pesos: quality=0.321 form=0.256 matchup=0.192 fatigue=0.128 platoon=0.103
```
(los 5 pesos suman exactamente 1.000 — `config.py:79-85`). Reproducido a mano para los dos abridores reales
del caso (823759, Sugano/Drohan):

**Sugano** (away, ajusta λ_home): quality=1.450, form=1.08519, matchup=1.000, platoon=0.97297, fatigue=1.008
```
Δ = 0.450×0.321 + 0.08519×0.256 + 0×0.192 + (−0.02703)×0.103 + 0.008×0.128
  = 0.14445 + 0.02181 + 0 − 0.00278 + 0.00102 = 0.16450
total = 1.16450   →  real: 1.164497683672599   (coincide a 5 cifras)
```

**Drohan** (home, ajusta λ_away): quality=0.84874, form=1.00773, matchup=1.000, platoon=1.05647, fatigue=1.008
```
Δ = (−0.15126)×0.321 + 0.00773×0.256 + 0 + 0.05647×0.103 + 0.008×0.128
  = −0.04855 + 0.00198 + 0 + 0.00582 + 0.00102 = −0.03973
total = 0.96027   →  real: 0.9602661283907692   (coincide a 6 cifras)
```

**Resuelve la aparente paradoja de la tarea** ("uno queda debajo del producto crudo, el otro arriba"): la
pregunta asumía implícitamente que la fórmula parte de un producto de los 5 factores y luego lo amortigua.
**No es así — nunca hay un producto**. Es una suma ponderada de desviaciones desde 1.0 desde el principio, así
que compararla contra un producto crudo hipotético compara dos objetos matemáticos distintos; no hay
contradicción real que explicar, solo una intuición inicial incorrecta sobre qué tipo de combinación es. El
que Sugano quede sobre 1.0 y Drohan bajo 1.0 es simplemente el signo neto de la suma ponderada de cada uno —
nada especial en la dirección relativa al producto crudo (que ni siquiera se calcula en ningún punto del
código real).

---

## VAL-7 — Agregación de bullpen

### 7.1 `tier_label: long_relief(blend=X)`

`bullpen_engine.py:600-613`. `avg_ips` (innings esperadas del abridor, clamp [4.0, 8.0]) determina qué tier de
relevista entra: si `avg_ips < 5.5` (abridor corto) → **long_relief**, `blend=min(1.0, (5.5−avg_ips)/1.5)`,
sube el ERA usado (`+0.80`, relevistas de baja categoría entran antes). Si `avg_ips > 6.5` (abridor profundo)
→ **high_leverage**, `blend=min(1.0,(avg_ips−6.5)/1.0)`, baja el ERA (`−0.70`, cierre/setup de alta categoría).
Rango observado 0.01-1.00 es exactamente el rango posible dado el clamp de `avg_ips` a [4.0,8.0]: `blend=1.0`
en el extremo (`avg_ips=4.0` → máximo efecto long-relief), `blend≈0` cerca del umbral (`avg_ips≈5.5`).
Confirmado con el caso real: away bp (Rockies) `avg_ips=5.42, blend=0.05`; home bp (Brewers) `avg_ips=5.48,
blend=0.01` — ambos abridores esperados a durar casi hasta el umbral, casi sin efecto de tier.

### 7.2 Tres conteos sin coordinar — HALLAZGO, severidad BAJA

Confirmado con el caso real (bullpen away = Rockies): `n_pitchers=2` (relievers con cobertura Savant
xwOBA/barrel, usados en el 55%+10%=65% del `quality_raw` cuando `used_siera=True`) vs `n_siera_pitchers=7`
(relievers con cobertura FanGraphs SIERA/xFIP, 35% del `quality_raw`) — **misma bullpen, mismo día, dos
fuentes de datos con cobertura muy distinta (2 vs 7)**. Un tercer número, el tamaño del roster de relevistas
que ve la UI (`api/mlb_presentation.py::fetch_bullpen_roster`, corregido esta misma sesión para usar
`_fetch_reliever_ids` real — ver resumen de la tarea anterior), viene de una llamada **independiente** a la
misma función de clasificación, no de los mismos `n_pitchers`/`n_siera_pitchers` que el engine ya calculó y
expone en metadata. Ninguno de los tres es, por construcción, igual a los otros. **Ninguno de los tres afecta
λ directamente** — lo que realmente pesa en la regresión Bayesiana es `total_pa`/`total_ip×_TBF_PER_IP`
(líneas 616, 645-646), no el conteo crudo de pitchers — pero un usuario viendo cualquiera de estos números sin
etiqueta clara podría razonablemente pensar que miden lo mismo. Severidad BAJA: es un problema de legibilidad/
definición, no de matemática incorrecta.

### 7.3 Reproducción a mano de `total_mult` (caso real, ambos bullpens)

`raw_mult = quality_mult × workload_mult`; `total_mult = clip(1 + innings_weight×(raw_mult−1), [0.85,1.15])`
(`bullpen_engine.py:684-689`):

```
Away (Rockies bp, ajusta λ_home):  0.9433×1.0036=0.9467 (real: 0.9467) ✓
  total = 1+0.398×(0.9467−1) = 0.97879  →  real: 0.9788 ✓

Home (Brewers bp, ajusta λ_away):  0.8647×0.9965=0.8617 (real: 0.8617) ✓
  total = 1+0.391×(0.8617−1) = 0.94593  →  real: 0.9459 ✓
```

Ambos reproducidos exactos a 4+ cifras.

---

## VAL-8 — Tests permanentes

`tests/verification/test_val_audit_invariants.py` — **17 tests, 15 verdes, 2 rojos intencionales**:

```
PASSED  test_moneyline_probabilities_sum_to_one
PASSED  test_runline_cover_probabilities_sum_to_one
PASSED  test_total_ou_sums_to_one_half_integer_line
PASSED  test_total_ou_push_handled_on_integer_line
PASSED  test_devig_multiplicative_sums_to_one[3 casos]
PASSED  test_ev_formula_matches_definition[3 casos]
PASSED  test_edge_antisymmetric_real_market
PASSED  test_kelly_clipped_to_configured_bounds
PASSED  test_simulator_reproducible_with_fixed_seed
PASSED  test_lambda_monotonicity_p_home_and_runline
PASSED  test_known_answer_independent_nb_matches_analytic
FAILED  test_walkoff_truncation_bias_is_absent          ← HALLAZGO VAL-1.3, severidad ALTA
FAILED  test_bootstrap_ci_width_reflects_actual_sample_size  ← HALLAZGO VAL-1.4, severidad MEDIA-ALTA
```

Los 2 rojos son **por diseño** — codifican los dos hallazgos cuantitativos más importantes de esta auditoría
como aserciones ejecutables, con la tolerancia puesta muy por encima del ruido de muestreo esperado para que
no puedan pasar "por accidente". Quedan rojos hasta que un fix real (priorizado más abajo) los apague. La
corrida completa de la suite (`pytest -m "not integration"`) ahora reporta **2 failed** por este archivo —
**esperado, no una regresión** — documentado aquí y en el mensaje de cierre de esta tarea para que no se
confunda con CI roto.

---

## Lista priorizada de fixes (SIN implementar — solo especificados)

### Tocan matemática (λ/probabilidad) — requieren esperar veredicto de D0/ventana O landearse ANTES de que
D0 arranque (D0 sigue "pendiente" — ver nota de régimen al inicio; decisión del dueño, no mía)

1. **[ALTA] Walk-off truncation** (`montecarlo/simulator.py`). Fix conceptual: truncar el sampleo del lado
   que va ganando cuando cruza la ventaja necesaria en la última entrada simulada (o, más simple y menos
   invasivo: aplicar una corrección post-hoc calibrada empíricamente al margen/total simulado, en vez de
   rehacer el sampleo inning-a-inning). Cualquiera de las dos formas mueve p_home/p_over/p_runline — pre-D0
   obligatorio si se decide tocar antes de que la ventana cierre.
2. **[MEDIA] ρ_game casi no-funcional** (`montecarlo/simulator.py`). Fix conceptual: mover la correlación al
   nivel del draw NB final (copula), no solo al ruido de λ — o, más simple, subir sustancialmente
   `lambda_noise`/`rho_game` sabiendo que hoy están, en la práctica, casi apagados. Toca la distribución
   conjunta → pre-D0 igual que el anterior.
3. **[BAJA] Split de empates uniforme 50/50** (`montecarlo/simulator.py`). Fix conceptual: ponderar el split
   por la fuerza relativa de λ en vez de 50/50 fijo. Efecto pequeño (~1pp en el caso real) — probablemente la
   prioridad más baja de las tres, pero técnicamente también mueve p_home.

### Tocan un archivo de la lista congelada pero NO mueven λ/probabilidad/Brier — mismo patrón ya aceptado en
este proyecto para `_SHARPE_SCORE_ANCHOR` (rescale de constante de score, no de motor) — **verificar contra
`docs/PROTOCOLO_CLV_V1.md` antes de asumir que aplica el mismo precedente**, la letra de `CLAUDE.md` nombra
`core/value_detector.py` sin excepción explícita para cambios "solo de UI"

4. **[MEDIA-ALTA] Bootstrap CI subsample fijo a 10K** (`core/value_detector.py::bootstrap_confidence_interval`).
   Fix conceptual: escalar `n_sub` con el n real (o eliminar el cap y aceptar el costo de cómputo — a
   n=5M, 1000 bootstraps sobre el array completo es más caro pero no prohibitivo; alternativa más barata:
   usar la aproximación normal analítica en vez de bootstrap cuando n es grande, que es exacta y O(1)).

### No tocan el pipeline en absoluto — libres de la ventana

5. **[BAJA-MEDIA] Piso de Kelly infla stakes marginales 3.6x** (`config.py::MIN_KELLY` /
   `value_detector.py::kelly_criterion`). No es un bug de código — es una decisión de producto a revisar:
   ¿bajar `MIN_KELLY`, o aceptar el piso como "apuesta mínima operacional" documentada?
6. **[BAJA] Tres conteos de relievers sin coordinar** (`bullpen_engine.py` + `api/mlb_presentation.py`).
   Fix conceptual: exponer los tres números con etiquetas claras en metadata/UI (`n_savant_coverage`,
   `n_siera_coverage`, `n_roster_relievers`) en vez de dejarlos a interpretación.
7. **[Nota, no fix]** `market_penalty` tiene rango dinámico práctico angosto (~0.68-0.72) — no amerita cambio,
   solo quedó documentado por si en el futuro se revisa el composite score completo.

---

## STOP (auditoría original, 2026-07-23)

Cero cambios a código de producción en esta sesión. Únicas escrituras: este reporte
(`audit_20260714/val_audit/reporte.md`) y `tests/verification/test_val_audit_invariants.py` (aditivo,
permanente, 2 tests intencionalmente rojos documentando HALLAZGO-1 y HALLAZGO-2). Los scripts ad-hoc de
verificación quedaron en el scratchpad de la sesión, no en el repo. Los fixes de la lista de arriba están
especificados, no implementados — priorización y decisión de timing (pre/post-D0) quedan con el dueño. Fuera
de alcance confirmado: ciclo de reconciliación live, learning loop/backtest (ya auditados), UI (fixes de
duelo/bullpen de la tarea anterior ya aplicados, independientes de esta auditoría).

---

## Addendum 2026-07-24 — Fixes aplicados (autorizado explícitamente por el dueño: "todo,
incluyendo simulator.py/value_detector.py", tras confirmar que ambos archivos están en la lista congelada
de `CLAUDE.md` y que D0 seguía "pendiente" en `docs/PROTOCOLO_CLV_V1.md`)

Se implementaron los 6 fixes de la lista priorizada arriba. **Esto SÍ mueve λ/probabilidad/EV** (fixes 1-3) —
el `engine_commit` congelado (`80d0faea8fae96a1bb11fd18d53cafcb382a1682`) queda superseded por este cambio;
es responsabilidad del dueño decidir si esto reinicia o no el reloj del protocolo dado que D0 seguía pendiente
en el momento del fix (ver nota de régimen al inicio del documento).

### 1-3. `montecarlo/simulator.py` — walk-off, correlación, split de empates

- **Walk-off**: nuevo parámetro `model_walkoff: bool = True`. Binomial-thinning de `home_runs` en
  "innings 1-8" / "9no inning" a `WALKOFF_9TH_SHARE=1/9` (asunción uniforme, documentada como tal — no hay
  una constante empírica por-inning en el repo, a diferencia de `F5_SCALE`); descarta la porción del 9no
  cuando el home ya iba estrictamente arriba con solo los primeros 8. `away_runs` nunca se trunca.
- **Correlación**: la implementación anterior perturbaba λ_noise con `rho_game` (~5% de λ en magnitud,
  swamped por la varianza propia del draw NB — atenuación >10x medida en el hallazgo original). Reemplazada
  por un mecanismo Gamma-Poisson: `NB(r,λ) ≡ Poisson(Gamma(r,λ/r))` exacto; home/away comparten una unidad de
  aleatoriedad Gamma con probabilidad `_p_share` (derivada en forma cerrada de `rho_game`/λh/λa vía
  `Var(-ln U)=π²/6` y `Cov(-ln U,-ln(1-U))=1-π²/6`, resultados conocidos), antitético para `rho_game<0`. Cada
  marginal queda exactamente `NB(r,λ)` sea cual sea la rama (compartida o no, suman exactamente
  `NB_DISPERSION` unidades). Un enfoque exacto por cópula Gaussiana (`nbinom.ppf`/`gamma.ppf`) se probó
  primero y sí da la correlación objetivo con precisión, pero **~22x más lento** — un run de 5M sims pasó de
  ~6s a ~34s (benchmark real). El mecanismo Gamma-share final corre en ~1.3-3.5s para 5M sims (comparable o
  mejor que el baseline pre-fix) y queda dentro de ~10-15% del `rho_game` objetivo (medido: target −0.008 →
  observado entre −0.0071 y −0.0085 según λh/λa, vs. el mecanismo anterior que daba solo −0.0007).
- **Split de empates**: reemplazado el 50/50 fijo por crédito proporcional a `lh_noise/(lh_noise+la_noise)`
  por simulación empatada (acumulado, no un draw extra) — `p_home+p_away` sigue sumando exactamente 1.0 por
  construcción (partición, no un sorteo independiente).

### 4. `core/value_detector.py::bootstrap_confidence_interval` — CI honesto

Detecta datos binarios (0/1 — el único tipo que cualquier call site real pasa hoy) y genera las medias
bootstrap directamente vía `Binomial(n_real, p_hat)/n_real` — exacto en distribución para ese caso, O(1) en
vez de O(n_bootstrap×n_sub), y usa el **n real** de la corrida en vez de un cap fijo de 10,000. Fallback sin
cambios para el caso no-binario (no ejercido hoy por ningún caller).

### 5. Transparencia del piso de Kelly (SIN tocar `MIN_KELLY`)

`unfractional_kelly()` (nueva función) expone el Kelly completo sin fraccionar ni recortar. Cada mercado en
`analyze_market_generic()` ahora incluye `kelly_unfractional` y `kelly_floor_applied` (bool) junto al `kelly`
ya existente (sin cambios) — el valor de negocio `MIN_KELLY=0.01` NO se tocó, es una decisión del dueño
pendiente, no un bug de código.

### 6. `bullpen_engine.py` — tercer conteo de relievers expuesto

Nuevo campo `n_reliever_ids` (tamaño del roster clasificado como reliever, `None` si la clasificación falló)
junto a `n_pitchers`/`n_siera_pitchers` ya existentes, con comentarios inline distinguiendo qué mide cada uno.
Aditivo puro, cero cambio a `total_mult`/`quality_mult`/`raw_mult`. Espejado en
`frontend/src/api/types.ts::BullpenAdjustment`.

### Validación

- **`tests/verification/test_val_audit_invariants.py`: 17/17 verdes** (antes: 15 verdes + 2 rojos
  intencionales). El test de HALLAZGO-1 se actualizó para ejercitar la fórmula real del fix (constante
  `WALKOFF_9TH_SHARE` importada, no reimplementada) — la brecha real-vs-modelo bajó de ~8.7pp a ~0.9pp. El
  test de HALLAZGO-2 pasa sin tocar su umbral. El test de "respuesta conocida" (VAL-1.5) se actualizó para
  usar `model_walkoff=False` (aísla el mecanismo NB puro que valida) y la nueva regla de empate proporcional
  en su fórmula analítica de referencia — sigue coincidiendo dentro de 5·SE.
- **Suite completa (`pytest -m "not integration"`): 653 passed, 0 failed** (antes del fix: 651 passed + 2
  rojos intencionales). 5 tests preexistentes (`test_montecarlo.py`×4, `test_validated_fixes.py`×1) asumían
  `E[home_runs]≈λ_home` exacto — verdadero bajo el mecanismo NB puro, intencionalmente falso ahora que el
  walk-off trunca algunas simulaciones. Actualizados para pasar `model_walkoff=False` (aíslan el mecanismo
  NB/correlación que esos tests específicamente verifican, no la regla de walk-off) — no se relajó ningún
  umbral, se corrigió el alcance de qué mecanismo cada test ejercita.
- **Sanity check con juego real** (`run_module(persist=False)`, hoy): anchos de `prob_ci` cayeron de ~0.019-
  0.020 a ~0.002 (10x más ajustado, consistente con el fix de CI); `bullpen_away` expone
  `n_reliever_ids=8, n_pitchers=2, n_siera_pitchers=7` simultáneamente; `kelly_unfractional`/
  `kelly_floor_applied` presentes y consistentes en moneyline. El juego real de hoy no tuvo runline/total con
  EV positivo grande (movimiento normal día a día del mercado, no comparable 1:1 contra el caso del
  2026-07-23) — la dirección del efecto del fix (EVs de runline/total dejan de estar sistemáticamente
  inflados) es la verificada en el reporte original vía datos históricos (881 juegos), no este único juego
  del día.
- **Performance**: peor caso (5,000,000 sims sin early-stop) pasó de ~6-7s (antes) a ~3.5s (después) — el
  mecanismo Gamma-share es más rápido que el sampling NB directo original, no solo "no más lento".

## STOP (addendum)

Cambios de producción: `modules/baseball_module/montecarlo/simulator.py`,
`core/value_detector.py`, `modules/baseball_module/context_engine/bullpen_engine.py`,
`frontend/src/api/types.ts` (aditivo), `tests/test_montecarlo.py` + `tests/test_validated_fixes.py`
(actualización de oráculos, no relajación), `tests/verification/test_val_audit_invariants.py` (actualización
de 2 oráculos + 1 nuevo import). Ningún archivo de `backtest_and_retrain.py` ni los builders PIT fue tocado —
fuera del alcance de esta auditoría y no mencionado en la autorización. **Pendiente, fuera de esta sesión**:
re-correr `backtest_and_retrain.py --season 2024,2025 --use-full-pit` para obtener el nuevo baseline Brier/
accuracy post-fix (el walk-off/correlación/tie-split cambian la distribución de probabilidad que ese backtest
mide) y decidir con el dueño si esto cuenta como el "cambio de motor a media ventana" que
`docs/PROTOCOLO_CLV_V1.md` dice que reinicia la muestra primaria — dado que D0 seguía pendiente al momento del
fix, es defendible que no, pero es una decisión de protocolo, no técnica, y no me corresponde tomarla
unilateralmente.
