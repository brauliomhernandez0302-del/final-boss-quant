# PASO 5c.1 — Alineación de métrica completa (barrel term), live vs. PIT adapter

**Fecha**: 2026-07-18. Antes de tocar código — citas exactas de ambos lados.

## Lado LIVE — `modules/baseball_module/offense/true_talent_engine.py`

```python
# true_talent_engine.py:70
LG_BARREL_PA = 0.088             # league barrel per PA
```

```python
# true_talent_engine.py:646-669
pa_cur       = hitting_cur.get("pa",     0.0)
attempts_cur = sc_cur.get("total_attempts", 0.0)
...
barrel_cur = sc_cur.get("barrel_pa",   LG_BARREL_PA)
...
barrel_reg = _regress(barrel_cur, LG_BARREL_PA, attempts_cur, _K_BARREL)
```

Y el origen de `sc_cur["barrel_pa"]` (pese al nombre, confirmado NO per-PA):

```python
# true_talent_engine.py:462,469 (_aggregate_statcast_for_team)
# Barrels per batted-ball-event (attempts), NOT per PA — LG_BARREL_PA=0.088
# ... aggregation already divides by attempts — this aligns the two.
"barrel_pa": round(barrels_sum / attempts_sum, 4) if attempts_sum > 0 else LG_BARREL_PA,
```

`_K_BARREL = 120` (`true_talent_engine.py`, mismo valor que `tte_formula.py`'s comentario de
unificación cita como ya compartido).

**Resumen live**: numerador = `barrels_sum`, denominador = `attempts_sum` (batted-ball-events
reales, del leaderboard público de Savant), prior = `0.088` (constante etiquetada "per PA" en
el comentario pero confirmada per-*attempt* por el propio comentario de la función que la
produce), K = `120`, n (muestra de shrinkage) = `attempts_cur`. **El nombre del campo/constante
("barrel_pa") es un residuo histórico engañoso en TODO el codebase — el valor real siempre fue
per-attempt del lado live.**

**Nota importante sobre el prior-season (`barrel_pri`)**: no se re-regresiona — se usa crudo
(`barrel_reg=barrel_pri` pasado directo a `season_lambda()`, línea 690-694, sin llamada a
`_regress()`). Solo el componente de temporada ACTUAL se regresiona. El adaptador PIT ya replica
esta misma estructura (ver abajo) — no requiere cambio en ese aspecto.

## Lado PIT — `modules/baseball_module/advanced_pit_enrichment/tte_pit_adapter.py` (ANTES del fix)

```python
# tte_pit_adapter.py:35-45 (comentario 2026-07-11)
# Re-centered 2026-07-11: this adapter reads the per-PA `barrel_pa` field
# (barrel_count / plate_appearances), but the constant was 0.088 ...
# Real per-PA league mean from savant.team_offense.prior_baseline (walk-forward,
# 2024: 0.0544, 2025: 0.0532).
LG_BARREL_PA = 0.054
...
barrel_cur = float(_first_present(snapshot, "team_barrel_pa", "barrel_pa"))
...
barrel_prior = float(snapshot["barrel_pa_prior"])
...
barrel_reg = _shared_regress(barrel_cur, lg_barrel_pa, pa, K_BARREL)
```

Y el origen del campo `barrel_pa` que lee (`savant_offense_daily_aggregator.py`):

```python
"barrel_pa": _rate(self.barrel_count, self.plate_appearances),
```

**Confirmado: el adaptador es una métrica per-PA de punta a punta** — numerador
`barrel_count/plate_appearances`, prior `0.054` (derivado él mismo per-PA, walk-forward desde
`prior_baseline`), n=`pa`. Es **internamente consistente**, pero mide algo **dimensionalmente
distinto** a lo que live mide. Esto confirma la señal de alerta del roadmap: el sufijo `_pa` no
era un error de nombre en este caso particular — describía con precisión lo que el campo hace,
que es precisamente el problema (es la métrica equivocada para lograr paridad con live).

## Veredicto — el fix debe cambiar métrica + prior + n juntos

Cambiar solo `n` (de `pa` a `bip`/`batted_ball_count`) dejando `barrel_cur`/`lg_barrel_pa` como
tasas per-PA sería, en efecto, regresionar una tasa per-PA hacia un prior per-PA usando un
tamaño de muestra basado en attempts — dimensionalmente parcheado pero NO equivalente a lo que
hace live, y el propio `season_lambda()` recibiría un `barrel_reg`/`lg_barrel_rate` per-PA
mientras el resto del pipeline (vía la constante compartida) asume la escala per-attempt de
live — no hay garantía de que el resultado sea comparable. **Paridad con live exige cambiar los
tres a la vez**, ya disponible sin tocar el snapshot builder:

- `team_brl_percent`/`brl_percent` (ya en el snapshot, YA corregido por el commit 5b — su
  denominador es `batted_ball_count`, exactamente `attempts` post-fix) dividido entre 100 =
  numerador per-attempt equivalente a `barrels_sum/attempts_sum`.
- `brl_percent_prior` (ya en el snapshot, mismo origen para la temporada previa) dividido entre
  100 = equivalente a `barrel_pri` de live.
- `bip`/`bip_prior` (ya en el snapshot, alias de `batted_ball_count`, YA corregido por 5b) como
  `n` de la regresión de temporada actual — igual que `attempts_cur` en live.
- `LG_BARREL_PA`: cambia de `0.054` (per-PA) a `0.088` (per-attempt, valor exacto de
  `true_talent_engine.py:70`). **No se importa directamente** — `test_no_live_backtest_or_run_module_imports`
  (`tests/test_tte_pit_adapter.py:187-195`) prohíbe explícitamente que este módulo importe
  `true_talent_engine` (aislamiento intencional de orquestación). Se hardcodea el mismo valor
  con un comentario citando la fuente exacta — mismo patrón de duplicación conocida que
  `tte_formula.py` ya documenta como riesgo aceptado (no unificado, solo la MATEMÁTICA se
  unificó en esa refactorización, no las constantes de liga).
- `K_BARREL = 120`: **sin cambio** — ya coincide con `_K_BARREL` de live. No se retunea.

Live tampoco fue impreciso, la premisa se confirma en el sentido inverso a la advertencia del
roadmap: **live SÍ es per-attempt** (confirmado por su propio comentario de código en la función
que produce `barrel_pa`), no hay imprecisión que reportar de ese lado — el único lado con
métrica-mezclada era el adaptador, y por una razón históricamente entendible (el fix de
2026-07-11 solo tenía el prior baseline per-PA disponible en ese momento, porque
`batted_ball_count`/`bip` todavía estaba roto — exactamente la razón por la que este roadmap
exigió resolver 5b primero).
