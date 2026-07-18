# PASO 5b.1 — Precondiciones (backups, gap de prior_baseline, consumers de _aggregate_common)

**Fecha**: 2026-07-18. Investigación read-only sobre el estado del repo (el fix de código de
5b.2 ya estaba aplicado sin commitear al empezar esta sesión — ver nota al final).

## 1. Backups

Ya existen, mismo patrón de nombres del proyecto:

```
data/pit_cache_2024_backup_pre_math002_20260718_111612.db
data/pit_cache_2025_backup_pre_math002_20260718_111612.db
data/pit_cache_merged_backup_pre_math002_20260718_111612.db
```

## 2. Gap de `savant.team_offense.prior_baseline` — RESUELTO, salida (i)

**El mecanismo SÍ existe** — el diagnóstico 5a no lo encontró porque buscó un script
*incremental* dedicado (no existe) pero el mecanismo real es un build de una sola pasada:

- `TTEPriorBaselineBuilder.persist_prior_baseline()`
  (`modules/baseball_module/advanced_pit_enrichment/tte_prior_baseline_builder.py:104`) llama
  a `SavantOffenseRollingBuilder.build_teams_for_as_of_date()` con la ventana completa de la
  temporada previa en una sola invocación (no es un walk-forward de N cutoffs — un prior
  baseline es un solo snapshot por equipo por temporada, no una serie).
- Invocado por `scripts/build_all_pit_caches.py::build_tte_prior()` (línea 77-95), a su vez
  parte de `main()` con `--target-season <N>` (usa `SEASON_WINDOWS` para resolver las fechas
  de la temporada previa, `data/pit_raw/raw_savant_{prior}.db`, offline).
- **Comando de rebuild**: `python3 scripts/build_all_pit_caches.py --target-season 2024
  --skip-defense --skip-bullpen` y lo mismo para `--target-season 2025`. Esto reconstruye
  TAMBIÉN `savant.team_offense.rolling` (ver hallazgo del punto 3 abajo — usa un code path
  DISTINTO al incremental) en la misma corrida; no hay flag para aislar solo el prior baseline
  dentro de "TTE", así que ambos se reconstruyen juntos. No hay problema: ambos ya están
  parcheados (ver punto 3).

## 3. Consumers de `_aggregate_common()` — hallazgo que corrige a 5a

**Pregunta del paso**: ¿solo builders de ofensa, o compartido con pitchers?

**Respuesta confirmada por grep** (`grep -rln "_aggregate_common"`): únicamente
`savant_offense_daily_aggregator.py` (el propio módulo) + su re-export en `__init__.py`.
**No hay pipeline de pitchers que importe o llame `_aggregate_common()`** — está limpio en
ese sentido específico, tal como asumía 5a.

**Pero 5a se equivocó en una premisa más fuerte**: afirmó que `_aggregate_common()` es "la
única función que produce `batted_ball_count`" en el path de ofensa. Es falso — existe una
**segunda reimplementación completamente independiente**, `_RawTeamAccumulator.add_event()`
(mismo archivo, línea ~536), con el **mismo bug de foul-inflation de forma independiente**
(constatado: el filtro `if event.launch_speed is not None` no estaba scoped a PA-terminal
antes del fix de esta sesión). Esta clase se usa en un code path real y activo:

- `SavantOffenseRollingBuilder.build_teams_for_as_of_date()` →
  `SavantOffenseDailyAggregator.aggregate_teams_rolling_by_date_range()` → `_RawTeamAccumulator`
  — este es exactamente el mecanismo que resuelve el gap del punto 2
  (`TTEPriorBaselineBuilder` y `scripts/build_all_pit_caches.py::build_tte_rolling()` lo usan).

Es decir: **`savant.team_offense.prior_baseline` y una vía alterna (no-incremental) de
`savant.team_offense.rolling` dependían de `_RawTeamAccumulator`, no de `_aggregate_common()`**,
y tenían el bug de foul-inflation de forma completamente separada. El fix ya presente en el
working tree (sin commitear al iniciar esta sesión) corrige ambas implementaciones de forma
consistente (mismo criterio: `is_pa_event = _is_plate_appearance_event(event)` antes de
contar). Verificado línea por línea — ambos fixes son equivalentes en criterio, y
`barrel_count` correctamente se dejó SIN scoping en ambas clases (confirmado en 5a: Statcast
nunca clasifica un foul como barrel, cero contaminación en el numerador).

**Runtime real de `savant.batter.rolling` + `savant.team_offense.rolling`**: se reconstruyen
vía `scripts/build_offense_savant_rolling_incremental.py` (usa `_aggregate_common` +
`_Accumulator`, O(días), ~27 min ambas temporadas, ver 5a). `prior_baseline` se reconstruye
vía `build_all_pit_caches.py` (usa `_RawTeamAccumulator`, O(días²) si se corre para rolling
también, pero el prior baseline en sí es una sola ventana, no walk-forward — rápido).

## 4. Hallazgo fuera de alcance — el mismo bug existe, de forma independiente, en el lado de PITCHERS

**No es el mismo código** (`_aggregate_common()` no es compartido, confirmado arriba), pero es
la **misma clase exacta de bug**, presente en dos implementaciones independientes del lado
pitcher:

1. `modules/baseball_module/advanced_pit_enrichment/savant_daily_aggregator.py:69` —
   `_aggregate_one()`, la función equivalente para pitchers (alimenta `savant.pitcher.rolling`
   vía `scripts/build_pitcher_savant_rolling_incremental.py`): `batted_ball_events = [event for
   event in events if event.launch_speed is not None]` — sin scoping de PA, idéntico patrón.
2. `modules/baseball_module/advanced_pit_enrichment/pitcher_prior_baseline.py:261-262` — el
   accumulator de prior baseline de pitchers: `if event.launch_speed is not None:
   self.batted_ball_count += 1` — mismo patrón, sin scoping.

Ambos NO tocados, tal como exige el alcance de este paso ("NO reconstruyas caches de pitcher
en este paso — repórtalo como decisión pendiente"). **Reporto esto como decisión pendiente del
dueño**: si `bip`/`batted_ball_count` del lado pitcher alimenta algún shrinkage o tasa
consumida hoy por `pitcher_engine.py` (no verificado en este paso, fuera de su alcance),
existiría un MATH-002 análogo del lado pitcher, mismo mecanismo, mismo orden de magnitud de
inflación esperado (~1.9x). Ameritaría su propio diagnóstico 5a-style antes de tocar código.

## 5. Nota sobre el estado del working tree al iniciar esta sesión

El fix de código de 5b.2 (`_aggregate_common()` y `_RawTeamAccumulator.add_event()`) y los
backups del punto 1 ya estaban presentes, sin commitear, al iniciar esta sesión — evidencia de
trabajo de una sesión previa interrumpida antes de documentar este paso o correr el rebuild.
Se verificó línea por línea (no se asumió correcto) antes de continuar: ambos fixes son
consistentes entre sí y con el criterio validado en 5a. Se procede con 5b.3 (rebuild) a partir
de este punto.
