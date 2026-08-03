# Fase 2B — B0: verificación previa (read-only)

**Fecha**: 2026-07-19.

## B0.1 — Agrupación interna de días por builder: LIMPIA en los 3

Los 3 builders (`team_defense_pit_builder.py`, `bullpen_pit_builder.py`,
`savant_offense_daily_aggregator.py`/TTE ofensa) leen del mismo store crudo compartido,
`RawSavantEventsCache` (confirmado en CONTRACTS.md: "the shared raw-event store behind all 4
Savant-derived PIT domains"). La ingesta (`savant_raw_ingestor.py::_fetch_statcast_day()`)
consulta el endpoint de Baseball Savant **por día exacto** (`game_date_gt=day,
game_date_lt=day` — el mismo día en ambos límites), usando el campo `game_date` propio de
Statcast — una fecha de calendario sin componente de hora, la convención oficial de "día de
partido" de Statcast, no un timestamp UTC.

**Verificación empírica** (no solo lectura de código) contra el caso conocido, `game_pk=745199`
(oficial=2024-09-18, contaminado en `game_outcomes.game_date`=2024-09-19):

```sql
SELECT DISTINCT game_date FROM raw_savant_events WHERE game_pk='745199'
-- resultado: ('2024-09-18',)
```

Los eventos crudos de este juego caen exactamente bajo `2024-09-18` — el día oficial correcto,
NO el día UTC contaminado. **Los 3 builders agrupan limpio. Cero rebuilds de cache — 2B es
estrictamente schema + derivación de cutoffs**, tal como el addendum del sweep anticipaba.

## B0.2 — Censo de consumidores de `get_latest()`

Grep de todos los call sites (`self.pit_cache.get_latest(` / `self.cache.get_latest(`):

| Archivo | Dominio |
|---|---|
| `team_defense_pit_builder.py` (×3) | Defense |
| `bullpen_pit_builder.py` (×2) | Bullpen |
| `savant_offense_daily_aggregator.py` (×2) | TTE ofensa (rolling) |
| `tte_daily_snapshot_builder.py` (×4) | TTE ofensa (snapshot compuesto: tte/team/batter/prior) |
| `tte_prior_baseline_builder.py` | TTE ofensa (prior) |
| `team_defense_prior_baseline.py` | Defense (prior) |
| `savant_pit_fetcher.py`, `fangraphs_pit_fetcher.py`, `fangraphs_daily_pit_persistence.py`, `advanced_pitcher_snapshot_builder.py` (×2), `advanced_pitcher_daily_snapshot_builder.py` (×3) | **Pitcher** |

**Veredicto**: hay más call sites de los 3 dominios "conocidos", pero TODOS caen en Pitcher,
Defense, Bullpen o TTE ofensa — los mismos 4 dominios que ya cubren "los 5 cutoffs" del reporte
V4 (Pitcher es uno de los 5: `_experimental_pitcher_pit_cutoff_for_row`). **No hay un dominio
nuevo/desconocido con un mecanismo de contaminación independiente** — Pitcher simplemente tiene
más call sites internos (combina Savant + FanGraphs) pero todos aguas abajo del mismo único
cutoff que `backtest_and_retrain.py` calcula una vez por fila.

Nota fuera de alcance (ya establecida en el roadmap): la limpieza del campo `game_date` de
FanGraphs (fuente adicional del lado pitcher) NO se verificó aquí — el fix de cutoffs de 2B
aplica igual (deriva de `official_date`, no de la fuente interna de cada builder), y cualquier
contaminación de FanGraphs específicamente sería un hallazgo de foul-inflation/pitcher-side,
explícitamente fuera de alcance de este paso.

## B0.3 — Los 5 cutoffs + un 6º sitio cronológico encontrado

| # | Función/sitio | Archivo:línea | Deriva de |
|---|---|---|---|
| 1 | `_prediction_cutoff_for_row` | `backtest_and_retrain.py:690-696` | `row["game_date"]` (fallback si no hay `prediction_cutoff_utc`/`prediction_cutoff`/`as_of_date` — nunca existen esos campos hoy, así que siempre cae aquí) |
| 2 | `_experimental_pitcher_pit_cutoff_for_row` | `backtest_and_retrain.py:699-708` | `row["game_date"]` |
| 3 | `_team_tte_pit_cutoff_for_row` | `backtest_and_retrain.py:711-716` | `row["game_date"]` |
| 4 | `_defense_pit_cutoff_for_row` | `backtest_and_retrain.py:719-724` | `row["game_date"]` |
| 5 | `_bullpen_pit_cutoff_for_row` | `backtest_and_retrain.py:727-732` | `row["game_date"]` |

Call sites de los 5: líneas 837, 932, 1181, 2964, 2984, 3020, 3038, 3056.

**Sexto sitio encontrado, no en la lista original de "los 5" pero de la misma naturaleza**:
`backtest_and_retrain.py:1785` — `_bias_before_date = str(game_data.get("game_date", "")) or
None`, pasado a `learning_engine.py`'s `compute_team_bias`/`compute_team_bias_kalman_adjusted`/
`compute_multidim_bias` como el parámetro `before_date` (el corte walk-forward anti-leak de
CHRON-001: `WHERE game_date < before_date` contra OTROS juegos, para team-bias/Kalman). Esto
**también deriva del campo contaminado** — y el propio punto B2.3 de este roadmap exige
consistencia ("nada de mezclar official en cutoffs PIT y UTC en bias windows"), así que este
sitio entra al alcance de B2 junto con los 5 originales. `game_data` en este contexto es el
dict de trabajo del loop principal del backtest (envuelve la `row` de `game_outcomes` más
campos calculados como `team_tte_pit_requested_as_of_date`) — una vez que B1 agregue
`official_date` a `game_outcomes`, el mismo patrón (`game_data["official_date"] = row["official_date"]`)
se puede sumar donde ya se setean los otros campos derivados de la fila.

La línea 1782 (`_game_month = int(game_data.get("game_date", "")[5:7])`) es, por su propio
comentario, "Diagnostic only" — no alimenta ningún cálculo real de bias (una versión anterior
que sí lo hacía causó un double-count y fue revertida). Se deja fuera del alcance de cambio de
B2 salvo que se decida limpiarla por prolijidad — no es funcionalmente parte del leak.

## Veredicto B0: alcance confirmado, tal como se esperaba

- 3 builders limpios por dentro → cero rebuilds.
- Censo de consumidores → sin dominio nuevo, Pitcher ya estaba entre "los 5".
- 5 cutoffs confirmados con cita exacta + 1 sitio adicional real (`_bias_before_date`) que debe
  entrar al mismo cutover por la misma razón de raíz.
- Continúo a B1 (schema + backfill) tal como estaba previsto, sin desviación de alcance.
