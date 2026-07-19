# Verificación operativa — sweep read-only, 2026-07-19

**Modo**: 100% read-only. Cero código modificado, cero cron instalado, cero commits (excepto
este reporte). Única escritura permitida y realizada: este directorio. Los cambios de código
hechos ANTES de esta orden (fix de formato EV, fix de game_date mislabel, parche de
closing-lines/lead-time) quedaron intactos, sin commitear, sin tocar más durante este sweep.

## Tabla resumen

| # | Claim | Veredicto | Severidad | Implicación fase 2 |
|---|---|---|---|---|
| V1 | "pct_pinnacle_present=0% porque la key estaba apagada" | **REFUTADO** — la key SÍ funcionaba; el 0% original no está re-explicado, y el 45.6% actual está contaminado por mis propias pruebas de hoy | Alta (integridad del monitor) | El monitor no es evaluable limpiamente ahora mismo — necesita que `run_module()` tenga un modo de prueba real |
| V2 | "ningún decimal odds pasa de 8x" vs. `odds_decimal=25.0` | **NO EVALUABLE** para el caso específico (dato histórico ya no existe) — pero se confirma un bug real y general, independiente del caso puntual | Alta (bug de mapeo confirmado, alcance general) | `get_best_odds_for_teams()` necesita desambiguación por fecha/game_pk antes de confiar en cualquier odds shopping |
| V3 | "el fix de formato es display-only, nada más mezcla unidades" | **VERIFICADO** — mapeo completo, un único patrón consistente en toda la app, ninguna otra mezcla encontrada | — | Cerrado, sin acción pendiente |
| V4 | "el patrón `game_date[:10]` en PIT es cosmético/conservador" | **REFUTADO** — es leak-risk, confirmado empíricamente contra la API real para juegos reales del backtest 2024 | **Crítica** | Requiere su propio paso con gate de identidad/delta — el más serio de todo este sweep |
| V5.1-5.4 | crontab vacío, 0 picks, sin locks, closure doc consistente | **VERIFICADO**, los 4 | — | Ninguna |
| V5.5 | Claims operativos de MEMORY.md | **VERIFICADO**, sin contradicciones | — | Ninguna |
| V5.6 | Suite 523/523 | **VERIFICADO** en HEAD committeado (528 con mis cambios sin commitear, 1 fallando — ver nota) | Baja | Terminar de depurar ese test antes de commitear la fase 2 |

---

## V1 — El alert del monitor, sin explicación

### V1.1 — `calibration_health()` re-corrido ahora

```
{'pct_platt_active': 100.0, 'pct_pinnacle_present': 45.588235294117645, 'n_rows': 68, 'window_days': 14}
```
Sin warning esta vez (ambos porcentajes ≥ `_CALIBRATION_HEALTH_PCT_THRESHOLD=5.0`, `n_rows=68 ≥ _CALIBRATION_HEALTH_MIN_ROWS=20`).

### V1.2 — Distribución temporal

```sql
SELECT game_date, COUNT(*) AS n, SUM(ml_home_pin IS NOT NULL) AS con_pin
FROM game_outcomes WHERE source='live' GROUP BY game_date ORDER BY game_date DESC LIMIT 25
```
```
2026-07-21   7   0
2026-07-20   8   0
2026-07-19  19  19
2026-07-18  12  12
2026-07-12  15   0
2026-07-07   1   0
...
2026-05-16  19   0
```

### V1.3 — ¿El dry-run persiste rows? SÍ, con cita exacta de línea

`modules/baseball_module/core/run_module.py:886` — `_learning.record_prediction(...)` se llama
**incondicionalmente**, sin ningún parámetro `dry_run` en `run_module()` (`grep dry_run` en ese
archivo: cero resultados). `run_daily_picks.py --dry-run` solo gatea el `db.publish_pick()` de
`track_record/publisher.py` — la llamada real a `run_mlb()`/`run_module()` (y por tanto a
`record_prediction()`) ocurre siempre, dry-run o no.

**Confirmado con timestamps exactos** — TODAS las 46 filas con `game_date >= 2026-07-18` fueron
creadas HOY:
```
824414  2026-07-18  created_at=2026-07-19 01:25:32  ml_home_pin=4.48
...  (30 filas más, created_at entre 01:25:32 y 01:31:36 — coincide exactamente con mis
       pruebas de run_module()/dry-runs de esta sesión)
824410  2026-07-20  created_at=2026-07-19 14:08:54  ml_home_pin=None
...  (14 filas más, created_at entre 14:08:54 y 14:09:34 — coincide con mi último dry-run
       "limpio" antes de la interrupción del usuario)
```

### V1.4 — Veredicto: ninguna de las tres opciones del menú original, una cuarta

- **(a) descartado**: los rows recientes con pin NO representan tráfico de producción real —
  son mis propias pruebas de diagnóstico de hoy.
- **(b) descartado**: cuando SÍ se fetchea pin, se persiste correctamente (12/12, 19/19 en los
  dos primeros batches) — no hay bug de persistencia del campo en sí.
- **(c) parcialmente cierto pero incompleto**: el dry-run (a nivel `track_record/`) no persiste
  a `picks`, pero SÍ persiste a `game_outcomes` vía la llamada real a `run_module()` que ocurre
  por debajo — el dry-run de este documento y el dry-run de `run_module()` son cosas distintas
  que comparten nombre.
- **(d) — el veredicto real**: **no existe modo de prueba en `run_module()`**. Cualquier
  invocación, con cualquier propósito (debug, curiosidad, un dry-run "de picks"), escribe una
  fila permanente en la tabla de calibración en vivo, indistinguible de una predicción real
  intencional. El 45.6% actual de `pct_pinnacle_present` es un artefacto de mis pruebas de hoy,
  no evidencia de que el pipeline en producción esté sano bajo tráfico orgánico real. El "0%"
  original tampoco queda explicado por esto — probablemente reflejaba una ventana de 14 días
  con cero actividad real Y cero pruebas, es decir, el sistema simplemente no se había usado,
  no que estuviera roto — pero esto es una hipótesis, no algo que este sweep confirme con
  evidencia directa (esa ventana de tiempo específica no la reconstruí).
- **Contaminación real, no hipotética**: 46 filas de `game_outcomes` (source='live', fechas
  2026-07-18 a 2026-07-21) son ahora datos de prueba permanentemente mezclados con el ledger de
  producción. Los juegos futuros (07-20/07-21) aún no tienen resultado (`home_won` presumibl.
  NULL, no verificado explícitamente aquí) — si se resuelven, alimentarán Kalman/team-bias como
  si fueran predicciones reales intencionales.

---

## V2 — Procedencia del `odds_decimal=25.0`

### V2.1 — Reconstrucción del sweep original de "≤8x"

El sweep original corrió en memoria (Python inline vía Bash), sin persistir a disco — los logs
de ese comando específico ya no existen en el scratchpad de `/tmp` (confirmado: no hay archivo
de esa sesión particular). Lo que persiste y usé para esta verificación: `.cache/odds_last.json`
(mtime 2026-07-19 10:21, POSTERIOR al sweep original) y la MLB Stats API en vivo (schedule
histórico, no expira).

### V2.2 — Rastreo del 25.0

`game_pk=824979` = Athletics (home) vs Washington Nationals (away), `officialDate=2026-07-18`
(confirmado vía `/api/v1/schedule?gamePk=824979`). El `.cache/odds_last.json` actual (10:21 AM
hoy) SÍ tiene un evento Athletics/Washington Nationals — pero con `commence_time` de
**2026-07-19T20:06:00Z** (un juego DISTINTO, un día después — consistente con ser una serie de
varios juegos entre los mismos dos equipos). Los 30 bookmakers de ESE evento están sanos:
Athletics 2.18–2.38, Nationals 1.60–1.71 — ninguno cerca de 25.0.

**No pude recuperar el snapshot exacto de odds del 07-18** (momento en que corrí el diagnóstico
original) — los datos de odds son efímeros, sin archivo por timestamp más allá del último
`_last.json` sobrescrito. El caso puntual queda **NO EVALUABLE** con la evidencia disponible.

### V2.3 — Grep de conversiones american↔decimal alcanzables

Único call site en todo el repo: `track_record/publisher.py:266`,
`_american_to_decimal(odds_raw) if odds_raw and abs(odds_raw) >= 100 else float(odds_raw)`.
- **No alcanzable desde el cómputo de EV real**: `core/value_detector.py` nunca llama a esta
  función; sus odds ya vienen decimales desde `odds_fetcher.py` (`oddsFormat=decimal` en la
  request a la API, confirmado en `odds_fetcher.py:187`).
- **No aplica al caso puntual**: `25.0 < 100`, así que el guard `abs(odds_raw) >= 100` nunca
  se activa para este valor — la rama de conversión americana ni se ejecuta. **Descartado** un
  bug tipo REG-027 para este caso específico.

### V2.4 — Hallazgo real, confirmado independientemente del caso puntual

`odds_fetcher.py::get_best_odds_for_teams()` (línea 510-652) — el match de evento es **solo por
nombre de equipo** (`home_team.lower() in g_home.lower()`, línea 530-531), **sin ningún chequeo
de fecha, `commence_time`, o `game_pk`**. El `return {...}` (línea 652) está DENTRO del
`for event in raw:` (línea 523) — se ejecuta en el PRIMER evento que matchee los dos nombres de
equipo, sin comparar contra ningún otro candidato. Si los mismos dos equipos juegan una serie de
varios días (lo normal en MLB — series de 3-4 juegos), y la API devuelve eventos de varios días
de esa serie simultáneamente (los libros postean líneas con días de anticipación), esta función
puede devolver las odds del juego **equivocado** de la serie, sin ningún mecanismo que lo
prevenga o lo detecte.

### V2.5 — Veredicto comprometido

**(a) confirmado como bug real y general** — mercado/lado mal mapeado por ausencia de
desambiguación temporal, reproducible en código, no hipotético. **(b) descartado** con evidencia
directa (guard de magnitud, no alcanzable desde EV). **(c) no evaluable** para el caso puntual
(dato ya no existe), pero irrelevante para la acción a tomar: sea cual sea la explicación exacta
de 824979, (a) es una vulnerabilidad real que debe cerrarse de todos modos — la fase 2 necesita
un filtro de sanidad de odds Y una desambiguación por fecha en `get_best_odds_for_teams()`,
no uno u otro.

---

## V3 — Unidades de `ev`/`ev_pct` de punta a punta

| Sitio | Archivo:línea | Unidad esperada | Unidad real | ¿Consistente? |
|---|---|---|---|---|
| Productor: `calculate_ev()` | `core/utils.py:10` | percent (docstring: "5.0 means +5% EV") | percent | ✅ fuente de verdad |
| Productor: `analyze_market_generic()`/`evaluate_value_ultra` | `core/value_detector.py` | percent | percent (confirmado: `ev=30.358` para prob=0.4919, odds=2.65 → `(0.4919*2.65-1)*100=30.36` ✓) | ✅ |
| Umbral de publicación | `config.py:36` `DEFAULT_MIN_EV=3.0  # % minimum EV` | percent | percent | ✅ |
| Tiers (ULTRA/HIGH/MEDIUM/SLIGHT) | `core/value_detector.py:44-46,388-394` (`ULTRA=7.25`, comparado directo contra `ev`) | percent | percent | ✅ |
| Kelly | independiente, no deriva de `ev` | fracción 0-1 | fracción 0-1 | ✅ (no mezcla con ev) |
| UI principal — `ui/mlb.py:366` | `f"{ev:+.1f}%"` (literal, NO `.1%`) | percent | percent | ✅ — **nunca tuvo el bug** |
| `ui/mlb.py:97` (guarda en `PredictionsDB`) | `ev = ev_pct / 100.0` | fracción 0-1 (conversión deliberada) | fracción 0-1 | ✅ conversión correcta y documentada |
| `ui/sidebar.py:93,110` (lee de `PredictionsDB`) | `history_df["ev"] * 100` | multiplica por 100 para mostrar | correcto — consistente con la conversión de arriba | ✅ |
| `track_record/publisher.py`, `run_daily_picks.py`, `track_record/ui.py` (6 sitios) | `f"{ev_pct:.2%}"` | ← **bug, ya corregido en el trabajo previo a este sweep** | — | Corregido, no commiteado aún |

### V3.2 — Respuesta pendiente: ¿app.py / ui/mlb.py tenían el mismo bug?

**NO.** `app.py` no muestra EV directamente en absoluto (grep de `'ev'`/`ev_pct` en `app.py`:
cero resultados). `ui/mlb.py:366` usa `{ev:+.1f}%` — formato literal, no el especificador `%` de
Python — nunca tuvo este bug. La única superficie afectada era `track_record/` (3 archivos, 6
sitios), ya identificados y corregidos antes de este sweep.

### V3.3 — Veredicto

**Consistente de punta a punta**, con la tabla de arriba como prueba. El único punto de mezcla
real (`ev_pct` percent → `ev` fracción en `ui/mlb.py:97`) es una conversión deliberada, correcta,
y ya usada consistentemente por su único consumidor (`ui/sidebar.py`). Test de unidades
propuesto (sin implementar): un test que verifique `calculate_ev(0.5, 2.0) == 0.0` (punto de
equilibrio exacto) y `calculate_ev(0.6, 2.0) == pytest.approx(20.0)` — ancla la convención
"percent, no fracción" en el productor, para que un cambio futuro accidental la rompa en CI.

---

## V4 — Mapa de `game_date[:10]` en el pipeline PIT y backtest

### V4.1 — Enumeración completa

| Archivo:línea | Consumidor | Qué alimenta |
|---|---|---|
| `data_fetchers.py:1132` (`get_team_days_rest`) | Live, contexto de descanso | Cálculo de días de descanso del equipo antes del juego actual |
| `data_fetchers.py:2052` (`game_date_str`) | Live enrichment | Lookup de % bateo zurdo/derecho por fecha |
| `modules/baseball_module/advanced_pit_enrichment/team_defense_pit_builder.py:337` | PIT Defense | Cutoff walk-forward para snapshot de defensa |
| `modules/baseball_module/advanced_pit_enrichment/bullpen_pit_builder.py:192` | PIT Bullpen | Cutoff walk-forward para snapshot de bullpen |
| `modules/baseball_module/advanced_pit_enrichment/tte_daily_snapshot_builder.py:152` (`previous_day_cutoff_for_game_date`) | PIT TTE (ofensa) | Cutoff walk-forward para snapshot de equipo/TTE |
| `modules/baseball_module/core/run_module.py:883` (`_gd`) | Live → persistencia | **Fuente original de la contaminación** — se escribe a `game_outcomes.game_date` |
| `backtest_and_retrain.py:696` (`_prediction_cutoff_for_row`) | Backtest, cutoff general | Fallback (siempre se activa — ver V4.3) |
| `backtest_and_retrain.py:705-708` (`_experimental_pitcher_pit_cutoff_for_row`) | Backtest PIT Pitcher | Cutoff walk-forward |
| `backtest_and_retrain.py:711-716` (`_team_tte_pit_cutoff_for_row`) | Backtest PIT TTE | Cutoff walk-forward |
| `backtest_and_retrain.py:719-724` (`_defense_pit_cutoff_for_row`) | Backtest PIT Defense | Cutoff walk-forward |
| `backtest_and_retrain.py:727-732` (`_bullpen_pit_cutoff_for_row`) | Backtest PIT Bullpen | Cutoff walk-forward |

### V4.2 — Prueba empírica de contaminación real (no hipotética) en el backtest 2024

```sql
SELECT game_pk, game_date, home_team, away_team FROM game_outcomes
WHERE season=2024 AND source='backtest' AND home_team LIKE '%Mariners%' ...
```
vs. MLB Stats API (`/api/v1/schedule?gamePk=<pk>`):

| game_pk | `game_outcomes.game_date` | `gameDate` (UTC, API) | `officialDate` (API, real) | ¿Contaminado? |
|---|---|---|---|---|
| 745199 | **2024-09-19** | 2024-09-19T01:40:00Z | **2024-09-18** | **SÍ — off by 1 día** |
| 745205 | **2024-09-18** | 2024-09-18T01:40:00Z | **2024-09-17** | **SÍ — off by 1 día** |
| 745201 | 2024-09-29 | 2024-09-29T19:10:00Z (día) | 2024-09-29 | No (juego de día, sin cruce de medianoche UTC) |

**Confirmado con datos reales del backtest**, no solo con lectura de código: `game_outcomes.game_date`
usa el timestamp UTC crudo (`gameDate`), no el día de calendario real del schedule
(`officialDate`) — para CUALQUIER juego nocturno que cruce medianoche UTC (la norma para
equipos de la costa oeste, y para muchos juegos nocturnos del este también).

### V4.3 — Clasificación por dirección, con razonamiento explícito (no la hipótesis global)

**Los 5 cutoffs de PIT en `backtest_and_retrain.py` y los 3 builders PIT — LEAK-RISK,
confirmado, no hipotético.** Razonamiento del cutoff específico:

Para `game_pk=745199` (real: officialDate=09-18, contaminado: game_date=09-19):
- Cutoff CORRECTO que debería usarse: `(09-18 00:00:00) - 1s` = **2024-09-17 23:59:59 UTC**.
- Cutoff que el código REALMENTE computa: `(09-19 00:00:00) - 1s` = **2024-09-18 23:59:59 UTC**.
- El cutoff contaminado es **un día completo MÁS TARDE** que el correcto — no conservador, no
  cosmético: **el snapshot PIT para este juego tiene acceso a todo el día 09-18 completo**,
  el mismísimo día en que el juego se jugó, que debería estar excluido del walk-forward cutoff.

Esta dirección (UTC empuja el juego HACIA ADELANTE en el calendario para juegos nocturnos) es
la dirección que SÍ produce leak, no la conservadora que se había asumido/insinuado antes de
verificarlo con evidencia. La hipótesis "UTC → conservador" queda **refutada por prueba
directa**, sitio por sitio, para los 5 sitios de `backtest_and_retrain.py` y los 3 builders PIT.

`data_fetchers.py:1132` (días de descanso) y `data_fetchers.py:2052` (bateo zurdo/derecho) —
**cosmético/negligible**: afectan solo enriquecimiento de contexto en vivo (no cruzan al futuro
respecto al propio historial del equipo), y su impacto (±1 día en conteo de descanso, o mirar la
composición de bateo de un día adyacente) es de magnitud despreciable frente al leak-risk de
arriba.

### V4.4 — STOP

Cero fixes en este sweep, tal como se pidió. Este hallazgo (leak-risk confirmado en 5+3 sitios,
afectando la integridad walk-forward de TODO el backtest histórico para juegos nocturnos/costa
oeste) es candidato a su propio paso dedicado, con gate de identidad/delta — dado el tamaño del
blast radius (podría mover el Brier/accuracy del baseline canónico actual, 0.24483/55.51%, en
cualquier dirección), no debe tocarse sin ese proceso completo.

---

## V5 — Sweep de claims operativos

| # | Claim | Comando | Resultado | Veredicto |
|---|---|---|---|---|
| 1 | crontab vacío | `crontab -l` | `no crontab for raulio` | VERIFICADO |
| 2 | `track_record.db::picks` = 0 rows | `SELECT COUNT(*) FROM picks` | `0` | VERIFICADO — los dry-runs no escribieron al ledger real de track_record |
| 3 | Sin lock de ml_state, sin proceso backtest colgado | `find *.lock`, `ps aux` | vacío ambos | VERIFICADO |
| 4 | `15_remediation_closure.md` post-13549f4 consistente | `git show HEAD:...` | sin más menciones de "reactivar la key" | VERIFICADO |
| 5 | Claims operativos de MEMORY.md | ver detalle abajo | sin contradicciones encontradas | VERIFICADO |
| 6 | Suite 523/523 | `pytest -q` | **528 total, 527 passed, 1 failed** (HEAD committeado = 523 limpio; +5 tests de mi trabajo sin commitear; 1 de esos 5 tiene un fallo intermitente sin resolver — interrumpido por esta orden antes de terminar de depurarlo) | VERIFICADO para HEAD; nota abierta para mi propio trabajo en curso |

### V5.5 detalle — verificación línea por línea de MEMORY.md

- "picks table still has 0 rows" → VERIFICADO (arriba).
- "run_daily_picks.py cron not yet running" → VERIFICADO (crontab vacío).
- "current baseline Brier 0.24483/55.51%" → VERIFICADO: `audit_20260714/paso5c_gate/backtest_report_20260718_1452.json`
  (commiteado, `git ls-files` lo confirma) contiene exactamente `accuracy_pct=55.51, brier_model=0.24483`.
- "ODDS_API_KEY confirmed ACTIVE" → VERIFICADO (ya establecido antes de este sweep, re-confirmado:
  `.env` tiene la key, `.cache/odds_last.json` se sigue actualizando en vivo durante este mismo sweep).

---

## Resumen para fase 2 (sin implementar nada aquí)

1. **V4 (leak-risk en PIT walk-forward)** — el hallazgo más serio. Necesita paso propio,
   gate de identidad/delta, antes de tocar cualquier otra cosa que dependa del baseline actual.
2. **V2 (odds shopping sin desambiguación de fecha)** — bug real confirmado, arreglo necesario
   independientemente de si explica el caso puntual de 824979.
3. **V1 (sin modo de prueba en `run_module()`)** — mientras no exista, cualquier verificación
   futura de `calibration_health()` seguirá contaminada por actividad de prueba. Vale la pena
   un flag explícito (p. ej. `persist_prediction: bool = True`) antes de confiar en el monitor.
4. Mi propio trabajo sin commitear (fix de formato EV, fix de game_date mislabel, parche de
   lead-time/closing-lines) sigue intacto y sin tocar — pendiente de terminar de depurar el test
   flaky (`test_game_far_enough_out_is_analyzed`) antes de commitear, y de decidir si se
   commitea antes o después de abordar V4/V2 dado que este sweep encontró issues más grandes.
