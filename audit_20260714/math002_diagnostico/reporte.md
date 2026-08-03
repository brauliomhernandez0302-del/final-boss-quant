# PASO 5a — Diagnóstico read-only: ¿existe el bug de foul-inflation en `batted_ball_count`?

**Fecha**: 2026-07-18. **Modo**: 100% read-only sobre el proyecto — cero código modificado,
cero caches reconstruidos, cero DB tocada, cero red. Única escritura: este directorio.

## Veredicto (adelantado, desarrollado abajo con evidencia)

**(a) — Bug confirmado tal como lo describe el comentario, y más severo de lo que estimaba.**
`batted_ball_count` en el pipeline PIT de ofensa cuenta cualquier lanzamiento con
`launch_speed` registrado, sin filtrar por si terminó la plate appearance — así que cada foul
tocado (que Statcast sí trackea con exit velocity/launch angle real) se cuenta como un
"batted ball" adicional. En los 16 bateador-temporada muestreados, el valor cacheado
sobreestima el conteo real de eventos de bola bateada en un **90% en promedio** (rango
70.3%–118.8%), y en **los 16/16 casos el `batted_ball_count` cacheado excede el propio
`pa`** — algo imposible para una métrica que debería ser, por definición, ≤ PA.

---

## TAREA 1 — Origen del campo

### 1.1 Dónde y cómo se computa

`modules/baseball_module/advanced_pit_enrichment/savant_offense_daily_aggregator.py`,
función `_aggregate_common()` (única función que produce `batted_ball_count`, compartida por
`_aggregate_batter()` y `_aggregate_team()` — línea 607-630):

```python
# savant_offense_daily_aggregator.py:633-660
def _aggregate_common(events: list[Any]) -> dict[str, Any]:
    pa_events = _plate_appearance_events(events)
    batted_ball_events = [event for event in events if event.launch_speed is not None]
    ...
    plate_appearances = len(pa_events)
    ...
    return {
        "plate_appearances": plate_appearances,
        "batted_ball_count": len(batted_ball_events),
        ...
        "brl_percent": _pct(barrel_count, len(batted_ball_events)),
        "barrel_pa": _rate(barrel_count, plate_appearances),
        ...
    }
```

El filtro `event.launch_speed is not None` (línea 635) se aplica sobre `events`, la lista
**cruda, sin deduplicar por plate appearance** — a diferencia de `pa_events =
_plate_appearance_events(events)` (línea 634), que sí filtra por eventos terminales de PA
(`_is_plate_appearance_event()`, línea 816-822, contra la whitelist `_PA_EVENT_LABELS`,
línea 830-852: `single`, `double`, ..., `strikeout`, `walk`, etc. — 21 labels).

**Confirmado con datos reales** (`data/pit_raw/raw_savant_2024.db`, consulta directa, sin
red): lanzamientos con `launch_speed` no nulo pero `events` vacío (no terminan la PA)
existen y son, exactamente, fouls:

```python
>>> con.execute("""SELECT raw_json FROM raw_savant_events
...     WHERE launch_speed IS NOT NULL AND (events IS NULL OR events = '') LIMIT 5""")
{'type': 'S', 'description': 'foul', 'events': '', 'launch_speed': '66',   'launch_angle': '6'}
{'type': 'S', 'description': 'foul', 'events': '', 'launch_speed': '72.2','launch_angle': '-30'}
{'type': 'S', 'description': 'foul', 'events': '', 'launch_speed': '74',  'launch_angle': '55'}
{'type': 'S', 'description': 'foul', 'events': '', 'launch_speed': '68.8','launch_angle': '-21'}
{'type': 'S', 'description': 'foul', 'events': '', 'launch_speed': '64.4','launch_angle': '-36'}
```

`type='S'` (strike, no ball-in-play) + `description='foul'` — Statcast sí registra exit
velocity/launch angle en fouls tocados (el bate hizo contacto físico real, medible), pero
`events` queda vacío porque el at-bat continúa. `_aggregate_common()` no distingue esto.

### 1.2 Esquema del cache y disponibilidad de la plomería

`data/pit_cache_merged.db`, tabla única `pit_metric_cache` (`namespace`, `entity_id`,
`season`, `as_of_date`, `data_json`, ...). El campo vive en `data_json` bajo la clave
`batted_ball_count` para los namespaces `savant.batter.rolling` (206,575 filas),
`savant.team_offense.rolling`, y — vía `bip_prior` — `savant.team_offense.prior_baseline`
(`tte_prior_baseline_builder.py:162`: `"bip_prior": metrics.batted_ball_count`, misma fuente
contaminada).

**La plomería ya existe, solo está desconectada de MATH-002**: `tte_daily_snapshot_builder.py:125`
ya surface el campo hacia el snapshot que consume el adaptador, bajo el alias `"bip"`:
```python
"bip": _first_present(team_data, "bip", "batted_ball_count"),
```
`tte_pit_adapter.py` simplemente no lo lee como base de shrinkage hoy (usa `pa` para las 4
regresiones, línea confirmada en sesiones previas de esta auditoría) — no es que falte
fetchear el dato, es que el dato que ya llega está contaminado, y por eso nadie lo conectó.

---

## TAREA 2 — Definición canónica del motor LIVE

`modules/baseball_module/offense/true_talent_engine.py::_fetch_savant_exitvelo()`
(línea 172-209) — el motor en vivo **no recomputa `attempts` desde eventos crudos**: lo
toma verbatim del leaderboard público de Baseball Savant:

```python
# true_talent_engine.py:181-203
csv_text = _get_csv(
    f"{SAVANT_BASE}/leaderboard/statcast",
    params={"year": season, "type": "batter", "min": "1", "csv": "true"},
)
...
for row in reader:
    pid      = int(row.get("player_id", 0))
    attempts = float(row.get("attempts", 0) or 0)
    ...
    result[pid] = {
        "barrels":  float(row.get("barrels",    0) or 0),
        "brl_pa":   float(row.get("brl_pa",     0) or 0),
        "attempts": attempts,
        ...
    }
```

Y en el punto de uso (línea 663-669):
```python
# barrel_cur is a per-batted-ball-event (attempts) rate, not per-PA —
# its shrinkage sample size must be attempts_cur, not pa_cur (PA always
# exceeds attempts since walks/Ks/HBP are plate appearances but not
# batted-ball events).
barrel_reg = _regress(barrel_cur, LG_BARREL_PA, attempts_cur, _K_BARREL)
```

**Implicación importante para el diagnóstico**: la definición "canónica" no es una función
Python en este repo que yo pueda comparar línea a línea — es lo que el endpoint público de
Savant define internamente como "attempts" (su propia agregación oficial de batted-ball
events para el leaderboard de barrels). El motor live confía en ese número ciegamente, sin
re-filtrar. Por diseño, esto hace que la Tarea 3 no pueda comparar "función live vs función
PIT" directamente — en su lugar, uso como proxy la definición operativa más defendible y ya
presente en el propio código del pipeline PIT: **`_is_plate_appearance_event()`**, la misma
función que ya usa `_aggregate_common()` para calcular `plate_appearances` correctamente.
Un evento con `launch_speed` no nulo que ADEMÁS es el evento terminal de su PA es,
inequívocamente, un batted-ball-event real — exactamente lo que "attempts" debe significar
bajo cualquier definición razonable, y consistente con el objetivo explícito de MATH-002
(paridad live/backtest, no una redefinición teórica).

---

## TAREA 3 — Ground truth offline

**Metodología**: 16 bateador-temporada (`data/pit_raw/raw_savant_{2024,2025}.db`, cero red),
8 con PA alto (>650, para maximizar señal) + 8 con PA medio (150-450, para diversidad),
mezclando 2024/2025. Para cada uno, tomo el snapshot cacheado del `2024-09-15` o `2025-09-15`
(`savant.batter.rolling`), extraigo `source_window_start_date`/`source_window_end_date` del
propio JSON cacheado (garantiza comparar exactamente la misma ventana walk-forward, sin
mezclar temporadas ni fechas), y recuento tres cosas directamente de
`raw_savant_events` para ese `batter` en esa ventana:

1. **`pa_correct_recount`**: aplico `_is_plate_appearance_event()` tal cual vive en el código,
   deduplicado por `(game_pk, at_bat_number)` quedándome con el pitch_number máximo.
2. **`bbc_naive_recount`**: replico EXACTAMENTE la línea 635 del builder (`launch_speed is not
   None`, sin scoping de PA) — esto valida que mi recount reproduce el mecanismo real, no una
   interpretación mía.
3. **`bbc_correct`**: intersección — eventos que son terminales de PA (mismo criterio que #1)
   Y tienen `launch_speed` no nulo. Este es el proxy de "attempts real" según la Tarea 2.

Tabla completa en `audit_20260714/math002_diagnostico/tabla_comparacion.csv`. Resumen:

| Verificación | Resultado |
|---|---|
| `pa_correct_recount == pa_cached` en los 16 casos | ✅ Sí (16/16) — confirma que mi metodología de ventana/recuento reproduce fielmente `plate_appearances` |
| `bbc_naive_recount == bbc_cached` en los 16 casos | ✅ Sí, **exacto, byte a byte** (16/16) — confirma que el cache real fue construido exactamente con la línea 635 sin scoping de PA, no es una sospecha, es el mecanismo verificado |
| `bbc_cached > pa_cached` en los 16 casos | ✅ Sí (16/16) — una métrica de "batted balls" mayor que el propio PA es imposible bajo la definición correcta |
| `ratio_cached_over_correct` (cuánto sobreestima el cache vs. attempts real) | min=1.703, max=2.188, **media=1.905** |
| `pa_over_attempts_actual` (pa / attempts real — lo que el comentario adivinó en ~1.47x) | min=1.298, max=1.837, **media=1.513** |

**Muestra completa** (season, batter_id, as_of, pa, bbc_cached, bbc_correcto, ratio):

```
2024 680776 2024-09-15  pa=685  bbc_cached=917  bbc_correct=481  ratio=1.906
2024 683002 2024-09-15  pa=672  bbc_cached=836  bbc_correct=445  ratio=1.879
2024 596019 2024-09-15  pa=671  bbc_cached=932  bbc_correct=479  ratio=1.946
2024 665742 2024-09-15  pa=670  bbc_cached=741  bbc_correct=435  ratio=1.703
2025 656941 2025-09-15  pa=681  bbc_cached=746  bbc_correct=387  ratio=1.928
2025 596019 2025-09-15  pa=678  bbc_cached=963  bbc_correct=477  ratio=2.019
2025 646240 2025-09-15  pa=678  bbc_cached=849  bbc_correct=388  ratio=2.188
2025 672695 2025-09-15  pa=671  bbc_cached=914  bbc_correct=492  ratio=1.858
2025 677587 2025-09-15  pa=338  bbc_cached=449  bbc_correct=241  ratio=1.863
2025 671289 2025-09-15  pa=405  bbc_cached=589  bbc_correct=312  ratio=1.888
2025 663886 2025-09-15  pa=316  bbc_cached=358  bbc_correct=172  ratio=2.081
2024 671056 2024-09-15  pa=233  bbc_cached=281  bbc_correct=160  ratio=1.756
2025 669257 2025-09-15  pa=431  bbc_cached=539  bbc_correct=275  ratio=1.960
2024 670032 2024-09-15  pa=423  bbc_cached=576  bbc_correct=320  ratio=1.800
2025 690987 2025-09-15  pa=180  bbc_cached=219  bbc_correct=119  ratio=1.840
2025 660162 2025-09-15  pa=271  bbc_cached=308  bbc_correct=165  ratio=1.867
```

### Hallazgo adicional relevante para el alcance del fix: `barrel_count` NO está contaminado

Verificado sobre la misma muestra de 16: `barrel_count` (numerador, línea 647:
`sum(1 for event in events if event.launch_speed_angle == 6)`, también sin scoping de PA en
apariencia) — **cero fouls clasificados como barrel en los 16/16 casos**
(`foul_barrels_included = 0` en todos). Es decir, el código de Statcast aparentemente nunca
asigna el código de clasificación "barrel" (`launch_speed_angle == 6`) a un foul — la métrica
de barrels en sí ya es correcta. Esto significa que **`barrel_pa` (barrels/PA), la tasa que
efectivamente usa `tte_pit_adapter.py` hoy, NO está sesgada por este bug** — ni el numerador
(`barrel_count`) ni el denominador (`plate_appearances`) están contaminados. **El bug vive
únicamente en `batted_ball_count`**, que hoy no se usa como denominador de ninguna tasa que
llegue al adaptador (`brl_percent` sí lo usa, pero `brl_percent` no está entre los campos que
lee `tte_pit_adapter.py`) — su único rol relevante para MATH-002 es como candidato a base de
shrinkage (`n` en `_regress()`), que es exactamente lo que MATH-002 se planteaba usar.

---

## TAREA 4 — Veredicto

**(a) — Bug confirmado tal como lo describe el comentario, con evidencia directa y
reproducible, y de magnitud mayor a la estimada.**

- El comentario en `tte_pit_adapter.py:176-178` decía "PA ~1.47x attempts" — la medición real
  da `pa/attempts_real` = 1.513 de media (bastante cerca, el comentario acertó en esa
  proporción).
- Pero el comentario **asumía implícitamente que `batted_ball_count` (el candidato a
  "attempts" real) YA es la cifra correcta**, y que el problema es solo "cuál `n` usar entre
  `pa` y `batted_ball_count`". La medición real muestra que `batted_ball_count` en sí está
  inflado 1.905x respecto al valor correcto — **más inflado que el propio `pa`** (`bbc_cached
  / bbc_correct` = 1.905 > `pa / bbc_correct` = 1.513). Es decir, **si el fix de MATH-002 se
  hubiera implementado ingenuamente usando `batted_ball_count` tal cual existe hoy como el
  nuevo `n`, el resultado habría sido PEOR que el bug actual** — menos shrinkage aún que
  usando `pa`, no más, exactamente lo opuesto de lo que MATH-002 buscaba corregir.
- Confirmado mecanismo exacto: fouls trackeados por Statcast (`type='S'`, `description='foul'`,
  con `launch_speed`/`launch_angle` reales) se cuentan como batted-ball-events en
  `_aggregate_common()` porque el filtro (línea 635) no exige que el evento sea terminal de
  PA, a diferencia de `plate_appearances` (línea 634), que sí lo exige correctamente.

---

## TAREA 5 — Scope del fix (sin implementar, solo mapeo)

**Archivo(s) a cambiar**:
- `modules/baseball_module/advanced_pit_enrichment/savant_offense_daily_aggregator.py`,
  función `_aggregate_common()` — el fix mínimo es intersectar `batted_ball_events` con
  `pa_events` (o, equivalentemente, filtrar por `_is_plate_appearance_event(event)` además de
  `launch_speed is not None`), en la línea 635. Esto corrige `batted_ball_count` y, en cascada,
  `brl_percent` (que ya usa `len(batted_ball_events)` como denominador) para que quede
  correcto también, aunque `brl_percent` no es consumido hoy por el adaptador.
- Después de corregir el builder: `tte_pit_adapter.py` necesitaría un segundo cambio (fuera de
  este diagnóstico) para efectivamente leer `attempts`/`bip` del snapshot en vez de `pa` para
  el shrinkage de barrel — la plomería (`tte_daily_snapshot_builder.py:125`, alias `"bip"`) ya
  existe, per Tarea 1.2.

**Caches a reconstruir** (todos derivados de `_aggregate_common()`, confirmado por grep):
- `savant.batter.rolling` (206,575 filas hoy) — `pit_cache_merged.db`.
- `savant.team_offense.rolling` — `pit_cache_merged.db`.
- `savant.team_offense.prior_baseline` (`bip_prior`, vía `tte_prior_baseline_builder.py:162`)
  — `pit_cache_merged.db`. **No localicé un script incremental dedicado para este namespace**
  (solo existe `scripts/build_pitcher_prior_baseline.py`, del lado pitcher) — el paso de
  implementación necesitará ubicar o construir el mecanismo de rebuild para este namespace
  específicamente.
- `savant.batter.rolling`/`savant.team_offense.rolling` de `pit_cache_2024.db`/
  `pit_cache_2025.db` individuales, si el pipeline los trata como fuente separada de
  `pit_cache_merged.db` — no verificado en este diagnóstico (fuera de su alcance), a
  confirmar en el paso de implementación.
- **No afectados** (no comparten `_aggregate_common()`, no verificado exhaustivamente pero sin
  evidencia de que compartan el mismo builder): `savant.team_bullpen.*`,
  `savant.team_defense.*` — hipótesis, no chequeado, fuera del alcance de este diagnóstico
  (MATH-002 es específicamente sobre ofensa/TTE).

**¿Rebuild 100% offline?** Sí, confirmado — `scripts/build_offense_savant_rolling_incremental.py`
lee exclusivamente de `data/pit_raw/raw_savant_{season}.db` (ya presentes localmente), cero
llamada de red, según su propio uso documentado.

**Runtime estimado — evidencia real, no estimación**: el mismo script ya reconstruyó ambas
temporadas completas en la sesión del 2026-07-12 (`reports/offense_rebuild/{2024,2025}.log`):
- 2024: 187 días, **796.1 segundos (~13.3 min)**.
- 2025: 186 días, **826.2 segundos (~13.8 min)**.
- Total ambas temporadas: **~27 minutos**, secuencial, offline.
Esto es consistente con el diseño O(días) incremental descrito en `CLAUDE.md`. El rebuild del
namespace `prior_baseline` (mecanismo no localizado en este diagnóstico) queda sin estimar.

---

## STOP

Diagnóstico completo. No se modificó código, no se reconstruyó ningún cache, no se tocó la
DB de producción, cero llamadas de red — confirmado por el propio historial de comandos de
esta sesión (todas las consultas fueron `SELECT` de solo lectura contra `data/pit_raw/*.db` y
`data/pit_cache_merged.db`). El fix (5b), el switch de MATH-002 (5c), y el análisis de
MATH-003 quedan para un paso posterior, con este veredicto en mano.
