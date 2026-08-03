# CHRON-001 — FASE 1: Forense de backups

## Procedimiento seguido

1. Primer momento en que un backtest tocó season 2026:
   ```
   SELECT MIN(backtest_run_at) FROM game_outcomes WHERE season=2026 AND backtest_run_at IS NOT NULL;
   → 2026-06-28T22:33:48.878536+00:00
   ```
2. Buscar el backup más reciente **anterior** a ese momento, entre TODOS los archivos
   `predictions_history*.db` existentes en el repo (no solo los que siguen el patrón de
   nombre `_backup_pre_*` — se revisaron los 19 archivos, incluyendo
   `data/predictions_history_pit_run.db`, que no sigue esa convención).

## Resultado: no existe ningún backup anterior al evento — desviación explícita del procedimiento

Los 18 backups `_backup_pre_*` están fechados (por nombre y por `mtime`) entre
**2026-07-05 08:05:52** y **2026-07-12 09:45:19** — todos **posteriores** al evento de
sobrescritura (2026-06-28 22:33:48 UTC) por 6-14 días.

El único archivo con `mtime` cercano al evento es `data/predictions_history_pit_run.db`
(`mtime` 2026-06-28 18:18:25 -0700 = 2026-06-29 01:18:25 UTC) — pero esto es **~2h45min
DESPUÉS** del evento (22:33:48 UTC), no antes. Verificado directamente contra su contenido:
también muestra los 563 rows ya con `backtest_run_at` fechado 2026-06-28 — es casi con
certeza el propio artefacto de salida de esa misma corrida de backtest del 2026-06-28
(coincide con las corridas registradas en `reports/baseline/` y `reports/full_pit/`, ambas
fechadas `20260628`), no un snapshot previo.

**Conclusión empírica, verificada contra los 19 archivos, no solo inferida por nombre**:
la disciplina de backups pre-operación-riesgosa (los 18 archivos `_backup_pre_*`) empezó a
practicarse a partir de 2026-07-05 — una semana después de que la corrida del 2026-06-28 ya
hubiera sobrescrito los 563 rows. No hay ningún punto de recuperación anterior a ese evento
en ningún archivo presente en este repositorio.

## Números exactos (sin redondear)

| Categoría | Cantidad |
|---|---|
| Rows season-2026 actualmente con `backtest_run_at` seteado (sobrescritos) | **563** |
| De esos 563, con fecha de sobrescritura distinta a 2026-06-28 | **0** (los 563 comparten una sola fecha — una única corrida de backtest los tocó, nunca más desde entonces) |
| Rows recuperables desde algún backup/snapshot anterior al evento | **0** |
| Rows irrecuperables | **563** (el 100% de los ya sobrescritos) |
| Rows live actuales, aún no tocados (`backtest_run_at IS NULL`) | **52** — no están en riesgo retroactivo, son el motivo por el que este fix se está construyendo hacia adelante |

## Progresión confirmada (evidencia adicional de consistencia)

El backup más antiguo disponible (`predictions_history_backup_pre_clean_pit_backtest_20260705_080552.db`,
2026-07-05) muestra exactamente los mismos 563 rows ya sobrescritos (misma fecha
2026-06-28) más 30 rows todavía live en ese momento (`game_date` hasta 2026-07-04). La DB
actual tiene 52 rows live (`game_date` hasta 2026-07-12). La diferencia (52−30=22 juegos
nuevos en 8 días, ~2.75/día) es consistente con el ritmo real de un calendario de MLB y con
que **ningún backtest ha vuelto a tocar season 2026 desde el 2026-06-28** — confirmado
también por el hecho de que la única fecha distinta en `backtest_run_at` para season 2026,
en la DB actual, sigue siendo 2026-06-28.

## Veredicto de FASE 1

**0 de 563 rows recuperados.** `audit_20260714/chron001_recovered_live_rows.csv` se generó
vacío (solo cabecera + nota explicativa) porque no hay nada que archivar — no existe ninguna
copia de los valores live originales de esos 563 juegos en ningún archivo de este
repositorio. La pérdida ya ocurrió, es permanente, y precede en una semana a la primera vez
que el proyecto empezó a tomar backups antes de operaciones riesgosas.

Esto no cambia el alcance de FASE 2-7 (siguen operando sobre los 52 rows live actuales y
hacia adelante), pero es la respuesta honesta a lo que pedía FASE 1: no hay nada que
restaurar porque no hay nada que recuperar.
