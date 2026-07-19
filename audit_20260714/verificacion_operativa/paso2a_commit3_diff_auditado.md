# Fase 2A — Commit 3: auditoría del parche de closing-lines pendiente

**Fecha**: 2026-07-19. El working tree traía un parche de closing-lines/lead-time escrito
ANTES del sweep de verificación, sin commitear ni revisar formalmente. Este documento es el
diff exacto entre lo que ese parche hacía originalmente y lo que quedó tras auditarlo contra la
especificación de la Fase 2A — commit 3.

## Lo que el parche original hacía (antes de esta auditoría)

1. `get_picks_needing_closing_capture()`: gate opcional `near_commence_minutes` — si se pasaba,
   excluía picks cuyo `commence_time` estuviera a más de N minutos en el futuro. **Diseño
   "captura única"**: `capture_closing_line()` seguía usando `WHERE closing_captured_at IS NULL`
   — una vez capturado, ninguna llamada posterior podía tocar ese pick de nuevo, sin importar
   qué tan temprano o tarde hubiera sido esa primera captura.
2. `capture_closing_lines.py` (script): exponía `--near-commence-minutes` como flag de CLI,
   pasándolo directo a la query de arriba.
3. Sin columna `minutes_before_start` — ninguna forma de saber qué tan "fresca" fue una
   captura específica.
4. Su propia llamada a `get_best_odds_for_teams()` no pasaba `commence_time` en absoluto
   (llamada solo por nombre de equipo — el mismo bug que el Commit 2 de esta fase arregló en
   otros call sites).
5. Picks sin `commence_time`: la query los seguía devolviendo como "pendientes de captura" sin
   ninguna distinción — capturados igual que cualquier otro, sin warning.

## Lo que quedó después de auditar contra la spec de la Fase 2A

| Punto de la spec | Diseño original del parche | Lo que quedó |
|---|---|---|
| 1. Última-pre-inicio gana | Captura única (`closing_captured_at IS NULL` bloqueaba cualquier repetición) | **Rediseñado**: `capture_closing_line()` ya no bloquea repetición — sobrescribe siempre que la captura sea ANTES de `commence_time`. Un intento en o después de `commence_time` se rechaza (`return False`, fila intacta). `get_picks_needing_closing_capture()` ahora devuelve TODO pick cuyo juego no ha empezado, incluso si ya fue capturado antes — es lo que permite que un sweep posterior lo sobrescriba. |
| 2. commence_time almacenado desde publicación | Ya existía (columna + `publish_pick(commence_time=...)`) | Sin cambios — se restauró la línea en `publisher.py` que quedó temporalmente removida durante el staging quirúrgico del Commit 2 |
| 3. captured_at + minutes_before_start | Solo `captured_at` existía | **Nueva columna** `minutes_before_start REAL` (migración idempotente), calculada como `(commence_time − captured_at)` en minutos, guardada en cada captura exitosa |
| 4. Sin commence_time → salteado con warning | La query los devolvía igual, sin distinción | `capture_closing_lines()` (la función de orquestación en el script) ahora chequea `if not pick["commence_time"]` ANTES de intentar nada, cuenta `summary["skipped_no_commence_time"]`, y hace `log.warning(...)` explícito — nunca los captura "de todos modos" |
| — (fuera de la spec original del parche, pero requerido por el Commit 2) | `get_best_odds_for_teams()` llamado sin `commence_time` | Ahora pasa `commence_time=pick["commence_time"]` — coherente con el Commit 2, que dejó esta actualización pendiente para este commit específicamente porque dependía de esta misma columna |

## Simplificación real: `near_commence_minutes` queda eliminado, no solo deprecado

El diseño original de `near_commence_minutes` (una ventana de proximidad configurable) queda
**completamente superado** por "última-pre-inicio gana" — con el nuevo diseño, cualquier sweep
puede correr en cualquier momento antes del inicio del juego sin riesgo de mal-etiquetar un
precio prematuro como "closing", porque la ÚLTIMA captura pre-inicio siempre gana sobre
cualquier captura anterior. Mantener AMBOS mecanismos simultáneamente habría sido complejidad
redundante — se eliminó el parámetro y el flag de CLI `--near-commence-minutes` por completo,
en vez de dejarlo como alternativa sin usar.

## Hallazgo adicional durante la auditoría: mismo anti-patrón de `sys.modules`, 3 veces más

Al escribir los tests de este commit until que uno fallaba de forma intermitente (mismo síntoma
que el bisectado en el Commit 1: un monkeypatch sobre `odds_fetcher` se volvía inefectivo al
correr la suite completa). Repetí el mismo bisecting y encontré que el anti-patrón arreglado en
el Commit 1 (`sys.modules.pop()` crudo en vez de `monkeypatch.delitem`) existía en **3 archivos
más** — los 4 comparten el mismo propósito ("verificar que un script no importa módulos del
pipeline en vivo") y los 4 incluían `odds_fetcher` en su lista de módulos prohibidos:

- `tests/test_savant_raw_ingestor.py`
- `tests/test_build_pitcher_pit_cache_script.py`
- `tests/test_build_experimental_pitcher_pit_cache.py`

Los 3 arreglados con el mismo patrón (`monkeypatch.delitem` en vez de `sys.modules.pop()`
crudo). Suite verde confirmado después.

## Tests nuevos/reescritos

- `tests/test_track_record_closing_lines.py`: reescrito — se eliminaron los 2 tests del gate
  `near_commence_minutes` (ya no existe ese parámetro) y el test de idempotencia de captura
  única (invertido: ahora prueba que una segunda captura SÍ sobrescribe). Nuevos: secuencia
  temprano→tarde-pre-inicio (gana el tardío, `minutes_before_start` refleja la captura más
  reciente), intento post-inicio (rechazado, fila intacta), intento exactamente en
  `commence_time` (rechazado), captura sin `commence_time` en absoluto (nunca rechazada por
  timing, al no haber base para comparar).
- `tests/test_capture_closing_lines_orchestration.py` (nuevo): prueba la función de
  orquestación completa (no solo los métodos de `db.py`) — pick sin `commence_time` se salta
  con warning; `commence_time` se pasa correctamente a `get_best_odds_for_teams()`; sin datos
  de mercado no se captura.
- `tests/test_track_record_publisher_lead_time.py`: escrito originalmente junto con el fix de
  `commence_raw` (el mismo que terminó commiteado en el Commit 2, ya que era un prerequisito
  real de esa corrección) — se commitea aquí en el Commit 3 por orden de secuencia, pero valida
  comportamiento que ya vive en HEAD desde el Commit 2.

545/545 tests verdes (suite completa, incluyendo los 3 tests marcados `integration` del Commit 1).
