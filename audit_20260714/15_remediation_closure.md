# 15 — CIERRE DE REMEDIACIÓN — AUDIT 2026-07-14

Fecha de cierre: 2026-07-18. Roadmap del audit: **5/5 completo**. Suite: 462 (audit) → **523/523** (cierre). Revisión independiente de cada paso: Claude (Fable), con evidencia fresca exigida en cada gate.

## Commits de la remediación

| Paso | Alcance | Commits | Gate | Resultado |
|---|---|---|---|---|
| 1 | CHRON-001: provenance de game_outcomes (source + columnas backtest_*), forense de backups | 175f497 | Identidad | Byte-idéntico vs round10 |
| 2 | CHRON-002 (namespaces ml_state/kalman_state + promote_calibration.py), fuente de entrenamiento Platt-2D, monitor LEARN-002 | 3 commits (incl. f369f12) | Identidad + smoke live + igualdad 2D | Todos PASS |
| 3 | ODDS-001 (_devig → import único) + MATH-001 (parámetro muerto) | 2a7d3b6 (+6d72089: borrado CLAUDE2.md) | Identidad | Byte-idéntico |
| 4 | FALL-001/002 (flags de procedencia, eliminado el 1000mi/1TZ fabricado) + test de consistencia de estadios | 74f6e24 | Identidad | Byte-idéntico |
| 5a | Diagnóstico read-only del foul-inflation (veredicto (a), mecanismo reproducido 16/16) | — (solo reporte) | STOP | Cumplido |
| 5b | Fix del conteo (hallado en 2 implementaciones independientes) + rebuild de 3 namespaces | aa1656b | Identidad | Byte-idéntico (6ª) |
| 5c | MATH-002: paridad exacta del término barrel con live (métrica completa, no solo n) | 9379c1a | **Delta + aprobación humana** | Aprobado: +0.17pp acc, Brier plano, máx 1.06pp/juego |
| 5d | MATH-003: script repetible de residual home-win; análisis sin cambio de modelo | 300f138 | Suite | Residual 2024 +0.08pp / 2025 +1.47pp — decisión abierta |

**Baseline canónico vigente**: Brier **0.24483** / accuracy **55.51%** — fuente de verdad: `audit_20260714/paso5c_gate/backtest_report_20260718_1452.json`. La prosa de CLAUDE.md cita este archivo; el JSON manda.

## Veredictos del audit, actualizados post-remediación

1. **"The current backtest is trustworthy for model evaluation"** — SÍ, con caveats reducidos de tres a dos: (i) sigue weather-blind por diseño (parity gap documentado e intencional; su cierre es decisión de blueprint, no defecto); (ii) los reportes aún no auto-declaran los flags PIT con que se generaron (riesgo de proceso, abierto). El caveat de MATH-002 (barrel% under-shrinkage) queda **retirado** — resuelto en 5b/5c con paridad exacta verificada contra el camino live.
2. **"The current ROI is trustworthy"** — el ROI de backtest es confiable al mismo nivel que el Brier (misma fuente de probabilidades) y sigue siendo delgado e inconsistente por bucket (≥8% +2.02%, ≥10% −2.45% en el baseline canónico — cifras del reporte, no de prosa). El ROI live **sigue sin existir** (0 picks resueltos; feed de Pinnacle apagado) — estado correcto y correctamente representado, no un defecto.
3. **"The live model is affected by the identified findings"** — los hallazgos que lo afectaban están **cerrados**: CHRON-001/002 (provenance de datos y de estado de calibración), FALL-001/002 (fallbacks honestos), LEARN-002 (monitor operando — disparó una alerta real el primer día: pct_pinnacle_present 0%). El riesgo residual del camino live es **operativo** (feed de Pinnacle desactivado → CLV no acumula, corrección Platt-2D en bypass elegante), no de correctitud.

**Estado global: PASS** (el audit cerró en PARTIAL). Todos los findings High y Medium de findings.csv están resueltos; los abiertos restantes son Low, informativos, o decisiones del dueño.

## Lo que la remediación encontró ADEMÁS del roadmap

- **CHRON-002** (descubierto en el cierre del Paso 1, resuelto en el Paso 2): ml_state/kalman_state compartidos entre live y backtest — el gemelo de CHRON-001 en la capa de estado. Incluye el cambio conceptual de "el backtest pisa la calibración live implícitamente" a promoción explícita opt-in (`scripts/promote_calibration.py`).
- **Pérdida histórica documentada**: 563 predicciones live de season 2026 sobrescritas el 2026-06-28, irrecuperables (el backup más antiguo es del 07-05). Sin picks perdidos (picks=0 en esa fecha); la pérdida es telemetría, no ledger. Salvage de baja prioridad posible vía tabla `predictions` (88 rows) y logs/.
- **Cuatro instancias de prose-drift corregidas**: REG-021 (Platt-2D nunca corrió en el backtest), MANIFIESTO.md (citado como archivo; no existe — decisión: no crearlo, consolidar en CLAUDE/CONTRACTS/blueprint), baseline de CLAUDE.md (0.24486→0.24482 real), buckets de ROI de CLAUDE.md (stale vs reporte).
- **Séptima instancia del patrón de duplicación**: el conteo con foul-inflation existía en dos implementaciones independientes (5b). El caso para el Feature Store (§1.1 del blueprint) siguió creciendo solo.
- **Regla de workflow nueva** (consecuencia de CHRON-002, escrita en CONTRACTS.md): todo cambio de modelo validado por backtest requiere `promote_calibration.py` antes del deploy — Kalman y team bias live se auto-mantienen; Platt 1D live solo se actualiza por promoción explícita.
- **Fixture permanente 16/16**: la tabla del diagnóstico 5a es test de regresión — un rebuild futuro que reintroduzca el conteo malo truena en CI.

## Registro de abiertos (post-cierre)

| # | Item | Tipo | Dueño de la decisión |
|---|---|---|---|
| 1 | MATH-003: `_UNIFORM_HOME_MULT=0.028` vs residual 2025 (+1.47pp, ~1σ dado SE≈±1.0pp/temporada) | Decisión de modelo | Braulio — recomendación: mantener 0.028, re-visitar con la 3ª temporada (2026) o al próximo cambio de modelo |
| 2 | Foul-inflation del lado de pitchers (reportado en 5b, no arreglado) | Diagnóstico condicional | Entra solo si un grep muestra consumidores actuales de esos campos |
| 3 | Reportes de backtest auto-declarando flags PIT | Proceso, chico | Pendiente de calendarizar |
| 4 | Secondary sort key en el ORDER BY del backtest (game_date, game_pk) | Cambio de comportamiento, paso propio con before/after | Pendiente |
| 5 | analyze_game_outcomes.py leyendo columnas live congeladas | Decisión (apuntar a backtest_* o dejar como vista histórica) | Braulio |
| 6 | INV-001/002/003 (higiene: .codex/, DBs huérfanas, weight_optimizer.py), DUP-001, REG-017/018/019 | Low / triage original del audit sin cambio | Cuando convenga |
| 7 | Feature Store (blueprint §1.1) — ahora con 7 instancias de evidencia | Arquitectura, multi-semana | Re-priorizar en el blueprint |
| 8 | Items restantes de hypotheses.md (roof default, sweep de silent-excepts, etc.) | Verificaciones futuras | Cuando convenga |

## Métricas del proceso de remediación

Seis validaciones de identidad byte a byte del baseline completo (que además re-confirmaron el determinismo de punta a punta seis veces). Dos checkpoints humanos ejercidos (aprobación del delta de 5c; decisión de MATH-003 dejada abierta). Cero regresiones de suite en toda la remediación. Un monitor de producción que detectó una condición real (Pinnacle 0%) en su primera ejecución.

## Siguientes acciones fuera del alcance del audit

1. Reactivar ODDS_API_KEY → arranca el reloj de acumulación de CLV (4-6 semanas de datos de Pinnacle; prerequisito de toda evaluación de edge en vivo).
2. Cron de `run_daily_picks.py` completando ciclos reales publicar→resolver.
3. Cablear `historical_weather.py` al backtest (cierra el caveat #1 del veredicto 1; el módulo ya existe, es un paso propio con before/after).
4. De ahí, el blueprint retoma el mando.
