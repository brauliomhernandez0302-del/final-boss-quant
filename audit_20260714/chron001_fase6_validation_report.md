# CHRON-001 — FASE 6: Validación por identidad

## Comando ejecutado

```bash
python3 backtest_and_retrain.py \
  --season 2024,2025 \
  --use-full-pit \
  --pitcher-pit-cache-db data/pit_cache_pitcher.db \
  --team-tte-pit-cache-db data/pit_cache_merged.db \
  --defense-pit-cache-db data/pit_cache_merged.db \
  --bullpen-pit-cache-db data/pit_cache_merged.db \
  --report-dir audit_20260714/fase6_validation
```

Permitido explícitamente bajo la regla interina (no incluye season 2026). Corrido contra la
DB real de producción (`data/predictions_history.db`, ya respaldada en FASE 0) — seguro
porque, como confirma FASE 2, los 4,830 rows de seasons 2024/2025 ya tenían `source='backtest'`
antes de este cambio, así que `update_game_outcomes()` solo escribe en sus columnas
`backtest_*`, nunca en columnas live (no hay ningún row live en esas dos temporadas).

Runtime: ~13 minutos (14:46:20 → 14:59:19), consistente con la corrida histórica equivalente.

## Resultado

```
BACKTEST REPORT — 4830 games | seasons: [2024, 2025]
Accuracy 55.34% | Brier score (model) 0.24482 | Log-loss 0.68281
BY SEASON: 2024 N=2415 accuracy=55.32% Brier=0.24579
           2025 N=2415 accuracy=55.36% Brier=0.24385
```

Report guardado en `audit_20260714/fase6_validation/backtest_report_20260717_1459.json`.

## Comparación — identidad confirmada a nivel de bytes

Comparé el JSON completo de este run contra el último reporte real en disco de una corrida
equivalente pre-cambio, `reports/round10_composite_weight_nudge/backtest_report_20260712_0827.json`
(2026-07-12, la última de las 10 rondas de esta sesión, `--season 2024,2025 --use-full-pit`,
4830 juegos — mismo comando exacto):

```bash
diff <(json_pretty round10_composite_weight_nudge/backtest_report_20260712_0827.json) \
     <(json_pretty audit_20260714/fase6_validation/backtest_report_20260717_1459.json)
```

**Resultado: 1 línea de diferencia en todo el JSON — el campo `run_at` (timestamp de cuándo
corrió cada uno). Todo lo demás — accuracy, Brier, log-loss, las 7 calibration buckets, los 5
umbrales de ROI simulation, el breakdown BY SEASON, la distribución de lambda, y los 3
resúmenes de cobertura PIT (team_tte/defense/bullpen) — es idéntico byte por byte.**

Dado que estas son estadísticas agregadas sobre 4,830 juegos calculadas con un seed
determinista por `game_pk`, una identidad tan exacta en docenas de cifras de precisión
completa (no solo redondeadas) solo es posible si cada predicción individual, juego por
juego, fue también idéntica. Esto es una confirmación mucho más fuerte que solo comparar el
Brier/accuracy resumido — es esencialmente una prueba de que absolutamente nada del cómputo
del modelo cambió.

## Discrepancia honesta con el número citado en CLAUDE.md — reportada, no escondida

El prompt de esta tarea esperaba "0.24486/55.42%" (citado de `CLAUDE.md`, sección
"Actualización 2026-07-11/12"). El resultado real, reproducido dos veces de forma
independiente (el reporte de round10 archivado en disco desde 2026-07-12, y esta corrida
fresca de 2026-07-17), es **0.24482/55.34%** — una diferencia pequeña (0.00004 en Brier,
0.08pp en accuracy) pero real, no cero.

**No es un bug de esta migración** — la prueba es la comparación byte-a-byte de arriba: mi
corrida coincide exactamente con round10, que se generó DÍAS ANTES de que este trabajo de
CHRON-001 empezara. La discrepancia ya existía entre `CLAUDE.md` (prosa) y el último reporte
real en disco, independientemente de este fix. Posibles explicaciones (no verificadas, fuera
de alcance de este paso): un redondeo o transcripción manual al escribir la nota de
`CLAUDE.md`, o una corrida adicional no capturada en `reports/` que produjo el número exacto
citado. Reportado aquí explícitamente en vez de forzar que mi resultado "coincida" con el
número esperado — la identidad real y verificable (mi corrida == round10) es la que importa
para el propósito de esta fase (¿el fix cambió algo?), y la respuesta es no.

## Veredicto de FASE 6

**✅ Identidad confirmada.** El fix de CHRON-001 no movió ni un decimal del modelo — exactamente
el resultado exigido. La pequeña discrepancia con la prosa de `CLAUDE.md` es preexistente,
documentada aquí con evidencia exacta, y no bloquea el criterio de salida de esta fase.

**La regla interina sobre season 2026 queda levantada** a partir de este punto — el camino de
escritura/lectura está protegido (FASE 3/4), validado por identidad (FASE 6), y las columnas
`source`/`backtest_*` existen y están pobladas correctamente (FASE 2). Una futura corrida de
backtest que incluya season 2026 ya no puede destruir una predicción en vivo — escribirá en
`backtest_lambda_home`/`backtest_p_home`/etc., nunca en las columnas live.
