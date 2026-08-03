# Fase 2B — Commit B2: gate de delta (leak V4 remediado)

**Fecha**: 2026-07-19. Comando: `backtest_and_retrain.py --season 2024,2025 --use-full-pit`
(mismos 4 flags de PIT cache de siempre). Reporte:
`audit_20260714/fase2b/gate_delta/backtest_report_20260719_1420.json`.
Baseline canónico previo: `audit_20260714/paso5c_gate/backtest_report_20260718_1452.json`
(Brier 0.24483 / accuracy 55.51%).

## Tabla completa

| Métrica | Antes (con leak V4) | Después (sin leak) | Δ |
|---|---|---|---|
| Juegos totales | 4830 | 4825 | **-5** |
| Accuracy global | 55.51% | 54.80% | **-0.71pp** |
| Brier (modelo) | 0.24483 | 0.24650 | **+0.00167** |
| Brier vs random | +2.07% | +1.40% | -0.67pp |
| Log-loss | 0.68283 | 0.68626 | +0.00343 |
| Accuracy 2024 | 55.57% | 54.77% | -0.80pp |
| Brier 2024 | 0.24584 | 0.24769 | +0.00185 |
| Accuracy 2025 | 55.45% | 54.83% | -0.62pp |
| Brier 2025 | 0.24383 | 0.24532 | +0.00149 |
| Favorite agreement vs Pinnacle | (no medido antes) | 75.47% | — |
| Model vs Pinnacle Brier | (no medido antes) | -2.49% | — |

## ROI por bucket de edge

| Edge | Antes: bets / ROI | Después: bets / ROI |
|---|---|---|
| ≥0% | 4604 / -0.42% | 4600 / **-2.55%** |
| ≥2% | 3531 / +0.78% | 3554 / **-2.93%** |
| ≥5% | 2115 / -0.40% | 2159 / **-2.82%** |
| ≥8% | 1083 / +2.02% | 1153 / **-4.07%** |
| ≥10% | 654 / -2.45% | 679 / **-6.81%** |

**Cada bucket empeoró, varios pasaron de positivo a negativo.**

## Calibración (bucket de probabilidad predicha)

| Bucket | Antes Pred%/Actual%/Diff | Después Pred%/Actual%/Diff |
|---|---|---|
| <40% | 35.7% / 40.9%(*) / +5.2% | 35.6% / 47.4% / **+11.8%** |
| 40-45% | 42.8% / 43.5% / +0.7% | 42.8% / 42.9% / +0.1% |
| 45-50% | 47.7% / 50.9% / +3.2% | 47.8% / 50.1% / +2.3% |
| 50-55% | 52.5% / 52.9% / +0.4% | 52.5% / 53.7% / +1.2% |
| 55-60% | 57.2% / 55.6% / -1.6% | 57.2% / 55.1% / -2.1% |
| 60-70% | 63.5% / 64.6% / +1.1% | 63.3% / 63.5% / +0.2% |
| >70% | 72.8% / 68.1% / -4.7% | 73.0% / 68.3% / -4.7% |

(*) Nota: la fila `<40%` "antes" citada aquí es del reporte 5c (post-MATH-002); el bucket con
peor deterioro (+11.8pp de diff) es justo el de probabilidades más bajas — plausible si el leak
inflaba artificialmente la confianza del modelo en sus predicciones más extremas hacia el
favorito equivocado.

## Movimiento por juego (p_home individual)

Fuente "antes": `data/predictions_history_backup_pre_fase2b_20260719_132946.db` (backup tomado
al INICIO de B1, antes de tocar cualquier cutoff — validado que corresponde al estado
pre-leak-fix porque B1 fue identidad confirmada contra el canónico 5c). "Después": DB actual
post-B2. 4,859 juegos comunes (2024+2025, `source='backtest'`, `backtest_p_home` no nulo).

| Umbral de movimiento en p_home | Juegos | % del total |
|---|---|---|
| > 0.5pp | 2,781 | **57.23%** |
| > 1.0pp | 1,534 | **31.57%** |
| > 2.0pp | 493 | **10.15%** |

Delta máximo observado: **16.63pp**. Delta medio (todos los juegos): 0.91pp.

**Correlación con B1.3(b)**: el backfill de B1 encontró 22.2% de filas con
`official_date != date(game_date)` (juegos nocturnos). El movimiento de `p_home` (57.2% de
juegos movidos >0.5pp) es MAYOR que ese 22.2% — consistente con que el leak no solo afectaba
directamente a los juegos nocturnos contaminados, sino que se propagaba en cascada a través del
entrenamiento walk-forward de team-bias/Kalman (un leak en CUALQUIER juego de un equipo afecta
el sesgo aprendido para ESE equipo en TODOS sus juegos posteriores de la temporada, no solo en
el juego contaminado puntual).

## Los -5 juegos: explicados, no un error

34 líneas de log `SKIPPED Team/TTE PIT coverage: missing_both_team_tte_pit` aparecen en esta
corrida (algunas coinciden con exclusiones que ya existían en el baseline previo por otras
razones). El conteo neto de juegos bajó de 4830 a 4825. Consistente con la explicación esperada:
con el cutoff correcto (un día antes del día oficial, no del día contaminado), un puñado de
juegos de inicio de temporada — que antes "colaban" porque el día extra que el leak les daba
alcanzaba a incluir la primera snapshot disponible — ahora correctamente no encuentran ninguna
cobertura PIT anterior a su propio primer juego, y se excluyen. Esto es exactamente el
comportamiento honesto esperado, no un bug nuevo introducido por este cambio.

## Lectura

**Peor en absolutamente todas las métricas — el patrón "peor pero honesto" que este roadmap
anticipó explícitamente como éxito, no como fracaso.** El leak V4 explicaba una porción real y
sustancial del baseline anterior (0.24483/55.51%): con el walk-forward genuinamente
point-in-time, el modelo cae a 0.24650/54.80%. El ROI, que ya era "delgado e inconsistente"
antes (CLAUDE.md), ahora es negativo en los 5 buckets de edge medidos — la pregunta de
rentabilidad que seguía abierta desde el leak-fix de team-bias (2026-07-08) queda, si acaso, más
lejos de responderse afirmativamente, no más cerca.

Nada de esto se auto-aprueba ni se auto-revierte. La dirección (empeorar) es exactamente la
esperada — pero por la propia regla de este roadmap, un delta "que empeora" tampoco es
auto-aprobación: el juicio de si la magnitud es razonable y explicable es humano.

## Decisión

**Pendiente de aprobación explícita del dueño antes de commitear B2.**
