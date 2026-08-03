# PASO 5c.4 — Gate de delta (MATH-002 commit 5c)

**Comando**: `backtest_and_retrain.py --season 2024,2025 --use-full-pit` (mismos 4 flags de PIT
cache que el resto de la sesión). Reporte: `audit_20260714/paso5c_gate/backtest_report_20260718_1452.json`.
Baseline canónico previo (post-5b, identity): `audit_20260714/paso5b_gate/backtest_report_20260718_1429.json`.

## Tabla before/after

| Métrica | Antes (post-5b) | Después (post-5c) | Δ |
|---|---|---|---|
| Accuracy global | 55.34% | 55.51% | **+0.17pp** |
| Brier (modelo) | 0.24482 | 0.24483 | +0.00001 |
| Log-loss | 0.68281 | 0.68283 | +0.00002 |
| Accuracy 2024 | 55.32% | 55.57% | +0.25pp |
| Brier 2024 | 0.24579 | 0.24584 | +0.00005 |
| Accuracy 2025 | 55.36% | 55.45% | +0.09pp |
| Brier 2025 | 0.24385 | 0.24383 | -0.00002 |

## ROI por bucket de edge (flat 1u, mejor lado @ Pinnacle)

| Edge | Antes: bets / ROI | Después: bets / ROI |
|---|---|---|
| ≥0% | 4604 / -0.70% | 4604 / -0.42% |
| ≥2% | 3525 / +0.90% | 3531 / +0.78% |
| ≥5% | 2117 / +0.29% | 2115 / -0.40% |
| ≥8% | 1087 / +1.36% | 1083 / +2.02% |
| ≥10% | 651 / -2.20% | 654 / -2.45% |

## Calibración (bucket de probabilidad predicha)

| Bucket | Antes Pred%/Actual%/Diff | Después Pred%/Actual%/Diff |
|---|---|---|
| <40% | 35.7% / 40.9% / +5.2% | 35.7% / 41.6% / +5.9% |
| 40-45% | 42.8% / 43.5% / +0.7% | 42.8% / 43.2% / +0.4% |
| 45-50% | 47.7% / 50.9% / +3.2% | 47.7% / 50.4% / +2.7% |
| 50-55% | 52.5% / 52.9% / +0.4% | 52.5% / 53.4% / +0.9% |
| 55-60% | 57.2% / 55.6% / -1.6% | 57.2% / 55.4% / -1.8% |
| 60-70% | 63.5% / 64.6% / +1.1% | 63.4% / 64.2% / +0.8% |
| >70% | 72.8% / 68.1% / -4.7% | 72.8% / 69.5% / -3.3% |

## Movimiento por juego (p_home individual)

Fuente "antes": `data/predictions_history_backup_pre_chron002_20260718_053008.db` (backup
2026-07-18 05:30, previo a cualquier corrida de esta sesión que tocara `game_outcomes` —
validado que corresponde al mismo estado pre-5c: sus agregados no se recomputaron aquí porque
el propósito es solo el delta por juego, no una tercera fuente de verdad de agregados).
"Después": `data/predictions_history.db` en su estado actual post-5c. 4,859 juegos comunes
(2024+2025, `source='backtest'`, `backtest_p_home` no nulo — el conteo incluye algunas filas
fuera del universo de 4,830 del reporte agregado, sin impacto en esta métrica de delta).

| Umbral de movimiento en p_home | Juegos | % del total |
|---|---|---|
| > 0.5pp | 203 | 4.18% |
| > 1.0pp | 1 | 0.02% |
| > 2.0pp | 0 | 0.00% |

Delta máximo observado: 1.058pp. Delta medio (todos los juegos): 0.186pp.

## Lectura

Movimiento acotado y explicable: barrel es uno de tres factores del composite (peso 0.20,
`BARREL_WEIGHT` en `tte_formula.py`), y el fix solo re-escala su regresión de shrinkage
(numerador, prior, n) — no toca xwOBA (peso 0.60) ni plate discipline (peso 0.20). Que el 95.8%
de los juegos se mueva ≤0.5pp en p_home, y prácticamente ninguno (1/4859) más de 1pp, es
consistente con un fix de precisión en un componente de peso parcial, no con un cambio
estructural del modelo. Accuracy sube +0.17pp, Brier se mueve +0.00001 (esencialmente plano),
ROI se mueve en ambas direcciones por bucket de edge (mejora en edge≥8%, empeora en edge≥5%/10%)
— exactamente el patrón que el propio roadmap anticipó como legítimo ("un fix de precisión
puede mover el Brier en cualquier dirección legítimamente").

## Decisión

**Pendiente de aprobación explícita del dueño antes de commitear 5c** (no auto-aprobado por
"mejora", no auto-revertido por "empeora" en algunos buckets — el propio roadmap prohíbe ambos
atajos).
