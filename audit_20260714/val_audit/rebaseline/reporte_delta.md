# Rebaseline post-VAL — delta vs. canónico fase2b

**Fecha**: 2026-07-25. **Estado**: **APROBADO POR EL DUEÑO** — rebaseline commiteado, `CLAUDE.md`
actualizado con el baseline nuevo (0.24675 / 55.05%) y con el framing textual obligatorio:
*"baseline del motor corregido, neutro en moneyline — la justificación del fix vive en
runline/total, que este backtest no mide"*. `engine_commit` de `docs/PROTOCOLO_CLV_V1.md`
re-apuntado a ese commit (D0 seguía pendiente, así que no se reinició ninguna muestra primaria).

## Comando

```bash
python3 backtest_and_retrain.py \
  --season 2024,2025 --use-full-pit \
  --pitcher-pit-cache-db data/pit_cache_pitcher.db \
  --team-tte-pit-cache-db data/pit_cache_merged.db \
  --defense-pit-cache-db data/pit_cache_merged.db \
  --bullpen-pit-cache-db data/pit_cache_merged.db \
  --report-dir audit_20260714/val_audit/rebaseline
```

- **Nuevo**: `audit_20260714/val_audit/rebaseline/backtest_report_20260725_0644.json`
- **Canónico (comparación)**: `audit_20260714/fase2b/gate_delta/backtest_report_20260719_1420.json`
- Backup pre-corrida: `data/predictions_history_backup_pre_valrebaseline_20260725_062930.db`
- Snapshot pre-corrida de `backtest_p_home`: `pre_run_p_home_snapshot.csv` (4,859 filas)
- El backtest usa `model_walkoff=True` por default — el simulador nuevo entra sin flags extra
  (`backtest_and_retrain.py:1949` no pasa el parámetro).

## 1. Titulares

| Métrica | Canónico (07-19) | Nuevo (07-25) | Δ | Dirección |
|---|---|---|---|---|
| Brier (modelo) | 0.24650 | **0.24675** | **+0.00025** | peor |
| Accuracy | 54.80% | **55.05%** | **+0.25pp** | mejor |
| LogLoss | 0.68626 | 0.68681 | +0.00055 | peor |
| Brier vs. random | +1.4% | +1.3% | −0.1pp | peor |
| Juegos | 4,825 | 4,825 | 0 | — |

**Brier subió, accuracy subió.** No es la mejora esperada; tampoco un deterioro claro. Ver §6.

## 2. Por temporada

| Temporada | N | Acc canónico | Acc nuevo | Δ acc | Brier canónico | Brier nuevo | Δ Brier |
|---|---|---|---|---|---|---|---|
| 2024 | 2,412 | 54.77% | 55.10% | **+0.33pp** | 0.24769 | 0.24822 | **+0.00053** (peor) |
| 2025 | 2,413 | 54.83% | 54.99% | **+0.16pp** | 0.24532 | 0.24528 | **−0.00004** (plano) |

Todo el deterioro de Brier viene de 2024. 2025 queda esencialmente idéntico.

## 3. Contra Pinnacle

| Métrica | Canónico | Nuevo | Δ |
|---|---|---|---|
| Brier Pinnacle | 0.24052 | 0.24052 | 0 |
| Brier modelo vs. Pin | −2.49% | −2.59% | **−0.10pp (peor)** |
| Acuerdo modelo-Pin | 75.47% | 75.79% | +0.32pp |
| edge vs. Pin (media) | −0.909 | −0.850 | +0.059 |
| edge vs. Pin (std) | 7.256 | 7.361 | +0.105 |

El modelo se acercó ligeramente a Pinnacle en dirección (acuerdo +0.32pp) pero se alejó en
calidad probabilística (Brier relativo −0.10pp).

## 4. Calibración por bucket

| Bucket | N canón → nuevo | Pred canón → nuevo | Real canón → nuevo | Gap canónico | Gap nuevo |
|---|---|---|---|---|---|
| <40% | 213 → 234 | 35.6 → 35.4 | 47.4 → 46.2 | **+11.8** | **+10.8** |
| 40-45% | 455 → 447 | 42.8 → 42.8 | 42.9 → 43.8 | +0.1 | +1.0 |
| 45-50% | 1102 → 1082 | 47.8 → 47.8 | 50.1 → 49.4 | +2.3 | +1.6 |
| 50-55% | 1397 → 1387 | 52.5 → 52.5 | 53.7 → 54.0 | +1.2 | +1.5 |
| 55-60% | 992 → 980 | 57.2 → 57.2 | 55.1 → 55.5 | −2.1 | −1.7 |
| 60-70% | 603 → 622 | 63.3 → 63.5 | 63.5 → 62.7 | +0.2 | **−0.8** |
| >70% | 63 → 73 | 73.0 → 73.6 | 68.3 → 67.1 | −4.7 | **−6.5** |

Mixto: el bucket de underdogs extremos (<40%) mejora ~1pp, los buckets centrales mejoran
marginalmente, pero **la cola de confianza alta empeora** — el bucket >70% pasa de −4.7 a
−6.5pp de sobreconfianza, y creció de 63 a 73 juegos. Consistente con el problema de
sobreconfianza en edge alto que VAL-5.3 dejó explícitamente abierto.

## 5. ROI por bucket de edge (flat 1u, mejor lado @ Pinnacle)

| Umbral | Bets canón → nuevo | ROI canónico | ROI nuevo | Δ ROI |
|---|---|---|---|---|
| edge≥0% | 4600 → 4600 | −2.55% | −2.65% | **−0.10pp** |
| edge≥2% | 3554 → 3557 | −2.93% | **−2.14%** | **+0.79pp** |
| edge≥5% | 2159 → 2166 | −2.82% | −3.29% | **−0.47pp** |
| edge≥8% | 1153 → 1158 | −4.07% | **−3.45%** | **+0.62pp** |
| edge≥10% | 679 → 698 | −6.81% | **−9.00%** | **−2.19pp** |

Sin patrón monótono: mejora en ≥2% y ≥8%, empeora en ≥0%, ≥5% y (fuerte) ≥10%. **Los 5 buckets
siguen negativos en ambas corridas** — ni antes ni ahora hay ROI positivo en moneyline.
El bucket ≥10% (n=698) es el más ruidoso; −2.19pp sobre 698 apuestas es del orden de la
variación muestral, no una señal limpia.

## 6. Distribución de movimiento de `p_home` (4,859 juegos comunes)

| Umbral de movimiento | Juegos | % |
|---|---|---|
| > 0.05pp | 4,214 | 86.7% |
| > 0.10pp | 3,639 | 74.9% |
| > 0.50pp | 1,007 | 20.7% |
| > 1.00pp | 208 | 4.3% |
| > 2.00pp | 0 | **0.0%** |

- media \|Δ\| = **0.318pp**, mediana 0.223pp, p90 0.767pp, p99 1.341pp, **máx 1.816pp**
- media con signo = **+0.055pp** (`p_home` medio 0.52331 → 0.52387)
- **Pre-Platt (`p_home_raw`)**: media con signo **+0.137pp**, media \|Δ\| **0.467pp**

**Lectura clave**: el movimiento es pequeño y acotado — ningún juego se movió más de 1.82pp.
Compárese con el gate de fase2b (B2), donde 57.2% de los juegos se movieron >0.5pp; acá solo
20.7%. El fix del simulador es una corrección **distribucional**, no un cambio de nivel.

## 7. Por qué el Brier de moneyline no es el instrumento que mide este fix

El hallazgo VAL-1.3 (walk-off) se cuantificó sobre **margen y total**:
P(margen local ≥2 | ganó) sobreestimado +8.72pp, P(total >9) sobreestimado +5.11pp, y home runs
condicional a victoria +0.568 carreras de más. Ese sesgo vive en la **cola** de la distribución
de carreras — exactamente donde salen `RUNLINE HOME −1.5` y `OVER`.

`backtest_and_retrain.py` **solo evalúa moneyline** (Brier/accuracy/ROI sobre `home_won`).
El signo de quién gana es casi insensible a truncar la baja del 9no: si el home ya iba arriba,
truncar no cambia que ganó. Por construcción, este backtest **no puede medir** la mejora que el
fix pretende. Lo que sí mide es el efecto colateral sobre moneyline — y ese efecto es
esencialmente neutro (+0.25pp accuracy, +0.00025 Brier, ambos dentro del rango de un cambio
distribucional menor).

**Consecuencia honesta**: no hay evidencia en esta corrida de que el fix sea una mejora, ni de
que sea un deterioro, *en moneyline*. La evidencia de que corrige un sesgo real sigue siendo la
del reporte VAL original (881 juegos reales, brecha real-vs-modelo bajó de ~8.7pp a ~0.9pp en el
test permanente) — y esa evidencia es sobre mercados que este backtest no evalúa.

## 8. `promote_calibration.py` — DECISIÓN TOMADA: no se corre

**Estado: decidido por el dueño el 2026-07-25, no es un pendiente.** `promote_calibration.py`
**no se corrió** y no se corre como parte de este rebaseline.

**Razón operativa de la decisión: la (b) de abajo** — *lo que promovería no es lo que hace
falta*. No hay una Platt de backtest 2026 fresca que promover; el comando movería una temporada
que no es la que corre en producción. Las razones (a) y (c) siguen siendo válidas y refuerzan la
misma conclusión, pero la que la decide por sí sola es (b): aunque el protocolo lo permitiera y
la magnitud lo justificara, no existe el objeto a promover.

Las tres razones, como se documentaron originalmente:

**(a) Protocolo.** `CLAUDE.md` y `docs/PROTOCOLO_CLV_V1.md` §"Condiciones de validez" prohíben
   correr `promote_calibration.py` durante la ventana. D0 sigue **pendiente**
   (`PROTOCOLO_CLV_V1.md:50`), así que el reloj no corre — pero promover es precisamente el acto
   que cambia probabilidades live, y merece ser una decisión deliberada tuya, no un paso
   automático detrás de este backtest.

**(b) Lo que promovería no es lo que hace falta** ← *razón operativa de la decisión*. La Platt-1D live vigente es
   `season=2026, state_source='live', a=0.7241 b=0.0758, n=586` (fit 2026-07-19). Esta corrida
   solo re-ajustó Platt para `season=2024/2025, state_source='backtest'`
   (a=0.5630/0.5881, escritas hoy 13:44 UTC — la separación de CHRON-001/002 aguantó, cero
   escrituras a filas `live`). `promote_calibration.py --season N` mueve backtest→live **de esa
   misma temporada**: no existe una Platt de backtest 2026 fresca que promover.

**(c) La magnitud no lo justifica.** El desplazamiento de `p_home_raw` es +0.137pp de media
   (\|Δ\| medio 0.467pp). La Platt-1D live está ajustada con n=586; su propia incertidumbre de
   parámetros es de un orden bastante mayor que ese desplazamiento. Re-ajustar por esto sería
   ruido, no corrección.

**Lo que sí correspondería, si apruebas el simulador nuevo**: dejar que la Platt-1D live de
2026 se re-ajuste sola conforme se acumulen juegos con el simulador nuevo, y **fijar D0 después**
de ese cambio de motor — no antes. Arrancar la ventana con el motor viejo y cambiarlo a mitad es
exactamente el escenario que el protocolo dice que reinicia la muestra primaria.

## 9. Qué falta para decidir bien

Este backtest no responde la pregunta que el fix pretende contestar. Si quieres evidencia real
del efecto sobre runline/total, hace falta un evaluador de esos mercados sobre
`actual_home_runs`/`actual_away_runs` (ambos ya están en `game_outcomes`) — no existe hoy en
`backtest_and_retrain.py`. Es trabajo acotado y no toca el camino de predicción de ML.

## Cierre (2026-07-25)

Aprobado y commiteado. `CLAUDE.md` cita ahora 0.24675/55.05% como baseline vigente, con la ruta
del JSON canónico y el framing textual obligatorio de §7 (que ninguna prosa futura reclame una
mejora que este instrumento no puede mostrar). `engine_commit` re-apuntado en
`docs/PROTOCOLO_CLV_V1.md`. Suite completa en verde: 656 passed, incluidos los 17 de
`tests/verification/test_val_audit_invariants.py` (los 2 escritos rojos a propósito contra
VAL-1.3/VAL-1.4 ahora pasan).

Escrituras de esta sesión: este reporte, el JSON del backtest, el CSV de snapshot, el log, el
backup de la DB, y las columnas `backtest_*` + `ml_state(state_source='backtest')` que la corrida
actualizó por diseño. Cero escrituras a filas `live` — la separación de CHRON-001/002 aguantó.

**Siguiente trabajo** (§9): el evaluador distribucional de runline/total sobre las
`backtest_lambda_*` guardadas — `audit_20260714/val_audit/derived_eval/`.
