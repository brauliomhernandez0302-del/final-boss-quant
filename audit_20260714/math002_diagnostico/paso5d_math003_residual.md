# PASO 5d — MATH-003: análisis residual de home-win-probability (análisis, sin implementar)

**Fecha**: 2026-07-18. Corrido sobre el baseline post-5c (`data/predictions_history.db`,
columnas `backtest_*`, comando: `python3 scripts/math003_home_win_residual.py`).

## Metodología (repetible, script en `scripts/math003_home_win_residual.py`)

Mismo diagnóstico original (`hfa_engine.py`, docstring del módulo, "Added 2026-07-11"): compara
`backtest_p_home_raw` (probabilidad RAW del modelo, antes de Platt — no `backtest_p_home`
calibrado, por la misma razón que el docstring original ya daba: Platt difiere
estructuralmente entre temporadas, 2024 corre en identidad, 2025 no) contra `home_won` real,
por temporada.

Para el escenario "sin el término uniforme", NO se usa Skellam como probabilidad absoluta —
verificado empíricamente antes de escribir el script que Skellam(λ_home, λ_away) diverge
sistemáticamente del `p_home_raw` real (~-5.8pp de sesgo medio en una muestra de 2024,
consistente con que el motor real usa Negative-Binomial, no Poisson independiente). En cambio,
Skellam se usa solo como **sensibilidad marginal** (exactamente el lenguaje del docstring
original, "Skellam sensitivity"): `delta = Skellam_p_home(λ_home, λ_away) −
Skellam_p_home(λ_home/1.028, λ_away)` por juego, y `p_home_raw_sin_termino = p_home_raw_real −
delta`. Esto cancela la mayor parte del sesgo sistemático Skellam-vs-MC real, porque es una
resta, no un valor absoluto.

**Caveat de conteo**: la consulta encuentra 2,429/2,430 juegos con `backtest_lambda_home` no
nulo para 2024/2025, vs. 2,415 en el reporte agregado oficial — 14-15 juegos de más en la tabla
cruda por temporada, probablemente filtrados del reporte agregado por algún criterio de calidad
de datos no investigado aquí (fuera de alcance de este análisis). No afecta la validez
direccional del resultado.

## Resultado

| Temporada | Home win rate real | p_home_raw CON término (actual) | Residual CON | p_home_raw SIN término (reestimado) | Residual SIN |
|---|---|---|---|---|---|
| 2024 (n=2429) | 52.16% | 52.08% | **+0.08pp** | 50.46% | +1.70pp |
| 2025 (n=2430) | 54.28% | 52.81% | **+1.47pp** | 51.18% | +3.10pp |

## Lectura

El término (`_UNIFORM_HOME_MULT=0.028`) SÍ está cerrando una fracción real y consistente del
gap en ambas temporadas — la reducción que produce (≈1.62pp en 2024, ≈1.63pp en 2025) es casi
idéntica entre temporadas, exactamente lo que se espera de un multiplicador fijo sobre λ en
rangos de λ similares. **Pero el gap subyacente que había que cerrar NO es igual entre
temporadas** (1.70pp en 2024 vs. 3.10pp en 2025, prácticamente el doble) — así que un solo
valor uniforme no puede cerrar ambas por diseño. El resultado neto: **2024 queda
excelentemente calibrado (residual +0.08pp, esencialmente cerrado)**, mientras que **2025
retiene un residual real de +1.47pp de sub-predicción de probabilidad de home-win, sin
corregir**.

Esto contradice parcialmente la premisa original del docstring de `hfa_engine.py` ("un shift
uniforme... presente en ambas temporadas" con magnitud "~1.6-1.7pp" citada para AMBAS) — el
residual real hoy (con datos post-5b/5c, un baseline distinto y más limpio que el de
2026-07-11) muestra que el gap de 2025 es sustancialmente mayor que el de 2024, no igual.

## Veredicto — SIN implementar

**El residual SÍ implica otro valor** (uno solo no cierra ambas temporadas por igual) — por
regla explícita de este paso, **no se cambia `_UNIFORM_HOME_MULT`**. Se reporta como decisión
abierta del dueño:

- Si se prioriza cerrar el residual promedio: un valor más alto (aprox. 0.028 × 3.10/1.63 ≈
  0.053 cerraría 2025) sobre-corregiría 2024 (produciría un residual negativo ahí, es decir,
  sobre-predicción de home-win en 2024).
- Alternativas que el dueño podría considerar (no evaluadas en profundidad aquí, fuera de
  alcance de este análisis): (a) dejar 0.028 como está — es un buen compromiso promedio, cierra
  2024 casi exactamente y deja a 2025 con un residual menor al que había antes del fix; (b) un
  valor intermedio que minimice el error cuadrático combinado entre ambas temporadas; (c)
  investigar POR QUÉ 2025 tiene un gap real casi el doble que 2024 antes de tocar la constante
  — podría ser una señal real específica de 2025 (cambio de reglas, composición de calendario,
  etc.) y no solo ruido de muestra, en cuyo caso "uniforme" (no por temporada) podría ser la
  premisa equivocada, no el valor numérico.
- Ninguna de estas se implementa en este paso.

## Ride-along: hallazgo incidental sobre `backtest_stage_factors_json`

Durante este análisis se notó que `backtest_stage_factors_json.home_hfa` está fijo en `1.0` en
absolutamente todos los juegos muestreados (`away_hfa` sí varía por travel fatigue) — es decir,
este campo de provenance NO refleja el `hfa_mult=1.028` real que `HFAEngine.get_adjusted_lambdas()`
sí aplica (confirmado leyendo el código fuente, y confirmado indirectamente por este mismo
análisis: si `home_hfa` reflejara el mult real, no habría sido necesario reconstruirlo vía
`backtest_lambda_home / 1.028`, hubiera bastado leer el campo). Esto es un campo de
**provenance/diagnóstico desactualizado, no un bug del pipeline real** (el λ que efectivamente
alimenta Monte Carlo sí incluye el mult, confirmado por el propio código de `hfa_engine.py`) —
pero significa que cualquier análisis futuro que confíe en `stage_factors_json` para
reconstruir el desglose por etapa de λ_home subestimará silenciosamente el HFA. Reportado aquí
como hallazgo incidental, fuera de alcance de este paso arreglarlo.
