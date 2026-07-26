# Evaluador distribucional de mercados derivados — runline y total

**Fecha**: 2026-07-26
**Script**: `audit_20260714/val_audit/derived_eval/evaluate_derived_markets.py` (repetible)
**Datos**: `game_outcomes.backtest_lambda_home/away` de la corrida canónica del rebaseline
(`backtest_run_at = 2026-07-25`), 4,825 juegos, temporadas 2024 y 2025 — exactamente el mismo
conjunto que `audit_20260714/val_audit/rebaseline/backtest_report_20260725_0644.json`.
**Salidas**: `results/per_game_20260726_1308.csv` (una fila por juego, ambas variantes),
`results/metrics_20260726_1308.json` (todas las agregaciones: overall, por temporada, por mes).

---

## 0. Qué es y qué NO es este número

**CALIBRACIÓN, NO ROI.** No existen líneas históricas de runline ni de total en esta DB —
`game_outcomes` solo guarda moneyline (`ml_*_open/cons/pin`). Sin línea y sin precio no hay
"cubrir el spread", no hay EV y no hay ROI; cualquier cifra de rentabilidad en estos mercados
sería inventada. Todo lo que sigue mide **qué tan bien la distribución simulada de carreras
describe los resultados reales**: sesgo, Brier, ECE, pendiente de calibración, RPS y PIT.

Un modelo puede mejorar su calibración y aun así no ser rentable contra el mercado. Esa pregunta
necesita líneas históricas de estos mercados y **este instrumento no la responde** — igual que
`backtest_and_retrain.py` no puede responder si el fix del simulador mejoró algo, que es
justamente por qué este evaluador existe.

**Por qué hacía falta**: el rebaseline del 2026-07-25 quedó explícitamente etiquetado como
*"baseline del motor corregido, neutro en moneyline — la justificación del fix vive en
runline/total, que este backtest no mide"*. Este es el instrumento que sí mide esa parte.

## 1. Método

Las λ son **entrada** del simulador, no salida: los tres cambios de b3325a5 (truncamiento de
walk-off, `rho_game` real, empates proporcionales) viven enteramente dentro del Monte Carlo. Por
eso re-simular sobre las λ ya guardadas aísla el efecto del fix de forma exacta — ninguna λ se
recalcula acá, ningún engine se ejecuta.

- **Sim viejo**: `b3325a5^:modules/baseball_module/montecarlo/simulator.py`, vendorizado en
  `_vendor/simulator_pre_b3325a5.py`. El script verifica en cada corrida que el vendor sea
  byte-idéntico a lo que git tiene en ese commit, y aborta si no lo es.
- **Sim nuevo**: el del working tree (`WALKOFF_9TH_SHARE=1/9`, `NB_DISPERSION=6.0`).
- **Pareado**: mismo `rng_seed = game_pk % 2**32` (el mismo esquema de `backtest_and_retrain.py`)
  en ambas variantes, 50,000 simulaciones por juego y por variante (el `N_MC` del backtest). El
  delta entre variantes no arrastra ruido de muestreo ni de selección.
- **Walk-forward**: las λ vienen de la corrida walk-forward del backtest (cada juego predicho
  solo con datos anteriores a su `official_date`, post-Fase 2B), así que la propiedad se hereda.
  Además, acá **no se ajusta ningún parámetro contra los resultados** — no hay Platt, ni bias, ni
  fit de ninguna clase; todas las probabilidades salen de la simulación cruda. No hay superficie
  de sobreajuste. Los cortes por temporada y por mes se reportan igual, para que una mejora
  agregada no pueda tapar deriva local (§5).

Mercados evaluados: runline ±1.5 de cada lado, totales sobre 6.5/7.5/8.5/9.5/10.5/11.5 (medias
líneas, sin push), y moneyline como **control** — debe moverse poco, y así fue.

---

## 2. Titular

| Métrica | Real | Sim viejo | Sim nuevo | Veredicto |
|---|---:|---:|---:|---|
| Total medio (carreras) | 8.837 | 9.112 (+0.275) | **8.850 (+0.013)** | el sesgo de total casi desaparece |
| Margen medio (local−visita) | +0.028 | +0.250 (+0.222) | **−0.011 (−0.039)** | el margen fantasma del local se corrige |
| P(margen local ≥2 \| ganó el local) — **VAL-1.3** | 0.6663 | 0.7820 (**+11.58pp**) | **0.6968 (+3.05pp)** | cierra ~3/4 de la brecha |
| Desv. típica del total | 4.455 | 4.144 | 4.038 | sub-dispersión, y el fix la empeora un poco (§4) |
| Desv. típica del margen | 4.501 | 4.150 | 3.871 | ídem, más marcado |

El fix hace lo que dice hacer, en la dirección y con el orden de magnitud que se le atribuyó —
pero **no cierra la brecha del todo**: quedan +3.05pp de sobre-predicción del margen ≥2
condicional a que el local ganó.

### 2.1 Reconciliación: ¿0.9pp o 3.05pp?

**El número canónico de VAL-1.3 post-fix es +3.05pp, el de este evaluador.** El ~0.9pp que cita
el reporte del fix (`audit_20260714/val_audit/reporte.md`, sección "Validación") no está mal —
mide **otra cosa**, y las dos se reconcilian exactamente:

| | reporte del fix (~0.9pp) | este evaluador (+3.05pp) |
|---|---|---|
| Muestra | ventana de λ del hallazgo: `5.0≤λ_h≤7.5`, `2.5≤λ_a≤4.5` (N≈881) | **los 4,825 juegos**, sin restricción de λ |
| Simulador | NB puro: sin ruido de λ, sin `rho_game`, sin Platt | configuración **de producción** completa |
| Sims por juego | 300–500 | 50,000 |
| Rol | test de regresión sobre la rebanada donde se detectó el sesgo | medición poblacional |

La diferencia es de **estimando, no de resultado**: la ventana original selecciona juegos con el
local muy favorito, que es justo donde el truncamiento muerde más fuerte y donde el fix por lo
tanto luce mejor. Corriendo **este mismo evaluador restringido a esa ventana de λ** (N=813 con
las λ de la corrida canónica actual):

| | viejo | nuevo |
|---|---:|---:|
| Ventana λ del hallazgo (N=813) | +8.90pp | **+1.23pp** |
| Población completa (N=4,825) | +11.58pp | **+3.05pp** |

Es decir: el evaluador **reproduce** el par ~8.7pp → ~0.9pp del reporte del fix cuando se le pide
la misma rebanada (+8.90 → +1.23, con la diferencia residual explicada por las 68 juegos de
membresía distinta —N=813 vs 881, las λ guardadas por el rebaseline no son idénticas a las que
leyó la auditoría— y por el ruido de MC de 300-500 sims). Sobre la población completa la brecha
es mayor, y esa es la que hay que citar: **para cualquier afirmación sobre runline/total, el
número canónico es el de este evaluador sobre los 4,825 juegos**, no el de la rebanada.

## 3. Por mercado (4,825 juegos)

Sesgo = probabilidad media predicha − frecuencia real, en puntos porcentuales.

| Mercado | Real | Sesgo viejo | Sesgo nuevo | Brier viejo | Brier nuevo | ECE viejo | ECE nuevo |
|---|---:|---:|---:|---:|---:|---:|---:|
| RL_HOME −1.5 | 0.3550 | +1.35pp | **−2.66pp** | 0.22781 | 0.22807 | 0.0297 | 0.0282 |
| RL_AWAY −1.5 | 0.3585 | −3.48pp | −3.41pp | 0.23049 | 0.23045 | 0.0449 | 0.0433 |
| TOTAL >6.5 | 0.6721 | +4.23pp | **+2.46pp** | 0.22155 | 0.22061 | 0.0449 | 0.0319 |
| TOTAL >7.5 | 0.5575 | +5.72pp | **+3.22pp** | 0.24873 | 0.24669 | 0.0600 | 0.0359 |
| TOTAL >8.5 | 0.4856 | +2.80pp | **+0.44pp** | 0.24945 | 0.24868 | 0.0345 | 0.0281 |
| TOTAL >9.5 | 0.3847 | +3.27pp | **+0.57pp** | 0.23707 | 0.23592 | 0.0375 | 0.0311 |
| TOTAL >10.5 | 0.3215 | +0.90pp | −1.43pp | 0.21830 | 0.21832 | 0.0271 | 0.0293 |
| TOTAL >11.5 | 0.2458 | +0.96pp | −1.36pp | 0.18621 | 0.18605 | 0.0291 | 0.0287 |
| ML local *(control)* | 0.5328 | −0.88pp | −0.75pp | 0.24707 | 0.24743 | 0.0285 | 0.0311 |

Tests pareados de Brier por juego (viejo − nuevo, positivo = mejora el nuevo):

| Mercado | Δ Brier | p | juegos mejor / peor |
|---|---:|---:|---:|
| TOTAL >7.5 | **+0.00205** | 3.4e-08 | 2,690 / 2,135 |
| TOTAL >9.5 | **+0.00115** | 3.6e-03 | 1,856 / 2,969 |
| TOTAL >6.5 | **+0.00094** | 2.1e-04 | 3,244 / 1,581 |
| TOTAL >8.5 | **+0.00077** | 3.0e-02 | 2,343 / 2,482 |
| TOTAL >10.5 | −0.00002 | 0.96 | 1,551 / 3,274 |
| TOTAL >11.5 | +0.00016 | 0.60 | 1,186 / 3,639 |
| RL_HOME −1.5 | −0.00026 | 0.64 | 1,713 / 3,112 |
| RL_AWAY −1.5 | +0.00004 | 0.38 | 2,577 / 2,237 |
| ML local *(control)* | −0.00037 | 8.1e-06 | 2,235 / 2,590 |

**Lectura honesta, mercado por mercado**:

1. **Totales: mejora real y significativa** en el rango donde se juegan (7.5–9.5). El sesgo de
   +5.72pp del viejo en la línea 7.5 baja a +3.22pp, y en 8.5/9.5 prácticamente desaparece
   (+0.44pp / +0.57pp). ECE mejora en 4 de 6 líneas, con Brier pareado significativo en las
   cuatro líneas centrales. Esta es la evidencia positiva más sólida del reporte.
2. **Runline local: el fix se pasa de largo.** El viejo sobre-predecía por +1.35pp; el nuevo
   sub-predice por −2.66pp. El Brier queda plano (−0.00026, p=0.64) — no es un deterioro
   medible, pero el sesgo **cambió de signo y creció en magnitud**. El sospechoso obvio es
   `WALKOFF_9TH_SHARE` (constante asumida, no medida) — se midió en §6, y resulta que no alcanza
   para explicarlo.
3. **Runline visitante: sin cambio** (−3.48 → −3.41pp), como debe ser: la truncación solo toca
   las carreras del local. Ese −3.4pp que sobrevive intacto en ambas variantes **no es de este
   fix** — es sub-dispersión general (§4) y estaba ahí desde antes.
4. **Moneyline (control): se movió poco y hacia peor**, −0.00037 de Brier (p=8e-06,
   estadísticamente sólido pero minúsculo). Coincide exactamente con lo que reportó el
   rebaseline (Brier +0.00025 peor en la corrida completa con Platt). Sirve de sanity check: el
   instrumento reproduce la neutralidad ya conocida en ML y detecta el efecto en los mercados
   derivados, que es la única razón por la que existe.

## 4. Distribución completa (PIT y RPS) — el residual dominante NO es el walk-off

| | viejo | nuevo |
|---|---:|---:|
| RPS total | 2.47007 | **2.46399** |
| RPS margen | **2.49288** | 2.49773 |
| PIT total: media (0.5 = calibrado) | 0.4747 | **0.4915** |
| PIT total: KS | 0.0596 | **0.0429** |
| PIT margen: media | 0.4897 | 0.5057 |
| PIT margen: KS | **0.0420** | 0.0483 |
| PIT margen: χ² (10 bins) | **136.0** | 261.8 |

El PIT del **total** mejora claramente (media 0.4747 → 0.4915, KS 0.0596 → 0.0429): la
distribución simulada del total quedó bastante mejor centrada. El PIT del **margen** cruza el
centro (0.4897 → 0.5057) pero su χ² empeora, y los bins dicen por qué:

```
margen, share por decil de PIT (uniforme = 0.100 en cada uno)
viejo  0.134 0.107 0.094 0.089 0.070 0.091 0.118 0.094 0.093 0.110
nuevo  0.134 0.105 0.095 0.089 0.061 0.076 0.103 0.094 0.100 0.144
```

Forma de U con las dos colas sobre-pobladas = **la distribución simulada es demasiado angosta**.
Los números lo confirman directo:

| | sim viejo | sim nuevo | real |
|---|---:|---:|---:|
| desv. típica del total | 4.144 | 4.038 | **4.455** |
| desv. típica del margen | 4.150 | 3.871 | **4.501** |

El margen simulado tiene **14% menos dispersión** que el real, y el truncamiento la reduce un
poco más (4.150 → 3.871) porque recortar la baja del 9no quita varianza además de quitar media.
Es exactamente la firma que se ve en las líneas extremas de total: sobre-predicción en 6.5
(+2.46pp) y sub-predicción en 11.5 (−1.36pp) con el centro ya bien calibrado.

**Esto no es un bug nuevo de b3325a5 — es `NB_DISPERSION = 6.0`.** El comentario del propio
simulador documenta que `r=3.0` ajustaba mucho mejor la cola cruda (var/mean 2.26 real vs 1.75
con r=6.0) y fue **rechazado a propósito** porque empeoraba las métricas de apuesta en
*moneyline* (ROI edge≥10% −5.70pp). Es decir: se eligió deliberadamente un peor ajuste
distribucional a cambio de mejor moneyline. Lo que este evaluador aporta es el precio de esa
decisión, ahora cuantificado en los mercados que sí dependen de la forma: −9% de dispersión en
el total, −14% en el margen, y un PIT en U que ninguna corrección de la media puede arreglar.

**No se cambió nada.** `NB_DISPERSION` sigue en 6.0. Es una decisión abierta del dueño, y ahora
tiene dos números en vez de uno: lo que r=6.0 gana en moneyline (ya medido en su momento) y lo
que cuesta en runline/total (este reporte).

Dos limitaciones estructurales, ninguna introducida por este fix, que ponen un piso a lo que
cualquier calibración de totales puede lograr acá:

- **Las entradas extra no se modelan.** Un empate en la simulación (10.5% de los casos) se queda
  en 9 innings; el juego real sigue y anota más. El total simulado está truncado por abajo en
  ese ~10% de juegos, y no hay forma de arreglarlo sin modelar el inning extra (con el corredor
  automático en segunda desde 2020, además, no es un inning normal).
- **El truncamiento de mitad de inning no se modela** (solo el caso "media entrada entera no se
  juega"), como ya dice el docstring del simulador.

## 5. Estabilidad temporal (cortes walk-forward)

Por temporada:

| | 2024 (n=2,412) | 2025 (n=2,413) |
|---|---|---|
| VAL-1.3 gap: viejo → nuevo | +10.33pp → **+1.78pp** | +12.78pp → **+4.28pp** |
| Total medio: real / viejo / nuevo | 8.779 / 9.059 / **8.802** | 8.896 / 9.164 / **8.899** |
| RL_HOME sesgo: viejo → nuevo | +1.07pp → −2.91pp | +1.62pp → −2.40pp |
| TOTAL >7.5 ECE: viejo → nuevo | 0.0532 → **0.0286** | 0.0701 → **0.0447** |

Por mes (sesgo en pp, viejo → nuevo):

| mes | n | RL_HOME | TOTAL >8.5 |
|---|---:|---|---|
| 2024-04 | 397 | +2.03 → −1.77 | +6.69 → +4.51 |
| 2024-05 | 409 | −3.02 → −6.98 | +2.89 → +0.57 |
| 2024-06 | 401 | −2.60 → −6.64 | +1.02 → −1.36 |
| 2024-07 | 369 | +3.95 → −0.09 | −2.46 → −4.88 |
| 2024-08 | 413 | +3.54 → −0.49 | +1.19 → −1.15 |
| 2024-09 | 385 | +1.72 → −2.30 | +6.79 → +4.43 |
| 2025-03 | 50 | −8.06 → −12.06 | +2.38 → −0.00 |
| 2025-04 | 391 | −2.17 → −6.30 | +4.14 → +1.60 |
| 2025-05 | 411 | +6.39 → +2.39 | +5.79 → +3.35 |
| 2025-06 | 397 | +2.92 → −1.05 | +2.93 → +0.60 |
| 2025-07 | 369 | −0.94 → −4.96 | −1.41 → −3.74 |
| 2025-08 | 421 | +2.57 → −1.45 | +0.18 → −2.14 |
| 2025-09 | 374 | +1.75 → −2.28 | +7.58 → +5.22 |

El efecto del fix es **un desplazamiento uniforme de ≈−4pp en RL_HOME en los 13 meses**, sin una
sola excepción — la firma de una constante, no de una interacción con el calendario. El swing
mes a mes (±6pp) es del tamaño esperado por muestreo con n≈400 (SE ≈ 2.4pp sobre una tasa de
0.355): es ruido, no deriva. La mejora en totales también aparece en todos los meses; su
magnitud sí varía (los meses con más sesgo previo son los que más ganan), lo cual es lo esperado
de una corrección de sesgo.

## 6. `WALKOFF_9TH_SHARE` — barrido de sensibilidad (REPORTE, no cambio)

`WALKOFF_9TH_SHARE = 1/9 ≈ 0.1111` es una **constante asumida**: no existe en este repo un split
real de carreras por inning, y el propio commit del fix la nombró explícitamente para que una
calibración futura tuviera dónde enchufar el número real. El residual de −2.66pp en RL_HOME
apunta directo a ella, así que se midió — con la misma regla que MATH-003: *un residual que
implica otro valor se reporta, no se cambia*.

Barrido sobre los mismos 4,825 juegos, mismos seeds, 30,000 sims por juego, parcheando la
constante **en memoria** (nada persiste; el código sigue en 1/9):
`python3 evaluate_derived_markets.py --walkoff-sweep 0,0.04,0.06,0.08,0.1111,0.13 --n-sims 30000`

Resultados (`results/walkoff_sweep_20260726_1317.json`):

| `WALKOFF_9TH_SHARE` | sesgo RL_HOME | brecha VAL-1.3 | total medio (sesgo) | margen medio (sesgo) |
|---|---:|---:|---:|---:|
| 0.0000 *(= sim viejo)* | +1.41pp | +11.67pp | 9.112 (+0.275) | +0.250 (+0.222) |
| 0.0400 | **−0.29pp** | +8.06pp | 9.011 (+0.174) | +0.150 (+0.122) |
| 0.0600 | −1.04pp | +6.48pp | 8.963 (+0.126) | +0.102 (+0.074) |
| 0.0800 | −1.72pp | +5.04pp | 8.918 (+0.081) | +0.056 (**+0.029**) |
| **0.1111 = 1/9 (actual)** | −2.65pp | +3.05pp | 8.850 (**+0.013**) | −0.011 (−0.039) |
| 0.1300 | −3.16pp | **+1.98pp** | 8.812 (−0.025) | −0.049 (−0.077) |

Primero, un chequeo de validez: con `share = 0` el simulador nuevo reproduce el sesgo del viejo
en RL_HOME (+1.41pp con 30K sims vs +1.35pp del viejo con 50K) — la diferencia es ruido de
muestreo, así que el barrido está midiendo lo que dice medir.

**El resultado importante es que ningún valor sirve para todo.** Cada criterio pide uno
distinto (cruce por cero interpolado sobre esta misma tabla):

| criterio | valor que lo cierra |
|---|---|
| sesgo de RL_HOME = 0 | ≈ **0.033** |
| margen medio = real | ≈ **0.093** |
| total medio = real | ≈ **0.118** ← el actual, 1/9, está prácticamente acá |
| brecha VAL-1.3 = 0 | ≈ **0.165** *(extrapolado fuera del rango medido)* |

Los cuatro objetivos están separados por un factor de 5. Subir la constante para cerrar VAL-1.3
llevaría el sesgo de RL_HOME a ≈−5pp; bajarla para cerrar RL_HOME devolvería la brecha de
VAL-1.3 a +8pp, casi la del simulador viejo. **Que un solo escalar no pueda satisfacer los cuatro
es en sí la evidencia de que el residual que queda no es una constante mal puesta** — es la
sub-dispersión de §4, que ninguna elección de `WALKOFF_9TH_SHARE` puede tocar, porque la
truncación mueve la ubicación de la distribución, no su ancho.

**Nada se cambió**: `WALKOFF_9TH_SHARE` sigue en `1.0/9.0` en `montecarlo/simulator.py`. El
valor actual resulta ser, casualmente, el que mejor centra el total medio — el objetivo con más
juegos detrás y el único de los cuatro que no está condicionado a un subconjunto. Si querés otro
criterio, la tabla dice exactamente qué costaría.

## 7. Decisiones (2026-07-26) y lo que sigue abierto

Las dos constantes que este reporte pone en la mesa quedaron **decididas por el dueño el
2026-07-26, con esta evidencia a la vista**. No son pendientes; cada una tiene su disparador de
re-visita escrito para que la próxima vez no se re-litigue desde cero.

### 7.1 `WALKOFF_9TH_SHARE = 1/9` — **DECIDIDA: se queda**

**Decidido con evidencia**, no por defecto ni por inercia: el barrido de §6 es la evidencia, y
dice que ningún valor único cierra los cuatro criterios (están separados por un factor de 5).
1/9 es, de los medidos, el que centra el total medio (+0.013 carreras sobre 4,825 juegos), que es
el objetivo con más datos detrás y el único no condicionado a un subconjunto. Los residuales que
deja —RL_HOME −2.65pp, VAL-1.3 +3.05pp— quedan **aceptados y documentados**, no ignorados.

**Disparador de re-visita: cuando se toque la dispersión.** Es la razón sustantiva y no una
fecha: §4 muestra que el ancho de la distribución está −9% (total) / −14% (margen) respecto al
real, y esos residuales de RL_HOME y VAL-1.3 se estiman sobre esa forma equivocada. Cambiar la
dispersión mueve el óptimo de `WALKOFF_9TH_SHARE` — recalibrar la constante ANTES de eso sería
ajustar un parámetro contra un sesgo que el otro cambio va a mover de todos modos. Re-correr
`--walkoff-sweep` es el primer paso obligatorio después de cualquier cambio de dispersión.

### 7.2 `NB_DISPERSION = 6.0` — **DECIDIDA: se queda durante la ventana de ML**

El valor se mantiene **mientras corra la ventana de evaluación de CLV de moneyline**
(`docs/PROTOCOLO_CLV_V1.md`). Razón: `NB_DISPERSION` está en `montecarlo/simulator.py`, dentro
del camino de predicción congelado; tocarlo a media ventana reinicia la muestra primaria. Y su
valor actual fue elegido *precisamente* por métricas de moneyline, que es lo que la ventana está
midiendo — cambiarlo ahora sería cambiar el objeto bajo medición.

**Disparador de re-visita: el arranque del motor de derivados.** Cuando runline/total dejen de
ser un subproducto y tengan motor propio, la balanza cambia de lado: r=6.0 optimiza el mercado
que ya se está midiendo por otra vía, y r≈3.0-3.5 ajusta la cola que los derivados necesitan.
**Este evaluador es la balanza** — el número que faltaba (lo que r=6.0 cuesta en forma
distribucional: −9%/−14% de dispersión, PIT en U) ya está medido acá, así que esa decisión no
arranca de cero. Corolario práctico: es plausible que la respuesta final no sea un solo r sino
uno por familia de mercado; medir eso es trabajo del motor de derivados, no de este reporte.

### 7.3 Abierto de verdad

**ROI real de runline/total**: sigue sin medirse y no se puede medir con lo que hay en la DB.
Haría falta ingerir líneas históricas de estos mercados (runline y totales con precio). Hasta
entonces, "mejor calibrado" ≠ "rentable", y este reporte no dice lo segundo en ninguna parte.

## 8. Reproducir

```bash
source mi_entorno/bin/activate
python3 audit_20260714/val_audit/derived_eval/evaluate_derived_markets.py           # ~4.5 min
python3 audit_20260714/val_audit/derived_eval/evaluate_derived_markets.py \
        --walkoff-sweep 0,0.04,0.06,0.08,0.1111,0.13 --n-sims 30000                 # ~8 min
```

Solo lee `game_outcomes` (conexión SQLite en modo lectura), no escribe una sola fila en ninguna
DB, no llama a ningún engine de predicción y no toca `ml_state`. Compatible con el congelamiento
del motor de `docs/PROTOCOLO_CLV_V1.md`.
