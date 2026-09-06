# Candidato v1 de P(gana el local) — resultado

**Preregistro**: `docs/PREREGISTRO_MODELO_V1_2026-09-06.md`, commit `59a39b5`,
escrito y commiteado **antes** de medir. **Nada de lo preregistrado se cambió
después de ver los resultados.**

> 📌 **SUPERSEDED en parte por la v1.1** (2026-09-06, mismo día):
> `docs/CONTROL_FUGA_INVARIANCIA_2026-09-06.md` reemplaza la cota de 8 h por el
> fin de partido MEDIDO. La población evaluable no cambia y los comparadores dan
> lo mismo; el Brier del modelo se mueve en la quinta cifra. Este documento se
> conserva entero: su conclusión sigue en pie y la comparación v1 ↔ v1.1 vive en
> el documento nuevo.

> ⚠️ **Evaluación HISTÓRICA, no fuera de muestra.** 2025 y 2026 ya fueron
> explorados exhaustivamente por este proyecto —siete baselines, una auditoría
> de dieciséis informes, nueve motores medidos sobre esos mismos años—. El
> diseño temporal impide que el MODELO vea el futuro; no impide que lo haya
> visto quien eligió las variables. La única prueba limpia posible es hacia
> adelante.

## Lo que se esperaba, escrito de antemano

> *"Se espera que **pierda contra Pinnacle**. […] Lo que se pide de él es que
> **le gane a la tasa base** y que la medición sea reproducible."*

**Es exactamente lo que pasó, en las dos temporadas.**

## El modelo

| | |
|---|---|
| forma | regresión logística, ridge L2 `λ=1.0`, intercepto sin penalizar |
| variables | `dif_pitagorica`, `dif_descanso` |
| corte de cada fila | el `captured_at` de su precio de referencia |
| ventana | últimos 162 partidos terminados por equipo, cruzando temporadas |
| regresión a la media | `k = 67` partidos de liga media |
| estandarización | media y desvío **del entrenamiento** |

Coeficientes (sobre variables estandarizadas):

| evaluación | entrena | n_train | intercepto | `dif_pitagorica` | `dif_descanso` |
|---|---|---|---|---|---|
| 2025 | 2024 | 1.988 | +0,1256 | **+0,2947** | −0,0800 |
| 2026 | 2024+2025 | 4.179 | +0,1641 | **+0,2965** | −0,0312 |

`dif_pitagorica` sale positiva y **estable** entre los dos ajustes, que es lo
que tiene que pasar si mide talento. `dif_descanso` sale **negativa** en los
dos: más descanso del local asociado a menos victorias locales. Es contra la
intuición, es chica y se reporta tal cual — con dos ajustes no alcanza para
llamarla señal ni para descartarla.

## Cobertura y exclusiones

| | n |
|---|---|
| filas predecibles | **5.736** |
| 2024 (sólo entrenamiento) | 1.988 |
| 2025 (evaluación) | 2.191 |
| 2026 (evaluación) | 1.557 |

| exclusión | n | por qué |
|---|---|---|
| `sin_precio_pinnacle_pre_juego` | **1.558** | sin nulo no hay con qué comparar |
| `historial_insuficiente` | **370** | menos de 30 partidos previos disponibles para alguno de los dos equipos — todas de abril de 2024, la primera temporada del almacén |

Ninguna se imputó. Un juego que no se puede predecir con lo disponible se
excluye; rellenarlo con la media lo convertiría en una predicción que no se
hizo.

## La comparación, sobre exactamente los mismos juegos

### 2025 — n = 2.191

| candidato | Brier | log-loss |
|---|---|---|
| **v1** | **0,243241** | 0,679410 |
| tasa base del entrenamiento (0,5307) | 0,247959 | 0,689060 |
| **Pinnacle** (nulo) | **0,241417** | 0,675417 |

### 2026 — n = 1.557

| candidato | Brier | log-loss |
|---|---|---|
| **v1** | **0,248113** | 0,689455 |
| tasa base del entrenamiento (0,5401) | 0,249470 | 0,692088 |
| **Pinnacle** (nulo) | **0,246411** | 0,685983 |

**El Brier del mercado está recalculado sobre la intersección evaluable**, como
corresponde. Resultó **idéntico al del marco completo de cada temporada**
(0,241417 y 0,246411) porque las 370 exclusiones por historial cayeron todas en
2024, que no se evalúa. Es una coincidencia de estos datos, no una propiedad:
se verificó, no se asumió. El **0,242124067** de
`docs/REFERENCIA_MERCADO_2026-09.md` es el agregado de las tres temporadas y no
es la barra de ninguna de estas dos comparaciones.

### Lectura

- **v1 le gana a la tasa base** en las dos temporadas: −0,0047 en 2025 y −0,0014
  en 2026. Poco, pero en la dirección correcta y consistente.
- **v1 pierde con Pinnacle** en las dos: +0,0018 y +0,0017. El mercado sabe
  cosas que dos variables sobre marcadores no pueden saber.
- Ordenado: `tasa base > v1 > Pinnacle`. Es la escalera que se esperaba.

## Veredicto del evaluador

`y ~ logit(mercado) + logit(candidato)`, con bootstrap **agrupado por equipo**:

| temporada | b_candidato | IC95 agrupado | P(b>0) | veredicto |
|---|---|---|---|---|
| 2025 | +0,1922 | **[−0,2361, +0,5707]** | 77,2% | **NO aporta sobre el precio** |
| 2026 | +0,0963 | **[−0,5170, +0,5710]** | 63,5% | **NO aporta sobre el precio** |

El coeficiente es positivo en las dos —a diferencia de `l0` del sistema
anterior, que salía **negativo**— pero el intervalo incluye el cero en las dos.
Sabiendo el precio, v1 no agrega nada demostrable.

## ROI — el que no hay que creerse

| 2025 | n | ROI | IC95 agrupado | P(ROI>0) |
|---|---|---|---|---|
| edge ≥ 2% | 1.628 | +3,37% | [−1,64%, +8,46%] | 89% |
| edge ≥ 4% | 1.099 | +3,99% | [−2,94%, +10,75%] | 85% |
| edge ≥ 6% | 667 | +5,07% | [−4,37%, +14,72%] | 84% |
| edge ≥ 8% | 372 | +2,67% | [−7,08%, +12,48%] | 69% |
| edge ≥ 10% | 197 | −1,96% | [−15,27%, +13,13%] | 38% |

| 2026 | n | ROI | IC95 agrupado | P(ROI>0) |
|---|---|---|---|---|
| edge ≥ 2% | 1.095 | +1,08% | [−5,12%, +6,38%] | 64% |
| edge ≥ 4% | 716 | +5,39% | [−3,23%, +14,08%] | 87% |
| edge ≥ 6% | 447 | +9,19% | [−1,13%, +20,38%] | 96% |
| edge ≥ 8% | 250 | +10,07% | [−4,62%, +25,24%] | 90% |
| edge ≥ 10% | 123 | +2,80% | [−23,54%, +32,02%] | 55% |

**Los diez intervalos cruzan el cero. Ninguno de estos ROI es evidencia de
nada**, y publicarlos sin intervalo habría sido repetir el error más caro del
proyecto: el único indicador de habilidad positivo que tuvo era el vig
disfrazado (+1,514% → −0,497% al desvigorizar).

Tres razones para no creerles, todas del propio catálogo del proyecto:

1. **Muestra.** El curso registra que a n=500 con cuota ~2 el error estándar del
   ROI ronda el 4,5%; probar habilidad por ROI pide **más de 1.000 apuestas** con
   p<0,001. Acá hay entre 123 y 1.628 en una sola temporada ya explorada.
2. **El coeficiente conjunto incluye el cero** en las dos temporadas. Un ROI
   positivo con un aporte no demostrable sobre el precio es la combinación que
   pide más muestra, no una conclusión.
3. **No es monótono.** Si el edge declarado midiera edge real, más edge debería
   dar más ROI. Acá sube y después se cae en los dos años.

## Control positivo de fuga

Se inyectó a propósito el marcador del **propio partido** dentro de
`dif_pitagorica`, una variable permitida, por dos caminos:

| camino | qué pasó |
|---|---|
| **A** · pidiendo el partido por la compuerta (`exigir_disponible`) | **Rechazado por el detector estructural**: *"FUGA: game_pk=745039 recién estaba disponible en 2024-03-29T07:35:00+00:00, y el corte de la fila es 2024-03-28T17:00:00+00:00"* |
| **B** · leyendo el marcador directo, **esquivando** la compuerta | La compuerta no lo vio: se construyeron las 5.736 filas envenenadas. **Rechazado por el detector estadístico**: Brier **0,00002**, muy por debajo del umbral 0,22 |

Los dos detectores hacen falta y hacen cosas distintas: el estructural impide la
fuga por construcción, el estadístico atrapa a quien esquivó la compuerta.

**Límite honesto del segundo**: atrapa fugas groseras. Una fuga sutil que dejara
el Brier en 0,235 pasaría el umbral de 0,22 sin encenderlo. Contra eso sólo
sirve el estructural, y por eso el umbral se eligió permisivo a propósito.

## Qué se reutilizó del sistema anterior

| pieza | de dónde | cómo se demostró su validez temporal |
|---|---|---|
| `regress(observado, media, n, k)` — encogimiento bayesiano | `modules/baseball_module/offense/tte_formula.py` | **Matemática pura**: recibe números, no accede a datos, no conoce fechas. Copiada con atribución (`fbq/` es autónomo por diseño) y vigilada por un test que la compara contra el original mientras exista |
| El día del juego sale de `official_date` | paso 4 de la auditoría | Ya encapsulado en `fbq/core/clock.py`; el 22-24% de los juegos difiere del UTC truncado |
| `ESTADOS_FINALES` | `results/store.py` → `core/identity.py` | Regla de identidad, sin componente temporal |
| Evaluador, bootstrap agrupado y portón | `fbq/evaluator/`, `fbq/features/` | Ya migrados a almacenes propios |
| **Resultados negativos ya pagados**: forma reciente anti-predictiva (paso 8); las nueve señales correlacionan 0,28-0,65 con el mercado y 0,05-0,10 con el resultado | auditoría paso-a-paso | Se reutilizan como **decisiones de diseño**: la forma reciente **no** se incluyó, y eso ahorró volver a medir lo ya medido |

**El código original no se tocó.** Nada de `modules/`, `core/` ni
`backtest_and_retrain.py` se modificó.

## Entregables y versiones

| | |
|---|---|
| detalle por juego | `docs/candidato_v1_2026-09-06.csv` — 3.748 filas |
| resumen | `docs/candidato_v1_2026-09-06.json` |
| código | rama `recuperacion/fbq` |
| `market.db` | 143.899 filas · 2024-03-20T17:00Z → 2026-08-07T23:37Z · 6.295 eventos enlazados |
| `results.db` | 7.664 juegos · 2024-03-20 → 2026-08-06 |

El CSV lleva por juego: `game_pk`, fecha, temporada, equipos, corte,
`inicio_utc`, las dos variables, los partidos previos de cada equipo, `p_v1`,
`p_tasa_base`, `p_mercado`, el resultado y las tres pérdidas de Brier.

## Para qué sirve esto

Es **la barra contra la cual medir los motores recuperados**. Cualquier motor
del sistema anterior que se reconstruya tiene que superar, sobre estos mismos
juegos y con este mismo corte, a un logístico de dos variables sobre marcadores.
Si no lo supera, no se paga a sí mismo.

## Lo que queda abierto, nombrado

1. **BaseRuns** en vez de la pitagórica. El propio curso lo señala como mejor
   estimador de talento. Necesita hits, bases por bolas y bases totales, y
   ninguna vive todavía en los almacenes propios.
2. **El `k` de regresión para carreras.** Se usó 67, que es la constante para
   *victorias*, y el curso dice que el diferencial de carreras regresiona menos.
   Sobre-encoge a propósito: medir el `k` correcto exigiría ajustarlo sobre los
   años que se evalúan.
3. **Las 370 exclusiones por historial** desaparecerían con una temporada más de
   almacén.
4. **La prueba limpia es 2027**, hacia adelante. Todo lo de acá es histórico.
