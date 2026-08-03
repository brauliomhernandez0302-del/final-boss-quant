# Paso 10 — el Kalman de ofensa: evidencia medida, decisión abierta

**Estado: NO SE CAMBIÓ NADA.** `_KALMAN_BLEND` sigue en 0.35. Este documento
guarda la medición para que la decisión se tome con números y no de nuevo desde
cero.

## Qué hace

`learning_engine.get_kalman_lambda_adjustment` mezcla
`0.65·λ_modelo + 0.35·λ_kalman` una vez que hay ≥10 observaciones. El Kalman
observa **carreras reales anotadas** (`update_kalman`, con destruncado de
walk-off para el local).

## El problema conceptual

El TTE produce una λ **merecida** (xwOBA/barrel/disciplina) que por construcción
filtra la suerte en pelotas en juego. El Kalman la tira 35% hacia el **resultado
real**, que es exactamente lo que ese filtrado quitó.

Segundo mecanismo, independiente: λ_base es **neutra de parque** por diseño, pero
las carreras reales llevan el parque adentro — y el factor de parque se vuelve a
aplicar en el PASO 5 del pipeline.

## Evidencia 1 — datos vivos (predicción del resto de temporada)

236 observaciones (30 equipos × 4 cortes × 2 temporadas), recursión de Kalman
real (Q=0.025, R=9.0), objetivo = carreras/juego del resto de temporada:

    w_kalman = 0.00   r = +0.5303
    w_kalman = 0.15   r = +0.5267
    w_kalman = 0.35   r = +0.4713   <- el actual
    w_kalman = 0.50   r = +0.4306
    w_kalman = 1.00   r = +0.3455

Monótona decreciente en los cinco puntos. Bootstrap agrupado por equipo de la
ventaja de no usarlo: IC95% [-0.0385, +0.1668] — cruza cero, 90.4% de los
remuestreos la favorecen. Sugerente por la forma de la curva, no concluyente por
el intervalo.

## Evidencia 2 — backtest completo (4.825 juegos, tres corridas)

| w_kalman | Brier | vs azar | log-loss | accuracy |
|---|---|---|---|---|
| 0.35 (actual) | 0.24675 | 1.300% | 0.68681 | 55.05% |
| 0.15 | 0.24642 | 1.430% | 0.68612 | 54.86% |
| 0.00 | 0.24624 | 1.510% | 0.68574 | 54.84% |

Brier, log-loss y ventaja-sobre-el-azar mejoran **monótonamente** hacia 0. La
ventaja sobre el azar crece +16.2% en términos relativos.

Accuracy **empeora** monótonamente, -0.21pp, y de forma consistente en las dos
temporadas.

ROI: signos alternados entre umbrales (+0.60, -0.58, +1.66, -2.13, +2.31) — ruido,
no aporta a la decisión.

## Lo que debilita el caso, y hay que decirlo

La mejora de Brier está **concentrada en una temporada**:

    2024   0.24822 -> 0.24727   (-0.00095)
    2025   0.24528 -> 0.24520   (-0.00008)

O sea que 2024 aporta ~92% de la ganancia y 2025 es prácticamente plano. Una
ganancia que depende de una sola temporada es evidencia mucho más floja que una
uniforme — y este proyecto ya tiene el precedente de un "arreglo" revertido dos
veces por regresión (ver `compute_team_bias_kalman_adjusted`).

La pérdida de accuracy, en cambio, sí es consistente en ambas.

## Nota técnica: el acoplamiento degenera limpio

`_KALMAN_BLEND` también gobierna el amortiguamiento del sesgo de equipo:
`dampened = raw_bias / ((1-B) + B·raw_bias)`. Con B=0 el denominador es 1.0 y
devuelve el sesgo crudo intacto — que es lo correcto, porque sin Kalman no hay
solapamiento que remover. Bajar el blend NO toca la parte frágil de esa fórmula.

## La decisión

El criterio correcto para un sistema de EV son las reglas de puntuación propias
(Brier, log-loss), no la accuracy: no se apuesta a quién gana sino a que
`p × cuota > 1`, y eso depende de que la probabilidad esté bien, no de acertar
por encima del 50%. Bajo ese criterio, bajar el blend a 0 es lo mejor de las tres
opciones medidas.

Contra eso: la ganancia se apoya en una temporada, y la accuracy cae en las dos.

**Queda como decisión del dueño.** Las tres corridas están guardadas acá
(`kalman_0.00.json`, `kalman_0.15.json`, y el canónico en
`../val_audit/rebaseline/`) para que no haya que repetirlas.
