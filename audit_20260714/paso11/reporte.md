# Paso 11 — el sesgo de equipo: quitarlo mejora el Brier y DESTRUYE el ROI

**Estado: NO SE CAMBIÓ NADA. `_BIAS_CLAMP` sigue en 0.30.** Y este paso corrige
además el criterio con que argumenté el paso 10.

## Qué hace

`compute_team_bias_kalman_adjusted` multiplica λ por `mean(carreras_reales / λ_predicha)`
del equipo, recortado a ±30%. Es el MISMO mecanismo de "tirar hacia el resultado
observado" que el Kalman del paso 10, expresado como razón en vez de como nivel.

Muy activo: mueve λ en el **94,5%** de los juegos, con ±10% habitual
(p10 0.896, p90 1.099) y 119 juegos pegados al recorte.

## Efecto colateral del paso 10 que hay que tener presente

`_KALMAN_BLEND` gobernaba también el amortiguamiento del sesgo:
`dampened = raw_bias / ((1-B) + B·raw_bias)`. Al pasar B a 0, el sesgo dejó de
amortiguarse y quedó MÁS fuerte:

    sesgo crudo 1.10  →  antes se aplicaba 1.0628, ahora 1.1000
    sesgo crudo 1.20  →  antes 1.1215, ahora 1.2000

Matemáticamente correcto (sin Kalman no hay solapamiento que remover), pero
significa que el paso 10 removió un canal de corrección-hacia-el-resultado y
reforzó el otro. El baseline 0.24624 es el NETO de ambas cosas.

## El experimento: `_BIAS_CLAMP = 0.0` (sesgo neutralizado)

Backtest completo, 4.825 juegos, contra el baseline del paso 10:

| | baseline (p10) | sin sesgo | |
|---|---|---|---|
| Brier | 0.24624 | 0.24501 | −0.00123 |
| ventaja sobre azar | 1.51% | **2.00%** | +32% relativo |
| log-loss | 0.68574 | 0.68303 | −0.00271 |
| accuracy | 54.84% | 54.65% | −0.19pp |

Acumulado desde el rebaseline VAL: la ventaja sobre el azar iría de 1.30% a 2.00%,
**+54% relativo**. Por Brier, el cambio parece excelente.

## Y el ROI dice lo contrario, monótonamente

| umbral | N orig | ROI orig | N p10 | ROI p10 | N sin sesgo | ROI sin sesgo |
|---|---|---|---|---|---|---|
| edge≥0% | 4600 | −2.65% | 4600 | −2.05% | 4600 | **−3.47%** |
| edge≥2% | 3557 | −2.14% | 3471 | −2.72% | 3265 | **−3.26%** |
| edge≥5% | 2166 | −3.29% | 2044 | −1.63% | 1653 | **−6.72%** |
| edge≥8% | 1158 | −3.45% | 1034 | −5.58% | 642 | **−10.59%** |
| edge≥10% | 698 | −9.00% | 600 | −6.69% | 297 | **−19.89%** |

Peor en los cinco umbrales, y **cada vez peor cuanto más alto el edge**. Además el
conteo de apuestas se derrumba: edge≥8% de 1158 a 642, edge≥10% de 698 a 297.

## La lectura, y por qué importa más allá de este paso

El Brier mide la calidad **promedio** de las probabilidades sobre los 4.825 juegos.
El ROI mide qué pasa en los juegos donde el modelo cree tener ventaja — la cola.
Quitar el sesgo volvió al modelo más "promedio": mejor calibrado en el centro, y sin
capacidad de identificar dónde apartarse del mercado. Para un sistema que sólo apuesta
cuando cree tener edge, **eso es exactamente el intercambio equivocado**.

La calibración por bucket lo confirma: mejora mucho en `<40%` (+10.80 → +3.40) pero
empeora en `45-50%` (+1.60 → +4.10) y `60-70%` (−0.80 → +3.40).

## Corrección al criterio que usé en el paso 10

En el paso 10 escribí que "el criterio correcto para un sistema de EV son las reglas
de puntuación propias (Brier, log-loss), no la accuracy". Eso era **incompleto**:
ignoré el ROI, que es el proxy más directo de para qué existe el sistema. Acá el
Brier y el ROI se separan de forma brutal y en direcciones opuestas.

El paso 10 NO queda invalidado — su ROI alternaba de signo (+0.60, −0.58, +1.66,
−2.13, +2.31), o sea ruido, no daño demostrable — pero el argumento con que lo
justifiqué sí. **De acá en adelante el ROI por umbral de edge es un gate obligatorio
para cualquier cambio de λ, no una métrica secundaria.**

## Conclusión

El sesgo de equipo se queda. Y si algún día se lo quiere tocar, la evidencia de que
mejora el Brier NO alcanza: hay que mostrar que no rompe el ROI en los umbrales altos,
que es donde este experimento lo destruyó.
