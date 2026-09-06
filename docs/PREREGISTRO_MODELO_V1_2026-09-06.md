# Preregistro — candidato v1 de P(gana el local)

**Fecha: 2026-09-06.** Este documento se commitea **antes de medir nada**. Ese
es su único valor: un preregistro escrito después de ver el resultado no es un
preregistro, es una racionalización. Si algo de acá cambia después de la primera
medición, se anota el cambio con su fecha y su motivo, no se edita en silencio.

## 0. Qué es y qué NO es

Es el **baseline sencillo** contra el cual se evaluarán después los motores
recuperados del sistema anterior. No pretende ganarle al mercado: pretende ser
el número honesto más simple que se pueda defender, construido sólo con hechos
que estaban disponibles al momento de cada predicción.

**Expectativa declarada de antemano, para que no se pueda mover después**: se
espera que **pierda contra Pinnacle**. El mercado mide 0,242124 sobre el marco
completo; nueve motores del sistema anterior, con Statcast y PIT, quedaron en
0,24620. Un logístico de dos variables sobre marcadores no tiene por qué
acercarse. Lo que se pide de él es que **le gane a la tasa base** y que la
medición sea reproducible.

## 1. Población de partidos

| criterio | valor |
|---|---|
| deporte | MLB |
| temporadas | 2024, 2025, 2026 |
| resultado | final legítimo en `results.db` (vista `resultado`) |
| precio de referencia | par de Pinnacle **pre-juego** en `market.db`, el último anterior al primer lanzamiento |
| exclusión | juego sin par de Pinnacle pre-juego: sin nulo no hay con qué comparar |
| exclusión | juego donde cualquiera de los dos equipos tiene menos de `MIN_JUEGOS_PREVIOS` partidos disponibles al corte |

Las exclusiones se **cuentan y se publican**, nunca se rellenan con un valor por
defecto. Un juego que no se puede predecir con lo disponible se excluye;
imputarle la media lo convertiría en una predicción que no se hizo.

## 2. El corte temporal de cada fila

**El corte de una fila es el `captured_at` de su precio de referencia.** No el
día del juego, no la medianoche anterior: el instante exacto en que se observó
el precio contra el que se va a comparar. Cualquier hecho que entre en esa fila
tiene que haber estado disponible en ese instante.

Un partido anterior está **disponible** en el corte `T` si:

```
inicio_utc(partido_anterior) + DURACION_MAXIMA  ≤  T
```

`DURACION_MAXIMA = 8 horas`, deliberadamente conservador: la mediana de un
partido de MLB ronda las 3 horas y los extremos de entradas extra no llegan a 7.
No conocemos la hora de FIN de los partidos —ninguna fuente propia la guarda— y
la alternativa a una cota es adivinar. Errar por exceso cuesta **cobertura**;
errar por defecto cuesta una **fuga**, que es lo que este proyecto ya pagó siete
veces.

**Partidos suspendidos y reanudados**: el schedule devuelve dos entradas con el
mismo `game_pk`. La disponibilidad se calcula sobre la **más tardía** — un
partido suspendido el día D y terminado el D+1 no estaba disponible el D.

**Etiquetas de entrenamiento**: la etiqueta de un juego de entrenamiento es su
propio resultado, y el diseño expansivo (§5) garantiza que todo juego de
entrenamiento pertenece a una temporada **estrictamente anterior** a la de
evaluación. Se verifica con una aserción, no se asume.

## 3. Variables

Sólo hechos deportivos que viven en los almacenes propios: marcadores y fechas.

| # | variable | definición |
|---|---|---|
| 1 | `dif_pitagorica` | esperanza pitagórica del local menos la del visitante |
| 2 | `dif_descanso` | días de descanso del local menos los del visitante, recortado a ±`TOPE_DESCANSO` |
| — | intercepto | absorbe la ventaja de local; **no** se modela aparte |

**Esperanza pitagórica**: `CF^e / (CF^e + CC^e)` con `e = 1.83`, sobre las
carreras a favor y en contra por juego de la ventana, **regresadas** hacia el
promedio de la liga (§4).

### Lo que se deja fuera a propósito

- **Forma reciente / rachas.** El paso 8 de la auditoría del pipeline la
  NEUTRALIZÓ tras medirla: sus señales salían de la misma lista de 5 arranques y
  captaban regresión a la media leída como persistencia; ninguna sobrevivía a
  control por nivel con errores agrupados. Volver a medirla sería pagar dos
  veces por la misma respuesta.
- **BaseRuns**, que el propio curso del proyecto señala como mejor estimador de
  talento que la pitagórica. Necesita hits, bases por bolas y bases totales, y
  ninguna vive en los almacenes propios. Queda anotado como la mejora obvia.
- **Abridores, bullpen, parque, clima, defensa.** Son el sistema anterior, y su
  dato no está en los almacenes propios. Este baseline existe justamente para
  darles una barra contra la cual medirse.

## 4. Ventana histórica y equipos con poco historial

| parámetro | valor | por qué |
|---|---|---|
| `VENTANA` | **162** partidos terminados por equipo | una temporada; cruza el borde de temporada a propósito, para que abril tenga historia en vez de un fallback |
| `MIN_JUEGOS_PREVIOS` | **30** | debajo de eso la pitagórica es ruido; el juego se **excluye**, no se imputa |
| `K_REGRESION` | **67** partidos de liga media | la constante que el propio curso del proyecto registra para regresar un récord de MLB a talento verdadero |
| `TOPE_DESCANSO` | **5** días | por encima es un parón (all-star, lesión larga) y deja de medir descanso |

**Regresión a la media**: se usa `regress(observado, media, n, k)` —encogimiento
bayesiano— **reutilizada del sistema anterior** (`modules/baseball_module/
offense/tte_formula.py`). Ver §8.

⚠️ **Decisión conservadora declarada**: `K_REGRESION=67` es la constante para
regresar **victorias**, y el curso dice explícitamente que *el diferencial de
carreras regresiona menos que las victorias*. Aplicarla a las carreras
**sobre-encoge** y debilita la feature. Se acepta a propósito: medir el `k`
correcto para carreras exigiría ajustarlo sobre los mismos años que se van a
evaluar. Queda como pendiente nombrado.

## 5. Entrenamiento y evaluación

**Expansivo, por temporadas enteras.** Nunca un corte aleatorio: pondría juegos
del mismo día a ambos lados y el ajuste aprendería del futuro por la puerta de
al lado.

| evaluación | entrenamiento |
|---|---|
| **2025** | 2024 |
| **2026** | 2024 + 2025 |

- **Estandarización** (media y desvío de cada variable): calculada **sólo sobre
  el pliegue de entrenamiento** y aplicada al de evaluación.
- **Regularización**: L2 (ridge) con `LAMBDA_L2 = 1.0` sobre las variables
  estandarizadas, **sin penalizar el intercepto**. Fijada a priori. **No se
  ajusta por validación cruzada**: cualquier búsqueda de hiperparámetro sobre
  2025 o 2026 sería tocar los años de evaluación.

### ⚠️ Esta evaluación es HISTÓRICA, no fuera de muestra

**2025 y 2026 ya fueron explorados exhaustivamente por este proyecto** —siete
baselines, una auditoría de 16 informes, nueve motores medidos sobre esos mismos
años. El diseño temporal impide que el MODELO vea el futuro, pero no impide que
lo haya visto **quien eligió las variables**. Ningún número de esta evaluación
puede citarse como validación fuera de muestra. La única prueba limpia posible
es 2027, hacia adelante.

## 6. Control positivo de fuga

Dos detectores, y el segundo existe porque el primero se puede esquivar.

1. **Estructural** (`fbq/model/pit.py`): todo hecho que entra en una fila pasa
   por una compuerta que verifica la condición de §2. Un hecho no disponible
   levanta `FugaDetectada`. La regla la impone el mecanismo, no la disciplina.
2. **Estadístico** (`fbq/model/detector.py`): un candidato cuyo Brier baje de
   `BRIER_IMPLAUSIBLE = 0.22` se **rechaza**. Justificación del umbral: Pinnacle
   —el libro más afilado del mercado más eficiente— mide 0,2421. Un 0,22 sería
   una mejora relativa del 9% sobre Pinnacle, algo que ninguna literatura
   reporta. En este problema, *demasiado bueno* es sinónimo de *fuga*.

**Prueba obligatoria**: se construye a propósito una variable permitida
(`dif_pitagorica`) que incluye el marcador **del propio partido**, y se verifica
que los detectores la rechacen — el estructural si pasa por la compuerta, el
estadístico si la esquiva. Un instrumento que no detecta una fuga plantada
tampoco detectaría una real.

## 7. Comparación

Sobre **exactamente los mismos juegos evaluables**, tres candidatos:

| candidato | qué es |
|---|---|
| **v1** | el logístico de este documento |
| **tasa base** | frecuencia de victoria local **calculada sobre el pliegue de entrenamiento**, constante |
| **Pinnacle** | el par desvigorizado, el nulo |

⚠️ **El Brier del mercado se recalcula sobre la intersección.** El 0,242124067
de `docs/REFERENCIA_MERCADO_2026-09.md` corresponde al marco COMPLETO de 6.106
juegos; el conjunto evaluable de este candidato es menor (§1) y comparar contra
el número del marco completo sería comparar dos muestras distintas — el error
que tiró siete baselines de este proyecto.

Métricas: Brier, log-loss, y el veredicto del evaluador ya existente
(`fbq.evaluator.evaluate`), que incluye la regresión conjunta
`y ~ logit(mercado) + logit(candidato)` con bootstrap **agrupado por equipo**.

## 8. Qué se reutiliza del sistema anterior, y por qué es válido

| pieza | origen | validez temporal |
|---|---|---|
| `regress(observado, media, n, k)` | `modules/baseball_module/offense/tte_formula.py` | **Trivial: es matemática pura.** Recibe números, no accede a ningún dato, no conoce fechas. Nada que pueda filtrarse. |
| El día del juego sale de `official_date`, no de `date(UTC)` | paso 4 de la auditoría; ya encapsulado en `fbq/core/clock.py` | Verificado: 22-24% de los juegos difieren |
| `ESTADOS_FINALES` para distinguir el cascarón pospuesto del partido jugado | `results/store.py`, ya movido a `core/identity.py` | Regla de identidad, sin componente temporal |
| El evaluador, su bootstrap agrupado y el portón | `fbq/evaluator/`, `fbq/features/gate.py` | Ya migrados a almacenes propios |
| **Resultados negativos ya pagados**: forma reciente anti-predictiva (paso 8); las nueve señales correlacionan 0,28-0,65 con el mercado y 0,05-0,10 con el resultado | auditoría paso-a-paso | Se reutilizan como **decisiones de diseño**: evitan volver a medir lo ya medido |

`regress` se **copia** con atribución en vez de importarse: `fbq/` es autónomo
del sistema anterior por diseño, y ese árbol está declarado inválido y puede
borrarse en cualquier momento. Como el propio proyecto ya pagó una duplicación
que derivó (la constante de barrel% arreglada en una copia y no en la otra), un
test compara la copia contra el original mientras el original exista.

**El código original no se toca.** Nada de `modules/`, `core/` o
`backtest_and_retrain.py` se modifica en este trabajo.

## 9. Constantes, en un solo sitio

```python
VENTANA              = 162     # partidos terminados por equipo
MIN_JUEGOS_PREVIOS   = 30      # debajo de esto se EXCLUYE el juego
K_REGRESION          = 67      # partidos de liga media (curso del proyecto)
EXPONENTE_PITAGORICO = 1.83    # sabermetría estándar
TOPE_DESCANSO        = 5       # días
DURACION_MAXIMA      = 8h      # cota conservadora de duración de un partido
LAMBDA_L2            = 1.0     # ridge, sin penalizar el intercepto
BRIER_IMPLAUSIBLE    = 0.22    # por debajo de esto se rechaza por fuga
```

## 10. Entregables

Un CSV por juego con `game_pk`, fecha, equipos, corte, las variables, `p_v1`,
`p_tasa_base`, `p_mercado`, el resultado y las tres pérdidas de Brier; más
cobertura, exclusiones por motivo, y las versiones de código y datos.
