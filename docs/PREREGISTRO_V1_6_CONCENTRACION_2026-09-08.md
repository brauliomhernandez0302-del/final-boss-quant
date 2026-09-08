# Preregistro — v1.6 experimental: concentración de la carga entre relevistas

**Escrito y commiteado el 2026-09-08, ANTES de calcular una sola métrica de
rendimiento.** Lo medido hasta acá es sólo **cobertura** (§3), que no toca
resultados, precios ni Brier.

v1.2 sigue siendo la referencia. **v1.4 `3258fd9e648a0135` sigue intacta** en su
evaluación prospectiva. v1.5 quedó **cerrada** (`docs/CIERRE_V1_5_2026-09-08.md`).

---

## 1. Qué cambia respecto de v1.5, y por qué UNA sola cosa

v1.5 medía **cuánto** lanzó el bullpen: un total por equipo. Ese total trata
igual a un bullpen que gastó 90 lanzamientos entre seis brazos y a uno que los
gastó en dos. v1.6 mide **entre cuántos brazos se repartió**, que es lo único
que el dato nuevo por relevista permite preguntar y el agregado no.

**Un solo candidato. No se prueban variantes.** Ni top-1, ni top-3, ni «número
efectivo de brazos», ni otras ventanas. Probar varias y quedarse con la que gana
es fabricar un hallazgo, y este proyecto ya pagó esa factura.

## 2. La variable — definición exacta

    dif_concentracion(juego, corte) = HHI_local(corte) − HHI_visita(corte)

    HHI_E(corte) = Σ_i ( p_i / P )²

donde `p_i` son los lanzamientos del relevista *i* del equipo E en los partidos
`p` con

    corte − 72 h  ≤  disponible_desde(p)  ≤  corte

y `P = Σ_i p_i`, sobre los relevistas con `p_i > 0`.

| elemento | definición | de dónde sale |
|---|---|---|
| relevista | rol `relevo` en `data/relevistas.db`: todo lanzador **después del primero que lanzó** | regla de rol de `bullpen_relief_appearance_builder`, con el arreglo de las entradas de cero lanzamientos |
| `p_i` | `numberOfPitches` del boxscore oficial | respuesta guardada en `data/boxscores/` |
| `disponible_desde` | fin medido de la última jugada + `MARGEN_FIN` (20 min) | `fbq/results/fines.py`, ya preregistrado |
| ventana 72 h | **la misma de v1.5**, sin volver a elegirla | `fbq/model/carga.py::VENTANA_HORAS` |
| `corte` | `captured_at` del último par de Pinnacle pre-juego | el mismo que usan v1.2 y v1.4 |

HHI vale **1** cuando un solo brazo cargó con todo y **1/k** cuando k brazos
cargaron por igual. Es una proporción: **no depende de cuánto se lanzó**, sólo
de cómo se repartió. Por eso no es una versión reescalada de v1.5 — su
correlación con aquélla se reporta como diagnóstico, después.

### Por qué HHI y no otro estadístico

Usa **todos** los brazos (no hay un N arbitrario que elegir), es adimensional, y
es el índice estándar de concentración. Elegirlo antes de mirar es lo que impide
que la elección la haga el resultado.

### Signo esperado, declarado antes de medir

Carga concentrada en pocos brazos → menos brazos frescos disponibles → **peor**
para ese equipo. Se espera **coeficiente negativo**. Un coeficiente positivo
significativo sería evidencia CONTRA esa hipótesis, no una historia nueva.

## 3. Cobertura — medida antes de definir el resto

### El almacén por relevista

| | |
|---|---|
| apariciones | **73.293** en 8.095 partidos |
| brazos distintos | **2.513** |
| brazos que **nunca** abrieron | **1.766 (70,3 %)** |
| brazos que el almacén de los 517 no tenía | **1.994** |
| apariciones de relevo | 57.100 — de ellas **40.397 (70,7 %) de brazos fuera de los 517** |
| `procedencia` | `medido` 73.222 · `desconocido` 71 |
| conciliación contra los agregados por equipo | **16.190 de 16.190 · 0 discrepancias** en lanzamientos de relevo, lanzamientos totales y número de relevistas |

### La variable sobre las filas evaluables

| | |
|---|---|
| filas predecibles | 5.741 |
| **con concentración computable** | **5.690 (99,11 %)** |
| no computables | 51, todas `sin_relevo_en_la_ventana` |
| por temporada | 2024: 1.971/1.988 · 2025: 2.175/2.191 · 2026: 1.544/1.562 |
| HHI local | media 0,1931 · mediana 0,1787 · rango [0,0865, 1,0000] |
| brazos en la ventana (local) | media 6,81 · rango [1, 13] |
| `dif_concentracion` | media −0,0033 · desvío 0,0831 · rango [−0,823, +0,841] |

## 4. Faltantes — fijado antes de medir

| caso | tratamiento |
|---|---|
| el equipo no tiene **ningún** lanzamiento de relevo en la ventana | **fila NO computable**: el HHI es 0/0 y no se inventa |
| un partido de la ventana sin fila de relevista | no ocurre: la conciliación cierra en 16.190/16.190 |
| `disponible_desde` nulo (`procedencia = desconocido`) | la aparición no entra a ninguna ventana, igual que en todo el proyecto |

**Nunca se imputa.** Una fila no computable se excluye de v1.6 **y de v1.2 y del
mercado**, para que las tres columnas se midan sobre exactamente las mismas
filas y los mismos cortes.

## 5. Estimación y comparación

Idénticas a v1.5, sin cambiar nada:

- pliegues expansivos (`evaluar_expansivo`): entrenar con temporadas
  **estrictamente anteriores**, evaluar 2025 y 2026 por separado;
- **el peso sale sólo del entrenamiento**; ridge λ=1,0 congelada, intercepto sin
  penalizar, estandarización con media y desvío del pliegue de entrenamiento;
- **v1.2 se re-ajusta sobre la misma intersección** — comparar contra el v1.2
  publicado, ajustado sobre un conjunto mayor, compararía dos muestras;
- nulo: Pinnacle desvigorizado, mismo devig, recomputado sobre la intersección;
- incertidumbre: diferencia **pareada** de Brier por juego, bootstrap **agrupado
  por equipo local** (29-30 clústeres, 2.000 remuestreos).

## 6. Qué es este dato, y qué NO

Es **carga observada por relevista**. No es disponibilidad.

Que un brazo haya lanzado 40 lanzamientos en dos días **no demuestra** que hoy
no esté disponible; que no aparezca **no demuestra** que esté lesionado ni
descansando. Un relevista puede faltar por rol —un cerrador que no entra porque
el juego no está cerrado—, por marcador, por decisión del entrenador, por una
bajada a ligas menores, o porque el partido no lo pidió. **Nada de eso está en
este dato y nada de eso se infiere de él.** Una concentración alta dice que la
carga cayó en pocos brazos; no dice por qué, ni qué pasará mañana.

## 7. Controles de fuga

Los mismos, sin versión relajada: compuerta estructural de `VentanaPIT`
heredada por leer la ventana por `disponible_desde`; pruebas de ventana
versionadas (un partido terminado después del corte no entra; uno de 73 h
tampoco; cambiar la carga de uno de dentro sí mueve la variable); control
positivo de fuga en CI; detector estadístico `BRIER_IMPLAUSIBLE = 0,22`.

## 8. Criterio de cierre, declarado ahora

v1.6 **no reemplaza a v1.2** salvo que la diferencia pareada de Brier sea
negativa y su **IC95 excluya el cero en los DOS pliegues**. Cualquier otro
resultado la cierra como no demostrada, igual que v1.3 y v1.5. Se publica el
resultado sea cual sea.

**La evaluación de 2025-2026 es EXPLORATORIA**: esos años ya fueron explorados
por este proyecto. El diseño temporal impide que el modelo vea el futuro; no
impide que lo haya visto quien eligió la variable.
