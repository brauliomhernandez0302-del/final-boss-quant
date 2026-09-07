# Estado de la evaluación prospectiva — 2026-09-07

Ajuste **congelado sin tocar**: `v1.2 = d88d41ecc7dcd437`,
`v1.4 = 3258fd9e648a0135`.

## 1. Las emisiones verificadas, desglosadas

Lo que el informe anterior llamó «45» eran **pares**, no filas. El desglose
completo:

| | |
|---|---|
| filas `prospectiva_verificada` | **90** |
| de v1.2 (sha `d88d41ec`) | 45 |
| de v1.4 (sha `3258fd9e`) | 45 |
| **partidos únicos** | **15** |
| pares completos (mismo juego y corte, las dos versiones) | **45** |
| **emisiones sin pareja** | **0** |

**Tres cortes por partido**, uno por corrida del generador:

| corte | pares |
|---|---|
| 2026-09-07T02:18:51Z | 15 |
| 2026-09-07T02:20:15Z | 15 |
| 2026-09-07T03:51:57Z | 15 |

### Por qué crecieron de 30 a 45

No aparecieron partidos nuevos: **apareció una tercera emisión de los mismos
15**. El generador corre cada hora a propósito —los abridores se anuncian a lo
largo del día y una corrida perdida no debe dejar hueco—, así que un partido
acumula tantas emisiones como corridas lo alcancen antes de empezar.

**30 → 45 es 15 partidos × 2 cortes → 15 partidos × 3 cortes.** Cero
información nueva sobre el mundo.

## 2. La regla de selección, declarada antes de los resultados

`docs/REGLA_SELECCION_PAREJA_2026-09-07.md`, commit `81523fd`, escrito y
commiteado con **0 de los 15 partidos evaluados** — verificado en el momento de
declararla: `results.db` llegaba hasta 2026-08-06 y ninguno de esos 15 tenía
resultado.

> **Para el informe principal se usa el PRIMER par completo y verificable de
> cada partido.**

Cuatro condiciones a la vez: las dos versiones, el mismo `game_pk` y el mismo
`corte`, las dos con `origen = prospectiva_verificada`, y el SHA de su ajuste
identificado en la fila. Desempate: `corte` más temprano, luego `id` menor.

**El primero y no el último** porque no depende del resultado, porque es el más
exigente —menos anuncios confirmados, menos horas de mercado— y porque «el
último antes del inicio» dependería de cuántas corridas alcanzó el partido, que
es una propiedad del cron y no del modelo.

Efecto inmediato: los 45 pares se convierten en **15 candidatos** al informe
principal, y **30 emisiones posteriores** que se conservan y se reportan aparte.
Fijado con siete tests, incluido uno donde el segundo par acierta más y aun así
se usa el primero.

## 3. Automatización del tramo que faltaba

Faltaban **los dos extremos del ciclo**: nada incorporaba resultados oficiales a
`fbq`, y nada actualizaba el informe. Sin eso, las predicciones se acumulaban
sin poder evaluarse nunca.

| cron | qué hace | costo |
|---|---|---|
| `*/30` | `fbq.anuncios.capturar --dias 3` | gratis |
| `:10` | `fbq.model.predecir --dias 3` | gratis |
| **`:35`** | **`fbq.results.fetch --dias 3`** | **gratis** |
| **`:20 cada 6 h`** | **`fbq.model.aperturas --desde-anuncios`** | **gratis** |
| **`:55`** | **`fbq.model.informe_pareado`** | local |

Las tres nuevas usan sólo la API de MLB. **Ninguna toca la cuota de The Odds
API**, y las 4 entradas de GANICUS siguen intactas.

Dos detalles que costaron una corrección:

- **Dos líneas quedaron sin `cd`** al instalarlas — el mismo error que ya había
  cometido con la captura. Corregido y **verificado simulando el entorno exacto
  del cron** (`env -i`), no leyendo el crontab: las dos devuelven código 0.
- La ventana de fechas se hacía con `$(date -u -d '3 days ago')` dentro del
  cron. Se reemplazó por un `--dias N` en el propio programa: la aritmética de
  fechas se escribe distinto en cada shell y es la clase de cosa que se ve bien
  en el archivo y falla al disparar.

**`fbq.model.aperturas --desde-anuncios`** es nuevo y hacía falta: sin él la
ventana de 40 aperturas se queda vieja, y un abridor anunciado hoy necesita sus
últimas aperturas, no las del mes pasado.

## 4. Un hueco encontrado al automatizar

`results.db` **no tenía nada entre 2026-08-07 y 2026-09-04**: 28 días. Rellenado
—**377 resultados nuevos**, ahora 8.087 filas sin días vacíos— con respaldo
previo.

⚠️ **Consecuencia que no se puede deshacer, y se declara**: las emisiones de los
15 partidos de hoy se calcularon **con ese hueco abierto**, así que sus perfiles
de equipo usaron una ventana incompleta. No es una fuga —todo lo usado era
anterior al corte— pero sí una degradación. Y la regla del primer par, ya
declarada, **congela justamente esas emisiones**.

Regenerar hoy con datos completos produciría cortes más tardíos que la regla
descartaría igual. Se deja como está y se anota: **el primer lote de 15 partidos
se emitió con un almacén de resultados incompleto**. Del lote siguiente en
adelante, completo.

## 5. Informe pareado — todavía sin resultados prospectivos

Son las **13:13 UTC** y los 15 partidos empiezan **17:05 UTC** o más tarde. No
hay ninguno terminado.

| población | pares evaluables | partidos únicos | Brier v1.2 | Brier v1.4 | diferencia | IC95 |
|---|---|---|---|---|---|---|
| **`prospectiva_verificada`** | **0** | 0 | — | — | — | — |
| `no_verificable` (cohorte histórica, reconstruida) | **37** | 37 | 0,242962 | 0,242829 | **−0,000133** | [−0,00766, +0,00740] |
| `reconstruccion` | 0 | 0 | — | — | — | — |

| pendientes | n |
|---|---|
| **partidos con par verificado, esperando resultado** | **15** |
| emisiones posteriores conservadas (no suman) | 30, sobre esos mismos 15 |
| pares sin las dos versiones (ajuste viejo) | 14 |

**No hay nada que concluir sobre v1.4.** El único número con muestra es el de la
cohorte histórica, que es una reconstrucción y cuyo intervalo cruza el cero por
un factor de 55.

Los resultados de esta noche entran solos por el cron de las `:35` y el informe
se regenera a las `:55`.

## 6. Cuándo se decide

n ≥ 900 **partidos** —no emisiones— con par verificado y resultado. A ~15 juegos
por jornada, unos **dos meses** de captura. Todo informe anterior a ese umbral
es descriptivo y no se cita como veredicto.
