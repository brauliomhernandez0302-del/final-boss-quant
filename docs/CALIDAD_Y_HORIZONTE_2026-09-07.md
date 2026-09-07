# Calidad de las entradas, y cuánto falta de verdad — 2026-09-07

Tres cosas, y ninguna cambia una sola predicción ya emitida:

1. **Se anota** —de forma persistente y auditable— qué emisiones se calcularon
   con el historial incompleto, sin tocar su condición prospectiva.
2. **Se corrige la estimación de tiempo** para reunir 900 pares, usando el
   calendario real y la cobertura elegible **medida**, no «~15 juegos/día».
3. **Se escribe qué garantiza y qué NO garantiza el umbral n ≥ 900.**

---

## 1. Dos preguntas distintas que se confundían en una etiqueta

| eje | pregunta | dónde vive |
|---|---|---|
| `origen` | ¿la predicción se **escribió** antes del primer lanzamiento? | `prediccion.origen`, derivado de `registrado_utc` |
| `calidad_datos` | ¿con qué **entradas** se calculó? | tabla `calidad_datos`, anotada aparte |

Una emisión puede ser impecablemente prospectiva **y** haber usado entradas
incompletas. Las dos cosas son ciertas a la vez y **ninguna anula a la otra**:

- degradarla a `no_verificable` por la calidad **escondería** que la predicción
  sí se escribió antes del partido — que es justo lo que cuesta demostrar;
- ignorar la calidad haría pasar por buena una entrada que no lo era.

Por eso el informe las cruza (`origen|calidad`) y **nunca las funde ni las
suma**.

### El caso concreto

`results.db` no tenía **nada entre 2026-08-07 y 2026-09-04** — 28 jornadas. El
hueco se descubrió el 2026-09-07 a las 13:13 UTC, al automatizar la
incorporación de resultados, y se rellenó (377 resultados). Pero las emisiones
de los **primeros 15 partidos prospectivos** ya se habían calculado con ese
hueco abierto: sus perfiles de equipo usaron una ventana de 162 partidos a la
que le faltaba casi un mes.

**No es una fuga** —todo lo que usaron era anterior a su corte— pero sí una
degradación real de la entrada.

### Qué se hizo, y qué NO

- ✅ Se **anota** cada emisión con la huella del almacén (`historial_hasta`,
  `historial_filas`, `dias_sin_datos`, `calidad`).
- ✅ Los 15 partidos **se conservan** en el registro y se reportan
  **identificados**, en su propia población.
- ❌ **No se re-generan** con el historial reparado. Sustituir hoy la predicción
  de ayer y presentarla como prospectiva sería fabricar un registro que no
  ocurrió — el error exacto que `registrado_utc` existe para impedir.
- ❌ Tampoco se descartan. Se emitieron antes del partido; su valor prospectivo
  es real y su limitación queda escrita al lado.

### El mecanismo, no la disciplina

| regla | quién la impone |
|---|---|
| una anotación no se pisa | trigger `trg_calidad_no_update` (ABORT en `UPDATE`) |
| una anotación equivocada **se corrige agregando otra** | `calidad_datos` **sin llave única**, a propósito; vale la de `MAX(id)` |
| lo que nadie miró no cuenta como bueno | `resumen()['sin_anotar']`, y el informe reporta `sin_anotar` como calidad propia |
| el par hereda la **peor** calidad de sus dos piernas | `informe_pareado`, igual que ya hacía con `origen` |

Esa segunda regla nació de un error propio de esta misma sesión: anoté las 229
emisiones como `historial_incompleto`, y **111 no lo eran** — los cortes de la
cohorte histórica (2 al 5 de agosto) **preceden** al hueco, así que su ventana
nunca lo tocó. La llave única de la primera versión de la tabla **impedía
corregirlo sin borrar**, contradiciendo el docstring del propio módulo. Se
migró la restricción (filas conservadas y contadas antes y después) y la
corrección entró como 111 filas nuevas, con la anotación equivocada a la vista.

### Estado vigente (229 emisiones, 53 juegos)

| `origen` | `calidad_datos` | filas | juegos | qué es |
|---|---|---|---|---|
| `no_verificable` | `completo` | 74 | 37 | cohorte histórica re-generada; su evidencia de emisión original se perdió |
| `no_verificable` | `historial_incompleto` | 28 | 14 | emisiones sin evidencia original **y** con el hueco abierto |
| `prospectiva_verificada` | `historial_incompleto` | 90 | 15 | **escritas antes del partido**, con el hueco abierto |
| `reconstruccion` | `completo` | 37 | 37 | calculadas después sobre un corte del pasado |

---

## 2. La estimación de tiempo, corregida

La cifra anterior —«~15 juegos/día»— era el **calendario**, no la cobertura. Un
partido programado sólo aporta un par si, además de jugarse, hay anuncio de
abridor, estadísticas del abridor y precio antes del corte.

### Lo medido, no lo supuesto

| dato | valor | fuente |
|---|---|---|
| cobertura elegible | **15 de 26 = 57,7 %** | 6/11 el 09-07 y 9/15 el 09-08, contra el schedule oficial |
| ritmo del calendario | **13,36 juegos/día** | 2.458 juegos de temporada regular en 184 días (2026) |
| ritmo efectivo | **7,71 pares/día** | 13,36 × 0,577 |
| fin de la regular 2026 | **2026-09-27** | schedule oficial |
| juegos que faltan en 2026 | **276** en 21 días | 11 hoy + 265 del 09-08 al 09-27 |

### La proyección

| tramo | pares | acumulado |
|---|---|---|
| resto de la regular 2026 | ~159 (de ellos ~15 con historial incompleto) | ~159 |
| postemporada 2026 (09-29 → 10-31) | ~23 sobre ≤53 programados | **no se suma** (ver abajo) |
| temporada 2027, desde ~03-25 | 7,71/día | 900 alrededor de **2027-06-29** |

Contando sólo pares con historial completo: **~2027-07-01**.

**La postemporada no se suma**: son 8-12 equipos seleccionados por ser los
mejores del año, con rotaciones comprimidas y descansos que la regular no tiene.
Es otra población; agruparla para llegar antes al número sería comprar muestra
con sesgo. Se captura y se reporta **aparte**.

**Supuesto declarado**: el calendario 2027 no está publicado. Se usa el mismo
día de arranque que 2026 (25 de marzo). Un arranque una semana después corre la
fecha una semana.

### Sensibilidad — la cobertura es la variable que manda

| cobertura elegible | pares/día | 2026 aporta | fecha de los 900 |
|---|---|---|---|
| 35 % | 4,68 | 97 | 2027-09-13 |
| 45 % | 6,01 | 124 | 2027-08-01 |
| **57,7 % (medida hoy)** | **7,71** | **159** | **2027-06-29** |
| 70 % | 9,35 | 193 | 2027-06-09 |
| 85 % | 11,35 | 235 | 2027-05-23 |

Subir la cobertura del 58 % al 85 % adelantaría el umbral **más de un mes**. Es
la única palanca disponible que no consume cuota paga, y la medición de por qué
se pierden 11 de 26 partidos es trabajo pendiente concreto.

---

## 3. El umbral n ≥ 900: objetivo, no garantía de conclusión

**Qué es**: n = ((1,96 + 0,84)·σ/Δ)² × deff con σ = 0,010024 (diferencia pareada
de Brier por juego, proxy v1.3−v1.2 sobre 3.748 juegos), Δ = 0,001 y efecto de
diseño 1,120 (bootstrap agrupado por equipo local) → **884**, redondeado a 900 y
pre-registrado en `PROSPECTIVA_ACTIVADA_2026-09-06.md` §6.

**Qué significa exactamente**: *si* el efecto real fuera de 0,001 de Brier,
*entonces* con 900 pares se detectaría el 80 % de las veces.

**Qué NO garantiza — y esto no se suaviza al reportarlo:**

| | |
|---|---|
| **no garantiza una conclusión** | alcanzar 900 no obliga al intervalo a excluir el cero. El desenlace más probable de un efecto nulo o minúsculo es un IC que sigue cruzando el cero — y ese es un resultado, no un fracaso de la medición |
| **el 20 % restante existe** | con potencia 80 %, uno de cada cinco efectos reales de ese tamaño **no** se detecta |
| **Δ = 0,001 es una elección** | es el orden que un motor de abridores debería producir si vale algo, no un umbral derivado de rentabilidad. Un efecto real de 0,0005 pide ~3.534 pares y a los 900 sería invisible |
| **σ es prestado** | sale del contraste v1.3−v1.2. El de v1.4−v1.2 tendrá el suyo, y se re-estimará con datos reales: si σ sube, 900 se queda corto |
| **deff puede cambiar** | 1,120 se midió agrupando por equipo local sobre la cohorte histórica; la estructura de agrupamiento de la muestra prospectiva no tiene por qué ser idéntica |
| **no autoriza mirar antes** | ningún informe previo a 900 es veredicto. La única salida anticipada pre-registrada es IC95 que excluya el cero **con n ≥ 221**, o sea un efecto de 0,002 o más |

---

## 4. El primer informe con resultados reales

**Todavía no hay ninguno**, y decirlo así es parte del informe.

Estado a las 13:47 UTC del 2026-09-07:

```
poblaciones:
  no_verificable|completo ............ 37 pares · 37 partidos únicos
      Brier v1.2 0,242962 · v1.4 0,242829
      diferencia pareada (v1.4 − v1.2) −0,000133
      IC95 [−0,007662, +0,007395]   ← cruza el cero con holgura
      37/900 del umbral de decisión
pendientes:
  prospectiva_verificada|historial_incompleto ... 15, sin resultado todavía
  sin_las_dos_versiones ......................... 14
emisiones_posteriores:
  prospectiva_verificada|historial_incompleto ... 30 filas, 15 partidos,
      conservadas y NO sumadas al informe principal
```

- Los **15 partidos prospectivos empiezan a las 17:05 UTC o después**. A las
  13:47 UTC ninguno se ha jugado: `results.db` tiene 0 de esos 15.
- Los 37 pares que sí tienen resultado son la cohorte **histórica**
  (`no_verificable`): sirven para desarrollar, **no acreditan nada** — su
  evidencia de emisión original se destruyó y así queda etiquetada.
- La diferencia observada, −0,000133, es **1/29 del ancho de su propio
  intervalo**. No dice que v1.4 sea mejor ni peor: dice que con 37 pares no se
  puede saber.

La cadena automática que lo cierra ya está verificada corriendo:

| cron | qué hace | evidencia |
|---|---|---|
| `:35` de cada hora | incorpora resultados oficiales (MLB Stats API, gratis) | disparó a las 13:35 UTC de hoy |
| `:55` de cada hora | regenera el informe pareado a `docs/informe_pareado_vigente.json` | sólo lee las bases |

**Cerrojo agregado hoy a la incorporación de resultados.** `results.db`
deduplica **comparando contra la última observación** de cada juego: dos
corridas simultáneas leen las dos «todavía no está» y las dos insertan. Es el
mismo modo de falla por el que la captura de anuncios ya llevaba `flock`, así
que el cerrojo se mudó a `fbq/core/cerrojo.py` y ahora lo usan las dos. El
informe no lo necesita: sólo lee.

**No hace falta intervenir**: los 15 partidos entran al informe solos, en su
población `prospectiva_verificada|historial_incompleto`, en cuanto terminen.

---

## 5. Reproducir

```bash
python3 -m fbq.model.calidad                  # huella del historial + anota pendientes
python3 -m fbq.model.informe_pareado          # informe cruzado origen × calidad
python3 -m pytest tests/test_fbq_prospectiva.py tests/test_fbq_informe_pareado.py
```
