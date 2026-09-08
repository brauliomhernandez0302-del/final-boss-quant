# Primer balance prospectivo de FBQ — qué pronosticamos y qué ocurrió

**Corrida del 2026-09-08 23:35 UTC.** Descriptivo. **No se declara ningún modelo
ganador**, no se ajusta ningún parámetro y no se cambia ninguna regla por estos
resultados.

- **v1.2** sigue siendo la referencia.
- **v1.4 `3258fd9e648a0135`** intacto: es el ajuste que emitió estas
  predicciones y no se tocó.
- **v1.6 cerrada** como mejora no demostrada (`docs/V1_6_CONCENTRACION_RESULTADO_2026-09-08.md`).
- Todas las emisiones originales se conservan; ninguna se regeneró ni se
  sustituyó.

## Regla aplicada, congelada antes de conocer resultados

`docs/REGLA_SELECCION_PAREJA_2026-09-07.md`: **el primer par completo y
verificable de cada partido**. Un partido con veinte emisiones aporta **un** par.
Las emisiones posteriores se conservan y se reportan aparte, nunca sumadas.

## La muestra creció entre la pregunta y esta corrida: de 6 a 9

Cuando se pidió este balance había **6** partidos terminados (2 con historial
completo, 4 con incompleto). El ciclo automático siguió corriendo y a esta hora
son **9** (3 y 6). Se reconstruyó con exactitud cuáles eran los 6 usando
`observacion.observado_en` —el instante en que NOSOTROS vimos cada resultado—,
así que van los dos cortes, separados. Recortar a 6 hoy exigiría un criterio
inventado después de ver los resultados.

| llegó | game_pk | visto por primera vez |
|---|---|---|
| en los 6 originales | 824229, 824715, 823415, 823742, 824793 | 2026-09-07 23:35 UTC |
| en los 6 originales | 823254 | 2026-09-08 00:35 UTC |
| después de la pregunta | 823902, 823175 | 2026-09-08 04:35 UTC |
| después de la pregunta | 824958 | 2026-09-08 05:35 UTC |

## Tabla completa — los 9 pares, sin recortes

| fecha | game_pk | visita @ local | historial | p(local) v1.2 | p(local) v1.4 | marcador | ganó | Brier v1.2 | Brier v1.4 | Δ v1.4−v1.2 | abridor local | abridor visita |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-07 | 823902 | Cincinnati Reds @ Los Angeles Dodgers | completo | 0,6386 | 0,6286 | 6-3 | local | 0,1306 | 0,1379 | **+0,0073** | coincidió | coincidió |
| 2026-09-07 | 824229 | Minnesota Twins @ Detroit Tigers | completo | 0,5932 | 0,5650 | 5-4 | local | 0,1655 | 0,1893 | **+0,0238** | coincidió | coincidió |
| 2026-09-07 | 824715 | Los Angeles Angels @ Boston Red Sox | completo | 0,6019 | 0,5684 | 5-2 | local | 0,1585 | 0,1862 | **+0,0278** | coincidió | coincidió |
| 2026-09-07 | 823175 | St. Louis Cardinals @ San Francisco Giants | incompleto | 0,5406 | 0,5684 | 5-4 | local | 0,2110 | 0,1863 | **−0,0248** | coincidió | coincidió |
| 2026-09-07 | 823254 | Washington Nationals @ San Diego Padres | incompleto | 0,5632 | 0,6027 | 3-2 | local | 0,1908 | 0,1578 | **−0,0330** | coincidió | coincidió |
| 2026-09-07 | 823415 | Atlanta Braves @ Philadelphia Phillies | incompleto | 0,4453 | 0,5131 | 1-0 | local | 0,3077 | 0,2371 | **−0,0706** | coincidió | coincidió |
| 2026-09-07 | 823742 | Chicago Cubs @ Milwaukee Brewers | incompleto | 0,5624 | 0,5670 | 4-3 | local | 0,1915 | 0,1875 | **−0,0040** | coincidió | coincidió |
| 2026-09-07 | 824793 | Cleveland Guardians @ Baltimore Orioles | incompleto | 0,4993 | 0,5248 | 6-4 | local | 0,2507 | 0,2259 | **−0,0249** | coincidió | coincidió |
| 2026-09-07 | 824958 | Toronto Blue Jays @ Athletics | incompleto | 0,4867 | 0,4732 | 6-5 | local | 0,2635 | 0,2775 | **+0,0141** | coincidió | coincidió |

Negativo = v1.4 perdió menos que v1.2 en ese partido.

**Abridores: 18 de 18 coincidieron.** El abridor anunciado vigente antes del
corte —lo último que se sabía en ese instante, vía `vigente_antes`— fue el que
efectivamente abrió, en los dos lados de los nueve partidos. Cero cambios, cero
partidos sin anuncio al corte. El abridor real sale de `data/relevistas.db`
(regla de rol: el primero de la lista que lanzó), no del probable del calendario.

## Resúmenes por cohorte — descriptivos, muestra minúscula

### Los 6 de la petición

| grupo | n | pérdida media v1.2 | pérdida media v1.4 | Δ v1.4−v1.2 | v1.4 mejor | v1.4 peor |
|---|---|---|---|---|---|---|
| historial **completo** | 2 | 0,161984 | 0,187756 | **+0,025772** | 0 | 2 |
| historial **incompleto** | 4 | 0,235184 | 0,202077 | **−0,033108** | 4 | 0 |
| los 6 juntos | 6 | 0,210784 | 0,197303 | −0,013481 | 4 | 2 |

### Los 9 de esta corrida

| grupo | n | pérdida media v1.2 | pérdida media v1.4 | Δ v1.4−v1.2 | v1.4 mejor | v1.4 peor |
|---|---|---|---|---|---|---|
| historial **completo** | 3 | 0,151527 | 0,171152 | **+0,019624** | 0 | 3 |
| historial **incompleto** | 6 | 0,235876 | 0,212021 | **−0,023855** | 5 | 1 |
| los 9 juntos | 9 | 0,207760 | 0,198398 | −0,009362 | 5 | 4 |

Los dos grupos **no se suman** para decidir nada: uno se emitió con el historial
reparado y el otro con el hueco del 2026-08-07 al 09-04 abierto. Su condición
prospectiva es idéntica —los nueve se escribieron antes del primer lanzamiento,
con `registrado_utc` que lo demuestra— y su calidad de entrada no.

## Por qué estos números no rankean nada, y no es humildad de cortesía

**Los nueve partidos los ganó el local.** Con `y = 1` en todos,

    Brier_v14 − Brier_v12 = (p14 − 1)² − (p12 − 1)² = (p14 − p12)(p14 + p12 − 2)

y el segundo factor es siempre negativo porque las dos probabilidades son
menores que 1. Es decir: **en esta muestra el signo de la diferencia lo fija
enteramente cuál de los dos modelos fue más optimista con el local.** No mide
quién predice mejor; mide quién apostó más alto por el que resultó ganar.

Y eso explica exactamente el patrón de las dos cohortes, sin necesidad de
ninguna historia sobre el historial: en los 3 de historial completo v1.4 dio una
probabilidad local **menor** que v1.2 en los 3 → pierde en los 3. En los 6 de
historial incompleto la dio **mayor** en 5 → gana en 5. Una moneda que cae nueve
veces del mismo lado no compara dos modelos.

Contexto: nueve locales seguidos tiene probabilidad ≈ 0,4 % a una tasa base de
0,54. Es una racha, no un descubrimiento, y los nueve son del mismo día
(2026-09-07). La prueba `test_con_y_igual_a_uno_el_signo_de_la_diferencia_lo_fija_quien_predijo_mas_alto`
deja esa aritmética versionada para que nadie lea un ranking donde no lo hay.

**Umbral de decisión, sin mover**: n ≥ 900 pares con resultado (potencia 80 %
para ΔBrier = 0,001). Vamos por **9/900**.

## Pendientes y exclusiones, con su motivo

| categoría | n | motivo |
|---|---|---|
| pendientes, historial completo | 13 | emitidos y sin resultado todavía |
| pendientes, historial incompleto | 9 | emitidos y sin resultado todavía |
| **excluidos: par incompleto bajo el ajuste vigente** | 14 | en el corte `2026-09-07T00:04:24Z` sólo la pierna v1.2 lleva el sha vigente; la v1.4 de ese corte se escribió con el ajuste **`e72f9b3b44a396b6`**, ya superado (51 filas, 51 juegos). Un par tiene que salir de **un** ajuste por versión: mezclar dos v1.4 compararía dos modelos distintos |
| emisiones posteriores, completo | 596 filas · 31 juegos | conservadas, **no sumadas** |
| emisiones posteriores, incompleto | 30 filas · 15 juegos | conservadas, **no sumadas** |

Las 14 excluidas **no se pierden**: tres de esos juegos (823415, 823742, 824793)
sí están en la tabla, con el primer par completo que existió después. Eso es la
regla funcionando, no un rescate.

## Rentabilidad: no se calcula

No hay precio de ejecución guardado junto a estas emisiones ni reglas de
liquidación verificadas. Sin las dos cosas, cualquier ROI sería inventado. Este
informe mide **pérdida de probabilidad** (Brier), que es un hecho comprobable
contra el marcador oficial, y nada más.

## Publicación automática

El desglose por partido ya no es un cálculo a mano: `fbq.model.informe_pareado`
lo emite en **cada corrida** bajo `desglose_por_partido`, con fecha, equipos,
marcador, ganador, las dos probabilidades guardadas, los dos Brier, la
diferencia y el cotejo de abridores por lado. El cron de los `:55` lo publica en
`docs/informe_pareado_vigente.json` sin intervención.

Cada población trae además `perdida_media_v1_2`, `perdida_media_v1_4`,
`juegos_donde_v1_4_mejora`, `juegos_donde_v1_4_empeora`, `juegos_sin_diferencia`
y el conteo de abridores por estado (`coincidio`, `cambio`,
`sin_anuncio_al_corte`, `sin_dato`).

## Reproducir

```bash
python3 -m fbq.model.informe_pareado                 # el informe entero, con el desglose
python3 -m pytest tests/test_fbq_informe_pareado.py
```
