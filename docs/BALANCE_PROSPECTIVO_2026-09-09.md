# Balance prospectivo de FBQ — corte del 2026-09-09 00:11:59 UTC

**Hora exacta del corte: `2026-09-09T00:11:59.901570+00:00`.** Desde esta
corrida el informe automático sella ese instante en cada salida
(`generado_utc`): la muestra crece sola cada hora, así que dos lecturas con
distinto número de partidos no son una contradicción sino dos cortes distintos.

Descriptivo. **No se declara ningún modelo ganador**, no se reajusta ningún
parámetro y no se cambia ninguna regla. v1.2 sigue de referencia, **v1.4
`3258fd9e648a0135` congelado**, v1.6 cerrada, todas las emisiones conservadas.

## Todos los partidos terminados disponibles al corte: 9

Regla congelada aplicada: **el primer par completo y verificable de cada
partido** (`docs/REGLA_SELECCION_PAREJA_2026-09-07.md`). Un partido aporta un
par; las emisiones posteriores se conservan aparte y no se suman.

| fecha | game_pk | visita @ local | historial | p(local) v1.2 | p(local) v1.4 | marcador | ganó | Brier v1.2 | Brier v1.4 | Δ v1.4−v1.2 | abr. local | abr. visita |
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

Negativo = v1.4 perdió menos que v1.2 en ese partido. **18 de 18 abridores
anunciados al corte coincidieron con quien abrió.**

### La jornada del 8 de septiembre no aporta partidos, y esto es por qué

**Al corte, ninguno de los 15 partidos del 2026-09-08 había terminado.** Estado
verificado contra el schedule oficial en ese mismo instante:

| estado | n |
|---|---|
| **In Progress** | 10 (entradas 1 a 6) |
| **Pre-Game** | 5 (inicio 01:40-02:10 UTC del 09-09) |
| Final | **0** |

Los 12 de ellos que ya tienen par emitido figuran como pendientes, no como
excluidos. El informe ahora publica `pendientes_por_fecha` para que esto se
responda sin abrir una base:

    {"2026-09-08": 12, "2026-09-09": 7, "2026-09-10": 3}

Entrarán solos en la corrida siguiente a que terminen.

## Resumen por cohorte

| grupo | n | pérdida media v1.2 | pérdida media v1.4 | Δ v1.4−v1.2 | IC95 | v1.4 mejor | v1.4 peor |
|---|---|---|---|---|---|---|---|
| historial **completo** | 3 | 0,151527 | 0,171152 | **+0,019624** | [+0,0074, +0,0319] | 0 | 3 |
| historial **incompleto** | 6 | 0,235876 | 0,212021 | **−0,023855** | [−0,0468, −0,0009] | 5 | 1 |

Los dos grupos no se suman: idéntica condición prospectiva —los nueve se
escribieron antes del primer lanzamiento, con `registrado_utc` que lo demuestra—
y distinta calidad de entrada.

⚠️ **Los dos intervalos excluyen el cero, en direcciones opuestas.** Salen de 3 y
6 diferencias pareadas con aproximación normal; a ese n el intervalo no tiene la
cobertura que su nombre promete. Que apunten a lados contrarios es precisamente
la señal de que no hay que leerlos como dos hallazgos.

## Corrección: qué permite y qué no permite una jornada de puras victorias locales

En el informe anterior escribí que estos números «no rankean nada». **Era
demasiado fuerte y se corrige.**

**Lo que sí permite.** La pérdida observada es un hecho: cada Brier se comprueba
contra el marcador oficial, y la diferencia pareada mide exactamente cuánto
perdió cada modelo en esos nueve partidos. Eso es comparable y es lo que esta
tabla reporta. En los nueve, v1.4 perdió menos en cinco y más en cuatro; por
cohorte, menos en las seis de historial incompleto salvo una, y más en las tres
de historial completo.

**Lo que no permite: concluir superioridad general.** Con `y = 1` en los nueve,

    Brier_v14 − Brier_v12 = (p14 − p12)(p14 + p12 − 2),   (p14 + p12 − 2) < 0

o sea que en esta muestra el orden de la diferencia coincide con el orden de las
probabilidades. No es un defecto de la medición —ser más optimista con el local
es justamente lo que hizo perder menos ese día— pero sí acota qué se puede
inferir: la muestra **sólo prueba un lado de la calibración**. Un modelo que
dijera 0,99 para el local ganaría esta comparación y sería pésimo. Nada de estos
nueve partidos dice cómo se comportan cuando gana el visitante.

Contexto: nueve locales seguidos tiene probabilidad ≈0,4 % a una tasa base de
0,54, y los nueve son del mismo día. Es una racha, y una racha informa poco
sobre el promedio.

**Umbral de decisión, sin mover:** n ≥ 900 pares con resultado (potencia 80 %
para ΔBrier = 0,001). Vamos por **9/900**.

## Cortes rechazados ≠ partidos excluidos

La corrección que faltaba. El informe anterior presentó «14 excluidos» y eso
**exageraba el daño en 13**:

| | n |
|---|---|
| **cortes** rechazados | **14** |
| partidos afectados | 14 |
| de ellos, **con otro par válido** (no pierden nada) | **13** |
| ├ ya evaluados en la tabla de arriba | 5 — 823175, 823415, 823742, 824793, 824958 |
| └ con par válido y partido sin terminar | 8 — 823500, 823738, 823821, 823901, 824551, 824714, 824792, 824957 |
| **partidos excluidos definitivamente** | **1** |

Los 14 cortes son todos el mismo: `2026-09-07T00:04:24Z`, donde la pierna v1.4
se escribió con el ajuste **`e72f9b3b44a396b6`**, ya superado por
`3258fd9e648a0135`. Un par tiene que salir de un solo ajuste por versión —
mezclar dos v1.4 compararía dos modelos— así que el corte se rechaza y el
partido busca su par en otro corte. Eso es la regla funcionando.

**El único partido excluido de verdad es `823903`** (Washington Nationals @ Los
Angeles Dodgers, 2026-09-06, 7-5): su única emisión es la de ese corte. Nota
adicional: su origen es `no_verificable`, así que tampoco habría entrado a la
cohorte prospectiva aunque el par estuviera completo.

Desde esta corrida el informe publica la distinción en su propia sección
`cortes_rechazados`, con `cortes`, `partidos_afectados`,
`partidos_con_otro_par_valido` y `partidos_excluidos_definitivamente`, más el
detalle corte a corte. `pendientes` queda sólo para lo que se resuelve
esperando.

## Emisiones posteriores y rentabilidad

Emisiones posteriores conservadas y **no sumadas**: 596 filas / 31 juegos
(completo) y 30 / 15 (incompleto).

**Sin rentabilidad**: no hay precio de ejecución guardado junto a estas
emisiones ni reglas de liquidación verificadas. Se mide pérdida de probabilidad
contra el marcador oficial, y nada más.

## Qué publica ahora el informe automático, en cada corrida

| campo | qué trae |
|---|---|
| `generado_utc` | la hora exacta del corte |
| `desglose_por_partido` | fecha, equipos, marcador, ganador, las dos probabilidades, los dos Brier, la diferencia y el cotejo de abridores por lado |
| `poblaciones[*]` | pérdidas medias, IC95, mejora/empeora/sin diferencia y abridores por estado |
| `cortes_rechazados` | cortes vs partidos afectados vs excluidos definitivamente, con detalle |
| `pendientes_por_fecha` / `pendientes_detalle` | por qué falta una jornada |
| `emisiones_posteriores` | conservadas, nunca sumadas |

Cron de los `:55` → `docs/informe_pareado_vigente.json`, sin intervención.

## Reproducir

```bash
python3 -m fbq.model.informe_pareado
python3 -m pytest tests/test_fbq_informe_pareado.py
```

---

## Re-corrida del 2026-09-09 00:23:30 UTC — mismo contenido

Se volvió a pedir el balance once minutos después del corte anterior. Resultado:
**idéntico**, y la razón está medida, no supuesta.

| | corte 00:11:59 | corte **00:23:30** |
|---|---|---|
| pares prospectivos evaluados | 9 | **9** |
| completo / incompleto | 3 / 6 | **3 / 6** |
| Δ completo · incompleto | +0,019624 · −0,023855 | **+0,019624 · −0,023855** |
| cortes rechazados / partidos excluidos | 14 / 1 | **14 / 1** |
| pendientes por fecha | 09-08: 12 · 09-09: 7 · 09-10: 3 | **igual** |

### Los partidos del 8 de septiembre, a las 00:23:30 UTC

Consultado al schedule oficial en ese instante: **0 de 15 terminados.**

| estado | n | detalle |
|---|---|---|
| **In Progress** | 10 | el más avanzado lleva 108 min y va por la 6ª entrada; tres llevan 43 min |
| **Pre-Game** | 5 | primer lanzamiento entre 76 y 106 minutos después del corte |

Un partido de nueve entradas dura unas tres horas, así que el primero de esa
jornada debería quedar disponible alrededor de las **01:30 UTC** y el último
—que empieza 02:10— cerca de las **05:10 UTC**.

### Cuándo entran solos

Sin intervención: el cron de resultados corre a los `:35` de cada hora y el
informe a los `:55`. Los primeros partidos del 09-08 entran en la corrida de las
**01:35 → 01:55 UTC**, y los últimos hacia las **05:35 → 05:55 UTC**.
`docs/informe_pareado_vigente.json` los recogerá con su propio `generado_utc`.

**Nada que corregir en esta re-corrida**: las cuatro correcciones pedidas —hora
exacta del corte, separación completo/incompleto, lectura de la jornada de puras
victorias locales, y cortes rechazados frente a partidos excluidos— ya están en
el informe automático desde el commit `d8f3615`, y siguen vigentes. Ajustes
congelados sin tocar; modelo sin cambios.
