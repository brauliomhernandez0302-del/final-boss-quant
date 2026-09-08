# v1.6 — concentración de la carga entre relevistas: mejora no demostrada

**2026-09-08.** Ejecuta `docs/PREREGISTRO_V1_6_CONCENTRACION_2026-09-08.md`,
commiteado en `7508cb6` **antes** de medir rendimiento.

**Veredicto por el criterio preregistrado: v1.6 NO reemplaza a v1.2.** La
diferencia pareada es **positiva** —o sea peor— en los dos pliegues, y su IC95
cruza el cero en los dos. v1.2 sigue siendo la referencia; **v1.4
`3258fd9e648a0135` sigue intacta** en su evaluación prospectiva.

---

## 1. Qué se recuperó, y de dónde

### Las respuestas originales no existían: las tiré yo

Búsqueda en todo el árbol: **cero** boxscores guardados. `.cache/` sólo tiene
derivados del sistema anterior (`bullpen_reliever_ids_*`, 6 equipos de 2026);
`/home/raulio/respaldos-fbq/` sólo bases; `.cache/backtest/` vacío.

La causa es mía y del 2026-09-07: `fbq/model/bullpen.py::descargar` parseaba la
respuesta, guardaba dos agregados por partido y **descartaba el documento**.
Cuando hizo falta el detalle por relevista —que venía en el mismo documento— no
quedaba nada que releer.

Corregido: `descargar` ahora escribe cada respuesta en `data/boxscores/<pk>.json.gz`
antes de parsearla (escritura a temporal + `rename`, para que un corte no deje un
archivo truncado que parezca válido), y la relee de disco en vez de pedirla.

| | |
|---|---|
| respuestas guardadas | **8.095**, una por partido |
| fallidas | **0** |
| tamaño | 155 MB comprimidas (~1,4 GB sin comprimir) |
| reutilizadas en la reconstrucción por relevista | **8.095 — cero llamadas de red** |

Consultas nuevas: 8.095 a MLB StatsAPI, **gratis y sin clave**. No se tocó The
Odds API ni ninguna cuota de GANICUS.

### La tabla por relevista

`data/relevistas.db`, una fila por `(game_pk, team_id, pitcher_id)`:

    game_pk · team_id · pitcher_id · es_local · official_date · season
    orden · rol · pitches · bf · outs · k · bb
    fin_medido · disponible_desde · procedencia

| | |
|---|---|
| apariciones | **73.293** en 8.095 partidos |
| brazos distintos | **2.513** |
| **brazos que nunca abrieron** | **1.766 (70,3 %)** |
| brazos que el almacén de los 517 no tenía | **1.994** |
| apariciones de relevo | 57.100, de ellas **40.397 (70,7 %) de brazos fuera de los 517** |
| roles | `abridor` 16.190 · `relevo` 57.100 · `sin_lanzar` 3 |
| `procedencia` | `medido` 73.222 · `desconocido` 71 |

Las 71 filas `desconocido` no tienen `disponible_desde` y por eso **nunca entran
a ninguna ventana** — el mismo tratamiento que en todo el proyecto.

### Reglas y compuertas reutilizadas, no reinventadas

| regla | origen | uso acá |
|---|---|---|
| rol por orden | `bullpen_relief_appearance_builder.ROLE_RULE_VERSION` | primero que lanzó = `abridor`; los demás = `relevo` |
| entradas de cero lanzamientos | arreglo propio del 2026-09-07 | `rol = sin_lanzar`, no desplazan al abridor |
| `disponible_desde` | `fbq/results/fines.py` | fin medido + 20 min, con `procedencia` |
| ventana de 72 h | `fbq/model/carga.py::VENTANA_HORAS` (v1.5) | **la misma**, sin volver a elegirla |

## 2. Conciliación con los agregados por equipo

Dos almacenes construidos del mismo documento por caminos distintos:

| comprobación | resultado |
|---|---|
| equipos-partido en ambos | **16.190 de 16.190** |
| sólo en uno | 0 y 0 |
| discrepancias en **lanzamientos de relevo** | **0** |
| discrepancias en **lanzamientos totales** | **0** |
| discrepancias en **número de relevistas** | **0** |

**Diferencias concretas: ninguna.** Se comprobó porque este proyecto ya pagó el
no hacerlo —combinó `fangraphs.pitcher.daily` con `savant.pitcher.rolling`, que
coincidían en 221 y 129 de 300—, no porque se esperara un problema.

Un defecto propio sí apareció y se corrigió antes de medir: el puente
juego→equipo se armaba sólo con filas de rol `relevo`, así que un equipo cuyo
abridor lanzó el partido completo quedaba «sin identidad de equipo». Confundía
*no usó el bullpen* con *no sé quién es*: 57 filas mal etiquetadas, ahora
correctamente `sin_relevo_en_la_ventana` (o computables). Cobertura 98,12 % →
**99,11 %**.

## 3. Qué es este dato — y qué NO

Es **carga observada por relevista**: cuántos lanzamientos hizo ese brazo, en
ese partido, y desde qué instante ese hecho se pudo usar sin fugar.

**No es disponibilidad.** Que un brazo haya lanzado 40 lanzamientos en dos días
no demuestra que hoy no esté disponible, ni que esté lesionado, ni qué va a
decidir el entrenador. Un relevista puede no aparecer por rol —un cerrador que
no entra porque el juego no está cerrado—, por marcador, por descanso
programado, por una bajada a ligas menores, o porque el partido no lo pidió.
**Nada de eso está en este dato y nada de eso se infiere de él.**

El sistema anterior convirtió carga en fatiga con un multiplicador de pendientes
elegidas a mano (`bullpen_engine._workload_mult`). Eso es una hipótesis; este
almacén no la contiene, la deja medible.

## 4. La variable y su cobertura

    dif_concentracion = HHI_local − HHI_visita,   HHI_E = Σ_i (p_i / P)²

sobre los relevistas con `p_i > 0` en los partidos con
`corte − 72 h ≤ disponible_desde ≤ corte`.

| | |
|---|---|
| filas predecibles | 5.741 |
| **con concentración computable** | **5.690 (99,11 %)** |
| no computables | 51, todas `sin_relevo_en_la_ventana` (HHI = 0/0, no se imputa) |
| por temporada | 2024: 1.971/1.988 · 2025: 2.175/2.191 · 2026: 1.544/1.562 |
| HHI local | media 0,1931 · mediana 0,1787 · rango [0,0865, 1,0000] |
| brazos en la ventana | media 6,81 · rango [1, 13] |
| `dif_concentracion` | media −0,0033 · desvío 0,0831 · rango [−0,823, +0,841] |

Las filas no computables se caen de v1.6 **y de v1.2 y del mercado**, así que
las tres columnas se miden sobre exactamente los mismos juegos y los mismos
cortes.

## 5. Comparación pareada contra v1.2 y Pinnacle

Peso estimado **sólo con entrenamiento**; v1.2 re-ajustada sobre la misma
intersección; ridge λ=1,0 congelada; bootstrap agrupado por equipo local.

| pliegue | n eval | entrena | Brier v1.2 | Brier v1.6 | Brier Pinnacle |
|---|---|---|---|---|---|
| 2025 | 2.175 | 2024 (1.971) | 0,243212 | 0,243636 | **0,241376** |
| 2026 | 1.544 | 2024-2025 (4.146) | 0,248407 | 0,248459 | **0,246467** |

| pliegue | **v1.6 − v1.2** | IC95 | deff |
|---|---|---|---|
| 2025 | **+0,000425** | [−0,000362, +0,001160] | 1,320 |
| 2026 | **+0,000052** | [−0,000204, +0,000312] | 0,882 |

Positivo es **peor**. Los dos intervalos cruzan el cero, así que lo honesto no
es «v1.6 empeora» sino **«no se distingue de v1.2, y lo poco que se ve apunta a
peor»**. Ninguno de los dos le gana a Pinnacle.

### El coeficiente sí sale con el signo esperado — y eso no alcanza

| pliegue | dif_pitagorica | dif_descanso | **dif_concentracion** |
|---|---|---|---|
| 2025 | +0,29153 | −0,06582 | **−0,06843** |
| 2026 | +0,29583 | −0,03177 | **−0,02022** |

Negativo en los dos, que es lo que el preregistro declaró esperar: más
concentración → peor para ese equipo. **Y aun así el Brier no mejora.** Un signo
consistente con la hipótesis no es evidencia de valor predictivo: dice hacia
dónde apunta el ajuste dentro de la muestra de entrenamiento, no que sirva fuera
de ella. Es exactamente la distinción que separó a v1.3 —signo estable, cero
mejora— de una variable útil.

## 6. Asociaciones observadas (n = 3.719) — descriptivas

Error típico de r bajo H₀ ≈ 0,0164.

| asociación | r | r² | IC95 |
|---|---|---|---|
| concentración ↔ carga total (v1.5) | −0,4242 | 0,180 | [−0,4502, −0,3974] |
| concentración ↔ gana el local | +0,0141 | 0,000 | [−0,0181, +0,0462] |
| concentración ↔ P(local) del mercado | +0,0060 | 0,000 | [−0,0262, +0,0381] |
| concentración ↔ brazos usados (local − visita) | −0,7488 | 0,561 | [−0,7626, −0,7344] |

Lo que permiten decir, y nada más:

1. **No es una v1.5 reescalada**: comparte el 18 % de la varianza con la carga
   total. Comparten información —más carga suele repartirse entre más brazos—
   pero son estadísticos distintos.
2. **Con el resultado no se distingue de cero.** El intervalo tampoco descarta
   un efecto pequeño.
3. **Con el precio tampoco.** A diferencia de la carga total, acá ni siquiera
   hay asociación pequeña que discutir.
4. **El 56 % de su varianza la comparte con cuántos brazos se usaron**, que a su
   vez depende de cuántos partidos se jugaron. Es la limitación estructural del
   índice y estaba anticipada: HHI es, en buena medida, un recuento de brazos
   disfrazado de proporción.

Por cuartiles la relación no es monótona (extremos 0,5441 y 0,5570 contra un
implícito de mercado de ~0,530; centro por debajo), pero cada tasa lleva ±0,032
de intervalo: ninguna brecha se distingue del ruido, y elegir la que más se
despega entre cuatro es fabricar un hallazgo.

## 7. Controles

| control | resultado |
|---|---|
| ventana: un partido terminado después del corte no entra | **prueba versionada, verde** |
| el abridor no cuenta como carga del bullpen | verde |
| HHI = 1 con un brazo · 0,5 con dos iguales · invariante al total | verde |
| sin relevo en la ventana → **no computable**, nunca cero | verde |
| un equipo con juego completo conserva identidad | verde (era el defecto de §2) |
| invariancia real: mismas apariciones con y sin campos de marcador → mismo índice | verde |
| conciliación con los agregados | 16.190/16.190, 0 discrepancias |

1.165 pruebas en verde.

## 8. Cierre

v1.6 queda **cerrada como mejora no demostrada**, igual que v1.3 y v1.5. No se
prueban variantes —ni top-N, ni número efectivo de brazos, ni otras ventanas—:
el preregistro lo prohíbe explícitamente y ese es todo su valor.

**Lo que sí queda**: `data/relevistas.db` con 73.293 apariciones y las 8.095
respuestas originales guardadas. Cualquier pregunta futura sobre el mismo
documento —quién lanzó, cuándo, cuánto— ya no cuesta una sola llamada.

## 9. Reproducir

```bash
python3 -m fbq.model.bullpen --seasons 2024 2025 2026 --rehacer  # relee lo guardado
python3 -m fbq.model.relevistas                                  # tabla + conciliación
python3 -m fbq.model.v16 --salida docs/v16_concentracion_2026-09-08
python3 -m pytest tests/test_fbq_control_fuga.py tests/test_fbq_model.py
```
