# Abridores anunciados — cobertura, evidencia y el límite

**Conclusión: la evaluación histórica de la v1.4 no es posible.** Hay **42**
partidos utilizables contra los 6.106 del marco. Lo que sí queda es la
evaluación **prospectiva**, con su umbral de arranque fijado y su fecha
estimada.

## 1. Cobertura

### Por temporada, contra el marco evaluable de la v1.2

| temporada | marco evaluable | registrados en el almacén | **ambos abridores conocidos antes del corte** | cobertura |
|---|---|---|---|---|
| 2024 | 2.358 | **0** | **0** | **0,0 %** |
| 2025 | 2.191 | **0** | **0** | **0,0 %** |
| 2026 | 1.557 | 63 | **42** | **2,7 %** |
| **total** | **6.106** | 63 | **42** | **0,7 %** |

Hay además **52 partidos registrados sin resultado todavía** — los de hoy y
mañana, capturados por la barrida nueva. No entran al marco porque aún no se
jugaron; son el comienzo de la serie prospectiva.

### Exclusiones, contadas

| motivo | n |
|---|---|
| `sin_precio_de_referencia` — el juego no tiene par de Pinnacle pre-juego, o todavía no se jugó | 68 |
| `anunciado_pero_vacio` — se observó antes del corte, pero sin abridor publicado | 3 |
| `sin_observacion_previa_al_corte` — sólo hay observaciones posteriores | 2 |

Ninguna se imputa. Un juego sin abridor conocido al corte se excluye, igual que
un juego sin precio.

## 2. Las fuentes, y la evidencia de la hora de captura

| fuente | qué aporta | evidencia del instante |
|---|---|---|
| **`track_record.pipeline_json`** del sistema anterior (commit `127bac6`, desde 2026-08-02) | 63 juegos con resultado, 2026-08-01 → 08-07 | `picks.published_at`, el instante real en que el proceso publicó. Mediana de anticipo sobre el primer lanzamiento: **23,1 h** (mín 2,6 · máx 29,8). **Observaciones posteriores al inicio: 0** |
| **`fbq.anuncios.capturar`** — schedule de MLB con `hydrate=probablePitcher` | 52 juegos futuros, desde 2026-09-06 | `observado_en` sellado en el momento de la barrida, y la barrida registrada en la tabla `barrida` aunque nada cambie |

Las dos son **gratis**. Cero créditos de pago.

**Lo que no es evidencia**: el `probablePitcher` que la API devuelve hoy para un
juego pasado. No trae instante de publicación ni historial, y está rellenado
—ver §3.

## 3. La prueba de que el relleno retroactivo es real

Dos mediciones del mismo fenómeno, sobre la misma pregunta, con resultados
opuestos:

| medición | discrepancia entre "abridor anunciado" y "abridor que abrió" |
|---|---|
| **API consultada hoy**, 400 juegos al azar (800 equipo-juego) | **0 de 800 — 0,00 %** |
| **Registro prospectivo real**, 42 juegos (84 equipo-juego) | **6 de 84 — 7,1 %** |

Un 7,1 % es lo que uno espera de la realidad: los abridores se caen por lesión,
enfermedad y lluvia. Un 0,00 % sobre 800 observaciones **no es posible** si el
campo guardara el anuncio. La conclusión no admite mucha discusión: **el campo
que la API publica para un juego pasado es el abridor que abrió, no el que se
anunció**.

Los seis casos, todos con el corte y las dos identidades:

| game_pk | lado | anunciado al corte | abrió | corte |
|---|---|---|---|---|
| 824891 | home | 527048 | 702275 | 2026-08-02T17:00 |
| 822783 | away | 700241 | 669461 | 2026-08-02T17:00 |
| 823270 | home | 650633 | 606996 | 2026-08-02T17:00 |
| 823432 | away | 641793 | 687223 | 2026-08-04T21:51 |
| 824402 | home | 677944 | 676440 | 2026-08-05T18:51 |
| 824402 | away | 690997 | 681035 | 2026-08-05T18:51 |

## 4. El registro conserva lo que se sabía entonces

El almacén guarda **5 llaves (juego, lado) con más de una identidad observada**,
cada reemplazo con su propia hora:

```
822867 home:  None   @ 2026-08-02T20:01:59  →  615698 @ 2026-08-03T20:00:47
824158 away: 656302  @ 2026-08-04T20:01:41  →  592791 @ 2026-08-05T14:00:32
824402 away: 690997  @ 2026-08-04T20:02:05  →  681035 @ 2026-08-05T20:00:38
824402 home: 677944  @ 2026-08-04T20:02:05  →  676440 @ 2026-08-05T20:00:38
824481 home: 668881  @ 2026-08-05T20:01:49  →  671096 @ 2026-08-06T14:00:29
```

**El caso 824402 es el que da sentido a todo el almacén.** El corte de su precio
fue el 2026-08-05 a las **18:51**; el anuncio correcto llegó ese mismo día a las
**20:00**, una hora y nueve minutos DESPUÉS. `vigente_antes()` devuelve los
abridores del 08-04 —los equivocados— porque **eso es lo que se sabía cuando se
observó el precio**. Un modelo que usara los otros estaría usando información que
no tenía.

Garantías, todas con test:

| garantía | mecanismo |
|---|---|
| un anuncio no se pisa | `anuncio` es **append-only por trigger**: un UPDATE aborta |
| cada reemplazo lleva su hora | `observado_en` por fila, normalizado a `+00:00` |
| pasar de "sin anuncio" a "anunciado" también es un cambio | se guarda `NULL` como valor, no como hueco |
| "no cambió" ≠ "no miramos" | deduplicado por cambio, con la barrida registrada igual |
| la importación trae el instante, no sólo la identidad | una fila sin `published_at` utilizable **no entra** |

## 5. El límite, y cuándo se levanta

**No hay evaluación histórica posible.** 2024 y 2025 tienen cobertura **cero** y
no la van a tener nunca: el dato no existe y no se puede reconstruir.

La evaluación **prospectiva** sí, y su umbral está fijado en el preregistro de
la v1.4. Medido sobre las 3.748 filas de la v1.3, la diferencia pareada de Brier
por juego tiene desvío **0,010024**:

| efecto a detectar (ΔBrier) | n necesario | días a ~15 juegos/día |
|---|---|---|
| 0,0020 | 97 | 7 |
| **0,0010** | **387** | **~26** |
| 0,0005 | 1.545 | ~103 |

**Umbral de arranque: n ≥ 400.** Hoy hay 42. A ritmo de temporada completa,
**un mes de captura** alcanza para un primer veredicto sobre un efecto de 0,001,
y una temporada entera para uno de 0,0005.

⚠️ **La captura no está programada.** `python3 -m fbq.anuncios.capturar --dias 3`
corre a mano; instalar el cron es decisión operativa del dueño, y este proyecto
convive con GANICUS, que tiene la suya. **Sin programarla, el reloj no arranca.**

## 6. Huecos, que quedan identificados

| hueco | estado | ¿se puede rellenar? |
|---|---|---|
| identidad del abridor anunciado, 2024-2025 | **0 %** | ❌ **nunca** — no existe el dato |
| identidad del abridor anunciado, 2026 en adelante | 2,7 % y creciendo | ✅ captura prospectiva, gratis, ya corriendo a mano |
| Statcast 2026 (`pit_raw` cubre 2023-2025) | falta | ⬇️ descarga gratis de Savant, ~2,8 GB |
| mano del bateador (`stand`) para splits | falta | ⬇️ re-descarga de Savant con la columna |
| `K%`/`BB%` por abridor y por apertura | falta | ⬇️ game logs de la API de MLB, gratis, fechados y por lo tanto reconstruibles con corte |
| SIERA / xFIP | falta | ❌ FanGraphs; se reemplazan por `K% − BB%` (ver preregistro v1.4 §1) |

## 7. Reproducir

```bash
python3 -m fbq.anuncios.importar_track_record   # rescata los 63 juegos históricos
python3 -m fbq.anuncios.capturar --dias 3       # barre el schedule, gratis
python3 -m fbq.anuncios.cobertura --salida docs/cobertura_anuncios_2026-09-06.json
```
