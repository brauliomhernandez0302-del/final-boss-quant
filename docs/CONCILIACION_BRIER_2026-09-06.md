# Conciliación de las tres cifras de Brier — 2026-09-06

Planteado por el dueño: si los tres números son medias simples por juego, los
dos grupos deberían reconstruir el total.

```
(6106 × 0,242124 + 139 × 0,227976) / 6245 = 0,241809097
publicado para "todos"                     = 0,241333
```

**La objeción es correcta.** La diferencia es de **0,000476** y no es ruido de
redondeo.

## Veredicto: la discrepancia estaba en el INFORME

No en los datos y no en el cálculo. Cada una de las tres cifras es correcta por
separado y se reproduce. Lo que estaba mal es **cómo se presentaron**: en una
tabla que invitaba a leerlas como una partición, cuando no lo son.

## La causa, medida

`solo_pregame=False` no significa "los mismos juegos, más los que se caen".
Significa **"quedate con la última cotización, sin mirar la hora"** — y para un
juego con captura en vivo propia, la última cotización *es* la de en vivo. O
sea que ese juego aparece en las dos filas de la tabla **con precios
distintos**.

Son **25 juegos**, todos de la ventana de captura propia del 2026-08-04 al
08-07, la única con trayectoria real:

| | Brier de esos 25 |
|---|---|
| con su precio **prepartido** | 0,234231 |
| con su precio **en vivo** | **0,115213** |

Seis de ellos, para ver qué información entra:

| game_pk | inicio | último precio prepartido | último precio (en vivo) | ganó local |
|---|---|---|---|---|
| 822865 | 2026-08-05T00:06 | 23:51 → p=0,6392 | 01:51 → **p=0,8483** | sí |
| 823108 | 2026-08-05T01:41 | 00:51 → p=0,5392 | 03:00 → **p=0,0983** | no |
| 823432 | 2026-08-04T22:40 | 21:51 → p=0,7407 | 23:51 → **p=0,9452** | sí |
| 823517 | 2026-08-04T23:06 | 22:51 → p=0,6520 | 00:51 → **p=0,8884** | sí |
| 823756 | 2026-08-04T23:40 | 22:51 → p=0,5736 | 01:51 → **p=0,9216** | sí |
| 824084 | 2026-08-04T23:41 | 21:51 → p=0,4229 | 01:51 → **p=0,9223** | sí |

En los seis el precio en vivo se mueve hacia el resultado. Es exactamente lo
que se espera de un mercado que ya vio parte del partido, y por eso ese precio
no puede ser el nulo.

## El puente aritmético

```
reconstrucción de las dos cohortes            0,241809162
efecto de los 25 juegos que cambian de precio −0,000476451
                                              ─────────────
Brier publicado para "todos"                   0,241332711  ✓ exacto
```

donde el efecto es `25 × (0,115213 − 0,234231) / 6245`.

## La partición que SÍ cierra

El universo de 6.245 se parte exactamente por **la clasificación temporal del
precio que cada juego usa**:

| cohorte | n | Brier |
|---|---|---|
| precio prepartido | 6.081 | 0,242156518 |
| precio en vivo | 164 | 0,210786560 |
| **total** | **6.245** | **0,241332711** |

Reconstrucción: `(6081×0,242156518 + 164×0,210786560)/6245 = 0,241332711`,
diferencia con lo publicado **0,00e+00**.

Y el marco de referencia también cierra desde su propio detalle:

| | n | Brier |
|---|---|---|
| su último precio ya era prepartido | 6.081 | 0,242156518 |
| tienen captura en vivo posterior, pero se usa la prepartido | 25 | 0,234230865 |
| **marco de referencia** | **6.106** | **0,242124067** |

Los 164 de en vivo se reparten en **139 sin ningún precio prepartido** (salen
del marco) y **25 que sí lo tienen** (se quedan, con el precio correcto).
`6.081 + 139 + 25 = 6.245` y `6.081 + 25 = 6.106`.

## Ponderación

**No hay pesos de ninguna clase.** `brier()` es `mean((p−y)²)` sobre las filas
del marco: **media simple por juego**. Una temporada pesa lo que pesa por su
número de juegos, y por eso 2026 —incompleta— pesa menos:

| temporada | n | Brier | peso en la referencia |
|---|---|---|---|
| 2024 | 2.358 | 0,239950202 | 38,6% |
| 2025 | 2.191 | 0,241417313 | 35,9% |
| 2026 | 1.557 | 0,246410820 | 25,5% |

Media simple de las tres medias de temporada: 0,242592778 — **distinta** de la
media por juego (0,242124067). La referencia es la **media por juego**; la de
temporadas se anota sólo para que nadie las confunda.

## La referencia no cambia

**n = 6.106 · Brier = 0,242124067.** Se calcula únicamente con precios
prepartido, así que nunca estuvo afectada por esto. Lo que se corrige es la
tabla comparativa de `docs/REFERENCIA_MERCADO_2026-09.md`, que ahora publica la
partición verdadera y esta advertencia.

## Reproducir

```bash
python3 -m fbq.evaluator.detalle --seasons 2024 2025 2026 \
        --salida docs/conciliacion_brier_2026-09-06.csv
```

Imprime el resumen **calculado desde el detalle**, no aparte de él. El CSV lleva
una fila por juego con `game_pk`, `official_date`, `season`, equipos,
`inicio_utc`, `snapshot_ts`, `p_home`, `brier`, `clasificacion_temporal`,
`snapshot_ts_prepartido`, `p_home_prepartido`, `brier_prepartido`,
`resultado_local`, `en_marco_prepartido` y `precio_cambia_al_filtrar`.

## Qué se hizo para que no vuelva a pasar

`tests/test_fbq_detalle.py` fija la reconstrucción como propiedad:

- `test_el_resumen_de_todos_se_reconstruye_desde_su_particion` — la partición
  por clasificación temporal cierra con tolerancia 1e-12.
- `test_prepartido_y_en_vivo_NO_son_una_particion_de_todos` — fija el error
  original: comprueba que **no** cierran, y que el puente publica la diferencia
  en vez de esconderla.
- `test_la_media_es_simple_por_juego_y_no_pondera_temporadas`.
