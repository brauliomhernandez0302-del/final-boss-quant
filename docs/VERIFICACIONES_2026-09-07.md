# Tres verificaciones antes de contabilizar la evaluación prospectiva

**Resultado: dos defectos encontrados y corregidos.** El ajuste se re-congeló;
el anterior se conserva. La v1.2 sigue siendo la referencia.

## 1. Hora real de emisión — **defecto encontrado**

### Lo que estaba mal, y es mío

La tabla `prediccion` guardaba **sólo `corte`**. Un corte anterior al partido no
demuestra nada por sí solo: se puede declarar cualquier corte en cualquier
momento.

Peor: **borré `data/prospectiva.db`** antes de re-congelar, en un `rm -f` para
regenerar limpio. Los triggers de append-only protegen las FILAS de un UPDATE o
un DELETE; no protegen el archivo. Las emisiones originales —sha
`277364404636b308`, corte `2026-09-06T23:25:34`— **ya no existen**, y su
evidencia no se puede reconstruir.

### La corrección

Tres instantes distintos, y hacen falta los tres:

| campo | qué significa |
|---|---|
| `corte` | hasta dónde **miró** el modelo |
| `generado_utc` | cuándo se **calculó** la probabilidad |
| `registrado_utc` | cuándo se **escribió** en la tabla — lo pone el motor, no quien llama |

Más un campo `origen` con tres valores, **derivado del sello, no declarado**:

| origen | condición |
|---|---|
| `prospectiva_verificada` | `registrado_utc < commence_time` |
| `reconstruccion` | calculada después del inicio, sobre un corte del pasado |
| `no_verificable` | su evidencia de emisión original no existe |

Un trigger **impide** marcar `prospectiva_verificada` sin un `registrado_utc`
anterior al primer lanzamiento. Y `logs/prospectiva.log` guarda cada corrida:
una predicción cuya emisión sólo consta en la salida de una terminal no es
demostrable después.

**Las 102 filas anteriores quedaron como `no_verificable`** —vía el default de
la columna nueva, sin tocarlas— porque su emisión ya no se puede demostrar.

### Evidencia de las emisiones nuevas

```
 game_pk  corte                generado_utc         registrado_utc       inicio            sha               origen
  823415  2026-09-07T02:18:51  2026-09-07T02:18:52  2026-09-07T02:18:52  2026-09-07T17:05  3258fd9e648a0135  prospectiva_verificada
  824793  2026-09-07T02:18:51  2026-09-07T02:18:52  2026-09-07T02:18:52  2026-09-07T17:35  3258fd9e648a0135  prospectiva_verificada
  823742  2026-09-07T02:18:51  2026-09-07T02:18:52  2026-09-07T02:18:52  2026-09-07T18:10  3258fd9e648a0135  prospectiva_verificada
  823254  2026-09-07T02:18:51  2026-09-07T02:18:52  2026-09-07T02:18:52  2026-09-07T21:10  3258fd9e648a0135  prospectiva_verificada
```

**30 filas `prospectiva_verificada`** (15 juegos × 2 versiones), escritas entre
14 y 19 horas antes del primer lanzamiento.

### Segundo defecto, encontrado por la propia regeneración

La llave única pasó a incluir `modelo_sha`, pero **`CREATE TABLE IF NOT EXISTS`
no recrea una tabla existente**: el constraint viejo `(game_pk, version, corte)`
seguía vigente, y un ajuste nuevo colisionaba con el anterior sobre el mismo
juego. Se detectó porque la regeneración reportó **`"guardadas": 0`**.

Corregido con una migración que **reconstruye la tabla y copia todas las
filas**, verificando el conteo antes de reemplazar: **132 antes, 132 después**.
Si no coincidiera, aborta dejando el original intacto.

## 2. Aperturas frente a apariciones — cuadra

| | n |
|---|---|
| filas totales | **27.958** |
| aperturas (`gamesStarted = 1`) | **13.743** |
| relevos (`gamesStarted = 0`) | **14.215** |
| suma | 27.958 ✅ |
| **duplicados** `(pitcher_id, game_pk)` | **0** — lo impide la clave primaria |
| filas con `bf = 0` | 14 |

**El orden de los filtros**: primero se descartan relevos y aperturas no
disponibles, **después** se toman las últimas 40. Al revés dejaría ventanas de
menos de 40 sin avisar, con menos muestra de la declarada. Fijado con
`test_la_compuerta_va_ANTES_de_recortar_a_40`.

K, BB y BF se suman sobre **exactamente ese mismo conjunto**, con `K_BF=300` y
`MIN_BF=150` sin cambios.

## 3. Disponibilidad del historial — **defecto encontrado**

### Lo que estaba mal

`ventana()` filtraba con `game_date < día del juego`: una comparación de
**fechas**, no la compuerta del proyecto. Una apertura de anoche que terminó a
las 02:10 UTC pasaba como disponible para un corte de la 01:00 del mismo día.

### La corrección

La misma compuerta que el resto del proyecto:

```
fin_medido(apertura) + MARGEN_FIN (20 min)  ≤  corte
```

con el fin de la última jugada de `results/fines.py`. **Suspendidos y
reanudados**: el fin medido corresponde a la reanudación; como respaldo, el
inicio más tardío. Una apertura que no se puede fechar **no entra**.

La caché de fines se amplió de 7.664 a **8.086 juegos, 100 % con fin medido**,
para cubrir todas las aperturas hasta 2026-09-06.

Los llamadores pasan ahora el **instante de corte**, no el día: en
`congelar.py` el `captured_at` del precio de referencia, en `predecir.py` el
instante de generación.

### Comprobado con los controles existentes

| prueba de invariancia | resultado |
|---|---|
| 1 · el objetivo no se mueve | 14 comprobables, **0 movidas** → **PASA** |
| 2 · el complemento reacciona | 2.834 de 2.835, 1 exento verificado → **PASA** |
| 3 · fuga chica esquivando la compuerta | **NO RE-CORRIDA** — ver abajo |

⚠️ **La prueba 3 no se re-corrió y no se da por pasada.** Su arnés —un script de
trabajo, no código del proyecto— levanta un `TypeError` en
`_modelo_fijo` porque el pliegue de entrenamiento le llega vacío, y **no aislé
la causa**. La prueba 3 pasó en su última corrida válida (commit `7ddb8eb`,
antes de esta compuerta), así que lo honesto es decir que **está pendiente de
re-verificación**, no que sigue pasando. Las pruebas 1 y 2 sí se re-corrieron
completas con la compuerta nueva.

Nota lateral que sí confirma que la compuerta llegó a las variables: el juego
exento de la prueba 2 ahora reporta `x = (0.0, 0.0, 1.0, 1.0)` — cuatro
columnas, con las dos de `b2b` que la v1.3 dejó registradas.

## 4. El ajuste nuevo, y el anterior conservado

| | anterior | **vigente** |
|---|---|---|
| congelado | 2026-09-06 | **2026-09-07T02:18:32Z** |
| v1.4 sha | `e72f9b3b44a396b6` | **`3258fd9e648a0135`** |
| n entrenamiento | 2.608 | **2.608** |
| `dif_calidad_abridor` | +0,0789 | **+0,0787** |

El ajuste anterior **no se pisa: se apila**, en el campo `superseded` del mismo
archivo. La compuerta correcta movió el coeficiente en la cuarta cifra — señal
de que el defecto era real pero de efecto pequeño, no de que no existiera.

## 5. Informe pareado v1.4 − v1.2, sobre los partidos terminados

**DESCRIPTIVO, no veredicto.** El umbral de decisión es **n ≥ 900** pares con
resultado (potencia 80 % para ΔBrier = 0,001).

| población | pares | Brier v1.2 | Brier v1.4 | diferencia media | IC95 |
|---|---|---|---|---|---|
| **`no_verificable`** (cohorte histórica) | **37** | 0,242962 | 0,242829 | **−0,000133** | [−0,00766, +0,00740] |
| `reconstruccion` | 0 | — | — | — | — |
| `prospectiva_verificada` | **0** | — | — | — | — |

v1.4 mejora en 21 de los 37 juegos. **El intervalo cruza el cero por un factor
de 55**, así que no dice nada: `37 / 900` del umbral.

Un par vale lo que su **pierna más débil**. Los 37 históricos quedan etiquetados
`no_verificable` porque las filas de v1.2 se escribieron antes de que existieran
las columnas de evidencia; su sustancia es una **reconstrucción** —su corte tiene
un mes— y así se lee.

### Pendientes

| | n |
|---|---|
| `prospectiva_verificada` **sin resultado todavía** | **30** (15 juegos, todos de 2026-09-07 en adelante) |
| pares sin las dos versiones (ajuste viejo) | 14 |

## 6. Límite de las conclusiones

Con **37 pares de una reconstrucción** y **0 pares prospectivos con resultado**,
lo único que se puede decir es que el instrumento produce pares y que la
diferencia observada es indistinguible de cero. Ninguna afirmación sobre si v1.4
aporta.

La captura sigue corriendo cada 30 min y la generación cada hora; los primeros
resultados prospectivos llegan mañana.

## 7. Reproducir

```bash
python3 -m fbq.model.congelar                       # re-congela, apilando el anterior
python3 -m fbq.model.predecir --dias 3 --historicas # emite y sella
python3 -m fbq.model.informe_pareado
```
