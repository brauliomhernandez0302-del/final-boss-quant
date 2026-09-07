# Cierre de los dos pendientes operativos

Ajuste **`3258fd9e648a0135` sin tocar**; las predicciones existentes se
conservan y ninguna se reescribió.

## 1. Control positivo de fuga

### El `TypeError`, aislado — y su causa raíz

El fallo **no se reproducía en aislamiento**: con el mismo constructor
envenenado, `verificar()` corría bien. Al reproducir la **secuencia exacta**
apareció la causa, y no estaba en el arnés sino en el módulo:

> `invariancia.verificar()` usaba una ruta temporal **fija**,
> `data/_invariancia_tmp.db`. Una corrida anterior caída dejaba ese archivo a
> medio copiar; la siguiente lo sobrescribía mientras `_precios_de_referencia`
> lo leía → **0 precios → 0 filas → pliegue de entrenamiento vacío**, y ahí
> `X.std(axis=0)` devolvía un escalar en vez de un vector:
>
> ```
> TypeError: 'numpy.float64' object does not support item assignment
>   logistica.py:62   sd[sd == 0] = 1.0
>   invariancia.py:147 _modelo_fijo → ajustar(X, y, tuple(nombres))
>   invariancia.py:175 verificar → modelo = _modelo_fijo(base, temporada, nombres)
> ```
>
> Un síntoma a veinte llamadas de distancia de la causa, y que no decía qué
> faltaba.

**Dos correcciones, ninguna relaja una aserción:**

1. **Base temporal independiente por corrida**, vía `tempfile.mkdtemp()` y fuera
   de `data/`. Un ensayo no comparte archivo con otro ensayo ni vive junto a la
   base de producción.
2. **`ajustar()` falla legible** con un pliegue vacío: `ValueError` que dice
   exactamente eso, en vez de un `TypeError` sobre un escalar.

Con eso, la secuencia completa pasa: pruebas 1 y 2 limpias, y la prueba 3
detecta la fuga — `movidas = 14/14`.

### Versionada, y corriendo en CI

`tests/test_fbq_control_fuga.py` — **7 tests**, sin red y sin las bases reales.
Comprueba las **dos direcciones**, porque una sola no alcanza:

| | |
|---|---|
| pipeline **limpio** | cambiar el marcador del propio partido **no** mueve la variable del abridor ni la predicción |
| **fuga chica** (0,005) inyectada en la variable del abridor | sí la mueve, y la diferencia es exactamente `2 × 0,005` |
| por qué hace falta la invariancia | con esa fuga el Brier **no** cruza el umbral de plausibilidad: el detector estadístico no se enciende |
| el detector estadístico sí atrapa una fuga grosera | control de que no está roto en la otra dirección |

**Comprobado que la prueba muerde.** Simulando un detector que no ve la fuga
(magnitud 0):

```
E   assert 0.06000000000000001 != 0.06000000000000001
FAILED tests/test_fbq_control_fuga.py::test_una_fuga_chica_..._es_DETECTADA
1 failed, 6 passed
```

Si el detector dejara pasar la fuga, **el CI cae**.

## 2. Conservación y recuperación

### Búsqueda única de las emisiones perdidas — nada auténtico que recuperar

Buscadas las filas con sha `277364404636b308` en todos los respaldos y
artefactos conocidos:

| dónde | resultado |
|---|---|
| copias `.db` de prospectiva en disco (2) | **0 filas** con ese sha |
| `/mnt/c/Backups`, `~`, scratchpad | ninguna otra copia |
| artefactos de texto | sólo mis propios documentos y el log del congelado |

Lo único que sobrevive es **metadato** en el log de aquella corrida: corte
`2026-09-06T23:25:34.459868+00:00`, 14 candidatos, 28 filas guardadas.
**Ninguna probabilidad.** No hay nada auténtico que recuperar: las 102 filas
afectadas **se conservan como `no_verificable`** y así se cuentan.

### Respaldos: API de SQLite, versionados, fuera del proyecto

`fbq/respaldo.py` → `~/respaldos-fbq/`

| decisión | por qué |
|---|---|
| `Connection.backup()`, no `cp` | instantánea **consistente** aunque haya un escritor; copiar el archivo a mitad de una transacción da una base corrupta que se ve sana |
| **fuera** del árbol del proyecto | un respaldo en `data/` desaparece con el mismo `rm -rf` que borra lo que respalda |
| versionados por instante, nunca sobrescritos | con manifiesto que anota filas por tabla y `PRAGMA integrity_check` |

**Enganchados donde importa**: `respaldar(motivo="antes_de_migracion")` antes de
tocar el esquema, y `motivo="tras_predicciones"` después de cada tanda. Verificado
en vivo:

```
respaldo: prospectiva.20260907T035158Z.db · ok
```

12 entradas de manifiesto, dos copias versionadas de `prospectiva.db`.

### Restauración comprobada en un directorio temporal

```json
{ "integridad": "ok",
  "filas": {"produccion": {"prediccion": 199}, "restaurada": {"prediccion": 199}},
  "filas_coinciden": true, "esquema_coincide": true,
  "predicciones_identicas": true, "n_predicciones": 199,
  "shas": ["3258fd9e648a0135", "d88d41ecc7dcd437", "e72f9b3b44a396b6"],
  "origenes": ["no_verificable", "prospectiva_verificada", "reconstruccion"],
  "con_sello_de_emision": 97 }
```

No alcanza con que el archivo abra: se comparan **fila por fila** las
probabilidades, los sellos `generado_utc`/`registrado_utc`, los SHA de los
ajustes y los orígenes. `predicciones_identicas: true`.

**Restaurar sobre producción está prohibido por código** — un ensayo que puede
destruir lo que verifica no es un ensayo. `tests/test_fbq_respaldo.py` (6 tests)
lo fija, incluido uno que altera una copia a propósito y comprueba que la
verificación lo detecta.

Un test encontró además un **error real**: el manifiesto se escribía en una ruta
fija en vez de junto a las copias, así que un respaldo a otro destino quedaba
sin registro. Corregido.

## 3. Informe prospectivo, actualizado

**DESCRIPTIVO, no veredicto.** Umbral de decisión: n ≥ 900 pares con resultado.

| población | pares con resultado | Brier v1.2 | Brier v1.4 | diferencia media | IC95 |
|---|---|---|---|---|---|
| **`no_verificable`** (cohorte histórica, reconstruida) | **37** | 0,242962 | 0,242829 | **−0,000133** | [−0,00766, +0,00740] |
| **`prospectiva_verificada`** | **0** | — | — | — | — |
| `reconstruccion` | 0 | — | — | — | — |

**Pendientes**: **45** filas `prospectiva_verificada` sin resultado todavía
—crecieron de 30 a 45 con la tanda nueva— y 14 pares del ajuste viejo.

El intervalo de los 37 cruza el cero por un factor de 55. **Con 0 pares
prospectivos con resultado, no hay nada que concluir sobre si v1.4 aporta**; los
primeros llegan cuando terminen los juegos de hoy.

## 4. Sigue corriendo

| | |
|---|---|
| captura de anuncios | cada 30 min, **12 corridas automáticas** |
| generación pareada | cada hora a los :10 |
| GANICUS | sus **4 entradas**, intactas |
