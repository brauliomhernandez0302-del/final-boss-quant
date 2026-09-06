# Preparación del motor de abridores — qué se reutiliza, qué falta, y por qué

**Veredicto: NO se puede preregistrar la incorporación todavía.** Falta un dato
que **no es recuperable hacia atrás** y que exige **captura prospectiva**. Lo que
sí se entrega es la alternativa concreta, implementada y corriendo.

## 1. El código reutilizable

| pieza | qué hace | tamaño |
|---|---|---|
| `context_engine/pitcher_engine.py` | `adjust_for_pitchers()` — PASO 2 del pipeline. El abridor visitante ajusta λ del local y viceversa | 579 líneas |
| `config.py::PITCHER_ENGINE_WEIGHTS` | los pesos de la combinación **lineal** de factores | — |
| la jerarquía de estimadores | SIERA → xFIP → xERA → FIP → ERA, con el ERA crudo como último recurso por ser el más contaminado por BABIP, defensa y suerte | — |
| `ip_mlb_equivalent` (paso 5 de la auditoría) | cuánto se le cree a un abridor según la **procedencia** del dato: MLB actual 1.0, temporada anterior 0.50, AAA 0.25, AA 0.15 | — |
| regresión de splits por tamaño de muestra (paso 7) | prior poblacional por mano: RHP 1.144, LHP 0.852, medidos | — |

**Lecciones ya pagadas que se reutilizan como decisiones, no como código:**

- **La forma reciente está NEUTRALIZADA** (paso 8): pesaba 0.256 y era
  anti-predictiva — sus tres señales salían de la misma lista de 5 arranques y
  captaban regresión a la media leída como persistencia. No se vuelve a medir.
- `context_engine/pitchers_regression.py` fue **borrado** a propósito para no
  contar dos veces la corrección de suerte que SIERA/xFIP ya hacen.
- **xERA es independiente** de SIERA/xFIP (r=0,70 contra r=0,94 entre ellos), así
  que aporta donde los otros dos no; ojo con el doble conteo contra Savant.

## 2. Las entradas que YA tenemos

| entrada | de dónde | disponibilidad temporal |
|---|---|---|
| `est_woba`, `brl_percent` (xwOBA y barriles del abridor) | `data/pit_raw/raw_savant_*.db` — **2,14 M de lanzamientos**, columnas `pitcher`, `game_date`, `game_pk`, `events`, `estimated_woba_using_speedangle`, `woba_value/denom`, `launch_speed/angle` | ✅ agregable con corte por fecha, PIT por construcción |
| `k_pct`, `bb_pct` | derivables de `events` del mismo crudo | ✅ |
| `days_rest` del abridor | fecha de su última aparición, del mismo crudo | ✅ |
| `throws` (mano) | atributo estático del jugador, endpoint gratis de MLB | ✅ sin componente temporal |
| plato del bateador para splits | ⚠️ `stand` **no está** entre las 16 columnas guardadas | ❌ falta |
| `era`, `whip`, `fip` | requieren carreras limpias e hits oficiales | ❌ no están en el crudo |
| `siera`, `xfip` | FanGraphs | ❌ no están en almacenes propios |

**Segundo hueco, éste sí rellenable**: `pit_raw` cubre **2023-2025** y no tiene
**2026**. Savant es gratis e histórico, así que es una **descarga**, no una
captura prospectiva. ~2,8 GB.

## 3. El dato que falta, y por qué no se puede rellenar

**La identidad del abridor anunciado antes del corte de cada precio.**

El abridor que aparece en el boxscore **no demuestra que esa identidad
estuviera disponible antes del partido**. Y la API tampoco lo demuestra:

> **Medido el 2026-09-06 sobre 400 juegos al azar (800 equipo-juego):**
> el `probablePitcher` que la API publica **hoy** para un juego ya jugado
> coincide con el abridor real en **799**, y **difiere en 0**.
> Sin anuncio publicado: 1.

Los abridores se cancelan por lesión, enfermedad o lluvia varias veces por
temporada. Una tasa de discrepancia de **0,00 %** sobre 800 observaciones no
dice que nunca fallen: dice que **el campo se rellena con lo que pasó**. La API
no guarda historial del anuncio ni un instante de publicación.

Usar ese campo para 2024-2026 metería en cada predicción la identidad que sólo
se conoció al empezar el partido. Es una fuga, y de las que el detector
estadístico **no** atraparía —cambiar de abridor mueve el Brier poco— pero el de
invariancia sí.

El propio proyecto ya había chocado con esto y lo dejó escrito
(`CLAUDE.md`, 2026-08-02): *"resultó IMPOSIBLE de medir porque la API sólo
devuelve el probable ACTUAL"*. Esta medición lo confirma con número.

### Lo único que existe: la captura prospectiva del sistema anterior

`data/track_record.db`, campo `pipeline_json`, desde el commit `127bac6`:

| | |
|---|---|
| juegos distintos con identidad de abridor capturada | **88** |
| rango | 2026-08-01 → 2026-08-07 |
| anticipo sobre el primer lanzamiento | mediana **23,1 h** (mín 2,6 · máx 29,8) |
| observaciones posteriores al inicio | **0** |
| dentro del marco evaluable | 46 |
| **y observadas antes del corte del precio** | **43** |

**43 juegos no alcanzan para nada** —el pliegue de evaluación más chico tiene
1.557— pero sirven para dos cosas: validar que la captura nueva produce el mismo
dato, y probar que la anticipación real es de ~23 h, muy por delante del corte.

## 4. La alternativa concreta, ya implementada

**`fbq/anuncios/`** — un paquete nuevo, y aparte de `market/` y `results/` a
propósito, porque guarda una tercera clase de dato:

```
market/     un PRECIO observado          — hecho sobre el presente
results/    un MARCADOR                  — hecho sobre el pasado
anuncios/   quién va a abrir mañana      — AFIRMACIÓN SOBRE EL FUTURO
```

Lo que hace útil a un anuncio no es su contenido —que se puede volver a pedir
cuando sea— sino **el instante en que se lo observó**, que no se puede
reconstruir.

| | |
|---|---|
| fuente | `GET /api/v1/schedule?hydrate=probablePitcher` — **gratis, sin clave, sin cuota de pago** |
| almacén | `data/anuncios.db`, **append-only por trigger**: un UPDATE aborta |
| deduplicado | sólo se inserta lo que cambió; la barrida queda registrada igual, así que "no cambió" se distingue de "no miramos" |
| lectura del modelo | `vigente_antes(game_pk, lado, corte)` — lo que se sabía en ese instante. Sin observación previa devuelve `None` y el juego se excluye |
| corrida | `python3 -m fbq.anuncios.capturar --dias 3` |

**Ya corriendo**: primera barrida el 2026-09-06, 26 juegos y 52 anuncios
guardados.

> ⚠️ **No se instaló ningún cron.** Programarlo es una decisión operativa del
> dueño, y este proyecto convive con GANICUS, que tiene su propia programación.
> La cadencia útil es la misma que la de la captura de precios —lo que importa
> es tener observaciones antes de cada corte—, y el costo es **cero créditos de
> pago**: la API de MLB es gratis.

## 5. Cuándo se podrá preregistrar la incorporación

| condición | estado |
|---|---|
| identidad del abridor anunciado, con instante de observación | ⏳ **requiere captura prospectiva** — arrancó hoy |
| muestra suficiente para un pliegue | ⏳ a ~15 juegos/día, un mes da ~450 juegos; una temporada, ~2.400 |
| Statcast 2026 | ⬇️ descarga gratis pendiente (~2,8 GB) |
| mano del bateador (`stand`) para splits | ⬇️ re-descarga de Savant con la columna |
| SIERA / xFIP | ❌ FanGraphs; o se reemplazan por xERA propio desde Statcast |

**Con lo que hay hoy no se preregistra nada**, porque el preregistro exigiría
declarar una variable cuya disponibilidad temporal no se puede verificar — que
es exactamente lo que este paso vino a comprobar.

## 6. Lo que se puede hacer sin esperar

Un motor de abridores necesita saber **quién** abre. Pero hay dos piezas del
mismo paquete que **no** dependen de esa identidad y sí son reconstruibles hoy
con Statcast 2023-2025:

1. **La calidad del bullpen del rival** por xwOBA permitido, agregada por equipo
   —no por jugador— y por lo tanto sin necesidad de anuncio.
2. **La rotación como estructura**: cuántos días lleva descansando *el equipo*
   entre aperturas, que es un agregado de equipo y no una identidad.

Las dos siguen el mismo procedimiento ya establecido: preregistro, contrato
temporal, comparación contra v1.2 y Pinnacle sobre las mismas filas, controles
de invariancia y diferencia por juego.
