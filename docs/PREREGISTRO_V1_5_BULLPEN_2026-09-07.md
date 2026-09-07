# Preregistro — v1.5 experimental: carga reciente del bullpen

**Escrito y commiteado el 2026-09-07, ANTES de calcular una sola métrica de la
variable nueva.** Lo único medido hasta acá es la COBERTURA de los datos (§2),
que no toca resultados ni precios.

v1.4 sigue en su evaluación prospectiva con el ajuste congelado `3258fd9e`, sin
tocar. Esto es una rama experimental aparte.

---

## 1. Qué existe en el sistema anterior, y cómo definía la ventana

Cuatro piezas, todas en el árbol declarado inválido (`modules/baseball_module/`).
Lo que importa de ellas no es el código: es **cómo definían la ventana temporal**,
porque ahí está lo que no se puede repetir.

| pieza | qué hace | **ventana temporal** | reutilizable |
|---|---|---|---|
| `data_fetchers.py::get_bullpen_workload` (l. 1000-1097) | IP de relevo reales leyendo boxscores | `[utcnow() − N días, utcnow()]`, por **cadena de fecha**; `abstractGameState == "Final"` | **NO** — ver abajo |
| `context_engine/bullpen_engine.py::_workload_mult` (l. 732) | convierte IP de 3 días en multiplicador de λ | consume `ip_last_3_days`, no define ventana | **NO** (es una forma funcional elegida a mano) |
| `advanced_pit_enrichment/bullpen_pit_builder.py::workload_facts` (l. 293) | `pitches_last_{1,3,7}_days` por equipo | `[cutoff − (N−1) días, cutoff]`, por **día de calendario**, INCLUYENDO el día del corte | **la idea sí, la ventana no** |
| `bullpen_relief_appearance_builder.py` | hechos de relevo desde Statcast crudo local | por partido; el relevo es «todo lanzador distinto después del primero» | **la REGLA DE ROL sí** |

### Lo que sí se reutiliza, explícito

1. **La regla de rol**: *el primer lanzador del equipo en el partido es el
   abridor; todos los demás son relevo.* Es de
   `bullpen_relief_appearance_builder.py` (`ROLE_RULE_VERSION =
   "bullpen_first_pitcher_excluded_v1"`) y es exactamente lo que hace
   `fbq/model/bullpen.py::filas_de_boxscore`. No hace falta clasificar rósters ni
   distinguir al *opener*: para el boxscore, quien abre es el primero.
2. **La variable cruda**: conteo de **lanzamientos** de relevo, no entradas.
   `workload_facts` ya usaba `pitch_count`; `get_bullpen_workload` usaba IP. Los
   lanzamientos son la unidad más fina y la que no depende de cómo se reparten
   los outs.
3. **La constante de referencia** `_NORMAL_IP_3D = 9.0` (≈3 IP de relevo por
   partido × 3 partidos) queda como **contexto documental**, no como parámetro:
   este preregistro no usa ningún multiplicador calibrado a mano.

### Lo que NO se reutiliza, y por qué

- **`[utcnow() − N días, utcnow()]`**. El extremo derecho es la hora de la
  CORRIDA, no el corte del precio. En vivo coincide por accidente; en cualquier
  reconstrucción histórica es el futuro. Es la misma clase de defecto que la
  Fase 2B ya pagó (`game_date` truncado en vez de `official_date`).
- **`abstractGameState == "Final"`**. Vale también para un partido **suspendido**.
  El propio proyecto lo documentó: el que distingue es `detailedState`.
- **La ventana por día de calendario, inclusive del día del corte**
  (`workload_facts`). Un partido jugado esa misma mañana entra en la ventana de
  un corte de la tarde **sólo si ya había terminado**, y una fecha sin hora no
  puede decidirlo. Con granularidad de día es indecidible; con `disponible_desde`
  es una comparación exacta.
- **`_workload_mult`**: multiplicador con pendientes (+1,2 % por IP arriba, −0,5 %
  por IP abajo) y topes elegidos a mano, sin ajuste. Acá el peso lo estima el
  ajuste, sólo con entrenamiento.

---

## 2. Cobertura — medida antes de definir nada

**Ninguna fuente local tiene relevistas por partido.** Comprobado:

| almacén | contenido | por qué no alcanza |
|---|---|---|
| `data/aperturas.db` | 27.958 filas · **517 lanzadores** | los 517 abrieron alguna vez (0 lanzadores con `MAX(es_apertura)=0`). El almacén nació de los abridores anunciados: **un brazo que nunca abrió no está** |
| `data/pit_cache_pitcher.db` | 463.967 instantáneas · 1.302 entidades · sólo 2024-2025 | acumulados de temporada por lanzador; sin conteo de lanzamientos y sin corte por partido |
| `data/results.db` | 8.087 partidos | marcador y fin medido; nada de pitcheo |
| `.cache/backtest/` | vacío | el Statcast crudo que alimentaba `bullpen_relief_appearance_builder` no está en este árbol |

Cobertura del almacén local para reconstruir la carga restando el abridor al
total del equipo: **13.743 de 16.174 equipos-partido = 85,0 %**, y ni siquiera
esos traen lanzamientos. Además mezclaría dos fuentes que no cuentan la misma
población — el error que este proyecto ya midió al intentar combinar
`fangraphs.pitcher.daily` con `savant.pitcher.rolling` (221/300 y 129/300).

**Se completa con lo mínimo, y gratis**: el boxscore oficial de MLB
(`/api/v1/game/{pk}/boxscore`), una llamada por partido, sin clave y sin cuota.
Da en el mismo documento la lista ORDENADA de lanzadores y el
`numberOfPitches` de cada uno, así que **un brazo que jamás abrió entra igual**:
no se enumera lanzadores, se lee quién lanzó en ese partido. Es además la fuente
que ya fecha los resultados y da `bf` en `fbq/model/aperturas.py`.

Ninguna consulta de pago. GANICUS no se toca.

---

## 3. La variable — definición exacta

**Una sola**, sumada a las dos de la v1.2:

    dif_carga_relevo(juego, corte)
        = ( P_local(corte) − P_visita(corte) ) / ESCALA_CARGA

donde para un equipo E:

    P_E(corte) = Σ pitches_relevo(p, E)  sobre los partidos p tales que
                 corte − 72 h  ≤  disponible_desde(p)  ≤  corte

y:

| elemento | definición |
|---|---|
| `pitches_relevo(p, E)` | suma de `numberOfPitches` de **todos** los lanzadores de E en `p` **menos el primero de la lista** (regla de rol reutilizada) |
| `disponible_desde(p)` | fin MEDIDO de la última jugada + `MARGEN_FIN` = 20 min (`fbq/results/fines.py`, ya preregistrado). Para un suspendido, el de la reanudación |
| `corte` | `captured_at` del precio de referencia — el último par de Pinnacle pre-juego. **El mismo corte que usan v1.2 y v1.4**, sin excepción |
| `ESCALA_CARGA` | **100,0**, constante fija. No se ajusta ni se estima: divide para que la variable entre a la ridge (λ=1,0) en un orden comparable al de `dif_descanso`. Estandarizar con media/desvío de la muestra metería el pliegue de prueba en el ajuste |
| 72 h | del enunciado del experimento. **No se prueban 24/48/96 y se elige la mejor**: eso sería elegir después de ver |

«Partidos finalizados durante las 72 h anteriores al corte» se traduce a
`disponible_desde`, que es cuándo el resultado se pudo usar, no cuándo empezó el
partido. Un partido que empezó hace 80 h y terminó hace 70 **cuenta**; uno que
empezó hace 3 h y sigue en juego **no**, y no puede colarse porque `VentanaPIT`
no lo tiene indexado.

### Signo esperado, declarado antes de medir

Más carga reciente del bullpen local → bullpen local más cansado → **peor** para
el local. Se espera **coeficiente negativo**. Un coeficiente positivo
significativo sería evidencia CONTRA la hipótesis de fatiga, no a favor de una
historia nueva.

## 4. Faltantes — tratamiento fijado antes de medir

| caso | tratamiento | por qué |
|---|---|---|
| el equipo no tiene ningún partido terminado en la ventana | **P_E = 0**, y se anota `juegos_ventana = 0` | no es un faltante: es un bullpen que no lanzó. El cero es el dato |
| un partido está en la ventana pero el almacén no tiene su fila | **la fila NO es computable** → `carga_ok = 0` | imputar la carga de un partido que existió es inventarla |
| un partido del equipo tiene `disponible_desde = None` (fin no fechable) y su día cae en la ventana | **la fila NO es computable** | no se puede afirmar que no aportó carga |
| la fila del almacén tiene `relevistas = 0` (nadie relevó) | **P contribuye 0** | el abridor lanzó completo: carga de relevo cero, verificada |

**Nunca se imputa.** Las filas no computables se excluyen y se cuentan, y —esto
es lo que protege la comparación— **se excluyen también de v1.2 y del mercado**,
de modo que las tres columnas se miden sobre exactamente las mismas filas.

## 5. Estimación y comparación

- **Peso estimado sólo con entrenamiento.** Pliegues expansivos ya existentes
  (`fbq/model/candidato.py::evaluar_expansivo`): entrenar con todas las
  temporadas ESTRICTAMENTE anteriores, evaluar la temporada T ∈ {2025, 2026}.
  Nunca un corte aleatorio.
- **Ajuste**: la misma logística ridge congelada — `LAMBDA_L2 = 1,0`, intercepto
  sin penalizar (`fbq/model/logistica.py`). Sin tocar.
- **v1.2 se re-ajusta sobre las mismas filas** de cada pliegue. Comparar el v1.2
  publicado (ajustado sobre un conjunto mayor) contra un v1.5 ajustado sobre la
  intersección compararía dos muestras.
- **Nulo**: Pinnacle desvigorizado, mismo devig multiplicativo, mismo precio de
  referencia, **recomputado sobre la intersección**.
- **Incertidumbre**: diferencia PAREADA de Brier por juego, con bootstrap
  **agrupado por equipo local** (30 clústeres, 2.000 remuestreos). Los juegos de
  un mismo equipo no son independientes; los errores iid inflan los t entre 5x y
  21x sobre este tipo de dato.

### Esta comparación es EXPLORATORIA, y así se reporta

2025 y 2026 ya fueron explorados por este proyecto: siete baselines, dieciséis
informes de auditoría, nueve motores medidos sobre esos mismos años. El diseño
temporal impide que el MODELO vea el futuro; **no impide que lo haya visto quien
eligió la variable**. Ninguna cifra de acá acredita nada. La única prueba limpia
es prospectiva, y v1.4 ya está en esa fila.

## 6. Controles de fuga que se aplican a la variable nueva

Los que ya existen, sin excepción y sin versión relajada:

1. **Compuerta estructural** (`VentanaPIT.exigir_disponible`): un partido
   posterior al corte levanta `FugaDetectada`. La variable lee la ventana por
   `disponible_desde`, así que hereda la compuerta por construcción.
2. **Control positivo**: una variable envenenada con el resultado del propio
   partido tiene que ser detectada. Si no lo es, el instrumento no sirve.
3. **Invariancia del proceso completo** (`fbq/model/invariancia.py`): perturbar
   el marcador de un partido POSTERIOR al corte no puede mover ni un bit de la
   fila. Se corre con la variable nueva incluida.
4. **Detector estadístico** (`BRIER_IMPLAUSIBLE = 0,22`): un Brier demasiado
   bueno se rechaza como implausible antes de celebrarlo.

## 7. Qué se publica pase lo que pase

Cobertura, código reutilizado, diferencias pareadas y su intervalo — **también
si el intervalo cruza el cero**, que es el desenlace más probable. Un
experimento que sólo se reporta cuando sale positivo no es un experimento.

**Criterio de cierre, declarado ahora**: v1.5 no reemplaza a v1.2 salvo que la
diferencia pareada de Brier sea negativa y su IC95 excluya el cero en los DOS
pliegues. Cualquier otro resultado cierra la variable como no demostrada, igual
que se cerró v1.3.
