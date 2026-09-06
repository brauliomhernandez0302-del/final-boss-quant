# Captura activada y predicciones pareadas — evidencia

**La v1.2 sigue siendo la referencia.** Nada de acá la reemplaza: lo que se
inicia es la única serie que producirá cifras que nadie escribió conociendo el
resultado.

## 1. Captura de anuncios: activada y verificada

| | |
|---|---|
| comando | `python3 -m fbq.anuncios.capturar --dias 3` |
| fuente | schedule de MLB con `hydrate=probablePitcher` — **gratis, sin clave** |
| frecuencia | cada 30 min |
| **cron instalado** | `*/30 * * * * …` |

**Exclusión mutua**: cerrojo `flock`, no "existe el archivo". Un cerrojo por
existencia deja el sistema trabado para siempre si el proceso muere; `flock` lo
suelta el kernel pase lo que pase. Verificado en vivo lanzando dos capturas a la
vez:

```
2026-09-06 19:17:17,470 INFO salteada: ya hay una captura corriendo (logs/anuncios.lock)
```

Una corrida que encuentra el cerrojo tomado **sale con código 0**: es lo
correcto para algo que dispara cada media hora.

**Registro de errores**: todo a `logs/anuncios.log`. Un fallo escribe la traza
completa y **sale con código distinto de cero** — un hueco en una captura
prospectiva no se rellena después, así que no puede quedarse en silencio.

### Evidencia de una ejecución AUTOMÁTICA real

No lanzada a mano. El cron disparó solo:

```
2026-09-06 19:30:02,699 INFO   2026-09-06: {'juegos': 15, 'nuevos': 0, 'sin_cambio': 30}
2026-09-06 19:30:02,963 INFO   2026-09-07: {'juegos': 11, 'nuevos': 0, 'sin_cambio': 22}
2026-09-06 19:30:03,222 INFO   2026-09-08: {'juegos': 15, 'nuevos': 0, 'sin_cambio': 30}
2026-09-06 19:30:03,224 INFO total: {'juegos': 41, 'nuevos': 0, 'sin_cambio': 82} | almacén:
   {'anuncios': 267, 'juegos': 130, 'barridas': 241, …}
```

Hora del sistema 19:30:02 (= 23:30:02 UTC), a los `:30` exactos. `nuevos: 0`
porque nada cambió desde la barrida anterior — y la barrida **igual quedó
registrada**, que es lo que distingue "no cambió" de "no miramos".

⚠️ **GANICUS intacto**: sus 4 entradas de cron siguen tal cual. Las dos nuevas
sólo usan la API de MLB; **no tocan la cuota de The Odds API**.

## 2. Con qué se entrena cada versión, y el ajuste congelado

`docs/modelos_congelados_2026-09-06.json`. Se entrena **una vez** y se lee de
ahí: un modelo que se re-entrena cada noche no produce una serie prospectiva,
produce una sucesión de modelos evaluados una vez cada uno.

| | v1.2 | v1.4 |
|---|---|---|
| entrenado con | **2024 + 2025** | **2024 + 2025** |
| n | **4.179** | **2.273** |
| variables | `dif_pitagorica`, `dif_descanso` | + `dif_calidad_abridor` |
| huella | `d88d41ecc7dcd437` | `277364404636b308` |
| intercepto | +0,1641 | +0,1337 |
| `dif_pitagorica` | +0,2965 | +0,2134 |
| `dif_descanso` | −0,0312 | −0,0531 |
| **`dif_calidad_abridor`** | — | **+0,1239** |

El coeficiente del abridor sale **positivo** —un mejor abridor local sube
P(gana el local)— y es **4 veces** el que tuvo `b2b_visita` en la v1.3. Eso es
una observación sobre el entrenamiento, **no evidencia**: el veredicto sale de
la serie prospectiva.

### La variable, y de dónde sale cada número

`dif_calidad_abridor = q(local) − q(visitante)`, con
`q = regress(K% − BB%, liga, BF, K_BF=300)`.

| dato | entrenamiento | aplicación prospectiva |
|---|---|---|
| `k_pct`, `bb_pct` | `pit_cache_pitcher.db` · `fangraphs.pitcher.daily`, instantánea PIT **anterior al día** | API de MLB, temporada en curso (gratis) |
| bateadores enfrentados | mismo almacén · `savant.pitcher.rolling.pa` | `battersFaced` de la misma respuesta |

**La sustitución de fuente está medida**, sobre 29 lanzadores con ≥100 BF en
2025: `K%` difiere en media **+0,00075** (máx 0,0152) y `BB%` en **−0,00038**
(máx 0,0124). Frente a una dispersión de K% entre lanzadores de 0,15-0,30, es
ruido de definición.

### La concesión declarada del entrenamiento

Para entrenar hace falta saber quién abrió en 2024-2025, y **el anuncio de
entonces no existe**. El entrenamiento usa el **abridor real del boxscore**.

- **No es fuga en la evaluación**, que es prospectiva y usa sólo el anuncio
  previo al corte.
- Magnitud acotada por la tasa de cambio de anuncio: **7,1 %**.
- El sesgo va **en contra** del modelo: entrena con identidades algo mejores que
  las que tendrá al aplicar.

Queda escrita en el propio archivo congelado, campo `concesion_declarada`.

## 3. Predicciones guardadas — evidencia

`data/prospectiva.db`, append-only por trigger, con **llave única
`(game_pk, version, corte)`** y un trigger que **rechaza cualquier corte
posterior al primer lanzamiento**.

| cohorte | versión | n | corte |
|---|---|---|---|
| **prospectiva** | v1.2 | **14** | 2026-09-06T23:25:34Z |
| **prospectiva** | v1.4 | **14** | 2026-09-06T23:25:34Z |
| **historica_42** | v1.2 | **30** | 2026-08-02 → 2026-08-05 |
| **historica_42** | v1.4 | **30** | 2026-08-02 → 2026-08-05 |

**14 pareados**, mismo corte, misma pasada, un solo reloj. Predicciones con
corte posterior al inicio: **0** — y no por disciplina, sino porque el trigger
las aborta.

Ejemplo, con su trazabilidad completa:

```
823903  Washington Nationals @ Los Angeles Dodgers   inicio 2026-09-07T02:10Z
  v1.2  p_home = 0.6367   sha d88d41ecc7dcd437
  v1.4  p_home = 0.6159   sha 277364404636b308
  abridores 680736 / 674841 · anuncio observado 2026-09-06T22:24:28Z
```

Las dos versiones se calculan **en la misma pasada**. Dos procesos separados no
lo garantizan: cada uno llamaría al reloj por su cuenta y leería el almacén en
un instante distinto. Es la misma lección que GANICUS dejó escrita para v2/v3.

Si a un juego le falta un insumo de **cualquiera** de las dos, **no se guarda
ninguna**: una serie pareada con huecos de un lado deja de ser pareada.

### La cohorte histórica va APARTE

Los 42 juegos con anuncio verificable quedan como `cohorte='historica_42'`, y
**no se mezclan**: otro capturador, otro corte, y las estadísticas del abridor
salen del almacén PIT en vez de la API. De los 42 quedaron **30**: 6 por
abridor con muestra insuficiente, 4 sin estadística, y **2 por un hallazgo
nuevo** —§5.

## 4. Cuotas moneyline: no hay precios nuevos

| | |
|---|---|
| fuente | **The Odds API**, vía `fbq/sources/odds_api.py` |
| requiere | `ODDS_API_KEY` — **no está en el entorno** |
| costo | **consume créditos**, de la misma bolsa que GANICUS usa para props |
| último precio propio | **2026-08-07T23:37Z**; posteriores: **ninguno** |
| juegos con predicción y evento de mercado enlazado | **0 de 14** |

**La comparación arranca entre modelos**: v1.4 contra v1.2, pareada, sobre los
mismos juegos y el mismo corte. **La comparación contra Pinnacle queda
pendiente** y no se abre sin decisión del dueño, porque exige consumo pagado.

Lo que sí se puede afirmar hoy es una diferencia relativa —si v1.4 mejora o
empeora sobre v1.2—, no una posición frente al mercado.

## 5. Hallazgo nuevo: el reloj del proveedor de cuotas va detrás del oficial

Al guardar la cohorte histórica, el trigger rechazó 2 juegos. Investigado:

| game_pk | corte del precio | inicio **oficial** | `commence_time` del proveedor |
|---|---|---|---|
| 823429 | 2026-08-05T22:51:02 | **22:40:00** | 22:41:00 |
| 823516 | 2026-08-05T23:51:01 | **23:05:00** | 23:06:00 |

El filtro pre-juego del almacén de precios compara contra el `commence_time`
**del proveedor**, que en estos casos va un minuto después del primer
lanzamiento oficial. Resultado: dos precios observados con el partido ya
empezado —uno a los **11 minutos**, otro a los **46**— pasaron como pre-juego.

**Alcance en el marco completo: 2 de 6.106 (0,03 %).** Chico, pero real, y
detectado sólo porque el almacén de predicciones compara contra el reloj
**oficial**. Los dos se excluyen de la cohorte histórica.

⚠️ **Queda identificado y NO corregido en el marco de la v1.2**: rehacer la
cadena entera por 2 juegos de 6.106 no se justifica hoy, pero el criterio
correcto es el reloj oficial y así queda escrito.

## 6. El cálculo de muestra, corregido

El número que publiqué antes —«n≈387 para detectar 0,001»— **estaba mal
etiquetado**. Tres correcciones:

**a) Origen del desvío.** Sale de la diferencia **pareada** de Brier por juego
entre v1.3 y v1.2, sobre 3.748 juegos de 2025-2026: **σ = 0,010024**. Es un
**proxy**: el contraste v1.4 − v1.2 es otro y tendrá su propio σ, que se
re-estimará con los primeros datos reales.

**b) Precisión no es potencia.** La fórmula que usé, `n = (1,96·σ/Δ)²`, da el n
para que el **IC95 tenga semiancho Δ**. Detectar un efecto de tamaño Δ con
**80 % de potencia** pide `(1,96 + 0,84)²` en vez de `1,96²`: **2,04× más
muestra**.

**c) Los partidos no son independientes.** Bootstrap agrupado por equipo local
(29 clústeres, 2.000 remuestreos): SE agrupado **0,00017331** contra SE iid
**0,00016374**. **Efecto de diseño = 1,120**, o sea que n juegos valen como
0,89·n independientes.

### Los números corregidos

| efecto ΔBrier | precisión (IC95 semiancho Δ) | **potencia 80 %** |
|---|---|---|
| 0,0020 | 108 | **221** |
| 0,0010 | 432 | **884** |
| 0,0005 | 1.730 | **3.534** |

**Todos incluyen ya el efecto de diseño.**

### Cuándo se decide, y cuándo se informa

| | |
|---|---|
| **informes descriptivos** | **desde el primer resultado**. Cobertura, cuántos pareados, la diferencia observada y su intervalo. Descriptivo, sin veredicto |
| **decisión sobre el modelo** | **al alcanzar n ≥ 900 juegos pareados con resultado** — potencia 80 % para ΔBrier = 0,001, el orden que un motor de abridores debería producir si vale algo |
| decisión anticipada | **sólo** si el IC95 de la diferencia pareada excluye el cero **y** n ≥ 221, o sea si el efecto real fuera de 0,002 o más |
| ritmo | ~15 juegos/día, pero limitado por la cobertura de anuncios y de estadísticas; hoy 14 de 15 juegos de la jornada |

**Ningún informe descriptivo previo a n ≥ 900 se cita como veredicto.** Mirar la
diferencia todos los días y decidir el día que da positivo es exactamente cómo
se fabrica un hallazgo.

## 7. Reproducir

```bash
python3 -m fbq.anuncios.capturar --dias 3        # cada 30 min por cron
python3 -m fbq.model.congelar                     # una sola vez
python3 -m fbq.model.predecir --dias 3            # cada hora por cron
python3 -m fbq.model.predecir --historicas        # la cohorte de 42, aparte
```
