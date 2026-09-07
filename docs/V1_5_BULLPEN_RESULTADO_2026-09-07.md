# v1.5 — carga reciente del bullpen: cerrada sin mejora demostrada

**2026-09-07.** Ejecuta `docs/PREREGISTRO_V1_5_BULLPEN_2026-09-07.md`, commiteado
en `2c5843c` **antes** de calcular una sola métrica.

**Veredicto por el criterio preregistrado**: v1.5 **no reemplaza a v1.2**. La
diferencia pareada de Brier es negativa en los dos pliegues pero su IC95 cruza
el cero en los dos, y el coeficiente sale con el **signo contrario** al esperado.
v1.2 sigue siendo la referencia; **v1.4 sigue intacta en su evaluación
prospectiva** con el ajuste congelado `3258fd9e`.

---

## 1. Código anterior: qué se reutilizó y cómo definía la ventana

| pieza del sistema anterior | ventana temporal que usaba | destino |
|---|---|---|
| `data_fetchers.py::get_bullpen_workload` (l. 1000) | `[utcnow() − N días, utcnow()]`, por cadena de fecha; `abstractGameState == "Final"` | **descartada** |
| `bullpen_pit_builder.py::workload_facts` (l. 293) | `[cutoff − (N−1) días, cutoff]`, por día de calendario, **incluyendo el día del corte** | **idea sí, ventana no** |
| `bullpen_relief_appearance_builder.py` (`ROLE_RULE_VERSION`) | por partido; relevo = todo lanzador distinto después del primero | **REUTILIZADA** |
| `bullpen_engine.py::_workload_mult` (l. 732) | consume `ip_last_3_days`; +1,2 %/IP arriba, −0,5 %/IP abajo, topes a mano | **descartada** (forma funcional sin ajustar) |

**Lo reutilizado, textual**: la regla de rol —*el primer lanzador del equipo en
el partido es el abridor; los demás son relevo*— y la unidad **lanzamientos**
(que ya usaba `workload_facts` con `pitch_count`; el camino en vivo usaba IP).

**Por qué se descartaron las dos ventanas**: `utcnow()` cierra la ventana en la
hora de la CORRIDA, que en cualquier reconstrucción histórica es el futuro; y una
ventana por día no puede decidir si el partido de esta mañana ya había terminado
al corte de la tarde. Acá la ventana se cierra en `disponible_desde` = fin
medido de la última jugada + 20 min (`fbq/results/fines.py`, ya preregistrado),
que es una comparación exacta.

**Un defecto de la regla de rol, encontrado y corregido**: en **3 de 16.174**
equipos-partido el primero de la lista figura con **cero lanzamientos** —
anunciado y retirado antes de lanzarle a nadie (Jon Gray, Mick Abel, Kyle
Freeland)— y el abridor real es el segundo. Con la regla literal, una apertura
entera (36, 33 y 24 lanzamientos) se contaba como relevo. Corregido saltando las
entradas de cero lanzamientos, que no son apariciones; en todos los demás casos
es idéntica a la regla original. Tras el arreglo, **0 filas** con
`abridor_gs ≠ 1`.

## 2. Cobertura

### Lo local no alcanzaba, y por cuánto

| almacén local | qué tiene | veredicto |
|---|---|---|
| `data/aperturas.db` | 27.958 filas · **517 lanzadores**, los 517 con ≥1 apertura | **24,9 % de cobertura** |
| `data/pit_cache_pitcher.db` | 463.967 instantáneas · 1.302 entidades · sólo 2024-2025 | sin lanzamientos ni corte por partido |
| `.cache/backtest/` | vacío | el Statcast crudo del builder anterior no está en este árbol |

**El número que decide**: apariciones de relevo reales **57.056**; las que tiene
el almacén de los 517: **14.215**. **Cobertura 24,9 % — faltaban 42.841.** La
advertencia era correcta: el almacén nació de los abridores anunciados y un brazo
que nunca abrió no está en él.

### Lo que se completó, gratis

Boxscore oficial de MLB, una llamada por partido, sin clave y sin cuota:

| | |
|---|---|
| partidos pedidos | 8.087 (todo `results.db`, 2024-2026) |
| **fallidos** | **0** |
| filas escritas | **16.174 = 2 × 8.087** — el 100 % de los equipos-partido |
| relevistas por equipo-partido | media 3,53 · máx 12 |
| lanzamientos de relevo | media 66,0 · rango 0-228 |
| equipos-partido sin relevo (el abridor completó) | 79 |

**Control independiente**: el `gameLog` de EQUIPO —otra ruta de la API, que
devuelve el total del staff por partido— contra la suma del boxscore, sobre
**2.805 equipos-partido** (6 equipos × 3 temporadas): **coinciden 2.805,
discrepan 0**.

### Cobertura de la variable sobre las filas evaluables

| | |
|---|---|
| filas predecibles | 5.741 |
| **con carga computable** | **5.741 (100 %)** |
| sin carga | 0 |
| partidos en la ventana de 72 h (local) | media 2,55 — distribución 0:49, 1:133, 2:2.238, 3:3.260, 4:61 |
| diferencia de lanzamientos, sin escalar | media +4,1 · desvío 67,7 · rango [−237, +270] |

Las 49 filas con **cero** partidos en la ventana son cero medido, no faltante:
un bullpen que no lanzó. Ninguna fila se imputó y ninguna se excluyó por falta
de carga, así que la intersección con v1.2 y con el mercado es **la misma que ya
existía** — las exclusiones son las de siempre (1.976 sin precio de Pinnacle
pre-juego, 370 por historial insuficiente).

## 3. Resultado — peso estimado sólo con entrenamiento

Pliegues expansivos: entrenar con las temporadas estrictamente anteriores,
evaluar la siguiente. Ridge λ=1,0 congelada, estandarización con media y desvío
**del pliegue de entrenamiento** (viaja dentro del modelo).

| pliegue | n eval | entrena | Brier v1.2 | Brier v1.5 | Brier Pinnacle |
|---|---|---|---|---|---|
| 2025 | 2.191 | 2024 (1.988) | 0,243238 | 0,243177 | **0,241417** |
| 2026 | 1.562 | 2024-2025 (4.179) | 0,248336 | 0,248228 | **0,246473** |

### Diferencias pareadas, con bootstrap agrupado por equipo local (29 clústeres, 2.000 remuestreos)

| pliegue | v1.5 − v1.2 | IC95 | efecto de diseño |
|---|---|---|---|
| 2025 | **−0,000061** | [−0,000369, +0,000227] | 1,353 |
| 2026 | **−0,000108** | [−0,000494, +0,000276] | 1,028 |

| pliegue | v1.5 − Pinnacle | IC95 |
|---|---|---|
| 2025 | +0,001760 | [−0,000488, +0,004140] |
| 2026 | +0,001755 | [−0,001033, +0,004678] |

**Los dos intervalos de v1.5−v1.2 cruzan el cero**, y la mejora puntual es de
1×10⁻⁴ — dos órdenes de magnitud por debajo de la distancia que separa al modelo
del precio. Ni v1.2 ni v1.5 le ganan a Pinnacle.

### El coeficiente sale con el signo CONTRARIO al preregistrado

| pliegue | dif_pitagorica | dif_descanso | **dif_carga_relevo** |
|---|---|---|---|
| 2025 | +0,29926 | −0,07369 | **+0,02744** |
| 2026 | +0,30159 | −0,02373 | **+0,03199** |

El preregistro §3 declaró: *más carga reciente del bullpen local → peor para el
local → coeficiente negativo*. Sale **positivo en los dos pliegues**. Por el
propio preregistro eso es evidencia **contra** la hipótesis de fatiga, no una
historia nueva que contar.

### Por qué, con los datos a la vista

| medición | valor |
|---|---|
| corr(carga, gana el local) | **−0,0046** — asociación cruda nula |
| corr(carga, P(local) del mercado) | **−0,1257** — el mercado YA la descuenta, y en la dirección de la fatiga |
| corr(carga, partidos jugados local − visita en la ventana) | **+0,4590** |

Esa última fila es lo que importa: casi la mitad de la variable es *cuántos
partidos jugó cada equipo en 72 h*, que es calendario — lo mismo que ya mide
`dif_descanso`, cuyo coeficiente se encoge al entrar la carga (−0,0800 → −0,0737
en 2025; −0,0312 → −0,0237 en 2026). No es una señal nueva: es una versión
ruidosa de una que ya estaba.

Por cuartiles de carga la relación tampoco es monótona (el cuartil más cargado
gana 0,5445 contra 0,5167 implícito del mercado, pero el MENOS cargado también
supera al mercado, 0,5505 contra 0,5440). Con ~930 juegos por cuartil el error
típico de una tasa es ~0,016: elegir el cuartil que más se despega entre cuatro
es fabricar un hallazgo.

## 4. Controles de fuga aplicados a la variable nueva

| control | resultado |
|---|---|
| **Invariancia del proceso completo** con `dif_carga_relevo` en el vector (27 partidos perturbados del 2025-06-10/11) | **14 comprobables · 0 variables movidas · 0 predicciones movidas** |
| **Complemento** (misma corrida) | 2.840 partidos posteriores usan los perturbados, **2.839 se movieron**; 1 exento por simetría documentada |
| **Control positivo** sobre la variable nueva (fuga de 0,005 inyectada) | **detectada** — `tests/test_fbq_control_fuga.py`, corre en CI |
| **Compuerta estructural** (`VentanaPIT`) | heredada: la ventana se lee por `disponible_desde` |
| **Detector estadístico** (`BRIER_IMPLAUSIBLE = 0,22`) | corrió en los 4 ajustes, ninguno implausible |

**Una limitación del control de invariancia, dicha explícitamente**: esa prueba
perturba **marcadores**, y la carga del bullpen no depende de marcadores, así
que su invariancia ahí es **trivial** — pasa sin demostrar nada sobre la
dimensión que importa. Lo que sí la demuestra son las pruebas de ventana
versionadas: un partido que terminó **después** del corte no entra aunque sea
del mismo día; uno que terminó 73 h antes tampoco; y cambiar la carga de un
partido **de dentro** de la ventana sí mueve la variable — sin esa última, una
constante pasaría todo lo demás con nota perfecta.

## 5. Dos correcciones al propio preregistro

1. **`ESCALA_CARGA = 100` es inocua pero redundante.** Se justificó como
   necesaria para que la ridge no aplastara la variable; en realidad `ajustar`
   ya estandariza con media y desvío **del pliegue de entrenamiento**, así que
   cualquier escala fija da el mismo ajuste. Se deja porque cambiarla ahora
   movería la definición después de medir; queda anotado que el mecanismo real
   de comparabilidad es la estandarización, no la constante.
2. **La regla de rol necesitó el arreglo de las 3 entradas de cero
   lanzamientos** descrito en §1, hecho antes de calcular ninguna métrica.

## 6. Esto es EXPLORATORIO

2025 y 2026 ya fueron explorados por este proyecto: siete baselines, dieciséis
informes de auditoría, nueve motores medidos sobre esos mismos años. El diseño
temporal impide que el MODELO vea el futuro; **no impide que lo haya visto quien
eligió la variable**. Ninguna cifra de acá acredita nada. La única prueba limpia
es prospectiva, y v1.4 sigue en esa fila sin que este experimento la toque.

## 7. Reproducir

```bash
python3 -m fbq.model.bullpen --seasons 2024 2025 2026     # 8.087 boxscores, gratis
python3 -m fbq.model.v15 --salida docs/v15_bullpen_2026-09-07
python3 -m pytest tests/test_fbq_control_fuga.py
```

Salidas: `docs/v15_bullpen_2026-09-07.csv` (3.753 juegos evaluados, uno por fila,
con la carga cruda de cada lado y los tres Brier) y `.json` (cobertura,
exclusiones, coeficientes, pareados con IC).
