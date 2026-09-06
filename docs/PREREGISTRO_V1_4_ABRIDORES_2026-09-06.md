# Preregistro — v1.4, una contribución del motor de abridores

**Commiteado antes de medir, y antes incluso de que exista la muestra.** Es la
forma más fuerte de preregistro disponible: cuando los datos lleguen, la
variable, el procedimiento y el criterio ya estarán fijados por escrito.

**La v1.2 sigue siendo la referencia.** Esto se compara contra ella.

## 1. La contribución elegida: UNA sola

| | |
|---|---|
| **archivo** | `modules/baseball_module/context_engine/pitcher_engine.py` |
| **qué se recupera** | el **primer eslabón de la jerarquía de estimadores**, reducido a lo que se puede reconstruir: la calidad del abridor por su tasa de ponches menos la de bases por bolas |
| **posición original** | PASO 2 del pipeline: el abridor visitante ajusta λ del local y viceversa |

### Por qué `K% − BB%` y no SIERA, xFIP o ERA

- **SIERA y xFIP son el objetivo real, y `K% − BB%` es su núcleo.** Las dos
  fórmulas se construyen sobre ponches, bases por bolas y batazos; sin acceso a
  FanGraphs, la parte reconstruible es exactamente ésa.
- El propio proyecto midió que **SIERA ≈ xFIP** (r=0,94, redundantes) y que
  **xERA es el independiente** (r=0,70). Traer los tres sería doble conteo.
- **ERA queda descartada por decisión ya tomada**: el motor la pone al final de
  la cadena de respaldo por ser la más contaminada por BABIP, defensa y suerte.
- `K% − BB%` sale de **conteos oficiales** (ponches, bases por bolas, bateadores
  enfrentados), no de estimaciones, así que su reconstrucción es verificable.

### El motivo deportivo

El abridor es el jugador individual con más peso en un partido de béisbol.
`K% − BB%` mide lo que un lanzador controla sin intermediación de su defensa ni
de la suerte del batazo: cuántos outs se lleva solo y cuántos corredores regala.
Es la señal de talento de abridor más estable y con mayor poder predictivo por
unidad de muestra.

## 2. La variable

```
dif_calidad_abridor = q(abridor LOCAL) − q(abridor VISITANTE)

q(p) = regress( K%(p) − BB%(p),  media de la liga,  BF(p),  K_BF )
```

**Signo**: el abridor local somete a la ofensa visitante, así que un local mejor
sube P(gana el local). Positivo favorece al local, igual que `dif_pitagorica`.

`K%` y `BB%` se calculan sobre **bateadores enfrentados (BF)** acumulados en las
aperturas del lanzador **disponibles al corte**, con la misma ventana y el mismo
contrato temporal que el resto del modelo.

Se reutiliza `regress()` —ya copiada con atribución de `tte_formula.py` y
vigilada por un test contra el original— para el encogimiento hacia la media de
la liga.

## 3. Constantes, fijadas ahora

| constante | valor | por qué |
|---|---|---|
| `K_BF` | **300** bateadores enfrentados | Un abridor de temporada completa ronda los 700-800 BF, así que 300 encoge fuerte a quien lleva pocas aperturas y casi nada a quien lleva media temporada. **Deliberadamente conservador**, por la misma razón que `K_REGRESION=67`: medir el valor óptimo exigiría ajustarlo sobre los años de evaluación |
| `MIN_BF_ABRIDOR` | **150** | debajo de eso la tasa es ruido; el juego se **excluye**, no se imputa |
| `VENTANA_ABRIDOR` | **40** aperturas | ~2 temporadas de un titular, cruzando el borde de temporada igual que la ventana de equipo |

**El único parámetro aprendido es el coeficiente de la logística**, y se ajusta
**sólo con el pliegue de entrenamiento**, igual que los otros. La magnitud del
motor anterior (sus `PITCHER_ENGINE_WEIGHTS`) **no se reutiliza**: se calibró
sobre datos que incluyen los años de evaluación.

## 4. Lo que queda congelado

`VENTANA=162`, `K_REGRESION=67`, `LAMBDA_L2=1.0`, `EXPONENTE_PITAGORICO=1.83`,
`MIN_JUEGOS_PREVIOS=30`, `TOPE_DESCANSO=5`, y las **dos variables de la v1.2**.
La v1.4 es v1.2 **más una** columna. `b2b_visita` de la v1.3 **no** entra: quedó
cerrada por confundido con la estructura de serie.

## 5. Requisitos de disponibilidad temporal — los tres, o no se corre

1. **Identidad del abridor anunciado**, observada **antes** del corte del
   precio, leída con `AnunciosStore.vigente_antes()`. Nunca del boxscore.
2. **Estadísticas del abridor** acumuladas sólo sobre aperturas **terminadas**
   antes del corte, con el mismo criterio de fin medido + margen que ya rige.
3. **Ambos** abridores conocidos. Con uno solo, el juego se excluye y se cuenta.

Si alguno falta, el juego **se excluye**; no se imputa un abridor promedio.

## 6. El procedimiento de evaluación, fijado ahora

- Entrenamiento **expansivo** por temporadas enteras, como en la v1.2.
- Estandarización con estadísticos **del entrenamiento** únicamente.
- Comparación contra **v1.2**, **tasa base del entrenamiento** y **Pinnacle**,
  sobre **exactamente los mismos partidos y los mismos cortes**, con el Brier
  del mercado recalculado sobre la intersección.
- **Diferencia pareada** de Brier por juego, con su intervalo: la media pareada
  y su error estándar `sd/√n`, más el bootstrap agrupado por equipo del
  evaluador.
- Los **tres controles de invariancia**, con la prueba 3 inyectando la fuga en
  la variable nueva.
- Exportación por juego con `p_v12`, `p_v14`, `delta_p` y `delta_brier`.

## 7. Criterio de arranque — cuándo se puede correr

Medido sobre las 3.748 filas de la v1.3, la diferencia pareada de Brier por
juego tiene **desvío 0,010024**. De ahí sale, para detectar un efecto real al
95 %:

| efecto a detectar (ΔBrier) | n necesario | días de captura a ~15 juegos/día |
|---|---|---|
| 0,0050 | 16 | 1 |
| 0,0020 | 97 | 7 |
| **0,0010** | **387** | **~26** |
| 0,0005 | 1.545 | ~103 |

**Umbral de arranque: n ≥ 400 juegos** con los tres requisitos cumplidos, que
permite detectar un ΔBrier de 0,001 — el orden del efecto que un motor de
abridores debería producir si vale algo.

Hoy hay **42**. Ver el límite documentado en
`docs/COBERTURA_ANUNCIOS_2026-09-06.md`.

## 8. Expectativa, declarada de antemano

**Se espera que aporte más que el componente de contexto, y aun así que NO le
gane a Pinnacle.** El abridor es la pieza con más peso del partido, así que un
ΔBrier favorable del orden de 0,001-0,003 sería plausible. Pero el portón fuera
de muestra del sistema anterior ya pasó `pitcher` y dio negativo
(2025→2024: 0,24042 → 0,24095, mejora −0,00054), así que **un resultado positivo
grande sería motivo de sospecha** y tendría que pasar antes por los controles de
invariancia.
