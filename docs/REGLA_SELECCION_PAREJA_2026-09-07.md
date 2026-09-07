# Regla de selección de pareja — declarada antes de conocer resultados

**Commiteada el 2026-09-07, con 0 partidos evaluados.** En este momento
`results.db` no tiene ningún resultado de los 15 partidos con predicción
prospectiva verificada, y el informe pareado reporta `prospectiva_verificada: 0
pares con resultado`. La regla se fija ahora justamente por eso: elegir qué
pareja cuenta **después** de ver cuál acierta es la forma más barata de
fabricar un hallazgo.

## El problema que resuelve

Hoy hay **90 emisiones** `prospectiva_verificada` = **45 pares completos**,
sobre **15 partidos únicos**. Tres cortes por partido:

| corte | pares |
|---|---|
| 2026-09-07T02:18:51Z | 15 |
| 2026-09-07T02:20:15Z | 15 |
| 2026-09-07T03:51:57Z | 15 |

El generador corre cada hora a propósito —los abridores se anuncian a lo largo
del día y una corrida perdida no debe dejar hueco—, así que un partido acumula
tantas emisiones como corridas lo alcancen antes de empezar.

**Contar los 45 como muestra sería contar 15 partidos tres veces.** El error
estándar caería por √3 sin que exista una sola observación nueva, y el intervalo
de confianza mentiría por construcción.

## La regla

> **Para el informe principal se usa el PRIMER par completo y verificable de
> cada partido.**

Un par es elegible cuando cumple las cuatro condiciones a la vez:

1. las **dos versiones** —v1.2 y v1.4— existen;
2. **el mismo `game_pk` y el mismo `corte`**;
3. las dos con `origen = 'prospectiva_verificada'`, o sea `registrado_utc`
   anterior al primer lanzamiento;
4. cada una con el **SHA de su ajuste vigente** identificado en la fila.

Entre los pares elegibles de un partido gana **el de `corte` más temprano**;
si hubiera empate exacto de `corte`, el de `id` menor.

### Por qué el primero y no el último

- Es la regla que **no depende del resultado**: se puede evaluar el mismo día
  que se emite, antes de que el partido empiece.
- Es la **más exigente**: el par más temprano tiene la información más pobre —
  menos anuncios confirmados, menos horas de mercado— así que si v1.4 aporta,
  tiene que aportar ahí.
- «El último antes del inicio» sería defendible pero **depende de cuántas
  corridas alcanzó el partido**, que es una propiedad del cron y no del modelo.

## Lo que NO se descarta

Las emisiones posteriores **se conservan enteras** y se reportan **aparte**,
como `emisiones_posteriores`. Sirven para dos cosas que no son el informe
principal: ver cuánto se mueve una predicción a medida que se acerca el partido,
y detectar una corrida que empezó a producir algo distinto.

**Nunca se suman al informe principal**, y un partido no puede aportar más de
un par a la muestra.

## Cohortes, que siguen separadas

| cohorte | qué es | ¿entra al informe principal? |
|---|---|---|
| `prospectiva_verificada` | escrita antes del primer lanzamiento, con sello | **sí**, un par por partido |
| `emisiones_posteriores` | los demás pares del mismo partido | no, se reportan aparte |
| `reconstruccion` | calculada después, sobre un corte del pasado | no |
| `no_verificable` | evidencia de emisión original destruida | no |
| cohorte histórica de 37 | reconstrucción sobre cortes de agosto | no |

## Umbral de decisión, sin cambios

n ≥ 900 pares con resultado, potencia 80 % para ΔBrier = 0,001. Con **un par
por partido**, eso son 900 partidos — no 900 emisiones.

## Ajuste congelado

`v1.2 = d88d41ecc7dcd437` · `v1.4 = 3258fd9e648a0135`. **Sin tocar.**
