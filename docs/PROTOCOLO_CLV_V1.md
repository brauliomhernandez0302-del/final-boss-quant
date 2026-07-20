# PROTOCOLO DE EVALUACIÓN DE CLV (PRE-REGISTRADO)

## Propósito
Decidir, con criterios fijados antes de recolectar datos, si los picks del modelo de moneyline MLB obtienen sistemáticamente mejor precio que la opinión final del mercado (closing line de Pinnacle) — el proxy estándar de edge sostenible. Este documento existe para que la respuesta la den números pre-comprometidos y no la tentación de leer ruido a favor.

## Definiciones
- **O_taken**: la cuota decimal del pick al momento de publicarse (el mejor precio disponible registrado en el pick).
- **p_close**: probabilidad justa del lado del pick al cierre = precio de cierre de Pinnacle de AMBOS lados, devigged con `remove_vig_multiplicative` (single source of truth). "Cierre" = último snapshot capturado antes del commence_time (mecanismo última-pre-inicio-gana de la Fase 2A).
- **EV-vs-close del pick** = p_close × O_taken − 1 (en %). Positivo = el pick obtuvo un precio mejor que el juicio final del mercado.

## Métrica primaria (la única que decide)
**Media de EV-vs-close** sobre la muestra primaria, con IC del 95% por bootstrap (10,000 remuestreos a nivel pick).

## Métricas secundarias (descriptivas, NO deciden)
1. % de picks con EV-vs-close > 0 (beat-close rate).
2. Brier del modelo vs Brier de p_close sobre los mismos juegos resueltos.
3. ROI/unidades del periodo — se registra y explícitamente NO decide: a esta n el ROI es ruido, y este protocolo pre-compromete que un ROI bueno no rescata un CLV malo ni viceversa.

## Cortes pre-registrados (descriptivos, máximo estos cuatro — ninguno adicional post hoc)
(a) por bucket de edge al publicar (<5%, 5–10%, >10%); (b) por staleness del cierre capturado (<60 min vs ≥60 min antes del inicio); (c) por libro de O_taken; (d) por mercado, si hubiera más de uno. La decisión vive SOLO en la métrica primaria sobre la muestra completa.

## Muestra primaria y D0
- **D0** = el primer día con: (i) veredicto V1 cerrado como (a) — telemetría de pins certificada — y (ii) keepalive de Windows instalado y verificado. D0 se escribe en la sección "Registro" de este documento al ocurrir (única edición permitida post-commit, logueada).
- Muestra primaria = todos los picks publicados desde D0 con cierre utilizable. Los picks de cuarentena anteriores a D0 son shakedown: se reportan como descriptivos, no entran a la primaria.
- **Atrición**: pick sin cierre utilizable (sin snapshot Pinnacle de ambos lados pre-inicio) → excluido de la primaria y contado. Si la atrición supera 15%, es un problema del instrumento: se arregla la captura y la ventana se extiende en consecuencia.

## Condiciones de validez (si se rompen, el reloj primario se reinicia)
1. **Motor congelado**: el engine de predicción de moneyline queda congelado en el commit registrado abajo durante toda la ventana. Esto incluye NO correr `promote_calibration.py` (una promoción cambia las probabilidades live). Cada pick se estampa con `engine_commit`; cualquier cambio de motor a media ventana reinicia la muestra primaria (lo anterior queda descriptivo).
2. La cuarentena publica TODO lo que el pipeline genera — sin banda de EV, sin filtros: la muestra es la opinión completa del modelo.
3. Construcción en paralelo permitida solo fuera del camino de predicción de ML: mercados derivados (F5, totals), higiene de backlog, tooling. Nada que toque probabilidades de ML.

## Calendario y decisión
- **Ventana base: 6 semanas desde D0** (objetivo ≥300 picks utilizables).
- **Miradas quincenales**: solo monitoreo de instrumento (atrición, staleness, volumen). Un resultado bueno a la semana 2 o 4 NO adelanta el veredicto — el optional stopping infla falsos positivos y aquí queda prohibido de fábrica.
- **Kill-switch (única decisión anticipada permitida)**: si en una mirada quincenal, con n≥150, el borde SUPERIOR del IC 95% de la primaria es < −1.0%, se cierra anticipadamente con veredicto NO-EDGE.
- **Al cierre de la ventana, tres veredictos posibles**:
  - **HAY SEÑAL DE EDGE**: borde inferior del IC 95% > 0 **y** estimador puntual ≥ +1.0%, con n≥300. → Se procede a diseñar la banda de publicación pública (2C) sobre estos datos.
  - **NO HAY EDGE** (de este modelo, contra estas líneas): borde superior del IC 95% < +1.0% — aun en el escenario optimista, el edge queda bajo el piso práctico (por debajo de ~1% de EV medio, costos de ejecución y límites se lo comen). → El moneyline no se publica públicamente; el esfuerzo pivota a mejora de modelo y/o mercados derivados.
  - **NO CONCLUYENTE**: el IC cruza ambos umbrales. → Extensión en bloques de 2 semanas hasta un máximo total de 10 semanas. Si a las 10 semanas sigue sin resolverse, el default pre-registrado es conservador: **tratar como no-edge para toda decisión de dinero**, y la evidencia acumulada informa la siguiente iteración del modelo.

## Lo que este protocolo NO decide
El veredicto aplica a ESTE modelo, en moneyline MLB, contra ESTOS libros. F5/derivados tendrán su propio protocolo cuando exista su motor. Un veredicto de no-edge no es un veredicto sobre el proyecto — es la respuesta honesta a una pregunta bien hecha, obtenida barata.

## Inmutabilidad
Desde D0, los umbrales y definiciones de este documento son inmutables. Solo se permiten clarificaciones que no cambien números ni definiciones, cada una logueada en "Registro" con fecha. Cualquier cosa mayor requiere un PROTOCOLO_CLV_V2 nuevo, commiteado ANTES de aplicarse, con changelog explícito — y reinicia el reloj.

## Registro
- Protocolo commiteado: 2026-07-20, commit siguiente a `80d0fae` en `feature/point-in-time-rebuild`
- engine_commit congelado: `80d0faea8fae96a1bb11fd18d53cafcb382a1682`
- D0: pendiente — completar cuando V1(a) + keepalive estén verificados
- Clarificaciones: (ninguna)
