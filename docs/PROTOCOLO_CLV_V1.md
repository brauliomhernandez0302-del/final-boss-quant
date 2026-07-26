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
- engine_commit congelado: `b3325a52680e44ea75e1b1dcc26f9d9af6369393` (2026-07-25 — rebaseline
  del simulador post-auditoría VAL: truncamiento de walk-off, `rho_game` efectivo sobre las
  carreras, empates proporcionales. Baseline 0.24675 Brier / 55.05% accuracy,
  `audit_20260714/val_audit/rebaseline/backtest_report_20260725_0644.json`)
  - Anterior: `80d0faea8fae96a1bb11fd18d53cafcb382a1682` (2026-07-20). **Re-apuntado sin
    reiniciar muestra primaria**: D0 seguía pendiente al momento del cambio, así que la ventana
    no había arrancado y no existían picks primarios que invalidar. La regla de "cambio de motor
    a media ventana reinicia la muestra" no se activó porque no había ventana corriendo.
- **D0 = 2026-07-26** (fijado ese mismo día). Cumple las dos condiciones y es posterior al
  engine_commit de arriba, como exige la regla:
  - **(i) V1 cerrado como (a) — telemetría de pins certificada.** Evidencia fresca de la cadena
    de cron, no de llamadas de sesión: en `game_outcomes(source='live')`, **25 de 25** juegos
    analizados el mismo día de su horario tienen `ml_home_pin` y `ml_away_pin` poblados (19/19 el
    2026-07-19, 5/5 el 2026-07-23, 1/1 el 2026-07-25), más un lote completo de 15/15 para juegos
    del día siguiente el 2026-07-20. La cobertura baja en juegos a 1-2 días vista (31/72 en total)
    queda explicada por horizonte de análisis —el libro todavía no postea `h2h`— y no por una
    falla de captura: la misma tubería, el mismo código, pobla al 100% cuando el mercado existe.
    Esto es lo que el reporte del Commit 4 de la Fase 2A dejó pendiente de confirmar con un dato
    fresco. Residual honesto, registrado y no bloqueante: el lote de 15 del 2026-07-21 13:03 PDT
    salió con 0 pins siendo todos de día siguiente — consistente con horizonte, aunque un lote
    entero en cero se parece más a una pasada sin odds que a 15 ausencias independientes.
  - **(ii) Keepalive de Windows instalado y verificado.** Las cinco tareas
    (`WSL Keepalive FBQ 0655/0910/1255/1525/1825`) responden `Scheduled Task State: Enabled`,
    `Last Result: 0` y próxima ejecución agendada, consultadas con `schtasks.exe /query /v` desde
    WSL el 2026-07-26.
  - **Primeros picks primarios**: 25 picks publicados el 2026-07-26 entre 14:00:14 y 14:04:50 UTC
    (`picks` #144–#168), todos `publish_mode='quarantine'`, todos con `decision_prob` poblado.
    Son los primeros picks posteriores al engine_commit: la cadena de cron llevaba desde el
    2026-07-23 sin publicar por un crash de serialización en el publisher, ajeno al motor y
    arreglado en `f4a4bf4` (ver Clarificaciones).
- Clarificaciones:
  - **2026-07-26 — qué certifica el congelamiento del motor (y qué no).** El stamp
    `picks.engine_commit` es el HEAD del repo (`track_record/publisher.py::_get_engine_commit()`),
    así que avanza con cualquier commit —UI, docs, tests— sin que el motor haya cambiado: los
    picks de D0 quedaron estampados `840aa6e`, que es el commit de docs cuyo padre es el
    `engine_commit` registrado y que no toca una sola línea de motor. Lo que certifica la
    condición de validez #1 es el diff de los paths del motor contra el `engine_commit`
    registrado, y tiene que estar vacío. Verificado el 2026-07-26 (0 archivos) y automatizado en
    `tests/test_engine_freeze.py`, que además falla si hay cambios sin commitear en esos paths
    (el cron corre desde el working tree, no desde HEAD). No cambia ningún número ni definición
    del protocolo: aclara con qué instrumento se comprueba una condición ya escrita.
