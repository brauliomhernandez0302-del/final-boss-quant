# MAPA DEL PROYECTO COMPLETO — paso 0 a paso 19, subsistema por subsistema

`mapa_pasos.md` recorre **una rama**: cron → pick → reconciliación. Este documento recorre **el
proyecto**: los 8 entry points de la raíz, los 3 deportes, los 6 subsistemas y los 17 scripts.
Inventario verificado contra el filesystem, no contra la documentación.

Misma tripleta: **(P)** pregunta que debería responder · **(C)** la que responde · **(=)** veredicto.

**Inventario real** (2026-07-26): 8 archivos ejecutables en raíz · `modules/` 49 archivos /
17,256 L · `scripts/` 17 archivos / 3,665 L · `track_record/` 8 / 2,233 L · `core/` 2 / 1,243 L ·
`ui/` 5 / 987 L · `api/` 3 / 402 L · frontend React aparte.

---

## P0. El proyecto · **el paso 0 de verdad**

**(P)** ¿Qué pregunta responde este sistema, para quién?
**(C)** Según `CLAUDE.md`: *"analiza MLB, NBA y UFC e identifica oportunidades de apuesta con EV
positivo"*. Lo que hace medido: **MLB y nada más**.

| evidencia | dato |
|---|---|
| picks publicados por deporte | **170, todos MLB** (0 NBA, 0 UFC) |
| `publish_daily_picks` | `sports = ["MLB"]` hardcodeado, con el comentario *"NBA / UFC can be added as modules mature"* |
| `modules/basketball_module.py` | 2,411 líneas, con su `run_module()` |
| `modules/ufc_module.py` | 626 líneas, con su `run_module()` |
| `modules/football_module.py` | **no existe** — `CLAUDE.md` lo documenta como "(disabled)" en una ruta vacía |

**(=) DESALINEADO — P-1 (MEDIA).** ~3,000 líneas de dos deportes que se ejecutan desde la UI y
**nunca entran al circuito que importa** (track record, CLV, protocolo, backtest, calibración). No
es código muerto: es código *vivo sin propósito conectado*. Toda la maquinaria de honestidad que se
construyó en julio —cuarentena, engine freeze, CLV, provenance— existe sólo para MLB. El proyecto
responde "¿hay valor en este juego de MLB?" y se presenta como si respondiera para tres deportes.

---

## P1. `app.py` — la entrada humana (338 L)

**(P)** ¿Cómo le pregunto al sistema por un juego?
**(C)** Streamlit: elegí deporte → elegí juego del dropdown de odds → corré el analizador → mirá el
render.
**(=) ALINEADO con su propósito, DESCONECTADO del circuito de producción.** Es un camino
**paralelo** al cron: usa `ui/odds_loader.py` (que llama `get_odds_data()`, la otra implementación
del parser de odds — ver P10) en vez de `get_best_odds_for_teams()`, y guarda en
`PredictionsDB`/`predictions` en vez de `picks`. **Dos entradas al mismo motor con dos caminos de
datos y dos destinos de persistencia distintos.** → **P-2 (MEDIA)**.

## P2. `ui/` (987 L) — sidebar, componentes, odds_loader, render MLB

**(P)** ¿Cómo muestro lo que el motor concluyó, sin mentir?
**(C)** Eso, y con historial de arreglos de honestidad reales (el λ final que mostraba la etapa
equivocada, la confianza hardcodeada en 0.5).
**(=) ALINEADO.** Nota: `ui/mlb.py` y el frontend React (P14) son **dos UIs** que responden la
misma pregunta con implementaciones separadas; hoy sólo la de React recibió el trabajo de
honestidad (waterfall, aportes VAL-6, bullpen usado). La de Streamlit no.

## P3. NBA (`modules/basketball_module.py`, 2,411 L)
**(P)** ¿Hay valor en este juego de NBA? **(C)** Produce un `run_module()` con la misma forma de
salida que MLB. **(=) Indeterminado — nunca se evaluó.** Sin backtest, sin calibración, sin picks
publicados, sin protocolo. Ninguna de las 11 desalineaciones de MLB se buscó acá; podrían estar
todas, o no. **No se auditó** — declararlo es parte del hallazgo.

## P4. UFC (`modules/ufc_module.py`, 626 L)
Igual que P3, a menor escala. **Indeterminado, nunca evaluado.**

## P5. Fútbol
**(C)** El archivo no existe. `CLAUDE.md` lo lista con ruta. → documentación que promete un
componente ausente.

---

## P6. El pipeline de MLB
→ auditado en `mapa_pasos.md` (fases A-H, 30 micro-pasos, 11 desalineaciones + 15 huecos).

---

## P7. `backtest_and_retrain.py` (3,463 L) — el que produce el número canónico

**(P)** ¿Qué tan bueno es **el modelo que corre en producción**?
**(C)** ¿Qué tan bueno es **un modelo con el mismo código de motor pero alimentado por otro camino
de datos, sin clima, con otros pesos de pipeline y otra procedencia de sesgo**?
**(=) DESALINEADO — P-3 (ALTA). Es la desalineación estructural más grande del proyecto.**

Cuatro divergencias medidas entre lo que el backtest evalúa y lo que producción ejecuta:

| dimensión | backtest (canónico `--use-full-pit`) | producción (`run_module`) |
|---|---|---|
| **fuente de features** | cachés PIT (`advanced_pit_enrichment`, 25 archivos / ~6,600 L) | `SavantFetcher`/`FanGraphsFetcher` en vivo — **0 referencias a PIT en `run_module.py`** |
| **clima** | inexistente (`# DEFERRED F7`, fetcher comentado) → 12 valores distintos de park | activo → 102 valores distintos en 140 juegos |
| **pesos del pipeline** | los del backtest (`pitcher` 0.96-1.06) | los live del 2026-07-17 (`pitcher` **1.23**) |
| **pool del sesgo** | filas `backtest_*` de esa temporada | 82% filas de otra procedencia (CHRON-001) |

Ninguna de las cuatro es un bug: cada una tiene su razón (PIT evita leakage, el clima histórico es
caro, los pesos live se promovieron en su momento). **El problema es acumulativo**: el número
0.24675 describe un sistema que no es el que publica los picks, y nada en el proyecto mide esa
brecha. Cada vez que se cita el baseline como "qué tan bueno es el modelo", se está citando otro
modelo.

## P8. `advanced_pit_enrichment/` (25 archivos, ~6,600 L) — la segunda implementación

**(P)** ¿Cuáles eran los features de este equipo/pitcher **en la fecha del juego**, sin mirar el
futuro?
**(C)** Eso, correctamente — es la infraestructura anti-leakage y funciona.
**(=) ALINEADO en su pregunta, DESALINEADO por duplicación — P-4 (ALTA).**

Consumidores verificados: `backtest_and_retrain.py`, `tte_formula.py` y **tests**. `run_module.py`:
**cero**. O sea que los features de producción y los del backtest se calculan con **dos
implementaciones independientes**, y ya se encontraron **dos derivas reales** entre ellas:

- `LG_BARREL_PA` en `tte_pit_adapter.py`: *"mismo bug ya arreglado en el motor en vivo, pero esta
  copia independiente del path PIT nunca recibió el fix"* (CLAUDE.md, 2026-07-11/12).
- MATH-002 (2026-07-18): el adaptador PIT regresionaba barrel% per-PA mientras el motor vivo usaba
  per-attempt — numerador, prior y n distintos. Corregido igualando al vivo.

Dos derivas encontradas en las dos veces que alguien fue a mirar. **No existe ningún test de
paridad PIT↔live** que compare, para el mismo juego y fecha, los features que produce cada camino.
→ es el hueco que hace que P-3 sea difícil de acotar.

## P9. `data_enrichment/` — los fetchers vivos
**(P)** ¿Cuáles son los features **actuales** de este equipo/pitcher? **(C)** Eso, con caché de 24h.
**(=) ALINEADO**, con el hueco de que **nada mide la antigüedad del dato en el pick**: un pick de
las 07:00 puede estar usando stats cacheadas de hace 24h y nada lo registra.

## P10. `odds_fetcher.py` (851 L) + `fetch_historical_odds.py` (581 L)
**(P)** ¿Qué precio ofrece el mercado, ahora y en el pasado?
**(C)** Eso — con **dos parsers paralelos del mismo JSON**: `_normalize_event()` (para el dropdown
de Streamlit) y `get_best_odds_for_teams()` (para el cron y el backtest).
**(=) ALINEADO con riesgo de deriva — P-5 (MEDIA).** Ya pasó: el bug de `g_home/g_away` obsoletos
vivía en uno solo, y ayer tuve que aplicar el fix de precios de Pinnacle **a los dos** para que no
divergieran. Misma clase que P-4: dos implementaciones de la misma pregunta.

## P11. `core/value_detector.py` (1,233 L)
→ auditado (fases E y F de `mapa_pasos.md`): **PURP-1** (runline), **PURP-4** (EV/tier),
**H-15** (umbral de moneyline aplicado a todo).

## P12. `calibration/learning_engine.py` (1,839 L)
→ auditado (**PURP-3/6/7**, medido: apagar el sesgo mejora el Brier 0.00181).

## P13. `track_record/` (8 archivos, 2,233 L)
**(P)** ¿Qué prometí antes del juego y qué pasó realmente?
**(C)** Eso, y es el subsistema **mejor alineado del proyecto**: `publish_pick` idempotente,
`VOID` explícito ante ambigüedad, rechazo de capturas post-inicio, cuarentena etiquetada.
**(=) ALINEADO**, con los huecos ya listados (H-8 fallback sintetizado, H-9 no puede cambiar de
opinión, H-10 `model_prob`≡`decision_prob`).

## P14. `api/` (402 L) + frontend React
**(P)** ¿Puedo ver por qué el modelo piensa lo que piensa? **(C)** Eso, y desde hoy con la cascada
de λ, los aportes VAL-6 y el bullpen realmente usado. **(=) ALINEADO** — es el único subsistema
donde "explicar" es el propósito declarado y cumplido.

## P15. `scripts/` (17 archivos, 3,665 L) — cuatro familias

| familia | archivos | (P) | (=) |
|---|---|---|---|
| **builders PIT** | 8 (`build_*`, `backfill_*`) | ¿Puedo reconstruir el estado histórico? | ✓ ALINEADOS |
| **diagnóstico** | 4 (`diagnose_gradient`, `math003_*`, `closing_capture_coverage`, `smoke_*`×3) | ¿Este componente hace lo que dice? | ✓ — y son el mejor hábito del proyecto |
| **promoción** | 1 (`promote_calibration.py`) | ¿Muevo la calibración validada a producción? | ⚠ existe y **está prohibido correrlo** por el protocolo; es el único puente backtest→live y hoy está clausurado |
| **reporte** | `clv_report.py` | ¿Estoy ganándole al cierre? | ⚠ ML-only, como todo lo de CLV hasta ayer |

**P-6 (MEDIA)**: `promote_calibration.py` es el **único** mecanismo diseñado para cerrar la brecha
P-3 (backtest→producción), y la política vigente lo prohíbe. Con lo cual la brecha no se cierra ni
se mide: se congela.

## P16. `analyze_game_outcomes.py` (558 L)
**(P)** ¿Dónde falla sistemáticamente el sistema (estadio, mes, drawdown)?
**(C)** Eso, como script de una sola vez sobre 5,422 juegos.
**(=) ALINEADO pero huérfano**: nadie lo corre periódicamente, no está en cron ni en tests. Es el
germen del monitoreo que falta (**H-11**), sin conectar.

## P17. Datos — `predictions_history.db`, `track_record.db`, `.cache/`, cachés PIT
**(P)** ¿Cuál es el registro de lo que el sistema predijo y de lo que pasó?
**(C)** Eso, con `source` por procedencia desde CHRON-001.
**(=) ALINEADO desde el fix**, con el daño histórico permanente (563 filas) que sigue alimentando
el aprendizaje live (**PURP-3**).

## P18. `tests/` (682 tests)
**(P)** ¿Sigue el sistema haciendo lo que decidimos que hiciera?
**(C)** Eso, con fixtures de regresión por cada bug real encontrado — hábito excelente.
**(=) ALINEADO con un hueco de cobertura**: los tests verifican **aritmética y contratos**, no
**propósito**. Ningún test falla hoy por PURP-1 (el runline precia el evento equivocado) porque
ningún test pregunta "¿la probabilidad corresponde al lado que se está preciando?". → **P-7
(MEDIA)**: falta una familia de tests de *alineación semántica*, del tipo "la p de este mercado
debe converger a la frecuencia real de ESE evento".

## P19. `docs/` — protocolo CLV, blueprint, auditorías
**(P)** ¿Bajo qué reglas decido si esto funciona?
**(C)** Eso, con pre-registro, congelamiento y veredictos posibles definidos antes de mirar datos.
**(=) ALINEADO — es lo más maduro del proyecto.** Su límite es el que ya se detectó: la métrica
primaria sólo existe para moneyline, y el 74% de los picks no son moneyline.

---

## Hallazgos nuevos a nivel proyecto

| # | Hallazgo | Severidad |
|---|---|---|
| **P-3** | El backtest evalúa un sistema distinto del que publica: otra fuente de features, sin clima, otros pesos, otra procedencia de sesgo | **ALTA** |
| **P-4** | Los features tienen **dos implementaciones** (PIT y live) sin ningún test de paridad; ya se hallaron 2 derivas reales | **ALTA** |
| **P-1** | El proyecto declara 3 deportes; 2 (~3,000 L) nunca entraron al circuito de track record/CLV/backtest | MEDIA |
| **P-2** | Dos entradas al motor (Streamlit y cron) con distinto camino de odds y distinta persistencia | MEDIA |
| **P-5** | Dos parsers paralelos del mismo JSON de odds — deriva ya ocurrida | MEDIA |
| **P-6** | El único puente backtest→producción (`promote_calibration`) está clausurado por política | MEDIA |
| **P-7** | Los tests cubren aritmética y contratos, no alineación semántica: PURP-1 pasa todos | MEDIA |

## La raíz que une P-3, P-4, P-5 y P-2

**El proyecto tiene dos de casi todo, y ningún mecanismo que verifique que las dos copias
coinciden**: dos caminos de features (PIT / live), dos parsers de odds, dos entradas al motor, dos
UIs, dos juegos de pesos, dos procedencias de datos de aprendizaje. Cada duplicación se creó por
una razón legítima; ninguna tiene un test de paridad. Las derivas encontradas hasta hoy
(`LG_BARREL_PA`, barrel% per-PA, `g_home/g_away`, precios de Pinnacle) **no se hallaron por diseño,
se hallaron por casualidad mientras alguien buscaba otra cosa**.

Es la misma forma que la RAÍZ A del reporte (falta una capa que pregunte "¿cuánto vale esto?"),
aplicada a la ingeniería en vez de a la probabilidad: falta una capa que pregunte **"¿estas dos
cosas que deberían ser iguales, lo son?"**.
