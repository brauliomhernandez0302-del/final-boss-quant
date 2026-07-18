# Roadmap Step 2, Commit B — FASE B.1: enumeración de lectores cross-season en modo live

## Metodología

1. Grep de todo patrón de comparación de `season` (`season <`, `season <=`, `season !=`,
   `season -`, `season IN`) dentro de `modules/baseball_module/calibration/learning_engine.py`
   (el único archivo, confirmado en CHRON-001/CHRON-002, que contiene lógica compartida
   live/backtest de lectura de `game_outcomes`).
2. Grep de `game_outcomes` en cada archivo del camino en vivo fuera de `learning_engine.py`:
   `run_module.py`, `ui/*.py`, `track_record/*.py`, `db/*.py`, `app.py`.

## Resultado

**Un solo lector cross-season de columnas de PREDICCIÓN de `game_outcomes` en modo live:
`recalibrate_platt_2d`'s training query** (`WHERE season < ?`, línea ~1382), exactamente lo
que el prompt anticipaba.

Todos los demás matches de `season <`/`season -` encontrados en `learning_engine.py` son
lecturas cross-season de **`ml_state`** (no `game_outcomes`), y por lo tanto fuera del
alcance textual de B.1 (que pide específicamente "columnas de predicción de `game_outcomes`
de seasons PASADAS"):

| Línea | Función | Qué lee | ¿`game_outcomes`? |
|---|---|---|---|
| `get_platt_params` (`season - 1`) | `ml_state` warm-start (`platt_params`) | No — cache de calibración, no columnas de predicción |
| `get_platt_2d_params` (`for prior_season in range(season-1, season-5, -1)`) | `ml_state` warm-start (`platt2d_params`) | No — mismo caso |
| `get_pipeline_weights` (`season - 1`) | `ml_state` warm-start (`pipeline_weights`) | No — mismo caso |
| `reset_kalman_for_seasons` (`season IN (...)`) | `kalman_state` DELETE | No — tabla de estado, no `game_outcomes`, y de todas formas es backtest-only (ver CHRON-002) |

Ningún archivo fuera de `learning_engine.py` en el camino en vivo (`run_module.py`,
`ui/`, `track_record/`, `db/`, `app.py`) hace ninguna consulta directa a `game_outcomes` —
confirmado por grep, cero resultados.

## Veredicto

**Enumeración cerrada — un solo lector a tratar: `recalibrate_platt_2d`.** No se encontraron
lectores adicionales que ameriten el mismo tratamiento.
