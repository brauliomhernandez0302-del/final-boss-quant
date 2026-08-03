# Nota — los 46 rows de prueba en `game_outcomes` (source='live')

**Fecha**: 2026-07-19. Contexto: el sweep de verificación (`reporte.md`, V1) encontró que 46
filas con `game_date` entre 2026-07-18 y 2026-07-21 fueron escritas hoy por mis propias pruebas
de diagnóstico y dry-runs de `run_daily_picks.py`, correlacionadas exactamente por `created_at`:

```
824414  2026-07-18  created_at=2026-07-19 01:25:32
...  (30 filas más, 01:25:32–01:31:36 — pruebas de diagnóstico + primeros dry-runs)
824410  2026-07-20  created_at=2026-07-19 14:08:54
...  (14 filas más, 14:08:54–14:09:34 — último dry-run antes de la interrupción)
```

## Decisión: SE QUEDAN

Estas 46 filas **no se borran ni se marcan de otra forma**. Razón:

1. **Son predicciones pre-juego legítimas del modelo comprometido** — `run_module()` corrió el
   pipeline real de punta a punta (sin ningún mock), con la key de odds activa, para juegos
   reales del schedule real, con el código de producción tal como estaba en cada momento. No son
   datos sintéticos ni basura — son exactamente lo que el modelo hubiera predicho si esas
   llamadas hubieran sido intencionales.
2. **Los cambios de código sin commitear en esos momentos eran de display/etiquetado, no de
   probabilidad**: el fix de formato de EV (`:.2%` → `:.2f}%`) y el fix de `game_date` mislabel
   en `track_record/publisher.py` — ninguno de los dos toca `p_home`/`p_away`/`lambda_home`/
   `lambda_away`, que es lo único que `record_prediction()` nunca sobreescribe una vez insertado
   (ver el docstring actualizado en `learning_engine.py`). Las probabilidades registradas en
   estas 46 filas son las probabilidades reales que el pipeline de esos momentos calculó.
3. **Borrarlas sería peor que dejarlas**: crearía un hueco de fechas sin explicación en
   `game_outcomes`, y el mecanismo real (falta de modo de prueba en `run_module()`) ya quedó
   arreglado en este mismo commit (`persist=False`) — no se va a repetir.

## Lo que sí cambia a partir de ahora

Con `persist=False` (o la variable de entorno `FBQ_NO_PERSIST=1`) disponible desde este commit,
cualquier prueba futura de `run_module()` debe usar ese flag explícitamente. Las 46 filas de hoy
son el último caso de esta clase de contaminación accidental.
