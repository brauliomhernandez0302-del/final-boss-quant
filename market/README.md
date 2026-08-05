# `market/` — almacén de precios append-only

Creado el 2026-08-04. Es la capa de datos de mercado escrita desde cero,
sabiendo lo que el proyecto ya aprendió a los golpes.

**No toca ningún motor de predicción, ninguna λ, ni el ledger de picks.** Es
un almacén independiente, en su propia base (`data/market.db`), que solo
inserta.

## Por qué existe

La captura de precios que ya corría (`track_record/capture_closing_lines.py`)
pide el board doce veces al día y hace `UPDATE picks SET closing_* = ...` en
cada barrida. Se pagan doce capturas y se guarda una: las once trayectorias
intermedias se sobreescriben. Y el histórico (`historical_odds`) tiene
`game_pk UNIQUE`, o sea una sola fila por juego, tomada siempre a las 17:00Z —
no es el cierre, es una foto de la mañana.

Un precio no se puede volver a comprar. Este módulo deja de tirarlos.

## Las cinco reglas, y quién las impone

| Regla | Cómo se impone |
|---|---|
| Append-only | Dos triggers de SQLite abortan `UPDATE` y `DELETE` |
| Los dos lados, siempre | `pair_before()` devuelve `None` con medio par |
| El punto FIRMADO y pegado a su lado | Una columna `point` por fila, por lado |
| `event_id` explícito | `NOT NULL`; sin `id` del proveedor no se guarda |
| El libro es parte del dato | `book NOT NULL` |

Tests: `tests/test_market_store.py` (16).

## Uso

```bash
# Barrida del board completo (cron, a los :51 — cache hit, cero cuota extra)
python3 -m market.capture

# Identidad evento↔game_pk (cron, una vez al día)
python3 -m market.link_events

# Estado del almacén
python3 -c "from market import MarketStore; print(MarketStore().summary())"
```

Lectura desde código:

```python
from market import MarketStore
s = MarketStore()

# El par de Pinnacle vigente en un momento dado, listo para desvigorizar
par = s.pair_before(event_id, "spreads", "home", published_at, book="pinnacle")
# → {'point': -1.5, 'price_side': 2.38, 'price_opposite': 1.65, ...}

# Toda la serie observada de una cotización
s.trajectory(event_id, "totals", "over", book="pinnacle")
```

`latest_before()` y `pair_before()` exigen un corte temporal explícito y lo
aplican **estricto** (`<`, no `<=`). No existe "el precio de este evento" sin
decir a qué momento — y un `<=` sobre el corte es literalmente el leak que la
Fase 2B tuvo que remediar en el camino PIT.

## Lo que NO está hecho

- **`picks.event_id`.** El ledger no guarda a qué evento de odds corresponde
  cada pick, así que todavía no se puede reconstruir el par de precios del
  momento en que se apostó. Con este almacén ya no hacen falta cuatro columnas
  nuevas en `picks` (lado tomado, lado opuesto, punto, libro): alcanza con
  `event_id`, y `pair_before(event_id, market, side, published_at)` reconstruye
  el resto exactamente. Es el siguiente paso.
- **Migrar `closing_*` a una vista sobre este almacén.** Mientras tanto
  `capture_closing_lines.py` sigue corriendo igual que antes: nada de lo que
  hoy funciona depende de este módulo.
- **Importar `historical_odds`** (4.695 juegos, un snapshot cada uno) como
  filas de este almacén, para tener un solo lugar donde buscar precio.
- **Barridas nocturnas.** Hoy se captura 08–19h, piggyback sobre la barrida de
  cierre. Cubrir la deriva nocturna cuesta ~3 llamadas más por día; es una
  decisión de cuota, no técnica.
