# `fbq/` — el sistema, en orden de dependencias

Cada paquete es un paso del build y **sólo puede depender de los anteriores**.
Esa regla es la arquitectura: si `features/` necesitara algo de `model/`, el
orden está mal, y el error aparece como un import circular en vez de como una
sorpresa seis meses después.

| | Paquete | Qué establece | Estado |
|---|---|---|---|
| 0 | `core/` | El contrato de tiempo e identidad | ✅ |
| — | `sources/` | Los fetchers: hablan con el exterior, devuelven crudo | parcial (MLB) |
| 1 | `market/` | Precios, append-only, con trayectoria | ✅ |
| 2 | `results/` | Los hechos: quién ganó | ✅ |
| 3-4 | `evaluator/` | La balanza, y v0 = el mercado | ✅ |
| 5-6 | `features/` | Señales con contrato as-of | vacío |
| 7-8 | `model/` | Probabilidad y detección de valor | vacío |
| 9 | `stake/` | Sizing | vacío |
| 10-11 | `ledger/` | Publicación y reconciliación | vacío |
| 12 | `app/` | La UI | vacío |

## Por qué `sources/` va aparte y no tiene número

No es un paso: es la frontera. Vive separado porque **traer el dato,
persistirlo y derivarlo son tres trabajos distintos**, y mezclarlos produjo dos
fallos concretos en el sistema anterior:

- Normalizar antes de guardar pierde lo que la normalización no contempló. Los
  precios de Pinnacle para total y runline llegaban en cada respuesta desde
  siempre y se descartaban en el parseo; cuando hicieron falta, no existía
  histórico.
- Leer del dato derivado en vez del crudo hace posible leer una clave que nadie
  emite. La telemetría de clima registró `None` en 52 de 52 picks porque leía
  `game_data['weather']['source']`, que el fetcher no devuelve — mientras el
  motor sí recibía el clima y sí movía λ.

## Las reglas que impone el motor, no la disciplina

Cada almacén rechaza por esquema lo que no debe entrar:

- `market.odds_snapshot` y `results.observacion` son **append-only por trigger**
  de SQLite. Un `UPDATE` aborta.
- `results.Final` no se puede **construir** con un estado que no sea final ni
  con un marcador empatado. El fetcher no tiene ninguna condición sutil que
  revisar.
- `market.pair_before()` devuelve `None` con medio par o con puntos que no
  coinciden. No hay forma de desvigorizar dos lados de mercados distintos.
- `core.elegir_unico()` **se abstiene** ante dos candidatos casi igual de cerca.

## Correr

```bash
python3 -m fbq.market.capture        # precios (cron :51, cero cuota extra)
python3 -m fbq.market.link_events    # identidad evento↔juego (cron 20:05)
python3 -m fbq.results.fetch         # hechos (cron 23:30)
python3 -m fbq.evaluator --seasons 2024 2025 --candidato mercado
```
