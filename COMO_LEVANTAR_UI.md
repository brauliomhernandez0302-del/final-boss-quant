# Cómo levantar la UI (matchup dashboard)

Dos servidores, cada uno en su propia terminal.

## 1. Backend — Flask API (`api/`)

```bash
source mi_entorno/bin/activate && FLASK_APP=api.server FBQ_NO_PERSIST=1 flask run --port 5000
```

`FBQ_NO_PERSIST=1` fuerza `persist=False` en toda llamada a `run_module()`
sin importar el argumento — necesario porque este servidor corre el pipeline
real para cualquier juego que el picker de React pida, y una llamada de
desarrollo nunca debe escribir una fila real en `game_outcomes`.

## 2. Frontend — React (`frontend/`)

```bash
cd frontend && npm run dev
```

Sirve en `http://localhost:5173` (Vite lee `VITE_API_BASE`, default
`http://localhost:5000` — solo hace falta si el backend corre en otro
puerto/host).

## Notas

- Primer uso: `cd frontend && npm install` antes del paso 2.
- El endpoint `/api/mlb/matchup/<game_pk>` corre Monte Carlo completo
  (~5-15s por request) — no es instantáneo.
- Cero cambios al pipeline vivo: la UI solo lee (`persist=False` +
  independiente de publisher/cron/quarantine).
