# FINAL BOSS QUANT G8+

Sistema cuantitativo que modela partidos de MLB como procesos de Poisson,
simula el marcador por Monte Carlo y compara las probabilidades resultantes
contra las cuotas del mercado para identificar precios mal puestos.

> Proyecto de investigación personal. Este README documenta cómo levantarlo y
> cómo correr sus pruebas; nada de lo que hay aquí es asesoramiento financiero.

## Requisitos

- Python 3.12
- Node 22 (sólo para el tablero React)
- Una clave de [The Odds API](https://the-odds-api.com)

Las demás fuentes de datos —MLB Stats API, Baseball Savant y FanGraphs— son
gratuitas y no piden autenticación.

## Instalación

```bash
python3 -m venv mi_entorno
source mi_entorno/bin/activate
pip install -r requirements.txt
```

Para desarrollo y pruebas, además:

```bash
pip install -r requirements-dev.txt
```

## Configuración

```bash
cp .env.example .env
```

Y rellenar:

| Variable | Necesaria | Para qué |
|---|---|---|
| `ODDS_API_KEY` | sí | Cuotas de moneyline, total y runline. Sin ella el pipeline calcula probabilidades pero no tiene contra qué compararlas. |
| `OPENWEATHER_API_KEY` | no | Temperatura, viento y lluvia por estadio. Si falta, el ajuste de parque corre con condiciones neutras y lo registra como tal. |

`.env.example` documenta el resto de variables opcionales. `.env` está
ignorado por git y no debe versionarse nunca.

## Uso

### Aplicación Streamlit

```bash
source mi_entorno/bin/activate
streamlit run app.py
```

### Tablero React

Backend Flask y frontend Vite, cada uno en su terminal. Instrucciones
completas en [`COMO_LEVANTAR_UI.md`](COMO_LEVANTAR_UI.md).

### Publicación automática

`run_daily_picks.py` es el punto de entrada del cron. La configuración en uso
está versionada en [`ops/crontab_20260726.txt`](ops/crontab_20260726.txt):
dos corridas diarias del pipeline, dos chequeos de salud y una barrida horaria
para capturar las líneas de cierre.

## Pruebas

```bash
source mi_entorno/bin/activate
python -m pytest -q
```

La suite corre sin bases de datos ni claves de API. Los tests que necesitan un
estado que sólo existe tras un backtest completo se marcan como `skipped`, no
como fallo.

Frontend:

```bash
cd frontend
npm ci
npm run build
npm test
```

## Datos

**Las bases de datos no vienen en el repositorio.** `data/*.db` está ignorado:
contienen predicciones, resultados reconciliados y estado de calibración
acumulados en local. Un clon limpio arranca sin ellas y las va construyendo.

Los cachés en disco (`.cache/`) se regeneran solos.

## Estructura

| Ruta | Qué hay |
|---|---|
| `app.py`, `ui/` | Aplicación Streamlit |
| `api/`, `frontend/` | Backend Flask y tablero React |
| `modules/baseball_module/` | El pipeline de MLB: motores de λ, Monte Carlo, calibración |
| `core/value_detector.py` | Detección de valor, EV y Kelly (compartido entre deportes) |
| `data_fetchers.py`, `odds_fetcher.py` | Capa de acceso a las APIs externas |
| `track_record/` | Publicación de picks, reconciliación y captura de cierre |
| `backtest_and_retrain.py` | Backtest walk-forward y reentrenamiento |
| `docs/`, `CONTRACTS.md` | Documentación técnica y contratos entre componentes |

`CLAUDE.md` lleva el registro cronológico de decisiones de ingeniería y del
estado de las auditorías.
