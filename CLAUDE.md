# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**FINAL BOSS QUANT G8+** — a quantitative sports betting prediction system. It analyzes MLB, NBA, and UFC games and identifies positive-EV betting opportunities using Monte Carlo simulation, Poisson modeling, Kelly criterion sizing, and odds API integration.

## Running the app

```bash
# Activate the virtual environment first
source mi_entorno/bin/activate

# Launch the Streamlit app
streamlit run app.py
```

## Environment setup

Create a `.env` file in the project root:

```
ODDS_API_KEY=your_key_from_the_odds_api
OPENWEATHER_API_KEY=your_openweather_key   # optional, for weather adjustments
```

- **MLB Stats API** (`statsapi.mlb.com/api/v1`): free, no key needed
- **The Odds API**: required for live odds; loaded via `odds_fetcher.py`
- **OpenWeather**: optional; used in `data_fetchers.py` for park weather factors

## Architecture

### Entry point

`app.py` is the single Streamlit entry point. It defines:
- `AppConfig` / `SportConfig` — global constants and per-sport settings
- `PredictionsDB` — SQLite wrapper (`data/predictions_history.db`)
- `BaseAnalyzer` / `MLBAnalyzer` / `NBAAnalyzer` / `UFCAnalyzer` — analyzer classes that call the sport modules
- UI rendering functions (`render_mlb_results`, `render_nba_results`, `render_ufc_results`)
- `main()` — Streamlit page layout

### Sport modules

Each sport has a `run_module()` function that returns `Dict[str, Any]` with keys: `status`, `game_info`/`fight_info`, `probabilities`, `lambdas_history`, `best_bets`, `metadata`.

| Sport | Module | Location |
|-------|--------|----------|
| MLB | `run_module()` | `modules/baseball_module/core/run_module.py` |
| NBA | `run_module()` | `modules/basketball_module.py` |
| UFC | `run_module()` | `modules/ufc_module.py` |
| Soccer | `run_module()` | `modules/football_module.py` (disabled) |

### MLB pipeline (most complex)

`run_module.py` orchestrates a sequential pipeline of engines, each adjusting Poisson λ (expected runs) values:

1. **`MLBStatsAPI`** (`data_fetchers.py`) — fetches game data, team stats, pitchers from MLB Stats API
2. **`AutoCalibrator`** (`calibration/auto_calibrator.py`) — adjusts λ based on 30-game historical performance, home/away splits, recent streak, season phase
3. **`get_adjusted_lambdas`** (`hfa/hfa_engine.py`) — applies Home Field Advantage using `StadiumFactors` (park factors, altitude, weather)
4. **`adjust_for_pitchers`** (`context_engine/pitcher_engine.py`) — adjusts λ for pitcher ERA/FIP/WHIP, recent form, bullpen workload, days rest
5. **`calculate_pitcher_regression`** (`context_engine/pitchers_regression.py`) — regresses lucky/unlucky ERA using BABIP, LOB%, HR/FB%
6. **`monte_carlo_advanced`** (`montecarlo/simulator.py`) — runs up to 5,000,000 Poisson simulations with early stopping at `SE < 0.003`; outputs win/total/run-line probabilities
7. **`evaluate_value_ultra`** (`value/value_detector.py`) — compares model probabilities to market odds; calculates EV, Kelly fraction, confidence tiers (ULTRA/HIGH/MEDIUM/SLIGHT)

### Supporting modules

- `bankroll.py` — `kelly_stake()` function with fractional Kelly and max-risk caps
- `storage.py` — `save_bets()` / `load_bets()` for CSV bet log (`bets_log.csv`)
- `odds_fetcher.py` — `get_odds_data()` using The Odds API; caches results
- `odds_api.py` — `get_best_odds_for_teams()` with fuzzy team name matching
- `data_fetchers.py` — `MLBStatsAPI`, `MLBDataIntegrator`, weather fetcher

### Data persistence

- `data/predictions_history.db` — SQLite; predictions saved by `PredictionsDB.save()`
- `data/bets_log.csv` — CSV bet history via `storage.py`
- `data/mlb_complete_data.json` — cached MLB data
- `.cache/` — JSON caches for bullpen data, H2H stats, odds

## Key design patterns

- **Lambda (λ) pipeline**: MLB scoring is modeled as Poisson processes. Each engine modifies λ_home and λ_away multiplicatively; the final adjusted λ values feed into Monte Carlo.
- **Fractional Kelly**: Bet sizing uses `kelly_fraction=0.25` by default (quarter Kelly). Max bet is capped at `max_risk_pct=5%` of bankroll.
- **EV calculation**: `EV = (prob × (odds - 1)) - (1 - prob)` in decimal odds. Positive EV + minimum rating threshold = actionable pick.
- **Safe module loading**: `safe_import()` in `app.py` gracefully handles missing modules; analyzers fall back gracefully when sport modules are unavailable.
- **Streamlit + terminal dual mode**: `run_module.py` detects Streamlit availability with a try/except around `import streamlit as st`; falls back to selecting the first game in terminal mode.
