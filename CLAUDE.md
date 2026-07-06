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

`app.py` is the Streamlit entry point. It directly defines `AppConfig`/`SportConfig` (global constants and per-sport settings), `PredictionsDB` (SQLite wrapper for `data/predictions_history.db`), the NBA/UFC analyzers and renderers, and `main()` (Streamlit page layout). MLB's analyzer and renderer (`MLBAnalyzer`, `render_mlb_results`) live in `ui/mlb.py`, imported into `app.py` via `from ui.mlb import MLBAnalyzer, render_mlb_results` — not defined directly in `app.py`.

### Sport modules

Each sport has a `run_module()` function that returns `Dict[str, Any]` with keys: `status`, `game_info`/`fight_info`, `probabilities`, `lambdas_history`, `best_bets`, `metadata`.

| Sport | Module | Location |
|-------|--------|----------|
| MLB | `run_module()` | `modules/baseball_module/core/run_module.py` |
| NBA | `run_module()` | `modules/basketball_module.py` |
| UFC | `run_module()` | `modules/ufc_module.py` |
| Soccer | `run_module()` | `modules/football_module.py` (disabled) |

### MLB pipeline (most complex)

`run_module.py` orchestrates a sequential pipeline of engines, each adjusting Poisson λ (expected runs) values. `AutoCalibrator` (`calibration/auto_calibrator.py`) no longer exists — it was superseded by the Kalman + multidim-bias system below. `context_engine/pitchers_regression.py` was also deleted (disconnected from the pipeline to avoid double-counting luck-correction with SIERA/xFIP; only a stale `.pyc` remains on disk). The real, current sequence:

1. **`MLBStatsAPI`/`MLBDataIntegrator`** (`data_fetchers.py`) — fetches game data, team stats, pitchers from MLB Stats API
2. **True Talent Engine** (`offense/true_talent_engine.py`) — computes park-neutral λ_base per team from Statcast xwOBA/barrel%/plate discipline (PIT-aware, lineup-filtered)
3. **Kalman offense + team bias + pipeline weights** (inline in `run_module.py`, backed by `calibration/learning_engine.py`) — pulls λ toward each team's observed run-scoring rate and applies a learned multidimensional bias
4. **`adjust_for_pitchers`** (`context_engine/pitcher_engine.py`, PASO 2) — adjusts λ for starter SIERA/xFIP/xERA/FIP/ERA (fallback hierarchy), recent form, matchup history, platoon splits, fatigue
5. **`adjust_for_context`** (`context_engine/contextual_engine.py`, PASO 3) — rest/back-to-back fatigue, asymmetric home/away
6. **`adjust_for_bullpen`** (`context_engine/bullpen_engine.py`, PASO 4) — bullpen quality (SIERA/ERA) and workload, tier-weighted by starter's expected innings
7. **`get_adjusted_lambdas`** (`hfa/park_weather_engine.py`, PASO 5) — park run-environment factor, weather (temp/wind/rain), retractable-roof status
8. **`adjust_for_defense`** (`context_engine/defensive_efficiency_engine.py`, PASO 6) — fielding-pure adjustment via DER (1 − BABIP) and OAA, independent of pitching
9. **`get_adjusted_lambdas`** (`hfa/hfa_engine.py`, PASO 7) — away-team travel fatigue only (home crowd boost removed, confirmed noise)
10. **`monte_carlo_advanced`** (`montecarlo/simulator.py`, PASO 8) — runs up to 5,000,000 Poisson/Negative-Binomial simulations with early stopping at `SE < 0.003`; outputs win/total/run-line probabilities
11. **`evaluate_value_ultra`** (`core/value_detector.py`, repo root — shared across sports, not under `modules/baseball_module/`, PASO 9) — Platt-2D corrects the model probability against the market before comparing to odds; calculates EV, Kelly fraction, confidence (real epistemic data-quality score, not MC sampling precision), composite score and tiers (ULTRA/HIGH/MEDIUM/SLIGHT)

### Supporting modules

- `odds_fetcher.py` — `get_odds_data()` (UI dropdown) and `get_best_odds_for_teams()` (fuzzy team name matching) both live here, using The Odds API; caches results
- `data_fetchers.py` — `MLBStatsAPI`, `MLBDataIntegrator`, weather fetcher

### Data persistence

- `data/predictions_history.db` — SQLite; predictions saved by `PredictionsDB.save()`
- `data/track_record.db` — SQLite; live pre-game picks + auto-reconciled results (`track_record/`)
- `data/mlb_complete_data.json` — cached MLB data
- `.cache/` — JSON caches for bullpen data, pitcher/team stats, odds

## Key design patterns

- **Lambda (λ) pipeline**: MLB scoring is modeled as Poisson processes. Each engine modifies λ_home and λ_away multiplicatively; the final adjusted λ values feed into Monte Carlo.
- **Fractional Kelly**: Bet sizing uses `KELLY_FRACTION=0.25` by default (quarter Kelly). `core.value_detector.kelly_criterion()` is the single source of truth — it clips the result to `[MIN_KELLY, MAX_KELLY]` = `[1%, 15%]` of bankroll (`config.py`). Every caller (pipeline `best_bets`, UI display cards, `track_record` fallback picks) delegates to this function so the cap can't drift between call sites.
- **EV calculation**: `EV = (prob × (odds - 1)) - (1 - prob)` in decimal odds. Positive EV + minimum rating threshold = actionable pick.
- **Safe module loading**: `safe_import()` in `app.py` gracefully handles missing modules; analyzers fall back gracefully when sport modules are unavailable.
- **Streamlit + terminal dual mode**: `run_module.py` detects Streamlit availability with a try/except around `import streamlit as st`; falls back to selecting the first game in terminal mode.
