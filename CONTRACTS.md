# CONTRACTS.md — System Map: Final Boss Quant G8+

**Generated:** 2026-05-13
**Rewritten:** 2026-07-06 — full verification against current code, following the 2026-07-06 audit (`docs/AUDITORIA_MLB_2026-07.md`). The 2026-05-13 version had drifted substantially: it cited files that no longer exist (`odds_api.py`, `storage.py`, `ablation_calibrator.py`, `post_game.py`, `train_historical.py`), described a component (`AutoCalibrator`) that was removed and superseded, and predated the entire point-in-time (PIT) rebuild, the `ui/` package split, and 5 pipeline engines that didn't exist yet. Every claim below was verified directly against the code on 2026-07-06.
**Purpose:** North-star reference — what every file does, what is alive vs dead, and the correct data flow for every sport.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [File Inventory](#2-file-inventory)
3. [Active vs Dead Code](#3-active-vs-dead-code)
4. [MLB Data Flow (canonical)](#4-mlb-data-flow)
5. [NBA Data Flow](#5-nba-data-flow)
6. [UFC Data Flow](#6-ufc-data-flow)
7. [Connected vs Orphaned Files](#7-connected-vs-orphaned-files)
8. [Database Schema](#8-database-schema)
9. [Import Graph](#9-import-graph)

---

## 1. System Overview

**Final Boss Quant G8+** is a Streamlit sports-betting prediction system.
Entry point: `app.py`. Run: `streamlit run app.py`.

The system models each sport as a probability estimator and compares those probabilities to market odds to find positive-EV bets. MLB is the primary sport with a full quantitative pipeline (9 sequential lambda-adjustment engines + point-in-time data infrastructure). NBA and UFC have self-contained modules but lack live data integrations (run on demo/passed-in data only). Soccer was disabled and later removed entirely.

**Three runtime modes:**

| Mode | Entry Point | Purpose |
|------|------------|---------|
| Live UI | `app.py` (imports `ui.mlb`, `ui.odds_loader`, `ui.sidebar`, `ui.components`) | Streamlit dashboard, one-game analysis + Track Record tab |
| Daily CLI | `run_daily_picks.py` | Publish picks to `track_record.db` + reconcile results |
| Dev/Research | `backtest_and_retrain.py`, `analyze_game_outcomes.py`, `fetch_historical_odds.py` | Offline model validation and historical data bootstrap |

**No longer part of this repo (verified removed):** `betting_ai/` and `bbets_ia_pro/` directories, `injuries_fetcher.py`, `nba_stats_fetcher.py`, `ufc_data_fetcher.py`, `modules/football_module.py`, `modules/boxing_module.py`, `ablation_calibrator.py`, `post_game.py`, `train_historical.py`, `odds_api.py`, `storage.py`, `modules/baseball_module/calibration/auto_calibrator.py`, `modules/baseball_module/context_engine/pitchers_regression.py` (only a stale `.pyc` remains). None of these should be referenced as live components going forward.

---

## 2. File Inventory

### Root-level Python files

| File | Lines | What it does |
|------|-------|-------------|
| `app.py` | 305 | **Streamlit entry point** — page config, `SPORT_CONFIGS`, NBA/UFC stubs (`_NBAAnalyzer`/`_UFCAnalyzer`), `_render_analysis_tab()`, `main()`. MLB's analyzer/renderer are NOT here — imported from `ui.mlb` (`from ui.mlb import MLBAnalyzer, render_mlb_results`). |
| `config.py` | 74 | **Single source of truth for constants.** `MLB_SIMULATIONS=5_000_000`, `KELLY_FRACTION=0.25`, `MIN_KELLY=0.01`/`MAX_KELLY=0.15`, `LEAGUE_AVG_RUNS=4.5`, `LEAGUE_AVG_ERA=4.15`, `LEAGUE_AVG_XWOBA=0.316`, `PITCHER_ENGINE_WEIGHTS` dict, `DEFAULT_MIN_EV=3.0`. No `MAX_RISK_PCT` (does not exist, despite prior versions of this doc citing it). |
| `data_fetchers.py` | 2,182 | **MLB data layer.** `MLBStatsAPI` (hits `statsapi.mlb.com`, free, no key): games, pitcher/team stats, bullpen ERA+workload, travel, roof status, batting handedness, Savant OAA. `MLBDataIntegrator` orchestrates enrichment. `WeatherAPI` (OpenWeather, optional). 5 dead fetchers removed 2026-07-06 (`get_head_to_head`, `get_standings_status`, `get_team_offensive_stats`, `get_team_recent_form`, `get_umpire_historical_stats` — zero downstream consumers, confirmed via grep). |
| `odds_fetcher.py` | 521 | **Odds API layer — both odds functions live here now.** `get_odds_data()` (UI dropdown, via `_normalize_event()`) and `get_best_odds_for_teams(home, away, sport)` (fuzzy-match lookup used by `run_module.py`'s auto-fetch path and `track_record/publisher.py`) — there is no separate `odds_api.py`. F5 field naming fixed 2026-07-06 (`f5_ml_home`/`f5_ml_away`/`f5_total_over`/`f5_total_under`, matching `GameOdds`' convention — previously `f5_home`/`f5_over` etc., a mismatch that silently meant F5 markets never activated via this path). `_normalize_event()` (feeds `get_odds_data()`) still uses a third, distinct naming scheme (`f5_home_odds`/`f5_over_odds`) — flagged inline as a land mine for any future refactor connecting it more directly to F5 analysis, not yet an active bug. |
| `run_daily_picks.py` | 140 | **Daily CLI runner.** Calls `track_record.publisher.publish_daily_picks()` then `track_record.reconciler.reconcile_all_sports()`, prints summary. Designed for cron. |
| `fetch_historical_odds.py` | 581 | **One-time historical odds downloader.** Pulls historical Pinnacle moneylines via The Odds API's historical endpoint. Costs quota; per project memory, has previously hit quota limits mid-run. |
| `backtest_and_retrain.py` | 3,178 | **Full pipeline backtest + walk-forward retrain.** Runs every game in `game_outcomes` through the complete live pipeline (not a separate reimplementation — imports and calls the same engines as `run_module.py`), refits Platt/Platt-2D, relearns gradient-descent pipeline weights. Supports 4 independent PIT modes (`--experimental-pitcher-pit-mode`, `--use-team-tte-pit`, `--use-defense-pit`, `--use-bullpen-pit`, or `--use-full-pit` for all 4) so backtest evaluation reads point-in-time-correct data instead of season aggregates. `_team_dict()`'s `use_team_full_season_pitching_base` flag (gated to `--use-defense-pit`) makes `team_era`/`team_whip`/`runs_allowed_per_game` fall back to league averages in PIT mode rather than leaking full-season data — confirmed present and correctly wired 2026-07-06 (a prior draft of this doc's audit had this flagged as an open leak; it is not). Writes model λ/p back to `game_outcomes`, produces a JSON report in `data/`. |
| `analyze_game_outcomes.py` | 558 | **Analytics script.** Reads `game_outcomes`, analyzes ROI by stadium/month/day-night, drawdown curves. Standalone, read-only against the pipeline. |

---

### `core/` — Shared utilities

| File | Lines | What it does |
|------|-------|-------------|
| `core/utils.py` | ~10 | `calculate_ev(prob, decimal_odds)` — single source of truth for the EV formula, imported by `basketball_module.py` and `value_detector.py`. |
| `core/value_detector.py` | 1,119 | **Multi-market value detector, sport-agnostic.** `evaluate_value_ultra()` — vig removal (3 methods: multiplicative/power/Shin), EV/edge/Kelly per market, `calculate_composite_score()`, `classify_value_tier()` (ULTRA/HIGH/MEDIUM/SLIGHT), Pinnacle as fair-line reference, optional Platt-2D correction via injected `p_home_corrector` callable. Markets: ML, totals, run line (±1.5), F5 (moneyline + totals). `all_opportunities` (source of `best_bets`) now spreads the full underlying bet dict (2026-07-06 fix — two independent downstream consumers each needed a field the old hand-picked projection didn't carry) except `tier_enum` (a raw Enum, not JSON-serializable). `GameOdds` dataclass defines the canonical field-naming convention (`f5_ml_home`, `f5_total_over`, etc.) that other files must match. |

---

### `modules/baseball_module/` — MLB pipeline (9-step lambda engine + PIT data layer)

| File | Lines | What it does |
|------|-------|-------------|
| `core/run_module.py` | 878 | **MLB pipeline orchestrator.** `run_module(game_id, ...)` drives the full PASO 0-9 sequence (§4). Applies gradient-descent pipeline weights (`LearningEngine.get_pipeline_weights()`) via a delta-weighted blend at every stage: `λ_out = λ_in × (1 + w×(raw_ratio−1))` — confirmed 2026-07-06 this is the actual combination method everywhere, not geometric mean. Stamps `lambdas_history['final']` (the exact λ fed to Monte Carlo) since 2026-07-06, so consumers don't have to guess "last stage present." |
| `offense/true_talent_engine.py` | 799 | **True Talent Engine (TTE) — offense, pre-pipeline.** Park-neutral λ_base from Statcast xwOBA/barrel%/plate discipline, lineup-filtered when a confirmed lineup exists, blended with a prior-season baseline (`prior_w = 1000/(1000+PA_current)`). PIT coverage ~99.4%. |
| `calibration/learning_engine.py` | 1,081 | **Adaptive ML engine, 6 mechanisms:** (1) team bias (mean actual/predicted λ per team/season/context, incl. multi-dim by home-away/month), (2) Kalman filter (hidden team run-rate state), (3) Platt recalibration (1D, weekly refit), (4) Platt-2D (`logit(p_corrected)=a+b·logit(p_home)+c·logit(market_prob)`, expanding-window, shipped 2026-07-05), (5) gradient-descent pipeline weights, (6) idempotent `update_outcome()`. **`auto_calibrator.py` does NOT exist** — the multiplicative-cap calibrator it used to describe was fully superseded by this file. |
| `context_engine/pitcher_engine.py` | 412 | **Pitcher Engine (PASO 2).** Fallback hierarchy SIERA→xFIP→xERA→FIP→raw ERA, branched TBF-based Bayesian shrinkage per estimator, conditional `kbb_mult` (only applied when the winning estimator doesn't already encode K%-BB%, avoiding double-counting), platoon splits, matchup history, fatigue. Quality clamp `[0.60,1.45]` (widened 2026-07-05 from real unclamped distribution). |
| `context_engine/contextual_engine.py` | 152 | **Contextual Engine (PASO 3).** Rest/back-to-back only, asymmetric (home B2B neutralized per audit finding, away B2B −4%). No umpire factor (removed, never had real data). |
| `context_engine/bullpen_engine.py` | 645 | **Bullpen Engine (PASO 4).** Composite of xwOBA/SIERA-or-ERA/K-BB%/barrel%, conditionally reweighted when team SIERA is available (drops K-BB weight to avoid double-counting, mirrors Pitcher Engine's `kbb_mult` fix). Workload/fatigue factor from real per-team boxscore innings (`ip_last_3_days`). PIT coverage 100%. |
| `hfa/park_weather_engine.py` | 484 | **Park + Weather Engine (PASO 5).** Symmetric park run-factor (static, 5-year FanGraphs-sourced `STADIUM_DATABASE`, no L/R split), temperature/wind/rain, retractable-roof handling. `roof_open`/`roof_closed` were never populated by any fetcher until 2026-07-06 (forced `roof_closed=True` unconditionally for all 8 retractable-roof parks) — fixed via `data_fetchers.py::get_roof_status()` reading the live game feed. Includes 2026 stadium renames (Daikin Park, UNIQLO Field at Dodger Stadium). Deliberately blind in the backtest (`historical_weather.py` instantiation is commented out — documented, intentional). |
| `context_engine/defensive_efficiency_engine.py` | 308 | **Defensive Efficiency Engine (PASO 6).** Fielding-only, isolated from pitching: DER (`1 − BABIP`, Bayesian-shrunk toward `_LG_DER=0.7097`) + OAA (Savant, now divided by an estimated real games-played denominator instead of a fixed 162, plus new Bayesian shrinkage — both fixed 2026-07-05/06). PIT coverage 100%. |
| `hfa/hfa_engine.py` | 145 | **HFA Engine (PASO 7).** Home crowd boost is permanently disabled (`hfa_boost=0.0`, confirmed noise — Pearson=-0.015). Away-team travel fatigue only (miles + timezone-crossing penalty, dynamic per-game, max −0.10 runs / ~2.2% λ). |
| `montecarlo/simulator.py` | 368 | **Monte Carlo engine.** `monte_carlo_advanced(lh, la, n_max=5_000_000)` — Negative Binomial (`NB_DISPERSION=6.0`, not Poisson), vectorized blocks, early stopping at `SE<0.003` (checked on `p_home` only — O/U and F5 ride on whatever `n` that produces, a known, low-priority, bounded gap). Bivariate correlation (`rho_game`) for run totals. `F5_SCALE=0.575` empirically validated against 60 real 2025 games. Rated the best-implemented file in the pipeline across multiple audits. |
| `advanced_pit_enrichment/` | 26 files | **Point-in-time data infrastructure**, distinct from the lambda-adjustment engines above — snapshot builders, daily aggregators, and prior-baseline builders that feed TTE/Bullpen/Defense/Pitcher with data "as of" a given date rather than season aggregates. Built incrementally across recent sessions; has never had a formal file-by-file audit despite being touched extensively (flagged in `docs/AUDITORIA_MLB_2026-07.md`). |
| `data_enrichment/savant_fetcher.py`, `fangraphs_fetcher.py` | ~120/~250 | Baseball Savant / FanGraphs enrichment (xERA, xwOBA, xFIP, SIERA, batter stats). 24h disk cache, graceful failure if unreachable. |

---

### `ui/` — Streamlit UI layer (did not exist in the 2026-05-13 version of this doc)

| File | Lines | What it does |
|------|-------|-------------|
| `ui/mlb.py` | 355 | **`MLBAnalyzer` + `render_mlb_results()`** — the actual MLB analysis/rendering logic, imported into `app.py`. `save_value_picks()` persists positive-EV picks to `PredictionsDB`. `market_odds` construction now includes F5 fields (fixed 2026-07-06 — previously the UI-selector-driven path had no way to carry F5 odds at all, a separate bug from the `odds_fetcher.py` naming mismatch). Displayed "λ Final" now reads `lambdas_history['final']` (fixed 2026-07-06 — previously read the wrong pipeline stage, post-Contextual instead of post-HFA). |
| `ui/components.py` | 277 | `SportConfig`, `ThemeColors`, `UIComponents` (render helpers), `calculate_ev`/`calculate_kelly` (delegates to `core.value_detector.kelly_criterion` — single source of truth, no local reimplementation). |
| `ui/sidebar.py` | 121 | `render_sidebar()` (settings: min_ev, kelly_factor, min_rating), `render_history()`. |
| `ui/odds_loader.py` | 177 | `load_odds_data()`, `filter_odds_by_sport()`, `build_game_selector()` — builds `GameData` objects for the Streamlit dropdown. Now maps F5 fields from `_normalize_event()`'s naming scheme into the canonical `GameData` convention (fixed 2026-07-06). |

---

### `db/` — Shared data models (did not exist in the 2026-05-13 version of this doc)

| File | Lines | What it does |
|------|-------|-------------|
| `db/predictions_db.py` | 232 | `GameData` (TypedDict — now includes `f5_ml_home`/`f5_ml_away`/`f5_total_line`/`f5_total_over`/`f5_total_under`, added 2026-07-06), `PredictionData`, `AnalysisResult`, `PredictionsDB` (SQLite wrapper for `data/predictions_history.db`). |

---

### `track_record/` — Live pick tracking

| File | Lines | What it does |
|------|-------|-------------|
| `track_record/db.py` | 360 | `TrackRecordDB` — SQLite CRUD for `data/track_record.db`. Tables: `picks`, `bankroll`, `daily_snapshots`. `resolve_pick()`/`upsert_daily_snapshot()` bankroll running-total fixed 2026-07-06 (was `MAX(running_total)`, corrupts the equity curve after any loss — now latest-by-insertion-order). |
| `track_record/publisher.py` | 343 | `publish_daily_picks()`/`publish_mlb_picks()` — runs the pipeline for today's games, saves qualifying picks pre-game. `_market_label()`'s F5 substring-collision fixed 2026-07-06 (F5 totals were colliding with full-game totals; F5 moneyline always resolved VOID). `mc_probs` now strips raw ndarray MC samples before `json.dumps()` (was an unconditional crash on the first real qualifying bet, 2026-07-06). |
| `track_record/reconciler.py` | 284 | `reconcile_pending()`/`reconcile_all_sports()` — fetches final scores via `statsapi` (fallback: `MLBStatsAPI`), resolves WIN/LOSS/PUSH/VOID per market. |
| `track_record/stats.py` | 184 | `compute_stats()` — headline (W-L, win rate, ROI, Sharpe; computed directly from `picks`, independent of the bankroll bug above), by-sport/market/tier/month, bankroll curve. |
| `track_record/ui.py` | 218 | `render_track_record()` — Streamlit page, "Track Record" tab in `app.py`. |
| `track_record/__init__.py` | 13 | Exports. |

**Status as of 2026-07-06**: all known bugs fixed and tested; `picks` table is empty (0 rows) — no real pick has completed the full publish→resolve cycle yet under the corrected code.

---

### `tests/` — Test suite

**39 test files, 448 tests total** (was "184+" as of the 2026-05-13 doc — grown substantially with the PIT infrastructure buildout). Notable groupings: PIT/Savant/FanGraphs snapshot+cache tests (~20 files), engine-specific tests (`test_pitcher_engine.py`, `test_defense_multiplier.py`, `test_hfa_pipeline.py`, `test_montecarlo.py`, `test_f5_lambda.py`), `test_learning_engine.py`, `test_validated_fixes.py` (regression tests for previously-fixed bugs), and 2 new files from the 2026-07-06 F5 fix (`test_odds_fetcher_f5_keys.py`, `test_ui_f5_odds_pipeline.py` — verify F5 activates end-to-end from synthetic payloads, no live `ODDS_API_KEY` needed).

---

## 3. Active vs Dead Code

### ACTIVE — Connected to the live MLB pipeline

```
app.py                                              ← Streamlit entry point
config.py                                           ← imported by nearly everyone
data_fetchers.py                                    ← MLB data source
odds_fetcher.py                                     ← both get_odds_data() and get_best_odds_for_teams() live here
core/utils.py                                       ← basketball_module + value_detector
core/value_detector.py                              ← PASO 9 of the MLB pipeline, sport-agnostic
db/predictions_db.py                                ← GameData/PredictionData/PredictionsDB
ui/mlb.py, ui/components.py, ui/sidebar.py, ui/odds_loader.py

modules/baseball_module/core/run_module.py          ← MLB pipeline entry
modules/baseball_module/offense/true_talent_engine.py
modules/baseball_module/calibration/learning_engine.py
modules/baseball_module/context_engine/pitcher_engine.py
modules/baseball_module/context_engine/contextual_engine.py
modules/baseball_module/context_engine/bullpen_engine.py
modules/baseball_module/hfa/park_weather_engine.py
modules/baseball_module/context_engine/defensive_efficiency_engine.py
modules/baseball_module/hfa/hfa_engine.py
modules/baseball_module/montecarlo/simulator.py
modules/baseball_module/advanced_pit_enrichment/ (26 files)
modules/baseball_module/data_enrichment/savant_fetcher.py       (optional, graceful fail)
modules/baseball_module/data_enrichment/fangraphs_fetcher.py    (optional, graceful fail)

modules/basketball_module.py                        ← self-contained, demo-data fallback
modules/ufc_module.py                                ← self-contained, demo-data fallback

track_record/db.py, publisher.py, reconciler.py, stats.py, ui.py
run_daily_picks.py
tests/ (39 files, 448 tests)
```

### ACTIVE — Dev/Research tools (not part of the live pipeline)

```
backtest_and_retrain.py          ← model validation + walk-forward retrain, imports the live engines
analyze_game_outcomes.py         ← analytics on historical data, read-only
fetch_historical_odds.py         ← historical odds download, quota-limited
```

### REMOVED since the 2026-05-13 version of this document (confirmed absent, do not reference as live)

```
odds_api.py, storage.py, ablation_calibrator.py, post_game.py, train_historical.py
injuries_fetcher.py, nba_stats_fetcher.py, ufc_data_fetcher.py
modules/football_module.py, modules/boxing_module.py
modules/baseball_module/calibration/auto_calibrator.py
modules/baseball_module/context_engine/pitchers_regression.py (only a stale .pyc remains)
betting_ai/ (entire directory), bbets_ia_pro/ (entire directory)
```

---

## 4. MLB Data Flow (canonical)

Full chain from "user clicks Analyze" to "best bet displayed," verified against `run_module.py` line numbers on 2026-07-06:

```
[Streamlit UI — app.py → ui/*]
│
├── render_sidebar() [ui/sidebar.py] → settings: kelly_factor, min_ev, min_rating
├── load_odds_data() [ui/odds_loader.py] → odds_fetcher.get_odds_data()
├── filter_odds_by_sport(), build_game_selector() → GameData (incl. F5 fields since 2026-07-06)
│
└── [User clicks "Analizar Evento (MLB)"]
    │
    ▼
[MLBAnalyzer.analyze() — ui/mlb.py]
│
├── find_game_id(game_data) — fuzzy-match via MLBDataIntegrator
│
└── run_module(game_id, market_odds=..., ...) [modules/baseball_module/core/run_module.py]
    │
    ├── PASO 0: MLBDataIntegrator.get_complete_game_data() — pitcher/team/bullpen stats,
    │           travel, roof status, lineup handedness, Savant/FanGraphs enrichment
    │
    ├── True Talent Engine (offense) — park-neutral λ_base per team (Statcast, PIT-aware,
    │   lineup-filtered when a confirmed lineup exists)
    │
    ├── Kalman + multidim team bias [calibration/learning_engine.py] — pulls λ toward each
    │   team's observed run-scoring rate, applies learned multidimensional bias
    │
    ├── PASO 2: Pitcher Engine — SIERA/xFIP/xERA/FIP/ERA fallback hierarchy, form, matchup,
    │           platoon, fatigue. Quality clamp [0.60, 1.45].
    │
    ├── PASO 3: Contextual Engine — rest/B2B only, asymmetric
    │
    ├── PASO 4: Bullpen Engine — quality composite + workload/fatigue
    │
    ├── PASO 5: Park + Weather Engine — park run-factor, weather, retractable roof
    │
    ├── PASO 6: Defensive Efficiency Engine — DER + OAA, fielding-only
    │
    ├── PASO 7: HFA Engine — away-team travel fatigue only (crowd boost disabled)
    │
    │   [λ clamp: 1.5 ≤ lh, la ≤ 12.0 — sanity guard, confirmed rarely/never binds]
    │   lambdas_history['final'] stamped here (2026-07-06) — the exact λ fed to Monte Carlo
    │
    ├── PASO 8: Monte Carlo [montecarlo/simulator.py] — Negative Binomial(NB_DISPERSION=6.0),
    │           early stopping SE<0.003 on p_home, F5 via F5_SCALE=0.575 (empirically validated)
    │
    └── PASO 9: Value Detection [core/value_detector.py::evaluate_value_ultra]
        ├── Platt-2D correction of p_home against the market (if fit available)
        ├── Vig removal (multiplicative/power/Shin), EV/edge/Kelly per market
        ├── calculate_composite_score() + classify_value_tier() (ULTRA/HIGH/MEDIUM/SLIGHT,
        │   percentile-refit 2026-07-06 against the real post-Platt-2D distribution)
        ├── Markets: ML, totals, run line (±1.5), F5 (moneyline + totals)
        └── RETURN: {status, game_info, probabilities, lambdas_history, best_bets, metadata}

[Back in ui/mlb.py]
├── render_mlb_results() — λ progression (reads lambdas_history['final']), probabilities,
│   value cards, best_bets by tier
└── save_value_picks() → PredictionsDB.save() → predictions_history.db::predictions
```

**Separately, `run_daily_picks.py` → `track_record/publisher.py::publish_mlb_picks()`** calls this same `run_module()` (without an explicit `market_odds` override, relying on `run_module`'s own internal `get_best_odds_for_teams()` fetch) to publish pre-game picks to `track_record.db`, then `track_record/reconciler.py` resolves them post-game via real final scores.

---

## 5. NBA Data Flow

Unchanged from the 2026-05-13 version — not touched by the 2026-07-06 review (explicitly out of scope). `app.py → _NBAAnalyzer.analyze() → modules/basketball_module.py::run_module()` — a 12-engine self-contained analyzer that falls back to demo data (Lakers @ Nuggets) unless real stats are explicitly passed in. Real-data fetchers (`nba_stats_fetcher.py`, `injuries_fetcher.py`) that used to exist for this purpose have been removed from the repo — this module currently has no live data path at all.

---

## 6. UFC Data Flow

Unchanged from the 2026-05-13 version — not touched by the 2026-07-06 review (explicitly out of scope). `app.py → _UFCAnalyzer.analyze() → modules/ufc_module.py::run_module()` — falls back to demo fighters unless real stats are passed in. `ufc_data_fetcher.py` (the scraper that used to exist for this) has been removed from the repo.

---

## 7. Connected vs Orphaned Files

No orphaned root-level fetchers remain as of 2026-07-06 (`injuries_fetcher.py`, `nba_stats_fetcher.py`, `ufc_data_fetcher.py` were all removed, along with 5 dead methods inside `data_fetchers.py` itself — see §2). The only known "fetched but never consumed" pattern still present is `_normalize_event()`'s F5 naming scheme in `odds_fetcher.py` (§2) — not orphaned code, just a latent naming inconsistency flagged for whoever connects it to F5 analysis next.

---

## 8. Database Schema

### `data/predictions_history.db`

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `predictions` | UI-generated predictions (one per analysis run) | `sport`, `home_team`, `away_team`, `ev`, `kelly`, `confidence`, `rating`, `p_home`, `p_away` |
| `game_outcomes` | Backtest: model predictions + actual scores, PIT metadata | `game_pk`, `game_date`, `season`, `lambda_home/away`, `p_home`/`p_home_raw`, `actual_home/away_runs`, `home_won`, `ml_home_pin`/`ml_away_pin`, `market_prob_home`/`market_prob_away`, `stage_factors_json`, `backtest_run_at` (note: the last 5 columns were added via idempotent `ALTER TABLE` migrations directly in `backtest_and_retrain.py`, independent of `learning_engine.py`'s own table-creation code — grep `backtest_and_retrain.py` for `ALTER TABLE game_outcomes` for the authoritative, current column list rather than treating any single file's schema definition as complete) |
| `ml_state` | Key-value store: team bias, Platt/Platt-2D params, pipeline weights | `key`, `scope`, `season`, `value_json` |
| `kalman_state` | Kalman filter state per team/context/season | `team`, `scope`, `season`, `x_est`, `p_est` |

### `data/track_record.db`

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `picks` | One row per published pick, pre-game timestamp | `pick_uid`, `published_at`, `sport`, `market` (`ML_HOME`\|`ML_AWAY`\|`RL_HOME`\|`RL_AWAY`\|`OVER`\|`UNDER`\|`F5_HOME`\|`F5_AWAY`\|`F5_OVER`\|`F5_UNDER`), `model_prob`, `ev_pct`, `odds_decimal`, `confidence_tier`, `result`, `profit_loss_units` |
| `bankroll` | Running P&L ledger (one row per resolved pick) | `pick_id`, `units_staked`, `units_pnl`, `running_total` (correctness of this column's ordering fixed 2026-07-06 — see §2) |
| `daily_snapshots` | Daily roll-up | `snap_date`, `picks_count`, `wins`/`losses`/`pushes`, `cumulative_pnl`, `roi_pct` |

### Other DBs

- `data/pit_cache_2024.db`, `pit_cache_2025.db`, `pit_cache_merged.db`, `pit_cache_pitcher.db` — point-in-time feature caches (TTE/Bullpen/Defense share the merged file, distinguished by `namespace`; Pitcher has its own).

---

## 9. Import Graph

```
                    config.py (root)
                        ▲
         ┌──────────────┼──────────────┬─────────────┐
         │              │              │             │
   data_fetchers    odds_fetcher    core/utils   db/predictions_db
         ▲  │                           ▲
         │  │ (direct: _fetch_team_roster,
         │  │  data_fetchers.py L.19)    │
         │  ▼                           │
         │  offense/true_talent_engine ◄┘ ── ALSO imported by run_module.py
         │        ▲                          (get_true_talent_lambda, L.56;
         │        │                           _fetch_game_lineup, L.376 —
         │        │                           lazy, inside a function body)
    run_module.py ────────────► core/value_detector
         ▲
         │
   ┌─────┴───────────────────────────────────────────────────┐
   │  calibration/learning_engine                              │
   │  context_engine/pitcher_engine, contextual_engine,        │
   │    bullpen_engine, defensive_efficiency_engine             │
   │  hfa/park_weather_engine, hfa_engine                       │
   │  montecarlo/simulator                                     │
   │  advanced_pit_enrichment/ (26 files, feeds the 4 PIT-aware │
   │    engines above)                                          │
   └─────────────────────────────────────────────────────────┘
         ▲
    ui/mlb.py ◄── ui/odds_loader.py, ui/sidebar.py, ui/components.py
         ▲
     app.py ◄──── track_record/ ◄──── run_daily_picks.py
         ▲
   [modules/basketball_module.py]  [modules/ufc_module.py]
   (self-contained, no live data path — see §5, §6)

backtest_and_retrain.py ──► imports the SAME live engines as run_module.py
                             (not a separate reimplementation), plus its own
                             use_team_full_season_defense/pitching_base gates
                             for PIT-safe backtest evaluation.
```

Confirmed via direct grep (2026-07-06) that no circular imports exist among the `context_engine/`, `hfa/`, and `offense/` engine packages — each imports `config`/`core` but never each other.

---

*End of CONTRACTS.md — verified against code 2026-07-06. Regenerate on the next major structural change rather than letting this drift again — see `docs/AUDITORIA_MLB_2026-07.md` for the audit that found the previous version stale.*
