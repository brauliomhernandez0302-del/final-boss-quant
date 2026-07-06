# CONTRACTS.md — System Map: Final Boss Quant G8+

**Generated:** 2026-05-13  
**Version:** 2.1  
**Purpose:** North-star reference — what every file does, what is alive vs dead, and the correct data flow for every sport.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [File Inventory](#2-file-inventory)
3. [Active vs Dead Code](#3-active-vs-dead-code)
4. [MLB Data Flow (canonical)](#4-mlb-data-flow)
5. [NBA Data Flow](#5-nba-data-flow)
6. [UFC Data Flow](#6-ufc-data-flow)
7. [betting_ai/ — What it contains](#7-betting_ai--what-it-contains)
8. [bbets_ia_pro/ — What it contains](#8-bbets_ia_pro--what-it-contains)
9. [Connected vs Orphaned Files](#9-connected-vs-orphaned-files)
10. [Database Schema](#10-database-schema)
11. [Import Graph](#11-import-graph)

---

## 1. System Overview

**Final Boss Quant G8+** is a Streamlit sports-betting prediction system.  
Entry point: `app.py`. Run: `streamlit run app.py`.

The system models each sport as a probability estimator and compares those probabilities to market odds to find positive-EV bets. MLB is the primary sport with a full quantitative pipeline. NBA and UFC have self-contained modules but lack live data integrations. Soccer and boxing exist as code but are disabled.

**Three runtime modes:**

| Mode | Entry Point | Purpose |
|------|------------|---------|
| Live UI | `app.py` | Streamlit dashboard, one-game analysis |
| Daily CLI | `run_daily_picks.py` | Publish picks to track_record.db + reconcile results |
| Dev/Research | `backtest_and_retrain.py`, `ablation_calibrator.py`, `analyze_game_outcomes.py` | Offline model validation |

---

## 2. File Inventory

### Root-level Python files

| File | Lines | What it does |
|------|-------|-------------|
| `app.py` | 1,591 | **Primary entry point.** Streamlit UI. Defines `AppConfig`, `SportConfig`, `PredictionsDB`, `BaseAnalyzer`/`MLBAnalyzer`/`NBAAnalyzer`/`UFCAnalyzer`, all `render_*` functions, and `main()`. Imports and calls sport modules. |
| `config.py` | 54 | **Single source of truth for constants.** `MLB_SIMULATIONS=5_000_000`, `KELLY_FRACTION=0.25`, `MAX_RISK_PCT=0.05`, `LEAGUE_AVG_RUNS=4.5`, `LEAGUE_AVG_ERA=4.15`, `DEFAULT_MIN_EV=3.0`, etc. Everything else imports from here. |
| `data_fetchers.py` | 2,157 | **MLB data layer.** `MLBStatsAPI` (hits `statsapi.mlb.com`): game schedule, team stats, pitcher stats, bullpen ERA, standings, travel. `MLBDataIntegrator` orchestrates parallel fetches. `ParkFactors` class with FanGraphs 2024 park factors. `WeatherFetcher` (OpenWeather, optional). Free, no API key. |
| `odds_fetcher.py` | 383 | **Odds API layer.** `get_odds_data()` — fetches h2h + totals + spreads for 15 sport keys via The Odds API. 3-region coverage (us/uk/eu). On-disk JSON cache with `ODDS_FILE_CACHE_TTL=600s`. Returns list of game dicts. |
| `odds_api.py` | 93 | **Odds lookup + fuzzy matching.** `get_best_odds_for_teams(home, away, sport)` — fetches odds from The Odds API for a specific matchup using substring fuzzy matching. Returns best ML odds across all bookmakers plus Pinnacle reference. |
| `storage.py` | 46 | **CSV bet log.** `save_bets(df)` / `load_bets()` — appends/reads `bets_log.csv`. Independent of SQLite. Used by app.py sidebar "Save Picks" button. |
| `run_daily_picks.py` | 140 | **Daily CLI runner.** Calls `track_record.publisher.publish_daily_picks()` then `track_record.reconciler.reconcile_all_sports()` then prints summary. Designed for cron at 18:00 UTC. |
| `post_game.py` | 311 | **Post-game reconciliation script.** Fetches actual scores via `LearningEngine.fetch_pending_outcomes()`, matches them to `predictions` table by team names + date, updates `results`, refreshes team bias cache in `ml_state`. Older predecessor to `run_daily_picks.py`. |
| `train_historical.py` | 359 | **One-time historical bootstrap.** Downloads every 2024-2025 MLB game result via MLB Stats API, inserts into `game_outcomes` using season RPG as predicted λ, then recomputes team bias. Safe to re-run (UNIQUE on game_pk). |
| `fetch_historical_odds.py` | 571 | **One-time historical odds downloader.** Pulls 2024-2025 MLB pre-game moneylines from The Odds API historical endpoint, one bulk call per game-day at 17:00 UTC. Writes to `historical_odds` table. ~7,380 total API calls. Costs quota. Per memory notes: 162 Athletics games pending re-fetch when API quota resets. |
| `backtest_and_retrain.py` | 1,094 | **Full pipeline backtest.** Runs every game in `game_outcomes` (4,695+ games) through the complete pipeline: `AutoCalibrator → HFA → PitcherEngine → Regression → MC(50k sims)`. Writes real model λ/p back to `game_outcomes`. Produces JSON report in `data/`. Uses disk-cached API calls in `.cache/backtest/`. |
| `ablation_calibrator.py` | 508 | **Ablation test harness.** Compares Scenario A (with `AutoCalibrator`) vs Scenario B (without, using Kalman+bias only). Uses `backtest_and_retrain.py` infrastructure. Zero new API calls. |
| `analyze_game_outcomes.py` | 558 | **Analytics script.** Reads `game_outcomes` table, analyzes ROI by stadium, month, day/night, drawdown curves, systematic failures. Standalone — no pipeline changes. |
| `injuries_fetcher.py` | 892 | NBA injury scraper (ESPN). **ORPHANED — never imported anywhere.** See §9. |
| `nba_stats_fetcher.py` | 783 | NBA team stats via `nba_api`. **ORPHANED — never imported anywhere.** See §9. |
| `ufc_data_fetcher.py` | 524 | UFC fighter stats scraper (web scraping, BeautifulSoup). **ORPHANED — never imported anywhere.** See §9. |

---

### `core/` — Shared utilities

| File | Lines | What it does |
|------|-------|-------------|
| `core/utils.py` | 10 | `calculate_ev(prob, decimal_odds) → float (%)` — single source of truth for EV formula. Imported by `basketball_module.py` and `value_detector.py`. |
| `core/value_detector.py` | 925 | **Multi-market value detector.** `evaluate_value_ultra(mc_result, odds, ...)` — full 9-step analysis: vig removal (3 methods), EV per market, Kelly sizing, bootstrap confidence intervals, tier classification (ULTRA/HIGH/MEDIUM/SLIGHT), Pinnacle line as fair reference. Markets: ML_HOME, ML_AWAY, OVER, UNDER, RL_HOME (+/-1.5), RL_AWAY, F5_HOME, F5_AWAY. Returns sorted `best_bets` list. |

---

### `modules/` — Sport analysis engines

| File | Lines | What it does |
|------|-------|-------------|
| `modules/baseball_module/core/run_module.py` | 630 | **MLB pipeline orchestrator.** `run_module(game_id, ...)` — drives the 6-step Poisson λ pipeline (see §4). Also defines `_compute_f5_lambda()` and `_platt()`. Calls every engine in sequence, applies gradient-descent pipeline weights from `LearningEngine`. |
| `modules/baseball_module/calibration/auto_calibrator.py` | 336 | **LambdaCalibrator.** Adjusts λ using 5 multiplicative factors: offense quality (30%), opponent defense (25%), recent form (25%), rest days (10%), season context (10%). Combined ±15% hard cap applied on the final product. wOBA league avg = 0.310 (FanGraphs 2024). |
| `modules/baseball_module/calibration/learning_engine.py` | 758 | **Adaptive ML engine.** 5 mechanisms: (1) team bias (mean actual/predicted λ per team/season), (2) multi-dim bias (home/away/month/stadium), (3) Kalman filter (hidden team run-rate state), (4) Platt recalibration (weekly logistic refit on outcomes), (5) gradient-descent pipeline weights. Tables in `predictions_history.db`. |
| `modules/baseball_module/hfa/hfa_engine.py` | 353 | **HFA Engine.** `get_adjusted_lambdas(lh, la, game_data)` — applies: (1) home crowd boost (asymmetric, home only, empirically ÷4), (2) FanGraphs 5-year park run-factor (symmetric), (3) away travel fatigue (miles + time zones, asymmetric). Does NOT touch team offense/defense (AutoCalibrator owns that). |
| `modules/baseball_module/context_engine/pitcher_engine.py` | 540 | **PitcherEngine.** `adjust_for_pitchers(lh, la, game_data)` — weights: quality (0.25), form (0.20), matchup vs lineup (0.15), platoon splits (0.08), fatigue (0.10), park-for-pitchers (0.10), travel (0.05), bullpen (0.07). Strictly pitcher signals only — does not touch team context. |
| `modules/baseball_module/context_engine/pitchers_regression.py` | 163 | **PitcherRegressionEngine.** `calculate_pitcher_regression(pitcher_stats, ...)` — BABIP luck, LOB% luck, HR/FB% luck vs career baselines (0.285 / 0.738 / 0.106 from FanGraphs 2024-2025). Returns `(factor, confidence)`. factor > 1 → pitcher was lucky → expect more runs. |
| `modules/baseball_module/montecarlo/simulator.py` | 251 | **Monte Carlo engine.** `monte_carlo_advanced(lh, la, n_max=5_000_000, ...)` — vectorized Poisson blocks of 200k, early stopping at SE < 0.003. Outputs p_home, p_away, p_tie, mean_total, p_over/under per line, plus F5 probabilities. λ noise = 0.05 (mild uncertainty). |
| `modules/baseball_module/data_enrichment/savant_fetcher.py` | ~120 | **Baseball Savant enrichment (optional).** `SavantFetcher.get_pitcher_stats(mlbam_id, year)` — fetches xERA, xwOBA from Savant leaderboard CSVs. 24h disk cache. Used in `run_module` if `_ENRICHMENT_AVAILABLE=True`. |
| `modules/baseball_module/data_enrichment/fangraphs_fetcher.py` | ~130 | **FanGraphs enrichment (optional).** `FanGraphsFetcher.get_pitcher_stats(mlbam_id, year)` — fetches xFIP, SIERA, WAR, K%, BB%, SwStr% from FanGraphs public API. 24h disk cache. |
| `modules/baseball_module/utils/helpers.py` | 28 | `clamp(x, a, b)`, `infer_decimal(odds)`, `prob_from_decimal(dec)`. |
| `modules/baseball_module/logging/__init__.py` | — | Empty. The `logging/` subdirectory is a naming curiosity — it does not shadow the stdlib `logging` module because Python resolves absolute imports from `sys.path`, not local package names at this depth. |
| `modules/basketball_module.py` | 2,411 | **NBA G10+ Ultra Pro V2.** `NBAAnalyzerG10PlusV2` with 12-engine analysis (see §5). `run_module(data=None, home_team=None, ...)` as public interface. Self-contained: no external API calls during analysis. Demo falls back to Lakers @ Nuggets. |
| `modules/ufc_module.py` | 626 | **UFC G8+ Ultra.** `UFCAnalyzer.analyze_fight()` — style matchup matrix, physical attrs, record quality, round simulation. `run_module(data=None, fighter1_data=None, ...)`. Self-contained: demo falls back to Fighter A vs Fighter B. |
| `modules/football_module.py` | 575 | **DISABLED soccer predictor.** `enabled=False` in `SPORT_CONFIGS`. Has `st.set_page_config()` at module level — importing it would crash the app. Dead code. |
| `modules/boxing_module.py` | 205 | **Disconnected boxing module.** Not in `SPORT_CONFIGS`. No `run_module()` function. Uses a separate `predictions_history.sqlite` (not the main DB). Dead code. |

---

### `track_record/` — Live pick tracking

| File | Lines | What it does |
|------|-------|-------------|
| `track_record/db.py` | ~120 | `TrackRecordDB` — SQLite CRUD for `data/track_record.db`. Tables: `picks` (one row per pick, with `published_at` timestamp as audit proof), `bankroll` (running P&L ledger). |
| `track_record/publisher.py` | ~200 | `publish_daily_picks(db, sports, dry_run)` — runs the full MLB/NBA/UFC analysis pipeline for today's games, saves every bet meeting EV/tier threshold to `picks` table with `published_at < game_start - 30min`. Only publishes once per `pick_uid`. |
| `track_record/reconciler.py` | ~180 | `reconcile_all_sports(db, lookback_days)` — fetches final MLB scores via `statsapi` (with `MLBStatsAPI` fallback), resolves WIN/LOSS/PUSH for each pending pick by market logic (ML, run line, total, F5). |
| `track_record/stats.py` | ~120 | `compute_stats(db, sport)` — calculates headline (W-L, win rate, ROI, Sharpe), by-sport, by-market, by-tier, by-month, bankroll curve, recent picks. Returns structured dict for Streamlit. |
| `track_record/ui.py` | ~100 | `render_track_record()` — Streamlit page shown in app.py "Track Record" tab. Filterable by sport and result. |
| `track_record/__init__.py` | 8 | Exports `TrackRecordDB`. |

---

### `tests/` — Test suite

| File | What it tests |
|------|--------------|
| `tests/conftest.py` | Adds project root to `sys.path`. |
| `tests/test_hfa_pipeline.py` | HFA engine multiplicative pipeline: each factor applies independently, home boost only to λ_home, travel only to λ_away, park factor symmetric. |
| `tests/test_f5_lambda.py` | `_compute_f5_lambda()` in `run_module.py` — F5 expected runs from starter avg_IPS + f5_ERA + bullpen ERA. |
| `tests/test_formulas.py` | Core math: EV, Kelly, prob conversions. |
| `tests/test_defense_multiplier.py` | AutoCalibrator's `_calculate_defense_multiplier`. |
| `tests/test_learning_engine.py` | `LearningEngine` table init, bias caching, Platt params. |
| `tests/test_montecarlo.py` | `monte_carlo_advanced()` — output shape, early stopping, F5 mode. |
| `tests/test_pitcher_engine.py` | `adjust_for_pitchers()` — direction and magnitude of adjustments. |

---

## 3. Active vs Dead Code

### ACTIVE — Connected to the live system

```
app.py                                         ← runs everything
config.py                                      ← imported by everyone
data_fetchers.py                               ← MLB data source
odds_fetcher.py                                ← odds data source
odds_api.py                                    ← odds lookup in run_module
storage.py                                     ← CSV bet log (manual save)
core/utils.py                                  ← basketball_module + value_detector
core/value_detector.py                         ← step 6 of MLB pipeline
modules/baseball_module/core/run_module.py     ← MLB pipeline entry
modules/baseball_module/calibration/auto_calibrator.py
modules/baseball_module/calibration/learning_engine.py
modules/baseball_module/hfa/hfa_engine.py
modules/baseball_module/context_engine/pitcher_engine.py
modules/baseball_module/context_engine/pitchers_regression.py
modules/baseball_module/montecarlo/simulator.py
modules/baseball_module/data_enrichment/savant_fetcher.py  (optional, graceful fail)
modules/baseball_module/data_enrichment/fangraphs_fetcher.py (optional, graceful fail)
modules/baseball_module/utils/helpers.py
modules/basketball_module.py
modules/ufc_module.py
track_record/db.py
track_record/publisher.py
track_record/reconciler.py
track_record/stats.py
track_record/ui.py
run_daily_picks.py
post_game.py                                   ← older reconciler, still functional
tests/ (all 7 files)
```

### ACTIVE — Dev/Research tools (not part of live pipeline)

```
backtest_and_retrain.py          ← used for model validation
ablation_calibrator.py           ← calls backtest_and_retrain infrastructure
analyze_game_outcomes.py         ← analytics on historical data
train_historical.py              ← one-time historical bootstrap (already run)
fetch_historical_odds.py         ← one-time historical odds download (partially run)
```

### ORPHANED — Written but never imported or called

```
injuries_fetcher.py              ← NBA injury scraper, 0 callers
nba_stats_fetcher.py             ← NBA stats via nba_api, 0 callers
ufc_data_fetcher.py              ← UFC fighter scraper, 0 callers
```

### DEAD CODE — Disabled or empty

```
modules/football_module.py       ← Soccer, enabled=False in SPORT_CONFIGS;
                                    st.set_page_config() at module level = crash on import
modules/boxing_module.py         ← No run_module(), not in SPORT_CONFIGS,
                                    writes to different SQLite file

betting_ai/main.py               ← EMPTY (0 bytes)
betting_ai/detector.py           ← EMPTY (0 bytes)
betting_ai/bankroll.py           ← EMPTY (0 bytes)
betting_ai/gestor_apuestas.py    ← EMPTY (0 bytes)
betting_ai/odds_api.py           ← EMPTY (0 bytes)
betting_ai/model/__init__.py     ← EMPTY (0 bytes)
betting_ai/model/betting_model.py ← EMPTY (0 bytes)
```

---

## 4. MLB Data Flow

Full chain from "user clicks Analyze" to "best bet displayed":

```
[Streamlit UI — app.py]
│
├── render_sidebar() → settings: kelly_factor, min_ev, min_rating
├── load_odds_data() → odds_fetcher.get_odds_data()
│   ├── Hits The Odds API: h2h + totals + spreads, 3 regions
│   ├── On-disk JSON cache (ODDS_FILE_CACHE_TTL=600s)
│   └── Returns DataFrame of all sport odds
│
├── filter_odds_by_sport(df, ["baseball","mlb"])
├── build_game_selector(sport_df) → user picks matchup from dropdown
│
└── [User clicks "Analizar Evento (MLB)"]
    │
    ▼
[MLBAnalyzer.analyze() — app.py:875]
│
├── find_game_id(game_data)
│   └── MLBDataIntegrator.get_complete_game_data() for today+tomorrow
│       → fuzzy-match home/away names → return game_pk
│
└── run_module(game_id=game_pk, ...) [modules/baseball_module/core/run_module.py]
    │
    ├── INIT: LearningEngine(db_path)
    │   └── fetch_pending_outcomes() → fills actual scores for last 7 days
    │       via MLB Stats API → triggers bias recalc + Platt refit if due
    │
    ├── PASO 0: Fetch game data
    │   ├── MLBStatsAPI.get_todays_games() + get_games_by_date(tomorrow)
    │   ├── [Streamlit] st.selectbox → user picks specific game
    │   ├── MLBDataIntegrator.get_complete_game_data(date)
    │   │   ├── Team offensive stats: wOBA, OPS, wRC+, RPG (season)
    │   │   ├── Team pitching stats: ERA, WHIP, runs_allowed_per_game
    │   │   ├── Pitcher stats: ERA, FIP, WHIP, K/9, last-5 ERA, f5_ERA,
    │   │   │                  avg_innings_per_start, platoon splits
    │   │   ├── Bullpen ERA (home + away)
    │   │   ├── Team form: wins, losses, last-10, streak
    │   │   ├── Travel: miles_traveled_away, time_zones_crossed_away, B2B flag
    │   │   ├── Park: venue name (ParkFactors provides run/HR factors)
    │   │   ├── H2H stats (from .cache/ JSON)
    │   │   └── Weather (OpenWeather, optional)
    │   │
    │   ├── [OPTIONAL] SavantFetcher.get_pitcher_stats(mlbam_id)
    │   │   → enriches pitcher dict with xERA, xwOBA (24h cache)
    │   │
    │   └── [OPTIONAL] FanGraphsFetcher.get_pitcher_stats(mlbam_id)
    │       → enriches pitcher dict with xFIP, SIERA, SwStr% (24h cache)
    │
    │   BASE LAMBDAS:
    │   lh = home_team.runs_per_game (season RPG)
    │   la = away_team.runs_per_game
    │
    │   PRE-CALIBRATION ADJUSTMENTS (inside run_module before Step 1):
    │   ├── LearningEngine.get_kalman_estimate(team, season)
    │   │   → Kalman-filtered team run rate (KF Q=0.025, R=9.0)
    │   │   → lh = 0.65*lh + 0.35*kalman_home; la = 0.65*la + 0.35*kalman_away
    │   └── LearningEngine.compute_team_bias(team, season)
    │       → mean(actual/predicted) from game_outcomes, min 10 samples
    │       → applied as multiplicative bias (clamped ±30%)
    │       → integrated into AutoCalibrator ±15% combined cap
    │
    ├── PASO 1: AutoCalibrator [auto_calibrator.py]
    │   LambdaCalibrator.calibrate(lh, la, game_data)
    │   ├── Factor 1 (30%): Offense — wOBA vs 0.310, OPS vs 0.735, wRC+ vs 100, RPG vs 4.5
    │   ├── Factor 2 (25%): Opponent defense — ERA vs 4.15, WHIP vs 1.30, DER vs 0.715
    │   ├── Factor 3 (25%): Recent form — last-10 win%, streak momentum
    │   ├── Factor 4 (10%): Rest days — bonus for 3+ days rest, penalty for B2B
    │   ├── Factor 5 (10%): Season context — early/mid/late phase adjustment
    │   ├── Combined product × team_bias → ±15% HARD CAP applied
    │   └── Pipeline weight from LearningEngine:
    │       lh = lh_pre × (1 + w_cal × (factor-1))
    │
    ├── PASO 2: HFA Engine [hfa_engine.py]
    │   get_adjusted_lambdas(lh, la, game_data)
    │   ├── Home crowd boost: +0.034 R/G asymmetric (empirically calibrated)
    │   ├── Park run-factor: FanGraphs 5yr (e.g. Coors=1.25, Petco=0.88), symmetric
    │   ├── Travel fatigue: miles × per-1000mi penalty + timezone penalty, away only
    │   └── Pipeline weight applied: lh = lh_pre × (1 + w_hfa × (factor-1))
    │
    ├── PASO 3: Pitcher Engine [pitcher_engine.py]
    │   adjust_for_pitchers(lh, la, game_data)
    │   ├── Quality (0.25): ERA, FIP, WHIP relative to league avg
    │   ├── Form (0.20): ERA last-5 trend, K/9 trend
    │   ├── Matchup (0.15): historical ERA vs this lineup (era_vs_opp)
    │   ├── Platoon (0.08): LHB/RHB splits × opposing lineup handedness
    │   ├── Fatigue (0.10): days rest, last pitch count
    │   ├── Park for pitchers (0.10)
    │   ├── Travel (0.05): pitcher-specific travel (away pitcher only)
    │   ├── Bullpen (0.07): bullpen ERA + workload × expected usage %
    │   └── Pipeline weight applied
    │
    ├── PASO 4: Pitcher Regression [pitchers_regression.py]
    │   calculate_pitcher_regression(pitcher_stats, opponent_stats)
    │   ├── BABIP luck: current vs career 0.285 baseline
    │   ├── LOB% luck: current vs career 0.738 baseline
    │   ├── HR/FB% luck: current vs career 0.106 baseline
    │   ├── Confidence: scales with innings_pitched, signal alignment
    │   └── factor_away applied to lh (home team runs against away pitcher)
    │       factor_home applied to la (away team runs against home pitcher)
    │
    ├── PASO 4b: Umpire zone adjustment (symmetric, ±4% max, if available)
    │   zone_factor from umpire_stats (strike%, games_worked ≥ 4)
    │
    ├── λ CLAMP: lh = max(3.0, min(lh, 7.0)); la = max(3.0, min(la, 7.0))
    │
    ├── PASO 5: Monte Carlo [simulator.py]
    │   monte_carlo_advanced(lh, la, n_max=5_000_000)
    │   ├── Vectorized Poisson in blocks of 200k
    │   ├── lambda_noise=0.05 (mild game-to-game uncertainty)
    │   ├── Early stopping: SE < 0.003 (typically stops at ~500k-2M sims)
    │   ├── Full-game: p_home, p_away, p_tie, mean_total, p_over/under
    │   ├── F5: lh_f5 / la_f5 from _compute_f5_lambda()
    │   │   (starter_ip × f5_ERA + bullpen_ip × bullpen_ERA) / 9
    │   ├── Platt calibration: get_platt_params() → logistic shrinkage
    │   │   p_home, p_away renormalized to sum to 1.0 after Platt
    │   └── LearningEngine.record_prediction() — persists to game_outcomes
    │
    ├── PASO 6: Value Detection [core/value_detector.py]
    │   odds_api.get_best_odds_for_teams(home, away) → market odds
    │   evaluate_value_ultra(mc_result, game_odds)
    │   ├── Vig removal: multiplicative + power + Shin methods → average
    │   ├── EV = (prob × odds) - 1, expressed as %
    │   ├── Kelly = ((prob × (odds-1)) - (1-prob)) / (odds-1) × 0.25, capped 15%
    │   ├── Bootstrap CI (1000 samples) on probability estimate
    │   ├── Tiers: ULTRA>15%, HIGH>8%, MEDIUM>4%, SLIGHT>1%
    │   └── Markets evaluated: ML_HOME, ML_AWAY, OVER, UNDER,
    │                          RL_HOME(-1.5), RL_AWAY(+1.5), F5_HOME, F5_AWAY
    │
    └── RETURN: {status, game_info, probabilities, lambdas_history, best_bets, metadata}

[Back in app.py]
├── render_mlb_results(result, game_data, settings)
│   ├── Shows λ progression table (per-stage)
│   ├── Win probabilities + total
│   ├── Value cards (EV, Kelly) per market
│   └── best_bets list by tier
│
├── PredictionsDB.save() → predictions_history.db:predictions table
└── [Optional] storage.save_bets() → bets_log.csv
```

---

## 5. NBA Data Flow

```
app.py → NBAAnalyzer.analyze()
│
├── Passes {home_team: {name}, away_team: {name}, game_context: {...}}
│   NOTE: Only team names are passed from odds data. No real stats.
│
└── basketball_module.run_module(data=payload)
    │
    └── NBAAnalyzerG10PlusV2.analyze_game()
        │
        ├── Fallback to demo data if no stats provided (Lakers @ Nuggets)
        │   ← nba_stats_fetcher.py is NOT called here (orphaned)
        │   ← injuries_fetcher.py is NOT called here (orphaned)
        │
        └── 12 WEIGHTED ENGINES (weights sum to 1.0):
            ├── Context (0.18): B2B(-3.0pt), timezone(-1.2), rest advantage(+2.0)
            ├── Injuries (0.20): player tiers × status (uses passed injury list only)
            ├── Pace (0.14): possessions / game → scoring projection
            ├── Monte Carlo (0.16): correlated Gaussian simulation (not Poisson)
            ├── Shooting Luck (0.08): eFG regression to mean
            ├── Sharp Money (0.12): line movement signals (hardcoded heuristics)
            ├── Matchups (0.10): paint pts, 3PT%, PnR, fastbreak, turnovers
            ├── HFA (0.04): home court + altitude (Denver = +3.2 pts)
            ├── Blowout (0.03): variance + consistency
            ├── Risk (0.05): opponent strength
            ├── Stability (0.05): scoring variance
            └── Trends (0.05): last 3/5/10 game scoring

        Output: {probabilities: {home_win, away_win, margin, total}, best_bets: [...]}

        EV: core.utils.calculate_ev(prob, decimal_odds)
        No Kelly sizing in NBA module output.
        No Odds API integration inside basketball_module — EV uses passed odds only.
```

**Critical gap:** NBA module runs on demo data unless the caller explicitly passes real team stats. `nba_stats_fetcher.py` has the fetching code but is never called.

---

## 6. UFC Data Flow

```
app.py → UFCAnalyzer.analyze()
│
├── Passes {fighter1_data: {name}, fighter2_data: {name}}
│   NOTE: Only fighter names from odds data. No real fight stats.
│
└── ufc_module.run_module(fighter1_data, fighter2_data, ...)
    │
    └── UFCAnalyzer.analyze_fight()
        │
        ├── Fallback to demo Fighter A vs Fighter B if no stats provided
        │   ← ufc_data_fetcher.py is NOT called here (orphaned)
        │
        ├── Style matchup matrix: striker/grappler/wrestler 3x3 advantage table
        ├── Physical factors: reach delta, age delta
        ├── Record quality: wins, losses, recent 5 results
        ├── Technical factors: striking/grappling accuracy + defense
        ├── Finish probability: KO%, submission%, decision%
        ├── Cardio/conditioning factor
        ├── n_simulations × round simulation with random outcome draws
        └── EV vs market odds

        Output: {probabilities: {p1_win, p2_win, p_ko, p_sub, p_dec}, best_bets: [...]}
```

**Critical gap:** UFC module runs on demo data unless the caller passes real fighter stats. `ufc_data_fetcher.py` has scraping code but is never called.

---

## 7. `betting_ai/` — What it contains

**All 7 files are completely empty (0 bytes):**

```
betting_ai/
├── main.py                — empty
├── detector.py            — empty
├── bankroll.py            — empty
├── gestor_apuestas.py     — empty
├── odds_api.py            — empty
└── model/
    ├── __init__.py        — empty
    └── betting_model.py   — empty
```

This folder was a placeholder created during early project planning. No code was ever written. The directory structure suggests an intended alternative value-detection system (detector, model, bankroll, odds_api), but none of the files were implemented. The live system uses `core/value_detector.py` and `odds_api.py` at the root instead.

**Action:** Can be safely deleted or used as a staging area for a v2 refactor.

---

## 8. `bbets_ia_pro/` — What it contains

```
bbets_ia_pro/
├── .env                   — separate environment variables (unused by main project)
└── venv/                  — complete Python 3.12 virtual environment
    ├── bin/
    ├── lib/
    └── ...
```

This is an early, abandoned project directory. It contains **no Python source code** — only a virtual environment and a `.env` file. The main project's virtual environment is at `/home/raulio/mi_entorno/` instead.

The `bbets_ia_pro/` name suggests it was an early version of the current system, then superseded when the project moved to `/home/raulio/` as the root.

**Action:** The venv (~700MB estimated) is a stranded artifact. Safe to delete `bbets_ia_pro/venv/`. The `.env` may contain a different API key — check before deleting.

---

## 9. Connected vs Orphaned Files

### Import dependency graph (simplified)

```
config.py ←────────────────────────────── everyone imports this
data_fetchers.py ←────────────────────── run_module, backtest, post_game, reconciler
odds_fetcher.py ←─────────────────────── app.py (via safe_import)
odds_api.py ←─────────────────────────── run_module (PASO 6)
core/utils.py ←───────────────────────── basketball_module, value_detector
core/value_detector.py ←──────────────── run_module (PASO 6), app.py

modules/baseball_module/core/run_module.py ←── app.MLBAnalyzer, publisher, backtest
  ├── data_fetchers (MLBStatsAPI, MLBDataIntegrator)
  ├── calibration/auto_calibrator (LambdaCalibrator)
  ├── calibration/learning_engine (LearningEngine)
  ├── hfa/hfa_engine (get_adjusted_lambdas)
  ├── context_engine/pitcher_engine (adjust_for_pitchers)
  ├── context_engine/pitchers_regression (calculate_pitcher_regression)
  ├── montecarlo/simulator (monte_carlo_advanced)
  ├── core/value_detector (evaluate_value_ultra)
  └── odds_api (get_best_odds_for_teams)

modules/basketball_module.py ←──────────── app.NBAAnalyzer
  ├── config (NBA_SIMULATIONS, KELLY_FRACTION)
  └── core/utils (calculate_ev)

modules/ufc_module.py ←─────────────────── app.UFCAnalyzer
  └── config (UFC_SIMULATIONS)

track_record/ ←─────────────────────────── app.py (Track Record tab), run_daily_picks
  publisher ← run_module (indirectly, via analyze pipeline)
  reconciler ← data_fetchers (MLBStatsAPI), statsapi
  stats ← db
  ui ← db, stats
```

### Orphaned (no callers, not connected)

| File | Why orphaned | Fix to connect |
|------|-------------|----------------|
| `injuries_fetcher.py` | NBA module uses only the `injuries` list passed from caller; ESPN scraping never triggered | Add call in `basketball_module.run_module()` or `publisher.py` |
| `nba_stats_fetcher.py` | NBA module falls back to demo data; no call site populates real stats | Add call in `publisher.py` NBA analysis branch |
| `ufc_data_fetcher.py` | UFC module falls back to demo data; no call site populates real fighter stats | Add call in `publisher.py` UFC analysis branch |
| `modules/football_module.py` | `enabled=False` in SPORT_CONFIGS; `st.set_page_config()` at module top would crash | Re-architect to separate page config from module body |
| `modules/boxing_module.py` | Not in SPORT_CONFIGS; no `run_module()` wrapper; uses different DB file | Add `run_module()`, register in SPORT_CONFIGS, migrate DB |

---

## 10. Database Schema

### `data/predictions_history.db`

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `predictions` | UI-generated predictions (one per analysis run) | `sport`, `home_team`, `away_team`, `ev`, `kelly`, `rating`, `p_home`, `p_away` |
| `game_outcomes` | LearningEngine: model predictions + actual scores | `game_pk`, `game_date`, `lambda_home/away`, `p_home/away`, `actual_home/away_runs`, `home_won` |
| `ml_state` | Key-value store for team bias, Platt params, pipeline weights | `key`, `scope`, `season`, `value_json`, `sample_count` |
| `kalman_state` | Kalman filter state per team/context/season | `team`, `scope`, `season`, `x_est`, `p_est` |
| `historical_odds` | Historical Pinnacle odds for backtesting | `game_pk`, `game_date`, `home_ml`, `away_ml`, `pinnacle_home`, `pinnacle_away` |

### `data/track_record.db`

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `picks` | One row per published pick, with pre-game timestamp | `pick_uid`, `published_at`, `sport`, `market`, `model_prob`, `ev_pct`, `confidence_tier`, `result`, `profit_loss_units` |
| `bankroll` | Running P&L ledger (one row per resolved pick) | `pick_id`, `units_staked`, `units_pnl`, `running_total` |

---

## 11. Import Graph

```
                    config.py (root)
                        ▲
         ┌──────────────┼──────────────┐
         │              │              │
   data_fetchers     odds_fetcher   core/utils
         ▲                              ▲
         │                              │
    run_module.py ────────────► core/value_detector
         ▲
         │
   ┌─────┴──────────────────────────────────┐
   │  auto_calibrator  learning_engine       │
   │  hfa_engine       pitcher_engine        │
   │  pitchers_regression  simulator         │
   └─────────────────────────────────────────┘
         ▲
     app.py ◄──── track_record/ ◄──── run_daily_picks.py
         ▲
   [basketball_module]  [ufc_module]
   (self-contained)     (self-contained)

ISOLATED (no callers):
   injuries_fetcher.py
   nba_stats_fetcher.py
   ufc_data_fetcher.py

EMPTY:
   betting_ai/* (all 7 files)

DEAD:
   modules/football_module.py
   modules/boxing_module.py
   bbets_ia_pro/ (only venv)
```

---

*End of CONTRACTS.md — Last updated 2026-05-13*
