# 02 — Project Inventory

Base source: `CONTRACTS.md` (rewritten 2026-07-06, verified against code that day) — reused
as the primary inventory since it was itself produced by a careful line-by-line audit and
this session independently re-verified several of its specific claims (§1 register). Deltas
against the current dirty working tree are called out explicitly below rather than silently
inherited.

## Entry points

| Mode | Entry point | Purpose |
|---|---|---|
| Live UI | `app.py` (329 lines — CONTRACTS.md said 305; grew slightly, not re-diffed) | Streamlit dashboard; imports `ui.mlb`, `ui.odds_loader`, `ui.sidebar`, `ui.components` |
| Daily CLI | `run_daily_picks.py` (140 lines) | cron target — `track_record.publisher.publish_daily_picks()` then `reconciler.reconcile_all_sports()` |
| Dev/Research | `backtest_and_retrain.py` (3,178+ lines pre-this-session, +190 lines uncommitted now — not re-counted), `analyze_game_outcomes.py` (558), `fetch_historical_odds.py` (581) | offline validation, historical bootstrap |

## Live pipeline (MLB, canonical — PASO 0-9)

Unchanged in structure from `CONTRACTS.md` §4 (re-verified this pass at the specific points
relevant to §4/§6/§9 of this audit — see those sections for line-level citations):

```
data_fetchers.MLBDataIntegrator.get_complete_game_data()  [PASO 0]
  -> offense/true_talent_engine.py (TTE, park-neutral λ_base)
  -> calibration/learning_engine.py (Kalman + multidim team bias)
  -> context_engine/pitcher_engine.py           [PASO 2]
  -> context_engine/contextual_engine.py        [PASO 3]
  -> context_engine/bullpen_engine.py           [PASO 4]
  -> hfa/park_weather_engine.py                 [PASO 5]
  -> context_engine/defensive_efficiency_engine.py [PASO 6]
  -> hfa/hfa_engine.py                          [PASO 7]
  -> montecarlo/simulator.py (Negative Binomial) [PASO 8]
  -> core/value_detector.py::evaluate_value_ultra [PASO 9]
```

## Backtest pipeline

`backtest_and_retrain.py` — not a separate reimplementation; imports and calls the same
engines as `run_module.py`. Supports 4 independent PIT modes plus `--use-full-pit`. Reads
`game_outcomes` filtered `WHERE actual_home_runs IS NOT NULL AND season IN (...)`
(backtest_and_retrain.py:2514-2523), writes back via `update_game_outcomes()`
(:1962-1984, unconditional `UPDATE ... WHERE game_pk = ?` — see **CHRON-001**, §8).

## Data fetchers

| File | Role |
|---|---|
| `data_fetchers.py` (2,182 lines + 11 uncommitted this session) | `MLBStatsAPI`, `MLBDataIntegrator`, `WeatherAPI` (OpenWeather, live forecast only) |
| `odds_fetcher.py` (521 lines, -4 net this session) | `get_odds_data()` (UI), `get_best_odds_for_teams()` (production auto-fetch) |
| `modules/baseball_module/hfa/historical_weather.py` | Open-Meteo archive, backtest-only, deliberately not wired into live path |
| `modules/baseball_module/data_enrichment/savant_fetcher.py`, `fangraphs_fetcher.py` | Statcast/FanGraphs enrichment, 24h cache, graceful failure |
| `scripts/build_offense_savant_rolling_incremental.py` (**new, untracked**, 187 lines) | O(days) incremental PIT offense-cache builder — per `CLAUDE.md`, built to fix the stale `savant.team_offense.rolling` cache that silently froze mid-2024/2025 |

## Feature engines (7 lambda-adjustment engines + TTE)

Same 7 engines as `CONTRACTS.md` §2/§3 (`pitcher_engine.py`, `contextual_engine.py`,
`bullpen_engine.py`, `park_weather_engine.py`, `defensive_efficiency_engine.py`,
`hfa_engine.py`, plus `true_talent_engine.py` pre-pipeline). **Delta this session**:
`true_talent_engine.py`'s composite-score math (weights, plate-factor clamp, current/prior
blend) was extracted into a new shared module, `modules/baseball_module/offense/
tte_formula.py` (145 lines, untracked) — `regress`, `clamp`, `plate_factor`,
`composite_score`, `season_lambda`, `blend_current_prior`. Stated purpose (per in-code
comment, true_talent_engine.py): unify this math with `tte_pit_adapter.py`'s independent
copy, which had drifted (§1 not directly registered but implied by REG-005's neighborhood;
treated as a real, in-progress refactor — checked for correctness in §5).

## Model/math modules

- `modules/baseball_module/montecarlo/simulator.py` (368 lines) — Negative Binomial,
  `NB_DISPERSION=6.0`, vectorized, early stopping `SE<0.003` on `p_home` only.
- `core/value_detector.py` (1,179 lines, grew from 1,119 per CONTRACTS.md — consistent with
  the 10-commit odds/value audit landing since 2026-07-06) — vig removal (3 methods),
  EV/edge/Kelly, composite score, tiers.
- `core/utils.py` (10 lines) — `calculate_ev`.

## Calibration / learning

`modules/baseball_module/calibration/learning_engine.py` — grew from 1,081 lines
(CONTRACTS.md) to +152 uncommitted lines this session (truncation-correction machinery, see
§1 REG-005/REG-006). 6 mechanisms: team bias, Kalman, Platt (1D), Platt-2D, gradient-descent
pipeline weights, idempotent `update_outcome()`/`record_prediction()`.

## Value detection / Kelly-staking

`core/value_detector.py::evaluate_value_ultra` (all markets), `kelly_criterion()` (single
source of truth, clipped `[MIN_KELLY=0.01, MAX_KELLY=0.15]` from `config.py`). Every caller
(pipeline `best_bets`, UI cards, track_record fallback) delegates to it — confirmed by
`CLAUDE.md`'s own architecture note, not independently re-grepped this pass (in-budget
callers already spot-checked individually in prior sessions per the register).

## Reporting

- `analyze_game_outcomes.py` — standalone analytics on `game_outcomes` (ROI by
  stadium/month/day-night, drawdown). **Note for §11**: reads the same `game_outcomes` table
  that CHRON-001 (§8) shows is shared, unprotected, between live and backtest writers — any
  of its "live performance" framing should be read with that caveat.
- `track_record/stats.py::compute_stats()` — headline W-L/ROI/Sharpe computed directly from
  `track_record.db::picks` (currently 0 rows — REG-031, re-confirmed this pass, see below).

## Database tables (read-only inspection performed this pass)

### `data/predictions_history.db` (7.5MB, last modified 2026-07-12 — the live/active DB)

| Table | Rows (this pass) | Note |
|---|---|---|
| `predictions` | 88 | UI-generated, one per analysis run |
| `game_outcomes` | 5,474 | seasons {2024, 2025, 2026}, `game_date` 2024-03-20 → 2026-07-12. **52 rows are pure live-production, unresolved** (`backtest_run_at IS NULL AND actual_home_runs IS NULL`, all season 2026, dated 2026-05-15 → 2026-07-12) — see CHRON-001, §8. **563 of 615 season-2026 rows already have `backtest_run_at` set**, i.e. a backtest run has already processed most of the 2026 season. |
| `ml_state` | 436 | Platt/Platt-2D params, team bias, pipeline weights |
| `kalman_state` | 360 | per team/context/season |
| `historical_odds` | 4,695 | from `fetch_historical_odds.py` |
| `results` | 0 | empty |
| `sport_stats` | 1 | — |

### `data/track_record.db`

| Table | Rows | Note |
|---|---|---|
| `picks` | **0** | Confirms REG-031 still true as of 2026-07-14 (was already 0 as of the 2026-07-06 audit). Full schema includes `closing_odds_decimal`/`closing_pin_home`/`closing_pin_away`/`closing_captured_at`/`clv_pct` — the CLV-capture columns referenced in `CLAUDE.md`'s memory of `track_record/capture_closing_lines.py` — present in schema, unpopulated in practice (nothing to capture yet, since 0 picks). |
| `bankroll`, `daily_snapshots` | not queried | consistent with 0 picks upstream |

### Stale / orphaned database files found in `data/` (not part of any documented schema)

| File | Size | Last modified | Note |
|---|---|---|---|
| `data/game_outcomes.db` | 0 bytes | 2026-05-24 | Empty, zero tables. Same base name as the live table inside `predictions_history.db` — a plausible source of confusion for anyone grepping for "the game_outcomes database" instead of the table. Not referenced by any `.py` file (confirmed: no `game_outcomes.db` string literal found in the codebase via grep). |
| `data/mlb_learning.db` | 45KB | 2026-05-27 (stale — predates the entire PIT rebuild and this session's fixes) | Contains `game_outcomes`/`kalman_state`/`ml_state` tables — an old, abandoned copy of the same schema as `predictions_history.db`, not the active one (`LearningEngine`'s actual `db_path` wiring confirmed elsewhere in this audit to point at `predictions_history.db`, not this file). |
| 18× `data/predictions_history_backup_pre_<label>_<timestamp>.db` | various | 2026-07-05 → 2026-07-12 | **Good practice, not a problem** — manual pre-backtest-round snapshots (`pre_round1_backtest`, `pre_teambias_leak_fix`, `pre_platt2d_refresh`, etc.), consistent with the project's own stated discipline of backing up before risky operations. Listed here for inventory completeness, not flagged as debt. |

Both stale files (`game_outcomes.db`, `mlb_learning.db`) are covered by `.gitignore`'s
`*.db` rule, so they pose no repo-hygiene risk, only a possible human/agent confusion risk
(picking the wrong file when asked to "check the game_outcomes DB"). Logged as a Low-severity
finding in `findings.csv` (**INV-002**).

## Scripts and caches

- `.cache/` — JSON caches (odds, Savant, FanGraphs, travel, roof, rest-days), TTLs per
  `CLAUDE.md`/`CONTRACTS.md`.
- `data/pit_cache_2024.db`, `pit_cache_2025.db`, `pit_cache_merged.db`, `pit_cache_pitcher.db`,
  `data/pit_raw/raw_savant_{2023,2024,2025}.db` — PIT feature caches, 32 `.db` files total
  under `data/` including backups.
- `reports/` (**new, untracked**, 1.8MB) — ad-hoc backtest run logs/JSON from this session's
  10 rounds (`round1_bullpen_tte_fix` … `round10_composite_weight_nudge`) plus `baseline`,
  `full_pit`, `offense_rebuild`. Read as historical evidence for this audit (not reproduced
  in full); consistent with `CLAUDE.md`'s narrative of one-change-at-a-time validated fixes.
- `.codex/` (**new, untracked**, 197MB) — **not a BetMindex artifact.** This is a separate
  CLI tool's home directory (its own SQLite state/logs/`auth.json`/plugins), sitting inside
  the repo root by coincidence of working directory, not `.gitignore`'d (unlike `.claude/`,
  which is). Zero references to it from any project `.py` file. Logged as **INV-001**
  (Low severity, hygiene only — recommend adding `.codex/` to `.gitignore` or relocating it
  outside the repo root; not a functional bug).

## Test suite

43 `.py` files under `tests/`, **462 tests total, 462 passed / 0 failed / 0 skipped** this
pass (§12 has the full transcript) — up from "39 files / 448 tests" in `CONTRACTS.md`
(2026-07-06), consistent with the 4 new untracked test files seen in `git status`
(`test_track_record_closing_lines.py`) plus whatever grew this session between commits.
