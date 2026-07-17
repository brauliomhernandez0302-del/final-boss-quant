# 03 — Call Graph

## Full chain: raw data → features → lambdas → adjustments → Monte Carlo → calibration → probability → edge → stake

```
[raw data]
MLBStatsAPI (data_fetchers.py) ── free, no key, statsapi.mlb.com
  │
  ▼
MLBDataIntegrator.get_complete_game_data()  [run_module.py PASO 0, imported line 17]
  ├── pitcher stats (fallback hierarchy, SIERA→xFIP→xERA→FIP→ERA)
  ├── WeatherAPI.get_weather_for_stadium()  (OpenWeather, live only — historical_weather.py
  │     is the backtest analog, deliberately NOT wired into the live path)
  ├── bullpen workload + ERA, team_pitching_stats, defense_home/away (DER+OAA), travel
  │     fatigue, roof status, days rest, lineup handedness
  │
  ▼
[features → λ_base]
true_talent_engine.get_true_talent_lambda()  (imported run_module.py:56, lazy inside function
  body — matches CONTRACTS.md's import-graph note) — Statcast xwOBA/barrel%/plate discipline,
  PIT-aware, lineup-filtered. Delegates composite-score math to tte_formula.py (NEW this
  session, see §2).
  │
  ▼
[Kalman + team bias]  calibration/learning_engine.py, inline in run_module.py
  compute_team_bias_kalman_adjusted() → dampened bias × Kalman-blended λ (see §1 REG-006 for
  why the dampening formula is empirically load-bearing despite a false derivation premise)
  │
  ▼ (PASO 2) adjust_for_pitchers()          context_engine/pitcher_engine.py   [run_module.py:528]
  ▼ (PASO 3) adjust_for_context()           context_engine/contextual_engine.py [:559]
  ▼ (PASO 4) adjust_for_bullpen()           context_engine/bullpen_engine.py   [:624]
  ▼ (PASO 5) adjust_for_park_and_weather()  hfa/park_weather_engine.py         [:651]
  ▼ (PASO 6) adjust_for_defense()           context_engine/defensive_efficiency_engine.py [:675]
  ▼ (PASO 7) get_adjusted_lambdas()         hfa/hfa_engine.py                  [:706]
  │   [λ clamp 1.5-12.0, lambdas_history['final'] stamped]
  ▼
[Monte Carlo]  monte_carlo_advanced()  montecarlo/simulator.py  [run_module.py:803]
  Negative Binomial, NB_DISPERSION=6.0, early stopping SE<0.003 on p_home only
  │
  ▼
[calibration]  Platt / Platt-2D  (calibration/learning_engine.py, applied inside
  evaluate_value_ultra via an injected p_home_corrector callable — see §9)
  │
  ▼
[probability → edge → stake]  evaluate_value_ultra()  core/value_detector.py  [run_module.py:928]
  vig removal → EV/edge → kelly_criterion() (single source of truth, clip [0.01, 0.15])
  → classify_value_tier() → best_bets
```

**Verified this pass**: every engine in the PASO 2-7 chain is imported and called **exactly
once** per `run_module()` invocation (grepped `adjust_for_*`/`get_adjusted_lambdas` call
sites — one occurrence each, run_module.py:528/559/624/651/675/706). **No module is applied
more than once** in the live lambda chain. `evaluate_value_ultra()` is likewise called once
(:928). This directly confirms/re-confirms `docs/AUDITORIA_MLB_2026-07.md` §4's own
conclusion (no geometric-mean confusion, no accidental double-invocation).

## Active modules

Same set as `CONTRACTS.md` §3 "ACTIVE" list — re-verified spot checks (this pass): all 7
lambda-adjustment engines import cleanly and are called by `run_module.py` as shown above;
`odds_fetcher.py`/`data_fetchers.py` both still live per §1/§2; `track_record/` 5 files still
present and now include the new `capture_closing_lines.py`.

## Legacy / removed modules (confirmed absent, not just "said to be absent")

Re-verified this pass via direct filesystem check:
```
$ find . -iname "auto_calibrator.py" -not -path "*/node_modules/*"
./modules/baseball_module/calibration/__pycache__/auto_calibrator.cpython-312.pyc
```
Only a stale bytecode file remains — matches `CLAUDE.md`/`CONTRACTS.md`, contradicts
`CLAUDE2.md` (see §1 REG-009/REG-033). `pitchers_regression.py` similarly reduced to a stale
`.pyc` only (per `CLAUDE.md`; not independently re-`find`-checked this pass, low risk given
the `auto_calibrator.py` spot check already validates the same claim pattern).

`odds_api.py`, `storage.py`, `ablation_calibrator.py`, `post_game.py`, `train_historical.py`,
`injuries_fetcher.py`, `nba_stats_fetcher.py`, `ufc_data_fetcher.py`, `modules/
football_module.py`, `modules/boxing_module.py`, `betting_ai/`, `bbets_ia_pro/` — per
`CONTRACTS.md`, confirmed removed 2026-07-06; not re-verified individually this pass (would
be pure repetition of that audit's own `find`/`grep` work with no new information expected).

## Experimental / not-yet-wired modules

- **F5 markets** — `odds_fetcher.py` MARKETS list omits `h2h_h1`/`totals_h1`/`spreads_h1`
  (REG-017); the normalization code exists and is exercised by tests
  (`test_odds_fetcher_f5_keys.py`, `test_ui_f5_odds_pipeline.py`, `test_f5_lambda.py`) using
  synthetic payloads, but the live bulk endpoint can never actually deliver this data. This is
  "wired for input that structurally cannot arrive yet" — a distinct category from dead code
  (the code path is live and tested, just permanently starved on the real network path until
  the per-event endpoint is implemented).
- **`historical_weather.py`** — fully built (Open-Meteo archive fetch/cache/parse), but its
  instantiation in `backtest_and_retrain.py` is commented out (per `CONTRACTS.md`
  `park_weather_engine.py` row: "Deliberately blind in the backtest... documented,
  intentional"). Re-verified this pass:
  ```
  backtest_and_retrain.py:63:  # from modules.baseball_module.hfa.historical_weather import HistoricalWeatherFetcher
  backtest_and_retrain.py:2627: # _weather_fetcher = HistoricalWeatherFetcher(...)
  ```
  Both references are commented out, confirming the backtest runs fully weather-blind by
  design (park factor still applies; only the temperature/wind/rain term is absent). This is
  a coverage gap worth naming precisely in §11: **the backtest's Brier/accuracy numbers
  contain zero weather signal, while live predictions do** — a live/backtest parity gap, not
  a leak (§5 has the parity classification).
- **NBA/UFC** (`basketball_module.py`, `ufc_module.py`) — self-contained, demo-data fallback,
  explicitly labeled `# NBA / UFC stubs (not yet connected to live data)` in `app.py:75`.
  Confirmed this session (odds_fetcher review) that fetching their odds via `SPORTS_KEYS` was
  therefore mostly wasted prior to this session's own trim.

## Duplicated logic

- **TTE composite-score formula** — was duplicated between `true_talent_engine.py` and
  `tte_pit_enrichment/tte_pit_adapter.py` (per this session's own in-code comment,
  true_talent_engine.py "2026-07-12: the composite-score math... moved to the shared
  `tte_formula` module... unifying it with the PIT-path adapter, which had drifted into an
  independently-maintained copy"). **Status: being actively de-duplicated in this exact
  working tree** — `tte_formula.py` (new, untracked) is the target of consolidation; checked
  for correctness of the merge in §5 (one real near-miss was already caught and fixed per the
  in-code comment: a metadata/log line had re-hardcoded the 0.50/0.30/0.20 weights right
  after the refactor was supposed to eliminate exactly that duplication).
- **Stadium name/coordinate tables** — three independent venue-keyed dictionaries
  (`park_weather_engine.STADIUM_DATABASE`, `data_fetchers.WeatherAPI.STADIUM_COORDS`,
  `historical_weather._STADIUM_COORDS`), serving different data (park factors vs. lat/lon for
  weather+travel-distance) but requiring synchronized alias updates whenever MLB renames a
  stadium — this desync is exactly what caused REG-015 (fixed this session, no regression
  test added — see `findings.csv`).
- **F5 field-naming** — three independent naming conventions for F5 fields across
  `odds_fetcher.py::_normalize_event()`, `odds_fetcher.py::get_best_odds_for_teams()`, and
  `core/value_detector.py`'s `GameOdds` canonical convention (REG-016). Two of the three are
  reconciled (`get_best_odds_for_teams`↔`GameOdds`, `ui/odds_loader.py`↔`_normalize_event`);
  the underlying `_normalize_event()` naming itself was deliberately left as-is (documented
  land mine, not yet a bug).

## Dead code

- 5 dead `data_fetchers.py` methods already removed (REG-014, re-verified absent this pass).
- `_l0_ratio()`'s `stage_factors_json` parameter (learning_engine.py:59-92, call sites
  :534/:536/:712) — accepted, never read in the function body. Not harmful (Python doesn't
  penalize an unused parameter), but it is a **concrete signal of an incompletely-cleaned-up
  revert** (see §1 REG-005/REG-006 postmortem — two earlier attempts *did* need this
  parameter to extract an L0 value from `stage_factors_json`; the final, kept version doesn't
  need it, and the 3 call sites were never simplified back down). Logged as **MATH-001** in
  `findings.csv` (Low severity, code-hygiene/confusion-risk, not a correctness bug).
- `weight_optimizer.py` — per `CLAUDE2.md` §"Baja prioridad" item 8: "archivo sin callers
  conocidos... revisar si es útil o eliminar." **Not independently re-verified this pass**
  (CLAUDE2.md is lower-trust per §1, but this specific claim is a "does this file have
  callers" question, not an architecture claim, so worth a cheap check):
  ```
  $ grep -rln "weight_optimizer" --include="*.py" . | grep -v __pycache__
  ./weight_optimizer.py
  ```
  Confirmed — **zero external callers**, the file only references itself. Logged as
  **INV-003** (Low severity, dead file candidate for removal).

## Modules applied more than once

**None found** in the live λ-adjustment chain (see verification note above). No evidence of
any engine being invoked twice, either directly or via a fallback path that re-runs the same
adjustment under a different name.
