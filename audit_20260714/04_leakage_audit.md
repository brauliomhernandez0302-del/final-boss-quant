# 04 — Point-in-Time and Leakage Audit (P0)

## Classification table

| Input | Source | Cache | Cutoff support | Full-season fallback? | Current-roster? | Sees future? | Classification |
|---|---|---|---|---|---|---|---|
| Team bias / multidim bias | `learning_engine.py::compute_team_bias`/`compute_multidim_bias` | `ml_state`, 24h TTL in live mode | **Yes** — `before_date` param, threaded to SQL WHERE (learning_engine.py:481,521-523,639,663-675,696-698) | No (walk-forward query) | N/A | **No — fixed 2026-07-08** | **PIT-safe** (REG-001, re-verified this pass) |
| Kalman offense/defense state | `learning_engine.py::update_kalman`, fed by `record_prediction`/`update_outcome` | `kalman_state` table | Implicit — only updates after a game resolves, in game order (see §8) | N/A | N/A | No, by construction (state machine only moves forward in time as outcomes arrive) | **PIT-safe**, contingent on §8's chronology (same-day/doubleheader ordering) holding |
| True Talent Engine (live path) | `offense/true_talent_engine.py`, Statcast/FanGraphs | `.cache/tte_*` 6-24h TTL | Uses **current-season-to-date** Statcast, park-neutral | Blends with prior-season baseline (`prior_w=1000/(1000+PA)`) | Yes — lineup-filtered when confirmed | No (live mode; "future" is meaningless for live) | **PIT-safe by construction** (live) |
| True Talent Engine (backtest, non-PIT default) | same file, called by `backtest_and_retrain.py` without PIT flags | same | **No explicit walk-forward cutoff found in the default path** — uses whatever the live fetchers return "as of" the API call time during the backtest run, i.e. **current, not historical, Statcast snapshots** for games in the past | — | uses whatever roster the API returns **today**, not as of the historical game date | **Yes, unless `--use-team-tte-pit` is passed** | **UNSAFE by default; PIT-safe only under `--use-team-tte-pit`/`--use-full-pit`** — see note below |
| True Talent Engine (backtest, `--use-team-tte-pit`) | `advanced_pit_enrichment/tte_pit_adapter.py` + snapshot builders | `pit_cache_*.db` | Yes — snapshot builders are explicitly "as of" a date | No look-ahead (walk-forward snapshots) | Uses the as-of-date roster | No | **PIT-safe**, with one known math imprecision (barrel% shrinkage basis, `MATH-002` below) — not a leak, a precision gap |
| Pitcher stats | `data_fetchers.py::get_pitcher_stats_full_fallback` (live); PIT snapshot builders (backtest PIT mode) | `.cache/`, `pit_cache_pitcher.db` | Yes in PIT mode (per `CONTRACTS.md`, "PIT coverage 96.5%+3.1%+0.9%" cascading fallback) | Falls back to league averages, not full-season leak, when PIT snapshot missing (per `CLAUDE.md`/CONTRACTS.md) | N/A | Live: no. Backtest non-PIT: **yes, same class of risk as TTE above** | **PIT-safe under `--experimental-pitcher-pit-mode`; unsafe by default** |
| Bullpen | `context_engine/bullpen_engine.py` (live), PIT builder (backtest PIT mode) | `.cache/bp_*`, PIT cache | Yes in PIT mode ("PIT coverage 100%" per CONTRACTS.md) | — | Real per-team boxscore innings, `ip_last_3_days` | Backtest non-PIT: same class of risk | **PIT-safe under `--use-bullpen-pit`; unsafe by default** |
| Defense (DER/OAA) | `context_engine/defensive_efficiency_engine.py`, `team_defense_pit_builder.py` (modified +19 lines this session) | PIT cache | Yes in PIT mode | `use_team_full_season_defense` gate confirmed (REG-004, re-verified, backtest_and_retrain.py:533-549) | — | Backtest non-PIT: same risk | **PIT-safe under `--use-defense-pit`; unsafe by default, but explicitly gated and documented** |
| `_team_dict()`'s `team_era`/`team_whip`/`runs_allowed_per_game` | `backtest_and_retrain.py:533-549` | — | Gated by `use_team_full_season_pitching_base` (`= not args.use_defense_pit`) | Yes when PIT mode is OFF (by design — this is the DEFAULT unless `--use-defense-pit` is passed) | — | Falls back to league-average constants in PIT mode, not a leak, a fidelity trade-off | **PIT-safe under `--use-defense-pit`; a genuine season-aggregate FIDELITY reduction (not a leak) otherwise** — matches REG-004 exactly |
| Weather (live) | `data_fetchers.WeatherAPI`, OpenWeather 5-day/3-hour forecast | none (live call each time) | N/A — always "now" relative to game time | N/A | N/A | N/A (live) | **PIT-safe by construction** (forecast, not backfill) |
| Weather (backtest) | `historical_weather.py`, Open-Meteo archive — **instantiation commented out** (backtest_and_retrain.py:63,2627, re-verified this pass) | `.cache/historical_weather_cache.json` | Would be exact-date archive if wired | N/A | N/A | No (archive data, strictly past) | **Not applicable — deliberately not fetched.** The backtest runs 100% weather-blind. This is not a leak; it is a **live/backtest parity gap** (§5 MATH classification) that could bias Brier/accuracy comparisons between live and backtest in either direction, magnitude unmeasured. |
| Odds / Pinnacle fair line | `odds_fetcher.py` (live), `fetch_historical_odds.py` (backtest) | `.cache/odds_last.json` (live) | Backtest uses the historical-odds script, which independently always had the correct `"pinnacle"` key (REG-007) — **backtest was never affected** by the live Pinnacle-exclusion bug | N/A | N/A | No | **PIT-safe (backtest); live path was broken until 2026-07-12, now fixed (REG-007)** |
| `game_outcomes` table itself (chronology/provenance) | shared by live `record_prediction()`/`update_outcome()` and backtest `update_game_outcomes()` | `predictions_history.db` | **No isolation between live-authored rows and backtest-authored overwrites** | N/A | N/A | Not a look-ahead leak (backtest still reads only pre-cutoff data for whichever game it recomputes) | **Not a leakage classification — a data-integrity/provenance gap, detailed fully in §8 as CHRON-001.** Flagged here because it was discovered during this section's investigation. |

## Narrative: the two most important findings

### 1. Non-PIT backtest paths are, by the project's own design, NOT point-in-time safe — this is documented, not hidden

`CLAUDE.md`'s own cited baseline command is `--season 2024,2025 --use-full-pit`, which
enables all 4 PIT modes together. Every "unsafe... unless a flag is passed" row above is
therefore **not a live bug in the cited baseline** — the project already runs its official
number with the safe flags on. The risk is entirely about **any other invocation** of
`backtest_and_retrain.py` without those flags (e.g. `reports/baseline/`,
`reports/full_pit/` in this session's own `reports/` directory suggest multiple historical
runs existed both with and without full PIT — a reader who picks up an old report from
`reports/baseline/` rather than a `*_pit*`-labeled one and treats its Brier as representative
of the current PIT-safe pipeline would be citing a leak-contaminated number without knowing
it). **Recommendation** (not implemented, per audit rules): the report filename/JSON payload
should self-declare which PIT flags were active, so a stale or non-PIT report can never be
silently mistaken for the current baseline. Not independently confirmed whether the JSON
report already includes this (would require reading `generate_report()`'s full field list,
out of this pass's remaining budget — flagged in `hypotheses.md`).

### 2. `tte_pit_adapter.py`'s barrel% Bayesian-shrinkage basis is wrong, but deliberately deferred, and only affects PIT-mode backtests, not live

Evidence (`modules/baseball_module/advanced_pit_enrichment/tte_pit_adapter.py`, current
working-tree diff, unified diff hunk around the `_compute_lambda` rewrite):

```python
# NOTE (2026-07-12, tte_formula unification): barrel% is regressed here
# using `pa` as the Bayesian sample size for ALL four metrics, including
# barrel — unlike the live engine, which uses `attempts` (batted-ball-
# events) specifically for barrel, since it's fundamentally a per-BBE
# rate. This under-shrinks barrel% here (PA ~1.47x attempts). Known,
# deliberately NOT fixed in this refactor...
xwoba_reg = _shared_regress(xwoba_cur, lg_xwoba, pa, K_XWOBA)
barrel_reg = _shared_regress(barrel_cur, lg_barrel_pa, pa, K_BARREL)
```

This is **not a temporal leak** (no future data involved) — it is a statistical-precision
bug: using plate appearances (≈1.47× batted-ball-event count) as the shrinkage sample size
for a rate that is fundamentally per-batted-ball-event makes the PIT adapter's barrel% signal
regress toward league-average *less* than it should for a given amount of real evidence,
i.e. it **overweights small/noisy barrel% samples** relative to the live engine's equivalent
computation. This is a genuine **live/backtest parity gap**: the officially-cited baseline
(`--use-full-pit`) computes offense λ with a systematically different (over-confident)
barrel% signal than live production does. Logged as **MATH-002** (Medium severity — real,
quantifiable, but self-documented, deliberately deferred pending a separate PIT-cache fix,
and does not affect live predictions at all).

## Verdict per classification scheme

- **PIT-safe**: team bias/multidim bias (REG-001), Kalman, TTE/pitcher/bullpen/defense **when
  run with their respective `--use-*-pit` flags** (which the project's own cited baseline
  does), `_team_dict()`'s season-aggregate fields (gated), odds/Pinnacle (both paths).
- **PIT-safe with explicit fallback**: `_team_dict()`'s three fields when PIT mode is off
  (falls back to league-average constants, not a leak — a fidelity trade-off, matches the
  project's own prior conclusion in `docs/AUDITORIA_MLB_2026-07.md` §3.1).
- **Unsafe**: TTE/pitcher/bullpen/defense **when PIT flags are NOT passed** — real, but
  scoped entirely to whichever specific invocation omits the flags; the project's own current
  baseline does not have this problem.
- **Not verifiable in this pass**: whether `advanced_pit_enrichment/`'s remaining ~24 files
  (beyond the 2 spot-checked here, `tte_pit_adapter.py` and `team_defense_pit_builder.py`)
  still hold the "1 gap found and fixed" clean bill of health the project's own blueprint
  claims for its 2026-07-06→09 formal audit (§0.1 exit criterion) — not re-walked file by
  file this pass given P0 budget was spent going deep on the two files that changed in this
  session's dirty working tree instead (higher marginal value: new code is more likely to
  hide a new bug than previously-audited, unchanged code).
