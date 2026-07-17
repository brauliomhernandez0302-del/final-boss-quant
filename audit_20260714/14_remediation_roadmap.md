# 14 — Recommended Remediation Roadmap

One change at a time, in this order, each locked in by its named regression test before
moving to the next — matching the project's own stated principle ("Un cambio a la vez.
Backtest después de cada uno.").

## Step 1 — CHRON-001 (High): protect `game_outcomes` provenance
**Why first**: highest severity, currently latent but trivially triggered by routine,
already-observed usage (a live app run followed by a backtest re-run touching season 2026).
Every day it stays unfixed is a day a routine backtest could silently destroy live-prediction
history with no error and no way to recover it after the fact.
**Change**: add a `source` provenance column to `game_outcomes`, set once at INSERT
(`record_prediction()`), never overwritten; gate `update_game_outcomes()` to respect it.
**Regression test**: insert via `record_prediction()` (source='live'), run
`update_game_outcomes()` against the same `game_pk`, assert the live values are preserved.
**Do NOT bundle with anything else** — this is a pure data-integrity fix with no model/λ
impact, so no backtest re-run is needed to validate it (matches the project's own documented
practice of not backtesting changes that provably can't move the number).

## Step 2 — LEARN-002 (Medium): minimal "is calibration alive" monitor
**Why second**: cheapest fix relative to the damage class it prevents (3 confirmed incidents
already). The project's own blueprint already specifies the exact metric.
**Change**: expose `% of live predictions with p_home != p_home_raw` and `% of rows with
ml_home_pin NOT NULL`, alert/log when either drops toward zero.
**Regression test**: a synthetic all-identity-calibration dataset should trip the alert.

## Step 3 — ODDS-001 + MATH-001 (Medium/Low, bundle — both are pure deduplication, zero
behavior change expected)
**Change**: `backtest_and_retrain.py` imports `remove_vig_multiplicative` from
`core/value_detector.py` instead of its own `_devig()`; remove `_l0_ratio()`'s dead
`stage_factors_json` parameter.
**Regression test**: assert byte-identical output before/after for a sample of odds pairs
(ODDS-001); existing 462-test suite already covers MATH-001's call sites.
**Backtest validation**: not required for MATH-001 (removing an unused parameter cannot
change output). For ODDS-001, a before/after backtest comparison is recommended anyway
purely to confirm the two implementations really were producing identical numbers all along
(should be a bit-for-bit no-op if this audit's math check was correct).

## Step 4 — FALL-001 + FALL-002 (Medium): stop conflating "missing" with "neutral"
**Change**: weather and travel-fatigue fallbacks return an explicit missing/estimated flag
instead of a value indistinguishable from a real measurement.
**Regression test**: synthetic unmapped-venue test cases for both, asserting the flag is set.
**Backtest validation**: not required (live-only code paths, per §7).

## Step 5 — MATH-002 (Medium) and MATH-003 (Low): PIT-adapter precision fixes
**Why last among the "real" fixes**: both are self-documented, deliberately deferred,
bounded-impact precision questions (not correctness bugs), and MATH-002 explicitly depends on
first fixing a separate PIT-cache `batted_ball_count` issue not otherwise in scope here.
**Change (MATH-002)**: once the batted-ball-count fix lands, switch `tte_pit_adapter.py`'s
barrel% shrinkage to use `attempts` instead of `pa`.
**Change (MATH-003)**: re-run the residual home-win-probability analysis with the truncation
fix already in place; confirm or re-derive `_UNIFORM_HOME_MULT`.
**Regression test**: both require a full before/after backtest comparison (these touch λ
directly) — follow the project's own established discipline (one change, full backtest,
compare Brier/accuracy before committing).

## Deliberately not scheduled by this roadmap

- Technical-debt items with no live defect (INV-001/002/003, DUP-001) — cheap, low-risk,
  do whenever convenient, no ordering constraint.
- REG-017/018/019 (F5 endpoint, bullpen trade edge case, gamePk match) — already correctly
  triaged and deferred by the project itself; this audit found no new evidence to reprioritize
  them.
- The Feature Store (`§1.1`) structural fix for the duplicated-constant pattern — this audit's
  finding (2 new instances) supports raising its priority, but implementing it is a multi-week
  architectural project, not a "step" in this list; recommend the project's own planning
  process revisit its scheduling with this audit's evidence in hand, rather than this roadmap
  prescribing a specific date.
