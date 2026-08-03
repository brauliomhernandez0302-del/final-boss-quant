# 06 — Double-Counting Audit (P1)

## Historical double-counting, confirmed fixed

- **AutoCalibrator** (offense_mult + defense_mult + rest + forma, triple-counted against
  pitcher/bullpen/defense engines) — deleted entirely, confirmed absent via direct filesystem
  check this pass (§1 REG-009/§3). Only a stale `.pyc` remains.
- **Pitcher Engine `kbb_mult`** vs. SIERA/xFIP already encoding K%-BB% — conditional gate,
  documented in `CONTRACTS.md`, not re-derived line-by-line this pass (REG-010).
- **Bullpen Engine** K-BB% reweighting when team SIERA available, mirrors the above
  (REG-011).
- **HFA crowd boost** (removed, confirmed noise) vs. park factor — architecturally separated
  into 2 engines specifically to prevent this (`park_weather_engine.py` = symmetric park/
  weather, `hfa_engine.py` = asymmetric travel + the new uniform home-win term only) —
  re-verified this pass via direct read of `hfa_engine.py`'s module docstring (§5), which
  explicitly states "NOT in scope (handled by ParkWeatherEngine, PASO 5): Park run-environment
  factor... Weather...".

## Interaction checked this pass: truncation-bias fix (REG-005) × `_UNIFORM_HOME_MULT` (REG-013)

These are the two most recent home-side λ/run corrections landed in the same session
(`CLAUDE.md` "Actualización 2026-07-11/12," items 4 and 5), and are the most plausible
candidate for a *new* double-count given they both touch "home side gets a boost/correction."
**Verified this pass they operate on structurally different things, not the same effect
counted twice**:
- `_UNIFORM_HOME_MULT` (`hfa_engine.py:52,97-98`) multiplies **λ_home directly, every
  prediction**, in PASO 7 of the live/backtest λ-adjustment chain.
- `untruncate_home_runs()`/`_HOME_RUNS_TRUNCATION_FACTOR` (`learning_engine.py:44-121`) is
  applied **only to the ground-truth `actual_home_runs` values fed to the learners**
  (Kalman, team-bias ratio, gradient-descent home role) — never to λ itself, never to the
  Monte Carlo simulation, never to the stored `actual_home_runs` column (kept as true ground
  truth for Brier/accuracy grading, per the in-code comment).

These affect **different stages of the same pipeline** (a permanent multiplicative shift to
the model's own λ output, vs. a correction to what the learners are told "really happened" so
they don't fight the first fix) — not a double-count of the same statistical effect. The
project's own documentation already frames these as sequenced/reconciled (`CLAUDE.md` item 5:
"...los learners... entrenaban contra la observación truncada y peleaban contra el fix #4" —
explicitly describing why #5 had to follow #4, not that they duplicate each other).
**Verdict: not a double-count.** The only open question is the narrower, already-logged
**MATH-003** (whether `_UNIFORM_HOME_MULT`'s specific numeric value was re-derived after #5
landed, or just carried through — a calibration-freshness question, not a double-counting
one).

## Overlap matrix (qualitative, cross-referenced against each engine's documented scope)

| Effect | Owning engine | Also touched by | Overlap risk |
|---|---|---|---|
| Starter pitching skill | Pitcher Engine | Bullpen (own SIERA/ERA), generic team pitching (`_team_dict`, backtest fallback only) | Low — `_team_dict()`'s season-aggregate fields are explicitly a *fallback for missing per-pitcher data*, gated by PIT flags (REG-004), not a parallel independent signal applied on top of a present starter signal |
| Bullpen quality | Bullpen Engine | — | Low — REG-029 (roster-vs-relievers-only fix) already resolved the one real bug found here this session |
| Defense (fielding) | Defensive Efficiency Engine | — | Low — architecturally isolated from pitching per its own docstring ("fielding puro, sin pitching") |
| Park run environment | Park+Weather Engine | — | Low — the one historical overlap (HFA used to also apply park factor) was removed (FIX C2, register) |
| Weather | Park+Weather Engine | — | N/A live; absent entirely in backtest (not an overlap, an omission, §4/§5) |
| Fatigue/rest (B2B) | Contextual Engine | HFA's travel-fatigue (different mechanism: overnight travel/timezone vs. simple B2B) | Low — `CONTRACTS.md` explicitly separates "Contextual Engine: rest/B2B only" from "HFA Engine: away-team travel fatigue only (miles + timezone)"; these are distinct real-world phenomena (playing yesterday vs. having flown across time zones), not two engines pricing the same thing |
| Travel | HFA Engine | Contextual Engine's B2B (see above) | Low, same reasoning |
| Calibration / bias | Team-bias + Kalman + Platt(-2D) | — | **Correlated by design, not duplicated** — Platt-2D intentionally chains off Platt-1D's output (§9); this is stacked calibration, a legitimate technique, not a double-count, though it does mean the two share failure modes rather than being independent checks (already noted in §9). |

## Verdict

No new double-counting defects found this pass beyond what the project's own history
already fixed. The one interaction worth a second look (truncation fix × uniform home mult)
was checked directly and found to be correctly sequenced, not duplicated — narrowed to a
calibration-freshness question (MATH-003) rather than a double-count.
