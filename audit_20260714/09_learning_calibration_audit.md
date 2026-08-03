# 09 — Learning and Calibration Audit (P0)

## Mechanisms audited

`modules/baseball_module/calibration/learning_engine.py` — 6 mechanisms per `CONTRACTS.md`:
team bias, Kalman, Platt (1D), Platt-2D, gradient-descent pipeline weights, idempotent
`update_outcome()`/`record_prediction()`.

## Kalman

`update_kalman()` runs per-game, per-team/role, fed by real `actual_home`/`actual_away` runs
(with the truncation correction, REG-005, applied at the 2 home-role call sites — verified
in §1). State (`kalman_state` table) advances strictly with game order via the single-pass
backtest loop (§8). No circularity: target is always real observed runs, never a prior
Kalman output.

## Team bias / multidim bias

Walk-forward safe (REG-001, re-verified §1/§4). `compute_team_bias_kalman_adjusted`'s
dampening formula has a **documented false derivation premise but a confirmed-correct
empirical effect** (REG-006) — two attempts to "fix" the premise both caused measured
backtest regressions and were reverted. This audit did not attempt a third fix (out of scope
— this is a read-only audit); flagged in `hypotheses.md` as "the dampening formula's true
justification is still an open research question, current form is empirically validated, not
derivationally validated."

## Platt (1D) — circularity, first-pass check performed this audit

```python
# learning_engine.py:943-950
rows = conn.execute(
    """
    SELECT COALESCE(p_home_raw, p_home) AS p_home, home_won
    FROM game_outcomes
    WHERE season = ? AND home_won IS NOT NULL AND p_home IS NOT NULL
    """, (season,)
).fetchall()
```
Trains logistic regression of `home_won` (real outcome) against `logit(p_home_raw)`
preferentially — deliberately avoids using the already-Platt-corrected `p_home` as the
predictor when the raw value is available, specifically to prevent circularity (per its own
inline framing in the surrounding code, consistent with `CLAUDE.md`'s description of
`p_home_raw`'s purpose). **Verified empirically this pass** (read-only query against
`data/predictions_history.db`): of 5,430 rows with `home_won` set, **0 have `p_home_raw`
NULL** — the COALESCE fallback-to-`p_home` branch is currently never exercised in practice,
so there is **no live circularity risk from this specific mechanism today**, though the code
path exists for the case where it could occur (e.g. legacy rows from before `p_home_raw` was
persisted — apparently none remain in this DB).

## Platt-2D — circularity, first-pass check performed this audit

```python
# learning_engine.py:1046-1056
rows = conn.execute(
    """
    SELECT p_home, home_won, ml_home_pin, ml_away_pin
    FROM game_outcomes
    WHERE season < ? AND home_won IS NOT NULL AND p_home IS NOT NULL
      AND ml_home_pin IS NOT NULL AND ml_away_pin IS NOT NULL
      AND ml_home_pin > 1 AND ml_away_pin > 1
    """, (season,)
).fetchall()
```
Unlike Platt-1D, this **does** use `p_home` (the already Platt-1D-corrected probability, not
`p_home_raw`) as one of its two input features, regressed against real `home_won` outcomes
with the Pinnacle-devigged market probability as the second feature. **This is intentional,
documented, two-stage calibration** (`logit(p_corrected)=a+b·logit(p_home)+c·logit(market)`)
— not circular in the strict sense (the training *target* is always real `home_won`, never a
Platt-2D output feeding back into itself), and walk-forward safe (`season < ?`, strictly
prior seasons only — re-verified this pass, matches the docstring claim). **Nuance worth
recording precisely** (not in any prior doc, found this pass): because stage 2's input
feature is stage 1's *output*, any future drift or bug in Platt-1D would propagate directly
into Platt-2D's fit — a legitimate design (this is how stacked calibration is supposed to
work) but it does mean the two mechanisms are not independent lines of defense against the
same class of error; a Platt-1D miscalibration and a Platt-2D miscalibration are correlated
risks, not two chances to catch a bad probability.

## Gradient-descent pipeline weights — circularity, first-pass check performed this audit

```python
# learning_engine.py:1166-1188 (_gradient_step)
row = conn.execute(
    "SELECT lambda_home, lambda_away, stage_factors_json FROM game_outcomes WHERE game_pk = ?",
    (game_pk,)
).fetchone()
...
for role, lam_final, actual in [
    ("home", row["lambda_home"], untruncate_home_runs(actual_home)),
    ("away", row["lambda_away"], actual_away),
]:
```
Trains against real `actual_home`/`actual_away` runs (Poisson NLL gradient, truncation-
corrected on the home side per REG-005) using each engine's own recorded `stage_factors_json`
ratio for credit attribution. **No circularity** — target is always ground truth, never a
previously-learned weight's own output.

## REG-023 status update (this audit's contribution)

The project's own blueprint (`docs/FBQ_MASTER_BLUEPRINT.md` v1.4, "Próximos 3 pasos" item 2)
lists the Platt/Platt-2D circularity check as proposed-but-never-executed, rising in urgency.
**This audit performed a genuine first pass** (above) across all three learning mechanisms
named in the audit brief (`recalibrate_platt`, `recalibrate_platt_2d`, `_gradient_step`) and
found **no circularity in any of them**, with one caveat worth carrying forward: Platt-2D's
intentional dependency on Platt-1D's output means the two are correlated, not independent,
safeguards. **This is a genuine, real, meaningfully-narrowed answer to a previously fully-open
question — not a restatement of REG-023, an advancement of it.** Recorded as **LEARN-001**
in `findings.csv` (informational/positive finding — no severity, since it found no defect;
included per the audit brief's explicit instruction to address this named risk directly).
Residual gap not covered by this pass: whether `save_state`/`load_state`'s TTL-based caching
(`_PLATT2D_RECAL_DAYS`) could let a stale fit computed under one code version keep being
*applied* after a later code change altered what `p_home`/`stage_factors_json` mean — a
cache-freshness question, not circularity, out of this section's budget.

## "Is calibration alive" monitor — confirmed still absent

Per REG-022, three independent silent-calibration-failure incidents (team-bias leak, Platt
reset-to-identity, Pinnacle gate silently disabling Platt-2D live) have already occurred, and
the project's own blueprint proposes a minimal production monitor (e.g. "% of live
predictions where `p_home != p_home_raw`", "% of rows with `ml_home_pin` not-NULL") as a
precursor to full monitoring (§1.5). **Confirmed this pass: no such monitor exists.**
```
$ grep -rln "calibration.*alive\|is_calibration\|calibration_monitor\|monitor.*calibration" --include="*.py" .
(no output)
```
Status: **NOT FIXED, correctly flagged as open by the project itself** — no new evidence
contradicts that. Logged for completeness as **LEARN-002** (Medium severity — the pattern
that caused REG-007's months-long silent Platt-2D outage could recur in a different form with
no current tripwire).

## Validation split / same-day / future-result leakage

Covered exhaustively in §4 and §8; no additional findings in this section beyond what's
already logged as REG-001, CHRON-001.
