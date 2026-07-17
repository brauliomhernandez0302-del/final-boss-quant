# 05 — Mathematical Audit (P1)

Per the audit's own priority rules, P1 sections receive solid but not exhaustive coverage;
depth was concentrated on files that changed in the current dirty working tree (higher
marginal risk than long-stable, previously-audited code).

## Combination method (confirmed, not re-derived from scratch)

`docs/AUDITORIA_MLB_2026-07.md` §4 already established, with line citations, that every
engine-combination point in `run_module.py` uses a delta-weighted blend
(`λ_out = λ_in × (1 + w×(ratio_crudo − 1))`), not geometric mean, and that this is
deliberate and consistent. Re-verified the pattern still holds in the current pipeline
ordering (§3's call graph) — not re-derived line-by-line again, since nothing in the dirty
diff touches `run_module.py`'s combination logic itself.

## New findings this pass

**MATH-001 — `_l0_ratio()`'s `stage_factors_json` parameter is dead (unused).**
`modules/baseball_module/calibration/learning_engine.py:59-92`, call sites `:534, :536, :712`.
```python
def _l0_ratio(actual_runs, lambda_final, stage_factors_json, is_home):
    """..."""
    if not lambda_final or lambda_final <= 0:
        return None
    numer = untruncate_home_runs(actual_runs) if is_home else actual_runs
    return numer / lambda_final
```
The parameter is accepted at all 3 call sites (which also modified their SQL SELECT to fetch
`stage_factors_json` purely to pass it in) but never read in the function body. Traced to a
real cause, not speculation: the in-code postmortem (§1 REG-006) documents that two earlier,
reverted attempts to change this function's denominator *did* need `stage_factors_json` (to
extract a logged L0 value); the final, kept version doesn't need it, and the call sites were
never simplified back down after the revert. **Severity: Low** (no behavior impact — Python
doesn't penalize an unused parameter) but it is exactly the kind of leftover that misleads a
future reader into thinking `stage_factors_json` still matters to this function's output.
Recommended fix: remove the parameter and the now-pointless `stage_factors_json` column
fetch at the 2 SQL call sites, OR add a one-line comment noting it's vestigial. Regression
test: none needed (removing an unused parameter cannot change behavior; a test would only
need to confirm the 3 call sites still compile/pass after the cleanup).

**MATH-002 — `tte_pit_adapter.py` shrinks barrel% using PA instead of batted-ball attempts
(already detailed in §4).** Cross-referenced here as a math-audit item: sample-size basis for
a Bayesian regression-to-mean should match the metric's own natural denominator (a per-BBE
rate needs a BBE-based `n`, not a PA-based one). Confirmed self-documented, deliberately
deferred, backtest-only (not live), Medium severity.

**MATH-003 (new, this audit) — `_UNIFORM_HOME_MULT`'s value may not have been re-derived
after the truncation-bias fix it was designed to coexist with.**
`modules/baseball_module/hfa/hfa_engine.py:15-34` documents that this constant (`0.028`) was
sized from a residual win-probability analysis on "the triple-clean, engine-hygiene-fixed
backtest" — but the file's own docstring explicitly flags: *"Needs re-validation against a
fresh backtest after landing — if the residual doesn't close to ~0 in both seasons, revert
and look at Platt's intercept instead."* Cross-referencing `reports/` directory timestamps
(§2): `round3_uniform_home_mult` (2026-07-11 14:09) precedes `round5_l0_reconstruction_fix`
(2026-07-11 22:35) and later rounds that touch the truncation-bias/dampening interaction
(REG-005/REG-006). This suggests the `0.028` constant itself was derived **before** the
truncation-bias reconciliation, and no evidence was found (within this pass's budget) of it
being explicitly re-derived from a fresh residual analysis afterward — only "kept" through
subsequent rounds.
**Important caveat, stated precisely to avoid overclaiming**: this is **not** evidence of a
current miscalibration. The project's own account (`CLAUDE.md`) describes the cited baseline
Brier/accuracy (0.24486/55.42%) as reflecting *all* fixes through round 10 together
end-to-end — so whatever residual imprecision `0.028` might carry is already priced into that
one validated number; a wrong `_UNIFORM_HOME_MULT` would have shown up as a worse Brier in the
final validated round, and it didn't regress. The finding is narrower and purely about
**methodology hygiene**: the constant's own justifying docstring asks for a re-validation step
that this audit found no direct evidence of having been performed as its own dedicated
check (as opposed to being implicitly carried through later end-to-end validation runs).
**Severity: Low** — recommended action: re-run the residual win-probability analysis the
docstring describes, now that truncation-correction is in place, and confirm `0.028` is still
the right sizing (or document explicitly that the end-to-end round-10 validation already
supersedes the need for a standalone re-check).

## Signs / units / normalization / home-away direction

`CLAUDE2.md`'s invariant table (§00, lower-trust doc) matches `CONTRACTS.md`'s canonical PASO
descriptions on direction (away pitcher→reduces λ_home, home pitcher→reduces λ_away, home
defense→reduces λ_away, away defense→reduces λ_home, HFA crowd→λ_home only [now disabled],
travel→λ_away only, park/umpire→symmetric) — no contradiction found between the two docs on
this specific point, so treated as reliable. Not independently re-derived engine-by-engine
this pass (would fully duplicate the already-thorough `docs/AUDIT_FINDINGS.md`
per-engine "Interface Contract" sections, table-of-contents-reviewed in §1 but not re-walked).

## Current/prior blending, sample-size basis

TTE's `blend_current_prior()` (now shared via `tte_formula.py`, §2/§3) uses
`prior_w = k/(k+PA_current)` — inverse-proportional Bayesian update, no arbitrary floor
(matches `CLAUDE.md`'s stated design). Re-verified this pass via the `tte_formula.py`/
`tte_pit_adapter.py` diff read in §4 — consistent between the live and PIT-adapter paths
after the unification refactor, **except** for the barrel%-specific shrinkage-basis
divergence already logged as MATH-002.

## Live/backtest parity — consolidated list (cross-referenced, not new)

1. Weather: backtest blind, live has real forecast (§3/§4/§11).
2. TTE barrel% shrinkage basis differs backtest-PIT vs. live (MATH-002).
3. Devig implementation duplicated (not yet inconsistent) live vs. backtest (ODDS-001, §10).
4. `_team_dict()`'s season-aggregate fields degrade to league averages in PIT backtest mode
   only; live always computes them for real (REG-004) — a fidelity difference, not a
   correctness bug, already accepted by the project.

## Duplicated implementations found across this audit (consolidated)

- TTE composite-score formula (being actively unified this session, §3).
- Stadium name/coordinate tables, 3 copies (REG-015, this session's own fix, §1).
- F5 field-naming, 3 conventions (REG-016).
- Devig multiplicative formula, 2 copies (ODDS-001, new this pass).

This is the single most recurring structural pattern found across the entire audit —
independently confirmed at least 4 separate times, spanning constants (`LG_XWOBA`, `LG_DER`,
`LG_BARREL_PA` — pre-existing, per `docs/FBQ_MASTER_BLUEPRINT.md` §1.1's own note),
dictionaries (stadium names), naming conventions (F5 fields), and now formulas (devig). The
project's own blueprint already proposes a structural fix (§1.1, Feature Store with a single
`features` table) but has explicitly deprioritized it pending real incidents. **This audit's
independent finding of a 4th and 5th instance (ODDS-001, plus the stadium-name gap this
session had to fix live) directly supports elevating that item's priority** — logged as a
technical-debt item in the final deliverables (§13), not a standalone severity-scored finding
in `findings.csv` (it's a pattern observation across already-logged findings, not a new
defect of its own).
