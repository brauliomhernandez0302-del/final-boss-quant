# 13 — Technical Debt

## The dominant pattern: duplicated constants/logic with no single source of truth

Found independently, at least 6 times across this audit and the project's own prior history:

1. `LG_XWOBA`/`LG_DER`/`LG_BARREL_PA` hardcoded identically-but-independently in 4 files
   (pre-existing, per `docs/FBQ_MASTER_BLUEPRINT.md` §1.1's own note).
2. Stadium name/coordinate dictionaries, 3 independent copies (REG-015, fixed this session
   with no regression test added).
3. F5 field-naming, 3 independent conventions (REG-016, 2 of 3 reconciled).
4. TTE composite-score formula, 2 independent copies (being actively unified this session via
   `tte_formula.py`, one near-miss already caught mid-refactor per its own in-code comment).
5. Devig multiplicative formula, 2 independent copies (ODDS-001, new this audit).
6. Travel-distance calculation, 2 independent implementations, possibly intentional
   (DUP-001, new this audit, unconfirmed whether deliberate).

The project's own blueprint (§1.1, Feature Store with a single `features(entity_id,
feature_name, value, as_of_date, computed_at)` table read by both backtest and production)
already proposes the structural fix, but has explicitly deprioritized it pending "a real
incident" as the trigger. **This audit's independent discovery of instances 5 and 6, on top
of the already-known 4, is itself evidence that the incident threshold has likely already
been crossed** — recommend re-scoring §1.1's priority given the rate at which this exact bug
class keeps recurring (roughly once per session, across unrelated files, for months).

## Documentation drift risk, again

`CLAUDE2.md` (untracked, undated relative to `CLAUDE.md`) reintroduces the exact class of
staleness that `CONTRACTS.md`'s full 2026-07-06 rewrite was meant to fix — it describes
`auto_calibrator.py` as still present (with a `tte_active` skip-gate), directly contradicted
by a direct filesystem check this audit performed (`find` confirms only a stale `.pyc`
remains). Recommend reconciling or removing `CLAUDE2.md` rather than letting two
authoritative-sounding docs disagree on a basic "does this file exist" question.

## Dead / orphaned code and files

- `weight_optimizer.py` — zero callers (INV-003).
- `_l0_ratio()`'s unused `stage_factors_json` parameter (MATH-001).
- `.codex/` (INV-001), stale DB files (INV-002) — repo hygiene, not code debt per se.

## Deliberately deferred, self-documented gaps (carried forward from the project's own
records, not re-litigated by this audit)

- F5 markets structurally unreachable via the bulk odds endpoint (REG-017) — needs the
  per-event endpoint, known, correctly gated so nothing downstream misbehaves in the meantime.
- Bullpen reliever-role misclassification for mid-season trades (REG-018) — rare edge case.
- `learning_engine.py`'s `abstractGameState=="Final"` check takes the first response game
  without an explicit `gamePk` match (REG-019) — low risk per the project's own assessment,
  not independently re-verified this pass.
- Edge/Kelly not yet push-adjusted for whole-number totals the same way EV already is
  (blueprint §0.6) — conservative-direction, minor.

## Monitoring gap

No "is calibration alive" production monitor exists (LEARN-002) despite three independent,
confirmed silent-calibration-failure incidents to date. This is the single highest-leverage
piece of debt found in this audit relative to its implementation cost — the project's own
blueprint already specifies exactly what to build.
