# 11 — Statistical Evaluation Audit (P0)

Every verdict below derives strictly from §4 (leakage), §8 (chronology), §9
(learning/calibration), §10 (odds/market) — no new computation performed in this section,
per the audit's own rule ("Do not recompute performance in this phase").

## Brier / accuracy (backtest)

**Cited baseline**: `CLAUDE.md` — Brier 0.24486 / accuracy 55.42% (`--season 2024,2025
--use-full-pit`, 4,830 games, 462 tests).

**Trustworthy, with named caveats**:
1. **PIT-safe as run** — the cited command explicitly enables all 4 PIT flags, which §4
   confirms are the correct, walk-forward-safe path (team-bias leak fixed and re-verified,
   `_team_dict()` gated correctly, TTE/pitcher/bullpen/defense all PIT-mode). ✅
2. **Weather-blind** (§4/§3) — the backtest computes this number with `historical_weather.py`
   deliberately not wired in. Live predictions include a real weather term; the backtest
   number does not. This is a known, documented, *intentional* gap, but it does mean the
   backtest Brier is not measuring the exact same model live users see — a live/backtest
   parity gap, magnitude unmeasured. Caveat, not disqualifying.
3. **`tte_pit_adapter.py`'s barrel% under-shrinkage** (`MATH-002`, §4) — a real, quantifiable
   precision bug specifically in the PIT offense path used by `--use-full-pit`, meaning the
   cited baseline's offense λ is computed with a systematically over-confident barrel%
   signal relative to what live production would compute for the same team/date. Caveat, not
   disqualifying (self-documented, bounded, doesn't affect live).
4. **Any *other* invocation of `backtest_and_retrain.py` without the PIT flags** (§4) would
   produce a leak-contaminated number that looks superficially like a fresh "backtest" result
   but isn't walk-forward safe. This is a **process risk**, not a defect in the cited number
   itself — but this audit has no way to verify that every number in `reports/` (e.g.
   `reports/baseline/`) was produced with PIT flags on, and no self-declaring metadata was
   confirmed present in the report JSON (flagged as unverified in §4).

**Verdict: the current cited Brier/accuracy backtest number is trustworthy for model
evaluation, with the three named caveats above** — none of which are leaks, all of which are
already either documented by the project or newly surfaced by this audit as bounded,
non-critical gaps.

## Calibration (Platt / Platt-2D)

**Trustworthy as a walk-forward mechanism** (§9 — genuine first-pass circularity check found
none in `recalibrate_platt`, `recalibrate_platt_2d`, or `_gradient_step`). **Historically
unreliable in production for reasons unrelated to the math**: three independent silent
failure modes have occurred (team-bias leak, Platt reset-to-identity, Pinnacle-gate silently
disabling Platt-2D live all season — REG-001/003/007), all now fixed, none currently
monitored (`LEARN-002`, §9). **The backtest's Platt/Platt-2D fits are not affected by the
live Pinnacle-gate bug** (REG-007 — backtest always used the historical-odds path with
correct keys) — this is an important, explicit distinction: **backtest calibration was never
broken by REG-007; live calibration was, silently, until 2026-07-12.**

## Accuracy vs. ROI relationship

Unchanged from the project's own long-standing position (`CLAUDE.md`): accuracy and ROI are
not the same question, and this audit found nothing to move that conclusion either direction.
ROI (backtest) has been "thin and inconsistent" per every version of the project's own
running notes — this audit did not recompute it and takes no position on whether it will
improve, only on whether it is being *measured* soundly (yes, per §10, modulo the same
weather-blind/barrel%-precision caveats as Brier above, since ROI is derived from the same
underlying probabilities).

## ROI (live)

**Does not exist as a number today.** §10: `track_record.db::picks` has 0 rows. Any
"ROI" figure anyone might quote for *live* performance right now is either (a) actually the
backtest ROI mislabeled, or (b) fabricated. This audit found no live ROI computation
anywhere producing a non-trivial number, and confirms the project's own documentation
(blueprint "Próximos 3 pasos") already states this plainly. **Not a defect — a "not yet
measured" state**, correctly represented in the project's own docs.

## CLV

**Not yet computable** (§10) — 0 resolved picks. The instrument
(`capture_closing_lines.py`, `picks.clv_pct`/`closing_*` columns) is built and ready, per
project memory only recently unblocked (Pinnacle live feed fix, `ml_home_pin` persistence
fix, both closed this session per `CLAUDE.md`/blueprint v1.4). **Verdict: CLV is currently an
undefined quantity, correctly represented as such by the project. Not trustworthy, not
untrustworthy — simply doesn't exist yet.**

## Edge / Kelly

**Trustworthy at the mechanism level** — `kelly_criterion()` confirmed (per `CLAUDE.md`'s
architecture note, not re-derived from scratch this pass beyond what §2 already restates) to
be the single source of truth with a hard clip `[0.01, 0.15]`; REG-024/025/026 (point-
matching, real runline, push handling) are all independently re-verified fixes as of this
session's own commits, directly improving edge accuracy for totals/runline specifically.
**One structural, self-documented gap remains** (from the register, REG-046-equivalent —
actually this is blueprint §0.6's own item, not separately re-derived this pass): edge/Kelly
are not yet push-adjusted the same way EV already is, for whole-number totals — a known,
minor, conservative-direction gap.

## Overall trustworthiness table

| Metric | Trustworthy? | Basis |
|---|---|---|
| Backtest Brier/log-loss/calibration | **Yes, with 3 named caveats** (weather-blind, MATH-002 barrel-shrinkage, non-PIT-flag-run risk for *other* invocations) | §4, §8, §9 |
| Backtest accuracy | **Yes**, same caveats as Brier | §4, §8, §9 |
| Backtest ROI | **Yes, at the same confidence level as Brier** (derived from the same probabilities) — but "thin and inconsistent" per the project's own long-standing finding, unchanged by this audit | §10 |
| Live ROI | **Does not exist** — 0 resolved picks | §10 |
| CLV | **Does not exist** — 0 resolved picks; instrument ready | §10 |
| Edge / Kelly (mechanism) | **Yes** | §10, register |
| Calibration (as a walk-forward mechanism) | **Yes** (first-pass circularity check clean, §9) | §9 |
| Calibration (production reliability) | **Historically no** (3 silent-failure incidents, now fixed, still unmonitored) | §9, register |
