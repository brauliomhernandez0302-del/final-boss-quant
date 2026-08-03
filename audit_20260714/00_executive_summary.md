# 00 — Executive Summary

**Audit date**: 2026-07-14. **Scope**: BetMindex (FINAL BOSS QUANT G8+/G10), read-only,
working-tree state on `feature/point-in-time-rebuild` @ `acb8d8a` + dirty uncommitted changes
(full listing in `00_first_actions.md`). **Priority discipline followed**: P0 sections
(4/8/9/10/11) received full depth including new live verification (Monte Carlo determinism
micro-check, Platt/Platt-2D/gradient-descent circularity first-pass, direct `game_outcomes`
DB inspection); P1 sections (5/6/7) received solid, evidence-cited coverage; P2-level depth
(exhaustive per-file silent-exception sweep, full 26-file `advanced_pit_enrichment/`
re-walk) was explicitly not attempted and is named as such in `hypotheses.md`, per the
brief's own instruction to cut breadth rather than depth under time pressure.

## What this audit found that was genuinely new (not already in the project's own records)

1. **CHRON-001 (High)** — `game_outcomes` is shared, unprotected, between live-production
   writes and backtest overwrites. Empirically confirmed via direct DB read (563 of 615
   season-2026 rows already backtest-overwritten; 52 live rows still pending). Not a leak —
   a data-integrity/provenance gap that will silently destroy live-prediction history the
   next time a backtest run touches a reconciled 2026 game. See `08_chronology_audit.md`.
2. **MATH-002 (Medium)** — `tte_pit_adapter.py` shrinks barrel% using the wrong sample-size
   basis (PA instead of batted-ball-event attempts), already self-documented in-code as a
   known, deferred gap; this audit independently confirmed and classified it precisely.
3. **ODDS-001 (Medium) / DUP-001 (Low)** — two more instances of this codebase's single most
   recurring structural pattern (duplicated constants/formulas with no shared source of
   truth), on top of the 4 instances the project already knew about.
4. **FALL-001 / FALL-002 (Medium)** — weather and travel-fatigue fallbacks fabricate
   plausible-looking values indistinguishable from real measurements when a venue lookup
   misses — the exact general shape of the bug (REG-015) this session already had to fix once
   for 4 renamed stadiums, with no structural protection against the next occurrence.
5. **A real, first-pass Platt/Platt-2D/gradient-descent circularity check** (the project's
   own blueprint had listed this as "proposed, never executed," rising in urgency across two
   versions) — found no circularity in any of the three mechanisms, narrowing a
   long-standing open question rather than leaving it open. See `09_learning_calibration_audit.md`.
6. Minor: `MATH-001` (dead parameter), `LEARN-002` (confirmed-absent calibration-health
   monitor), `INV-001/002/003` (repo hygiene), and a direct contradiction between `CLAUDE2.md`
   and `CLAUDE.md`/`CONTRACTS.md` regarding whether `auto_calibrator.py` still exists
   (it does not — confirmed by direct filesystem check).

Everything else of substance in this audit **confirms, with fresh evidence, findings the
project had already made about itself** — the known-issues register (`01_known_issues_register.md`,
34 entries) shows a project with an unusually good track record of finding and fixing its own
real bugs (team-bias leak, Platt corruption, Pinnacle exclusion, point-mismatch odds shopping,
push-probability, bullpen roster misclassification, and more), each validated with a real
before/after backtest per its own stated discipline. This audit re-verified the most
consequential of those claims against the current code rather than taking them on faith.

## Final verdicts

**Overall project audit status: PARTIAL.**
Not FAIL: no active leakage was found in the project's own cited baseline configuration, the
test suite is fully green (462/462), and the model/lambda pipeline itself is in good, well-
documented shape. Not a clean PASS: one High-severity, currently-latent data-integrity gap
(CHRON-001) has no protection today and is easily triggered by completely routine future
usage, and several Medium-severity gaps (ODDS-001, FALL-001/002, LEARN-002, MATH-002) are
real and unaddressed, even though each is individually bounded and mostly self-documented.

**"The current backtest is trustworthy for model evaluation"** — for the project's own cited
command (`--season 2024,2025 --use-full-pit`), **yes**, with three named, bounded caveats:
it is weather-blind by design (§4), its PIT-mode offense path has a known barrel%-shrinkage
imprecision (MATH-002), and this audit could not confirm that report files self-declare which
PIT flags produced them (a process-hygiene risk for any *other* invocation, not a defect in
the cited number itself).

**"The current ROI is trustworthy"** — the **backtest** ROI is trustworthy at the same
confidence level as Brier/accuracy (it is derived from the same probabilities), but remains
"thin and inconsistent" per the project's own long-standing, unchanged finding — this audit
did not recompute it and takes no new position on profitability. **No live ROI number exists
yet** (`track_record.db::picks` has 0 rows) — this is correctly represented by the project as
"not yet measured," not a false claim needing correction.

**"The live model is affected by the identified findings"** — **yes, partially**: CHRON-001
(live prediction history at risk on the next qualifying backtest run), FALL-001/FALL-002
(live-only weather/travel fallback ambiguity), and LEARN-002 (no live calibration-health
monitor) are all live-path issues. The live model is **not** affected by MATH-002 (backtest-
PIT-only) and is only theoretically affected by ODDS-001/MATH-003 (currently consistent /
already backtested end-to-end respectively). **None of the findings in this audit are
leakage or probability-correctness bugs in the live model** — they are data-integrity,
fallback-design, and monitoring gaps.

## Roadmap

See `14_remediation_roadmap.md` for the full sequenced plan. Order: **CHRON-001 → LEARN-002 →
(ODDS-001 + MATH-001, bundled) → (FALL-001 + FALL-002) → (MATH-002, MATH-003)**, each gated
by its own named regression test, matching the project's own "one change at a time, backtest
after each" discipline.

## Deliverables index

| File | Content |
|---|---|
| `00_first_actions.md` | Exact git state, dirty-tree listing, docs read as declared intent |
| `01_known_issues_register.md` | 34 previously-documented issues, status re-verified against current code |
| `02_project_inventory.md` | Entry points, pipeline, DB tables (read live), scripts/caches |
| `03_call_graph.md` | Full raw-data→stake chain, active/legacy/experimental/duplicated/dead modules |
| `04_leakage_audit.md` | PIT-safety classification table, MATH-002 detail |
| `05_math_audit.md` | Formula/constant review, MATH-001/002/003 |
| `06_double_counting_audit.md` | Overlap matrix, truncation-fix × uniform-home-mult interaction check |
| `07_fallback_audit.md` | Per-layer fallback inventory, FALL-001/002 |
| `08_chronology_audit.md` | Ordering, doubleheaders, seeds (live-verified), **CHRON-001** |
| `09_learning_calibration_audit.md` | Kalman/Platt/Platt-2D/gradient-descent, live circularity check |
| `10_odds_market_audit.md` | Devig, CLV/ROI instrument status, ODDS-001 |
| `11_statistical_evaluation_audit.md` | Trustworthiness verdicts per metric |
| `12_operational_audit.md` | Test suite (462/462), retries/locks/logs/resumability |
| `13_technical_debt.md` | Duplicated-logic pattern (6 instances), doc drift, dead code |
| `14_remediation_roadmap.md` | Sequenced fix order with regression tests |
| `findings.csv` | Machine-readable finding list (11 rows) |
| `hypotheses.md` | 8 unproven suspicions, explicitly excluded from findings |
