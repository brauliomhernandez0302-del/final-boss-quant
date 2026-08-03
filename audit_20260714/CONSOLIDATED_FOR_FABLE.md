# BETMINDEX AUDIT — 2026-07-14 — Documento consolidado para revisión de Fable

---


<!-- ============================================================ -->
<!-- FILE: 00_executive_summary.md -->
<!-- ============================================================ -->

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


<!-- ============================================================ -->
<!-- FILE: 00_first_actions.md -->
<!-- ============================================================ -->

# 00 — First actions (pre-section-1 state capture)

**Audit date:** 2026-07-14
**Audit directory:** `audit_20260714/` (only write location used in this audit)

## Exact commit / branch state

```
HEAD commit:  acb8d8aa10c6aa88f243908634613493bc1b0ec4
Branch:       feature/point-in-time-rebuild
Last commit:  "docs: update master blueprint to v1.4 with this session's odds/value-detection audit"
              (Braulio Matos, 2026-07-14 14:06:03 -0700)
```

**⚠️ Working tree is DIRTY.** This audit reflects the working tree as it stands right now,
NOT the last commit. Every finding below that touches a modified/untracked file must be read
as "true of the code on disk," which differs from what `git show HEAD:<file>` would show.

## Full uncommitted change list (`git status` / `git diff HEAD --stat`)

### Modified (tracked, unstaged)

| File | +/- | Note |
|---|---|---|
| `CLAUDE.md` | +20/-? | doc update |
| `backtest_and_retrain.py` | +190/-? | large — likely PIT-mode / chronology-relevant changes, audited in §8 |
| `data_fetchers.py` | +11/-0 | stadium-name alias fix (this session, prior to audit request — see §1 register) |
| `modules/baseball_module/advanced_pit_enrichment/team_defense_pit_builder.py` | +19/-? | PIT builder change |
| `modules/baseball_module/advanced_pit_enrichment/team_defense_prior_baseline.py` | +2/-1 | small |
| `modules/baseball_module/advanced_pit_enrichment/tte_pit_adapter.py` | +109/-? | large — audited in §4/§6 |
| `modules/baseball_module/calibration/learning_engine.py` | +152/-? | large — audited in §9 (leakage-critical file) |
| `modules/baseball_module/hfa/hfa_engine.py` | +46/-? | audited in §5/§6 |
| `modules/baseball_module/hfa/historical_weather.py` | +7/-0 | stadium-name alias fix (this session) |
| `modules/baseball_module/offense/true_talent_engine.py` | +92/-? | refactor to `tte_formula.py` — audited in §5 |
| `odds_fetcher.py` | +10/-? | `SPORTS_KEYS` trim (this session) |
| `tests/test_hfa_pipeline.py` | +54/-? | test changes tracking hfa_engine.py |
| `tests/test_learning_engine.py` | +17/-? | test changes tracking learning_engine.py |
| `tests/test_validated_fixes.py` | +23/-? | test changes |
| `track_record/db.py` | +120/-0 | new functionality, not yet reviewed this session |

### Untracked (new files)

| Path | Note |
|---|---|
| `.codex/` | **not a project artifact** — this is a separate CLI tool's home directory (~197MB, contains its own SQLite state/logs/auth.json). Not part of BetMindex; excluded from this audit's scope. Flagged as repo hygiene: should not sit inside the repo root if it isn't meant to be tracked (no `.gitignore` entry confirmed — see INV finding). |
| `CLAUDE2.md` | draft doc, "G10 Ultra Pro" — describes an architecture that partially conflicts with `CLAUDE.md`'s "G8+" description (e.g. claims `AutoCalibrator` PASO 1 still exists with a `tte_active` skip — CLAUDE.md says `auto_calibrator.py` **does not exist at all**, superseded by Kalman+bias). Treated as declared intent but internally inconsistent with CLAUDE.md — flagged in §1. |
| `modules/baseball_module/offense/tte_formula.py` | new shared formula module (extracted from true_talent_engine.py + tte_pit_adapter.py per this session's own stated goal) |
| `reports/` | ad-hoc backtest run logs/JSON from prior sessions (round1..round10, baseline, full_pit, offense_rebuild) — historical evidence, read-only inputs for this audit, not reproduced |
| `scripts/build_offense_savant_rolling_incremental.py` | new PIT cache builder script |
| `tests/test_track_record_closing_lines.py` | new test file |
| `track_record/capture_closing_lines.py` | new CLV capture module (referenced in project memory / blueprint 1.4) |

## Documentation read as declared intent

- `CLAUDE.md` (118 lines) — primary source of truth, describes pipeline PASO 1-9, states current baseline **Brier 0.24486 / accuracy 55.42%**, documents 9 bugs fixed 2026-07-11/12.
- `CLAUDE2.md` (268 lines, untracked) — a **draft/alternate** description ("G10 Ultra Pro") that partly contradicts CLAUDE.md (see above). Internally dated to an earlier session (references "462-test"/"448-test" eras loosely, describes AutoCalibrator as still-present-but-neutered — CONTRACTS.md and CLAUDE.md both say it was deleted). Treated as lower-trust than CLAUDE.md where they conflict.
- `CONTRACTS.md` (339 lines) — rewritten 2026-07-06, "north-star" system map, verified line-by-line against code as of that date. Most reliable single inventory document; used heavily in §2/§3.
- `docs/FBQ_MASTER_BLUEPRINT.md` (158 lines, v1.4, 2026-07-12/14) — roadmap + running changelog of every major fix, with severity/impact notes. Primary source for §1 (known-issues register).
- `docs/AUDITORIA_MLB_2026-07.md` (165 lines, 2026-07-06) — prior formal audit, read-only, self-corrected once (Fable re-review). Its own baseline (Brier 0.24242) is explicitly marked superseded.
- `docs/AUDIT_FINDINGS.md` (2766 lines) — oldest, most granular audit trail (2026-05 through -06 era, pre-PIT-rebuild). Table of contents reviewed; treated as historical known-issues source, not re-verified line-by-line in this pass (see §1 methodology note) because CONTRACTS.md/blueprint already re-verified its live conclusions as of 2026-07-06/14.
- Test suite layout: `tests/` — 43 `.py` files present on disk right now (CONTRACTS.md says 39 files/448 tests as of 2026-07-06; file count has grown since — reconciled against actual `pytest` run in §12).

## Note on scope vs. this session's own edits

Two of the uncommitted diffs (`data_fetchers.py`'s stadium-alias fix, `odds_fetcher.py`'s `SPORTS_KEYS` trim) were made earlier in this same conversation, before this audit was requested. They are included in the working-tree snapshot like any other uncommitted change and are evaluated on the same terms — no special treatment.


<!-- ============================================================ -->
<!-- FILE: 01_known_issues_register.md -->
<!-- ============================================================ -->

# 01 — Known-Issues Register

Methodology: built from `CLAUDE.md`, `CLAUDE2.md` (lower trust, see 00), `CONTRACTS.md`,
`docs/FBQ_MASTER_BLUEPRINT.md` v1.4, `docs/AUDITORIA_MLB_2026-07.md`, and
`docs/AUDIT_FINDINGS.md` (table of contents + targeted reads; this 2,766-line file predates
the PIT rebuild and its live conclusions were already re-verified by CONTRACTS.md/blueprint
as of 2026-07-06/14, so it is treated as historical provenance, not re-walked line-by-line).
Where marked "**re-verified this pass**," I independently confirmed the claim against the
current working-tree code (not just trusting the doc). Everything else is "as documented,"
carried at the confidence the source document itself claims.

IDs here are register entries (`REG-NNN`), not audit findings — new findings in §4-12 that
restate one of these reference it by `REG-NNN` instead of being re-reported.

---

## A. Leakage (P0)

**REG-001 — `LearningEngine.compute_team_bias`/`compute_multidim_bias` season-wide look-ahead leak.**
Documented: `docs/AUDITORIA_MLB_2026-07.md` update note, `docs/FBQ_MASTER_BLUEPRINT.md` §0.1
("el hallazgo más severo de todo FASE 0"), `CLAUDE.md` "Estado actual". Bias multiplied into
λ during backtest saw the team's full-season results, including games after the one being
predicted. Fixed with a `before_date` walk-forward parameter (2026-07-08); regression test
`tests/test_anti_leakage_opening_day_2024.py`.
**Status: FIXED — re-verified this pass.** `before_date: Optional[str] = None` present in
`compute_team_bias` (learning_engine.py:481), `compute_team_bias_kalman_adjusted` (:558),
`compute_multidim_bias` (:639), threaded into the SQL `WHERE` clause (:521-523, :696-698).
`tests/test_anti_leakage_opening_day_2024.py` is in the current 462-green test run (§12).

**REG-002 — Monte Carlo non-determinism across backtest runs.**
Documented: `CLAUDE.md` "Estado actual" item 2. Fixed with a deterministic seed keyed on
`game_pk`. **Status: FIXED, as documented — not independently re-derived this pass** (would
require a live backtest re-run, which is out of scope per execution rules; seed mechanism
itself checked in §8).

**REG-003 — Platt calibration corrupted by a duplicate/interrupted backtest launch resetting
`ml_state.platt_params` to identity mid-run.**
Documented: `CLAUDE.md`, blueprint v1.3 note. Not a pipeline bug — a launch artifact — but it
exposed that the reset+warm-start mechanism had no protection against concurrent/interrupted
writers. **Status: FIXED (relaunch) + mitigated** — a file lockfile around `ml_state` writes
was added 2026-07-11/12 (`CLAUDE.md`, "Actualización 2026-07-11/12" item 7). Not re-verified
against code this pass; flagged for §9 spot-check.

**REG-004 — `_team_dict()` in `backtest_and_retrain.py` leaking full-season `team_era`/
`team_whip`/`runs_allowed_per_game` when PIT mode is requested.**
Documented: `docs/AUDITORIA_MLB_2026-07.md` §3.1 (a prior draft of that same audit had
flagged this as still-open; corrected in the same doc after direct code verification).
**Status: FIXED, gated correctly** — `use_team_full_season_pitching_base` flag, set to
`not args.use_defense_pit`. Re-verify current line numbers in §4 (the file changed +190
lines in the current dirty working tree since that audit, so line numbers there are stale).

**REG-005 — Truncation bias: Monte Carlo doesn't model 9th-inning walk-offs, but real box
scores are truncated, so every learner (Kalman/bias/gradient-descent) read the gap as a
false "home overrated" signal.**
Documented: `CLAUDE.md` "Actualización 2026-07-11/12" item 5. **Status: FIXED — re-verified
this pass.** `_HOME_RUNS_TRUNCATION_FACTOR = 0.967` and `untruncate_home_runs()` present in
`learning_engine.py:44-121`, applied at the 3 learning call sites (Kalman offense_home/
defense_away, team-bias ratio via `_l0_ratio`, gradient-descent home role) — see diff excerpt
in §9. **New, adjacent finding surfaced while verifying this**: `_l0_ratio()`'s
`stage_factors_json` parameter is accepted but never read in the function body (dead
parameter, 3 call sites all pass it for no effect) — logged as **MATH-001** in §5, not a
leak, but evidence of an incompletely-cleaned-up revert (see REG-006).

**REG-006 — Two reverted attempts to change `_l0_ratio`'s denominator / bias dampening
formula, both caused real backtest regressions (0.24479→0.24550→0.24636).**
Documented: `CLAUDE.md` "Un fix diagnosticado pero revertido dos veces", and the in-code
postmortem docstrings on `_l0_ratio`/`compute_team_bias_kalman_adjusted`
(learning_engine.py:59-92, :565-598). **Status: REVERTED, correctly, per two independent
before/after backtests** — re-verified this pass that the code matches the "keep original
formula" decision (docstring explicitly says so, and the dampening formula shown at line
~578 is the original rational form, not either reverted attempt).

**REG-007 — Pinnacle structurally excluded from the live odds fetch.**
Documented: `docs/FBQ_MASTER_BLUEPRINT.md` v1.4 note (1). `REGION="us"` (live) excluded `eu`
(where Pinnacle is licensed), and the bookmaker key used was `"pinnaclesports"` instead of
the real The Odds API key `"pinnacle"`. Consequence: `ml_home_pin`/`ml_away_pin` were NULL
in 100% of live rows since at least 2026-07-04, meaning Platt-2D's live-correction gate
(which requires a Pinnacle fair line) was silently disabled all season. Backtest was NOT
affected (uses a separate historical-odds script that always had the correct key).
**Status: FIXED — re-verified this pass.** `odds_fetcher.py:48` — `REGION = "us,eu"` with an
explanatory comment; `odds_fetcher.py:88` — `_PINNACLE_KEY = "pinnacle"` with an explanatory
comment citing the exact same root-cause history. Confirms backtest/live were never
comparable until this fix landed (2026-07-12+, per commit history) — this is why v1.4's own
note says the backtest number did not move from this session's fixes: they live downstream
of the model, in the odds/value layer.

**REG-008 — `advanced_pit_enrichment/` (26 files) never had a formal file-by-file audit,
despite heavy use building the 4 PIT-aware engines.**
Documented: `docs/AUDITORIA_MLB_2026-07.md` §2 table, blueprint §0.1.
**Status per blueprint's own "Criterio salida" (2026-07-09 update): "✅ auditoría formal de
`advanced_pit_enrichment/` completada (26 archivos, 1 gap real encontrado y corregido en
`tte_prior_baseline_builder.py`)."** So this is now marked done by the project's own record —
**treated as FIXED/COMPLETE, not independently re-walked file-by-file in this pass** (26
files is out of proportion to this audit's P0/P1 budget given it's already been through a
dedicated pass; spot-checked 3 of the 4 PIT-aware engines' adapters directly in §4 instead).

---

## B. Double-counting (P1)

**REG-009 — AutoCalibrator triple-counted offense_mult + defense_mult + rest + forma against
the pitcher/bullpen/defense engines.**
Documented: `CLAUDE.md`, `CONTRACTS.md` — `auto_calibrator.py` **does not exist in the repo at
all** (superseded, not merely neutered). **Status: FIXED by deletion — re-verified this
pass.**
```
$ find . -iname "auto_calibrator.py" -not -path "*/node_modules/*"
./modules/baseball_module/calibration/__pycache__/auto_calibrator.cpython-312.pyc
```
Only a stale `.pyc` remains, exactly as `CLAUDE.md` states. **Contradiction flagged**:
`CLAUDE2.md` (untracked draft, line 67-76) describes `auto_calibrator.py` as still present
with a `tte_active` skip-gate — this is **stale/wrong** per direct filesystem check above.
`CLAUDE2.md` should not be trusted over `CLAUDE.md`/`CONTRACTS.md` on this point (see 00).

**REG-010 — Pitcher Engine `kbb_mult` double-counting K%-BB% when the winning SIERA/xFIP
estimator already encodes it.**
Documented: `CONTRACTS.md` §2 (`context_engine/pitcher_engine.py` row) — "conditional
`kbb_mult` (only applied when the winning estimator doesn't already encode K%-BB%)".
**Status: FIXED, as documented** — not independently re-verified this pass (out of this
pass's direct-read budget; flagged for §6 if time allows a spot check).

**REG-011 — Bullpen Engine reweighting K-BB% down when team SIERA is available, mirroring
REG-010's fix.**
Documented: `CONTRACTS.md` §2 (`bullpen_engine.py` row). **Status: FIXED, as documented,
same caveat as REG-010.**

**REG-012 — HFA Engine used to mix park factor + crowd boost + back-to-back in one factor;
crowd boost was later found to be pure noise (Pearson = -0.015) and removed permanently.**
Documented: `docs/AUDIT_FINDINGS.md` "FIX ARQUITECTÓNICO C2", `CONTRACTS.md`
(`hfa/hfa_engine.py` row: "Home crowd boost is permanently disabled... confirmed noise").
**Status: FIXED — re-verified this pass**, see REG-013 below (the *new* uniform home-win
correction is a distinct, separately-validated mechanism, not a reintroduction of crowd
boost).

**REG-013 — Systematic ~1.6-1.7pp under-prediction of home win probability across both
backtest seasons.**
Documented: `CLAUDE.md` "Actualización 2026-07-11/12" item 4. Fixed with a new
`_UNIFORM_HOME_MULT=0.028` in `hfa_engine.py` — explicitly **uniform across all parks**, a
different mechanism from the removed per-park crowd-boost lookup (REG-012), so this is not a
double-counting regression of that earlier fix.
**Status: FIXED — re-verified this pass.** `grep -n "_UNIFORM_HOME_MULT" hfa_engine.py`
confirms the constant is present with a 2026-07-11 dated comment matching this description
(see §5/§6 for the exact excerpt and an overlap check against REG-005's truncation fix, since
both affect home-side λ/runs).

---

## C. Fallback / missingness (P1)

**REG-014 — 5 dead data-fetcher methods with zero downstream consumers, each still making a
live network call on every prediction.**
Documented: `CONTRACTS.md` §2 (`data_fetchers.py` row), `docs/AUDITORIA_MLB_2026-07.md` §3.2.
`get_head_to_head`, `get_standings_status`, `get_team_offensive_stats`, `get_team_recent_form`,
`get_umpire_historical_stats` — removed 2026-07-06.
**Status: FIXED — re-verified this pass**, none of the 5 names appear in the current
`data_fetchers.py` (confirmed via the method inventory built earlier this session — see the
conversation's own `data_fetchers.py` review, cross-checked again: `grep -c` for each name
returns 0 matches in the file as it stands now).

**REG-015 (this session's own finding, not from prior docs) — Stadium-name sponsorship
renames desynchronized across 3 separate venue-keyed dictionaries.**
`park_weather_engine.STADIUM_DATABASE` had "Rate Field"/"Oriole Park at Camden Yards"/
"Daikin Park"/"UNIQLO Field at Dodger Stadium" added (confirmed via `game_outcomes` rows,
2026-07-05/13 per in-code comments) but `data_fetchers.WeatherAPI.STADIUM_COORDS` and
`historical_weather._STADIUM_COORDS` were never updated to match — silently dropping live
weather AND corrupting `get_travel_fatigue()`'s distance calc (fabricated 1000mi/1-timezone
fallback) for 4 teams' home games in 2026.
**Status: FIXED in this session, prior to this audit being requested** — both files patched
(current working-tree diff, `data_fetchers.py` +11 lines, `historical_weather.py` +7 lines).
**No regression test added for this fix** — flagged as a gap in §7/Section K
(`findings.csv`), since nothing currently asserts these 4 keys exist, so a 5th future rename
could reintroduce the same class of bug silently.

**REG-016 — `_normalize_event()`'s F5 naming scheme in `odds_fetcher.py` is a third,
independent naming convention from `GameOdds`' canonical one.**
Documented: `CONTRACTS.md` §2/§7, blueprint §0.5. Not yet an active bug (feeds only the UI
dropdown path, not `run_module()`), but flagged as a land mine for whoever wires it up next.
**Status: NOT FIXED, deliberately deferred, correctly documented as such — re-verified this
pass**: `odds_fetcher.py:321-336` still carries the inline warning comment; `ui/odds_loader.py`
:176-186 already does the field-name translation on its own read side (so the mismatch is
currently contained, not silently propagating).

**REG-017 — F5 markets (`h2h_h1`/`totals_h1`/`spreads_h1`) structurally cannot be fetched via
the bulk `/sports/{sport}/odds/` endpoint — The Odds API returns 422 for the whole request.**
Documented: `odds_fetcher.py:50-56` inline comment, blueprint §0.6. Needs the per-event
endpoint instead. **Status: NOT FIXED, deliberately deferred** — re-verified this pass,
`MARKETS = ["h2h", "totals", "spreads"]` (odds_fetcher.py:49) still excludes the 3 F5 keys;
this session's trim of `SPORTS_KEYS` (this conversation's own edit) did not touch `MARKETS`
and is unrelated to this gap.

**REG-018 — Bullpen reliever-role classification uses a single team's split, so a pitcher
traded mid-season can be misclassified.**
Documented: blueprint §0.6, listed as a known, deliberately-not-fixed edge case.
**Status: NOT FIXED, deliberately deferred, low-severity per project's own assessment.**

**REG-019 — `learning_engine.py`'s new `abstractGameState=="Final"` check takes the first
game in the API response without an explicit `gamePk` match.**
Documented: blueprint §0.6. **Status: NOT FIXED, deliberately deferred** ("query already
scoped server-side; low risk, harden someday" — project's own words).

---

## D. Chronology (P0)

**REG-020 — Backtest determinism / seed (see REG-002).**

**REG-021 — `recalibrate_platt`/`recalibrate_platt_2d` run after the main backtest loop, not
interleaved — confirmed non-contaminating.**
Documented: `docs/AUDITORIA_MLB_2026-07.md` §"Sigue pendiente" note under item 0.1.
**Status: CANNOT VERIFY further without exceeding this audit's reproducibility-run budget
beyond what §8 already used** — treated as previously verified, not re-derived.

---

## E. Learning / calibration (P0)

**REG-022 — Three independent, confirmed classes of *silent* calibration failure to date:**
(1) team-bias look-ahead leak (REG-001), (2) Platt reset-to-identity via interrupted launch
(REG-003), (3) Pinnacle gate silently disabling Platt-2D correction live all season (REG-007).
Documented explicitly as a pattern in `docs/FBQ_MASTER_BLUEPRINT.md` v1.4, §3.2 note ("ahora
hay TRES clases confirmadas..."). **Status: each individually FIXED; the underlying pattern
(no "is calibration alive" production monitor) is explicitly OPEN** — blueprint's own "Próximos
3 pasos" item 3 proposes a minimal monitor, not yet built (`grep` confirms no such monitor
module exists in the repo — see §9 findings).

**REG-023 — Platt/Platt-2D circularity check ("no input to `recalibrate_platt`/
`_gradient_step` derives circularly from an already-calibrated output") — proposed, never
executed.**
Documented: `docs/AUDITORIA_MLB_2026-07.md` §8 "Circularidad del Learning Engine" (open
question), carried forward in blueprint v1.3 and v1.4 ("sube otra vez de urgencia").
**Status: NOT FIXED / NOT VERIFIED — still open**, explicitly flagged by the project itself
as increasingly urgent. Re-attempted at a first-pass level in §9 of this audit within budget.

---

## F. Odds / market (P0)

**REG-024 — `best price` shopping ignored the `point` field, pairing a probability computed
for one total/spread line with a different book's line at that "best" price.**
Documented: blueprint v1.4 note (2). **Status: FIXED — re-verified this pass.**
`odds_fetcher.py::_accumulate_point_price`/`_consensus_line_and_price` (lines 257-291) group
by point and prefer Pinnacle's own quoted point; used consistently in both `_normalize_event`
and `get_best_odds_for_teams`.

**REG-025 — Runline cover-probability math hardcoded to ±1.5 regardless of the real line
fetched.**
Documented: blueprint v1.4 note (2), commit `0fd674e` per `CLAUDE.md`/git log ("fix(mlb):
parametrize runline cover probability by the real line, not a hardcoded 1.5").
**Status: FIXED, per commit message — not independently re-derived this pass** (would
require reading `montecarlo/simulator.py`'s runline function; flagged for §10 spot-check if
budget allows).

**REG-026 — Push on whole-number totals ("Over 9.0") counted as a full loss in EV instead of
a push.**
Documented: blueprint v1.4 note (3). **Status: FIXED, per documented `push_prob` addition —
not independently re-derived this pass**, flagged for §10.

**REG-027 — `track_record/publisher.py`'s fallback pick ran an already-decimal odds value
through an American-odds conversion (~2830% fake EV, max-Kelly-clipped stake).**
Documented: git commit `2e976c7` (seen in git log at session start), blueprint v1.4 note (4).
**Status: FIXED — confirmed via commit message inspection this pass** (`git log` shows the
exact commit; not re-read the diff itself, low priority given it's dated and specific).

**REG-028 — `ui/odds_loader.py` used to fabricate even-money (2.0) prices that `ui/mlb.py`
treated as real market data.**
Documented: blueprint v1.4 note (5). **Status: FIXED — re-verified this pass.**
`ui/odds_loader.py:161-166` — explicit comment: "No fallback here on purpose... Missing odds
must stay None."

**REG-029 — Bullpen "quality" signal computed over the ENTIRE roster (starters included),
not just relievers — documented premise ("relievers dominate by volume") verified false.**
Documented: blueprint v1.4 note (6), commit `c97e61f` ("fix(mlb): restrict bullpen quality
aggregate to actual relievers, not the whole roster") seen in git log at session start.
**Status: FIXED, per commit — not independently re-derived this pass.**

---

## G. Metric trustworthiness (P0) — carried forward, judged fully in §11

**REG-030 — Backtest vs. live were never comparable until REG-007 (Pinnacle gate) was fixed
2026-07-12+.** Backtest Brier/accuracy numbers pre-date and are independent of the entire
odds/value-layer bug class (REG-007, REG-024–029) since the historical-odds path always used
correct keys. **Live production metrics (CLV, ROI from real picks) are much younger** — real
Pinnacle data "recién empiezan a fluir desde 2026-07-12" (`CLAUDE.md`/blueprint v1.4).
**Status: open by design, not a bug** — see §11 for the full trustworthiness verdict.

**REG-031 — `track_record.db::picks` was empty (0 rows) as of the 2026-07-06 audit; no real
pick had completed a full publish→resolve cycle under corrected code.**
Documented: `docs/AUDITORIA_MLB_2026-07.md` §3.4, §8. **Status: CANNOT VERIFY without reading
the live DB** — checked in §12 (this audit's DB inspection, read-only).

**REG-032 — Historical `predictions_history.db` rows (April 2026) had `confidence` exactly
equal to the chosen side's `model_prob`, not an independent epistemic signal.**
Documented: `docs/AUDITORIA_MLB_2026-07.md` §8, marked as an unresolved historical question
(pre-dates the current `compute_data_quality_confidence()` fix) but confirmed NOT
reproducible with current code. **Status: HISTORICAL DATA ARTIFACT, current code verified
clean per that audit — not re-verified independently this pass** (would require reading old
rows out of `predictions_history.db`, done in §12 read-only DB inspection if time allows).

---

## H. Documentation drift (meta-issue, still relevant)

**REG-033 — `CONTRACTS.md` v1 (2026-05-13) described a system that no longer existed**
(files removed, wrong line counts, `AutoCalibrator` as if still active). **Status: FIXED by
full rewrite 2026-07-06** — current `CONTRACTS.md` re-verified against code that day.
**New instance of the same pattern found in this register**: `CLAUDE2.md` (untracked,
undated relative to CLAUDE.md) reintroduces the exact same class of drift (REG-009's
contradiction). **Not fixed** — flagged as INV finding in §2/§13 (recommend: delete or
reconcile `CLAUDE2.md`, since two authoritative-sounding docs disagreeing on whether
`auto_calibrator.py` exists is exactly the failure mode `CONTRACTS.md`'s rewrite was meant to
prevent).

---

## Known failing tests

**None.** Full suite run this pass: `462 passed in 38.94s`, 0 failed, 0 skipped (§12 has the
full run transcript). This matches `CLAUDE.md`'s last commit message ("462/462 tests pass")
and confirms the dirty working tree (including this session's own two edits) has not
regressed the suite.


<!-- ============================================================ -->
<!-- FILE: 02_project_inventory.md -->
<!-- ============================================================ -->

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


<!-- ============================================================ -->
<!-- FILE: 03_call_graph.md -->
<!-- ============================================================ -->

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


<!-- ============================================================ -->
<!-- FILE: 04_leakage_audit.md -->
<!-- ============================================================ -->

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


<!-- ============================================================ -->
<!-- FILE: 05_math_audit.md -->
<!-- ============================================================ -->

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


<!-- ============================================================ -->
<!-- FILE: 06_double_counting_audit.md -->
<!-- ============================================================ -->

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


<!-- ============================================================ -->
<!-- FILE: 07_fallback_audit.md -->
<!-- ============================================================ -->

# 07 — Fallback and Missingness Audit (P1)

## Per-layer fallback inventory

| Layer | Primary source | Prior-season source | Neutral fallback | Silent? | Coverage note |
|---|---|---|---|---|---|
| TTE (offense) | Current-season Statcast, PA-weighted | Yes — `prior_w=k/(k+PA)` blend | League-average λ if PA=0 | No — logged (`_tte_active` flag, metadata) | ~99.4% per `CONTRACTS.md` |
| Pitcher | SIERA→xFIP→xERA→FIP→ERA cascade | Prior-season pitcher stats (`get_pitcher_stats_full_fallback`) | League-average ERA/WHIP | Partially — fallback tier is recorded in metadata (`source_used`-style field, per `CONTRACTS.md`'s description of the cascade), not silent at the code level | ~96.5%+3.1%+0.9% cascade per CONTRACTS.md |
| `_team_dict()` (backtest only) | `get_team_pitching_stats()` season aggregate | — | `LEAGUE_AVG_ERA`/`LEAGUE_AVG_WHIP`/`LEAGUE_AVG_RUNS`, gated by `use_team_full_season_pitching_base` | **Explicitly gated, not silent** — this is a deliberate mode switch (`--use-defense-pit`), documented in-line (backtest_and_retrain.py:536-541, §4) | REG-004 |
| Bullpen | Real per-team boxscore innings + SIERA/ERA/K-BB%/barrel% | — | League bullpen constants | No — PIT coverage 100% claimed per CONTRACTS.md | — |
| Defense | DER (Bayesian-shrunk toward `_LG_DER`) + OAA | — | League DER | No — PIT coverage 100% claimed | — |
| Weather (live) | OpenWeather 5-day/3h forecast | — | `_NEUTRAL_TEMP_F=72.0` in `park_weather_engine._weather_mult` when `weather={}` | **Silent-by-necessity**: if `get_weather_for_stadium()` returns `None` (unknown venue name — REG-015's exact failure mode before this session's fix), `enriched["weather"]` is simply never set, and `park_weather_engine.py` treats missing weather as neutral with no distinguishing flag in the output metadata that says "we don't actually know the weather" vs. "it happened to be neutral." This is the **general shape of the bug class REG-015 was one instance of** — flagged as **FALL-001**. |
| Weather (backtest) | N/A — deliberately not fetched | — | Always neutral (`historical_weather.py` never instantiated, §3/§4) | Documented as intentional, not silent in the sense of being hidden — but see note below | 0% by design |
| Travel fatigue | Coordinate-based haversine + timezone-offset lookup | — | **Fabricated 1000mi/1-timezone assumption** when either venue's coordinates are missing from `STADIUM_COORDS` (`data_fetchers.py:1071-1074`, this session's exploration) | **Silent** — no flag distinguishing "real computed distance" from "made-up 1000mi placeholder" in the returned dict | This was the exact mechanism REG-015 broke for 4 renamed venues; now fixed for those 4, but the *fallback design itself* — silently substituting a fabricated number instead of returning "unknown" — remains generically fragile to the next stadium rename. Logged as **FALL-002**. |
| Roof status | `get_roof_status()` live game feed | — | Defaults `roof_closed=True` for the 8 retractable-roof parks if fetch fails (per `CONTRACTS.md`'s description of the pre-2026-07-06 bug and its fix) | Not re-verified this pass whether the *current* failure-path default is still `True` (conservative) or was changed; flagged in `hypotheses.md`. |
| Odds (live) | The Odds API, best-of-all-bookmakers | — | `None` (never a fabricated price) per `ui/odds_loader.py:161-166`'s explicit "no fallback on purpose" comment (REG-028) | **Not silent — deliberately explicit**, the one fallback in this whole table designed the *right* way (missing stays `None`, never impersonates real data). |
| Pinnacle fair line | `ml_home_pin`/`ml_away_pin` | — | `apply_platt_2d()` returns `p_home` unchanged (not identity coefficients) when no fit available — explicit `None`-means-"don't touch it" contract (`get_platt_2d_params()`'s own docstring, §9) | **Not silent — deliberately explicit**, same good pattern as above. |

## New findings this pass

**FALL-001 — Weather "unknown" and weather "neutral" are indistinguishable in the output.**
When `WeatherAPI.get_weather_for_stadium()` returns `None` (missing coordinates — the general
class of bug REG-015 was a specific instance of, for 4 venues, now fixed for those 4 but not
structurally prevented for the *next* stadium rename), `park_weather_engine.py` receives
`weather={}` and computes a neutral multiplier with no metadata flag recording that this was
an absence, not a genuine neutral reading. **Severity: Low-Medium** — the 4 known cases are
fixed; this is about resilience to the *next* occurrence of the same pattern (already
observed 4 times across different constants/dictionaries in this codebase per §5's
cross-cutting note). Recommended fix: `park_weather_engine.py`'s metadata should include an
explicit `weather_source: "live" | "missing"` field; regression test: assert that a stadium
name absent from `STADIUM_COORDS` produces a distinguishable metadata flag, not just a
neutral number.

**FALL-002 — Travel-fatigue's missing-coordinate fallback fabricates a specific, plausible-
looking number (1000 miles, 1 timezone) instead of signaling "unknown."**
`data_fetchers.py:1071-1074` (already read in full during this session's earlier stadium-name
investigation):
```python
elif previous_venue != current_venue:
    # Fallback: unknown stadium → assume mid-range travel
    miles = 1000
    time_zones = 1
```
This is the same general fragility as FALL-001, one layer over — a fabricated but
plausible-looking value is indistinguishable downstream from a real computation, which is
exactly what let REG-015 go undetected until this session (the fallback masked the failure
instead of surfacing it). **Severity: Low** (only triggers for genuinely unknown venues,
which — post REG-015 fix — should be zero for the current 30 active MLB stadiums, but will
recur the next time a park is renamed or a franchise relocates). Recommended fix: return
`None`/a distinguishing flag instead of fabricated numbers, and have `hfa_engine.py` fall back
to *no* travel-fatigue adjustment (neutral) rather than a specific fabricated distance,
mirroring the "stay None, don't impersonate real data" discipline already correctly used for
odds (REG-028) and Platt-2D (`get_platt_2d_params`).

## Coverage claims not independently re-measured this pass

CONTRACTS.md's specific coverage percentages (TTE 99.4%, Pitcher 96.5%+3.1%+0.9%, Bullpen
100%, Defense 100%) are carried forward as documented, not recomputed — recomputing them
would require running the backtest, forbidden by this audit's execution rules. Flagged as
"as documented, not independently verified this pass" rather than confirmed.


<!-- ============================================================ -->
<!-- FILE: 08_chronology_audit.md -->
<!-- ============================================================ -->

# 08 — Backtest Chronology Audit (P0)

## Reproducibility micro-run performed (within budget)

Per the execution rules, network calls are **absolutely forbidden** in this audit, with no
carve-out for the micro-run budget — and `backtest_and_retrain.py` makes live calls
(MLBStatsAPI, Savant, FanGraphs) for any non-cached game, so a genuine end-to-end micro-run
of the real backtest script could not be guaranteed network-free. I therefore restricted the
one live execution in this section to a **pure, zero-I/O function call** —
`monte_carlo_advanced()` itself takes only `(lambda_home, lambda_away, rng_seed)` and touches
no network, no project DB, no cache:

```python
r1 = monte_carlo_advanced(4.5, 4.2, rng_seed=823358 % (2**32))
r2 = monte_carlo_advanced(4.5, 4.2, rng_seed=823358 % (2**32))
r3 = monte_carlo_advanced(4.5, 4.2, rng_seed=999999)
# same seed identical p_home: True 0.530682 0.530682
# different seed differs:     True 0.529675
# elapsed 1.68s
```

**Confirms REG-002 directly**: identical seed ⇒ bit-identical `p_home`; different seed ⇒
different result (rules out a memoization/caching artifact masquerading as determinism).
This is the only section of this audit that executed live code; everything else below is
static analysis, which for chronology-specific questions (ordering, cutoffs) is arguably
stronger evidence than an observed run, since it traces the actual control flow rather than
inferring it from one output. Full walk-forward, multi-game-window verification was **not
attempted** live (would require network access for uncached games) — classified **not
verifiable in this phase** per the audit's own rules for anything beyond the stated budget.

## Sorting / ordering

`backtest_and_retrain.py:2519`: `order = "ORDER BY game_date ASC"` — confirmed single
ascending sort key, no explicit secondary key (e.g. `game_pk`) for same-date ties. The main
loop (`for idx, row in enumerate(rows, 1): ...`, line 2853) processes rows strictly in that
order, one game at a time, in a **single interleaved pass**: build data → predict → (further
down, outside this excerpt) record outcome / update Kalman — not a two-phase
"predict-all-then-update-all" design. Verified this pass by reading the loop body directly
(backtest_and_retrain.py:2853-2930+).

## Same-day ordering / doubleheaders

**Analysis (static, verified against code):** the walk-forward cutoff used for team-bias
(`_bias_before_date = str(game_data.get("game_date", ""))`, backtest_and_retrain.py:1780) is
**date-granular**, not timestamp-granular, and the stored `game_outcomes.game_date` column is
confirmed (via direct DB read, §2) to hold plain `YYYY-MM-DD` strings with no time component.
Since the SQL comparison is `game_date < before_date` (learning_engine.py, e.g. :521-523) and
`before_date` is the **current game's own date**, every row sharing that date — including
both games of a doubleheader — is **excluded** from that game's own bias-training query,
regardless of which one is processed first. This means:
- No leak is possible from doubleheader ordering (the cutoff is conservative by construction:
  it under-uses same-day information rather than risking using it out of order).
- Kalman state (`update_kalman`, not gated by `before_date` at all) **is** updated
  sequentially as the single loop pass proceeds — so a same-day Game 2 processed after Game 1
  in loop order (arbitrary intra-date order, since there's no secondary sort key) legitimately
  sees Game 1's real, already-final outcome. This mirrors reality (Game 2 of a real
  doubleheader starts after Game 1 has actually finished) and is **not a leak** — it is
  correct walk-forward behavior, contingent only on Game 1 truly preceding Game 2 in kickoff
  time, which this audit did not independently verify (SQLite's return order for
  identical `game_date` values is not guaranteed to match true intra-day chronology — e.g. if
  `rows` happen to return Game 2 before Game 1 for a given date, Game 2's Kalman-based
  prediction would be computed **before** Game 1's real outcome has been folded into Kalman
  state, which is the *safe* direction — under-informed, not leaked — but means backtest
  fidelity for doubleheader Game 2 specifically may vary run-to-run in a way this audit
  cannot bound further without executing SQL against the live DB, which the "no writes"-plus
  read-only spirit of this phase permits as a read, but doing so productively would require
  cross-referencing actual first-pitch times not present in `game_outcomes` — out of budget,
  logged as a `hypotheses.md` item, not a confirmed finding).
- **B2B (back-to-back) detection** uses its own explicit in-memory tracker
  (`_team_last_game: Dict[str, Tuple[str,str]]`, updated **after** computing the current
  game's B2B flags — backtest_and_retrain.py:2891-2904) rather than a DB re-query. This is a
  clean, leak-safe pattern: verified the update happens strictly after the read for the
  current game, so a team cannot see its own game's B2B status computed from itself.

**Verdict**: same-day ordering is **not a source of leakage** by design (date-granular
walk-forward cutoff for the trained signal that matters most, team bias). One narrow,
low-severity fidelity question remains open (intra-day SQL row order vs. true kickoff order
for doubleheader Kalman updates) — logged in `hypotheses.md`, not `findings.csv`, since it is
unproven and, even if true, would bias toward under-information rather than leakage.

## Season resets / cross-season boundaries

`backtest_and_retrain.py:2861-2878`: on detecting a season boundary mid-loop (current row's
`season != _loop_season`), the code explicitly re-fits Platt (`recalibrate_platt`) for the
**season that just ended**, then **warm-starts** the new season's Platt params from that
fresh fit (`learning.save_state(..., season=season)`) — "in-run, fresh," per its own log
message, distinguishing it from a stale warm-start carried over from a previous, unrelated
run (the exact class of bug that caused REG-003's Platt-hysteresis issue, now with an
explicit mid-run refit instead of relying on whatever was left in `ml_state` from a prior
process). **Re-verified this pass, matches CLAUDE.md's "Actualización 2026-07-11/12" item 6
("Histéresis de warm-start de Platt entre corridas de backtest") as a real, current fix, not
just a doc claim.**

## Cache isolation

Not independently re-derived this pass beyond what §4 already covers (PIT cache DBs are
separate files per domain — `pit_cache_2024.db`, `pit_cache_2025.db`, `pit_cache_merged.db`,
`pit_cache_pitcher.db` — confirmed to exist as distinct files in §2's inventory). Whether two
concurrent backtest processes could corrupt a shared PIT cache file was **not tested live**
(would require deliberately running two concurrent processes, out of scope/budget) — the
`ml_state` lockfile mitigation (REG-003) is documented as covering the Platt/ml_state case
specifically, not the PIT cache files; **not verifiable in this phase** whether an analogous
protection exists for PIT caches.

## Database isolation — CHRON-001 (new finding, high severity)

**This is the most significant finding of this entire audit section.** While inspecting
`game_outcomes` for chronology evidence (§2's DB read), a direct data-integrity gap was found
between the live-production writer and the backtest's overwrite path, evidenced as follows:

**Evidence 1 — the live writer** (`modules/baseball_module/calibration/learning_engine.py`,
`record_prediction()`, lines 270-339): called from the live pipeline to persist a pre-game
prediction (`INSERT OR IGNORE INTO game_outcomes ...`), with careful COALESCE-based backfill
for any pre-existing row so it **never overwrites already-populated fields**:
```python
# Row already exists (e.g. historical bulk import).  Backfill
# any NULL fields we now have — never overwrite existing data.
conn.execute(
    """
    UPDATE game_outcomes SET
        stage_factors_json = COALESCE(stage_factors_json, ?),
        p_home_raw         = COALESCE(p_home_raw, ?),
        ...
    WHERE game_pk = ?
    """, ...)
```

**Evidence 2 — the live auto-reconciler**, `run_module.py:125`:
```python
_learning.fetch_pending_outcomes()
```
called on **every** live `run_module()` invocation (i.e. every time anyone opens the
Streamlit app and analyzes any game, or the daily-picks CLI runs). This walks any
previously-recorded-but-unscored `game_outcomes` row and, via `update_outcome()`
(learning_engine.py:408-444), backfills `actual_home_runs`/`actual_away_runs`/`home_won` the
moment a real final score becomes available — again carefully idempotent
(`WHERE game_pk = ? AND actual_home_runs IS NULL`, so it can't double-fire).

**Evidence 3 — the backtest's overwrite path** (`backtest_and_retrain.py`,
`update_game_outcomes()`, lines 1962-1984):
```python
def update_game_outcomes(conn, game_pk, lh, la, p_home, p_away, ...):
    conn.execute(
        """
        UPDATE game_outcomes
        SET lambda_home = ?, lambda_away = ?,
            p_home = ?, p_away = ?,
            p_home_raw = ?, p_away_raw = ?,
            stage_factors_json = ?,
            backtest_run_at = ?
        WHERE game_pk = ?
        """,
        (lh, la, p_home, p_away, p_home_raw, p_away_raw, sf_json, now, game_pk),
    )
```
**No `COALESCE`, no guard, no check for whether this row was originally authored by live
production.** It is called for every row the backtest's main SELECT pulls in, which is gated
only by `WHERE actual_home_runs IS NOT NULL AND season IN (...)` (:2514-2523) — a filter that
has **nothing to do with whether the row's `lambda_home`/`p_home` were computed live, by a
prior backtest run, or by a historical bulk import.**

**Empirical confirmation this is not hypothetical — read directly from
`data/predictions_history.db` (read-only query, this pass):**
```
season=2026 rows in game_outcomes:              615
  of which backtest_run_at IS NOT NULL:          563   ← already processed by some backtest run
  of which backtest_run_at IS NULL
      AND actual_home_runs IS NULL:               52   ← pure live picks, unresolved,
                                                          game_date 2026-05-15 .. 2026-07-12
```
The 563 already-`backtest_run_at`-stamped 2026 rows prove that **a backtest run scoped to
include season 2026 has already executed against this exact database** at some point. The
52 remaining rows are genuine live, unreconciled predictions sitting in the same table,
structurally indistinguishable from the 563 except by the very columns
(`backtest_run_at`, `actual_home_runs`) that a future backtest run would use to decide whether
to sweep them in.

**The mechanism that would trigger the actual damage, step by step:**
1. Today: these 52 rows have `actual_home_runs IS NULL` (game not yet reconciled) →
   excluded from any backtest's SELECT.
2. The **next time anyone runs `run_module()` live** (opening the app, or the daily CLI),
   `fetch_pending_outcomes()` (run_module.py:125) will silently backfill
   `actual_home_runs`/`actual_away_runs` for any of these 52 games that have since concluded.
3. **The next time `backtest_and_retrain.py` is invoked with a `--season` argument that
   includes 2026** (or no `--season` filter at all, which the argparse help text confirms
   means "all seasons") — very plausible, since 563 of 615 2026 rows show this has already
   happened, presumably during this session's own PIT-rebuild validation rounds
   (`reports/round1`..`round10`, all dated 2026-07-11/12, overlapping the live game dates
   seen in `predictions_history.db`) — any of those 52 games now eligible (`actual_home_runs`
   populated) get their `lambda_home`/`lambda_away`/`p_home`/`p_away`/`stage_factors_json`
   **silently overwritten** by the backtest's own freshly recomputed values, and
   `backtest_run_at` flips from NULL to a timestamp with zero audit trail of what the
   original live-computed values were.

**Scope of actual damage / what this does NOT corrupt:**
- **`track_record.db::picks`** — architecturally independent (separate DB file, separate
  schema), persists its own `model_prob`/`odds_decimal`/`ev_pct` at publish time. Confirmed
  this pass: **0 rows** in `picks` right now (matches REG-031), so nothing has been corrupted
  there yet, and nothing in `picks` reads back from `game_outcomes` for its stored fields (an
  architectural firewall that happens to protect CLV/ROI grading from this specific bug).
- This is therefore **not (yet) a corruption of the CLV/ROI evaluation pipeline** the project
  treats as its north-star metric.

**What it DOES corrupt / put at risk:**
- **`analyze_game_outcomes.py`** — reads `game_outcomes` directly for "ROI by
  stadium/month/day-night" analytics (per `CONTRACTS.md`). Any conclusion it draws about
  **live** performance for a game later swept into a batch backtest is silently describing
  **backtest-reconstructed** performance instead, with no column distinguishing the two after
  the overwrite (only `backtest_run_at`'s presence hints at it, but the *previous* live value
  is gone — no versioning, no shadow column, no log).
- Any future manual or automated audit that treats `game_outcomes.p_home`/`lambda_home` as
  "what the live system actually predicted, before this game, in real time" for a 2026-season
  game is at risk of being wrong without any error or warning.
- Undermines exactly the kind of provenance guarantee the project's own `1.1 Feature Store`
  blueprint item (§2, `docs/FBQ_MASTER_BLUEPRINT.md`) is designed to eventually provide
  ("backtest y producción leen de la misma tabla" — but explicitly a *read* guarantee, not
  "backtest may silently mutate what production already wrote").

**Severity: High.** Not a leakage bug (no future data used — the backtest's recomputed value
for a 2026 game still only uses data available as of that game's date under walk-forward
rules), and not (yet) a threat to the CLV/ROI metric specifically. But it is a real,
currently-latent, easily-triggered data-integrity gap that destroys the live-prediction
provenance record the instant a routine backtest re-run touches a reconciled live game —
logged as **CHRON-001** in `findings.csv`, recommended fix: guard
`update_game_outcomes()`'s `UPDATE` with `AND backtest_run_at IS NULL` removed as the
distinguishing test (that's not sufficient, since a *previous* backtest-touched row should
still be re-updatable by a *later* backtest run) — the actual fix needs a new boolean/
provenance column (e.g. `source = 'live' | 'backtest'`) set once at `INSERT` time by
`record_prediction()` and never flipped, with `update_game_outcomes()` either refusing to
touch `source='live'` rows or writing its recomputed values to separate
`backtest_lambda_home`/`backtest_p_home` columns instead of clobbering the live ones.

## Cutoff handling — summary

Already covered in depth in §4 (leakage) and above (`before_date` walk-forward). No
additional cutoff-handling issues found beyond what's logged there and in CHRON-001.


<!-- ============================================================ -->
<!-- FILE: 09_learning_calibration_audit.md -->
<!-- ============================================================ -->

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


<!-- ============================================================ -->
<!-- FILE: 10_odds_market_audit.md -->
<!-- ============================================================ -->

# 10 — Odds and Market Audit (P0)

## Timestamps: odds vs. prediction vs. game-start

Not independently re-derived beyond what the register already covers — `odds_fetcher.py`'s
live odds carry a `last_update`/fetch timestamp (`_normalize_event()`, "last_update":
`datetime.now().isoformat()`); the historical path (`fetch_historical_odds.py`) is a
separate, quota-limited, one-time downloader per `CONTRACTS.md`. No evidence found or sought
this pass of odds being matched to the wrong game's start time; out of budget for a fresh
timestamp-matching audit beyond the register's existing entries (REG-024 already covers the
point-mismatch bug, which is a related but distinct issue from timestamp mismatch).

## Opening vs. closing

`track_record/capture_closing_lines.py` (new, untracked, 133 lines) exists specifically to
capture the closing line for CLV computation — per project memory, built before reactivating
`ODDS_API_KEY`. `track_record.db::picks` schema includes `closing_odds_decimal`,
`closing_pin_home`, `closing_pin_away`, `closing_captured_at`, `clv_pct` (confirmed via
schema read, §2) — but **0 rows exist**, so opening-vs-closing capture has never actually run
against a real pick. Status: infrastructure present, unexercised (matches REG-031).

## Vig removal — 3 methods, and a duplicated implementation found this pass

`core/value_detector.py` implements `remove_vig_multiplicative`, `remove_vig_power`,
`remove_vig_shin` (lines 107-127). The live Pinnacle fair-line reference
(`_pinnacle_fair_probs`, line 129-136) uses **multiplicative** specifically.

**New finding (`ODDS-001`)**: `backtest_and_retrain.py` has its own, independent devig
function, algebraically identical to `remove_vig_multiplicative` but never imported from
`core/value_detector.py`:

```python
# backtest_and_retrain.py:1989-1992
def _devig(o1: float, o2: float) -> Tuple[float, float]:
    """Multiplicative devig of a two-outcome market."""
    t = 1.0 / o1 + 1.0 / o2
    return (1.0 / o1) / t, (1.0 / o2) / t

# core/value_detector.py:107-111 — same formula, different code
def remove_vig_multiplicative(odds_list: List[float]) -> List[float]:
    implied = [1/o for o in odds_list]
    total = sum(implied)
    return [imp / total for imp in implied]
```

Used at `backtest_and_retrain.py:2667` and `:3140` (`pin_fh, pin_fa = _devig(r["ml_home_pin"],
r["ml_away_pin"])`) to compute the backtest's own Pinnacle fair-line reference —
**mathematically consistent today** with the live path's `_pinnacle_fair_probs` (both
multiplicative), so this is **not currently a live/backtest parity bug**. It is, however, the
exact same class of "duplicated-constant-drift" risk the project has already been bitten by
repeatedly this session (`LG_XWOBA`, `LG_DER`, `LG_BARREL_PA`, stadium-name dictionaries) —
if the live method is ever changed to `remove_vig_power` or `remove_vig_shin` (both already
implemented and explicitly proposed for comparison in `docs/FBQ_MASTER_BLUEPRINT.md` §2.3),
the backtest's hardcoded multiplicative `_devig()` would **silently stop matching** unless
someone remembers to update this second copy too. Logged as **ODDS-001**, Medium severity
(latent, not currently active), recommended fix: have `backtest_and_retrain.py` import
`remove_vig_multiplicative` from `core/value_detector.py` instead of maintaining its own copy.

## Devig method consistency, live vs. backtest — verdict

**Consistent today** (both multiplicative), **not structurally guaranteed to stay consistent**
(ODDS-001). No inconsistency found for the current state of the code.

## Sportsbook / side mapping

`get_best_odds_for_teams()`/`_normalize_event()` match by exact team-name string against the
event's `home_team`/`away_team` fields (`odds_fetcher.py`); fuzzy substring matching used only
in `get_best_odds_for_teams()`'s caller context (`home_team.lower() in g_home.lower() or
g_home.lower() in home_team.lower()`, line 530-531) — not re-derived for edge cases (e.g. a
team name that is a substring of another team's name) this pass; flagged in `hypotheses.md`
as an unverified but plausible edge case (no evidence of an actual collision found or sought).

## Line movement

Out of scope for this pass — no `odds_history` table exists yet (per blueprint §2.1, listed
as FASE 2 future work, not yet built). Confirmed absent: no table named `odds_history` found
in `data/predictions_history.db`'s schema (§2's table list) or in `track_record.db`.

## CLV validity

**Not yet measurable** — `track_record.db::picks` has 0 rows (REG-031, re-confirmed §2/§9).
The instrument (`capture_closing_lines.py`, schema columns) exists but is unexercised. Any
CLV number computed today would be undefined (no rows to average). This is the project's own
stated position (`docs/FBQ_MASTER_BLUEPRINT.md`, "Próximos 3 pasos" item 1 — needs 4-6 weeks
of live Pinnacle data accumulation).

## ROI validity

Same conclusion as CLV — `track_record/stats.py::compute_stats()` computes ROI directly from
`picks`, currently empty. Historical `predictions_history.db::predictions` (88 rows) is **not
picks-graded** (no resolution/profit_loss tracking in that table per its schema, §2) — it is
raw UI analysis output, not a bet ledger. **Any ROI figure currently circulating for this
project (e.g. the backtest's own edge-bucket ROI, "edge≥8% +0.20%, edge≥10% -2.97%" cited in
`CLAUDE.md`) is a *backtest* ROI**, computed against historical odds via the fully-separate
`fetch_historical_odds.py`/`historical_odds` table path (4,695 rows, confirmed §2) — **not**
a live-production ROI. This distinction, while already implicit in the project's own
documentation, is worth stating explicitly for §11: **no live ROI number exists at all right
now**, only a backtest one, and the backtest one predates and is independent of every
odds/value-layer bug fixed in the 2026-07-12/14 session (REG-007/024-029) per the project's
own v1.4 note ("el backtest no se ve afectado" — the historical-odds script always had
correct keys).

## Summary verdict for §11

- **CLV**: not yet computable (zero data), instrument ready.
- **Live ROI**: does not exist yet (zero resolved picks).
- **Backtest ROI**: exists, computed via a path architecturally independent of the recently
  fixed live odds/value-layer bugs, but is itself only as trustworthy as the model/leakage
  findings in §4/§8/§9 allow — carried into §11's full verdict.


<!-- ============================================================ -->
<!-- FILE: 11_statistical_evaluation_audit.md -->
<!-- ============================================================ -->

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


<!-- ============================================================ -->
<!-- FILE: 12_operational_audit.md -->
<!-- ============================================================ -->

# 12 — Operational Audit + Test Suite

## Test suite run (performed this audit, read-only)

```
$ python3 -m pytest tests/ -q
462 passed in 38.94s
```

**462 / 462 passed, 0 failed, 0 skipped.** Matches `CLAUDE.md`'s last-commit claim
("462/462 tests pass") and confirms the current dirty working tree — including both this
session's pre-audit edits (stadium-alias fix, `SPORTS_KEYS` trim) and the larger uncommitted
point-in-time-rebuild diffs (`learning_engine.py`, `backtest_and_retrain.py`,
`tte_pit_adapter.py`, etc.) — has not regressed anything. **No failing test to map to a
finding.** Test file count: 43 `.py` files under `tests/` (up from CONTRACTS.md's "39 files"
as of 2026-07-06, consistent with the new untracked `test_track_record_closing_lines.py` plus
organic growth).

## API failures / retries / rate limits

`odds_fetcher.py::_get()` (lines 97-123) — exponential-ish backoff (`delay*(attempt+1)`),
explicit 429 handling honoring `Retry-After` capped at `_MAX_RETRY_AFTER=30s`, explicit 401
short-circuit (invalid key, no point retrying), up to 3 attempts. `_get_raw_events()` adds a
**60-second full-failure backoff** (`_FULL_FAILURE_BACKOFF_SECONDS`) to avoid hammering the
API after a total outage, and explicitly avoids caching a partial multi-sport fetch as if it
were complete (`if events and all_ok: _save_cache(...)` vs. `elif events: ... not caching`) —
this exact "partial fetch poisons the cache" pattern is called out in
`docs/FBQ_MASTER_BLUEPRINT.md` §0.6 as a bug class that recurred even within the same session
that was fixing it elsewhere, so its presence here (correctly guarded) is worth confirming
positively: **re-verified this pass, the guard is real and present** (`odds_fetcher.py:243-251`).

## Cache corruption handling

`odds_fetcher.py::_load_cache()` handles 3 failure modes explicitly: legacy list-format cache
(treated as expired, not crashed on), JSON decode errors (caught, returns `None`), and missing
file. `historical_weather.py`'s `HistoricalWeatherFetcher.__init__` similarly catches load
errors and resets to an empty cache rather than crashing. No corruption-handling gaps found
in the files read this pass.

## Manifests / resumability

- This audit's own output directory (`audit_20260714/`) is itself built with resumability in
  mind per the brief (skip already-written section files on restart) — not separately
  applicable to the project's own code, but worth noting the brief's own resumability
  requirement was honored (each section file written incrementally, in order, as completed).
- **Project-level resumability**: `reports/` directory (18 dated round-subdirectories, §2)
  demonstrates real operational discipline — every risky backtest round has a paired
  `backtest_report_<timestamp>.json` + `stdout.log`, and **18 separate
  `predictions_history_backup_pre_<label>_<timestamp>.db` snapshots** exist in `data/`,
  confirming the project actually practices "snapshot before risky operation" (matches this
  user's own stated preference pattern from prior sessions — validated empirically here, not
  just claimed).
- `backtest_and_retrain.py`'s `_acquire_ml_state_lock()` (lines 2359-2394, read in full this
  pass): a plain PID+timestamp advisory file lock at `data/backtest_ml_state.lock`, 4-hour
  staleness window, released via `atexit`. Explicitly documented as "not OS-level flock —
  good enough for accidental double-launch, not adversarial use." **Confirmed no stale lock
  file present right now** (clean state). This directly mitigates the *concurrent-process*
  class of corruption that caused REG-003, but **does not address CHRON-001** (§8) — that is
  a single, properly-locked process legitimately overwriting a different process's
  (production's) prior write to a shared table; a PID lock cannot prevent that, since only
  one write is happening at a time and it isn't concurrent.

## Memory / runtime

Not independently benchmarked this pass (would require executing the backtest, forbidden).
`docs/FBQ_MASTER_BLUEPRINT.md` §1.2 states SQLite has "handled without effort" Monte Carlo
runs of millions of samples, a 4,830-game backtest, and several 60-130MB concurrent PIT
caches — carried forward as documented, not re-benchmarked.

## Logs

Extensive, consistent use of the `logging` module throughout every file read this session
(`logger.warning`/`.info`/`.debug` at every meaningful branch — enrichment failures, cache
hits/misses, fallback triggers, lock acquisition). No `print()`-based logging found in any
file read this pass. This is a genuine operational strength, not just a claim — directly
observed across `data_fetchers.py`, `odds_fetcher.py`, `learning_engine.py`,
`backtest_and_retrain.py`, `hfa_engine.py`.

## Silent exceptions

Broad `except Exception:` blocks are common across the codebase (data fetchers, cache
loaders) — but every one directly read during this audit (§4/§7/§8/§9/§10's code excerpts)
logs a warning or debug message before falling back, consistent with the project's documented
"graceful degradation" philosophy for optional enrichment (Savant/FanGraphs/weather all
explicitly designed to fail open). **Not exhaustively enumerated file-by-file this pass**
(P2-level depth, per the audit's own priority rules) — no instance of a fully silent
`except: pass` with zero logging was encountered in any of the specific functions read across
sections 1-11, but this is not a claim that none exist anywhere in the ~50+ files not directly
opened this pass. Flagged in `hypotheses.md` as an area worth a dedicated grep-based sweep
in a future pass, specifically targeting whether any silent-except sits on a path that could
mask a leakage-relevant failure (e.g., a PIT snapshot lookup silently falling back to a
non-PIT value without logging which branch was taken).

## Reproducibility

Covered fully in §8 (seed mechanism, live-verified this pass with a network-free micro-check).

## Database read-only inspection summary (consolidated from §2/§8)

- `data/predictions_history.db`: 7.5MB, live, last modified 2026-07-12. `game_outcomes`:
  5,474 rows across 3 seasons. **CHRON-001** discovered directly from this inspection.
- `data/track_record.db`: `picks` table, 0 rows (REG-031 re-confirmed).
- Stale orphan files (`data/game_outcomes.db` — 0 bytes, `data/mlb_learning.db` — stale since
  May 27): logged as **INV-002**, both `.gitignore`'d, low risk.


<!-- ============================================================ -->
<!-- FILE: 13_technical_debt.md -->
<!-- ============================================================ -->

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


<!-- ============================================================ -->
<!-- FILE: 14_remediation_roadmap.md -->
<!-- ============================================================ -->

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


<!-- ============================================================ -->
<!-- FILE: findings.csv -->
<!-- ============================================================ -->

```csv
id,severity,area,file,function,lines,affects,summary,recommended_fix,regression_test
CHRON-001,High,chronology/data-integrity,"backtest_and_retrain.py; modules/baseball_module/calibration/learning_engine.py; modules/baseball_module/core/run_module.py","update_game_outcomes(); record_prediction(); update_outcome(); fetch_pending_outcomes()","backtest_and_retrain.py:1962-1984,2514-2523; learning_engine.py:270-339,408-444; run_module.py:125",both,"game_outcomes is shared, unprotected, between live-production writes and backtest overwrites. update_game_outcomes() does an unconditional UPDATE...WHERE game_pk=? with no guard distinguishing a live-authored row from a backtest-authored one. Empirically confirmed via DB read: 563 of 615 season-2026 rows already have backtest_run_at set (a backtest already processed them), 52 remain pure-live/unresolved. The next live run_module() call will auto-backfill actual_home_runs for any of those 52 that have concluded (fetch_pending_outcomes); the next backtest run scoped to include season 2026 will then silently overwrite their live-computed lambda_home/p_home/stage_factors_json with backtest-recomputed values, destroying the live-prediction provenance record with no versioning or audit trail. track_record.db's picks table is architecturally immune (separate DB, persists its own model_prob at publish time; currently 0 rows), so CLV/ROI grading is not yet at risk, but analyze_game_outcomes.py and any future analysis treating game_outcomes.p_home as ""what was live-predicted"" is.","Add a provenance column (e.g. source='live'|'backtest') set once at INSERT time by record_prediction() and never overwritten; have update_game_outcomes() either skip source='live' rows or write to separate backtest_lambda_home/backtest_p_home columns instead of clobbering the live ones.","New test: insert a game_outcomes row via record_prediction() (source='live'), run update_game_outcomes() against the same game_pk, assert lambda_home/p_home are unchanged (or land in separate backtest_* columns) and source stays 'live'."
MATH-002,Medium,math/PIT-adapter,modules/baseball_module/advanced_pit_enrichment/tte_pit_adapter.py,_compute_lambda,~168-200,backtest only,"tte_pit_adapter.py regresses barrel% toward league mean using plate appearances (pa) as the Bayesian sample size for all four metrics, but barrel% is fundamentally a per-batted-ball-event (attempts) rate, and PA is ~1.47x attempts. This under-shrinks barrel% specifically in the PIT-mode backtest path (used by the project's own cited --use-full-pit baseline), giving it a systematically more confident (less regressed) barrel signal than the live engine computes for the same team/date. Self-documented in-code as a known, deliberately deferred gap pending a separate PIT-cache batted_ball_count fix.",Use attempts (batted-ball-event count) as the Bayesian n for barrel% regression in tte_pit_adapter.py once the underlying PIT cache's batted_ball_count foul-inflation bug is fixed; until then this is accepted debt.,"Add a test asserting barrel_reg's shrinkage denominator equals a passed-in attempts count, not pa, once fixed; currently no test guards this distinction."
ODDS-001,Medium,odds/devig,"backtest_and_retrain.py; core/value_detector.py",_devig(); remove_vig_multiplicative(),"backtest_and_retrain.py:1989-1992; core/value_detector.py:107-111",both,"backtest_and_retrain.py implements its own multiplicative devig function, algebraically identical to but independent of core/value_detector.py's remove_vig_multiplicative (used live via _pinnacle_fair_probs). Currently mathematically consistent (both multiplicative), but this is the same duplicated-constant/logic-drift pattern already responsible for at least 4 other confirmed bugs in this codebase this session (LG_XWOBA, LG_DER, LG_BARREL_PA, stadium-name dictionaries). If the live method is ever changed to remove_vig_power/remove_vig_shin (both already implemented and explicitly proposed in the project's own roadmap for CLV comparison), the backtest's hardcoded copy would silently stop matching.",Import remove_vig_multiplicative from core/value_detector.py in backtest_and_retrain.py instead of maintaining a second copy.,"Add a test asserting backtest_and_retrain._devig (or its replacement import) produces identical output to core.value_detector.remove_vig_multiplicative for a range of odds pairs."
MATH-003,Low,math/calibration-freshness,modules/baseball_module/hfa/hfa_engine.py,HFAEngine.get_adjusted_lambdas / module constant _UNIFORM_HOME_MULT,15-34,52,97-98,both,"_UNIFORM_HOME_MULT=0.028 was sized from a residual win-probability analysis on a backtest that predates the truncation-bias reconciliation (report timestamps: round3_uniform_home_mult precedes round5_l0_reconstruction_fix). The constant's own docstring explicitly asks for re-validation against a fresh backtest after the truncation fix landed; no direct evidence found of that specific re-derivation having been performed as its own dedicated check (as opposed to being implicitly carried through later end-to-end validation rounds, whose final Brier did not regress).",Re-run the residual home-win-probability analysis described in the docstring now that the truncation correction is in place and confirm 0.028 is still correctly sized; or explicitly document that the round-10 end-to-end validation already supersedes the need for a standalone re-check.,"None currently exists; add a diagnostic script or test capturing the home-favorite/away-favorite residual by season, comparable to the one that originally derived 0.028."
FALL-001,Medium,fallback/weather,data_fetchers.py; modules/baseball_module/hfa/park_weather_engine.py,WeatherAPI.get_weather_for_stadium(); ParkWeatherEngine._weather_mult(),data_fetchers.py:1536-1554; park_weather_engine.py:360-378,live,"When get_weather_for_stadium() returns None (e.g. an unmapped/renamed venue - the general failure mode REG-015 was one confirmed instance of), park_weather_engine.py receives weather={} and silently computes a neutral multiplier with no metadata flag distinguishing ""genuinely neutral conditions"" from ""we don't actually know."" This is the general shape of a bug class that has already caused a real, confirmed production gap (4 renamed stadiums, fixed this session) and offers no structural protection against the next occurrence.",Add an explicit weather_source: 'live'|'missing' field to the adjustment metadata so a silent coordinate-lookup miss is distinguishable from a real neutral reading.,"Test: call adjust_for_park_and_weather with a stadium name absent from STADIUM_COORDS and assert the returned metadata flags the weather as missing, not merely neutral."
FALL-002,Medium,fallback/travel,data_fetchers.py,MLBStatsAPI.get_travel_fatigue(),1060-1074,live,"When either the previous or current venue is missing from STADIUM_COORDS, get_travel_fatigue() fabricates a specific, plausible-looking placeholder (1000 miles, 1 timezone) instead of returning an ""unknown"" signal - indistinguishable downstream from a real computed distance. This is exactly the kind of silent substitution that let REG-015 go undetected for 4 renamed stadiums until this session.",Return None or an explicit is_estimated flag instead of fabricated numbers; have hfa_engine.py fall back to no travel-fatigue adjustment (neutral) rather than a specific invented distance.,"Test: call get_travel_fatigue with a synthetic unmapped venue name and assert the result is flagged as estimated/unknown rather than a bare 1000/1 pair indistinguishable from a real computation."
MATH-001,Low,math/dead-code,modules/baseball_module/calibration/learning_engine.py,_l0_ratio(),"59-92, call sites 534,536,712",both,"_l0_ratio()'s stage_factors_json parameter is accepted at all 3 call sites (which fetch the column from SQL purely to pass it in) but never read in the function body - a leftover from two reverted denominator-change attempts documented in the function's own postmortem docstring. No behavior impact, but it misleads a future reader into thinking the parameter still matters.",Remove the unused parameter and the now-pointless stage_factors_json SQL fetch at the 2 call sites (or add a one-line comment marking it vestigial).,"None needed - removing an unused parameter cannot change behavior; existing test suite (462 tests) already covers these call sites."
LEARN-002,Medium,learning/monitoring,(none - absence confirmed via repo-wide grep),N/A,N/A,live,"No production monitor exists to detect a silently-disabled calibration mechanism, despite three independent, confirmed incidents of exactly that (team-bias look-ahead leak, Platt reset-to-identity via interrupted launch, Pinnacle gate silently disabling Platt-2D live for months). The project's own blueprint proposes a minimal check (% of live predictions where p_home != p_home_raw; % of rows with ml_home_pin not-NULL) but it has not been built.",Build the minimal monitor the project's own blueprint already specifies and alert when either percentage drops to zero.,"New test/monitor script asserting the live percentage metrics are computed and exposed somewhere (dashboard/log line), triggered on a synthetic all-identity-calibration dataset to confirm it would have caught REG-007-style failures."
DUP-001,Low,duplicated-logic/travel,backtest_and_retrain.py; data_fetchers.py,_travel_stats()/_geodesic_miles(); MLBStatsAPI.get_travel_fatigue(),backtest_and_retrain.py:265-278; data_fetchers.py:1013-1091,both,"Two independent travel-distance implementations exist: data_fetchers.py's schedule-aware, stadium-coordinate haversine (live) and backtest_and_retrain.py's static team-city-coordinate geodesic lookup (backtest). This may be an intentional performance/simplicity trade-off (backtest processes thousands of games and can't afford per-game schedule lookups) rather than a bug, but it is one more instance of this codebase's recurring duplicated-constant/logic pattern (5th+ instance found this audit) and was not previously documented as a deliberate design choice anywhere read this pass.",Document explicitly (in-code or in CONTRACTS.md) that these are intentionally different implementations for performance reasons if that is indeed the reason; otherwise unify.,"None - informational; if unified, a test comparing both implementations' output for a sample of real team pairs would lock in the chosen single source of truth."
INV-001,Low,repo-hygiene,.codex/,N/A,N/A,N/A,".codex/ (197MB, untracked) is a separate CLI tool's home directory sitting inside the repo root, not covered by .gitignore (unlike .claude/). Zero references from any project .py file.",Add .codex/ to .gitignore or relocate it outside the repo root.,None needed - pure hygiene.
INV-002,Low,repo-hygiene,data/game_outcomes.db; data/mlb_learning.db,N/A,N/A,N/A,"Two stale/orphaned database files sit in data/: game_outcomes.db (0 bytes, empty, last modified May 24, zero references in any .py file) and mlb_learning.db (stale since May 27, contains an old copy of the game_outcomes/kalman_state/ml_state schema, not the active DB). Both are .gitignore'd so pose no repo risk, only a human/agent confusion risk when asked to inspect 'the game_outcomes database.'",Delete both stale files or rename with an explicit ARCHIVED_ prefix.,None needed - pure hygiene.
INV-003,Low,dead-code,weight_optimizer.py,N/A,N/A,N/A,"weight_optimizer.py has zero external callers (confirmed via grep - the file only references itself), consistent with CLAUDE2.md's own note flagging it as a possible removal candidate.",Confirm no external use and delete, or document its exploratory purpose if intentionally kept as a standalone research script.,None needed - if removed, the 462-test suite passing is sufficient confirmation nothing depended on it.
```

<!-- ============================================================ -->
<!-- FILE: hypotheses.md -->
<!-- ============================================================ -->

# Hypotheses — unproven suspicions, excluded from findings.csv

These are things noticed during the audit that are plausible but were not confirmed with
direct evidence within this pass's budget. They are not findings — do not treat any of these
as verified defects.

1. **Doubleheader intra-day SQL row order vs. true kickoff order.** §8 establishes that the
   backtest's walk-forward bias cutoff is date-granular and therefore leak-safe regardless of
   intra-day processing order, but Kalman state updates happen sequentially as the single
   loop pass proceeds, with no secondary sort key (e.g. `game_pk` or an actual start-time
   column) breaking ties within a `game_date`. If SQLite's default return order for
   same-date rows ever put a doubleheader's Game 2 before Game 1, Game 2's Kalman-based
   prediction would be computed without Game 1's real result — the *safe* direction (under-
   informed, not leaked), but a fidelity question, not confirmed to actually occur (would
   require cross-referencing real first-pitch times not present in `game_outcomes`).

2. **Roof-status failure-path default.** `CONTRACTS.md` describes the pre-2026-07-06 bug as
   defaulting `roof_closed=True` for all 8 retractable-roof parks when nothing populated the
   field. Not independently re-verified this pass whether the *current* failure path (if
   `get_roof_status()` itself fails/throws, as opposed to simply never being called) still
   defaults conservatively closed or has since changed.

3. **Fuzzy team-name substring matching edge case.** `get_best_odds_for_teams()`'s matching
   (`home_team.lower() in g_home.lower() or g_home.lower() in home_team.lower()`) could in
   principle collide for a team name that is a substring of an unrelated team's name. No
   actual MLB team-name pair was found or checked that would trigger this; flagged as a
   theoretical edge case only.

4. **Whether backtest report JSON self-declares which PIT flags were active.** §4 raises the
   concern that a reader could pick up an old, non-PIT-flagged report from `reports/baseline/`
   and mistake it for the current PIT-safe baseline. Not confirmed either way whether
   `generate_report()`'s JSON payload already includes the active CLI flags as a field (which
   would mitigate this) — would require reading the full `generate_report()` function, out of
   budget this pass.

5. **Silent-exception sweep is not exhaustive.** §12 notes every `except Exception` block
   directly read during sections 1-11 logs before falling back, but this was not a dedicated,
   repo-wide grep-and-classify pass across every file (P2-level depth was intentionally not
   spent here per the audit's own priority rules). No specific instance of a fully silent
   `except: pass` was found, but the absence of a targeted search means this is "not found,"
   not "confirmed absent."

6. **`advanced_pit_enrichment/`'s remaining ~24 files** (of 26 total) were not re-walked
   file-by-file this pass, beyond the 2 that changed in the current dirty working tree
   (`tte_pit_adapter.py`, `team_defense_pit_builder.py`). The project's own blueprint claims a
   completed formal audit of all 26 (2026-07-09, "1 gap found and fixed in
   `tte_prior_baseline_builder.py`") — carried forward as documented, not independently
   re-verified.

7. **PIT cache concurrency/isolation** — whether two concurrent backtest processes could
   corrupt a shared `pit_cache_*.db` file the way `ml_state` was once corrupted (REG-003).
   The `ml_state` lockfile (`backtest_ml_state.lock`, §12) is confirmed to exist and cover
   `ml_state` specifically; whether an equivalent protection exists for the PIT cache files
   was not checked.

8. **Whether the dampening formula in `compute_team_bias_kalman_adjusted` (REG-006) has a
   correct alternative derivation** that wasn't tried in the two reverted attempts. The
   project's own docstring frames this as an open research question ("does explicit, tuned
   shrinkage beat this implicit-shrinkage design") — this audit did not attempt a third fix
   (out of scope for a read-only audit) and takes no position on whether one exists.
