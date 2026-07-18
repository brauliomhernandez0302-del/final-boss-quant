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
**Errata (2026-07-18, roadmap Step 2 Commit C)**: this entry's premise was itself doc drift,
carried forward uncritically from `AUDITORIA_MLB_2026-07.md` (2026-07-06) without
re-verification against the current code. `recalibrate_platt_2d` has **zero call sites**
in `backtest_and_retrain.py` — confirmed repeatedly this session by direct grep (see
`audit_20260714/chron002_commitB_enumeration.md`) — it is exclusively invoked from the live
path (`get_platt_2d_params` → `run_module.py`). There is nothing to "run after the main
backtest loop, not interleaved" for `recalibrate_platt_2d` specifically, because it never
runs in the backtest at all. `recalibrate_platt` (1D) genuinely does run inside the backtest
(mid-run season-boundary refits, end-of-run per-season refit — see CHRON-002's commit), and
*that* half of the original claim holds. **Status: CORRECTED** — `recalibrate_platt_2d`
removed from this claim; `recalibrate_platt`'s non-interleaved, non-contaminating behavior
remains confirmed (verified directly this session via the CHRON-001/CHRON-002 identity gates,
which re-ran the full backtest and reproduced byte-identical reports).

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
