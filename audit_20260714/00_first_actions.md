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
