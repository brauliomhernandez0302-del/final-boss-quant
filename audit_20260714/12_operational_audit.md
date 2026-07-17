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
