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
