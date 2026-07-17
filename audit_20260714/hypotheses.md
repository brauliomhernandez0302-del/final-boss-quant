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
