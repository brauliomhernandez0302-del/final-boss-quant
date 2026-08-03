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
