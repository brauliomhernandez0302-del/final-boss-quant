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
