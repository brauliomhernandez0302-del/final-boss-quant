"""Shared TTE (True Talent Offense) math core.

2026-07-12 (Fable audit finding, unification item #1): `offense/
true_talent_engine.py` (live engine) and `advanced_pit_enrichment/
tte_pit_adapter.py` (backtest PIT path) had independently duplicated copies
of the same Bayesian-regression + composite-score + current/prior-blend
formula — same weights and constants (0.50/0.30/0.20, k=150/120/120/60,
PRIOR_PA_EQUIVALENT=1000), but two separately-maintained implementations
that had already drifted once (the barrel% league constant got fixed in one
copy in a prior session and not the other; found again as a live bug in
`tte_pit_adapter.py` this session). Measured effect of the drift: the two
paths only correlated 0.733 on real per-team λ, with disagreements up to
±0.4-0.5 runs for some teams.

This module is the single source of truth for the MATH only — it takes
already-extracted primitive metrics (xwOBA, barrel rate, BB%/K%, PA) and
league constants as plain arguments. It intentionally does NOT unify how
each caller SOURCES those inputs (player-roster Statcast aggregation vs.
PIT-cache team-event aggregation remain separate, legitimate data layers)
and does NOT change either path's current behavior — this is a pure
refactor, verified byte-identical against both callers' pre-refactor output
before landing.

One known, NOT-yet-unified divergence, left as a follow-up (Fable audit
finding #5, low priority): the live engine regresses barrel% toward its
league mean using `attempts` (batted-ball-events) as the Bayesian sample
size `n`, since barrel% is fundamentally a per-BBE rate; the PIT adapter
currently uses `pa` (plate appearances, ~1.47x attempts) for the same
regression, which under-shrinks it. Fixing this requires first resolving
the known foul-ball inflation of the PIT cache's own `batted_ball_count`
field (see savant_offense_daily_aggregator.py) so that fix doesn't
introduce a corrupted count into a now-load-bearing computation — do not
"fix" this by just switching PIT's n to `bip`/`batted_ball_count` without
addressing that first.
"""

from __future__ import annotations

# Composite weights — three regressed, league-normalized factors.
# f_xwoba (0.60): contact quality net of BABIP luck.
# f_barrel (0.20): power/exit-velocity, predicts future HR.
# f_plate (0.20): BB%-K% discipline, most stable signal.
#
# Re-centered 2026-07-12 (was 0.50/0.30/0.20) — a directional, Fable-gated
# nudge, NOT a full regression fit. An out-of-sample test (n=240 team-cutoff
# observations, 2024+2025, 4 cutoffs/season) found xwOBA ALONE predicts a
# team's rest-of-season RPG better than the full composite in both seasons
# (r=0.559/0.468 vs composite's 0.522/0.426), and barrel-alone is the
# weakest single predictor in both (r=0.394/0.264) — xwOBA already contains
# most of barrel's and plate-discipline's signal by construction (it's
# built from contact quality and walk/strikeout events), so the composite's
# original equal-ish split was diluting a strong, already-inclusive signal
# with two weaker, partially-redundant ones.
#
# A full walk-forward regression fit was attempted first and explicitly
# REJECTED: with only 2 real seasons of data (~30 effective team-season
# units per fold once cutoff-overlap non-independence is accounted for),
# fitting 3-4 free parameters produced wildly fold-inconsistent coefficients
# (e.g. barrel's fitted weight swung from 0.02 to 0.85 depending on which
# season was held out) — the same overfitting-instability failure pattern
# already hit and reverted once this session for a different fix (see
# learning_engine.py's compute_team_bias_kalman_adjusted docstring). This
# constrained, single-degree-of-freedom nudge (only xwoba/barrel traded off,
# plate left untouched since its own signal was inconsistent between
# seasons) is what ~30 effective units per fold can actually support.
# Gated on a full backtest before shipping: Brier 0.24486 (unchanged
# baseline) -> confirm no regression before trusting this number long-term.
XWOBA_WEIGHT = 0.60
BARREL_WEIGHT = 0.20
PLATE_WEIGHT = 0.20

# Plate-discipline factor: 1.0 + (BB%-K% differential vs league) * PLATE_SCALE,
# clamped. Both paths use the same scale and bounds.
PLATE_DISC_SCALE = 3.5
PLATE_FACTOR_MIN = 0.85
PLATE_FACTOR_MAX = 1.15

# Current/prior blend: prior_weight = k / (k + PA_current). At PA=0 -> 100%
# prior; at PA=k -> 50/50; at PA=inf -> 0% prior. k is TEAM-season PA
# (~6,150-6,300 full season), not one batter's PA.
PRIOR_PA_EQUIVALENT = 1000

LAMBDA_MIN = 3.0
LAMBDA_MAX = 7.0


def regress(observed: float, mean: float, n: float, k: float) -> float:
    """Bayesian shrinkage toward `mean`. At n=0 -> mean; at n=k -> 50/50;
    at n=inf -> observed. `n`/`k` must be on the same sample-size basis
    (e.g. both PA, or both batted-ball-events) — mixing bases under- or
    over-shrinks (see this module's docstring for the one known instance
    of this in the codebase, not yet fixed)."""
    if n <= 0:
        return mean
    return (observed * n + mean * k) / (n + k)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def plate_factor(bb_reg: float, k_reg: float, lg_bb_pct: float, lg_k_pct: float) -> float:
    """1.0 = league-average BB%-K% differential."""
    disc = (bb_reg - k_reg) - (lg_bb_pct - lg_k_pct)
    return clamp(1.0 + disc * PLATE_DISC_SCALE, PLATE_FACTOR_MIN, PLATE_FACTOR_MAX)


def composite_score(f_xwoba: float, f_barrel: float, f_plate: float) -> float:
    """Weighted composite of three league-normalized factors (1.0 = average)."""
    return f_xwoba * XWOBA_WEIGHT + f_barrel * BARREL_WEIGHT + f_plate * PLATE_WEIGHT


def season_lambda(
    *,
    xwoba_reg: float,
    barrel_reg: float,
    bb_reg: float,
    k_reg: float,
    lg_xwoba: float,
    lg_barrel_rate: float,
    lg_bb_pct: float,
    lg_k_pct: float,
    lg_rpg: float,
) -> tuple[float, dict[str, float]]:
    """One season's (current OR prior) composite lambda, plus its factors.
    `lg_barrel_rate` must be on the same basis (per-PA or per-BBE) as
    `barrel_reg` — each caller owns that consistency, this function doesn't
    police it."""
    f_xwoba = xwoba_reg / lg_xwoba
    f_barrel = barrel_reg / lg_barrel_rate
    f_plate = plate_factor(bb_reg, k_reg, lg_bb_pct, lg_k_pct)
    composite = composite_score(f_xwoba, f_barrel, f_plate)
    lam = composite * lg_rpg
    return lam, {"f_xwoba": f_xwoba, "f_barrel": f_barrel, "f_plate": f_plate}


def blend_current_prior(
    lambda_cur: float, lambda_prior: float, pa_cur: float,
    prior_pa_equivalent: float = PRIOR_PA_EQUIVALENT,
) -> tuple[float, float, float]:
    """Returns (lambda_blended_clamped, current_weight, prior_weight)."""
    prior_w = prior_pa_equivalent / (prior_pa_equivalent + pa_cur)
    current_w = pa_cur / (prior_pa_equivalent + pa_cur)
    lam = round(current_w * lambda_cur + prior_w * lambda_prior, 4)
    return clamp(lam, LAMBDA_MIN, LAMBDA_MAX), current_w, prior_w
