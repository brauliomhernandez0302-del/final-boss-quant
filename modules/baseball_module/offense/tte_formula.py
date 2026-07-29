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

RESUELTO (verificado 2026-07-28, auditoría paso 9) — este párrafo describía
una divergencia pendiente que YA SE ARREGLÓ y su texto anterior era
activamente engañoso: avisaba de un bug inexistente y advertía cómo NO
arreglarlo, cuando el arreglo correcto ya estaba hecho.

Lo que decía: el motor en vivo regresaba barrel% con `attempts` (eventos de
batazo) como n bayesiana, y el adaptador PIT usaba `pa` (~1.47x attempts),
sub-encogiendo. El arreglo estaba bloqueado esperando la corrección de la
inflación por fouls de `batted_ball_count`.

Ese bloqueo se levantó en MATH-002 (2026-07-18) y el arreglo se aplicó en el
mismo paso. Verificado línea por línea: `true_talent_engine.py:669` usa
`attempts_cur`, `tte_pit_adapter.py:206` usa `bip_cur`, y ambos con
`K_BARREL = 120`. Los dos caminos miden lo mismo sobre la misma base.
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
#
# RE-MEDIDO 2026-07-28 (auditoría paso 9), y el resultado cambia la lectura de
# arriba en dos puntos:
#
# 1. EL COMPUESTO YA NO ES MEDIBLEMENTE PEOR QUE xwOBA SOLA — pero tampoco es
#    mediblemente mejor. La comparación citada arriba se hizo con los pesos
#    VIEJOS (0.50/0.30/0.20). Con los actuales, sobre 236 observaciones (equipo ×
#    4 cortes, 2024+2025) prediciendo carreras/juego del RESTO de temporada:
#    compuesto r=+0.5303, xwOBA sola r=+0.5043.
#
#    Esa ventaja de +0.026 NO sobrevive a un bootstrap agrupado por equipo:
#    IC95% = [-0.0135, +0.0752], cruza el cero (90.4% de los remuestreos la
#    favorecen — sugerente, no concluyente). Las 236 observaciones son 30 equipos
#    con 4 cortes cada uno, y los cortes de un mismo equipo-temporada están
#    fuertemente correlacionados; tratarlas como independientes infla cualquier
#    diferencia.
#
#    Lo que SÍ se puede afirmar: el re-centrado quitó una desventaja que estaba
#    medida (el compuesto perdía en AMBAS temporadas con los pesos viejos) y hoy
#    los dos son estadísticamente indistinguibles. Dejar el párrafo de arriba sin
#    esta nota le decía a quien lo leyera que el compuesto sigue siendo peor que
#    su propio componente principal, y eso ya no está sostenido por el dato.
#
# 2. NO MOVER MÁS ESTE PESO SIN DATOS NUEVOS. Barrido de w_xwoba con el resto
#    repartido 50/50, misma muestra:
#
#        w      2024      2025     juntas
#        0.4  +0.5441   +0.4970   +0.5303
#        0.5  +0.5494   +0.4943   +0.5310
#        0.6  +0.5542   +0.4908   +0.5303   <- actual
#        0.7  +0.5584   +0.4863   +0.5278
#        1.0  +0.5636   +0.4657   +0.5043
#
#    Las dos temporadas apuntan en direcciones OPUESTAS y de forma monótona:
#    2024 quiere subirlo hasta 1.0, 2025 quiere bajarlo. La curva conjunta es
#    plana entre 0.4 y 0.7 (rango 0.003 de r). El óptimo NO es identificable con
#    dos temporadas — es la misma inestabilidad que hizo rechazar el ajuste
#    completo, y confirma que el nudge acotado fue la decisión correcta. Mover
#    0.60 a 0.50 "ganaría" 0.0007 de r: ruido de una temporada.
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
