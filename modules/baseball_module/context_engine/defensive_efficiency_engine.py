"""
DEFENSIVE EFFICIENCY ENGINE — Fielding-pure lambda adjustment
=============================================================

Answers one question per team:
    "How well does this team's FIELDING convert balls-in-play to outs?"

This is FIELDING ONLY — not pitching.  ERA/WHIP include both; DER and OAA
isolate the fielding signal from the pitching signal.  Mixing them (as the
old defense_mult in AutoCalibrator did) created multicollinearity: ERA was
already priced into the Pitcher Engine, so adding ERA-based defense_mult
double-counted strikeout-heavy/weak-contact staffs.

Two inputs:
  1. DER (Defensive Efficiency Ratio) = 1 − BABIP_allowed
     ≈ fraction of BIP converted to outs (fielding + sequencing noise).
     Available from MLB Stats API pitching group (computed from H, AB, K, HR, SF).
     Stabilises at ~500 BIP (one team season ≈ 4,200 BIP; stabilises fast).

  2. OAA (Outs Above Average) — Statcast per-team aggregate per season.
     Isolates RANGE (hardest fielding component); ~0.80 runs saved per OAA.
     Available from Baseball Savant team OAA leaderboard CSV.
     Optional — engine degrades gracefully to DER-only when absent.

Convention (consistent with Bullpen and HFA engines):
  • Home defence → adjusts λ_away  (home fielders face away batters)
  • Away defence → adjusts λ_home  (away fielders face home batters)

Output range: ±8% (hard cap, expanded from ±5% — see _MAX_DEF_ADJ below).
DER varies ≈ 0.68–0.74 across MLB; ±0.015 DER → ±2.0% λ change ×
innings-weight, well within cap.

Pipeline position: PASO 6 (after Park+Weather, before HFA).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from config import LEAGUE_AVG_RUNS

log = logging.getLogger(__name__)

# ── League constants ─────────────────────────────────────────────────────────
# Empirically measured 2026-07-05 via live MLB Stats API across all 30 teams:
# 2024 avg DER=0.7097, 2025 avg DER=0.7097 (identical). The prior 0.715 value
# was stale/wrong even for the 2024 season it claimed to represent — a
# regression screen on the clean PIT backtest (4,830 games) found this
# systematically skewed the defense stage factor above 1.0 for most teams
# (p=0.0006 unexplained residual correlation, see run_module.py's
# stage_factors_naming TODO near the defense engine call).
_LG_DER       = 0.7097   # 2024-2025 MLB average (1 − BABIP_allowed ≈ 0.2903)
_LG_OAA_RUN   = 0.80     # runs saved per net OAA (Dewan/StatCast research)
_BIP_PER_GAME = 25.9     # used to estimate games-played from BIP (see OAA fix below)

# Bayesian stabilisation: DER stabilises ≈500 BIP; BIP/game ≈ 26 → ~19 games
# k=500 BIP; at 1 full team-season (4,200 BIP) → ~89% observed, 11% prior
_K_BIP_DER    = 500

# OAA Bayesian stabilisation (games-played-equivalent). No citation found for
# team-aggregate OAA specifically (unlike _K_BIP_DER, which has one) — set
# conservatively above DER's ~19-game-equivalent k since this file's own
# docstring already calls OAA "the hardest fielding component" to measure,
# i.e. presumed noisier than DER at the same sample size. Flagged 2026-07-05
# as an interim estimate needing proper derivation against the clean
# backtest, same category as bullpen_engine.py's still-missing _K_BARREL_BP —
# implemented now because fixing the OAA/games-played bug below (see
# games_played_est) supplies the sample-size input this needs, but the
# constant itself is a placeholder, not a researched value.
_K_GAMES_OAA  = 40

# Weight split when both metrics are available
_DER_WEIGHT   = 0.55
_OAA_WEIGHT   = 0.45

# Hard cap on total fielding adjustment: ±8 % (B2: expanded from ±5%)
# Public alias for tests and external callers.
_MAX_DEF_ADJ  = 0.08
MAX_DEF_ADJ   = _MAX_DEF_ADJ


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TeamDefense:
    """Fielding metrics for one team, one season."""
    team_id:    int   = 0
    team_name:  str   = ""
    der:        float = _LG_DER     # observed DER
    bip:        int   = 0           # balls-in-play sample size
    oaa:        Optional[float] = None   # Outs Above Average (season total, net)
    der_regressed: float = _LG_DER  # Bayesian-regressed DER (populated by engine)


# ── Engine ────────────────────────────────────────────────────────────────────

class DefensiveEfficiencyEngine:
    """
    Converts team fielding quality into a symmetric λ multiplier on the
    OPPONENT's expected run total.

    Strong home defence → lower λ_away.
    Strong away defence → lower λ_home.
    """

    def __init__(self) -> None:
        self.name = "DefensiveEfficiencyEngine"

    # ── Public interface ─────────────────────────────────────────────────────

    def adjust_for_defense(
        self,
        lh: float,
        la: float,
        game_data: Dict[str, Any],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Apply fielding adjustments to (lh, la).

        Args:
            lh: Home expected runs after HFA engine.
            la: Away expected runs after HFA engine.
            game_data: Must contain 'defense_home' and/or 'defense_away' dicts
                       with keys: 'der', 'bip', 'oaa' (optional).

        Returns:
            (lh_adjusted, la_adjusted, metadata)
        """
        home_def = game_data.get("defense_home") or {}
        away_def = game_data.get("defense_away") or {}

        home_td = self._build_team_defense("home", home_def)
        away_td = self._build_team_defense("away", away_def)

        # Home defence → multiplier on λ_away (away batters face home fielders)
        home_mult, home_meta = self._fielding_mult(home_td, label="home")
        # Away defence → multiplier on λ_home (home batters face away fielders)
        away_mult, away_meta = self._fielding_mult(away_td, label="away")

        lh_new = lh * away_mult  # cap already enforced inside _fielding_mult
        la_new = la * home_mult

        meta = {
            "home_defense": home_meta,
            "away_defense": away_meta,
            "home_mult_on_away": round(home_mult, 4),   # reduces λ_away
            "away_mult_on_home": round(away_mult, 4),   # reduces λ_home
        }

        log.info(
            "DefEff | home_def×λ_away: %.3f×%.3f→%.3f  away_def×λ_home: %.3f×%.3f→%.3f",
            home_mult, la, la_new, away_mult, lh, lh_new,
        )
        return lh_new, la_new, meta

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_team_defense(self, side: str, def_dict: Dict) -> TeamDefense:
        """Populate a TeamDefense from the game_data sub-dict."""
        der = float(def_dict.get("der", _LG_DER))
        bip = int(def_dict.get("bip", 0))
        oaa = def_dict.get("oaa")  # None when absent

        # Bayesian regression of DER toward league average
        bip_eff   = max(bip, 0)
        shrink_w  = _K_BIP_DER / (_K_BIP_DER + bip_eff)
        der_reg   = der * (1.0 - shrink_w) + _LG_DER * shrink_w

        td = TeamDefense(
            team_name     = def_dict.get("team_name", side),
            der           = der,
            bip           = bip_eff,
            oaa           = float(oaa) if oaa is not None else None,
            der_regressed = round(der_reg, 4),
        )
        return td

    def _fielding_mult(self, td: TeamDefense, label: str) -> Tuple[float, Dict]:
        """
        Compute the multiplier this defence applies to the OPPONENT's λ.

        multiplier < 1.0 → elite defence → fewer opponent runs
        multiplier > 1.0 → weak defence  → more opponent runs
        """
        # ── DER factor ──────────────────────────────────────────────────────
        # LG_DER / regressed_DER: higher DER → lower ratio → fewer runs allowed
        # e.g. DER=0.730 → 0.715/0.730 = 0.979 (−2.1% λ for opponent)
        der_factor = _LG_DER / td.der_regressed if td.der_regressed > 0 else 1.0

        # ── OAA factor (optional) ────────────────────────────────────────────
        oaa_factor = None
        games_played_est = None
        oaa_per_game_reg = None
        if td.oaa is not None:
            # td.oaa is a SEASON-TO-DATE cumulative total from Savant (grows as
            # the season progresses), not a fixed final-season number — dividing
            # by a hardcoded 162 (the old behaviour) understated real per-game
            # impact by up to ~1.8x mid-season and only became correct once the
            # season was nearly over. td.bip already exists on this same
            # TeamDefense object and accumulates in lockstep with games played
            # (this file's own docstring cites BIP/game≈26), so it's a
            # zero-new-plumbing proxy for games-played-so-far. Fixed 2026-07-05.
            games_played_est = max(1.0, td.bip / _BIP_PER_GAME)
            oaa_per_game_raw = td.oaa / games_played_est

            # Bayesian shrinkage toward 0 (OAA is defined relative to league
            # average, so a thin sample should regress toward "no net
            # advantage", not toward some other constant) — same
            # games-played-equivalent sample size used for the fix above.
            # DER gets this kind of shrinkage already (_K_BIP_DER); OAA
            # previously got none at all.
            shrink_w_oaa = _K_GAMES_OAA / (_K_GAMES_OAA + games_played_est)
            oaa_per_game_reg = oaa_per_game_raw * (1.0 - shrink_w_oaa)

            oaa_run_pg = oaa_per_game_reg * _LG_OAA_RUN
            # Positive OAA → fewer runs for opponent → factor < 1
            # Negative OAA → more runs for opponent  → factor > 1
            # Adjust relative to league-average λ; clip to ±8%
            oaa_factor = max(0.92, min(1.08, 1.0 - oaa_run_pg / LEAGUE_AVG_RUNS))

        # ── Combined multiplier ──────────────────────────────────────────────
        if oaa_factor is not None:
            raw_mult = der_factor * _DER_WEIGHT + oaa_factor * _OAA_WEIGHT
        else:
            raw_mult = der_factor   # DER only

        # Apply ±8% hard cap
        mult = max(1.0 - _MAX_DEF_ADJ, min(1.0 + _MAX_DEF_ADJ, raw_mult))

        meta = {
            "der_observed":  round(td.der, 4),
            "der_regressed": round(td.der_regressed, 4),
            "bip_sample":    td.bip,
            "der_factor":    round(der_factor, 4),
            "oaa":           td.oaa,
            "games_played_est": round(games_played_est, 1) if games_played_est is not None else None,
            "oaa_per_game_regressed": round(oaa_per_game_reg, 4) if oaa_per_game_reg is not None else None,
            "oaa_factor":    round(oaa_factor, 4) if oaa_factor is not None else None,
            "raw_mult":      round(raw_mult, 4),
            "final_mult":    round(mult, 4),
        }

        log.debug(
            "DefEff [%s] DER=%.4f(reg=%.4f, BIP=%d) OAA=%s → mult=%.4f",
            label, td.der, td.der_regressed, td.bip,
            f"{td.oaa:.1f}" if td.oaa is not None else "n/a",
            mult,
        )
        return mult, meta


# ── DER calculator (used in data_fetchers.py enrichment) ──────────────────────

def calculate_der(
    hits: int,
    at_bats: int,
    strikeouts: int,
    home_runs: int,
    sac_flies: int = 0,
) -> Tuple[float, int]:
    """
    DER = 1 − BABIP_allowed
    BABIP_allowed = (H − HR) / (AB − K − HR + SF)

    Returns (der, bip) where bip is the balls-in-play sample size.
    """
    bip_numerator   = hits - home_runs
    bip_denominator = at_bats - strikeouts - home_runs + sac_flies

    if bip_denominator <= 0:
        return _LG_DER, 0

    babip = bip_numerator / bip_denominator
    der   = round(1.0 - babip, 4)
    return der, bip_denominator


# ── Module-level singleton + public interface ─────────────────────────────────

_engine: Optional[DefensiveEfficiencyEngine] = None


def adjust_for_defense(
    lh: float,
    la: float,
    game_data: Dict[str, Any],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Convenience wrapper for run_module.py:
        from context_engine.defensive_efficiency_engine import adjust_for_defense
        lh, la, meta = adjust_for_defense(lh, la, game_data)

    Requires game_data to contain 'defense_home' and/or 'defense_away' dicts
    (populated by MLBDataIntegrator or data_fetchers enrichment).
    When both are absent the function returns unchanged lambdas.
    """
    global _engine
    if _engine is None:
        _engine = DefensiveEfficiencyEngine()

    home_def = game_data.get("defense_home") or {}
    away_def = game_data.get("defense_away") or {}
    if not home_def and not away_def:
        return lh, la, {"skipped": "no defense data"}

    return _engine.adjust_for_defense(lh, la, game_data)
