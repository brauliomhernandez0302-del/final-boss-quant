"""
CONTEXTUAL ENGINE — Game-night binary context factors
=====================================================

Consolidates small, game-specific adjustments that are:
  (a) Known only on game day, not from season stats.
  (b) Binary or discrete — yes/no, integer rest days.

Factors:
  1. Rest / fatigue — per team, asymmetric
       B2B  (rest_days == 0): −4 % on that team's λ
       Rust (rest_days >= 3): −2 % on that team's λ
       1–2 days            : neutral (optimal MLB rhythm)

  2. Home plate umpire zone — symmetric (same multiplier on both λ)
       zone_factor < 1.0 → tight zone → fewer runs
       zone_factor > 1.0 → wide zone  → more runs
       Clipped to [0.96, 1.04]; requires ≥ 4 game sample.

Why here and not in AutoCalibrator?
  AutoCalibrator applies a ±8 % hard cap on form + season context.
  Moving rest out gives it clean, uncapped accounting at its true
  empirical value without competing for budget inside that cap.

Pipeline position: PASO 7 (after Bullpen, before Monte Carlo).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_B2B_MULT      = 0.960   # −4 % for back-to-back (empirical MLB: ~3–5 %)
_RUST_MULT     = 0.980   # −2 % for extended rest ≥ 3 days
_UMP_CLIP_LOW  = 0.960   # umpire zone_factor floor
_UMP_CLIP_HIGH = 1.040   # umpire zone_factor ceiling
_UMP_MIN_GAMES = 4       # minimum sample to trust ump data


# ── Engine ────────────────────────────────────────────────────────────────────

class ContextualEngine:
    """
    Applies rest/fatigue and umpire-zone adjustments to (λ_home, λ_away).
    """

    def adjust(
        self,
        lh: float,
        la: float,
        game_data: Dict[str, Any],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Args:
            lh: Home expected runs (after Bullpen Engine).
            la: Away expected runs (after Bullpen Engine).
            game_data: Contains 'home_team', 'away_team' dicts (with 'rest_days')
                       and optionally 'umpire_stats', 'back_to_back_home/away'.

        Returns:
            (lh_adjusted, la_adjusted, metadata)
        """
        home = game_data.get("home_team") or {}
        away = game_data.get("away_team") or {}

        # ── 1. Rest / fatigue (asymmetric) ───────────────────────────────────
        home_rest = self._rest_days(home, game_data.get("back_to_back_home", False))
        away_rest = self._rest_days(away, game_data.get("back_to_back_away", False))

        lh_new = lh * home_rest.mult
        la_new = la * away_rest.mult

        # ── 2. Umpire zone (symmetric) ────────────────────────────────────────
        ump_meta = self._umpire_factor(game_data)
        lh_new  *= ump_meta["factor"]
        la_new  *= ump_meta["factor"]

        meta = {
            "home_rest_days":  home_rest.days,
            "home_rest_mult":  round(home_rest.mult, 4),
            "home_rest_reason": home_rest.reason,
            "away_rest_days":  away_rest.days,
            "away_rest_mult":  round(away_rest.mult, 4),
            "away_rest_reason": away_rest.reason,
            "umpire":          ump_meta,
        }

        log.info(
            "ContextualEngine | home_rest=%s(×%.3f)  away_rest=%s(×%.3f)"
            "  ump=%s(×%.3f) → λ_h=%.3f  λ_a=%.3f",
            home_rest.reason, home_rest.mult,
            away_rest.reason, away_rest.mult,
            ump_meta.get("name", "?"), ump_meta["factor"],
            lh_new, la_new,
        )
        return lh_new, la_new, meta

    # ── Helpers ───────────────────────────────────────────────────────────────

    class _RestResult:
        __slots__ = ("days", "mult", "reason")
        def __init__(self, days: int, mult: float, reason: str):
            self.days   = days
            self.mult   = mult
            self.reason = reason

    def _rest_days(self, team: Dict, back_to_back_flag: bool) -> "_RestResult":
        """
        Determine rest multiplier for one team.

        Priority: rest_days field (API-populated integer) → back_to_back flag.
        Default rest_days = 1 when absent (neutral — optimal MLB rhythm).
        """
        rest = team.get("rest_days")

        # Infer B2B from explicit flag when rest_days is absent or defaulted
        if rest is None and back_to_back_flag:
            rest = 0

        rest = int(rest) if rest is not None else 1

        if rest == 0:
            return self._RestResult(rest, _B2B_MULT, "b2b")
        if rest >= 3:
            return self._RestResult(rest, _RUST_MULT, "rust")
        return self._RestResult(rest, 1.0, "optimal")

    def _umpire_factor(self, game_data: Dict) -> Dict:
        """
        Return clipped zone_factor and metadata for the HP umpire.
        Returns factor=1.0 when umpire data is absent or sample too small.
        """
        stats = game_data.get("umpire_stats") or {}
        _gw   = stats.get("games_worked"); games = int(_gw if _gw is not None else 0)

        if games < _UMP_MIN_GAMES:
            return {
                "name":   game_data.get("hp_umpire_name", "unknown"),
                "factor": 1.0,
                "games":  games,
                "skipped": True,
            }

        _zf          = stats.get("zone_factor"); raw_factor = float(_zf if _zf is not None else 1.0)
        clipped      = max(_UMP_CLIP_LOW, min(_UMP_CLIP_HIGH, raw_factor))
        strike_pct   = stats.get("strike_pct")

        return {
            "name":       game_data.get("hp_umpire_name", "unknown"),
            "factor":     round(clipped, 4),
            "raw_factor": round(raw_factor, 4),
            "games":      games,
            "strike_pct": round(float(strike_pct), 4) if strike_pct is not None else None,
            "clipped":    clipped != raw_factor,
            "skipped":    False,
        }


# ── Module-level singleton + public interface ─────────────────────────────────

_engine: Optional[ContextualEngine] = None


def adjust_for_context(
    lh: float,
    la: float,
    game_data: Dict[str, Any],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Convenience wrapper for run_module.py:
        from context_engine.contextual_engine import adjust_for_context
        lh, la, meta = adjust_for_context(lh, la, game_data)
    """
    global _engine
    if _engine is None:
        _engine = ContextualEngine()
    return _engine.adjust(lh, la, game_data)
