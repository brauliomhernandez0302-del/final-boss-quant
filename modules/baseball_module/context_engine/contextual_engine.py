"""
CONTEXTUAL ENGINE — Game-night binary context factors
=====================================================

Consolidates small, game-specific adjustments that are:
  (a) Known only on game day, not from season stats.
  (b) Binary or discrete — yes/no, integer rest days.

Factors:
  1. Rest / fatigue — per team, asymmetric
       B2B home (rest_days == 0): 0 % — empirical audit: actual +7.87 % (FIX D4)
       B2B away (rest_days == 0): −4 % on λ_away — empirical: −4.41 % ✓
       1–2 days                 : neutral (optimal MLB rhythm)

  Removed (E1/E2 cleanup):
       Rust factor (rest_days ≥ 3): 0 activations across 5,422 games — dead code.
       Umpire zone factor: umpire_stats never populated in pipeline — always 1.0.

Why here and not in AutoCalibrator?
  AutoCalibrator (now removed — calibration/auto_calibrator.py no longer
  exists; superseded by the Kalman + multidim_bias system in
  learning_engine.py) used to apply a ±8 % hard cap on form + season
  context. Moving rest out gave it clean, uncapped accounting at its true
  empirical value without competing for budget inside that cap — a
  decision that still holds even though AutoCalibrator itself is gone.

Pipeline position: PASO 3 (after Pitcher Engine, before Bullpen).
  Intentional: the F5 (first-5-innings) snapshot is taken right after this
  engine runs, since starters — not the bullpen — cover the first 5 innings,
  so rest/B2B (which affects the starter's team, not bullpen usage) belongs
  in the F5 signal while Bullpen adjustments correctly don't.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_B2B_MULT_AWAY = 0.960   # −4 % for away B2B (empirical: −4.41 %, correct direction)
_B2B_MULT_HOME = 1.000   # home B2B: audit confirmed +7.87 % actual vs −4 % model — neutralized


# ── Engine ────────────────────────────────────────────────────────────────────

class ContextualEngine:
    """
    Applies rest/fatigue adjustments to (λ_home, λ_away).
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
                       and optionally 'back_to_back_home/away'.

        Returns:
            (lh_adjusted, la_adjusted, metadata)
        """
        home = game_data.get("home_team") or {}
        away = game_data.get("away_team") or {}

        home_rest = self._rest_days(home, game_data.get("back_to_back_home", False), is_home=True)
        away_rest = self._rest_days(away, game_data.get("back_to_back_away", False), is_home=False)

        lh_new = lh * home_rest.mult
        la_new = la * away_rest.mult

        meta = {
            "home_rest_days":   home_rest.days,
            "home_rest_mult":   round(home_rest.mult, 4),
            "home_rest_reason": home_rest.reason,
            "away_rest_days":   away_rest.days,
            "away_rest_mult":   round(away_rest.mult, 4),
            "away_rest_reason": away_rest.reason,
        }

        log.info(
            "ContextualEngine | home_rest=%s(×%.3f)  away_rest=%s(×%.3f) → λ_h=%.3f  λ_a=%.3f",
            home_rest.reason, home_rest.mult,
            away_rest.reason, away_rest.mult,
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

    def _rest_days(self, team: Dict, back_to_back_flag: bool, is_home: bool = False) -> "_RestResult":
        """
        Determine rest multiplier for one team.

        Priority:
          1. back_to_back_flag=True → rest=0 (B2B), regardless of rest_days.
             The flag is set from schedule data and is more specific than the
             default rest_days=1 placeholder.  An explicit API rest_days=0 is
             consistent; rest_days=1 is the default and should be overridden.
          2. rest_days from team dict (API-populated) for non-B2B cases.
          3. Absent/None → default 1 (optimal MLB rhythm).

        B2B multipliers differ by side (FIX D4):
          Home B2B: 1.000 — audit showed actual +7.87 % effect (homestand confound)
          Away B2B: 0.960 — audit confirmed −4.41 % actual, model −4 % ✓
        """
        rest = team.get("rest_days")

        if back_to_back_flag and (rest is None or rest <= 1):
            rest = 0

        rest = int(rest) if rest is not None else 1

        if rest == 0:
            b2b_mult = _B2B_MULT_HOME if is_home else _B2B_MULT_AWAY
            return self._RestResult(rest, b2b_mult, "b2b")
        return self._RestResult(rest, 1.0, "optimal")



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
