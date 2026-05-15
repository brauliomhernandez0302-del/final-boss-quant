# ==========================================================
# PITCHER REGRESSION ENGINE — LUCK INDICATORS ONLY
# ==========================================================
# NOT CONNECTED TO THE PIPELINE.
#
# This engine requires per-start BABIP, LOB%, and HR/FB% to be
# meaningful. Neither the MLB Stats API game log endpoint nor
# Ball Don't Lie API (nor any other free API) provides these
# metrics at per-start granularity. FanGraphs only publishes
# season-to-date totals, making it impossible to distinguish
# recent-start luck from season-level drift.
#
# Re-connect (run_module.py PASO 3) only if a per-start data
# source becomes available.
# ==========================================================

from typing import Dict, Tuple
import numpy as np


class PitcherRegressionEngine:
    """
    Adjusts λ for pitcher luck regression using BABIP, LOB%, and HR/FB%
    relative to career baselines.

    factor > 1 → pitcher has been lucky, expect more runs allowed
    factor < 1 → pitcher has been unlucky, expect fewer runs allowed
    """

    def calculate_regression_factor(
        self,
        pitcher_stats: Dict,
        opponent_stats: Dict,
    ) -> Tuple[float, float]:
        """
        Returns (factor, confidence).
        confidence is 0.0–1.0 based on innings pitched and aligned luck signals.
        opponent_stats retained for API compatibility but not used.
        """
        if not pitcher_stats:
            return 1.0, 0.0

        babip         = pitcher_stats.get("babip",             0.285)
        lob_pct       = pitcher_stats.get("lob_pct",           0.738)
        hr_fb_pct     = pitcher_stats.get("hr_fb_pct",         0.106)

        # Career baselines: pitcher-specific when available; otherwise use the
        # empirical FanGraphs 2024–2025 qualified-pitcher league averages
        # (BABIP 0.285, LOB% 0.738, HR/FB 10.6%) derived from 780+ pitcher-seasons.
        # These are materially different from the old "textbook" values
        # (0.300 / 0.720 / 0.125), which created 3:1 lucky/unlucky asymmetry.
        career_babip  = pitcher_stats.get("career_babip",      0.285)
        career_lob    = pitcher_stats.get("career_lob_pct",    0.738)
        career_hr_fb  = pitcher_stats.get("career_hr_fb_pct",  0.106)

        innings_pitched = pitcher_stats.get("innings_pitched", 50)

        factor = 1.0

        # ── BABIP luck ─────────────────────────────────────────────────────────
        # Low pitcher BABIP → batters hitting less on BIP than usual → pitcher lucky
        # High pitcher BABIP → batters hitting more on BIP than usual → pitcher unlucky
        babip_diff = babip - career_babip
        if babip_diff < -0.030:
            factor *= 1.04   # lucky: expect regression (more runs)
        elif babip_diff > 0.030:
            factor *= 0.97   # unlucky: expect improvement (fewer runs)

        # ── LOB% luck ──────────────────────────────────────────────────────────
        # High LOB% → pitcher stranding more runners than career norm → lucky
        # Low LOB%  → pitcher stranding fewer runners than career norm → unlucky
        lob_diff = lob_pct - career_lob
        if lob_diff > 0.030:
            factor *= 1.03   # lucky: expect regression (more runs allowed)
        elif lob_diff < -0.030:
            factor *= 0.98   # unlucky: expect improvement (fewer runs allowed)

        # ── HR/FB% luck ────────────────────────────────────────────────────────
        # Low HR/FB%  → pitcher giving up fewer HRs per flyball than career → lucky
        # High HR/FB% → pitcher giving up more  HRs per flyball than career → unlucky
        hr_fb_diff = hr_fb_pct - career_hr_fb
        if hr_fb_diff > 0.030:
            factor *= 0.98   # unlucky: expect fewer HRs going forward
        elif hr_fb_diff < -0.030:
            factor *= 1.02   # lucky: expect more HRs going forward

        # ── Confidence ─────────────────────────────────────────────────────────
        if innings_pitched >= 100:
            sample_conf = 0.7
        elif innings_pitched >= 60:
            sample_conf = 0.5
        elif innings_pitched >= 30:
            sample_conf = 0.3
        else:
            sample_conf = 0.1

        luck_signals = sum([
            1 if abs(babip_diff)  > 0.020 else 0,
            1 if abs(lob_diff)    > 0.025 else 0,
            1 if abs(hr_fb_diff)  > 0.025 else 0,
        ])
        confidence = min(0.85, sample_conf + luck_signals * 0.15)

        factor = float(np.clip(factor, 0.90, 1.10))
        return round(factor, 3), round(confidence, 2)

    def get_regression_explanation(
        self,
        pitcher_stats: Dict,
        opponent_stats: Dict,
    ) -> str:
        factor, confidence = self.calculate_regression_factor(
            pitcher_stats, opponent_stats
        )

        babip     = pitcher_stats.get("babip",             0.285)
        lob_pct   = pitcher_stats.get("lob_pct",           0.738)
        hr_fb_pct = pitcher_stats.get("hr_fb_pct",         0.106)
        c_babip   = pitcher_stats.get("career_babip",      0.285)
        c_lob     = pitcher_stats.get("career_lob_pct",    0.738)
        c_hr_fb   = pitcher_stats.get("career_hr_fb_pct",  0.106)

        reasons = []
        if babip - c_babip < -0.030:
            reasons.append(f"BABIP {babip:.3f} vs career {c_babip:.3f} (lucky)")
        elif babip - c_babip > 0.030:
            reasons.append(f"BABIP {babip:.3f} vs career {c_babip:.3f} (unlucky)")

        if lob_pct - c_lob > 0.030:
            reasons.append(f"LOB% {lob_pct:.3f} vs career {c_lob:.3f} (lucky)")
        elif lob_pct - c_lob < -0.030:
            reasons.append(f"LOB% {lob_pct:.3f} vs career {c_lob:.3f} (unlucky)")

        if hr_fb_pct - c_hr_fb > 0.030:
            reasons.append(f"HR/FB {hr_fb_pct:.3f} vs career {c_hr_fb:.3f} (unlucky)")
        elif hr_fb_pct - c_hr_fb < -0.030:
            reasons.append(f"HR/FB {hr_fb_pct:.3f} vs career {c_hr_fb:.3f} (lucky)")

        if factor > 1.02:
            direction = "NEGATIVE regression expected (pitcher has been lucky)"
        elif factor < 0.98:
            direction = "POSITIVE regression expected (pitcher has been unlucky)"
        else:
            direction = "Minimal luck regression expected"

        explanation = direction
        if reasons:
            explanation += "\n  Signals: " + "; ".join(reasons)
        explanation += f"\n  Factor: {factor:.3f} | Confidence: {confidence:.2f}"
        return explanation


# ── Helper for run_module ──────────────────────────────────────────────────────

def calculate_pitcher_regression(
    pitcher_stats: Dict,
    opponent_stats: Dict,
) -> Tuple[float, float]:
    """
    Usage:
        factor, conf = calculate_pitcher_regression(pitcher, opp)
        lambda_adjusted = lambda_base * factor
    """
    engine = PitcherRegressionEngine()
    return engine.calculate_regression_factor(pitcher_stats, opponent_stats)
