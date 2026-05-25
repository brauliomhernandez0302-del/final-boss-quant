"""
PITCHER ENGINE — Starting-pitcher quality and game-context adjustments
======================================================================

Adjusts λ_home / λ_away based on the STARTING PITCHER's characteristics.
Bullpen adjustments are handled by a separate Bullpen Engine (not this file).

Responsibilities:
  Starter quality    — SIERA → xFIP → xERA → FIP → ERA composite
                       + Statcast xwOBA allowed, barrel%, and K%-BB% overlays
                       + Bayesian regression to mean based on innings pitched
  Recent form        — era_last_5 level vs season, ERA slope trend, QS%
  Matchup history    — era_vs_opp shrunk toward season ERA at < 15 IP
  Platoon splits     — L/R WHIP split × opposing lineup handedness composition
  Fatigue            — smooth gradient over days-rest and last-start pitch count

Does NOT handle:
  Home field advantage           — HFA Engine
  Ballpark / park factor         — HFA Engine (or future Park Engine)
  Team offense / defense         — True Talent Engine + AutoCalibrator
  Bullpen quality / workload     — Bullpen Engine (separate)
  Travel fatigue of position players — HFA Engine
"""

from typing import Dict, Any, Tuple
import logging
from config import PITCHER_ENGINE_WEIGHTS, LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP

logger = logging.getLogger(__name__)

# League averages for normalisation (2024/2025 combined)
_LG_ERA            = LEAGUE_AVG_ERA     # 4.15
_LG_WHIP           = LEAGUE_AVG_WHIP    # 1.30
_LG_K_PCT          = 0.220              # starter league-avg K%
_LG_BB_PCT         = 0.080              # starter league-avg BB%
_LG_K_BB           = _LG_K_PCT - _LG_BB_PCT   # 0.140
_LG_XWOBA_ALLOWED  = 0.320              # Statcast xwOBA allowed, starter avg
_LG_BRL_PCT        = 8.0               # barrel% allowed, starter avg

# Bayesian stabilisation constant for ERA estimators (TBF at 50% reliability).
# Research-based: FIP/xFIP stabilise around 300-400 TBF; SIERA around 250.
# Using 350 as a reasonable average across the fallback chain.
_K_TBF_ERA   = 350

# Away pitchers allow ~0.15 more ERA when pitching away from home.
# Source: documented across multiple baseball research papers (e.g., Baseball Prospectus, FG).
# Applied as a context adjustment on top of the Bayesian-regressed quality estimate.
_AWAY_ERA_PENALTY = 0.15


class PitcherEngine:
    """
    Adjusts Poisson λ values for the starting pitcher matchup.
    Each sub-factor returns a multiplier around 1.0; factors are combined
    via the delta formula: 1 + Σ((factor − 1) × weight).
    """

    def __init__(self):
        self.name    = "PitcherEngine"
        self.weights = dict(PITCHER_ENGINE_WEIGHTS)

    def adjust_for_pitchers(
        self,
        lh: float,
        la: float,
        game_data: Dict[str, Any],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Adjusts λ_home and λ_away for the starter matchup.

        Away pitcher faces the home lineup  → adjusts λ_home.
        Home pitcher faces the away lineup  → adjusts λ_away.

        Returns (lh_adjusted, la_adjusted, metadata).
        """
        logger.info("Pitcher Engine — adjusting lambdas")
        logger.info("   Input: λ_h=%.3f  λ_a=%.3f", lh, la)

        pitcher_home = game_data.get("pitcher_home") or {}
        pitcher_away = game_data.get("pitcher_away") or {}

        # Away starter holds down the home lineup
        adj_away = self._calculate_pitcher_adjustment(pitcher_away, game_data, is_home=False)
        lh_new   = lh * adj_away["total_multiplier"]

        # Home starter holds down the away lineup
        adj_home = self._calculate_pitcher_adjustment(pitcher_home, game_data, is_home=True)
        la_new   = la * adj_home["total_multiplier"]

        logger.info(
            "   Away starter (%s): mult=%.3f  λ_home %.3f → %.3f",
            pitcher_away.get("name", "?"), adj_away["total_multiplier"], lh, lh_new,
        )
        logger.info(
            "   Home starter (%s): mult=%.3f  λ_away %.3f → %.3f",
            pitcher_home.get("name", "?"), adj_home["total_multiplier"], la, la_new,
        )

        metadata = {
            "pitcher_home": adj_home,
            "pitcher_away": adj_away,
        }
        return lh_new, la_new, metadata

    # ── Factor dispatch ────────────────────────────────────────────────────────

    def _calculate_pitcher_adjustment(
        self,
        pitcher: Dict[str, Any],
        game_data: Dict[str, Any],
        is_home: bool,
    ) -> Dict[str, Any]:
        quality  = self._adjust_pitcher_quality(pitcher, is_home=is_home)
        form     = self._adjust_pitcher_form(pitcher)
        matchup  = self._adjust_pitcher_matchup(pitcher, game_data, is_home)
        platoon  = self._adjust_pitcher_platoon(pitcher, game_data, is_home)
        fatigue  = self._adjust_pitcher_fatigue(pitcher)

        # Delta-weighted combination: preserves the direction and magnitude of each
        # sub-factor while mixing them proportionally by their assigned weights.
        total = 1.0 + (
            (quality - 1.0) * self.weights["pitcher_quality"] +
            (form    - 1.0) * self.weights["pitcher_form"]    +
            (matchup - 1.0) * self.weights["pitcher_matchup"] +
            (platoon - 1.0) * self.weights["pitcher_platoon"] +
            (fatigue - 1.0) * self.weights["pitcher_fatigue"]
        )

        return {
            "pitcher_name":     pitcher.get("name", "Unknown"),
            "quality_mult":     quality,
            "form_mult":        form,
            "matchup_mult":     matchup,
            "platoon_mult":     platoon,
            "fatigue_mult":     fatigue,
            "total_multiplier": max(0.65, min(1.45, total)),
        }

    # ── Quality ───────────────────────────────────────────────────────────────

    def _adjust_pitcher_quality(self, pitcher: Dict, is_home: bool = True) -> float:
        """
        Starter quality multiplier combining four orthogonal signals:

        1. ERA estimator (SIERA → xFIP → xERA → FIP → ERA)
           Bayesian-regressed toward league average based on IP (TBF proxy),
           so small samples early in the season don't overfit.

        2. Statcast xwOBA allowed — expected batting value vs this pitcher
           per PA (removes BABIP luck from batted-ball outcomes).

        3. Statcast barrel% allowed — predicts HR/XBH better than HR/9.

        4. K%-BB% differential — most reliably stabilises and predicts
           future ERA; reflects true command and swing-and-miss ability.

        Output clamped to [0.70, 1.35].
        """
        _era  = pitcher.get("era")
        era   = float(_era if _era is not None else _LG_ERA)
        fip   = pitcher.get("fip")
        xfip  = pitcher.get("xfip")
        xera  = pitcher.get("xera")
        siera = pitcher.get("siera")

        # Best available ERA estimator (most predictive → least)
        primary = float(
            siera if siera is not None else
            xfip  if xfip  is not None else
            xera  if xera  is not None else
            fip   if fip   is not None else
            era
        )

        # Bayesian regression: regress primary toward league avg based on IP sample.
        # At IP=0 → 100% league avg. At IP≈81 (350 TBF) → 50/50. At IP=∞ → raw value.
        ip_cur   = float(pitcher.get("innings_pitched") or 0)
        tbf_est  = max(0.0, ip_cur * 4.3)   # ~4.3 TBF per IP for starters
        shrink_w = _K_TBF_ERA / (_K_TBF_ERA + tbf_est)
        primary_reg = primary * (1.0 - shrink_w) + _LG_ERA * shrink_w
        if not is_home:
            primary_reg += _AWAY_ERA_PENALTY
        skill_mult  = primary_reg / _LG_ERA

        # xwOBA allowed overlay — each 0.010 above avg ≈ +3% runs allowed
        est_woba = pitcher.get("est_woba")
        woba_mult = (
            1.0 + (float(est_woba) - _LG_XWOBA_ALLOWED) * 3.0
            if est_woba is not None else 1.0
        )

        # Barrel% allowed overlay — each 1 ppt above avg ≈ +1.2% runs allowed
        brl_pct = pitcher.get("brl_percent")
        brl_mult = (
            1.0 + max(0.0, float(brl_pct) - _LG_BRL_PCT) * 0.012
            if brl_pct is not None else 1.0
        )

        # K%-BB% overlay — elite command (high K, low BB) means fewer runs
        # Each 1% above league avg K-BB (14%) → ~1.5% fewer runs allowed
        k_pct  = pitcher.get("k_pct")
        bb_pct = pitcher.get("bb_pct")
        if k_pct is not None and bb_pct is not None:
            k_bb_diff = (float(k_pct) - float(bb_pct)) - _LG_K_BB
            kbb_mult  = max(0.88, min(1.12, 1.0 - k_bb_diff * 1.5))
        else:
            kbb_mult = 1.0

        multiplier = max(0.70, min(1.35, skill_mult * woba_mult * brl_mult * kbb_mult))
        logger.debug(
            "   quality: prim=%.2f→%.2f(reg)  xwOBA=%.3f  brl=%.1f  kbb_mult=%.3f  → %.3f",
            primary, primary_reg,
            float(est_woba) if est_woba is not None else _LG_XWOBA_ALLOWED,
            float(brl_pct)  if brl_pct  is not None else _LG_BRL_PCT,
            kbb_mult, multiplier,
        )
        return multiplier

    # ── Form ──────────────────────────────────────────────────────────────────

    def _adjust_pitcher_form(self, pitcher: Dict) -> float:
        """
        Three recent-form signals combined multiplicatively:
          1. ERA level  — era_last_5 vs season ERA
          2. ERA trend  — signed slope of per-start ERA (improving vs worsening)
          3. QS%        — fraction of recent starts that were quality starts

        Clamped to [0.85, 1.15].
        """
        _era_last  = pitcher.get("era_last_5")
        _era_seas  = pitcher.get("era")
        season_era = float(_era_seas  if _era_seas  is not None else _LG_ERA)
        recent_era = float(_era_last  if _era_last  is not None else season_era)

        level_diff = recent_era - season_era
        level_adj  = 1.0 + level_diff * 0.06

        era_trend = pitcher.get("era_trend", 0.0)
        trend_adj = 1.0 + float(era_trend or 0.0) * 0.03

        qs_pct  = pitcher.get("quality_start_pct", 0.50)
        qs_adj  = 1.0 - (float(qs_pct or 0.50) - 0.50) * 0.06

        form_adj = level_adj * trend_adj * qs_adj
        return max(0.85, min(1.15, form_adj))

    # ── Matchup ───────────────────────────────────────────────────────────────

    def _adjust_pitcher_matchup(
        self,
        pitcher: Dict,
        game_data: Dict,
        is_home: bool,
    ) -> float:
        """
        Historical ERA vs today's opponent, shrunk toward season ERA
        when sample is < 15 IP.
        """
        era_vs_team = pitcher.get("era_vs_opp")
        if era_vs_team is None:
            return 1.0

        _se         = pitcher.get("era")
        season_era  = float(_se if _se is not None else _LG_ERA)
        _ip         = pitcher.get("ip_vs_opp")
        ip_vs_opp   = float(_ip if _ip is not None else 0)

        if ip_vs_opp < 15:
            w = ip_vs_opp / 15.0
            era_vs_team = float(era_vs_team) * w + season_era * (1.0 - w)

        diff = float(era_vs_team) - season_era
        return max(0.85, min(1.15, 1.0 + diff * 0.06))

    # ── Platoon ───────────────────────────────────────────────────────────────

    def _adjust_pitcher_platoon(
        self,
        pitcher: Dict,
        game_data: Dict,
        is_home: bool,
    ) -> float:
        """
        WHIP-based platoon mismatch: lineup-composition-weighted WHIP vs overall WHIP.
        Home pitcher faces away lineup (away_lineup_lhb_pct) and vice versa.
        Returns 1.0 when split data is insufficient.
        """
        platoon = pitcher.get("platoon_splits")
        if not platoon:
            return 1.0

        vs_lhb = platoon.get("vs_lhb") or {}
        vs_rhb = platoon.get("vs_rhb") or {}
        if not vs_lhb or not vs_rhb:
            return 1.0

        _lhb_key    = "away_lineup_lhb_pct" if is_home else "home_lineup_lhb_pct"
        _lhb_raw    = game_data.get(_lhb_key)
        opp_lhb_pct = float(_lhb_raw if _lhb_raw is not None else 0.45)
        opp_rhb_pct = 1.0 - opp_lhb_pct

        lhb_whip    = float(vs_lhb.get("whip", _LG_WHIP))
        rhb_whip    = float(vs_rhb.get("whip", _LG_WHIP))
        lineup_whip = opp_lhb_pct * lhb_whip + opp_rhb_pct * rhb_whip

        overall_whip = float(pitcher.get("whip", _LG_WHIP))
        if overall_whip <= 0:
            return 1.0

        return max(0.93, min(1.07, lineup_whip / overall_whip))

    # ── Fatigue ───────────────────────────────────────────────────────────────

    def _adjust_pitcher_fatigue(self, pitcher: Dict) -> float:
        """
        Smooth fatigue gradient from days rest and last-start pitch count.

        Days rest (optimal window: 4–5 days):
          0 d  → +9.0%  (back-to-back, extremely rare for starters)
          1 d  → +7.5%
          2 d  → +5.0%
          3 d  → +2.5%
          4–5 d → neutral
          6 d  → +0.8%  (rust begins)
          7 d  → +1.6%
          8+ d → +2.4% cap

        Pitch count (above 100 in last start):
          +0.12% per pitch above 100; capped at +4.8% (at 140 pitches).
        """
        _dr       = pitcher.get("days_rest")
        days_rest = int(_dr if _dr is not None else 4)

        if days_rest == 0:
            rest_adj = 1.090
        elif days_rest <= 3:
            rest_adj = 1.0 + (4 - days_rest) * 0.025
        elif days_rest >= 6:
            rest_adj = 1.0 + min((days_rest - 5) * 0.008, 0.024)
        else:
            rest_adj = 1.0   # 4–5 days: optimal

        _lpc     = pitcher.get("last_pitch_count")
        last_pc  = float(_lpc if _lpc is not None else 90)
        pc_mult  = (
            1.0 + min((last_pc - 100.0) * 0.0012, 0.048)
            if last_pc > 100 else 1.0
        )

        return max(0.95, min(1.12, rest_adj * pc_mult))


# ── Module-level helper ────────────────────────────────────────────────────────

def adjust_for_pitchers(
    lh: float,
    la: float,
    game_data: Dict[str, Any],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Convenience wrapper used by run_module.py:
        from context_engine.pitcher_engine import adjust_for_pitchers
        lh, la, meta = adjust_for_pitchers(lh, la, game_data)
    """
    return PitcherEngine().adjust_for_pitchers(lh, la, game_data)
