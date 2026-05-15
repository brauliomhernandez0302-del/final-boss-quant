"""
AUTO CALIBRATOR — Recent form + season context adjustment
=========================================================

Pipeline position: PASO 1 (after TTE λ_base + Kalman, before Park+Weather).

Original design: was THE quality engine before specialized engines existed.
Current role after architecture split:

  KEPT (unique signals not modeled elsewhere):
    1. Recent form     — last-10 record + active streak  (±8%)
    2. Season context  — late-season motivation / early-season noise

  REMOVED (now owned by specialized engines):
    • offense_mult (rpg/wOBA/OPS/wRC+) — double-counts TTE when active.
      TTE already encodes xwOBA+barrel%+wRC+ from Statcast.  Applied only
      as fallback when TTE was not available (no team IDs or import error).
    • defense_mult (RA/G, ERA, WHIP)   — triple-counts PASO 4+5+6.
      DefensiveEfficiencyEngine (PASO 4) owns fielding; PitcherEngine (PASO 5)
      owns the starter; BullpenEngine (PASO 6) owns the relief staff.
    • rest/fatigue — moved to ContextualEngine (PASO 7).
    • DER          — moved to DefensiveEfficiencyEngine (PASO 4).

Cap tightened: ±15% → ±8%.  With only form+season remaining the old cap
allowed more swing than both factors combined could ever produce.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple
import logging
from config import LEAGUE_AVG_RUNS, LEAGUE_AVG_OPS, LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP

if TYPE_CHECKING:
    from modules.baseball_module.calibration.learning_engine import LearningEngine

logger = logging.getLogger(__name__)

# 2024 wOBA constant (FanGraphs Guts! page)
LEAGUE_AVG_WOBA = 0.310
# League home win rate (structural; HFA engine handles the run delta)
LEAGUE_HOME_WIN_PCT = 0.520
LEAGUE_AWAY_WIN_PCT = 0.480

# Offense weights: used ONLY when TTE is unavailable (fallback path)
_DEFAULT_OFF_W = {"rpg": 0.35, "woba": 0.30, "ops": 0.20, "wrc_plus": 0.15}

_WEIGHTS_FILE = Path(__file__).parent.parent.parent.parent / "data" / "calibration_weights.json"


def _load_offense_weights() -> Dict[str, float]:
    """Load learned offense weights from JSON; fall back to defaults silently."""
    try:
        if _WEIGHTS_FILE.exists():
            data = json.loads(_WEIGHTS_FILE.read_text())
            return data.get("offense", _DEFAULT_OFF_W)
    except Exception as e:
        logger.warning("Could not load calibration_weights.json: %s — using defaults", e)
    return _DEFAULT_OFF_W


class LambdaCalibrator:
    """
    Applies two unique signals not covered by specialized engines:

      1. Recent form      — last-10 record + active winning/losing streak
      2. Season context   — late-season playoff push / cellar effect

    When TTE was unavailable (fallback λ_base), also applies:
      3. Offense quality  — rpg/wOBA/OPS/wRC+ composite

    A ±8% hard cap is applied to the combined product to prevent
    stacking beyond what form and season context can realistically produce.
    The team-bias term from LearningEngine is applied before the cap.
    """

    def __init__(self, learning_engine: "Optional[LearningEngine]" = None):
        self.name = "LambdaCalibrator"
        self.learning_engine = learning_engine
        self.league_avg_runs = LEAGUE_AVG_RUNS
        self.league_avg_ops  = LEAGUE_AVG_OPS
        self._off_w = _load_offense_weights()

    # ── Public interface ───────────────────────────────────────────────────────

    def calibrate(
        self,
        lh_base: float,
        la_base: float,
        game_data: Dict,
        tte_active: bool = True,
    ) -> Tuple[float, float]:
        """
        Returns (lh_calibrated, la_calibrated).

        Args:
            lh_base:    Home team expected runs (from TTE or legacy fallback).
            la_base:    Away team expected runs.
            game_data:  Must contain 'home_team' and 'away_team' sub-dicts.
            tte_active: True when TTE produced λ_base (offense_mult skipped).
                        False when legacy get_team_lambda was the fallback
                        (offense_mult applied as the primary quality signal).
        """
        home_team = game_data.get("home_team", {})
        away_team = game_data.get("away_team", {})

        if not home_team or not away_team:
            logger.warning("LambdaCalibrator: missing team data, returning base λ")
            return lh_base, la_base

        lh_new = self._calibrate_team(lh_base, home_team, tte_active)
        la_new = self._calibrate_team(la_base, away_team, tte_active)

        # Learned bias correction (applied only when LearningEngine is active).
        # Kalman covers 35% of model error; bias gets the remaining 65%.
        if self.learning_engine:
            season    = datetime.now().year
            home_name = home_team.get("name", "") if isinstance(home_team, dict) else str(home_team)
            away_name = away_team.get("name", "") if isinstance(away_team, dict) else str(away_team)
            home_bias = self.learning_engine.compute_team_bias_kalman_adjusted(
                home_name, season, "offense_home"
            )
            away_bias = self.learning_engine.compute_team_bias_kalman_adjusted(
                away_name, season, "offense_away"
            )
            lh_new *= home_bias
            la_new *= away_bias
            if home_bias != 1.0 or away_bias != 1.0:
                logger.debug("LambdaCalibrator bias — home %.4f  away %.4f", home_bias, away_bias)

        # ±8% hard cap — tightened from ±15% because only form+season remain.
        # The old ±15% was sized for 5 combined factors; 2 factors need less room.
        lh_new = max(lh_base * 0.92, min(lh_base * 1.08, lh_new))
        la_new = max(la_base * 0.92, min(la_base * 1.08, la_new))

        logger.debug(
            "LambdaCalibrator | tte_active=%s  λh: %.3f→%.3f  λa: %.3f→%.3f",
            tte_active, lh_base, lh_new, la_base, la_new,
        )
        return lh_new, la_new

    # ── Core calibration ───────────────────────────────────────────────────────

    def _calibrate_team(
        self,
        lambda_base: float,
        team: Dict,
        tte_active: bool,
    ) -> float:
        """
        Apply form and season-context multipliers.
        When TTE was unavailable, also applies offense quality as primary signal.
        """
        form_mult   = self._recent_form(team)
        season_mult = self._season_context(team)

        if not tte_active:
            # TTE fallback: use traditional offense composite as quality baseline.
            # defense_mult is intentionally absent even in fallback mode — the
            # pitcher/bullpen/defEff engines will apply those adjustments later.
            offense_mult = self._offense_multiplier(team)
            total = offense_mult * form_mult * season_mult
        else:
            total = form_mult * season_mult

        return float(lambda_base * total)

    # ── Factor calculators ─────────────────────────────────────────────────────

    def _offense_multiplier(self, team: Dict) -> float:
        """
        Traditional offense composite — used ONLY in TTE fallback path.
        rpg/wOBA/OPS/wRC+ relative to league average.
        """
        rpg      = team.get("runs_per_game", self.league_avg_runs)
        ops      = team.get("ops",           self.league_avg_ops)
        woba     = team.get("woba",          None)
        wrc_plus = team.get("wrc_plus",      100.0)

        w = self._off_w
        rpg_factor = rpg      / self.league_avg_runs
        ops_factor = ops      / self.league_avg_ops
        wrc_factor = wrc_plus / 100.0

        if woba is not None:
            woba_factor = woba / LEAGUE_AVG_WOBA
            return (
                rpg_factor  * w.get("rpg",      0.35) +
                woba_factor * w.get("woba",     0.30) +
                ops_factor  * w.get("ops",      0.20) +
                wrc_factor  * w.get("wrc_plus", 0.15)
            )

        # wOBA absent: redistribute its weight to rpg
        woba_w = w.get("woba", 0.30)
        return (
            rpg_factor * (w.get("rpg", 0.35) + woba_w) +
            ops_factor *  w.get("ops",         0.20) +
            wrc_factor *  w.get("wrc_plus",     0.15)
        )

    def _recent_form(self, team: Dict) -> float:
        """
        Recent performance trend from last-10 record and active streak.
        Range: [0.92, 1.08 × 1.02] — narrow to avoid over-weighting hot streaks.

        This is the primary unique contribution of AutoCalibrator vs TTE:
        TTE uses season Statcast quality; form captures the last 2–3 weeks.
        """
        last_10 = team.get("last_10", "5-5")

        try:
            wins = int(last_10.split("-")[0])
            # 0 wins → 0.92,  5 wins → 1.00,  10 wins → 1.08
            form_mult = 0.92 + (wins / 10.0) * 0.16
        except (ValueError, IndexError):
            form_mult = 1.0

        # Active streak bonus/penalty capped at ±2% to avoid excess stacking
        streak = team.get("streak", "")
        if "W" in streak:
            try:
                if int(streak.replace("W", "")) >= 5:
                    form_mult *= 1.02
            except ValueError:
                pass
        elif "L" in streak:
            try:
                if int(streak.replace("L", "")) >= 5:
                    form_mult *= 0.98
            except ValueError:
                pass

        return form_mult

    def _season_context(self, team: Dict) -> float:
        """
        Late-season motivation adjustment.

        Early season (<30 games) is intentionally NOT penalised here —
        TTE's Bayesian regression already handles small-sample uncertainty
        by blending observed xwOBA with the league prior.

        Late season (>130 games):
          Contenders (win% > .550): mild +2% (playoff intensity)
          Cellar     (win% < .420): mild -2% (possible low motivation)
        """
        wins   = team.get("wins",   81)
        losses = team.get("losses", 81)
        games  = wins + losses

        if games > 130:
            win_pct = wins / games if games > 0 else 0.500
            if win_pct > 0.550:
                return 1.02
            if win_pct < 0.420:
                return 0.98

        return 1.0

    def _home_away_split(self, team: Dict, is_home: bool) -> float:
        """
        Team-specific home/away deviation above/below structural baseline.
        Kept for reference but NOT called — HFA engine owns location advantage.
        """
        baseline = LEAGUE_HOME_WIN_PCT if is_home else LEAGUE_AWAY_WIN_PCT
        record   = team.get("home_record" if is_home else "away_record", "")
        loc_rpg  = team.get("home_runs_per_game" if is_home else "away_runs_per_game",
                             self.league_avg_runs)
        try:
            w, l    = map(int, record.split("-"))
            win_pct = w / (w + l) if (w + l) > 0 else baseline
        except (ValueError, AttributeError):
            win_pct = baseline

        return (win_pct / baseline) * 0.60 + (loc_rpg / self.league_avg_runs) * 0.40


# ── Module-level helper (public interface) ─────────────────────────────────────

def calibrate_lambdas(
    lh_base: float,
    la_base: float,
    game_data: Dict,
    learning_engine=None,
    tte_active: bool = True,
) -> Tuple[float, float]:
    """
    Convenience wrapper for run_module.py.

    Usage:
        from calibration.auto_calibrator import calibrate_lambdas
        lh, la = calibrate_lambdas(lh_base, la_base, game_data, tte_active=True)
    """
    return LambdaCalibrator(learning_engine=learning_engine).calibrate(
        lh_base, la_base, game_data, tte_active=tte_active,
    )
