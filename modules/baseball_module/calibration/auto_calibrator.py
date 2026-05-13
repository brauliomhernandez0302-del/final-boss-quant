"""
AUTO CALIBRATOR — Lambda calibration from team-stat signals
============================================================

Adjusts the base Poisson λ values for each team using six independent
factors derived from team statistics. Each factor is an independent
adjustment, so they are combined multiplicatively (not as a weighted
sum, which compresses their joint effect).

Pipeline position: runs BEFORE HFA engine and pitcher adjustments.

Fixes applied vs prior version:
  - Weighted sum → multiplicative product (factors are independent)
  - wOBA league avg 0.320 → 0.310 (FanGraphs 2024 constant)
  - `if woba:` → `if woba is not None:` (silent zero-wOBA fallback)
  - wOBA weight redistributed to rpg/ops when woba absent
  - Travel removed from _calculate_rest_travel (HFA engine owns it;
    run_module copies miles_traveled_away → team dict, creating
    double-counting across both engines)
  - DER league avg 0.700 → 0.715 (2024 empirical MLB average)
  - Home/away split normalisation 0.50 → 0.52/0.48 to isolate
    team-specific tendencies from structural home advantage
"""

from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional, Tuple
import logging
from config import LEAGUE_AVG_RUNS, LEAGUE_AVG_OPS, LEAGUE_AVG_ERA, LEAGUE_AVG_WHIP

if TYPE_CHECKING:
    from modules.baseball_module.calibration.learning_engine import LearningEngine

logger = logging.getLogger(__name__)

# 2024 wOBA constant (FanGraphs Guts! page — changes slightly each season)
LEAGUE_AVG_WOBA = 0.310
# 2024 MLB defensive efficiency (share of BIP converted to outs)
LEAGUE_AVG_DER  = 0.715
# League home win rate (structural; HFA engine handles the run delta)
LEAGUE_HOME_WIN_PCT = 0.520
LEAGUE_AWAY_WIN_PCT = 0.480


class LambdaCalibrator:
    """
    Adjusts λ_home and λ_away using five team-quality signals:
      1. Offense quality        (30 % weight)
      2. Opponent defence       (25 %)
      3. Recent form            (25 %)
      4. Rest days              (10 %)
      5. Season context         (10 %)

    Each factor targets a different dimension; they are multiplied
    together so their effects compound correctly.  A single ±15% hard
    cap is applied in calibrate() on the COMBINED product of all stat
    factors plus team bias — preventing the old two-level stacking
    (±20% factors × ±30% bias = ±56%) that was inverting the spread.
    """

    def __init__(self, learning_engine: "Optional[LearningEngine]" = None):
        self.name = "LambdaCalibrator"
        self.learning_engine = learning_engine
        self.league_avg_runs = LEAGUE_AVG_RUNS
        self.league_avg_ops  = LEAGUE_AVG_OPS
        self.league_avg_era  = LEAGUE_AVG_ERA
        self.league_avg_whip = LEAGUE_AVG_WHIP

    # ── Public interface ───────────────────────────────────────────────────────

    def calibrate(
        self,
        lh_base: float,
        la_base: float,
        game_data: Dict,
    ) -> Tuple[float, float]:
        """
        Returns (lh_calibrated, la_calibrated).

        Args:
            lh_base: Home team expected runs (from team RPG).
            la_base: Away team expected runs (from team RPG).
            game_data: Must contain 'home_team' and 'away_team' sub-dicts.
        """
        home_team = game_data.get("home_team", {})
        away_team = game_data.get("away_team", {})

        if not home_team or not away_team:
            logger.warning("LambdaCalibrator: missing team data, returning base λ")
            return lh_base, la_base

        lh_new = self._calibrate_team(lh_base, home_team, away_team, is_home=True)
        la_new = self._calibrate_team(la_base, away_team, home_team, is_home=False)

        # Learned bias correction (applied only when LearningEngine is active).
        # Uses Kalman-adjusted bias to avoid double-counting the same error signal:
        # Kalman covers blend=35% of model error; bias gets the remaining 65%.
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
            lh_new   *= home_bias
            la_new   *= away_bias
            if home_bias != 1.0 or away_bias != 1.0:
                logger.debug("LambdaCalibrator bias — home %.4f  away %.4f", home_bias, away_bias)

        # Single combined ±15% cap applied AFTER bias.
        # Replaces the old two-level arrangement (±20% stat cap then ±30% bias cap)
        # which allowed up to ±56% total swing and was overriding the ~3-5% HFA
        # signal, inverting the λ spread (λ_away > λ_home across the full dataset).
        lh_new = max(lh_base * 0.85, min(lh_base * 1.15, lh_new))
        la_new = max(la_base * 0.85, min(la_base * 1.15, la_new))

        logger.debug(
            "LambdaCalibrator | λh: %.3f→%.3f  λa: %.3f→%.3f",
            lh_base, lh_new, la_base, la_new,
        )
        return lh_new, la_new

    # ── Core calibration ───────────────────────────────────────────────────────

    def _calibrate_team(
        self,
        lambda_base: float,
        team: Dict,
        opponent: Dict,
        is_home: bool,
    ) -> float:
        """
        Apply five independent multipliers to lambda_base and return the raw
        product. The caller (calibrate()) applies the combined ±15% hard cap
        after adding the team-bias term, so no cap is applied here.

        _home_away_split is intentionally excluded: home/away advantage is
        handled by the HFA engine. Including it here double-counts the effect
        and, because default records ("40-41") are asymmetrically neutral vs
        the 0.52/0.48 baselines, it created a systematic −0.19 run/game
        away bias that inverted the spread.
        """
        offense_mult = self._offense_multiplier(team)
        defense_mult = self._defense_multiplier(opponent)
        form_mult    = self._recent_form(team)
        rest_mult    = self._rest_days(team)
        season_mult  = self._season_context(team)

        total = offense_mult * defense_mult * form_mult * rest_mult * season_mult
        return float(lambda_base * total)

    # ── Factor calculators ─────────────────────────────────────────────────────

    def _offense_multiplier(self, team: Dict) -> float:
        """
        Offensive quality relative to league average.
        Combines four correlated metrics as a weighted average (correct
        for correlated signals measuring the same underlying quantity).
        """
        rpg     = team.get("runs_per_game", self.league_avg_runs)
        ops     = team.get("ops",           self.league_avg_ops)
        woba    = team.get("woba",          None)
        wrc_plus = team.get("wrc_plus",     100.0)

        rpg_factor = rpg / self.league_avg_runs
        ops_factor = ops / self.league_avg_ops
        wrc_factor = wrc_plus / 100.0

        if woba is not None:
            # woba=0.0 (missing data sentinel) treated correctly as missing
            woba_factor = woba / LEAGUE_AVG_WOBA  # 2024 FanGraphs constant = 0.310
            offense_mult = (
                rpg_factor  * 0.35 +
                woba_factor * 0.30 +
                ops_factor  * 0.20 +
                wrc_factor  * 0.15
            )
        else:
            # Redistribute wOBA's 0.30 weight: rpg is most reliable predictor
            offense_mult = (
                rpg_factor  * 0.50 +
                ops_factor  * 0.35 +
                wrc_factor  * 0.15
            )

        return offense_mult

    def _defense_multiplier(self, opponent: Dict) -> float:
        """
        Opponent pitching / defence quality.
        Higher value → easier to score (weaker opponent).
        All four metrics are normalised so 1.0 = league average.
        """
        runs_allowed = opponent.get("runs_allowed_per_game", self.league_avg_runs)
        team_era     = opponent.get("team_era",              self.league_avg_era)
        team_whip    = opponent.get("team_whip",             self.league_avg_whip)
        der          = opponent.get("der",                   LEAGUE_AVG_DER)

        defense_factor = runs_allowed / self.league_avg_runs   # higher RA → easier
        era_factor     = team_era     / self.league_avg_era    # higher ERA → easier
        whip_factor    = team_whip    / self.league_avg_whip   # higher WHIP → easier
        # Lower DER (fewer BIP converted to outs) → easier to score → factor > 1
        der_factor     = LEAGUE_AVG_DER / der                  # 0.715 / der

        defense_mult = (
            defense_factor * 0.40 +
            era_factor     * 0.30 +
            whip_factor    * 0.20 +
            der_factor     * 0.10
        )

        return defense_mult

    def _recent_form(self, team: Dict) -> float:
        """
        Recent performance trend from last-10 record and active streak.
        Range: [0.90, 1.10] — deliberately narrow to avoid over-weighting
        hot/cold streaks vs underlying team quality.
        """
        last_10 = team.get("last_10", "5-5")

        try:
            wins     = int(last_10.split("-")[0])
            # 0-10 wins → 0.92–1.08, centred at 5 wins = 1.00
            form_mult = 0.92 + (wins / 10.0) * 0.16
        except (ValueError, IndexError):
            form_mult = 1.0

        # Active streak bonus/penalty capped at ±2 % to avoid stacking
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

    def _home_away_split(self, team: Dict, is_home: bool) -> float:
        """
        Team-specific home / away performance deviation.

        Normalised against the LEAGUE-AVERAGE home win rate (0.52) and
        away win rate (0.48) rather than 0.50.  This isolates each team's
        tendencies above / below what is structurally expected at home or
        away, since the overall home-field advantage is already handled by
        the HFA engine.
        """
        if is_home:
            record  = team.get("home_record",        "42-39")  # ≈ 0.519 ≈ home baseline
            loc_rpg = team.get("home_runs_per_game", self.league_avg_runs)
            baseline_win_pct = LEAGUE_HOME_WIN_PCT  # 0.52

        else:
            record  = team.get("away_record",        "39-42")  # ≈ 0.481 ≈ away baseline
            loc_rpg = team.get("away_runs_per_game", self.league_avg_runs)
            baseline_win_pct = LEAGUE_AWAY_WIN_PCT  # 0.48

        try:
            w, l    = map(int, record.split("-"))
            win_pct = w / (w + l) if (w + l) > 0 else baseline_win_pct
        except (ValueError, AttributeError):
            win_pct = baseline_win_pct

        split_mult = (
            (win_pct / baseline_win_pct) * 0.60 +
            (loc_rpg / self.league_avg_runs) * 0.40
        )

        return split_mult

    def _rest_days(self, team: Dict) -> float:
        """
        Fatigue and rust adjustment based on days of rest.
        Travel is intentionally excluded here — the HFA engine applies
        travel fatigue to λ_away, and run_module copies the away travel
        fields into the team dict, which would cause double-counting.
        """
        rest_days = team.get("rest_days", 1)

        if rest_days == 0:
            return 0.96   # Back-to-back penalty
        if rest_days >= 3:
            return 0.98   # Rust penalty (extended rest, all-star break, etc.)
        return 1.0         # 1–2 days: optimal rest

    def _season_context(self, team: Dict) -> float:
        """
        Minor contextual adjustment for season phase.
        Early season: lower confidence in small-sample stats (symmetric,
        cancels out in spread but reduces noise vs league average).
        Late season: mild incentive effects for contenders / tail-enders.
        """
        wins   = team.get("wins",   81)
        losses = team.get("losses", 81)
        games  = wins + losses

        if games < 30:
            return 0.98   # Low sample: regress toward league avg

        if games > 130:
            win_pct = wins / games
            if win_pct > 0.550:
                return 1.02   # Playoff-push motivation
            if win_pct < 0.420:
                return 0.98   # Deep cellar: possible low motivation
        return 1.0


# ── Module-level helper (public interface) ─────────────────────────────────────

def calibrate_lambdas(
    lh_base: float,
    la_base: float,
    game_data: Dict,
    learning_engine=None,
) -> Tuple[float, float]:
    """
    Convenience wrapper — matches the import used in run_module.py.

    Usage:
        from calibration.auto_calibrator import calibrate_lambdas
        lh, la = calibrate_lambdas(lh_base, la_base, game_data)
    """
    return LambdaCalibrator(learning_engine=learning_engine).calibrate(
        lh_base, la_base, game_data
    )
