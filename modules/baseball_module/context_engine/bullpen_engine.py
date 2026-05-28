"""
BULLPEN ENGINE — Relief-corps quality and workload adjustments
=============================================================

Adjusts λ_home / λ_away based on each team's relief corps, using the
same tier of advanced metrics as the Pitcher Engine for starters:

  Quality signals (parallel to starter engine):
    xwOBA against   (0.40) — removes BABIP luck from contact outcomes
    K% − BB%        (0.30) — most stable command/whiff signal for relievers
    ERA             (0.20) — reliever-specific (pitcherType=R endpoint)
    Barrel% against (0.10) — predicts HR/XBH allowed

  All metrics are Bayesian-regressed toward league averages based on
  total batters faced (TBF), so early-season samples don't dominate.

  Innings weighting:
    The adjustment is scaled by (9 − avg_ips) / 9 so the bullpen
    only moves the needle for the innings it actually pitches.

Data sources:
  MLB Stats API pitcherType=R  → ERA, WHIP, K%, BB%, TBF (reliever-only ✓)
  Baseball Savant pitcher leaderboard → xwOBA against, barrel% against
    (team-aggregate over all pitchers who appear in Savant; relievers
     dominate by volume once starters are at 25–30 starts)

Pipeline position: PASO 6 — after Pitcher Engine, before Contextual.
Convention (same as Pitcher Engine):
  Away bullpen suppresses home-team scoring → adjusts λ_home
  Home bullpen suppresses away-team scoring → adjusts λ_away
"""

from __future__ import annotations

import csv as _csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent.parent.parent
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

MLB_BASE    = "https://statsapi.mlb.com/api/v1"
SAVANT_BASE = "https://baseballsavant.mlb.com"

# ── League averages for RELIEVERS (2024/2025 combined) ────────────────────────
_LG_BP_ERA       = 4.10   # team reliever ERA (pitcherType=R)
_LG_BP_WHIP      = 1.26
_LG_BP_K_PCT     = 0.248  # relievers K% avg (higher than starters ~22%)
_LG_BP_BB_PCT    = 0.086
_LG_BP_K_BB      = _LG_BP_K_PCT - _LG_BP_BB_PCT  # 0.162
_LG_XWOBA_AG     = 0.312  # xwOBA against league avg (pitcher-side)
_LG_BARREL_PA_AG = 0.088  # barrel% against league avg

# Bayesian stabilisation constants for team-aggregate bullpen stats (TBF).
# Team bullpen accumulates TBF fast (many pitchers × high appearance rate).
# These constants reflect when TEAM aggregate (not individual) is 50% reliable.
_K_ERA_BP    = 250   # ERA / WHIP for team bullpen aggregate
_K_XWOBA_BP  = 200   # xwOBA against for team bullpen aggregate
_K_K_BB_BP   = 180   # K%-BB% for team bullpen aggregate

# Workload: "normal" bullpen throws ~3 IP per game × 3 games = 9 IP over 3 days
_NORMAL_IP_3D  = 9.0
_DEFAULT_AVG_IPS = 5.5

# Bullpen tier ERA adjustments (empirical MLB averages).
# When starter exits early, Long Relief pitchers fill innings — ERA runs above avg.
# When starter goes deep, High Leverage (setup/closer) pitches — ERA runs below avg.
_LONG_RELIEF_ERA_DELTA  = 0.80   # Long Relief ERA ≈ avg bullpen ERA + 0.80
_HIGH_LEVERAGE_ERA_DELTA = 0.70  # Closer/Setup ERA ≈ avg bullpen ERA − 0.70
_LONG_RELIEF_IPS_THRESHOLD  = 5.5  # avg_ips below this → long relief blends in
_HIGH_LEVERAGE_IPS_THRESHOLD = 6.5  # avg_ips above this → high leverage blends in


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get_json(url: str, params: dict = None, timeout: tuple = (5, 20)) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("GET %s → %s", url, e)
        return None


def _get_csv(url: str, params: dict = None, timeout: tuple = (5, 30)) -> Optional[str]:
    try:
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.text
    except Exception as e:
        log.debug("CSV %s → %s", url, e)
        return None


def _cache_valid(path: Path, ttl: int = 86400) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl


# ── Savant pitcher data (same pattern as TTE for batters) ─────────────────────

def _fetch_savant_pitcher_expected(season: int) -> Dict[int, dict]:
    """
    Returns {player_id: {est_woba, pa, bip}} for ALL pitchers in MLB.
    Checks data_enrichment SavantFetcher cache first to avoid duplicate downloads.
    """
    cache = CACHE_DIR / f"bp_savant_expected_{season}.json"
    if _cache_valid(cache):
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}

    # Reuse SavantFetcher cache if run_module already downloaded it this session.
    _enrichment_cache = ROOT / "data" / ".cache" / f"savant_expected_{season}.json"
    if _cache_valid(_enrichment_cache):
        try:
            raw = json.loads(_enrichment_cache.read_text())
            result = {
                int(k): {
                    "est_woba": float(v.get("est_woba") or _LG_XWOBA_AG),
                    "pa":       float(v.get("pa") or 0),
                    "bip":      float(v.get("bip") or 0),
                }
                for k, v in raw.items()
                if float(v.get("pa") or 0) > 0
            }
            cache.write_text(json.dumps(result))
            log.info("Savant pitcher expected %d: reused enrichment cache (%d pitchers)", season, len(result))
            return result
        except Exception:
            pass

    csv_text = _get_csv(
        f"{SAVANT_BASE}/leaderboard/expected_statistics",
        params={"type": "pitcher", "year": season, "position": "",
                "team": "", "min": "1", "csv": "true"},
    )
    if not csv_text:
        log.warning("Savant pitcher expected stats unavailable for %d", season)
        return {}

    result: dict = {}
    lines = csv_text.lstrip("﻿").splitlines()
    reader = _csv.DictReader(lines)
    for row in reader:
        try:
            pid = int(row.get("player_id", 0) or 0)
            pa  = float(row.get("pa", 0) or 0)
            if pid and pa > 0:
                result[pid] = {
                    "est_woba": float(row.get("est_woba", _LG_XWOBA_AG) or _LG_XWOBA_AG),
                    "pa":       pa,
                    "bip":      float(row.get("bip", 0) or 0),
                }
        except (ValueError, KeyError):
            continue

    cache.write_text(json.dumps(result))
    log.info("Savant pitcher expected %d: %d pitchers cached", season, len(result))
    return result


def _fetch_savant_pitcher_exitvelo(season: int) -> Dict[int, dict]:
    """
    Returns {player_id: {barrels, brl_pa, attempts}} for ALL pitchers.
    Checks data_enrichment SavantFetcher cache first to avoid duplicate downloads.
    """
    cache = CACHE_DIR / f"bp_savant_exitvelo_{season}.json"
    if _cache_valid(cache):
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}

    # Reuse SavantFetcher EV cache if already downloaded this session.
    _enrichment_cache = ROOT / "data" / ".cache" / f"savant_ev_{season}.json"
    if _cache_valid(_enrichment_cache):
        try:
            raw = json.loads(_enrichment_cache.read_text())
            result = {
                int(k): {
                    "barrels":  float(v.get("barrels")  or 0),
                    "brl_pa":   float(v.get("brl_pa")   or 0),
                    "attempts": float(v.get("attempts")  or 0),
                }
                for k, v in raw.items()
                if float(v.get("attempts") or 0) > 0
            }
            cache.write_text(json.dumps(result))
            log.info("Savant pitcher exitvelo %d: reused enrichment cache (%d pitchers)", season, len(result))
            return result
        except Exception:
            pass

    csv_text = _get_csv(
        f"{SAVANT_BASE}/leaderboard/statcast",
        params={"year": season, "type": "pitcher", "min": "1", "csv": "true"},
    )
    if not csv_text:
        log.warning("Savant pitcher exit-velo unavailable for %d", season)
        return {}

    result: dict = {}
    lines = csv_text.lstrip("﻿").splitlines()
    reader = _csv.DictReader(lines)
    for row in reader:
        try:
            pid      = int(row.get("player_id", 0) or 0)
            attempts = float(row.get("attempts", 0) or 0)
            if pid and attempts > 0:
                result[pid] = {
                    "barrels":  float(row.get("barrels",  0) or 0),
                    "brl_pa":   float(row.get("brl_pa",   0) or 0),
                    "attempts": attempts,
                }
        except (ValueError, KeyError):
            continue

    cache.write_text(json.dumps(result))
    log.info("Savant pitcher exitvelo %d: %d pitchers cached", season, len(result))
    return result


def _fetch_team_roster(team_id: int, season: int) -> Dict[int, str]:
    """Gameday roster {player_id: name}. Shared cache with TTE. Cached 24h."""
    cache = CACHE_DIR / f"tte_roster_{team_id}_{season}.json"
    if _cache_valid(cache, ttl=86400):
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}

    data = _get_json(
        f"{MLB_BASE}/teams/{team_id}/roster",
        params={"rosterType": "gameday", "season": season},
    )
    if not data:
        return {}

    roster: Dict[int, str] = {}
    for p in data.get("roster", []):
        person = p.get("person", {})
        pid    = person.get("id")
        if pid:
            roster[int(pid)] = person.get("fullName", "")

    cache.write_text(json.dumps(roster))
    return roster


# ── Bayesian regression helper ─────────────────────────────────────────────────

def _regress(observed: float, mean: float, n: float, k: float) -> float:
    """Regress toward mean. At n=0 → mean. At n=k → 50/50. At n=∞ → observed."""
    if n <= 0:
        return mean
    return (observed * n + mean * k) / (n + k)


# ── Savant aggregation for a team's pitchers ──────────────────────────────────

def _aggregate_team_savant(
    roster: Dict[int, str],
    savant_exp: Dict[int, dict],
    savant_ev:  Dict[int, dict],
) -> dict:
    """
    PA-weighted Statcast aggregation for all team pitchers found in Savant.
    Relievers dominate by volume once the team has 80+ games (starters cap
    out at ~25 appearances while the bullpen accumulates 200+ combined).
    """
    xwoba_sum    = 0.0
    total_pa     = 0.0
    barrels_sum  = 0.0
    attempts_sum = 0.0
    n_found      = 0

    for pid in roster:
        exp = savant_exp.get(pid)
        ev  = savant_ev.get(pid)
        if exp:
            pa          = float(exp["pa"])
            xwoba_sum  += float(exp["est_woba"]) * pa
            total_pa   += pa
            n_found    += 1
        if ev:
            att          = float(ev["attempts"])
            barrels_sum += float(ev["barrels"])
            attempts_sum += att

    if total_pa <= 0:
        return {"xwoba_against": _LG_XWOBA_AG, "barrel_pa_against": _LG_BARREL_PA_AG,
                "total_pa": 0.0, "total_attempts": 0.0, "n_pitchers": 0}

    return {
        "xwoba_against":     round(xwoba_sum   / total_pa,     4),
        "barrel_pa_against": round(barrels_sum  / attempts_sum, 4) if attempts_sum > 0 else _LG_BARREL_PA_AG,
        "total_pa":          total_pa,
        "total_attempts":    attempts_sum,
        "n_pitchers":        n_found,
    }


# ── Main engine ───────────────────────────────────────────────────────────────

class BullpenEngine:
    """
    Adjusts λ for the relief corps using four quality signals (xwOBA, K%-BB%,
    ERA, barrel%) with Bayesian regression and innings-proportional weighting.
    """

    def __init__(self) -> None:
        self._savant_exp: Dict[int, dict] = {}
        self._savant_ev:  Dict[int, dict] = {}
        self._season_loaded: Optional[int] = None

    def _ensure_savant_loaded(self, season: int) -> None:
        if self._season_loaded != season:
            self._savant_exp = _fetch_savant_pitcher_expected(season)
            self._savant_ev  = _fetch_savant_pitcher_exitvelo(season)
            self._season_loaded = season

    def adjust_for_bullpen(
        self,
        lh: float,
        la: float,
        game_data: Dict[str, Any],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Returns (lh_adjusted, la_adjusted, metadata).
        Away bullpen → adjusts λ_home.
        Home bullpen → adjusts λ_away.
        """
        from datetime import datetime
        season = int(game_data.get('season', datetime.now().year))
        self._ensure_savant_loaded(season)

        log.info("Bullpen Engine — adjusting lambdas")
        log.info("   Input: λ_h=%.3f  λ_a=%.3f", lh, la)

        bp_away      = game_data.get("bullpen_away")  or {}
        bp_home      = game_data.get("bullpen_home")  or {}
        starter_away = game_data.get("pitcher_away")  or {}
        starter_home = game_data.get("pitcher_home")  or {}

        home_team_id = game_data.get("home_team_id")
        away_team_id = game_data.get("away_team_id")

        away_result = self._calculate_mult(
            bp_away, starter_away, away_team_id, season, label="away"
        )
        lh_new = lh * away_result["total_mult"]

        home_result = self._calculate_mult(
            bp_home, starter_home, home_team_id, season, label="home"
        )
        la_new = la * home_result["total_mult"]

        log.info(
            "   Away BP: ERA=%.2f  xwOBA=%.3f  K-BB=%.3f  IP_3d=%.1f  "
            "bp_w=%.2f  mult=%.3f  λ_home %.3f→%.3f",
            away_result["era"], away_result["xwoba_ag"],
            away_result["k_bb"], away_result["ip_3d"],
            away_result["innings_weight"], away_result["total_mult"], lh, lh_new,
        )
        log.info(
            "   Home BP: ERA=%.2f  xwOBA=%.3f  K-BB=%.3f  IP_3d=%.1f  "
            "bp_w=%.2f  mult=%.3f  λ_away %.3f→%.3f",
            home_result["era"], home_result["xwoba_ag"],
            home_result["k_bb"], home_result["ip_3d"],
            home_result["innings_weight"], home_result["total_mult"], la, la_new,
        )

        return lh_new, la_new, {
            "bullpen_away": away_result,
            "bullpen_home": home_result,
        }

    # ── Per-team calculation ───────────────────────────────────────────────────

    def _calculate_mult(
        self,
        bullpen:  Dict[str, Any],
        starter:  Dict[str, Any],
        team_id:  Optional[int],
        season:   int,
        label:    str = "",
    ) -> dict:
        # ── Starter depth (computed first — needed for tier ERA adjustment) ──
        avg_ips     = float(starter.get("avg_innings_per_start") or _DEFAULT_AVG_IPS)
        avg_ips     = max(4.0, min(8.0, avg_ips))
        innings_weight = (9.0 - avg_ips) / 9.0

        # ── MLB API metrics (reliever-specific) ───────────────────────────
        _era  = bullpen.get("era");          era  = float(_era  if _era  is not None else _LG_BP_ERA)
        _whip = bullpen.get("bullpen_whip"); whip = float(_whip if _whip is not None else _LG_BP_WHIP)
        _tbf  = bullpen.get("tbf");          tbf  = float(_tbf  if _tbf  is not None else 0)
        k_pct  = bullpen.get("k_pct")   # may be None if not yet fetched
        bb_pct = bullpen.get("bb_pct")

        # ── Tier ERA adjustment: who actually pitches depends on starter depth ──
        # Short starter → Long Relief (worse ERA); deep starter → High Leverage (better).
        # Blend is linear: full effect at threshold extremes, neutral in the middle.
        if avg_ips < _LONG_RELIEF_IPS_THRESHOLD:
            tier_blend = min(1.0, (_LONG_RELIEF_IPS_THRESHOLD - avg_ips) / 1.5)
            tier_era   = era + _LONG_RELIEF_ERA_DELTA * tier_blend
            tier_label = f"long_relief(blend={tier_blend:.2f})"
        elif avg_ips > _HIGH_LEVERAGE_IPS_THRESHOLD:
            tier_blend = min(1.0, (avg_ips - _HIGH_LEVERAGE_IPS_THRESHOLD) / 1.0)
            tier_era   = era - _HIGH_LEVERAGE_ERA_DELTA * tier_blend
            tier_label = f"high_leverage(blend={tier_blend:.2f})"
        else:
            tier_era   = era
            tier_label = "average"

        # Bayesian regression on tier-adjusted ERA/WHIP based on TBF
        era_reg  = _regress(tier_era, _LG_BP_ERA,  tbf, _K_ERA_BP)
        whip_reg = _regress(whip,     _LG_BP_WHIP, tbf, _K_ERA_BP)
        era_factor  = era_reg  / _LG_BP_ERA
        whip_factor = whip_reg / _LG_BP_WHIP

        # K%-BB% differential (regressed) — most stable relief quality signal
        if k_pct is not None and bb_pct is not None:
            k_bb_obs = float(k_pct) - float(bb_pct)
            k_bb_reg = _regress(k_bb_obs, _LG_BP_K_BB, tbf, _K_K_BB_BP)
            delta_k_bb = k_bb_reg - _LG_BP_K_BB
            # Positive = better command → fewer runs → factor < 1
            k_bb_factor = max(0.85, min(1.15, 1.0 - delta_k_bb * 1.5))
        else:
            k_bb_factor = 1.0
            k_bb_reg    = _LG_BP_K_BB

        # ── Statcast metrics (team aggregate, PA-weighted) ─────────────────
        xwoba_ag   = _LG_XWOBA_AG
        barrel_ag  = _LG_BARREL_PA_AG
        savant_pa  = 0.0
        savant_att = 0.0
        n_pitchers = 0

        if team_id:
            roster = _fetch_team_roster(int(team_id), season)
            if roster:
                sv = _aggregate_team_savant(roster, self._savant_exp, self._savant_ev)
                xwoba_ag      = sv["xwoba_against"]
                barrel_ag     = sv["barrel_pa_against"]
                savant_pa     = sv["total_pa"]
                savant_att    = sv["total_attempts"]
                n_pitchers    = sv["n_pitchers"]

        xwoba_reg  = _regress(xwoba_ag,  _LG_XWOBA_AG,     savant_pa,  _K_XWOBA_BP)
        barrel_reg = _regress(barrel_ag, _LG_BARREL_PA_AG, savant_att, _K_XWOBA_BP)

        # Higher xwOBA against → more runs → factor > 1
        xwoba_factor  = xwoba_reg  / _LG_XWOBA_AG
        barrel_factor = barrel_reg / _LG_BARREL_PA_AG

        # ── Composite quality: four signals, ERA replaces raw league proxy ─
        # Weights: xwOBA 0.40, K-BB 0.30, ERA 0.20, Barrel 0.10
        quality_raw = (
            xwoba_factor  * 0.40 +
            k_bb_factor   * 0.30 +
            era_factor    * 0.20 +
            barrel_factor * 0.10
        )
        quality_mult = max(0.75, min(1.30, quality_raw))

        # ── Workload fatigue ───────────────────────────────────────────────
        _ip3  = bullpen.get("ip_last_3_days"); ip_3d = float(_ip3 if _ip3 is not None else _NORMAL_IP_3D)
        workload_mult = self._workload_mult(ip_3d)

        raw_mult = quality_mult * workload_mult

        # ── Innings weighting (avg_ips already computed above for tier) ───

        total_mult = 1.0 + innings_weight * (raw_mult - 1.0)
        total_mult = max(0.85, min(1.15, total_mult))  # B1: expanded from [0.90,1.10]

        log.debug(
            "   [%s bp] tier=%s ERA %.2f→%.2f(tier)→%.2f(reg) xwOBA=%.3f kbb=%.3f brl=%.3f "
            "→ quality=%.3f  workload=%.3f  raw=%.3f  bp_w=%.2f  total=%.3f",
            label, tier_label, era, tier_era, era_reg,
            xwoba_reg, k_bb_reg, barrel_reg,
            quality_mult, workload_mult, raw_mult, innings_weight, total_mult,
        )

        return {
            "era":            round(era,            2),
            "tier_era":       round(tier_era,       2),
            "tier_label":     tier_label,
            "era_reg":        round(era_reg,        3),
            # Composite quality expressed as an ERA rate: quality_mult × LG_BP_ERA.
            # Used by _compute_f5_lambda to estimate bullpen runs in F5 innings.
            "effective_era":  round(quality_mult * _LG_BP_ERA, 3),
            "xwoba_ag":       round(xwoba_ag,       3),
            "xwoba_reg":      round(xwoba_reg,      3),
            "k_bb":           round(k_bb_reg,       3),
            "barrel_ag":      round(barrel_ag,      3),
            "barrel_reg":     round(barrel_reg,     3),
            "ip_3d":          round(ip_3d,          1),
            "quality_mult":   round(quality_mult,   4),
            "workload_mult":  round(workload_mult,  4),
            "raw_mult":       round(raw_mult,        4),
            "avg_ips":        round(avg_ips,         2),
            "innings_weight": round(innings_weight,  3),
            "total_mult":     round(total_mult,      4),
            "n_pitchers":     n_pitchers,
        }

    @staticmethod
    def _workload_mult(ip_3d: float) -> float:
        """
        Smooth fatigue gradient from innings thrown in the last 3 days.
        Normal: ~9 IP over 3 games. Each IP above → +1.2% penalty (cap +10%).
        Each IP below → up to −3% bonus (rested corps).
        """
        delta = ip_3d - _NORMAL_IP_3D
        if delta > 0:
            return float(min(1.0 + delta * 0.012, 1.10))
        return float(max(1.0 + delta * 0.005, 0.97))


# ── Module-level convenience function ─────────────────────────────────────────

_engine: Optional[BullpenEngine] = None


def adjust_for_bullpen(
    lh: float,
    la: float,
    game_data: Dict[str, Any],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Convenience wrapper for run_module.py:
        from context_engine.bullpen_engine import adjust_for_bullpen
        lh, la, meta = adjust_for_bullpen(lh, la, game_data)
    Re-uses a module-level engine instance to share cached Savant data.
    """
    global _engine
    if _engine is None:
        _engine = BullpenEngine()
    return _engine.adjust_for_bullpen(lh, la, game_data)
