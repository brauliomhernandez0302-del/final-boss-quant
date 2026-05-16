"""
true_talent_engine.py — True Talent Offense Engine
====================================================

Estimates how many runs a team's lineup would score against average
pitching in a **park-neutral** context.  The Park Engine and HFA Engine
apply game-specific location adjustments on top of this baseline.

Data sources
------------
  Baseball Savant (Statcast)
      Player-level xwOBA (expected wOBA from exit velocity / launch angle)
      Player-level barrel% and brl_pa
      → joined to team via MLB Stats API roster endpoint

  MLB Stats API
      Team season stats: wOBA (computed from components), OBP, SLG,
      BB, K, PA, HBP — used for BB%, K%, and wRC+ approximation

Method
------
  1.  For each team, pull season stats from both sources.
  2.  Aggregate player-level Statcast data to team level (PA-weighted).
  3.  Apply Bayesian regression to mean for each metric based on sample
      size (PA).  This handles early-season noise correctly without any
      hardcoded season_context fudge factor.
  4.  Compute composite True Talent score from four orthogonal signals:
        xwOBA factor   (0.45) — removes BABIP luck from contact outcomes
        wRC+ factor    (0.30) — park-adjusted comprehensive run creation
        Contact factor (0.15) — barrel% predicts future power output
        Plate disc     (0.10) — BB%-K% differential, most stable metric
  5.  λ_talent = composite × league_avg_rpg

Regression constants (half-reliability PA from FanGraphs / BIS research):
    xwOBA  → k=150 PA   BB%  → k=120 PA
    wRC+   → k=400 PA   K%   → k=60  PA
    barrel → k=120 BIP
"""

from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent.parent.parent
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

MLB_BASE   = "https://statsapi.mlb.com/api/v1"
SAVANT_BASE = "https://baseballsavant.mlb.com"

# ── League averages (2024/2025 combined — updated annually) ──────────────────
LG_RPG       = 4.50
LG_XWOBA     = 0.312
LG_WOBA      = 0.310
LG_WOBA_SCALE = 1.157   # FanGraphs wOBAscale 2024
LG_BB_PCT    = 0.086
LG_K_PCT     = 0.224
LG_BARREL_PA = 0.088    # league barrel per PA

# Bayesian regression constants (PA at which metric is 50% reliable)
_K_XWOBA  = 150
_K_WRC    = 400
_K_BARREL = 120
_K_BB     = 120
_K_K      = 60

# Bayesian prior-season equivalent PA (crossover point where prior = current = 50%)
# At PA=0 → 100% prior. At PA=1000 → 50/50. At PA=∞ → 0% prior.
_PRIOR_PA_EQUIVALENT = 1000


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _get_json(url: str, params: dict = None, timeout: int = 12) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("GET %s → %s", url, e)
        return None


def _get_csv(url: str, params: dict = None, timeout: int = 15) -> Optional[str]:
    try:
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.text
    except Exception as e:
        log.debug("CSV %s → %s", url, e)
        return None


def _cache_path(name: str) -> Path:
    return CACHE_DIR / name


def _cache_valid(path: Path, ttl_seconds: int = 86400) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds


# ── Bayesian regression to mean ───────────────────────────────────────────────


def _regress(observed: float, mean: float, n: float, k: float) -> float:
    """
    Bayesian shrinkage toward the population mean.
    At n=0  → returns mean (pure prior).
    At n=k  → returns (observed + mean) / 2  (50% signal).
    At n=∞  → returns observed (full trust).
    """
    if n <= 0:
        return mean
    return (observed * n + mean * k) / (n + k)


# ── Baseball Savant data fetchers ─────────────────────────────────────────────


def _fetch_savant_expected(season: int) -> dict:
    """
    Returns {player_id: {xwoba, woba, pa, bip}} for the given season.
    Cached for 6 hours.
    """
    cache = _cache_path(f"tte_savant_expected_{season}.json")
    if _cache_valid(cache):
        return json.loads(cache.read_text())

    csv_text = _get_csv(
        f"{SAVANT_BASE}/leaderboard/expected_statistics",
        params={"type": "batter", "year": season, "position": "",
                "team": "", "min": "1", "csv": "true"},
    )
    if not csv_text:
        log.warning("Savant expected stats unavailable for %d", season)
        return {}

    import csv as csv_mod
    result: dict = {}
    lines = csv_text.lstrip("﻿").splitlines()
    reader = csv_mod.DictReader(lines)
    for row in reader:
        try:
            pid = int(row.get("player_id", 0))
            pa  = float(row.get("pa", 0) or 0)
            if pid and pa > 0:
                result[pid] = {
                    "xwoba": float(row.get("est_woba", LG_XWOBA) or LG_XWOBA),
                    "woba":  float(row.get("woba",     LG_WOBA)   or LG_WOBA),
                    "pa":    pa,
                    "bip":   float(row.get("bip", 0) or 0),
                }
        except (ValueError, KeyError):
            continue

    cache.write_text(json.dumps(result))
    log.info("Savant expected stats %d: %d players cached", season, len(result))
    return result


def _fetch_savant_exitvelo(season: int) -> dict:
    """
    Returns {player_id: {barrels, brl_pa, attempts}} for the given season.
    Cached for 6 hours.
    """
    cache = _cache_path(f"tte_savant_exitvelo_{season}.json")
    if _cache_valid(cache):
        return json.loads(cache.read_text())

    csv_text = _get_csv(
        f"{SAVANT_BASE}/leaderboard/statcast",
        params={"year": season, "type": "batter", "min": "1", "csv": "true"},
    )
    if not csv_text:
        log.warning("Savant exit velo unavailable for %d", season)
        return {}

    import csv as csv_mod
    result: dict = {}
    lines = csv_text.lstrip("﻿").splitlines()
    reader = csv_mod.DictReader(lines)
    for row in reader:
        try:
            pid      = int(row.get("player_id", 0))
            attempts = float(row.get("attempts", 0) or 0)
            if pid and attempts > 0:
                result[pid] = {
                    "barrels":  float(row.get("barrels",    0) or 0),
                    "brl_pa":   float(row.get("brl_pa",     0) or 0),
                    "attempts": attempts,
                    "avg_ev":   float(row.get("avg_hit_speed", 86) or 86),
                }
        except (ValueError, KeyError):
            continue

    cache.write_text(json.dumps(result))
    log.info("Savant exit velo %d: %d players cached", season, len(result))
    return result


# ── MLB Stats API fetchers ────────────────────────────────────────────────────


def _fetch_team_roster(team_id: int, season: int) -> Dict[int, str]:
    """
    Returns {mlbam_player_id: player_full_name} for the team's 40-man roster.
    Cached per team/season for 24 hours.
    """
    cache = _cache_path(f"tte_roster_{team_id}_{season}.json")
    if _cache_valid(cache, ttl_seconds=86400):
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}

    data = _get_json(
        f"{MLB_BASE}/teams/{team_id}/roster",
        params={"rosterType": "40Man", "season": season},
    )
    if not data:
        return {}

    roster: Dict[int, str] = {}
    for p in data.get("roster", []):
        person = p.get("person", {})
        pid  = person.get("id")
        name = person.get("fullName", "")
        if pid:
            roster[int(pid)] = name

    cache.write_text(json.dumps(roster))
    log.debug("Roster %d/%d: %d players", team_id, season, len(roster))
    return roster


def _fetch_team_hitting_stats(team_id: int, season: int) -> dict:
    """
    Returns {pa, runs, bb, k, hbp, singles, doubles, triples, hr, ab, sf,
             ops, obp, slg, woba, games}
    from MLB Stats API season hitting endpoint. Cached 6h.
    """
    cache = _cache_path(f"tte_hitting_{team_id}_{season}.json")
    if _cache_valid(cache):
        return json.loads(cache.read_text())

    data = _get_json(
        f"{MLB_BASE}/teams/{team_id}/stats",
        params={"stats": "season", "season": season, "group": "hitting", "sportId": 1},
    )
    if not data:
        return {}

    splits = data.get("stats", [{}])[0].get("splits", [{}])
    if not splits:
        return {}
    stat = splits[0].get("stat", {})

    games = int(stat.get("gamesPlayed") or 0)
    if games < 5:
        return {}

    bb   = int(stat.get("baseOnBalls") or 0)
    ibb  = int(stat.get("intentionalWalks") or 0)
    hbp  = int(stat.get("hitByPitch") or 0)
    hits = int(stat.get("hits") or 0)
    dbl  = int(stat.get("doubles") or 0)
    trp  = int(stat.get("triples") or 0)
    hr   = int(stat.get("homeRuns") or 0)
    ab   = int(stat.get("atBats") or 0)
    sf   = int(stat.get("sacrificeFlies") or 0)
    so   = int(stat.get("strikeOuts") or 0)
    runs = int(stat.get("runs") or 0)

    ubb     = bb - ibb
    singles = hits - dbl - trp - hr
    pa      = ab + ubb + hbp + sf + ibb
    denom   = ab + ubb + hbp + sf

    woba = (
        round((0.690*ubb + 0.722*hbp + 0.888*singles +
               1.271*dbl + 1.616*trp + 2.101*hr) / denom, 4)
        if denom > 0 else LG_WOBA
    )
    obp = float(stat.get("obp") or 0)
    slg = float(stat.get("slg") or 0)
    ops = round(obp + slg, 4) if obp and slg else 0.735

    result = {
        "pa":     pa,
        "runs":   runs,
        "games":  games,
        "rpg":    round(runs / games, 4) if games else LG_RPG,
        "bb":     ubb,
        "k":      so,
        "bb_pct": round(ubb / pa, 4)  if pa > 0 else LG_BB_PCT,
        "k_pct":  round(so  / pa, 4)  if pa > 0 else LG_K_PCT,
        "woba":   woba,
        "ops":    ops,
        "obp":    round(obp, 4),
        "slg":    round(slg, 4),
    }
    cache.write_text(json.dumps(result))
    time.sleep(0.1)
    return result


# ── Team-level Statcast aggregation ──────────────────────────────────────────


def _aggregate_statcast_for_team(
    roster: Dict[int, str],
    savant_exp: dict,
    savant_ev: dict,
) -> dict:
    """
    PA-weighted aggregation of player-level Statcast data for a roster.
    Returns {xwoba, barrel_pa, avg_ev, total_pa, n_players}.
    Falls back to league averages for missing players.
    """
    total_pa      = 0.0
    xwoba_sum     = 0.0
    barrels_sum   = 0.0
    attempts_sum  = 0.0
    ev_sum        = 0.0
    n_found       = 0

    for pid in roster:
        exp = savant_exp.get(pid) or savant_exp.get(str(pid))
        ev  = savant_ev.get(pid)  or savant_ev.get(str(pid))
        if exp:
            pa          = float(exp["pa"])
            xwoba_sum  += float(exp["xwoba"]) * pa
            total_pa   += pa
            n_found    += 1
        if ev:
            attempts    = float(ev["attempts"])
            barrels_sum += float(ev["barrels"])
            attempts_sum += attempts
            ev_sum      += float(ev["avg_ev"]) * attempts

    if total_pa <= 0:
        return {
            "xwoba":      LG_XWOBA,
            "barrel_pa":  LG_BARREL_PA,
            "avg_ev":     87.0,
            "total_pa":   0.0,
            "n_players":  0,
        }

    return {
        "xwoba":     round(xwoba_sum  / total_pa,     4),
        "barrel_pa": round(barrels_sum / total_pa,     4) if total_pa > 0 else LG_BARREL_PA,
        "avg_ev":    round(ev_sum / attempts_sum,      2) if attempts_sum > 0 else 87.0,
        "total_pa":  total_pa,
        "n_players": n_found,
    }


# ── Composite True Talent computation ────────────────────────────────────────


def _composite_score(
    xwoba_reg:  float,
    wrc_reg:    float,
    barrel_reg: float,
    plate_reg:  float,
) -> float:
    """
    Weighted composite of four regressed, normalised factors.
    Each factor is centred at 1.0 = league average.
    """
    return (
        xwoba_reg  * 0.45 +
        wrc_reg    * 0.30 +
        barrel_reg * 0.15 +
        plate_reg  * 0.10
    )


# ── Public interface ──────────────────────────────────────────────────────────


class TrueTalentOffenseEngine:
    """
    Computes park-neutral expected runs per game for a team using
    Statcast xwOBA, barrel%, plate discipline, and wRC+ derived from
    MLB Stats API.  Applies Bayesian regression to mean based on PA.
    """

    def __init__(self) -> None:
        self._savant_exp:     Dict[int, dict] = {}
        self._savant_ev:      Dict[int, dict] = {}
        self._savant_exp_pri: Dict[int, dict] = {}  # prior season — shared across calls
        self._savant_ev_pri:  Dict[int, dict] = {}
        self._season_loaded:  Optional[int] = None
        self._prior_loaded:   Optional[int] = None

    def _ensure_savant_loaded(self, season: int) -> None:
        prior_season = season - 1
        if self._season_loaded != season:
            self._savant_exp = {
                int(k): v for k, v in _fetch_savant_expected(season).items()
            }
            self._savant_ev = {
                int(k): v for k, v in _fetch_savant_exitvelo(season).items()
            }
            self._season_loaded = season
            log.info(
                "Savant season %d loaded: %d expected, %d exitvelo players",
                season, len(self._savant_exp), len(self._savant_ev),
            )
        if self._prior_loaded != prior_season:
            self._savant_exp_pri = {
                int(k): v for k, v in _fetch_savant_expected(prior_season).items()
            }
            self._savant_ev_pri = {
                int(k): v for k, v in _fetch_savant_exitvelo(prior_season).items()
            }
            self._prior_loaded = prior_season
            log.info(
                "Savant prior season %d loaded: %d expected, %d exitvelo players",
                prior_season, len(self._savant_exp_pri), len(self._savant_ev_pri),
            )

    def get_lambda(
        self,
        team_id:   int,
        team_name: str,
        season:    Optional[int] = None,
    ) -> Tuple[float, dict]:
        """
        Returns (λ_talent, metadata_dict).

        λ_talent — park-neutral expected runs per game.
        metadata  — all intermediate values for logging / debugging.
        """
        if season is None:
            season = datetime.now().year
        prior_season = season - 1

        self._ensure_savant_loaded(season)

        # ── Current season data ────────────────────────────────────────────
        roster_cur  = _fetch_team_roster(team_id, season)
        hitting_cur = _fetch_team_hitting_stats(team_id, season)
        sc_cur      = _aggregate_statcast_for_team(
            roster_cur, self._savant_exp, self._savant_ev
        )

        # ── Prior season data (for blending) ──────────────────────────────
        hitting_pri = _fetch_team_hitting_stats(team_id, prior_season)
        roster_pri  = _fetch_team_roster(team_id, prior_season)
        # Prior-season Savant is cached in the instance by _ensure_savant_loaded.
        sc_pri      = _aggregate_statcast_for_team(
            roster_pri, self._savant_exp_pri, self._savant_ev_pri
        )

        # ── Extract raw metrics ────────────────────────────────────────────
        pa_cur     = hitting_cur.get("pa",     0.0)
        xwoba_cur  = sc_cur.get("xwoba",       LG_XWOBA)
        barrel_cur = sc_cur.get("barrel_pa",   LG_BARREL_PA)
        bb_cur     = hitting_cur.get("bb_pct", LG_BB_PCT)
        k_cur      = hitting_cur.get("k_pct",  LG_K_PCT)
        woba_cur   = hitting_cur.get("woba",   LG_WOBA)

        xwoba_pri  = sc_pri.get("xwoba",       LG_XWOBA)
        barrel_pri = sc_pri.get("barrel_pa",   LG_BARREL_PA)
        bb_pri     = hitting_pri.get("bb_pct", LG_BB_PCT)
        k_pri      = hitting_pri.get("k_pct",  LG_K_PCT)
        woba_pri   = hitting_pri.get("woba",   LG_WOBA)
        rpg_pri    = hitting_pri.get("rpg",    LG_RPG)

        # ── Bayesian regression to mean (current season) ──────────────────
        xwoba_reg  = _regress(xwoba_cur,  LG_XWOBA,     pa_cur, _K_XWOBA)
        barrel_reg = _regress(barrel_cur, LG_BARREL_PA, pa_cur, _K_BARREL)
        bb_reg     = _regress(bb_cur,     LG_BB_PCT,    pa_cur, _K_BB)
        k_reg      = _regress(k_cur,      LG_K_PCT,     pa_cur, _K_K)
        woba_reg   = _regress(woba_cur,   LG_WOBA,      pa_cur, _K_WRC)

        # wRC+ approximation from regressed wOBA (no park adjustment needed
        # because we output park-neutral λ and the Park Engine handles venue)
        wrc_reg_val = round(((woba_reg - LG_WOBA) / LG_WOBA_SCALE) * 100 + 100, 2)
        wrc_reg_val = max(50.0, min(165.0, wrc_reg_val))

        # Plate discipline: (BB% - K%) differential vs league
        # Positive = more walks, fewer Ks (good); normalised to 1.0 = average
        disc_cur   = (bb_reg - k_reg) - (LG_BB_PCT - LG_K_PCT)
        plate_reg_val = 1.0 + disc_cur * 3.5   # ±10% range for realistic spreads
        plate_reg_val = max(0.85, min(1.15, plate_reg_val))

        # ── Normalise factors (1.0 = league average) ──────────────────────
        f_xwoba  = xwoba_reg  / LG_XWOBA
        f_wrc    = wrc_reg_val / 100.0
        f_barrel = barrel_reg / LG_BARREL_PA
        f_plate  = plate_reg_val

        # ── Composite score → current season λ ────────────────────────────
        composite = _composite_score(f_xwoba, f_wrc, f_barrel, f_plate)
        lambda_cur = composite * LG_RPG

        # ── Prior season λ (using same formula on prior-season metrics) ────
        wrc_pri_val = max(50.0, min(165.0,
            ((woba_pri - LG_WOBA) / LG_WOBA_SCALE) * 100 + 100
        ))
        disc_pri  = (bb_pri - k_pri) - (LG_BB_PCT - LG_K_PCT)
        plate_pri = max(0.85, min(1.15, 1.0 + disc_pri * 3.5))
        composite_pri = _composite_score(
            xwoba_pri / LG_XWOBA,
            wrc_pri_val / 100.0,
            barrel_pri / LG_BARREL_PA,
            plate_pri,
        )
        lambda_pri = composite_pri * LG_RPG

        # ── Blend current + prior — Bayesian inverse-proportional update ────
        # prior_w = k / (k + PA): at PA=0 → 100% prior; at PA=k → 50/50; at PA=∞ → 0%
        # No arbitrary floor: the formula is naturally correct at all sample sizes.
        prior_w   = _PRIOR_PA_EQUIVALENT / (_PRIOR_PA_EQUIVALENT + pa_cur)
        current_w = pa_cur / (_PRIOR_PA_EQUIVALENT + pa_cur)
        lambda_talent = round(current_w * lambda_cur + prior_w * lambda_pri, 4)

        # Guard against extreme values
        lambda_talent = max(3.0, min(7.0, lambda_talent))

        metadata = {
            "team_id":      team_id,
            "team_name":    team_name,
            "season":       season,
            "pa_current":   int(pa_cur),
            "n_statcast_players": sc_cur["n_players"],
            "metrics": {
                "xwoba_raw":       round(xwoba_cur,  4),
                "xwoba_regressed": round(xwoba_reg,  4),
                "barrel_pa_raw":   round(barrel_cur, 4),
                "barrel_regressed":round(barrel_reg, 4),
                "bb_pct":          round(bb_reg,     4),
                "k_pct":           round(k_reg,      4),
                "wrc_plus_approx": round(wrc_reg_val,2),
            },
            "factors": {
                "f_xwoba":  round(f_xwoba,  4),
                "f_wrc":    round(f_wrc,    4),
                "f_barrel": round(f_barrel, 4),
                "f_plate":  round(f_plate,  4),
            },
            "composite":    round(composite,    4),
            "lambda_cur":   round(lambda_cur,   4),
            "lambda_prior": round(lambda_pri,   4),
            "prior_weight": round(prior_w,      3),
            "lambda_talent":lambda_talent,
        }

        log.info(
            "TTO %s %d | PA=%d  xwOBA=%.3f→%.3f  wRC+=%.0f  "
            "barrel=%.1f%%  λ=%.3f (prior_w=%.0f%%)",
            team_name, season, int(pa_cur),
            xwoba_cur, xwoba_reg, wrc_reg_val,
            barrel_reg * 100, lambda_talent, prior_w * 100,
        )
        return lambda_talent, metadata


# ── Module-level convenience function ────────────────────────────────────────

_engine: Optional[TrueTalentOffenseEngine] = None


def get_true_talent_lambda(
    team_id:   int,
    team_name: str,
    season:    Optional[int] = None,
) -> Tuple[float, dict]:
    """
    Convenience wrapper — returns (λ_talent, metadata).
    Re-uses a module-level engine instance to share cached Savant data
    across multiple calls in the same process (e.g. home + away in one game).
    """
    global _engine
    if _engine is None:
        _engine = TrueTalentOffenseEngine()
    return _engine.get_lambda(team_id, team_name, season)
