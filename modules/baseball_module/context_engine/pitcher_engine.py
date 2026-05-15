"""
PITCHER ENGINE G13 PRO - VERSIÓN ENFOCADA
===========================================

Módulo especializado en ajustar lambdas SOLO por factores de PITCHERS.

Responsabilidades:
✅ Pitcher quality (ERA, WHIP, FIP, xFIP, SIERA)
✅ Pitcher recent form (últimos 5-10 starts)
✅ Pitcher vs team matchup (histórico vs este lineup)
✅ Pitcher fatigue (pitch count, days rest)
✅ Park factors PARA PITCHERS (no para bateadores - eso es HFA)
✅ Travel fatigue DEL PITCHER (no del equipo - eso es HFA)
✅ Bullpen workload & quality
✅ Closer availability

NO toca:
❌ Home advantage del equipo (eso es HFA)
❌ Park factors para bateadores (eso es HFA)
❌ Travel fatigue del equipo bateo/fielding (eso es HFA)
❌ Offense/defense del equipo (eso es HFA)

Autor: Braulio & Claude
Versión: G13 Pro Focused
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from config import PITCHER_ENGINE_WEIGHTS

logger = logging.getLogger(__name__)


class PitcherEngine:
    """
    Motor de ajuste por factores de pitchers.
    
    Ajusta lambdas de Poisson basado en:
    - Calidad del pitcher starter
    - Forma reciente del pitcher
    - Matchup histórico pitcher vs equipo
    - Park factors específicos para pitchers
    - Fatiga y travel del pitcher
    - Calidad y disponibilidad del bullpen
    """
    
    def __init__(self):
        self.name = "PitcherEngine G13 Pro Focused"
        self.weights = dict(PITCHER_ENGINE_WEIGHTS)
    
    
    def adjust_for_pitchers(
        self,
        lh: float,
        la: float,
        game_data: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Ajusta lambdas por todos los factores de pitchers.
        
        Args:
            lh: Lambda home después de HFA/Calibration
            la: Lambda away después de HFA/Calibration
            game_data: Dict completo con info del juego
        
        Returns:
            (lh_adjusted, la_adjusted, metadata)
        """
        
        logger.info(f"🎯 Pitcher Engine G13 Pro - Ajustando lambdas")
        logger.info(f"   Input: λ_h={lh:.3f}, λ_a={la:.3f}")
        
        metadata = {
            'adjustments': {},
            'pitcher_home': {},
            'pitcher_away': {}
        }
        
        # Extraer datos
        pitcher_home = game_data.get('pitcher_home', {})
        pitcher_away = game_data.get('pitcher_away', {})
        park = game_data.get('park', {})
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 1: AJUSTAR POR PITCHER AWAY (afecta λ_home)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        adj_away = self._calculate_pitcher_adjustment(
            pitcher_away,
            game_data,
            is_home=False
        )
        
        lh_new = lh * adj_away['total_multiplier']
        metadata['pitcher_away'] = adj_away
        
        logger.info(f"   Pitcher Away ({pitcher_away.get('name', 'Unknown')}):")
        logger.info(f"   └─ Multiplier: {adj_away['total_multiplier']:.3f}")
        logger.info(f"   └─ λ_home: {lh:.3f} → {lh_new:.3f}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 2: AJUSTAR POR PITCHER HOME (afecta λ_away)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        adj_home = self._calculate_pitcher_adjustment(
            pitcher_home,
            game_data,
            is_home=True
        )
        
        la_new = la * adj_home['total_multiplier']
        metadata['pitcher_home'] = adj_home
        
        logger.info(f"   Pitcher Home ({pitcher_home.get('name', 'Unknown')}):")
        logger.info(f"   └─ Multiplier: {adj_home['total_multiplier']:.3f}")
        logger.info(f"   └─ λ_away: {la:.3f} → {la_new:.3f}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # RESULTADO FINAL
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        logger.info(f"✅ Pitcher Engine completado:")
        logger.info(f"   λ_home: {lh:.3f} → {lh_new:.3f} (Δ={lh_new-lh:+.3f})")
        logger.info(f"   λ_away: {la:.3f} → {la_new:.3f} (Δ={la_new-la:+.3f})")
        
        return lh_new, la_new, metadata
    
    
    def _calculate_pitcher_adjustment(
        self,
        pitcher: Dict[str, Any],
        game_data: Dict[str, Any],
        is_home: bool
    ) -> Dict[str, Any]:
        """
        Calcula el ajuste completo para UN pitcher.
        
        Returns:
            Dict con multiplicador total y desglose
        """
        
        result = {
            'pitcher_name':     pitcher.get('name', 'Unknown'),
            'quality_mult':     1.0,
            'form_mult':        1.0,
            'matchup_mult':     1.0,
            'platoon_mult':     1.0,
            'fatigue_mult':     1.0,
            'total_multiplier': 1.0,
        }

        # 1. PITCHER QUALITY (SIERA → xFIP → FIP → ERA, + Savant overlays)
        result['quality_mult'] = self._adjust_pitcher_quality(pitcher)

        # 2. PITCHER FORM (era_last_5 level + trend slope + quality_start_pct)
        result['form_mult'] = self._adjust_pitcher_form(pitcher)

        # 3. PITCHER VS TEAM MATCHUP
        result['matchup_mult'] = self._adjust_pitcher_matchup(
            pitcher, game_data, is_home
        )

        # 4. PLATOON SPLITS × LINEUP HANDEDNESS
        result['platoon_mult'] = self._adjust_pitcher_platoon(
            pitcher, game_data, is_home
        )

        # 5. PITCHER FATIGUE (days rest, pitch count)
        result['fatigue_mult'] = self._adjust_pitcher_fatigue(pitcher)

        # Combine via delta formula: 1.0 + Σ((factor - 1.0) × weight)
        result['total_multiplier'] = 1.0 + (
            (result['quality_mult']  - 1.0) * self.weights['pitcher_quality'] +
            (result['form_mult']     - 1.0) * self.weights['pitcher_form'] +
            (result['matchup_mult']  - 1.0) * self.weights['pitcher_matchup'] +
            (result['platoon_mult']  - 1.0) * self.weights['pitcher_platoon'] +
            (result['fatigue_mult']  - 1.0) * self.weights['pitcher_fatigue']
        )

        return result
    
    
    def _adjust_pitcher_quality(self, pitcher: Dict) -> float:
        """
        Pitcher quality multiplier.

        ERA estimator fallback (most predictive → least):
          SIERA → xFIP → FIP → ERA

        Contact quality overlay from Baseball Savant:
          est_woba (xwOBA allowed)  — expected batting value per PA
          brl_percent               — barrel rate allowed; predicts HR/XBH

        League averages: ERA ≈ 4.20, xwOBA ≈ 0.320, barrel% ≈ 8.0
        Output clamped to [0.70, 1.30].
        """
        era   = pitcher.get("era", 4.50)
        fip   = pitcher.get("fip")
        xfip  = pitcher.get("xfip")
        siera = pitcher.get("siera")

        primary = (
            siera if siera is not None else
            xfip  if xfip  is not None else
            fip   if fip   is not None else
            era
        )
        skill_mult = primary / 4.20

        # xwOBA penalty: each 0.010 above league avg (0.320) ≈ +3% runs allowed
        est_woba = pitcher.get("est_woba")
        if est_woba is not None:
            woba_mult = 1.0 + (float(est_woba) - 0.320) * 3.0
        else:
            woba_mult = 1.0

        # Barrel% penalty: each 1 ppt above league avg (8%) ≈ +1.2% runs allowed
        brl_pct = pitcher.get("brl_percent")
        if brl_pct is not None:
            brl_mult = 1.0 + max(0.0, float(brl_pct) - 8.0) * 0.012
        else:
            brl_mult = 1.0

        multiplier = float(np.clip(skill_mult * woba_mult * brl_mult, 0.70, 1.30))
        return multiplier
    
    
    def _adjust_pitcher_form(self, pitcher: Dict) -> float:
        """
        Adjust for recent form using three signals:

        1. ERA level  — era_last_5 vs season ERA (how does recent compare overall?)
        2. ERA trend  — slope of per-start ERA, oldest→newest (getting better/worse?)
        3. QS%        — fraction of recent starts that were quality starts (≥6 IP ≤3 ER)

        All three push in the same direction so no signal cancels another.
        Clamped to [0.85, 1.15] to match the other factor limits.
        """
        recent_era = pitcher.get('era_last_5', pitcher.get('era', 4.50))
        season_era = pitcher.get('era', 4.50)

        # Level: positive diff = pitcher ERA worse recently = more opponent runs
        level_diff = recent_era - season_era
        level_adj  = 1.0 + (level_diff * 0.06)

        # Trend: negative slope = ERA dropping (improving) → opponent scores less
        era_trend  = pitcher.get('era_trend', 0.0)
        trend_adj  = 1.0 + (era_trend * 0.03)

        # Quality-start %: higher = pitcher dominant = less opponent scoring
        qs_pct    = pitcher.get('quality_start_pct', 0.50)
        qs_adj    = 1.0 - (qs_pct - 0.50) * 0.06  # neutral at 50%, ±3% at extremes

        form_adj = level_adj * trend_adj * qs_adj
        return float(np.clip(form_adj, 0.85, 1.15))

    def _adjust_pitcher_platoon(
        self, pitcher: Dict, game_data: Dict, is_home: bool
    ) -> float:
        """
        Adjust for the platoon mismatch between the pitcher's L/R splits
        and the opposing lineup's actual handedness composition.

        Data flow:
          pitcher['platoon_splits'] = {'vs_lhb': {ip, whip, ...},
                                       'vs_rhb': {ip, whip, ...}}
          game_data['home_lineup_lhb_pct'] / 'away_lineup_lhb_pct'  (0–1)

        Home pitcher faces the away lineup (away_lineup_lhb_pct).
        Away pitcher faces the home lineup (home_lineup_lhb_pct).

        A pitcher with a large WHIP differential between splits will be
        penalised/rewarded based on how many of that type are in today's lineup.
        Returns 1.0 when data is insufficient.
        """
        platoon = pitcher.get('platoon_splits')
        if not platoon:
            return 1.0

        vs_lhb = platoon.get('vs_lhb') or {}
        vs_rhb = platoon.get('vs_rhb') or {}
        if not vs_lhb or not vs_rhb:
            return 1.0

        # Fraction of LHB in the *opposing* lineup
        if is_home:
            opp_lhb_pct = float(game_data.get('away_lineup_lhb_pct', 0.45))
        else:
            opp_lhb_pct = float(game_data.get('home_lineup_lhb_pct', 0.45))
        opp_rhb_pct = 1.0 - opp_lhb_pct

        # Lineup-composition-weighted WHIP
        lhb_whip = float(vs_lhb.get('whip', 1.30))
        rhb_whip = float(vs_rhb.get('whip', 1.30))
        lineup_whip = opp_lhb_pct * lhb_whip + opp_rhb_pct * rhb_whip

        overall_whip = float(pitcher.get('whip', 1.30))
        if overall_whip <= 0:
            return 1.0

        # lineup_whip > overall_whip → today's lineup is harder for this pitcher
        platoon_mult = lineup_whip / overall_whip
        return float(np.clip(platoon_mult, 0.93, 1.07))
    
    
    def _adjust_pitcher_matchup(
        self,
        pitcher: Dict,
        game_data: Dict,
        is_home: bool
    ) -> float:
        """
        Ajusta por matchup histórico pitcher vs este equipo.
        era_vs_opp is populated by data_fetchers via the MLB Stats API vsTeam endpoint.
        """
        era_vs_team = pitcher.get("era_vs_opp")
        if era_vs_team is None:
            return 1.0

        season_era = pitcher.get('era', 4.50)
        ip_vs_opp = pitcher.get('ip_vs_opp', 0)

        # Shrink toward season ERA when sample is small (< 15 IP)
        if ip_vs_opp and ip_vs_opp < 15:
            weight = ip_vs_opp / 15.0
            era_vs_team = era_vs_team * weight + season_era * (1 - weight)

        diff = era_vs_team - season_era
        matchup_mult = 1.0 + (diff * 0.06)
        matchup_mult = np.clip(matchup_mult, 0.85, 1.15)

        return float(matchup_mult)
    
    
    def _adjust_pitcher_fatigue(self, pitcher: Dict) -> float:
        """
        Ajusta por fatiga del pitcher (days rest, pitch count).
        """
        
        days_rest = pitcher.get('days_rest', 4)
        last_pitch_count = pitcher.get('last_pitch_count', 90)
        
        fatigue_mult = 1.0
        
        # Days rest
        if days_rest < 4:  # Menos descanso de lo normal
            fatigue_mult *= 1.05  # Peor rendimiento
        elif days_rest > 5:  # Demasiado descanso (rust)
            fatigue_mult *= 1.02
        
        # Pitch count alto en último start
        if last_pitch_count > 105:
            fatigue_mult *= 1.04  # Cansancio residual
        
        fatigue_mult = np.clip(fatigue_mult, 0.95, 1.10)
        
        return fatigue_mult
    
    


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FUNCIÓN HELPER PARA USAR EN RUN_MODULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def adjust_for_pitchers(
    lh: float,
    la: float,
    game_data: Dict[str, Any]
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Función helper para importar en run_module.
    
    Usage:
        from context_engine.pitcher_engine import adjust_for_pitchers
        
        lh_adj, la_adj, meta = adjust_for_pitchers(lh, la, game_data)
    """
    
    engine = PitcherEngine()
    return engine.adjust_for_pitchers(lh, la, game_data)
