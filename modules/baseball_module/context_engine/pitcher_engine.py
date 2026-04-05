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
        
        # Pesos para cada factor (ajustables)
        self.weights = {
            'pitcher_quality': 0.30,      # Calidad base del pitcher
            'pitcher_form': 0.20,         # Forma reciente
            'pitcher_matchup': 0.15,      # Vs este equipo
            'pitcher_fatigue': 0.10,      # Cansancio/days rest
            'park_for_pitcher': 0.10,     # Park effect en pitchers
            'travel_pitcher': 0.05,       # Viaje del pitcher
            'bullpen_quality': 0.10       # Calidad del bullpen
        }
        
        # Park factors para PITCHERS (diferente a bateadores)
        # Valores: multiplicador de ERA (>1.0 = duro, <1.0 = fácil)
        self.pitcher_park_factors = {
            'Coors Field': 1.25,          # Altitude = pesadilla
            'Great American Ball Park': 1.12,
            'Yankee Stadium': 1.08,
            'Fenway Park': 1.06,
            'Rogers Centre': 1.05,
            'Chase Field': 1.04,
            'Camden Yards': 1.03,
            'Citizens Bank Park': 1.02,
            'Guaranteed Rate Field': 1.02,
            'Minute Maid Park': 1.01,
            
            # Neutral
            'Truist Park': 1.00,
            'Busch Stadium': 1.00,
            
            # Pitcher friendly
            'Oracle Park': 0.92,          # Grande, foul territory
            'T-Mobile Park': 0.94,
            'Dodger Stadium': 0.95,
            'Petco Park': 0.93,
            'Kauffman Stadium': 0.96,
            'Comerica Park': 0.97,
            'Marlins Park': 0.98,
            'Tropicana Field': 0.98,
        }
    
    
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
            'pitcher_name': pitcher.get('name', 'Unknown'),
            'quality_mult': 1.0,
            'form_mult': 1.0,
            'matchup_mult': 1.0,
            'fatigue_mult': 1.0,
            'park_mult': 1.0,
            'travel_mult': 1.0,
            'bullpen_mult': 1.0,
            'total_multiplier': 1.0
        }
        
        # 1. PITCHER QUALITY (ERA, FIP, xFIP)
        result['quality_mult'] = self._adjust_pitcher_quality(pitcher)
        
        # 2. PITCHER FORM (últimos 5-10 starts)
        result['form_mult'] = self._adjust_pitcher_form(pitcher)
        
        # 3. PITCHER VS TEAM MATCHUP
        result['matchup_mult'] = self._adjust_pitcher_matchup(
            pitcher, game_data, is_home
        )
        
        # 4. PITCHER FATIGUE (days rest, pitch count)
        result['fatigue_mult'] = self._adjust_pitcher_fatigue(pitcher)
        
        # 5. PARK FACTORS FOR PITCHER
        result['park_mult'] = self._adjust_park_for_pitcher(
            pitcher, game_data, is_home
        )
        
        # 6. TRAVEL FATIGUE OF PITCHER
        result['travel_mult'] = self._adjust_pitcher_travel(
            pitcher, game_data, is_home
        )
        
        # 7. BULLPEN QUALITY
        result['bullpen_mult'] = self._adjust_bullpen(
            game_data, is_home
        )
        
        # COMBINAR TODO
        result['total_multiplier'] = (
            result['quality_mult'] * self.weights['pitcher_quality'] +
            result['form_mult'] * self.weights['pitcher_form'] +
            result['matchup_mult'] * self.weights['pitcher_matchup'] +
            result['fatigue_mult'] * self.weights['pitcher_fatigue'] +
            result['park_mult'] * self.weights['park_for_pitcher'] +
            result['travel_mult'] * self.weights['travel_pitcher'] +
            result['bullpen_mult'] * self.weights['bullpen_quality']
        )
        
        # Normalizar (los weights suman 1.0)
        # El multiplicador ya está ponderado correctamente
        
        return result
    
    
    def _adjust_pitcher_quality(self, pitcher: Dict) -> float:
        """
        Ajusta por calidad base del pitcher (ERA, FIP, xFIP).
        
        Returns:
            Multiplicador (1.0 = average, <1.0 = elite, >1.0 = malo)
        """
        
        # Obtener stats (con defaults)
        era = pitcher.get('era', 4.50)
        fip = pitcher.get('fip', era)  # Usar ERA si no hay FIP
        xfip = pitcher.get('xfip', fip)  # Usar FIP si no hay xFIP
        
        # Ponderar: xFIP > FIP > ERA
        # xFIP es mejor predictor futuro
        composite = (
            era * 0.30 +
            fip * 0.35 +
            xfip * 0.35
        )
        
        # League average ≈ 4.20
        league_avg = 4.20
        
        # Convertir a multiplicador
        # ERA 3.00 (elite) → ~0.85x (menos runs)
        # ERA 4.20 (average) → 1.00x
        # ERA 5.50 (malo) → ~1.15x (más runs)
        
        multiplier = composite / league_avg
        
        # Clamp para evitar extremos
        multiplier = np.clip(multiplier, 0.70, 1.30)
        
        return multiplier
    
    
    def _adjust_pitcher_form(self, pitcher: Dict) -> float:
        """
        Ajusta por forma reciente (últimos 5-10 starts).
        """
        
        recent_era = pitcher.get('era_last_5', pitcher.get('era', 4.50))
        season_era = pitcher.get('era', 4.50)
        
        # Si recent ERA mucho mejor/peor que season, ajustar
        diff = recent_era - season_era
        
        # Hot: recent_era 2.50, season 3.50 → diff = -1.00
        # Cold: recent_era 5.50, season 4.00 → diff = +1.50
        
        # Convertir diff a multiplicador
        # -1.00 ERA diff → ~0.92x (mejor forma)
        # +1.00 ERA diff → ~1.08x (peor forma)
        
        form_adjustment = 1.0 + (diff * 0.08)
        
        form_adjustment = np.clip(form_adjustment, 0.85, 1.15)
        
        return form_adjustment
    
    
    def _adjust_pitcher_matchup(
        self,
        pitcher: Dict,
        game_data: Dict,
        is_home: bool
    ) -> float:
        """
        Ajusta por matchup histórico pitcher vs este equipo.
        """
        
        # Obtener opponent team
        if is_home:
            opp_team = game_data.get('away_team', {})
        else:
            opp_team = game_data.get('home_team', {})
        
        # ERA vs este equipo (career)
        era_vs_team = pitcher.get(f"era_vs_{opp_team.get('abbr', '')}", None)
        
        if era_vs_team is None:
            return 1.0  # Sin datos de matchup
        
        season_era = pitcher.get('era', 4.50)
        
        # Si ERA vs team >> season ERA → malo matchup
        # Si ERA vs team << season ERA → buen matchup
        
        diff = era_vs_team - season_era
        
        matchup_mult = 1.0 + (diff * 0.06)
        matchup_mult = np.clip(matchup_mult, 0.85, 1.15)
        
        return matchup_mult
    
    
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
    
    
    def _adjust_park_for_pitcher(
        self,
        pitcher: Dict,
        game_data: Dict,
        is_home: bool
    ) -> float:
        """
        Ajusta por park factors ESPECÍFICOS PARA PITCHERS.
        
        NOTA: Esto es diferente al park factor para bateadores (HFA).
        """
        
        park_name = game_data.get('park', {}).get('name', 'Unknown')
        
        # Obtener park factor para pitchers
        park_mult = self.pitcher_park_factors.get(park_name, 1.0)
        
        # Si pitcher es away, juega en este park
        # Si pitcher es home, ya está acostumbrado (mitad del efecto)
        if is_home:
            # Pitcher home ya conoce el park
            park_mult = 1.0 + (park_mult - 1.0) * 0.5
        
        return park_mult
    
    
    def _adjust_pitcher_travel(
        self,
        pitcher: Dict,
        game_data: Dict,
        is_home: bool
    ) -> float:
        """
        Ajusta por travel fatigue DEL PITCHER.
        
        NOTA: Diferente al travel del equipo (HFA).
        """
        
        if is_home:
            return 1.0  # Pitcher home no viajó
        
        # Pitcher away viajó
        miles_traveled = game_data.get('miles_traveled_away', 0)
        time_zones_crossed = game_data.get('time_zones_crossed_away', 0)
        
        travel_mult = 1.0
        
        # Distance
        if miles_traveled > 2000:  # Cross-country
            travel_mult *= 1.04
        elif miles_traveled > 1000:
            travel_mult *= 1.02
        
        # Time zones (afecta sleep/circadian rhythm)
        if time_zones_crossed >= 2:
            travel_mult *= 1.03
        elif time_zones_crossed == 1:
            travel_mult *= 1.01
        
        travel_mult = np.clip(travel_mult, 1.0, 1.08)
        
        return travel_mult
    
    
    def _adjust_bullpen(
        self,
        game_data: Dict,
        is_home: bool
    ) -> float:
        """
        Ajusta por calidad y disponibilidad del bullpen.
        """
        
        if is_home:
            bullpen = game_data.get('bullpen_home', {})
        else:
            bullpen = game_data.get('bullpen_away', {})
        
        # Bullpen ERA
        bullpen_era = bullpen.get('era', 4.20)
        league_avg_bullpen = 4.20
        
        # Workload reciente (innings últimos 3 días)
        ip_last_3 = bullpen.get('ip_last_3_days', 3.0)
        
        # Closer disponible
        closer_available = bullpen.get('closer_available', True)
        
        # Multiplicador base por ERA
        bullpen_mult = bullpen_era / league_avg_bullpen
        
        # Ajustar por workload
        if ip_last_3 > 6.0:  # Bullpen cansado
            bullpen_mult *= 1.05
        
        # Ajustar por closer
        if not closer_available:
            bullpen_mult *= 1.03  # Sin closer = peor
        
        bullpen_mult = np.clip(bullpen_mult, 0.85, 1.15)
        
        return bullpen_mult


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
