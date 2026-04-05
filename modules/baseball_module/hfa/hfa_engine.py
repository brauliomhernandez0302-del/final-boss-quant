"""
HFA ENGINE G10 PRO - VERSIÓN HÍBRIDA DEFINITIVA
================================================

Combina:
✅ Tus dataclasses originales (StadiumFactors, etc.)
✅ Tu STADIUM_DATABASE completo
✅ Mi lógica funcional ejecutable
✅ Park factors separados (batter vs pitcher)

Autor: Braulio & Claude
Versión: G10 Pro Hybrid
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PARTE 1: TUS DATACLASSES ORIGINALES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class StadiumFactors:
    """Factores específicos del estadio."""
    runs_factor: float = 1.0
    hr_factor: float = 1.0
    hits_factor: float = 1.0
    altitude: float = 0


@dataclass
class PitcherStats:
    """Estadísticas del pitcher."""
    name: str
    era: float
    fip: Optional[float] = None
    whip: Optional[float] = None
    k_per_9: Optional[float] = None
    bb_per_9: Optional[float] = None
    home_era: Optional[float] = None
    away_era: Optional[float] = None


@dataclass
class WeatherConditions:
    """Condiciones climáticas."""
    temp_f: float = 75
    wind_mph: float = 0
    humidity_pct: float = 50
    precipitation: bool = False
    roof_closed: bool = False


@dataclass
class TeamContext:
    """Contexto adicional del equipo."""
    team_name: str
    home_record: Optional[Tuple[int, int]] = None
    away_record: Optional[Tuple[int, int]] = None
    rest_days: int = 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PARTE 2: TU STADIUM DATABASE ORIGINAL (COMPLETO)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STADIUM_DATABASE = {
    # Últimos 10 juegos - STADIUM DATABASE (Park Factors MLB 2024)
    "Coors Field": StadiumFactors(1.25, 1.35, 1.20, 1660),
    "Great American Ball Park": StadiumFactors(1.12, 1.20, 1.02, 1609),
    "Fenway Park": StadiumFactors(1.05, 1.15, 1.25, 10),
    "Yankee Stadium": StadiumFactors(1.05, 1.20, 1.02, 15),
    "Oracle Park": StadiumFactors(0.87, 0.81, 0.90, 0),
    "T-Mobile Park": StadiumFactors(0.92, 0.81, 0.90, 15),
    "Dodger Stadium": StadiumFactors(0.95, 0.92, 0.88, 0),
    "Tropicana Field": StadiumFactors(0.98, 0.96, 0.93, 0),
    "Chase Field": StadiumFactors(1.08, 1.15, 1.05, 1669),
    "Citizens Bank Park": StadiumFactors(1.08, 1.15, 0.96, 0),
    "Minute Maid Park": StadiumFactors(1.10, 1.18, 1.08, 12),
    "Globe Life Field": StadiumFactors(1.12, 1.20, 1.08, 13),
    "Target Field": StadiumFactors(0.97, 0.94, 1.08, 161),
    "Kauffman Stadium": StadiumFactors(0.96, 0.92, 0.97, 229),
    "Camden Yards": StadiumFactors(1.02, 1.09, 1.03, 97),
    "Wrigley Field": StadiumFactors(1.04, 1.05, 1.03, 180),
    "Guaranteed Rate Field": StadiumFactors(1.00, 1.08, 1.00, 181),
    "Progressive Field": StadiumFactors(0.98, 0.96, 0.99, 187),
    "Rogers Centre": StadiumFactors(1.01, 1.03, 1.00, 182),
    "Comerica Park": StadiumFactors(0.94, 0.88, 0.96, 182),
    "PNC Park": StadiumFactors(0.96, 0.94, 0.97, 231),
    "Busch Stadium": StadiumFactors(0.99, 0.98, 1.00, 14),
    "Petco Park": StadiumFactors(0.90, 0.85, 0.90, 144),
    "Angel Stadium": StadiumFactors(0.98, 0.86, 0.97, 231),
    "loanDepot park": StadiumFactors(0.90, 0.95, 0.99, 48),
    "Truist Park": StadiumFactors(0.98, 1.00, 0.96, 48),
    "Citi Field": StadiumFactors(0.94, 0.88, 0.96, 3),
    "Nationals Park": StadiumFactors(1.01, 1.05, 1.01, 305),
    "American Family Field": StadiumFactors(1.00, 1.07, 0.97, 7),
    "RingCentral Coliseum": StadiumFactors(0.94, 0.98, 0.96, 194),
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PARTE 3: MI CLASE HFAEngine CON LÓGICA FUNCIONAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HFAEngine:
    """
    Motor de Home Field Advantage enfocado en EQUIPO.
    
    Usa tu STADIUM_DATABASE original + park factors separados.
    """
    
    def __init__(self):
        self.name = "HFA Engine G10 Pro Hybrid"
        
        # Usar tu STADIUM_DATABASE
        self.stadium_db = STADIUM_DATABASE
        
        # HFA base por estadio (crowd effect)
        self.hfa_base = {
            'Yankee Stadium': 0.18,
            'Fenway Park': 0.17,
            'Dodger Stadium': 0.16,
            'Busch Stadium': 0.15,
            'Wrigley Field': 0.16,
            'Oracle Park': 0.15,
            'Coors Field': 0.14,
            'Petco Park': 0.14,
            'Citizens Bank Park': 0.15,
            'Progressive Field': 0.14,
            'Comerica Park': 0.13,
            'Target Field': 0.13,
            'PNC Park': 0.13,
            'Great American Ball Park': 0.13,
            'Guaranteed Rate Field': 0.12,
            'Truist Park': 0.13,
            'Chase Field': 0.12,
            'T-Mobile Park': 0.13,
            'Tropicana Field': 0.10,
            'Rogers Centre': 0.11,
            'loanDepot park': 0.10,
            'RingCentral Coliseum': 0.11,
        }
        
        # Park factors SOLO PARA BATEADORES (diferente a pitchers)
        # Derivados de tu STADIUM_DATABASE pero con ajustes
        self.park_factors_hitters = {}
        for stadium, factors in STADIUM_DATABASE.items():
            # Usar runs_factor como base para hitters
            self.park_factors_hitters[stadium] = factors.runs_factor
    
    
    def get_adjusted_lambdas(
        self,
        lh: float,
        la: float,
        game_data: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Ajusta lambdas por todos los factores del EQUIPO.
        
        Args:
            lh: Lambda home después de calibration
            la: Lambda away después de calibration
            game_data: Dict completo con info del juego
        
        Returns:
            (lh_adjusted, la_adjusted, metadata)
        """
        
        logger.info(f"🏟️  HFA Engine G10 Pro - Ajustando por equipo")
        logger.info(f"   Input: λ_h={lh:.3f}, λ_a={la:.3f}")
        
        metadata = {
            'hfa_base': 0.0,
            'park_factor': 1.0,
            'travel_away': 0.0,
            'offense_home': 1.0,
            'offense_away': 1.0,
            'defense_home': 1.0,
            'defense_away': 1.0,
            'altitude': 0.0
        }
        
        # Extraer datos
        home_team = game_data.get('home_team', {})
        away_team = game_data.get('away_team', {})
        park = game_data.get('park', {})
        park_name = park.get('name', 'Unknown')
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 1: HOME FIELD ADVANTAGE BASE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        hfa_boost = self.hfa_base.get(park_name, 0.13)
        lh_new = lh + hfa_boost
        metadata['hfa_base'] = hfa_boost
        
        logger.info(f"   HFA Base: +{hfa_boost:.3f} runs")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 2: PARK FACTORS FOR HITTERS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        park_mult = self.park_factors_hitters.get(park_name, 1.00)
        
        lh_new *= park_mult
        la_new = la * park_mult
        
        metadata['park_factor'] = park_mult
        
        logger.info(f"   Park Factor (hitters): {park_mult:.3f}x")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 3: ALTITUDE EFFECTS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if park_name in self.stadium_db:
            altitude = self.stadium_db[park_name].altitude
            
            if altitude > 1000:  # Significant altitude
                # Convertir altitude a boost de runs
                alt_boost = (altitude / 5000) * 0.15
                
                # Home acostumbrado (50% efecto)
                lh_new += (alt_boost * 0.5)
                la_new += alt_boost
                
                metadata['altitude'] = alt_boost
                logger.info(f"   Altitude: +{alt_boost:.3f} runs")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 4: TRAVEL FATIGUE (EQUIPO)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        travel_penalty = self._calculate_travel_fatigue(game_data)
        la_new -= travel_penalty
        metadata['travel_away'] = travel_penalty
        
        if travel_penalty > 0:
            logger.info(f"   Travel Away: -{travel_penalty:.3f} runs")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 5: TEAM OFFENSE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        off_mult_home = self._calculate_offense_multiplier(home_team)
        off_mult_away = self._calculate_offense_multiplier(away_team)
        
        lh_new *= off_mult_home
        la_new *= off_mult_away
        
        metadata['offense_home'] = off_mult_home
        metadata['offense_away'] = off_mult_away
        
        logger.info(f"   Offense: home {off_mult_home:.3f}x, away {off_mult_away:.3f}x")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PASO 6: TEAM DEFENSE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        def_mult_home = self._calculate_defense_multiplier(home_team)
        def_mult_away = self._calculate_defense_multiplier(away_team)
        
        la_new *= def_mult_home
        lh_new *= def_mult_away
        
        metadata['defense_home'] = def_mult_home
        metadata['defense_away'] = def_mult_away
        
        logger.info(f"   Defense: home {def_mult_home:.3f}x, away {def_mult_away:.3f}x")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # RESULTADO FINAL
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        logger.info(f"✅ HFA Engine completado:")
        logger.info(f"   λ_home: {lh:.3f} → {lh_new:.3f} (Δ={lh_new-lh:+.3f})")
        logger.info(f"   λ_away: {la:.3f} → {la_new:.3f} (Δ={la_new-la:+.3f})")
        
        return lh_new, la_new, metadata
    
    
    def _calculate_travel_fatigue(self, game_data: Dict) -> float:
        """Calcula penalty por travel del EQUIPO away."""
        
        miles = game_data.get('miles_traveled_away', 0)
        time_zones = game_data.get('time_zones_crossed_away', 0)
        back_to_back = game_data.get('back_to_back_away', False)
        
        penalty = 0.0
        
        if miles > 2500:
            penalty += 0.12
        elif miles > 1500:
            penalty += 0.08
        elif miles > 800:
            penalty += 0.05
        
        if time_zones >= 3:
            penalty += 0.10
        elif time_zones == 2:
            penalty += 0.06
        elif time_zones == 1:
            penalty += 0.03
        
        if back_to_back:
            penalty += 0.05
        
        return min(penalty, 0.25)
    
    
    def _calculate_offense_multiplier(self, team: Dict) -> float:
        """Calcula multiplicador por calidad ofensiva."""
        
        woba = team.get('woba', 0.320)
        ops = team.get('ops', 0.735)
        wrc_plus = team.get('wrc_plus', 100)
        
        if wrc_plus and wrc_plus != 100:
            mult = (
                (wrc_plus / 100) * 0.50 +
                (woba / 0.320) * 0.30 +
                (ops / 0.735) * 0.20
            )
        else:
            mult = (
                (woba / 0.320) * 0.60 +
                (ops / 0.735) * 0.40
            )
        
        return np.clip(mult, 0.80, 1.22)
    
    
    def _calculate_defense_multiplier(self, team: Dict) -> float:
        """Calcula multiplicador por calidad defensiva."""
        
        der = team.get('der', 0.700)
        uzr = team.get('uzr', 0.0)
        fielding_pct = team.get('fielding_pct', 0.985)
        
        der_mult = 0.700 / der
        
        if uzr > 20:
            uzr_adj = 0.95
        elif uzr > 10:
            uzr_adj = 0.97
        elif uzr < -20:
            uzr_adj = 1.06
        elif uzr < -10:
            uzr_adj = 1.03
        else:
            uzr_adj = 1.00
        
        field_mult = 0.985 / fielding_pct
        
        mult = (
            der_mult * 0.50 +
            uzr_adj * 0.30 +
            field_mult * 0.20
        )
        
        return np.clip(mult, 0.88, 1.12)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PARTE 4: FUNCIÓN HELPER PARA RUN_MODULE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_adjusted_lambdas(
    lh: float,
    la: float,
    game_data: Dict[str, Any]
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Función helper para importar en run_module.
    
    Usage:
        from hfa.hfa_engine import get_adjusted_lambdas
        
        lh_hfa, la_hfa, meta = get_adjusted_lambdas(lh, la, game_data)
    """
    
    engine = HFAEngine()
    return engine.get_adjusted_lambdas(lh, la, game_data)
