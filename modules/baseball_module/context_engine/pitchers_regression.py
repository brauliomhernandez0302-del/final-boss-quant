# ==========================================================
# PITCHER REGRESSION ENGINE G2 PRO - ENHANCED
# ==========================================================
# Autor: Braulio & Claude ⚾
# Mejoras sobre G1:
#   + BABIP regression (luck indicator)
#   + LOB% regression (strand rate luck)
#   + Sample size correction (IP-based confidence)
#   + HR/FB% regression (home run luck)
# ==========================================================

from typing import Dict, Tuple
import numpy as np


class PitcherRegressionEngine:
    """
    Evalúa si un pitcher está sobre-rindiendo o sub-rindiendo
    en relación con sus métricas periféricas y luck indicators.
    
    Devuelve un factor de ajuste (λ) y su nivel de confianza.
    """
    
    # League averages para regresión
    LEAGUE_AVG = {
        'babip': 0.300,
        'lob_pct': 0.720,
        'hr_fb_pct': 0.125,
        'era': 4.20,
        'fip': 4.10
    }
    
    def calculate_regression_factor(
        self, 
        pitcher_stats: Dict, 
        opponent_stats: Dict
    ) -> Tuple[float, float]:
        """
        Retorna (factor, confidence)
        factor > 1 → regresión negativa (peor rendimiento esperado)
        factor < 1 → regresión positiva (mejora esperada)
        confidence → 0.0-1.0 basado en sample size y magnitud
        """
        if not pitcher_stats:
            return 1.0, 0.0
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # MÉTRICAS BASE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        era = pitcher_stats.get("era", 4.00)
        fip = pitcher_stats.get("fip", 4.10)
        xfip = pitcher_stats.get("xfip", 4.20)
        siera = pitcher_stats.get("siera", 4.10)
        
        # Recent performance
        last3_era = pitcher_stats.get("last3_era", era)
        
        # Fatigue indicators
        days_rest = pitcher_stats.get("days_rest", 4)
        innings_last3 = pitcher_stats.get("innings_last3", 15)
        
        # Sample size
        innings_pitched = pitcher_stats.get("innings_pitched", 50)
        
        # Luck indicators
        babip = pitcher_stats.get("babip", 0.300)
        lob_pct = pitcher_stats.get("lob_pct", 0.720)
        hr_fb_pct = pitcher_stats.get("hr_fb_pct", 0.125)
        
        # Career baselines (para comparar)
        career_babip = pitcher_stats.get("career_babip", 0.300)
        career_lob = pitcher_stats.get("career_lob_pct", 0.720)
        career_hr_fb = pitcher_stats.get("career_hr_fb_pct", 0.125)
        
        # Opponent
        opponent_ops = opponent_stats.get("ops", 0.740)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. SKILL-BASED REGRESSION (ERA vs peripherals)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # True talent estimate (ponderar FIP/xFIP/SIERA)
        true_talent = (fip * 0.40 + xfip * 0.35 + siera * 0.25)
        
        # Diferencia entre recent ERA y true talent
        diff = last3_era - true_talent
        
        factor = 1.0
        
        if diff < -1.0:
            # Overperformance: ERA mucho mejor que peripherals
            factor *= (1.05 + abs(diff) * 0.02)
        elif diff > 1.0:
            # Underperformance: ERA mucho peor que peripherals
            factor *= (0.96 - diff * 0.01)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2. LUCK INDICATORS REGRESSION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # BABIP (balls in play → luck)
        babip_diff = babip - career_babip
        if babip_diff < -0.030:  # .270 vs .300 → lucky
            factor *= 1.04  # Expect regression up
        elif babip_diff > 0.030:  # .330 vs .300 → unlucky
            factor *= 0.97  # Expect improvement
        
        # LOB% (strand rate → luck)
        lob_diff = lob_pct - career_lob
        if lob_diff > 0.030:  # 75% vs 72% → lucky (stranding too many)
            factor *= 1.03
        elif lob_diff < -0.030:  # 69% vs 72% → unlucky
            factor *= 0.98
        
        # HR/FB% (home run rate → luck)
        hr_fb_diff = hr_fb_pct - career_hr_fb
        if hr_fb_diff > 0.03:  # 15% vs 12% → unlucky (too many HRs)
            factor *= 0.98
        elif hr_fb_diff < -0.03:  # 9% vs 12% → lucky (too few HRs)
            factor *= 1.02
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 3. SAMPLE SIZE CORRECTION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Si sample pequeño, regresión más agresiva hacia career/league avg
        if innings_pitched < 30:
            # Muy pequeño sample → peso a career
            weight_current = 0.30
            weight_career = 0.70
            
            # Regresionar hacia career stats
            regressed_babip = babip * weight_current + career_babip * weight_career
            if abs(babip - regressed_babip) > 0.020:
                factor *= 1.03
        
        elif innings_pitched < 60:
            # Moderado sample → ajuste menor
            weight_current = 0.60
            weight_career = 0.40
            
            regressed_babip = babip * weight_current + career_babip * weight_career
            if abs(babip - regressed_babip) > 0.015:
                factor *= 1.02
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 4. FATIGUE / WORKLOAD
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if days_rest < 4:
            factor *= 1.03  # Fatigue
        elif days_rest > 6:
            factor *= 0.98  # Extra rest (fresher)
        
        if innings_last3 > 18:
            factor *= 1.04  # Overworked
        elif innings_last3 < 12:
            factor *= 0.97  # Under-worked (lack of rhythm)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 5. OPPONENT STRENGTH
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if opponent_ops > 0.780:
            factor *= 1.03  # Tough opponent
        elif opponent_ops < 0.700:
            factor *= 0.97  # Weak opponent
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 6. CONFIDENCE SCORE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Base confidence on:
        # - Magnitude of discrepancy (larger diff = more confident)
        # - Sample size (more IP = more confident)
        # - Number of luck indicators aligned
        
        # Magnitude component
        magnitude_conf = min(0.5, abs(diff) / 4.0)
        
        # Sample size component
        if innings_pitched < 30:
            sample_conf = 0.3
        elif innings_pitched < 60:
            sample_conf = 0.5
        elif innings_pitched < 100:
            sample_conf = 0.7
        else:
            sample_conf = 0.9
        
        # Luck indicators alignment
        luck_signals = 0
        if abs(babip_diff) > 0.020:
            luck_signals += 1
        if abs(lob_diff) > 0.025:
            luck_signals += 1
        if abs(hr_fb_diff) > 0.025:
            luck_signals += 1
        
        luck_conf = min(0.4, luck_signals * 0.15)
        
        # Combined confidence
        confidence = min(0.95, magnitude_conf + sample_conf * 0.4 + luck_conf)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FINAL ADJUSTMENTS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Clip to reasonable range
        factor = np.clip(factor, 0.88, 1.14)
        
        return round(factor, 3), round(confidence, 2)
    
    
    def get_regression_explanation(
        self,
        pitcher_stats: Dict,
        opponent_stats: Dict
    ) -> str:
        """
        Genera explicación human-readable de la regresión.
        Útil para logging/debugging.
        """
        
        factor, confidence = self.calculate_regression_factor(
            pitcher_stats, opponent_stats
        )
        
        era = pitcher_stats.get("era", 4.00)
        fip = pitcher_stats.get("fip", 4.10)
        babip = pitcher_stats.get("babip", 0.300)
        lob_pct = pitcher_stats.get("lob_pct", 0.720)
        
        if factor > 1.05:
            direction = "NEGATIVE regression expected (worse performance)"
            reason = []
            if era < fip - 0.50:
                reason.append("ERA significantly better than FIP (overperforming)")
            if babip < 0.270:
                reason.append("BABIP unusually low (getting lucky)")
            if lob_pct > 0.750:
                reason.append("LOB% very high (strand rate luck)")
            
            explanation = f"{direction}\n  Reasons: " + ", ".join(reason)
        
        elif factor < 0.95:
            direction = "POSITIVE regression expected (improvement)"
            reason = []
            if era > fip + 0.50:
                reason.append("ERA significantly worse than FIP (underperforming)")
            if babip > 0.330:
                reason.append("BABIP unusually high (unlucky)")
            if lob_pct < 0.690:
                reason.append("LOB% very low (strand rate bad luck)")
            
            explanation = f"{direction}\n  Reasons: " + ", ".join(reason)
        
        else:
            explanation = "Minimal regression expected (performing close to true talent)"
        
        explanation += f"\n  Factor: {factor:.3f} | Confidence: {confidence:.2f}"
        
        return explanation


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_pitcher_regression(
    pitcher_stats: Dict,
    opponent_stats: Dict
) -> Tuple[float, float]:
    """
    Helper function para usar en otros módulos.
    
    Usage:
        factor, conf = calculate_pitcher_regression(pitcher, opp)
        lambda_adjusted = lambda_base * factor
    """
    engine = PitcherRegressionEngine()
    return engine.calculate_regression_factor(pitcher_stats, opponent_stats)
