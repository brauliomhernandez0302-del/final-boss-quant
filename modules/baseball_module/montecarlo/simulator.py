
# ==========================================================
# MONTE CARLO ENGINE G9 PRO ULTRA v2.0 FIXED
# ==========================================================

import numpy as np
import math
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MonteCarloLimits:
    MIN_LAMBDA: float = 0.1
    MAX_LAMBDA: float = 20.0
    MIN_SIMS: int = 10_000
    MAX_SIMS: int = 10_000_000
    DEFAULT_BLOCK: int = 200_000
    MIN_BLOCK: int = 10_000
    MAX_NOISE: float = 0.30
    MIN_SE: float = 0.0001
    MAX_SE: float = 0.01

LIMITS = MonteCarloLimits()

def validate_inputs(lh, la, n_max, block, lambda_noise, early_stop_se, total_line):
    if not (LIMITS.MIN_LAMBDA <= lh <= LIMITS.MAX_LAMBDA):
        raise ValueError(f"λ_home={lh:.2f} fuera de rango")
    if not (LIMITS.MIN_LAMBDA <= la <= LIMITS.MAX_LAMBDA):
        raise ValueError(f"λ_away={la:.2f} fuera de rango")
    if not (LIMITS.MIN_SIMS <= n_max <= LIMITS.MAX_SIMS):
        raise ValueError(f"n_max={n_max:,} fuera de rango")
    if not (LIMITS.MIN_BLOCK <= block <= n_max):
        raise ValueError(f"block={block:,} inválido")
    if not (0 <= lambda_noise <= LIMITS.MAX_NOISE):
        raise ValueError(f"lambda_noise={lambda_noise} inválido")
    if not (LIMITS.MIN_SE <= early_stop_se <= LIMITS.MAX_SE):
        raise ValueError(f"early_stop_se={early_stop_se} inválido")
    if total_line is not None and not (0 < total_line < 50):
        raise ValueError(f"total_line={total_line} fuera de rango")

def monte_carlo_advanced(
    lh: float,
    la: float,
    n_max: int = 5_000_000,
    block: int = 200_000,
    total_line: Optional[float] = None,
    rng_seed: Optional[int] = None,
    lambda_noise: float = 0.05,
    early_stop_se: float = 0.003,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    store_samples: bool = True,
    analyze_f5: bool = False,
) -> Dict[str, Any]:
    """Simulación Monte Carlo avanzada CORREGIDA."""
    
    validate_inputs(lh, la, n_max, block, lambda_noise, early_stop_se, total_line)
    
    rng = np.random.default_rng(rng_seed)
    sims_done = 0
    
    logger.info(f"🎲 Monte Carlo: λ_h={lh:.2f}, λ_a={la:.2f}, max={n_max:,}")
    
    # ✅ ACUMULADORES GLOBALES (CRÍTICO)
    wins_home_total = 0
    wins_away_total = 0
    ties_total = 0
    
    if store_samples:
        all_home, all_away, all_total = [], [], []
    
    # Simulación por bloques
    while sims_done < n_max:
        b = min(block, n_max - sims_done)
        
        # Varianza dinámica
        lh_noise = np.clip(
            rng.normal(lh, lambda_noise * max(lh, 0.5), size=b),
            LIMITS.MIN_LAMBDA, LIMITS.MAX_LAMBDA
        )
        la_noise = np.clip(
            rng.normal(la, lambda_noise * max(la, 0.5), size=b),
            LIMITS.MIN_LAMBDA, LIMITS.MAX_LAMBDA
        )
        
        home_runs = rng.poisson(lh_noise)
        away_runs = rng.poisson(la_noise)
        total_runs = home_runs + away_runs
        
        # ✅ ACUMULAR CONTADORES
        wins_home_total += int(np.sum(home_runs > away_runs))
        wins_away_total += int(np.sum(away_runs > home_runs))
        ties_total += int(np.sum(home_runs == away_runs))
        
        if store_samples:
            all_home.append(home_runs)
            all_away.append(away_runs)
            all_total.append(total_runs)
        
        sims_done += b
        
        # Progress callback con protección
        if progress_callback:
            try:
                progress_callback(sims_done, n_max)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
        
        # ✅ EARLY STOP CORREGIDO
        if sims_done >= 500_000 and sims_done % 100_000 == 0:
            p_home_est = (wins_home_total + 0.5 * ties_total) / sims_done  # ✅ Usa acumulado
            se_home = math.sqrt(max(p_home_est * (1 - p_home_est), 1e-9) / sims_done)
            
            if se_home < early_stop_se:
                logger.info(f"✅ Convergencia en {sims_done:,} sims (SE={se_home:.5f})")
                break
    
    # ✅ CALCULAR ESTADÍSTICAS FINALES
    if store_samples:
        final_home = np.concatenate(all_home)
        final_away = np.concatenate(all_away)
        final_total = np.concatenate(all_total)
    else:
        # ✅ REGENERAR todas las simulaciones si no guardamos
        logger.debug("Regenerando muestras para estadísticas finales...")
        lh_noise = np.clip(
            rng.normal(lh, lambda_noise * max(lh, 0.5), size=sims_done),
            LIMITS.MIN_LAMBDA, LIMITS.MAX_LAMBDA
        )
        la_noise = np.clip(
            rng.normal(la, lambda_noise * max(la, 0.5), size=sims_done),
            LIMITS.MIN_LAMBDA, LIMITS.MAX_LAMBDA
        )
        final_home = rng.poisson(lh_noise)
        final_away = rng.poisson(la_noise)
        final_total = final_home + final_away
    
    # Estadísticas
    p_home = (wins_home_total + 0.5 * ties_total) / sims_done
    p_away = (wins_away_total + 0.5 * ties_total) / sims_done
    mean_home = float(np.mean(final_home))
    mean_away = float(np.mean(final_away))
    mean_total = float(np.mean(final_total))
    std_total = float(np.std(final_total))
    
    # ✅ PERCENTILES CON KEYS CORRECTAS
    percentiles = {
        f"p{p}": float(np.percentile(final_total, p))
        for p in [10, 25, 50, 75, 90]
    }
    
    results = {
        "n": sims_done,
        "p_home": float(p_home),
        "p_away": float(p_away),
        "mean_home": mean_home,
        "mean_away": mean_away,
        "mean_total": mean_total,
        "std_total": std_total,
        "percentiles": percentiles,
        "converged_early": sims_done < n_max,
    }
    
    # Totales O/U
    if total_line:
        over = int(np.sum(final_total > total_line))
        under = int(np.sum(final_total < total_line))
        push = int(np.sum(final_total == total_line))
        
        results.update({
            "p_over": float(over / len(final_total)),
            "p_under": float(under / len(final_total)),
            "p_push": float(push / len(final_total)),
            "total_line": float(total_line),
        })
    
    logger.info(f"✅ Completado: {sims_done:,} sims | P(Home)={p_home:.3f}")
    return results

def monte_carlo_simple(lh: float, la: float, total_line: float = 8.5):
    return monte_carlo_advanced(
        lh, la,
        n_max=200_000,
        block=50_000,
        total_line=total_line,
        rng_seed=42,
        store_samples=True  # ✅ Para evitar regeneración
    )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    print("🧪 TEST: Monte Carlo G9 PRO ULTRA v2.0 FIXED")
    result = monte_carlo_advanced(
        lh=4.3, la=3.9,
        total_line=8.0,
        n_max=1_000_000,
        store_samples=True  # Para test
    )
    
    print(f"\n📊 RESULTADOS:")
    print(f"  Sims: {result['n']:,}")
    print(f"  P(Home): {result['p_home']:.3f}")
    print(f"  P(Away): {result['p_away']:.3f}")
    print(f"  P(Over): {result.get('p_over', 'N/A'):.3f}")
    print(f"  Media Total: {result['mean_total']:.2f} ± {result['std_total']:.2f}")
    print(f"  Percentiles: {result['percentiles']}")
    print(f"  Convergió temprano: {result['converged_early']}")
