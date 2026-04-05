# ==========================================================
#  HELPERS - Utilidades generales para MLB Module
# ==========================================================

import math
import numpy as np

def clamp(x: float, a: float, b: float) -> float:
    """Limita un valor dentro de un rango [a, b]."""
    return max(a, min(b, x))

def infer_decimal(odds: float) -> float:
    """Convierte odds American ↔ Decimal automáticamente."""
    if odds is None or not np.isfinite(odds):
        return np.nan
    if 1.20 <= odds < 100.0:
        return float(odds)
    if odds <= -100:
        return 1.0 + (100.0 / abs(odds))
    if odds >= 100:
        return 1.0 + (odds / 100.0)
    return np.nan

def prob_from_decimal(dec: float) -> float:
    """Convierte cuota decimal en probabilidad implícita."""
    if dec is None or not np.isfinite(dec) or dec <= 1.0:
        return np.nan
    return 1.0 / dec
