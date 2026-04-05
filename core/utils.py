from __future__ import annotations
import math
import time
import random
from dataclasses import dataclass

def implied_prob(american_odds: float) -> float | None:
    if american_odds is None or american_odds == 0:
        return None
    return (100 / (american_odds + 100)) if american_odds > 0 else (-american_odds / (-american_odds + 100))

def fair_to_american(p: float) -> int:
    p = max(1e-6, min(1 - 1e-6, p))  # keep within valid range (avoid division by zero)
    return int(round(100 * (1 - p) / p)) if p >= 0.5 else int(round(-100 * p / (1 - p)))

def kelly_fraction(p: float, odds_us: int, b: float = 1.0) -> float:
    q = 1 - p
    b_ret = odds_us / 100 if odds_us > 0 else 100 / (-odds_us)
    edge = (p * b_ret - q)
    f = edge / b_ret
    return max(0.0, min(b, f * b))  # clamp to [0, b]

def ts() -> int:
    return int(time.time())

def rng(seed: int | None = None) -> random.Random:
    return random.Random(seed if seed is not None else ts())

@dataclass
class Pick:
    market: str
    selection: str
    odds_us: int
    prob_model: float
    prob_implied: float
    edge: float

