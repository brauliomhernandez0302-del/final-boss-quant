"""Shared math utilities — single source of truth for core formulas."""


def calculate_ev(prob: float, decimal_odds: float) -> float:
    """Expected value as a percentage.

    decimal_odds must include stake return (e.g. 1.91 for -110).
    Returns a percentage: 5.0 means +5 % EV.
    """
    return (prob * decimal_odds - 1) * 100
