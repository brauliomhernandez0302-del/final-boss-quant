"""Anchors the EV unit convention across the app: calculate_ev() (and
everything downstream that compares against its output — publish
thresholds, tiers) works in PERCENTAGE POINTS, not a 0-1 fraction.

This is the test the verification sweep proposed but didn't implement
(audit_20260714/verificacion_operativa/reporte.md, V3.3), after mapping
every ev/ev_pct producer and consumer in the app and finding the display
bug fixed in track_record/ (Fase 2A precursor work) was the only place that
had gotten the units wrong — a Python "%" format specifier (`:.2%`) treats
its input as a fraction and multiplies by 100, but calculate_ev()'s own
docstring says "5.0 means +5% EV": already in percent, not a fraction.
Freezing this here so a future accidental unit flip (e.g. someone "fixing"
calculate_ev() to return a 0-1 fraction to match some other convention)
breaks CI immediately instead of silently reintroducing the same class of
100x-inflated-EV bug this phase spent real effort tracking down and fixing.
"""
import pytest

from config import DEFAULT_MIN_EV
from core.utils import calculate_ev
from core.value_detector import ValueTier


def test_calculate_ev_breakeven_is_zero_not_a_fraction():
    # prob * decimal_odds == 1.0 exactly -> breakeven, EV == 0.0 regardless
    # of which unit convention you'd assume — the real discriminating case
    # is the next one.
    assert calculate_ev(0.5, 2.0) == pytest.approx(0.0)


def test_calculate_ev_returns_percentage_points_not_a_fraction():
    # (0.6 * 2.0 - 1) = 0.2 as a fraction -> in PERCENTAGE POINTS that's 20.0,
    # not 0.2. If this ever returns 0.2, calculate_ev() silently flipped to
    # a 0-1 fraction convention and every consumer comparing against
    # percent-scale thresholds (DEFAULT_MIN_EV, ValueTier cutoffs) would be
    # instantly and silently miscalibrated by a factor of 100.
    assert calculate_ev(0.6, 2.0) == pytest.approx(20.0)
    assert calculate_ev(0.6, 2.0) != pytest.approx(0.2)


def test_min_ev_threshold_is_percentage_points():
    # config.py's own comment says "% minimum EV to flag a bet" — anchor the
    # actual value is on the same percent scale as calculate_ev()'s output,
    # not a 0-1 fraction (a value like 0.03 here would silently let through
    # almost every bet, since virtually all real EVs exceed 0.03 percentage
    # points).
    assert DEFAULT_MIN_EV >= 1.0, (
        "DEFAULT_MIN_EV looks like a 0-1 fraction, not percentage points — "
        "this would silently disable meaningful EV filtering"
    )


def test_value_tier_cutoffs_are_percentage_points():
    # ValueTier's own cutoffs (core/value_detector.py) must be on the same
    # percent scale as calculate_ev()'s output for tier assignment
    # (`ev >= ValueTier.X.value[1]`) to mean anything.
    for tier in (ValueTier.ULTRA, ValueTier.HIGH, ValueTier.MEDIUM):
        cutoff = tier.value[1]
        assert cutoff >= 1.0, (
            f"{tier.name} cutoff={cutoff} looks like a 0-1 fraction, not "
            "percentage points"
        )
