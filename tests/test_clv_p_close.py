"""Tests for track_record/clv.py — the shared p_close / EV-vs-close
computation docs/PROTOCOLO_CLV_V1.md's primary metric depends on.

Verifies: (1) the primary path devigs Pinnacle's own two-sided close via
core.value_detector.remove_vig_multiplicative (no reimplementation), (2) the
pre-registered SECONDARY fallback (median devigged price across books) only
activates when Pinnacle's close is unavailable, and is correctly labeled so
a caller never mistakes it for the primary, (3) a pick with no usable close
at all is attrition (None), never a fabricated value.
"""
from __future__ import annotations

import json

import pytest

from core.value_detector import remove_vig_multiplicative
from track_record.clv import compute_ev_vs_close, compute_p_close


def _pin_pick(market="ML_HOME", pin_home=1.80, pin_away=2.05, odds_decimal=None,
              all_books=None):
    return {
        "market": market,
        "closing_pin_home": pin_home,
        "closing_pin_away": pin_away,
        "closing_all_books_json": json.dumps(all_books) if all_books else None,
        "odds_decimal": odds_decimal,
    }


def test_primary_path_devigs_pinnacle_close():
    pick = _pin_pick(market="ML_HOME", pin_home=1.80, pin_away=2.05)
    expected_fair_home = remove_vig_multiplicative([1.80, 2.05])[0]

    result = compute_p_close(pick)

    assert result["source"] == "pinnacle"
    assert result["p_close"] == pytest.approx(expected_fair_home)


def test_primary_path_picks_away_side_for_ml_away_pick():
    pick = _pin_pick(market="ML_AWAY", pin_home=1.80, pin_away=2.05)
    expected_fair_away = remove_vig_multiplicative([1.80, 2.05])[1]

    result = compute_p_close(pick)

    assert result["source"] == "pinnacle"
    assert result["p_close"] == pytest.approx(expected_fair_away)


def test_fallback_median_used_only_when_pinnacle_missing():
    all_books = [
        {"book": "A", "home": 1.80, "away": 2.05},
        {"book": "B", "home": 1.90, "away": 1.95},
    ]
    pick = _pin_pick(market="ML_HOME", pin_home=None, pin_away=None, all_books=all_books)

    fair_a = remove_vig_multiplicative([1.80, 2.05])[0]
    fair_b = remove_vig_multiplicative([1.90, 1.95])[0]
    expected_median = (fair_a + fair_b) / 2.0

    result = compute_p_close(pick)

    assert result["source"] == "fallback_median"
    assert result["n_books"] == 2
    assert result["p_close"] == pytest.approx(expected_median)


def test_pinnacle_present_never_falls_through_to_median():
    all_books = [{"book": "A", "home": 5.0, "away": 1.05}]  # wildly different
    pick = _pin_pick(market="ML_HOME", pin_home=1.80, pin_away=2.05, all_books=all_books)

    result = compute_p_close(pick)

    assert result["source"] == "pinnacle"
    assert result["p_close"] == pytest.approx(remove_vig_multiplicative([1.80, 2.05])[0])


def test_one_sided_book_in_fallback_pool_is_ignored():
    all_books = [
        {"book": "A", "home": 1.80, "away": 2.05},
        {"book": "OneSided", "home": 1.99, "away": None},
    ]
    pick = _pin_pick(market="ML_HOME", pin_home=None, pin_away=None, all_books=all_books)

    result = compute_p_close(pick)

    assert result["n_books"] == 1
    assert result["p_close"] == pytest.approx(remove_vig_multiplicative([1.80, 2.05])[0])


def test_no_usable_close_is_attrition_not_a_guess():
    pick = _pin_pick(market="ML_HOME", pin_home=None, pin_away=None, all_books=None)
    assert compute_p_close(pick) is None


def test_non_moneyline_market_is_out_of_scope():
    pick = _pin_pick(market="OVER", pin_home=1.80, pin_away=2.05)
    assert compute_p_close(pick) is None


def test_ev_vs_close_matches_protocol_formula():
    pick = _pin_pick(market="ML_HOME", pin_home=1.80, pin_away=2.05, odds_decimal=1.95)
    p_close = remove_vig_multiplicative([1.80, 2.05])[0]
    expected_ev_pct = (p_close * 1.95 - 1.0) * 100.0

    result = compute_ev_vs_close(pick)

    assert result["ev_vs_close_pct"] == pytest.approx(expected_ev_pct)
    assert result["source"] == "pinnacle"


def test_ev_vs_close_is_attrition_without_odds_decimal():
    pick = _pin_pick(market="ML_HOME", pin_home=1.80, pin_away=2.05, odds_decimal=None)
    assert compute_ev_vs_close(pick) is None


def test_ev_vs_close_is_attrition_without_usable_close():
    pick = _pin_pick(market="ML_HOME", pin_home=None, pin_away=None, odds_decimal=1.95)
    assert compute_ev_vs_close(pick) is None
