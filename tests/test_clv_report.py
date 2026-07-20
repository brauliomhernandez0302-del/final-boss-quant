"""Tests for scripts/clv_report.py — docs/PROTOCOLO_CLV_V1.md's evaluation
instrument.

Verifies: D0 is read from the protocol document itself (never hardcoded),
shakedown/primary split by D0, attrition counting, the fallback-median path
never contaminating the primary metric, reproducible bootstrap CI with a
fixed seed, and — critically — that build_report() never emits a verdict
field (the script is the instrument, not the judge).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.value_detector import remove_vig_multiplicative
from scripts.clv_report import bootstrap_mean_ci, build_report, read_d0


def _pick(
    game_date="2026-08-01", market="ML_HOME", odds_decimal=1.95,
    pin_home=1.80, pin_away=2.05, all_books_json=None,
    ev_pct=6.0, minutes_before_start=45.0, odds_book=None,
    result=None, profit_loss_units=None, model_prob=0.55,
):
    return {
        "game_date": game_date, "market": market, "odds_decimal": odds_decimal,
        "closing_pin_home": pin_home, "closing_pin_away": pin_away,
        "closing_all_books_json": all_books_json,
        "ev_pct": ev_pct, "minutes_before_start": minutes_before_start,
        "odds_book": odds_book, "result": result,
        "profit_loss_units": profit_loss_units, "model_prob": model_prob,
    }


# --------------------------------------------------------------------- D0

def test_read_d0_parses_a_real_date(tmp_path):
    doc = tmp_path / "protocol.md"
    doc.write_text("## Registro\n- D0: 2026-08-01\n- Clarificaciones: (ninguna)\n")
    assert read_d0(doc) == "2026-08-01"


def test_read_d0_pending_returns_none(tmp_path):
    doc = tmp_path / "protocol.md"
    doc.write_text("## Registro\n- D0: pendiente — completar cuando...\n")
    assert read_d0(doc) is None


def test_read_d0_missing_line_returns_none(tmp_path):
    doc = tmp_path / "protocol.md"
    doc.write_text("## Registro\n- Clarificaciones: (ninguna)\n")
    assert read_d0(doc) is None


# --------------------------------------------------------------- bootstrap

def test_bootstrap_is_reproducible_with_fixed_seed():
    values = [1.0, 2.0, -0.5, 3.0, 0.2, -1.1, 4.0]
    r1 = bootstrap_mean_ci(values, n_resamples=500, seed=42)
    r2 = bootstrap_mean_ci(values, n_resamples=500, seed=42)
    assert r1 == r2


def test_bootstrap_ci_contains_the_sample_mean():
    values = [1.0, 2.0, -0.5, 3.0, 0.2, -1.1, 4.0]
    r = bootstrap_mean_ci(values, n_resamples=2000, seed=1)
    assert r["ci_lo"] <= r["mean"] <= r["ci_hi"]


def test_bootstrap_empty_values_returns_none():
    assert bootstrap_mean_ci([]) is None


# ------------------------------------------------------------- build_report

def test_no_d0_means_everything_is_shakedown_no_primary():
    picks = [_pick(game_date="2026-07-01"), _pick(game_date="2026-07-02")]
    report = build_report(picks, d0=None)
    assert report["primary"] is None
    assert report["shakedown"]["n_picks"] == 2


def test_picks_split_correctly_by_d0():
    picks = [
        _pick(game_date="2026-07-25"),  # before D0 -> shakedown
        _pick(game_date="2026-08-01"),  # == D0 -> primary
        _pick(game_date="2026-08-05"),  # after D0 -> primary
    ]
    report = build_report(picks, d0="2026-08-01")
    assert report["shakedown"]["n_picks"] == 1
    assert report["primary"]["n_picks"] == 2


def test_known_ev_vs_close_matches_hand_computed_value():
    pin_home, pin_away, odds_decimal = 1.80, 2.05, 1.95
    p_close = remove_vig_multiplicative([pin_home, pin_away])[0]
    expected_mean = (p_close * odds_decimal - 1.0) * 100.0

    picks = [_pick(game_date="2026-08-01", pin_home=pin_home, pin_away=pin_away,
                    odds_decimal=odds_decimal)]
    report = build_report(picks, d0="2026-08-01")

    pm = report["primary"]["primary_metric_mean_ev_vs_close_pct"]
    assert pm["mean"] == pytest.approx(expected_mean)
    assert pm["n"] == 1


def test_attrition_pick_is_excluded_from_primary_metric_but_counted():
    good = _pick(game_date="2026-08-01")
    no_close = _pick(game_date="2026-08-02", pin_home=None, pin_away=None,
                      all_books_json=None)
    report = build_report([good, no_close], d0="2026-08-01")

    primary = report["primary"]
    assert primary["n_picks"] == 2
    assert primary["n_attrition"] == 1
    assert primary["attrition_pct"] == 50.0
    assert primary["primary_metric_mean_ev_vs_close_pct"]["n"] == 1


def test_fallback_median_pick_never_enters_primary_metric():
    all_books = [{"book": "A", "home": 1.80, "away": 2.05}]
    fallback_pick = _pick(game_date="2026-08-01", pin_home=None, pin_away=None,
                           all_books_json=json.dumps(all_books))
    report = build_report([fallback_pick], d0="2026-08-01")

    primary = report["primary"]
    assert primary["n_attrition"] == 0
    assert primary["n_fallback_median_only"] == 1
    assert primary["n_primary_eligible"] == 0
    assert primary["primary_metric_mean_ev_vs_close_pct"] is None


def test_edge_bucket_cut_groups_correctly():
    picks = [
        _pick(game_date="2026-08-01", ev_pct=2.0),
        _pick(game_date="2026-08-01", ev_pct=7.0),
        _pick(game_date="2026-08-01", ev_pct=15.0),
    ]
    report = build_report(picks, d0="2026-08-01")
    cuts = report["primary"]["cuts"]["a_edge_bucket"]
    assert cuts["<5%"]["n"] == 1
    assert cuts["5-10%"]["n"] == 1
    assert cuts[">10%"]["n"] == 1


def test_staleness_bucket_cut_groups_correctly():
    picks = [
        _pick(game_date="2026-08-01", minutes_before_start=30.0),
        _pick(game_date="2026-08-01", minutes_before_start=90.0),
    ]
    report = build_report(picks, d0="2026-08-01")
    cuts = report["primary"]["cuts"]["b_staleness_bucket"]
    assert cuts["<60min"]["n"] == 1
    assert cuts[">=60min"]["n"] == 1


def test_book_cut_groups_by_odds_book_including_unknown():
    picks = [
        _pick(game_date="2026-08-01", odds_book="Pinnacle"),
        _pick(game_date="2026-08-01", odds_book=None),
    ]
    report = build_report(picks, d0="2026-08-01")
    cuts = report["primary"]["cuts"]["c_book"]
    assert cuts["Pinnacle"]["n"] == 1
    assert cuts["unknown"]["n"] == 1


def test_market_cut_only_appears_with_more_than_one_market():
    only_home = [_pick(game_date="2026-08-01", market="ML_HOME")]
    report_single = build_report(only_home, d0="2026-08-01")
    assert "d_market" not in report_single["primary"]["cuts"]

    both = [
        _pick(game_date="2026-08-01", market="ML_HOME"),
        _pick(game_date="2026-08-01", market="ML_AWAY", pin_home=1.80, pin_away=2.05,
              odds_decimal=2.00),
    ]
    report_both = build_report(both, d0="2026-08-01")
    assert "d_market" in report_both["primary"]["cuts"]


def test_roi_units_computed_over_all_resolved_regardless_of_close():
    no_close_but_resolved = _pick(
        game_date="2026-08-01", pin_home=None, pin_away=None,
        result="WIN", profit_loss_units=0.9,
    )
    report = build_report([no_close_but_resolved], d0="2026-08-01")
    secondary = report["primary"]["secondary"]
    assert secondary["roi_units_total"] == pytest.approx(0.9)
    assert secondary["roi_units_n_resolved"] == 1


def test_model_brier_vs_close_brier_only_uses_resolved_primary_eligible():
    pick = _pick(game_date="2026-08-01", result="WIN", model_prob=0.60)
    report = build_report([pick], d0="2026-08-01")
    secondary = report["primary"]["secondary"]
    p_close = remove_vig_multiplicative([1.80, 2.05])[0]
    assert secondary["model_brier"] == pytest.approx((0.60 - 1) ** 2, abs=1e-5)
    assert secondary["p_close_brier"] == pytest.approx((p_close - 1) ** 2, abs=1e-5)


def test_beat_close_rate_over_primary_eligible():
    winner = _pick(game_date="2026-08-01", odds_decimal=2.50)  # beats close
    loser = _pick(game_date="2026-08-01", odds_decimal=1.50)   # worse than close
    report = build_report([winner, loser], d0="2026-08-01")
    assert report["primary"]["secondary"]["beat_close_rate_pct"] == 50.0


def test_staleness_distribution_reported_for_closes_used():
    picks = [
        _pick(game_date="2026-08-01", minutes_before_start=30.0),
        _pick(game_date="2026-08-01", minutes_before_start=90.0),
    ]
    report = build_report(picks, d0="2026-08-01")
    dist = report["primary"]["staleness_distribution"]
    assert dist["n"] == 2
    assert dist["min_minutes"] == 30.0
    assert dist["max_minutes"] == 90.0


def test_report_never_emits_a_verdict_field():
    picks = [_pick(game_date="2026-08-01")]
    report = build_report(picks, d0="2026-08-01")

    forbidden = {"verdict", "veredicto", "decision", "decisión", "hay_edge", "no_edge"}
    assert forbidden.isdisjoint(report.keys())
    assert forbidden.isdisjoint(report["primary"].keys())
    # The thresholds are shown for reference only, clearly labeled as such —
    # not a computed decision.
    assert "thresholds_for_reference_only_not_a_verdict" in report
