"""
Regression test for the F5 key-naming mismatch found 2026-07-06:
get_best_odds_for_teams() used to return "f5_home"/"f5_away"/"f5_over"/
"f5_under", but run_module.py's GameOdds construction reads
"f5_ml_home"/"f5_ml_away"/"f5_total_over"/"f5_total_under" — a silent
mismatch that meant F5 markets could never activate in production
whenever run_module() relies on this function for real odds (no explicit
market_odds override), since only "f5_total_line" happened to match by
coincidence.

This is a pure dict-key-naming issue, not a data-correctness/API-behavior
one, so it's fully testable with a synthetic payload — no live
ODDS_API_KEY required.
"""

import odds_fetcher
from core.value_detector import GameOdds


def _synthetic_event():
    """One event shaped like a real Odds API response, with h1 (F5) markets."""
    return {
        "id": "evt1",
        "sport_key": "baseball_mlb",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "commence_time": "2026-07-06T23:00:00Z",
        "bookmakers": [
            {
                "key": "pinnaclesports",
                "title": "Pinnacle",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "New York Yankees", "price": 1.80},
                        {"name": "Boston Red Sox", "price": 2.05},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "price": 1.91, "point": 8.5},
                        {"name": "Under", "price": 1.91, "point": 8.5},
                    ]},
                    {"key": "spreads", "outcomes": [
                        {"name": "New York Yankees", "price": 1.95},
                        {"name": "Boston Red Sox", "price": 1.88},
                    ]},
                    {"key": "h2h_h1", "outcomes": [
                        {"name": "New York Yankees", "price": 1.75},
                        {"name": "Boston Red Sox", "price": 2.15},
                    ]},
                    {"key": "totals_h1", "outcomes": [
                        {"name": "Over", "price": 1.90, "point": 4.5},
                        {"name": "Under", "price": 1.92, "point": 4.5},
                    ]},
                    {"key": "spreads_h1", "outcomes": [
                        {"name": "New York Yankees", "price": 1.95},
                        {"name": "Boston Red Sox", "price": 1.87},
                    ]},
                ],
            }
        ],
    }


def test_get_best_odds_for_teams_returns_gameodds_compatible_f5_keys(monkeypatch):
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: [_synthetic_event()])

    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox", sport="baseball_mlb"
    )

    # The exact keys GameOdds/run_module.py read (core/value_detector.py's
    # GameOdds field names, confirmed as the established convention via its
    # own self-test data).
    assert result["f5_ml_home"] == 1.75
    assert result["f5_ml_away"] == 2.15
    assert result["f5_total_line"] == 4.5
    assert result["f5_total_over"] == 1.90
    assert result["f5_total_under"] == 1.92

    # The old, mismatched key names must not resurface.
    assert "f5_home" not in result
    assert "f5_away" not in result
    assert "f5_over" not in result
    assert "f5_under" not in result


def test_f5_odds_activate_analyze_first5_gate(monkeypatch):
    """End-to-end shape check: feed get_best_odds_for_teams()'s real output
    into GameOdds exactly as run_module.py does, and confirm F5 activates."""
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: [_synthetic_event()])
    fetched = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox", sport="baseball_mlb"
    )

    odds = GameOdds(
        ml_home=fetched["ml_home"],
        ml_away=fetched["ml_away"],
        total_line=fetched["total_line"],
        total_over=fetched["total_over"],
        total_under=fetched["total_under"],
        f5_ml_home=fetched.get("f5_ml_home"),
        f5_ml_away=fetched.get("f5_ml_away"),
        f5_total_line=fetched.get("f5_total_line"),
        f5_total_over=fetched.get("f5_total_over"),
        f5_total_under=fetched.get("f5_total_under"),
    )

    # Mirrors value_detector.py's analyze_first5 gating condition exactly.
    assert odds.f5_ml_home and odds.f5_ml_away
    assert odds.f5_total_line and odds.f5_total_over and odds.f5_total_under
