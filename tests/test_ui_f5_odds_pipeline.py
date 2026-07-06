"""
Regression test for the F5 gap found 2026-07-06 in the Streamlit-selector-
driven analysis path: db/predictions_db.py::GameData had no F5 fields at
all, so ui/odds_loader.py::build_game_selector() never populated them and
ui/mlb.py::MLBAnalyzer.analyze()'s market_odds construction never read them
— independent of (and not fixed by) the get_best_odds_for_teams() naming
fix in test_odds_fetcher_f5_keys.py, since a human picking a game from the
dropdown never goes through that function at all.

Walks the full real chain: a synthetic get_odds_data()-shaped DataFrame row
-> build_game_selector() -> GameData -> the same market_odds dict
MLBAnalyzer.analyze() builds -> GameOdds -> analyze_first5()'s real gate.
"""

import pandas as pd

from ui.odds_loader import build_game_selector
from core.value_detector import GameOdds


def _synthetic_odds_df_row():
    """Shaped like one row of odds_fetcher.py::get_odds_data()'s output
    (i.e. after _normalize_event() — the third F5 naming scheme)."""
    return {
        "sport_key": "baseball_mlb",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "commence_time": "2026-07-06T23:00:00Z",
        "home_odds": 1.80,
        "away_odds": 2.05,
        "pin_home": 1.82,
        "pin_away": 2.02,
        "total_line": 8.5,
        "over_odds": 1.91,
        "under_odds": 1.91,
        "runline_home": 1.95,
        "runline_away": 1.88,
        "f5_home_odds": 1.75,
        "f5_away_odds": 2.15,
        "f5_total_line": 4.5,
        "f5_over_odds": 1.90,
        "f5_under_odds": 1.92,
    }


def test_build_game_selector_populates_f5_fields_on_gamedata():
    df = pd.DataFrame([_synthetic_odds_df_row()])

    options, mapping = build_game_selector(df)
    game_data = mapping[options[0]]

    assert game_data["f5_ml_home"] == 1.75
    assert game_data["f5_ml_away"] == 2.15
    assert game_data["f5_total_line"] == 4.5
    assert game_data["f5_total_over"] == 1.90
    assert game_data["f5_total_under"] == 1.92


def test_ui_selector_path_activates_analyze_first5_gate_end_to_end():
    """Mirrors MLBAnalyzer.analyze()'s market_odds construction exactly,
    then feeds it into GameOdds -> confirms analyze_first5()'s real gate."""
    df = pd.DataFrame([_synthetic_odds_df_row()])
    options, mapping = build_game_selector(df)
    game_data = mapping[options[0]]

    market_odds = {
        "ml_home":      game_data.get("home_odds"),
        "ml_away":      game_data.get("away_odds"),
        "pin_home":     game_data.get("pin_home"),
        "pin_away":     game_data.get("pin_away"),
        "total_line":   game_data.get("total_line"),
        "total_over":   game_data.get("total_over"),
        "total_under":  game_data.get("total_under"),
        "runline_home": game_data.get("runline_home"),
        "runline_away": game_data.get("runline_away"),
        "f5_ml_home":    game_data.get("f5_ml_home"),
        "f5_ml_away":    game_data.get("f5_ml_away"),
        "f5_total_line": game_data.get("f5_total_line"),
        "f5_total_over": game_data.get("f5_total_over"),
        "f5_total_under": game_data.get("f5_total_under"),
    }

    odds = GameOdds(
        ml_home=market_odds["ml_home"],
        ml_away=market_odds["ml_away"],
        total_line=market_odds["total_line"],
        total_over=market_odds["total_over"],
        total_under=market_odds["total_under"],
        f5_ml_home=market_odds["f5_ml_home"],
        f5_ml_away=market_odds["f5_ml_away"],
        f5_total_line=market_odds["f5_total_line"],
        f5_total_over=market_odds["f5_total_over"],
        f5_total_under=market_odds["f5_total_under"],
    )

    # Mirrors value_detector.py's analyze_first5 gating condition exactly.
    assert odds.f5_ml_home and odds.f5_ml_away
    assert odds.f5_total_line and odds.f5_total_over and odds.f5_total_under
