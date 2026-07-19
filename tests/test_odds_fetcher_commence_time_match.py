"""Regression tests for get_best_odds_for_teams()'s commence_time
disambiguation (Fase 2A commit 2, V2).

Found during the verification sweep (audit_20260714/verificacion_operativa/
reporte.md, V2): the function matched odds events by team name ALONE,
returning on the first match with zero date/commence_time check. For any
multi-game series between the same two teams (the MLB norm — series are
3-4 games), or a doubleheader, this could silently return the wrong day's
game's odds. Fixed by requiring the caller to supply the target game's
commence_time and matching within a +/-6h window, refusing to guess (None)
on no match or an exact tie.
"""
from __future__ import annotations

import odds_fetcher


def _event(event_id: str, commence_time: str, home_price: float, away_price: float):
    return {
        "id": event_id,
        "sport_key": "baseball_mlb",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "commence_time": commence_time,
        "bookmakers": [
            {
                "key": "pinnaclesports",
                "title": "Pinnacle",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "New York Yankees", "price": home_price},
                        {"name": "Boston Red Sox", "price": away_price},
                    ]},
                ],
            }
        ],
    }


def test_happy_path_single_game_still_works(monkeypatch):
    monkeypatch.setattr(
        odds_fetcher, "_get_raw_events",
        lambda: [_event("evt1", "2026-07-19T23:00:00Z", 1.80, 2.05)],
    )
    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox",
        commence_time="2026-07-19T23:00:00Z", sport="baseball_mlb",
    )
    assert result["ml_home"] == 1.80
    assert result["ml_away"] == 2.05
    assert result["game_id"] == "evt1"


def test_multi_day_series_picks_the_correct_days_game(monkeypatch):
    # Same two teams, 3-game series on consecutive days — distinct prices
    # per day so a wrong pick is unmistakable.
    events = [
        _event("evt_day1", "2026-07-18T23:00:00Z", 1.70, 2.20),
        _event("evt_day2", "2026-07-19T23:00:00Z", 1.80, 2.05),
        _event("evt_day3", "2026-07-20T23:00:00Z", 1.95, 1.90),
    ]
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: events)

    # Asking for day 2's game must return day 2's odds, not day 1's or day 3's.
    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox",
        commence_time="2026-07-19T23:00:00Z", sport="baseball_mlb",
    )
    assert result["game_id"] == "evt_day2"
    assert result["ml_home"] == 1.80
    assert result["ml_away"] == 2.05


def test_doubleheader_closest_commence_wins(monkeypatch):
    # Two games same day, same two teams, a few hours apart — well within
    # +/-6h of each other, so distance-to-target is what must disambiguate.
    events = [
        _event("evt_game1", "2026-07-19T17:00:00Z", 1.75, 2.10),
        _event("evt_game2", "2026-07-19T22:30:00Z", 1.85, 2.00),
    ]
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: events)

    # Target close to game 2's start (22:30Z) — within its own window, and
    # far enough from game 1 (17:00Z, 5.5h away) that only game 2 should win.
    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox",
        commence_time="2026-07-19T22:15:00Z", sport="baseball_mlb",
    )
    assert result["game_id"] == "evt_game2"


def test_no_event_within_window_returns_empty_not_a_guess(monkeypatch):
    # Only a game 3 days away — well outside the +/-6h window.
    monkeypatch.setattr(
        odds_fetcher, "_get_raw_events",
        lambda: [_event("evt_far", "2026-07-22T23:00:00Z", 1.80, 2.05)],
    )
    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox",
        commence_time="2026-07-19T23:00:00Z", sport="baseball_mlb",
    )
    assert result == {}


def test_exact_tie_in_distance_returns_empty_not_a_guess(monkeypatch):
    # Two candidate events exactly equidistant from the target — genuinely
    # ambiguous, must refuse to pick either rather than arbitrarily pick one.
    events = [
        _event("evt_before", "2026-07-19T20:00:00Z", 1.70, 2.20),
        _event("evt_after", "2026-07-20T02:00:00Z", 1.95, 1.90),
    ]
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: events)

    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox",
        commence_time="2026-07-19T23:00:00Z", sport="baseball_mlb",
    )
    assert result == {}


def test_unparseable_commence_time_returns_empty(monkeypatch):
    monkeypatch.setattr(
        odds_fetcher, "_get_raw_events",
        lambda: [_event("evt1", "2026-07-19T23:00:00Z", 1.80, 2.05)],
    )
    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox",
        commence_time="not-a-real-timestamp", sport="baseball_mlb",
    )
    assert result == {}


def test_event_missing_commence_time_is_ignored_as_a_candidate(monkeypatch):
    events = [
        {**_event("evt_no_commence", "2026-07-19T23:00:00Z", 1.99, 1.99), "commence_time": ""},
        _event("evt_real", "2026-07-19T22:00:00Z", 1.80, 2.05),
    ]
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: events)

    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox",
        commence_time="2026-07-19T23:00:00Z", sport="baseball_mlb",
    )
    assert result["game_id"] == "evt_real"
