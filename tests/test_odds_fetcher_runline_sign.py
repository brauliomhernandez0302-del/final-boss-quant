"""Regression tests for get_best_odds_for_teams()'s signed run-line points
(runline_home_point/runline_away_point).

Found during a run-line/totals detection review (2026-07-20): the function
only ever returned the run-line MAGNITUDE (runline_line), discarding which
side was actually favored — every downstream consumer assumed "home is
always the favorite," which is false whenever the away team is favored.
These fields expose the real, signed point per side so track_record's
post-game grading (reconciler.py) can know which side was actually favored
for a given pick, instead of guessing.
"""
from __future__ import annotations

import odds_fetcher


def _spread_event(commence_time: str, home_point: float, away_point: float,
                   home_price: float = 1.90, away_price: float = 1.90):
    return {
        "id": "evt1",
        "sport_key": "baseball_mlb",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "commence_time": commence_time,
        "bookmakers": [
            {
                "key": "pinnaclesports", "title": "Pinnacle",
                "markets": [
                    {"key": "spreads", "outcomes": [
                        {"name": "New York Yankees", "price": home_price, "point": home_point},
                        {"name": "Boston Red Sox", "price": away_price, "point": away_point},
                    ]},
                ],
            }
        ],
    }


def test_home_favored_gives_negative_home_point_positive_away_point(monkeypatch):
    events = [_spread_event("2026-07-19T23:00:00Z", home_point=-1.5, away_point=1.5)]
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: events)

    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox",
        commence_time="2026-07-19T23:00:00Z", sport="baseball_mlb",
    )
    assert result["runline_home_point"] == -1.5
    assert result["runline_away_point"] == 1.5
    assert result["runline_line"] == 1.5  # magnitude, unaffected


def test_away_favored_gives_negative_away_point_positive_home_point(monkeypatch):
    # The scenario the old code silently mishandled: the ROAD team is the
    # run-line favorite.
    events = [_spread_event("2026-07-19T23:00:00Z", home_point=1.5, away_point=-1.5)]
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: events)

    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox",
        commence_time="2026-07-19T23:00:00Z", sport="baseball_mlb",
    )
    assert result["runline_home_point"] == 1.5
    assert result["runline_away_point"] == -1.5
    assert result["runline_line"] == 1.5  # magnitude still just 1.5


def test_no_spreads_market_gives_none_points(monkeypatch):
    event = {
        "id": "evt1", "sport_key": "baseball_mlb",
        "home_team": "New York Yankees", "away_team": "Boston Red Sox",
        "commence_time": "2026-07-19T23:00:00Z",
        "bookmakers": [{"key": "pinnaclesports", "title": "Pinnacle", "markets": []}],
    }
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: [event])

    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox",
        commence_time="2026-07-19T23:00:00Z", sport="baseball_mlb",
    )
    assert result["runline_home_point"] is None
    assert result["runline_away_point"] is None
    assert result["runline_line"] is None
