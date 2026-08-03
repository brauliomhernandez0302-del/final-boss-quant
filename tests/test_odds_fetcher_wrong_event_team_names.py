"""Regression test for a real bug in get_best_odds_for_teams(): the winning
event's h2h/spreads outcome matching and the returned home_team/away_team
used g_home/g_away — variables left over from the SEARCH loop over every
raw event, holding whichever MLB game was last examined, not the actual
matched event's own teams.

Found 2026-07-21 while building the React matchup dashboard: a live call for
"New York Yankees" vs "Pittsburgh Pirates" returned ml_home=None, ml_away=None
and home_team="Seattle Mariners"/away_team="Cincinnati Reds" — a completely
different game — even though real Yankees/Pirates moneyline odds existed in
the same raw event list. Every existing fixture in
tests/test_odds_fetcher_commence_time_match.py uses the SAME two teams for
every event in its mocked raw list, so g_home/g_away's stale value always
happened to equal the correct one — masking this until a real multi-game
schedule (the actual `_get_raw_events()` shape in production, which returns
every MLB game for the day together) was fetched.
"""
from __future__ import annotations

import odds_fetcher


def _event(event_id: str, home: str, away: str, commence_time: str,
           home_price: float, away_price: float):
    return {
        "id": event_id,
        "sport_key": "baseball_mlb",
        "home_team": home,
        "away_team": away,
        "commence_time": commence_time,
        "bookmakers": [
            {
                "key": "pinnaclesports",
                "title": "Pinnacle",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": home, "price": home_price},
                        {"name": away, "price": away_price},
                    ]},
                ],
            }
        ],
    }


def test_moneyline_matches_the_target_game_not_the_last_event_in_the_list(monkeypatch):
    # Real production shape: _get_raw_events() returns EVERY MLB game for the
    # day together, not just the two teams being queried. The target game
    # (Yankees/Pirates) is placed BEFORE an unrelated game in the list, so a
    # stale "last event examined" variable would leak that unrelated game's
    # teams instead.
    events = [
        _event("evt_target", "New York Yankees", "Pittsburgh Pirates",
               "2026-07-21T23:06:00Z", 1.81, 2.22),
        _event("evt_other", "Seattle Mariners", "Cincinnati Reds",
               "2026-07-21T23:10:00Z", 1.90, 1.95),
    ]
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: events)

    result = odds_fetcher.get_best_odds_for_teams(
        home_team="New York Yankees", away_team="Pittsburgh Pirates",
        commence_time="2026-07-21T23:05:00Z", sport="baseball_mlb",
    )

    assert result["game_id"] == "evt_target"
    assert result["home_team"] == "New York Yankees"
    assert result["away_team"] == "Pittsburgh Pirates"
    assert result["ml_home"] == 1.81
    assert result["ml_away"] == 2.22
    assert result["pin_home"] == 1.81
    assert result["pin_away"] == 2.22


def test_moneyline_matches_the_target_game_when_it_is_last_in_the_list(monkeypatch):
    # Same scenario, order reversed — the unrelated game is examined FIRST,
    # so a stale-variable bug would (partially) self-mask here. Both
    # orderings must pass for the fix to be real, not order-dependent luck.
    events = [
        _event("evt_other", "Seattle Mariners", "Cincinnati Reds",
               "2026-07-21T23:10:00Z", 1.90, 1.95),
        _event("evt_target", "New York Yankees", "Pittsburgh Pirates",
               "2026-07-21T23:06:00Z", 1.81, 2.22),
    ]
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: events)

    result = odds_fetcher.get_best_odds_for_teams(
        home_team="New York Yankees", away_team="Pittsburgh Pirates",
        commence_time="2026-07-21T23:05:00Z", sport="baseball_mlb",
    )

    assert result["game_id"] == "evt_target"
    assert result["home_team"] == "New York Yankees"
    assert result["away_team"] == "Pittsburgh Pirates"
    assert result["ml_home"] == 1.81
    assert result["ml_away"] == 2.22
