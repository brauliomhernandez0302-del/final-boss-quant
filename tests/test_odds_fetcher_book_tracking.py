"""Regression tests for get_best_odds_for_teams()'s per-book tracking
(docs/PROTOCOLO_CLV_V1.md CLV-C2): which bookmaker gave the winning
ml_home/ml_away price (cut (c), "por libro de O_taken"), and every book's
own two-sided h2h quote (the pre-registered fallback: median devigged price
across books when no Pinnacle close is available for a pick).
"""
from __future__ import annotations

import odds_fetcher


def _event_multi_book(commence_time: str, books: list[dict]):
    return {
        "id": "evt1",
        "sport_key": "baseball_mlb",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
        "commence_time": commence_time,
        "bookmakers": [
            {
                "key": b["key"], "title": b["title"],
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "New York Yankees", "price": b["home"]},
                        {"name": "Boston Red Sox", "price": b["away"]},
                    ]},
                ],
            }
            for b in books
        ],
    }


def test_winning_price_records_its_own_book(monkeypatch):
    events = [_event_multi_book("2026-07-19T23:00:00Z", [
        {"key": "pinnaclesports", "title": "Pinnacle", "home": 1.80, "away": 2.05},
        {"key": "draftkings", "title": "DraftKings", "home": 1.91, "away": 1.95},
    ])]
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: events)

    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox",
        commence_time="2026-07-19T23:00:00Z", sport="baseball_mlb",
    )
    assert result["ml_home"] == 1.91
    assert result["ml_home_book"] == "DraftKings"
    assert result["ml_away"] == 2.05
    assert result["ml_away_book"] == "Pinnacle"


def test_all_books_h2h_includes_every_two_sided_quote(monkeypatch):
    events = [_event_multi_book("2026-07-19T23:00:00Z", [
        {"key": "pinnaclesports", "title": "Pinnacle", "home": 1.80, "away": 2.05},
        {"key": "draftkings", "title": "DraftKings", "home": 1.91, "away": 1.95},
        {"key": "fanduel", "title": "FanDuel", "home": 1.85, "away": 2.00},
    ])]
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: events)

    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox",
        commence_time="2026-07-19T23:00:00Z", sport="baseball_mlb",
    )
    books = {b["book"]: b for b in result["all_books_h2h"]}
    assert len(books) == 3
    assert books["Pinnacle"] == {"book": "Pinnacle", "home": 1.80, "away": 2.05}
    assert books["DraftKings"] == {"book": "DraftKings", "home": 1.91, "away": 1.95}
    assert books["FanDuel"] == {"book": "FanDuel", "home": 1.85, "away": 2.00}


def test_one_sided_book_excluded_from_all_books_h2h(monkeypatch):
    # A bookmaker offering only one side of the h2h market (or none at all)
    # must not contribute a fabricated pair to the fallback pool.
    event = _event_multi_book("2026-07-19T23:00:00Z", [
        {"key": "pinnaclesports", "title": "Pinnacle", "home": 1.80, "away": 2.05},
    ])
    # Manually strip one outcome from a second bookmaker to simulate a
    # partial/one-sided quote.
    event["bookmakers"].append({
        "key": "onesided", "title": "OneSided",
        "markets": [{"key": "h2h", "outcomes": [
            {"name": "New York Yankees", "price": 1.99},
        ]}],
    })
    monkeypatch.setattr(odds_fetcher, "_get_raw_events", lambda: [event])

    result = odds_fetcher.get_best_odds_for_teams(
        home_team="Yankees", away_team="Red Sox",
        commence_time="2026-07-19T23:00:00Z", sport="baseball_mlb",
    )
    books = [b["book"] for b in result["all_books_h2h"]]
    assert "OneSided" not in books
    assert "Pinnacle" in books


def test_no_h2h_market_gives_empty_book_fields(monkeypatch):
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
    assert result["ml_home_book"] is None
    assert result["ml_away_book"] is None
    assert result["all_books_h2h"] == []
