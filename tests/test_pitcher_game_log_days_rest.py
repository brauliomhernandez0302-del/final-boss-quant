from datetime import datetime

import pytest

import data_fetchers
from data_fetchers import MLBStatsAPI


class _FakeResponse:
    def __init__(self, splits):
        self._splits = splits

    def raise_for_status(self):
        return None

    def json(self):
        return {"stats": [{"splits": self._splits}]}


def _split(date, earned_runs=2, innings=6.0, pitches=90):
    return {
        "date": date,
        "stat": {
            "gamesStarted": 1,
            "earnedRuns": earned_runs,
            "inningsPitched": innings,
            "numberOfPitches": pitches,
            "strikeOuts": 6,
            "baseOnBalls": 2,
        },
    }


def _api_with_splits(monkeypatch, tmp_path, splits):
    monkeypatch.setattr(data_fetchers, "CACHE_DIR", tmp_path)
    api = MLBStatsAPI()
    monkeypatch.setattr(api.session, "get", lambda *args, **kwargs: _FakeResponse(splits))
    return api


def test_historical_pitcher_days_rest_uses_supplied_game_date(monkeypatch, tmp_path):
    api = _api_with_splits(
        monkeypatch,
        tmp_path,
        [
            _split("2024-04-10", earned_runs=9),  # future relative to game date
            _split("2024-04-01", earned_runs=1),
        ],
    )

    result = api.get_pitcher_game_log(607192, 2024, as_of_date="2024-04-05")

    assert result["days_rest"] == 4
    assert result["starts_analyzed"] == 1
    assert result["last_pitch_count"] == 90


def test_live_pitcher_days_rest_falls_back_to_current_utc(monkeypatch, tmp_path):
    class FixedDateTime(datetime):
        @classmethod
        def utcnow(cls):
            return cls(2024, 4, 20)

    monkeypatch.setattr(data_fetchers, "datetime", FixedDateTime)
    api = _api_with_splits(
        monkeypatch,
        tmp_path,
        [
            _split("2024-04-15", earned_runs=2),
            _split("2024-04-01", earned_runs=1),
        ],
    )

    result = api.get_pitcher_game_log(607192, 2024)

    assert result["days_rest"] == 5
    assert result["starts_analyzed"] == 2


def test_backtest_days_rest_does_not_call_utcnow_when_game_date_supplied(monkeypatch, tmp_path):
    class NoUtcNowDateTime(datetime):
        @classmethod
        def utcnow(cls):
            raise AssertionError("datetime.utcnow() must not be used in historical mode")

    monkeypatch.setattr(data_fetchers, "datetime", NoUtcNowDateTime)
    api = _api_with_splits(
        monkeypatch,
        tmp_path,
        [
            _split("2024-04-10", earned_runs=9),
            _split("2024-04-01", earned_runs=1),
        ],
    )

    result = api.get_pitcher_game_log(607192, 2024, as_of_date="2024-04-05")

    assert result["days_rest"] == 4
