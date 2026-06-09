import importlib
import sys

import pytest

import modules.baseball_module.advanced_pit_enrichment.savant_raw_ingestor as raw_ingestor
from modules.baseball_module.advanced_pit_enrichment import RawSavantEventsCache
from modules.baseball_module.advanced_pit_enrichment.savant_raw_ingestor import (
    SavantRawIngestor,
)


PIPELINE_MODULES = {
    "app",
    "backtest_and_retrain",
    "data_fetchers",
    "odds_fetcher",
    "run_daily_picks",
}


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self, by_day):
        self.headers = {}
        self.by_day = by_day
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        day = params["game_date_gt"]
        assert params["game_date_lt"] == day
        return _FakeResponse(self.by_day[day])


def _csv(*rows):
    header = (
        "game_date,game_pk,at_bat_number,pitch_number,pitcher,batter,"
        "player_name,events,launch_speed,launch_angle,"
        "estimated_woba_using_speedangle,woba_value,woba_denom,launch_speed_angle\n"
    )
    return header + "".join(rows)


def test_savant_raw_ingestor_ingests_fake_csv_rows(tmp_path):
    session = _FakeSession(
        {
            "2024-04-01": _csv(
                "2024-04-01,746001,1,1,605400,592450,\"Nola, Aaron\",field_out,95.0,20,0.320,0,1,6\n",
                "2024-04-01,746001,1,2,605400,592450,\"Nola, Aaron\",strikeout,,,,0,1,\n",
            )
        }
    )
    ingestor = SavantRawIngestor(tmp_path / "raw.db", session=session)

    summary = ingestor.ingest_date_range(start_date="2024-04-01", end_date="2024-04-01")
    events = ingestor.cache.get_events_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )

    assert summary.events_saved == 2
    assert summary.distinct_dates == 1
    assert summary.distinct_pitchers == 1
    assert summary.source_fingerprint.startswith("savant:statcast-raw:v1:2024-04-01:2024-04-01:")
    assert len(session.calls) == 1
    assert session.calls[0]["params"]["player_type"] == "pitcher"
    assert ingestor.cache.count_events() == 2
    assert events[0].raw_json["player_name"] == "Nola, Aaron"
    assert events[0].source_fingerprint == summary.source_fingerprint


def test_savant_raw_ingestor_dedupes_duplicate_events_on_reingest(tmp_path):
    session = _FakeSession(
        {
            "2024-04-01": _csv(
                "2024-04-01,746001,1,1,605400,592450,\"Nola, Aaron\",field_out,95.0,20,0.320,0,1,6\n",
            )
        }
    )
    ingestor = SavantRawIngestor(tmp_path / "raw.db", session=session)

    first = ingestor.ingest_date_range(start_date="2024-04-01", end_date="2024-04-01")
    second = ingestor.ingest_date_range(start_date="2024-04-01", end_date="2024-04-01")

    assert first.events_saved == 1
    assert second.events_saved == 1
    assert ingestor.cache.count_events() == 1


def test_savant_raw_ingestor_respects_date_range(tmp_path):
    session = _FakeSession(
        {
            "2024-04-01": _csv(
                "2024-04-01,746001,1,1,605400,592450,\"Nola, Aaron\",field_out,95.0,20,0.320,0,1,6\n",
            ),
            "2024-04-02": _csv(
                "2024-04-02,746002,1,1,999999,592450,\"Other, Pitcher\",single,80.0,10,0.500,0.9,1,5\n",
            ),
        }
    )
    ingestor = SavantRawIngestor(tmp_path / "raw.db", session=session)

    summary = ingestor.ingest_date_range(start_date="2024-04-01", end_date="2024-04-02")

    assert summary.events_saved == 2
    assert summary.distinct_dates == 2
    assert summary.distinct_pitchers == 2
    assert [call["params"]["game_date_gt"] for call in session.calls] == [
        "2024-04-01",
        "2024-04-02",
    ]
    events = ingestor.cache.get_events_by_date_range(
        start_date="2024-04-02",
        end_date="2024-04-02",
    )
    assert len(events) == 1
    assert events[0].game_date == "2024-04-02"


def test_savant_raw_ingestor_rejects_rows_outside_requested_day(tmp_path):
    session = _FakeSession(
        {
            "2024-04-01": _csv(
                "2024-04-02,746001,1,1,605400,592450,\"Nola, Aaron\",field_out,95.0,20,0.320,0,1,6\n",
            )
        }
    )
    ingestor = SavantRawIngestor(tmp_path / "raw.db", session=session)

    with pytest.raises(RuntimeError, match="expected '2024-04-01'"):
        ingestor.ingest_date_range(start_date="2024-04-01", end_date="2024-04-01")


def test_savant_raw_ingestor_preserves_null_fields(tmp_path):
    session = _FakeSession(
        {
            "2024-04-01": _csv(
                "2024-04-01,746001,1,1,605400,592450,\"Nola, Aaron\",strikeout,,,,0,1,\n",
            )
        }
    )
    ingestor = SavantRawIngestor(tmp_path / "raw.db", session=session)

    ingestor.ingest_date_range(start_date="2024-04-01", end_date="2024-04-01")
    event = ingestor.cache.get_events_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )[0]

    assert event.launch_speed is None
    assert event.launch_angle is None
    assert event.estimated_woba_using_speedangle is None
    assert event.launch_speed_angle is None
    assert event.woba_value == 0.0
    assert event.woba_denom == 1.0
    assert event.raw_json["launch_speed"] == ""


def test_savant_raw_ingestor_rejects_possible_daily_truncation(monkeypatch, tmp_path):
    monkeypatch.setattr(raw_ingestor, "_MAX_ROWS_PER_DAY", 1)
    session = _FakeSession(
        {
            "2024-04-01": _csv(
                "2024-04-01,746001,1,1,605400,592450,\"Nola, Aaron\",field_out,95.0,20,0.320,0,1,6\n",
            )
        }
    )
    ingestor = SavantRawIngestor(tmp_path / "raw.db", session=session)

    with pytest.raises(RuntimeError, match="may be truncated"):
        ingestor.ingest_date_range(start_date="2024-04-01", end_date="2024-04-01")


def test_savant_raw_ingestor_does_not_import_live_or_backtest_modules(tmp_path):
    for module_name in PIPELINE_MODULES:
        sys.modules.pop(module_name, None)

    importlib.import_module("modules.baseball_module.advanced_pit_enrichment.savant_raw_ingestor")
    cache = RawSavantEventsCache(tmp_path / "raw.db")
    assert cache.count_events() == 0

    assert PIPELINE_MODULES.isdisjoint(sys.modules)
