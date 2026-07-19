import importlib
import json
import sys
from pathlib import Path

import pytest
import requests

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


class _FailingSession(_FakeSession):
    def get(self, url, *, params, timeout):
        day = params["game_date_gt"]
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        value = self.by_day[day]
        if isinstance(value, Exception):
            raise value
        return _FakeResponse(value)


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


def test_savant_raw_ingestor_does_not_import_live_or_backtest_modules(tmp_path, monkeypatch):
    # monkeypatch.delitem (not a raw sys.modules.pop()) restores these to
    # their pre-test cached state after this test — a raw pop() here
    # permanently evicted them from sys.modules for the rest of the pytest
    # session, silently breaking any later test's monkeypatch.setattr() on
    # one of these already-imported modules (found 2026-07-19 bisecting a
    # flaky odds_fetcher test; see test_build_raw_savant_events_script.py's
    # identical fix for the first instance of this pattern).
    for module_name in PIPELINE_MODULES:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    importlib.import_module("modules.baseball_module.advanced_pit_enrichment.savant_raw_ingestor")
    cache = RawSavantEventsCache(tmp_path / "raw.db")
    assert cache.count_events() == 0

    assert PIPELINE_MODULES.isdisjoint(sys.modules)


def test_historical_resume_skips_completed_dates_and_updates_manifest(tmp_path):
    session = _FakeSession(
        {
            "2023-03-30": _csv(
                "2023-03-30,700001,1,1,605400,592450,Nola,field_out,95,20,0.320,0,1,6\n",
            ),
            "2023-03-31": _csv(
                "2023-03-31,700002,1,1,605401,592451,Other,single,98,18,0.500,0.9,1,6\n",
            ),
        }
    )
    db_path = tmp_path / "raw_savant_2023.db"
    manifest_path = tmp_path / "raw_savant_2023.manifest.json"
    ingestor = SavantRawIngestor(db_path, session=session, sleep=lambda _: None)

    first = ingestor.ingest_historical_date_range(
        season=2023,
        start_date="2023-03-30",
        end_date="2023-03-31",
        manifest_path=manifest_path,
    )
    second = ingestor.ingest_historical_date_range(
        season=2023,
        start_date="2023-03-30",
        end_date="2023-03-31",
        manifest_path=manifest_path,
    )
    manifest = json.loads(manifest_path.read_text())

    assert first.fetched_dates == ("2023-03-30", "2023-03-31")
    assert second.fetched_dates == ()
    assert second.skipped_dates == ("2023-03-30", "2023-03-31")
    assert len(session.calls) == 2
    assert ingestor.cache.count_events() == 2
    assert manifest["completed_dates"] == ["2023-03-30", "2023-03-31"]
    assert manifest["failed_dates"] == {}
    assert manifest["total_events_saved"] == 2
    assert manifest["distinct_game_dates"] == 2
    assert manifest["season"] == 2023
    assert manifest["schema_version"] == "raw_savant_events_v1"
    assert manifest["source_fingerprint"].startswith("savant:historical-cache:")
    assert Path(manifest["database_path"]) == db_path.resolve()


def test_historical_partial_failure_preserves_completed_data_and_records_failure(tmp_path):
    session = _FailingSession(
        {
            "2023-03-30": _csv(
                "2023-03-30,700001,1,1,605400,592450,Nola,field_out,95,20,0.320,0,1,6\n",
            ),
            "2023-03-31": requests.ConnectionError("temporary outage"),
        }
    )
    db_path = tmp_path / "raw_savant_2023.db"
    manifest_path = tmp_path / "raw_savant_2023.manifest.json"
    ingestor = SavantRawIngestor(db_path, session=session, sleep=lambda _: None)

    summary = ingestor.ingest_historical_date_range(
        season=2023,
        start_date="2023-03-30",
        end_date="2023-03-31",
        manifest_path=manifest_path,
        max_retries=2,
    )
    manifest = json.loads(manifest_path.read_text())

    assert summary.completed_dates == ("2023-03-30",)
    assert "2023-03-31" in summary.failed_dates
    assert ingestor.cache.count_events() == 1
    assert manifest["completed_dates"] == ["2023-03-30"]
    assert "ConnectionError" in manifest["failed_dates"]["2023-03-31"]
    assert manifest["total_events_saved"] == 1
    assert manifest_path.exists()


def test_historical_rerun_retries_failed_date_without_duplicate_events(tmp_path):
    db_path = tmp_path / "raw_savant_2023.db"
    manifest_path = tmp_path / "raw_savant_2023.manifest.json"
    failing = _FailingSession(
        {
            "2023-03-30": _csv(
                "2023-03-30,700001,1,1,605400,592450,Nola,field_out,95,20,0.320,0,1,6\n",
            ),
            "2023-03-31": requests.Timeout("timeout"),
        }
    )
    SavantRawIngestor(db_path, session=failing, sleep=lambda _: None).ingest_historical_date_range(
        season=2023,
        start_date="2023-03-30",
        end_date="2023-03-31",
        manifest_path=manifest_path,
        max_retries=1,
    )

    resumed = _FakeSession(
        {
            "2023-03-31": _csv(
                "2023-03-31,700002,1,1,605401,592451,Other,single,98,18,0.500,0.9,1,6\n",
            )
        }
    )
    summary = SavantRawIngestor(
        db_path, session=resumed, sleep=lambda _: None
    ).ingest_historical_date_range(
        season=2023,
        start_date="2023-03-30",
        end_date="2023-03-31",
        manifest_path=manifest_path,
    )

    assert summary.skipped_dates == ("2023-03-30",)
    assert summary.fetched_dates == ("2023-03-31",)
    assert summary.failed_dates == {}
    assert RawSavantEventsCache(db_path).count_events() == 2
