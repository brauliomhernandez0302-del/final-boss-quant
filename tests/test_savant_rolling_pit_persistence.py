import inspect
import sqlite3

import pytest

import modules.baseball_module.advanced_pit_enrichment.savant_rolling_pit_persistence as persistence_module
from modules.baseball_module.advanced_pit_enrichment import (
    PITCache,
    RawSavantEventsCache,
    SavantRollingPITPersistence,
)


def _event(**overrides):
    event = {
        "game_date": "2024-04-01",
        "game_pk": "746001",
        "at_bat_number": "1",
        "pitch_number": "1",
        "pitcher": "605400",
        "batter": "592450",
        "events": "field_out",
        "launch_speed": "95.0",
        "launch_angle": "20",
        "estimated_woba_using_speedangle": "0.300",
        "woba_value": "0",
        "woba_denom": "1",
        "launch_speed_angle": "6",
    }
    event.update(overrides)
    return event


def _raw_cache(tmp_path, events):
    db_path = tmp_path / "raw_savant.db"
    cache = RawSavantEventsCache(db_path)
    cache.save_events(events, source_fingerprint="raw-test", fetched_at="2026-06-09T12:00:00Z")
    return db_path


def _persister(tmp_path, events):
    raw_db = _raw_cache(tmp_path, events)
    pit_db = tmp_path / "pit.db"
    return SavantRollingPITPersistence(raw_cache_db=raw_db, pit_cache_db=pit_db), pit_db


def test_persist_one_cutoff_writes_one_row_per_pitcher(tmp_path):
    persister, pit_db = _persister(
        tmp_path,
        [
            _event(game_pk="1", pitch_number="1", pitcher="605400", launch_speed="95", woba_value="0", woba_denom="1", launch_speed_angle="6"),
            _event(game_pk="1", pitch_number="2", pitcher="605400", launch_speed="", launch_angle="", estimated_woba_using_speedangle="", woba_value="0", woba_denom="1", launch_speed_angle=""),
            _event(game_pk="2", pitch_number="1", pitcher="999999", launch_speed="80", woba_value="0.9", woba_denom="1", launch_speed_angle="5"),
        ],
    )

    persisted = persister.persist_cutoff(
        season=2024,
        season_start_date="2024-04-01",
        as_of_date="2024-04-01",
        fetched_at="2026-06-10T12:00:00Z",
    )

    assert sorted(persisted) == [605400, 999999]
    cache = PITCache(pit_db)
    record = cache.get_record(
        namespace=SavantRollingPITPersistence.NAMESPACE,
        entity_id=605400,
        season=2024,
        as_of_date="2024-04-01T00:00:00Z",
        source=SavantRollingPITPersistence.SOURCE,
    )

    assert record is not None
    assert record.source == "baseball_savant_rolling"
    assert record.source_fingerprint.startswith("savant:raw:rolling:savant_rolling_v1:2024-04-01:2024-04-01:3:1:")
    assert record.data["metric_version"] == "savant_rolling_v1"
    assert record.data["source_window_start_date"] == "2024-04-01"
    assert record.data["source_window_end_date"] == "2024-04-01"
    assert record.data["batted_ball_count"] == 1
    assert record.data["barrel_count"] == 1
    assert record.data["brl_percent"] == 100.0
    assert record.data["woba_numerator"] == 0.0
    assert record.data["woba_denominator"] == 2.0
    with sqlite3.connect(pit_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM pit_metric_cache").fetchone()[0]
    assert count == 2


def test_persist_multiple_cutoffs_keeps_each_cutoff(tmp_path):
    persister, pit_db = _persister(
        tmp_path,
        [
            _event(game_date="2024-04-01", game_pk="1", pitcher="605400", launch_speed="95", woba_value="0", woba_denom="1", launch_speed_angle="6"),
            _event(game_date="2024-04-02", game_pk="2", pitcher="605400", launch_speed="100", estimated_woba_using_speedangle="0.700", woba_value="0.9", woba_denom="1", launch_speed_angle="5"),
        ],
    )

    first = persister.persist_cutoff(season=2024, season_start_date="2024-04-01", as_of_date="2024-04-01")
    second = persister.persist_cutoff(season=2024, season_start_date="2024-04-01", as_of_date="2024-04-02")

    assert first[605400]["batted_ball_count"] == 1
    assert second[605400]["batted_ball_count"] == 2
    cache = PITCache(pit_db)
    assert cache.list_cutoffs(
        namespace=SavantRollingPITPersistence.NAMESPACE,
        season=2024,
        source=SavantRollingPITPersistence.SOURCE,
    ) == ["2024-04-01T00:00:00+00:00", "2024-04-02T00:00:00+00:00"]


def test_latest_lte_as_of_date_retrieval(tmp_path):
    persister, _ = _persister(
        tmp_path,
        [
            _event(game_date="2024-04-01", game_pk="1", pitcher="605400", launch_speed="95"),
            _event(game_date="2024-04-02", game_pk="2", pitcher="605400", launch_speed="100"),
        ],
    )
    persister.persist_cutoff(season=2024, season_start_date="2024-04-01", as_of_date="2024-04-01")
    persister.persist_cutoff(season=2024, season_start_date="2024-04-01", as_of_date="2024-04-02")

    record = persister.get_latest_pitcher_snapshot(
        pitcher=605400,
        season=2024,
        requested_as_of_date="2024-04-02T12:00:00Z",
    )

    assert record is not None
    assert record.as_of_date == "2024-04-02T00:00:00+00:00"
    assert record.data["batted_ball_count"] == 2


def test_no_future_cutoff_retrieval_and_before_first_is_none(tmp_path):
    persister, _ = _persister(
        tmp_path,
        [
            _event(game_date="2024-04-01", game_pk="1", pitcher="605400", launch_speed="95"),
            _event(game_date="2024-04-03", game_pk="3", pitcher="605400", launch_speed="100"),
        ],
    )
    persister.persist_cutoff(season=2024, season_start_date="2024-04-01", as_of_date="2024-04-01")
    persister.persist_cutoff(season=2024, season_start_date="2024-04-01", as_of_date="2024-04-03")

    before_first = persister.get_latest_pitcher_snapshot(
        pitcher=605400,
        season=2024,
        requested_as_of_date="2024-03-31T23:59:59Z",
    )
    before_future = persister.get_latest_pitcher_snapshot(
        pitcher=605400,
        season=2024,
        requested_as_of_date="2024-04-02T23:59:59Z",
    )

    assert before_first is None
    assert before_future is not None
    assert before_future.as_of_date == "2024-04-01T00:00:00+00:00"
    assert before_future.data["source_window_end_date"] == "2024-04-01"


def test_duplicate_persist_upserts_and_dedupes(tmp_path):
    persister, pit_db = _persister(
        tmp_path,
        [_event(game_pk="1", pitcher="605400", launch_speed="95")],
    )

    persister.persist_cutoff(season=2024, season_start_date="2024-04-01", as_of_date="2024-04-01", fetched_at="2026-06-10T12:00:00Z")
    persister.persist_cutoff(season=2024, season_start_date="2024-04-01", as_of_date="2024-04-01", fetched_at="2026-06-10T13:00:00Z")

    with sqlite3.connect(pit_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM pit_metric_cache").fetchone()[0]
        fetched_at = conn.execute("SELECT fetched_at FROM pit_metric_cache").fetchone()[0]

    assert count == 1
    assert fetched_at == "2026-06-10T13:00:00+00:00"


def test_null_denominator_is_preserved(tmp_path):
    persister, _ = _persister(
        tmp_path,
        [
            _event(
                game_pk="1",
                pitcher="605400",
                launch_speed="",
                launch_angle="",
                estimated_woba_using_speedangle="",
                woba_value="",
                woba_denom="",
                launch_speed_angle="",
            )
        ],
    )

    persisted = persister.persist_cutoff(
        season=2024,
        season_start_date="2024-04-01",
        as_of_date="2024-04-01",
    )

    row = persisted[605400]
    assert row["woba"] is None
    assert row["woba_numerator"] is None
    assert row["woba_denominator"] is None
    assert row["brl_percent"] is None
    assert row["ev95percent"] is None


def test_real_zero_barrel_percent_is_preserved(tmp_path):
    persister, _ = _persister(
        tmp_path,
        [
            _event(game_pk="1", pitch_number="1", pitcher="605400", launch_speed="88", launch_speed_angle="4"),
            _event(game_pk="1", pitch_number="2", pitcher="605400", launch_speed="89", launch_speed_angle="5"),
        ],
    )

    persisted = persister.persist_cutoff(
        season=2024,
        season_start_date="2024-04-01",
        as_of_date="2024-04-01",
    )

    row = persisted[605400]
    assert row["barrel_count"] == 0
    assert row["batted_ball_count"] == 2
    assert row["brl_percent"] == 0.0


def test_persistence_module_does_not_import_live_backtest_or_model_code():
    source = inspect.getsource(persistence_module)

    assert "backtest" not in source
    assert "model" not in source
    assert "data_fetchers" not in source
    assert "odds_fetcher" not in source
