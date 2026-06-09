import sqlite3

from modules.baseball_module.advanced_pit_enrichment import RawSavantEventsCache


def _event(**overrides):
    event = {
        "game_date": "2024-04-01",
        "game_pk": "746001",
        "at_bat_number": "12",
        "pitch_number": "3",
        "pitcher": "605400",
        "batter": "592450",
        "events": "field_out",
        "launch_speed": "95.2",
        "launch_angle": "18",
        "estimated_woba_using_speedangle": "0.320",
        "woba_value": "0",
        "woba_denom": "1",
        "launch_speed_angle": "6",
        "description": "hit_into_play",
    }
    event.update(overrides)
    return event


def test_raw_savant_events_schema_creation(tmp_path):
    db_path = tmp_path / "raw_savant.db"
    RawSavantEventsCache(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {
            row[1]: {"type": row[2], "pk": row[5]}
            for row in conn.execute("PRAGMA table_info(raw_savant_events)")
        }

    assert columns["game_date"]["type"] == "TEXT"
    assert columns["game_pk"]["pk"] == 1
    assert columns["at_bat_number"]["pk"] == 2
    assert columns["pitch_number"]["pk"] == 3
    assert "raw_json" in columns
    assert "source_fingerprint" in columns
    assert "fetched_at" in columns


def test_save_events_inserts_and_round_trips_typed_values(tmp_path):
    cache = RawSavantEventsCache(tmp_path / "raw_savant.db")

    cache.save_events(
        [_event()],
        source_fingerprint="savant:2024-04-01",
        fetched_at="2026-06-09T12:00:00Z",
    )

    events = cache.get_events_by_date_range(start_date="2024-04-01", end_date="2024-04-01")

    assert cache.count_events() == 1
    assert cache.count_distinct_dates() == 1
    assert len(events) == 1
    event = events[0]
    assert event.game_pk == 746001
    assert event.at_bat_number == 12
    assert event.pitch_number == 3
    assert event.pitcher == 605400
    assert event.batter == 592450
    assert event.events == "field_out"
    assert event.launch_speed == 95.2
    assert event.launch_angle == 18.0
    assert event.estimated_woba_using_speedangle == 0.320
    assert event.woba_value == 0.0
    assert event.woba_denom == 1.0
    assert event.launch_speed_angle == 6
    assert event.raw_json["description"] == "hit_into_play"
    assert event.source_fingerprint == "savant:2024-04-01"
    assert event.fetched_at == "2026-06-09T12:00:00+00:00"


def test_duplicate_pitch_identity_upserts_without_duplicate_rows(tmp_path):
    cache = RawSavantEventsCache(tmp_path / "raw_savant.db")
    identity = {
        "game_pk": "746001",
        "at_bat_number": "12",
        "pitch_number": "3",
    }

    cache.save_events(
        [_event(**identity, launch_speed="95.2", events="field_out")],
        source_fingerprint="old",
        fetched_at="2026-06-09T12:00:00Z",
    )
    cache.save_events(
        [_event(**identity, launch_speed="101.5", events="single")],
        source_fingerprint="new",
        fetched_at="2026-06-09T13:00:00Z",
    )

    events = cache.get_events_by_date_range(start_date="2024-04-01", end_date="2024-04-01")

    assert cache.count_events() == 1
    assert events[0].launch_speed == 101.5
    assert events[0].events == "single"
    assert events[0].source_fingerprint == "new"
    assert events[0].fetched_at == "2026-06-09T13:00:00+00:00"


def test_get_events_by_date_range_filters_and_orders(tmp_path):
    cache = RawSavantEventsCache(tmp_path / "raw_savant.db")
    cache.save_events(
        [
            _event(game_date="2024-04-03", game_pk="746003", at_bat_number="2", pitch_number="1"),
            _event(game_date="2024-04-01", game_pk="746001", at_bat_number="1", pitch_number="1"),
            _event(game_date="2024-04-02", game_pk="746002", at_bat_number="1", pitch_number="2"),
        ],
        source_fingerprint="window",
        fetched_at="2026-06-09T12:00:00Z",
    )

    events = cache.get_events_by_date_range(start_date="2024-04-02", end_date="2024-04-03")

    assert [event.game_date for event in events] == ["2024-04-02", "2024-04-03"]
    assert [event.game_pk for event in events] == [746002, 746003]
    assert cache.count_events() == 3
    assert cache.count_distinct_dates() == 3


def test_null_like_optional_values_remain_none(tmp_path):
    cache = RawSavantEventsCache(tmp_path / "raw_savant.db")

    cache.save_events(
        [
            _event(
                pitcher="",
                batter=None,
                events="",
                launch_speed="",
                launch_angle="NULL",
                estimated_woba_using_speedangle=None,
                woba_value="null",
                woba_denom="",
                launch_speed_angle=None,
            )
        ],
        source_fingerprint="nulls",
        fetched_at="2026-06-09T12:00:00Z",
    )

    event = cache.get_events_by_date_range(start_date="2024-04-01", end_date="2024-04-01")[0]

    assert event.pitcher is None
    assert event.batter is None
    assert event.events is None
    assert event.launch_speed is None
    assert event.launch_angle is None
    assert event.estimated_woba_using_speedangle is None
    assert event.woba_value is None
    assert event.woba_denom is None
    assert event.launch_speed_angle is None


def test_raw_savant_events_cache_uses_only_requested_temp_db(tmp_path):
    db_path = tmp_path / "nested" / "raw_savant.db"
    cache = RawSavantEventsCache(db_path)

    cache.save_events(
        [_event()],
        source_fingerprint="temp-only",
        fetched_at="2026-06-09T12:00:00Z",
    )

    assert db_path.exists()
    assert cache.count_events() == 1
