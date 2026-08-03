import sqlite3

import pytest

from modules.baseball_module.advanced_pit_enrichment import PITCache


def test_pit_cache_round_trips_exact_record(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    payload = {"siera": 3.21, "xfip": 3.44, "nested": {"k_bb_pct": 22.4}}

    cache.save_record(
        namespace="fangraphs.pitcher",
        entity_id=669923,
        season=2025,
        as_of_date="2025-04-05T12:00:00Z",
        source="fangraphs",
        source_fingerprint="fg:2025-03-27:2025-04-05",
        data=payload,
        fetched_at="2026-06-06T12:00:00Z",
    )

    record = cache.get_record(
        namespace="fangraphs.pitcher",
        entity_id=669923,
        season=2025,
        as_of_date="2025-04-05T12:00:00+00:00",
        source="fangraphs",
    )

    assert record is not None
    assert record.namespace == "fangraphs.pitcher"
    assert record.entity_id == "669923"
    assert record.season == 2025
    assert record.as_of_date == "2025-04-05T12:00:00+00:00"
    assert record.source == "fangraphs"
    assert record.source_fingerprint == "fg:2025-03-27:2025-04-05"
    assert record.data == payload
    assert record.fetched_at == "2026-06-06T12:00:00+00:00"


def test_get_latest_never_returns_future_record(tmp_path):
    cache = PITCache(tmp_path / "pit.db")

    cache.save_record(
        namespace="savant.pitcher",
        entity_id=123,
        season=2025,
        as_of_date="2025-04-01T00:00:00Z",
        source="baseball_savant",
        source_fingerprint="savant:old",
        data={"xera": 3.80},
        fetched_at="2026-06-06T12:00:00Z",
    )
    cache.save_record(
        namespace="savant.pitcher",
        entity_id=123,
        season=2025,
        as_of_date="2025-04-10T00:00:00Z",
        source="baseball_savant",
        source_fingerprint="savant:future",
        data={"xera": 2.95},
        fetched_at="2026-06-06T12:00:00Z",
    )

    before_first = cache.get_latest(
        namespace="savant.pitcher",
        entity_id=123,
        season=2025,
        as_of_date="2025-03-31T23:59:59Z",
        source="baseball_savant",
    )
    before_future = cache.get_latest(
        namespace="savant.pitcher",
        entity_id=123,
        season=2025,
        as_of_date="2025-04-05T19:00:00Z",
        source="baseball_savant",
    )

    assert before_first is None
    assert before_future is not None
    assert before_future.as_of_date == "2025-04-01T00:00:00+00:00"
    assert before_future.data == {"xera": 3.80}


def test_source_filter_keeps_providers_isolated(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    common = {
        "namespace": "pitcher.metric",
        "entity_id": "42",
        "season": 2025,
        "as_of_date": "2025-05-01T00:00:00Z",
        "fetched_at": "2026-06-06T12:00:00Z",
    }

    cache.save_record(
        **common,
        source="fangraphs",
        source_fingerprint="fg",
        data={"metric": "fg"},
    )
    cache.save_record(
        **common,
        source="baseball_savant",
        source_fingerprint="savant",
        data={"metric": "savant"},
    )

    fg = cache.get_latest(
        namespace="pitcher.metric",
        entity_id="42",
        season=2025,
        as_of_date="2025-05-02T00:00:00Z",
        source="fangraphs",
    )
    savant = cache.get_latest(
        namespace="pitcher.metric",
        entity_id="42",
        season=2025,
        as_of_date="2025-05-02T00:00:00Z",
        source="baseball_savant",
    )

    assert fg is not None
    assert savant is not None
    assert fg.data == {"metric": "fg"}
    assert savant.data == {"metric": "savant"}


def test_save_record_upserts_same_identity(tmp_path):
    db_path = tmp_path / "pit.db"
    cache = PITCache(db_path)
    identity = {
        "namespace": "fangraphs.pitcher",
        "entity_id": 99,
        "season": 2025,
        "as_of_date": "2025-06-01T00:00:00Z",
        "source": "fangraphs",
    }

    cache.save_record(
        **identity,
        source_fingerprint="old",
        data={"siera": 4.00},
        fetched_at="2026-06-06T12:00:00Z",
    )
    cache.save_record(
        **identity,
        source_fingerprint="new",
        data={"siera": 3.75},
        fetched_at="2026-06-06T13:00:00Z",
    )

    record = cache.get_record(**identity)

    assert record is not None
    assert record.source_fingerprint == "new"
    assert record.data == {"siera": 3.75}
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM pit_metric_cache").fetchone()[0]
    assert count == 1


@pytest.mark.parametrize(
    "field, value",
    [
        ("as_of_date", "not-a-date"),
        ("fetched_at", "2025-13-01T00:00:00Z"),
    ],
)
def test_invalid_datetime_is_rejected(tmp_path, field, value):
    cache = PITCache(tmp_path / "pit.db")
    kwargs = {
        "namespace": "fangraphs.pitcher",
        "entity_id": 1,
        "season": 2025,
        "as_of_date": "2025-04-01T00:00:00Z",
        "source": "fangraphs",
        "source_fingerprint": "bad",
        "data": {},
        "fetched_at": "2026-06-06T12:00:00Z",
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match="invalid ISO datetime"):
        cache.save_record(**kwargs)
