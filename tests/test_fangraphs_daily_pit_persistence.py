import inspect
import sqlite3

from modules.baseball_module.advanced_pit_enrichment import (
    FanGraphsDailyPITPersistence,
    PITCache,
)
from modules.baseball_module.advanced_pit_enrichment.fangraphs_pit_fetcher import (
    FanGraphsPITFetcher,
)
import modules.baseball_module.advanced_pit_enrichment.fangraphs_daily_pit_persistence as persistence_module


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, by_end_date):
        self.headers = {}
        self.by_end_date = by_end_date
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return _FakeResponse(self.by_end_date[params["enddate"]])


def _payload(start_date, end_date, *rows):
    return {"dateRange": f"{start_date} and {end_date}", "data": list(rows)}


def _row(**overrides):
    row = {
        "xMLBAMID": "669923",
        "xFIP": "3.44",
        "SIERA": "3.21",
        "FIP": "3.60",
        "xERA": "3.12",
        "K%": "28.5",
        "BB%": "7.1",
        "IP": "33.2",
        "playerid": "12345",
        "Name": '<a href="/players/example">Example Pitcher</a>',
    }
    row.update(overrides)
    return row


def _persister(tmp_path, by_end_date):
    pit_db = tmp_path / "pit.db"
    return FanGraphsDailyPITPersistence(
        pit_cache_db=pit_db,
        session=_FakeSession(by_end_date),
    ), pit_db


def test_persist_one_cutoff_writes_one_row_per_pitcher(tmp_path):
    persister, pit_db = _persister(
        tmp_path,
        {
            "2025-04-05": _payload(
                "2025-03-27",
                "2025-04-05",
                _row(),
                _row(xMLBAMID="999999", Name="Other Pitcher", playerid="54321", SIERA="4.10"),
                _row(xMLBAMID="", SIERA="2.00"),
            )
        },
    )

    persisted = persister.persist_cutoff(
        season=2025,
        season_start_date="2025-03-27",
        as_of_date="2025-04-05",
        fetched_at="2026-06-13T12:00:00Z",
    )

    assert sorted(persisted) == [669923, 999999]
    assert persisted[669923] == {
        "siera": 3.21,
        "xfip": 3.44,
        "xera": 3.12,
        "fip": 3.6,
        "k_pct": 28.5,
        "bb_pct": 7.1,
        "ip": 33.2,
        "player_name": "Example Pitcher",
        "fg_playerid": "12345",
        "mlbam_id": 669923,
        "source_window_start_date": "2025-03-27",
        "source_window_end_date": "2025-04-05",
        "metric_version": "fangraphs_daily_cutoff_v1",
    }

    cache = PITCache(pit_db)
    record = cache.get_record(
        namespace=FanGraphsDailyPITPersistence.NAMESPACE,
        entity_id=669923,
        season=2025,
        as_of_date="2025-04-05T00:00:00Z",
        source=FanGraphsDailyPITPersistence.SOURCE,
    )
    assert record is not None
    assert record.source_fingerprint.startswith(
        "fangraphs:daily-cutoff:fangraphs_daily_cutoff_v1:2025:2025-03-27:2025-04-05:2:"
    )
    with sqlite3.connect(pit_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM pit_metric_cache").fetchone()[0]
    assert count == 2


def test_persist_multiple_cutoffs_keeps_each_cutoff(tmp_path):
    persister, pit_db = _persister(
        tmp_path,
        {
            "2025-04-01": _payload("2025-03-27", "2025-04-01", _row(SIERA="3.50", IP="5.0")),
            "2025-04-02": _payload("2025-03-27", "2025-04-02", _row(SIERA="3.25", IP="11.0")),
        },
    )

    first = persister.persist_cutoff(season=2025, season_start_date="2025-03-27", as_of_date="2025-04-01")
    second = persister.persist_cutoff(season=2025, season_start_date="2025-03-27", as_of_date="2025-04-02")

    assert first[669923]["siera"] == 3.5
    assert second[669923]["siera"] == 3.25
    assert PITCache(pit_db).list_cutoffs(
        namespace=FanGraphsDailyPITPersistence.NAMESPACE,
        season=2025,
        source=FanGraphsDailyPITPersistence.SOURCE,
    ) == ["2025-04-01T00:00:00+00:00", "2025-04-02T00:00:00+00:00"]


def test_latest_lte_requested_as_of_date_retrieval(tmp_path):
    persister, _ = _persister(
        tmp_path,
        {
            "2025-04-01": _payload("2025-03-27", "2025-04-01", _row(SIERA="3.50")),
            "2025-04-02": _payload("2025-03-27", "2025-04-02", _row(SIERA="3.25")),
        },
    )
    persister.persist_cutoff(season=2025, season_start_date="2025-03-27", as_of_date="2025-04-01")
    persister.persist_cutoff(season=2025, season_start_date="2025-03-27", as_of_date="2025-04-02")

    record = persister.get_latest_pitcher_snapshot(
        pitcher=669923,
        season=2025,
        requested_as_of_date="2025-04-02T12:00:00Z",
    )

    assert record is not None
    assert record.as_of_date == "2025-04-02T00:00:00+00:00"
    assert record.data["siera"] == 3.25


def test_no_future_cutoff_retrieval_and_before_first_is_none(tmp_path):
    persister, _ = _persister(
        tmp_path,
        {
            "2025-04-01": _payload("2025-03-27", "2025-04-01", _row(SIERA="3.50")),
            "2025-04-03": _payload("2025-03-27", "2025-04-03", _row(SIERA="2.95")),
        },
    )
    persister.persist_cutoff(season=2025, season_start_date="2025-03-27", as_of_date="2025-04-01")
    persister.persist_cutoff(season=2025, season_start_date="2025-03-27", as_of_date="2025-04-03")

    before_first = persister.get_latest_pitcher_snapshot(
        pitcher=669923,
        season=2025,
        requested_as_of_date="2025-03-31T23:59:59Z",
    )
    before_future = persister.get_latest_pitcher_snapshot(
        pitcher=669923,
        season=2025,
        requested_as_of_date="2025-04-02T23:59:59Z",
    )

    assert before_first is None
    assert before_future is not None
    assert before_future.as_of_date == "2025-04-01T00:00:00+00:00"
    assert before_future.data["source_window_end_date"] == "2025-04-01"


def test_duplicate_persist_upserts_and_dedupes(tmp_path):
    persister, pit_db = _persister(
        tmp_path,
        {"2025-04-01": _payload("2025-03-27", "2025-04-01", _row())},
    )

    persister.persist_cutoff(season=2025, season_start_date="2025-03-27", as_of_date="2025-04-01", fetched_at="2026-06-13T12:00:00Z")
    persister.persist_cutoff(season=2025, season_start_date="2025-03-27", as_of_date="2025-04-01", fetched_at="2026-06-13T13:00:00Z")

    with sqlite3.connect(pit_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM pit_metric_cache").fetchone()[0]
        fetched_at = conn.execute("SELECT fetched_at FROM pit_metric_cache").fetchone()[0]

    assert count == 1
    assert fetched_at == "2026-06-13T13:00:00+00:00"


def test_missing_siera_is_preserved_as_none(tmp_path):
    persister, _ = _persister(
        tmp_path,
        {"2025-04-01": _payload("2025-03-27", "2025-04-01", _row(SIERA=""))},
    )

    persisted = persister.persist_cutoff(
        season=2025,
        season_start_date="2025-03-27",
        as_of_date="2025-04-01",
    )

    assert persisted[669923]["siera"] is None


def test_daily_rows_do_not_collide_with_existing_aggregate_fangraphs_rows(tmp_path):
    pit_db = tmp_path / "pit.db"
    cache = PITCache(pit_db)
    cache.save_record(
        namespace=FanGraphsPITFetcher.PITCHER_NAMESPACE,
        entity_id=669923,
        season=2025,
        as_of_date="2025-04-01T00:00:00Z",
        source=FanGraphsPITFetcher.SOURCE,
        source_fingerprint="aggregate-existing",
        data={"siera": 9.99},
        fetched_at="2026-06-13T11:00:00Z",
    )
    persister = FanGraphsDailyPITPersistence(
        pit_cache_db=pit_db,
        session=_FakeSession({"2025-04-01": _payload("2025-03-27", "2025-04-01", _row(SIERA="3.50"))}),
    )

    persister.persist_cutoff(season=2025, season_start_date="2025-03-27", as_of_date="2025-04-01")

    aggregate = cache.get_record(
        namespace=FanGraphsPITFetcher.PITCHER_NAMESPACE,
        entity_id=669923,
        season=2025,
        as_of_date="2025-04-01T00:00:00Z",
        source=FanGraphsPITFetcher.SOURCE,
    )
    daily = persister.get_latest_pitcher_snapshot(
        pitcher=669923,
        season=2025,
        requested_as_of_date="2025-04-01T12:00:00Z",
    )

    assert aggregate is not None
    assert aggregate.source_fingerprint == "aggregate-existing"
    assert aggregate.data == {"siera": 9.99}
    assert daily is not None
    assert daily.data["siera"] == 3.5
    with sqlite3.connect(pit_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM pit_metric_cache").fetchone()[0]
    assert count == 2


def test_persistence_module_does_not_import_live_backtest_or_model_code():
    source = inspect.getsource(persistence_module)

    assert "backtest" not in source
    assert "model" not in source
    assert "data_fetchers" not in source
    assert "odds_fetcher" not in source
