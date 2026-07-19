import importlib
import inspect
import sqlite3
import sys

from modules.baseball_module.advanced_pit_enrichment import (
    AdvancedPitcherDailySnapshotBuilder,
    FanGraphsDailyPITPersistence,
    PITCache,
    SavantRollingPITPersistence,
)


SCRIPT_MODULE = "scripts.build_experimental_pitcher_pit_cache"
PIPELINE_MODULES = {
    "app",
    "backtest_and_retrain",
    "data_fetchers",
    "odds_fetcher",
    "run_daily_picks",
}


class _FakeSavantRawIngestor:
    calls = []

    def __init__(self, raw_savant_db):
        self.raw_savant_db = raw_savant_db

    def ingest_historical_date_range(self, *, season, start_date, end_date, manifest_path):
        self.calls.append(
            {
                "raw_savant_db": self.raw_savant_db,
                "season": season,
                "start_date": start_date,
                "end_date": end_date,
                "manifest_path": manifest_path,
            }
        )
        return object()


class _FakeFanGraphsDailyPITPersistence:
    calls = []
    NAMESPACE = FanGraphsDailyPITPersistence.NAMESPACE
    SOURCE = FanGraphsDailyPITPersistence.SOURCE

    def __init__(self, *, pit_cache_db):
        self.pit_cache_db = pit_cache_db

    def persist_cutoff(self, *, season, season_start_date, as_of_date):
        self.calls.append(
            {
                "pit_cache_db": self.pit_cache_db,
                "season": season,
                "season_start_date": season_start_date,
                "as_of_date": as_of_date,
            }
        )
        return {1: {"siera": 3.21}}


class _FakeSavantRollingPITPersistence:
    calls = []
    NAMESPACE = SavantRollingPITPersistence.NAMESPACE
    SOURCE = SavantRollingPITPersistence.SOURCE

    def __init__(self, *, raw_cache_db, pit_cache_db):
        self.raw_cache_db = raw_cache_db
        self.pit_cache_db = pit_cache_db

    def persist_cutoff(self, *, season, season_start_date, as_of_date):
        self.calls.append(
            {
                "raw_cache_db": self.raw_cache_db,
                "pit_cache_db": self.pit_cache_db,
                "season": season,
                "season_start_date": season_start_date,
                "as_of_date": as_of_date,
            }
        )
        return {1: {"est_woba": 0.300}}


def _load_script(monkeypatch):
    # monkeypatch.delitem (not a raw sys.modules.pop()) restores these after
    # the test — a raw pop() permanently evicts them from sys.modules for
    # the rest of the pytest session, silently breaking a later test's
    # monkeypatch.setattr() on one of these already-imported modules (found
    # 2026-07-19; see test_build_raw_savant_events_script.py's identical fix).
    for module_name in PIPELINE_MODULES:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    script = importlib.import_module(SCRIPT_MODULE)
    monkeypatch.setattr(script, "SavantRawIngestor", _FakeSavantRawIngestor)
    monkeypatch.setattr(script, "FanGraphsDailyPITPersistence", _FakeFanGraphsDailyPITPersistence)
    monkeypatch.setattr(script, "SavantRollingPITPersistence", _FakeSavantRollingPITPersistence)
    _FakeSavantRawIngestor.calls = []
    _FakeFanGraphsDailyPITPersistence.calls = []
    _FakeSavantRollingPITPersistence.calls = []
    return script


def _argv(tmp_path):
    return [
        "--pit-cache-db",
        str(tmp_path / "pit.db"),
        "--raw-savant-db",
        str(tmp_path / "raw_savant.db"),
        "--season",
        "2024",
        "--season-start-date",
        "2024-03-20",
        "--start-date",
        "2024-04-02",
        "--end-date",
        "2024-04-03",
        "--cutoff-policy",
        "previous_day",
    ]


def test_cli_parses_args(tmp_path, monkeypatch):
    script = _load_script(monkeypatch)

    args = script.parse_args(_argv(tmp_path))

    assert args.pit_cache_db == tmp_path / "pit.db"
    assert args.raw_savant_db == tmp_path / "raw_savant.db"
    assert args.season == 2024
    assert args.season_start_date == "2024-03-20"
    assert args.start_date == "2024-04-02"
    assert args.end_date == "2024-04-03"
    assert args.cutoff_policy == "previous_day"


def test_previous_day_cutoff_policy():
    script = importlib.import_module(SCRIPT_MODULE)

    plan = script.build_cutoff_plan(
        start_date="2024-04-02",
        end_date="2024-04-03",
        cutoff_policy="previous_day",
    )

    assert [(p.game_date, p.cutoff_date) for p in plan] == [
        ("2024-04-02", "2024-04-01"),
        ("2024-04-03", "2024-04-02"),
    ]


def test_builder_calls_canonical_providers_for_previous_day_cutoffs(tmp_path, monkeypatch, capsys):
    script = _load_script(monkeypatch)

    assert script.main(_argv(tmp_path)) == 0

    assert _FakeSavantRawIngestor.calls == [
        {
            "raw_savant_db": tmp_path / "raw_savant.db",
            "season": 2024,
            "start_date": "2024-03-20",
            "end_date": "2024-04-02",
            "manifest_path": tmp_path / "raw_savant.manifest.json",
        }
    ]
    assert [c["as_of_date"] for c in _FakeFanGraphsDailyPITPersistence.calls] == [
        "2024-04-01",
        "2024-04-02",
    ]
    assert [c["as_of_date"] for c in _FakeSavantRollingPITPersistence.calls] == [
        "2024-04-01",
        "2024-04-02",
    ]
    output = capsys.readouterr().out
    assert "cutoff=2024-04-01" in output
    assert "cutoff=2024-04-02" in output


def test_script_helper_produces_expected_namespaces_and_no_old_namespaces(tmp_path):
    pit_db = tmp_path / "pit.db"
    cache = PITCache(pit_db)
    _seed_canonical_rows(cache, as_of_date="2024-04-01T00:00:00Z")

    with sqlite3.connect(pit_db) as conn:
        rows = conn.execute(
            "SELECT namespace, source, COUNT(*) FROM pit_metric_cache GROUP BY namespace, source"
        ).fetchall()

    assert rows == [
        (FanGraphsDailyPITPersistence.NAMESPACE, FanGraphsDailyPITPersistence.SOURCE, 1),
        (SavantRollingPITPersistence.NAMESPACE, SavantRollingPITPersistence.SOURCE, 1),
    ]
    assert all(row[0] not in {"fangraphs.pitcher", "savant.pitcher"} for row in rows)


def test_no_future_or_same_day_rows_for_previous_day_requested_game_date(tmp_path):
    pit_db = tmp_path / "pit.db"
    cache = PITCache(pit_db)
    _seed_canonical_rows(cache, as_of_date="2024-04-01T00:00:00Z", siera=3.21)
    _seed_canonical_rows(cache, as_of_date="2024-04-02T00:00:00Z", siera=9.99)

    builder = AdvancedPitcherDailySnapshotBuilder(cache_db=pit_db)
    snapshot = builder.build_pitcher_snapshot(
        pitcher=669923,
        season=2024,
        requested_as_of_date="2024-04-01T23:59:59Z",
    )

    assert snapshot["found"] is True
    assert snapshot["fangraphs_as_of_date"] == "2024-04-01T00:00:00+00:00"
    assert snapshot["siera"] == 3.21


def test_explicit_same_day_request_can_use_same_day_cutoff(tmp_path):
    pit_db = tmp_path / "pit.db"
    cache = PITCache(pit_db)
    _seed_canonical_rows(cache, as_of_date="2024-04-01T00:00:00Z", siera=3.21)
    _seed_canonical_rows(cache, as_of_date="2024-04-02T00:00:00Z", siera=2.22)

    builder = AdvancedPitcherDailySnapshotBuilder(cache_db=pit_db)
    snapshot = builder.build_pitcher_snapshot(
        pitcher=669923,
        season=2024,
        requested_as_of_date="2024-04-02T23:59:59Z",
    )

    assert snapshot["fangraphs_as_of_date"] == "2024-04-02T00:00:00+00:00"
    assert snapshot["siera"] == 2.22


def test_duplicate_cutoffs_upsert_cleanly(tmp_path):
    pit_db = tmp_path / "pit.db"
    cache = PITCache(pit_db)
    _seed_canonical_rows(cache, as_of_date="2024-04-01T00:00:00Z", siera=3.21)
    _seed_canonical_rows(cache, as_of_date="2024-04-01T00:00:00Z", siera=4.44)

    with sqlite3.connect(pit_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM pit_metric_cache").fetchone()[0]

    builder = AdvancedPitcherDailySnapshotBuilder(cache_db=pit_db)
    snapshot = builder.build_pitcher_snapshot(
        pitcher=669923,
        season=2024,
        requested_as_of_date="2024-04-01T23:59:59Z",
    )

    assert count == 2
    assert snapshot["siera"] == 4.44


def test_script_does_not_import_live_or_backtest_modules(tmp_path, monkeypatch):
    script = _load_script(monkeypatch)

    script.main(_argv(tmp_path))

    assert PIPELINE_MODULES.isdisjoint(sys.modules)
    source = inspect.getsource(script)
    assert "backtest" not in source
    assert "data_fetchers" not in source
    assert "odds_fetcher" not in source


def _seed_canonical_rows(cache, *, as_of_date, siera=3.21):
    cache.save_record(
        namespace=FanGraphsDailyPITPersistence.NAMESPACE,
        entity_id=669923,
        season=2024,
        as_of_date=as_of_date,
        source=FanGraphsDailyPITPersistence.SOURCE,
        source_fingerprint=f"fg-{as_of_date}",
        data={
            "siera": siera,
            "xfip": 3.44,
            "fip": 3.60,
            "k_pct": 28.5,
            "bb_pct": 7.1,
            "ip": 12.2,
            "player_name": "Example Pitcher",
            "mlbam_id": 669923,
        },
        fetched_at="2026-06-13T12:00:00Z",
    )
    cache.save_record(
        namespace=SavantRollingPITPersistence.NAMESPACE,
        entity_id=669923,
        season=2024,
        as_of_date=as_of_date,
        source=SavantRollingPITPersistence.SOURCE,
        source_fingerprint=f"sv-{as_of_date}",
        data={
            "est_woba": 0.300,
            "brl_percent": 6.5,
            "ev95percent": 34.2,
            "pa": 50,
            "bip": 31,
        },
        fetched_at="2026-06-13T12:00:00Z",
    )
