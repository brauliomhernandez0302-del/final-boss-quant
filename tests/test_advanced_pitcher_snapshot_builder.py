from modules.baseball_module.advanced_pit_enrichment import AdvancedPitcherSnapshotBuilder
from modules.baseball_module.context_engine.pitcher_engine import PitcherEngine
from modules.baseball_module.advanced_pit_enrichment.fangraphs_pit_fetcher import (
    FanGraphsPITFetcher,
)
from modules.baseball_module.advanced_pit_enrichment.pit_cache import PITCache
from modules.baseball_module.advanced_pit_enrichment.savant_pit_fetcher import (
    SavantPITFetcher,
)


def test_pitcher_snapshot_builder_merges_base_fangraphs_and_savant_pit(tmp_path):
    db_path = tmp_path / "pit.db"
    cache = PITCache(db_path)
    cache.save_record(
        namespace=FanGraphsPITFetcher.PITCHER_NAMESPACE,
        entity_id=605400,
        season=2025,
        as_of_date="2025-04-05T00:00:00Z",
        source=FanGraphsPITFetcher.SOURCE,
        source_fingerprint="fg-window",
        data={
            "fg_name": "Nola, Aaron",
            "siera": 3.21,
            "xfip": 3.44,
            "fip": 3.60,
            "era": 3.80,
            "ip": 12.2,
            "k_pct": 28.5,
            "bb_pct": 7.1,
            "swstr_pct": 13.2,
        },
        fetched_at="2026-06-06T12:00:00Z",
    )
    cache.save_record(
        namespace=SavantPITFetcher.PITCHER_NAMESPACE,
        entity_id=605400,
        season=2025,
        as_of_date="2025-04-05T00:00:00Z",
        source=SavantPITFetcher.SOURCE,
        source_fingerprint="savant-window",
        data={
            "player_name": "Nola, Aaron",
            "est_woba": 0.301,
            "brl_percent": 6.5,
            "avg_hit_speed": 88.1,
            "ev95percent": 34.2,
            "pitches": 97,
            "pa": 27,
            "bip": 20,
        },
        fetched_at="2026-06-06T12:01:00Z",
    )

    builder = AdvancedPitcherSnapshotBuilder(db_path)
    snapshot = builder.build_pitcher_snapshot(
        mlbam_id=605400,
        season=2025,
        as_of_date="2025-04-05T12:00:00Z",
        base_stats={
            "name": "Aaron Nola",
            "era": 4.10,
            "fip": 4.00,
            "whip": 1.20,
            "days_rest": 5,
            "last_pitch_count": 92,
        },
    )

    assert snapshot["name"] == "Aaron Nola"
    assert snapshot["mlbam_id"] == "605400"
    assert snapshot["era"] == 4.10
    assert snapshot["fip"] == 4.00
    assert snapshot["siera"] == 3.21
    assert snapshot["xfip"] == 3.44
    assert snapshot["innings_pitched"] == 12.2
    assert snapshot["k_pct"] == 0.285
    assert snapshot["bb_pct"] == 0.071
    assert snapshot["swstr_pct"] == 0.132
    assert snapshot["est_woba"] == 0.301
    assert snapshot["brl_percent"] == 6.5
    assert snapshot["savant_pitches"] == 97
    assert snapshot["pit_metadata"]["fangraphs"]["found"] is True
    assert snapshot["pit_metadata"]["savant"]["found"] is True

    quality_mult = PitcherEngine()._adjust_pitcher_quality(snapshot)
    assert 0.70 <= quality_mult <= 1.35


def test_pitcher_snapshot_builder_never_uses_future_pit_records(tmp_path):
    db_path = tmp_path / "pit.db"
    cache = PITCache(db_path)
    cache.save_record(
        namespace=FanGraphsPITFetcher.PITCHER_NAMESPACE,
        entity_id=1,
        season=2025,
        as_of_date="2025-04-10T00:00:00Z",
        source=FanGraphsPITFetcher.SOURCE,
        source_fingerprint="future",
        data={"siera": 1.23},
        fetched_at="2026-06-06T12:00:00Z",
    )

    snapshot = AdvancedPitcherSnapshotBuilder(db_path).build_pitcher_snapshot(
        mlbam_id=1,
        season=2025,
        as_of_date="2025-04-05T00:00:00Z",
        base_stats={"era": 4.00},
    )

    assert "siera" not in snapshot
    assert snapshot["pit_metadata"]["fangraphs"]["found"] is False


def test_pitcher_snapshot_builder_builds_home_and_away_game_snapshots(tmp_path):
    db_path = tmp_path / "pit.db"
    cache = PITCache(db_path)
    cache.save_record(
        namespace=FanGraphsPITFetcher.PITCHER_NAMESPACE,
        entity_id="home",
        season=2025,
        as_of_date="2025-04-05T00:00:00Z",
        source=FanGraphsPITFetcher.SOURCE,
        source_fingerprint="fg-home",
        data={"siera": 3.0},
        fetched_at="2026-06-06T12:00:00Z",
    )
    cache.save_record(
        namespace=SavantPITFetcher.PITCHER_NAMESPACE,
        entity_id="away",
        season=2025,
        as_of_date="2025-04-05T00:00:00Z",
        source=SavantPITFetcher.SOURCE,
        source_fingerprint="sv-away",
        data={"est_woba": 0.290},
        fetched_at="2026-06-06T12:00:00Z",
    )

    snapshots = AdvancedPitcherSnapshotBuilder(db_path).build_for_game(
        season=2025,
        as_of_date="2025-04-05T00:00:00Z",
        home_pitcher_id="home",
        away_pitcher_id="away",
        home_pitcher_name="Home Starter",
        away_pitcher_name="Away Starter",
        home_base_stats={"era": 4.0},
        away_base_stats={"era": 5.0},
    )

    assert snapshots["pitcher_home"]["name"] == "Home Starter"
    assert snapshots["pitcher_home"]["siera"] == 3.0
    assert snapshots["pitcher_away"]["name"] == "Away Starter"
    assert snapshots["pitcher_away"]["est_woba"] == 0.290
