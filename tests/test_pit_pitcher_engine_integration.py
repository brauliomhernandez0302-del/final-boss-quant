import pytest

from modules.baseball_module.advanced_pit_enrichment import AdvancedPitcherSnapshotBuilder
from modules.baseball_module.advanced_pit_enrichment.fangraphs_pit_fetcher import (
    FanGraphsPITFetcher,
)
from modules.baseball_module.advanced_pit_enrichment.pit_cache import PITCache
from modules.baseball_module.advanced_pit_enrichment.savant_pit_fetcher import (
    SavantPITFetcher,
)
from modules.baseball_module.context_engine.pitcher_engine import adjust_for_pitchers


def _seed_fangraphs(cache, *, entity_id, siera, xfip, fip, era, ip, k_pct, bb_pct):
    cache.save_record(
        namespace=FanGraphsPITFetcher.PITCHER_NAMESPACE,
        entity_id=entity_id,
        season=2025,
        as_of_date="2025-04-05T00:00:00Z",
        source=FanGraphsPITFetcher.SOURCE,
        source_fingerprint=f"fg-{entity_id}",
        data={
            "siera": siera,
            "xfip": xfip,
            "fip": fip,
            "era": era,
            "ip": ip,
            "k_pct": k_pct,
            "bb_pct": bb_pct,
        },
        fetched_at="2026-06-06T12:00:00Z",
    )


def _seed_savant(cache, *, entity_id, est_woba, brl_percent):
    cache.save_record(
        namespace=SavantPITFetcher.PITCHER_NAMESPACE,
        entity_id=entity_id,
        season=2025,
        as_of_date="2025-04-05T00:00:00Z",
        source=SavantPITFetcher.SOURCE,
        source_fingerprint=f"savant-{entity_id}",
        data={
            "est_woba": est_woba,
            "brl_percent": brl_percent,
            "avg_hit_speed": 88.0,
            "ev95percent": 32.0,
            "pitches": 90,
            "pa": 25,
            "bip": 18,
        },
        fetched_at="2026-06-06T12:01:00Z",
    )


def _game_data(pitcher_home, pitcher_away):
    return {
        "pitcher_home": pitcher_home,
        "pitcher_away": pitcher_away,
        "home_team": {"name": "Home"},
        "away_team": {"name": "Away"},
        "park": {"name": "Neutral Park"},
        "home_lineup_lhb_pct": 0.45,
        "away_lineup_lhb_pct": 0.45,
    }


def test_pit_snapshots_feed_pitcher_engine_with_expected_directionality(tmp_path):
    db_path = tmp_path / "pit.db"
    cache = PITCache(db_path)

    _seed_fangraphs(
        cache,
        entity_id="home_bad",
        siera=5.10,
        xfip=5.00,
        fip=5.20,
        era=5.40,
        ip=120.0,
        k_pct=16.0,
        bb_pct=10.0,
    )
    _seed_savant(cache, entity_id="home_bad", est_woba=0.355, brl_percent=13.0)

    _seed_fangraphs(
        cache,
        entity_id="away_elite",
        siera=2.70,
        xfip=2.80,
        fip=2.90,
        era=2.60,
        ip=130.0,
        k_pct=32.0,
        bb_pct=5.0,
    )
    _seed_savant(cache, entity_id="away_elite", est_woba=0.275, brl_percent=4.0)

    snapshots = AdvancedPitcherSnapshotBuilder(db_path).build_for_game(
        season=2025,
        as_of_date="2025-04-05T12:00:00Z",
        home_pitcher_id="home_bad",
        away_pitcher_id="away_elite",
        home_pitcher_name="Home Bad",
        away_pitcher_name="Away Elite",
        home_base_stats={"whip": 1.35, "days_rest": 4, "last_pitch_count": 90, "era_last_5": 5.20},
        away_base_stats={"whip": 1.05, "days_rest": 5, "last_pitch_count": 88, "era_last_5": 2.70},
    )

    lh, la, metadata = adjust_for_pitchers(4.50, 4.50, _game_data(**snapshots))

    assert lh < 4.50
    assert la > 4.50
    assert metadata["pitcher_away"]["quality_mult"] < 1.0
    assert metadata["pitcher_home"]["quality_mult"] > 1.0
    assert snapshots["pitcher_away"]["pit_metadata"]["fangraphs"]["found"] is True
    assert snapshots["pitcher_home"]["pit_metadata"]["savant"]["found"] is True


def test_pit_pitcher_engine_integration_is_neutral_without_future_cache(tmp_path):
    db_path = tmp_path / "pit.db"
    cache = PITCache(db_path)
    _seed_fangraphs(
        cache,
        entity_id="future_only",
        siera=1.00,
        xfip=1.00,
        fip=1.00,
        era=1.00,
        ip=100.0,
        k_pct=40.0,
        bb_pct=2.0,
    )
    cache.save_record(
        namespace=FanGraphsPITFetcher.PITCHER_NAMESPACE,
        entity_id="future_only",
        season=2025,
        as_of_date="2025-04-10T00:00:00Z",
        source=FanGraphsPITFetcher.SOURCE,
        source_fingerprint="future",
        data={"siera": 1.00, "xfip": 1.00, "ip": 100.0},
        fetched_at="2026-06-06T12:00:00Z",
    )

    snapshots = AdvancedPitcherSnapshotBuilder(db_path).build_for_game(
        season=2025,
        as_of_date="2025-04-04T00:00:00Z",
        home_pitcher_id="future_only",
        away_pitcher_id="future_only",
        home_base_stats={"era": 4.15, "fip": 4.15, "innings_pitched": 100, "days_rest": 4, "last_pitch_count": 90},
        away_base_stats={"era": 4.15, "fip": 4.15, "innings_pitched": 100, "days_rest": 4, "last_pitch_count": 90},
    )

    lh, la, metadata = adjust_for_pitchers(4.50, 4.50, _game_data(**snapshots))

    assert lh == pytest.approx(4.50, abs=0.08)
    assert la == pytest.approx(4.50, abs=0.08)
    assert metadata["pitcher_home"]["quality_mult"] == pytest.approx(1.0, abs=0.02)
    assert snapshots["pitcher_home"]["pit_metadata"]["fangraphs"]["found"] is False
