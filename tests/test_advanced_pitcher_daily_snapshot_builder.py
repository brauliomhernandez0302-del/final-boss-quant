import inspect

from modules.baseball_module.advanced_pit_enrichment import (
    AdvancedPitcherDailySnapshotBuilder,
    FanGraphsDailyPITPersistence,
    PITCache,
    PitcherPriorBaselinePersistence,
    SavantRollingPITPersistence,
)
import modules.baseball_module.advanced_pit_enrichment.advanced_pitcher_daily_snapshot_builder as snapshot_module


def _cache(tmp_path):
    return PITCache(tmp_path / "pit.db")


def _seed_fangraphs(
    cache,
    *,
    entity_id=605400,
    as_of_date="2025-04-05T00:00:00Z",
    source_fingerprint="fg-window",
    data=None,
):
    payload = {
        "siera": 3.21,
        "xfip": 3.44,
        "xera": 3.12,
        "fip": 3.60,
        "k_pct": 28.5,
        "bb_pct": 7.1,
        "ip": 12.2,
        "player_name": "Nola, Aaron",
        "fg_playerid": "12345",
        "mlbam_id": 605400,
        "source_window_start_date": "2025-03-27",
        "source_window_end_date": as_of_date[:10],
        "metric_version": "fangraphs_daily_cutoff_v1",
    }
    if data:
        payload.update(data)
    cache.save_record(
        namespace=FanGraphsDailyPITPersistence.NAMESPACE,
        entity_id=entity_id,
        season=2025,
        as_of_date=as_of_date,
        source=FanGraphsDailyPITPersistence.SOURCE,
        source_fingerprint=source_fingerprint,
        data=payload,
        fetched_at="2026-06-13T12:00:00Z",
    )


def _seed_savant(
    cache,
    *,
    entity_id=605400,
    as_of_date="2025-04-05T00:00:00Z",
    source_fingerprint="savant-window",
    data=None,
):
    payload = {
        "est_woba": 0.301,
        "woba": 0.290,
        "brl_percent": 6.5,
        "barrel_count": 2,
        "batted_ball_count": 31,
        "ev95percent": 34.2,
        "sweet_spot_pct": 38.7,
        "pa": 50,
        "bip": 31,
        "source_window_start_date": "2025-03-27",
        "source_window_end_date": as_of_date[:10],
        "metric_version": "savant_rolling_v1",
    }
    if data:
        payload.update(data)
    cache.save_record(
        namespace=SavantRollingPITPersistence.NAMESPACE,
        entity_id=entity_id,
        season=2025,
        as_of_date=as_of_date,
        source=SavantRollingPITPersistence.SOURCE,
        source_fingerprint=source_fingerprint,
        data=payload,
        fetched_at="2026-06-13T12:01:00Z",
    )


def _seed_prior(
    cache,
    *,
    entity_id=605400,
    target_season=2025,
    as_of_date="2024-09-29T23:59:59Z",
    source_fingerprint="prior-window",
    data=None,
):
    payload = {
        "prior_season": target_season - 1,
        "est_woba": 0.315,
        "woba": 0.305,
        "brl_percent": 7.2,
        "barrel_count": 20,
        "batted_ball_count": 280,
        "ev95percent": 36.5,
        "sweet_spot_pct": 34.0,
        "pa": 420,
        "bip": 280,
        "source_window_start_date": f"{target_season - 1}-03-28",
        "source_window_end_date": f"{target_season - 1}-09-29",
        "metric_version": "pitcher_prior_baseline_v1",
    }
    if data:
        payload.update(data)
    cache.save_record(
        namespace=PitcherPriorBaselinePersistence.NAMESPACE,
        entity_id=entity_id,
        season=target_season,
        as_of_date=as_of_date,
        source=PitcherPriorBaselinePersistence.SOURCE,
        source_fingerprint=source_fingerprint,
        data=payload,
        fetched_at="2026-06-26T12:00:00Z",
    )


def test_merge_both_sources(tmp_path):
    cache = _cache(tmp_path)
    _seed_fangraphs(cache)
    _seed_savant(cache)

    snapshot = AdvancedPitcherDailySnapshotBuilder(cache).build_pitcher_snapshot(
        pitcher=605400,
        season=2025,
        requested_as_of_date="2025-04-05T12:00:00Z",
    )

    assert snapshot["found"] is True
    assert snapshot["fangraphs_found"] is True
    assert snapshot["savant_found"] is True
    assert snapshot["mlbam_id"] == 605400
    assert snapshot["player_name"] == "Nola, Aaron"
    assert snapshot["fg_playerid"] == "12345"
    assert snapshot["siera"] == 3.21
    assert snapshot["xfip"] == 3.44
    assert snapshot["xera"] == 3.12
    assert snapshot["fip"] == 3.60
    assert snapshot["k_pct"] == 0.285
    assert snapshot["bb_pct"] == 0.071
    assert snapshot["innings_pitched"] == 12.2
    assert snapshot["ip"] == 12.2
    assert snapshot["est_woba"] == 0.301
    assert snapshot["woba"] == 0.290
    assert snapshot["brl_percent"] == 6.5
    assert snapshot["barrel_count"] == 2
    assert snapshot["batted_ball_count"] == 31
    assert snapshot["ev95percent"] == 34.2
    assert snapshot["sweet_spot_pct"] == 38.7
    assert snapshot["pa"] == 50
    assert snapshot["bip"] == 31
    assert snapshot["savant_pa"] == 50
    assert snapshot["savant_bip"] == 31


def test_current_pit_takes_priority_over_prior_baseline(tmp_path):
    cache = _cache(tmp_path)
    _seed_savant(cache, data={"est_woba": 0.299})
    _seed_prior(cache, data={"est_woba": 0.399})

    snapshot = AdvancedPitcherDailySnapshotBuilder(cache).build_pitcher_snapshot(
        pitcher=605400,
        season=2025,
        requested_as_of_date="2025-04-05T12:00:00Z",
    )

    assert snapshot["provenance_source"] == "current_pit"
    assert snapshot["prior_baseline_found"] is False
    assert snapshot["est_woba"] == 0.299


def test_prior_baseline_used_only_when_current_pit_is_missing(tmp_path):
    cache = _cache(tmp_path)
    _seed_prior(cache)

    snapshot = AdvancedPitcherDailySnapshotBuilder(cache).build_pitcher_snapshot(
        pitcher=605400,
        season=2025,
        requested_as_of_date="2025-04-05T12:00:00Z",
    )

    assert snapshot["found"] is True
    assert snapshot["fangraphs_found"] is False
    assert snapshot["savant_found"] is False
    assert snapshot["prior_baseline_found"] is True
    assert snapshot["provenance_source"] == "prior_season_baseline"
    assert snapshot["prior_season"] == 2024
    assert snapshot["est_woba"] == 0.315
    assert snapshot["source_fingerprints"]["prior_season_baseline"] == "prior-window"


def test_fangraphs_only(tmp_path):
    cache = _cache(tmp_path)
    _seed_fangraphs(cache)

    snapshot = AdvancedPitcherDailySnapshotBuilder(cache).build_pitcher_snapshot(
        pitcher=605400,
        season=2025,
        requested_as_of_date="2025-04-05T12:00:00Z",
    )

    assert snapshot["found"] is True
    assert snapshot["fangraphs_found"] is True
    assert snapshot["savant_found"] is False
    assert snapshot["siera"] == 3.21
    assert "est_woba" not in snapshot


def test_savant_only(tmp_path):
    cache = _cache(tmp_path)
    _seed_savant(cache)

    snapshot = AdvancedPitcherDailySnapshotBuilder(cache).build_pitcher_snapshot(
        pitcher=605400,
        season=2025,
        requested_as_of_date="2025-04-05T12:00:00Z",
    )

    assert snapshot["found"] is True
    assert snapshot["fangraphs_found"] is False
    assert snapshot["savant_found"] is True
    assert snapshot["est_woba"] == 0.301
    assert "siera" not in snapshot


def test_both_missing_returns_found_false(tmp_path):
    snapshot = AdvancedPitcherDailySnapshotBuilder(_cache(tmp_path)).build_pitcher_snapshot(
        pitcher=605400,
        season=2025,
        requested_as_of_date="2025-04-05T12:00:00Z",
    )

    assert snapshot["found"] is False
    assert snapshot["fangraphs_found"] is False
    assert snapshot["savant_found"] is False
    assert snapshot["fangraphs_as_of_date"] is None
    assert snapshot["savant_as_of_date"] is None
    assert snapshot["prior_baseline_found"] is False
    assert snapshot["provenance_source"] == "league_average_safe_fallback"


def test_latest_lte_requested_as_of_date(tmp_path):
    cache = _cache(tmp_path)
    _seed_fangraphs(cache, as_of_date="2025-04-01T00:00:00Z", data={"siera": 4.10})
    _seed_fangraphs(cache, as_of_date="2025-04-03T00:00:00Z", data={"siera": 3.50})
    _seed_savant(cache, as_of_date="2025-04-01T00:00:00Z", data={"est_woba": 0.330})
    _seed_savant(cache, as_of_date="2025-04-03T00:00:00Z", data={"est_woba": 0.300})

    snapshot = AdvancedPitcherDailySnapshotBuilder(cache).build_pitcher_snapshot(
        pitcher=605400,
        season=2025,
        requested_as_of_date="2025-04-03T12:00:00Z",
    )

    assert snapshot["fangraphs_as_of_date"] == "2025-04-03T00:00:00+00:00"
    assert snapshot["savant_as_of_date"] == "2025-04-03T00:00:00+00:00"
    assert snapshot["siera"] == 3.50
    assert snapshot["est_woba"] == 0.300


def test_does_not_use_future_source_rows(tmp_path):
    cache = _cache(tmp_path)
    _seed_fangraphs(cache, as_of_date="2025-04-01T00:00:00Z", data={"siera": 4.10})
    _seed_fangraphs(cache, as_of_date="2025-04-03T00:00:00Z", data={"siera": 1.00})
    _seed_savant(cache, as_of_date="2025-04-01T00:00:00Z", data={"est_woba": 0.330})
    _seed_savant(cache, as_of_date="2025-04-03T00:00:00Z", data={"est_woba": 0.100})

    snapshot = AdvancedPitcherDailySnapshotBuilder(cache).build_pitcher_snapshot(
        pitcher=605400,
        season=2025,
        requested_as_of_date="2025-04-02T23:59:59Z",
    )

    assert snapshot["fangraphs_as_of_date"] == "2025-04-01T00:00:00+00:00"
    assert snapshot["savant_as_of_date"] == "2025-04-01T00:00:00+00:00"
    assert snapshot["siera"] == 4.10
    assert snapshot["est_woba"] == 0.330


def test_preserves_none_values(tmp_path):
    cache = _cache(tmp_path)
    _seed_fangraphs(cache, data={"siera": None, "xera": None, "k_pct": None})
    _seed_savant(cache, data={"est_woba": None, "brl_percent": None, "ev95percent": None})

    snapshot = AdvancedPitcherDailySnapshotBuilder(cache).build_pitcher_snapshot(
        pitcher=605400,
        season=2025,
        requested_as_of_date="2025-04-05T12:00:00Z",
    )

    assert snapshot["siera"] is None
    assert snapshot["xera"] is None
    assert snapshot["k_pct"] is None
    assert snapshot["est_woba"] is None
    assert snapshot["brl_percent"] is None
    assert snapshot["ev95percent"] is None


def test_source_provenance_correct(tmp_path):
    cache = _cache(tmp_path)
    _seed_fangraphs(cache, as_of_date="2025-04-01T00:00:00Z", source_fingerprint="fg-old")
    _seed_savant(cache, as_of_date="2025-04-02T00:00:00Z", source_fingerprint="savant-current")

    snapshot = AdvancedPitcherDailySnapshotBuilder(cache).build_pitcher_snapshot(
        pitcher=605400,
        season=2025,
        requested_as_of_date="2025-04-02T23:59:59Z",
    )

    assert snapshot["requested_as_of_date"] == "2025-04-02T23:59:59Z"
    assert snapshot["fangraphs_as_of_date"] == "2025-04-01T00:00:00+00:00"
    assert snapshot["savant_as_of_date"] == "2025-04-02T00:00:00+00:00"
    assert snapshot["source_fingerprints"] == {
        "fangraphs": "fg-old",
        "savant": "savant-current",
        "prior_season_baseline": None,
    }
    assert snapshot["snapshot_version"] == "advanced_pitcher_daily_snapshot_v1"


def test_output_keys_match_current_pitcher_engine_expected_advanced_fields(tmp_path):
    cache = _cache(tmp_path)
    _seed_fangraphs(cache)
    _seed_savant(cache)

    snapshot = AdvancedPitcherDailySnapshotBuilder(cache).build_pitcher_snapshot(
        pitcher=605400,
        season=2025,
        requested_as_of_date="2025-04-05T12:00:00Z",
    )

    expected_keys = {
        "siera",
        "xfip",
        "xera",
        "fip",
        "k_pct",
        "bb_pct",
        "innings_pitched",
        "ip",
        "player_name",
        "fg_playerid",
        "mlbam_id",
        "est_woba",
        "woba",
        "brl_percent",
        "barrel_count",
        "batted_ball_count",
        "ev95percent",
        "sweet_spot_pct",
        "pa",
        "bip",
        "savant_pa",
        "savant_bip",
    }

    assert expected_keys.issubset(snapshot.keys())


def test_snapshot_module_does_not_import_live_backtest_or_model_code():
    source = inspect.getsource(snapshot_module)

    assert "backtest" not in source
    assert "model" not in source
    assert "data_fetchers" not in source
    assert "odds_fetcher" not in source
