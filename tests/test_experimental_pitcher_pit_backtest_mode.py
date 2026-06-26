import inspect

from backtest_and_retrain import (
    LEAGUE_AVG_ERA,
    _experimental_pitcher_pit_cutoff_for_row,
    _prediction_cutoff_for_row,
    apply_experimental_pitcher_pit_mode,
    build_game_data,
)
from modules.baseball_module.advanced_pit_enrichment import (
    AdvancedPitcherDailySnapshotBuilder,
    FanGraphsDailyPITPersistence,
    PITCache,
    SavantRollingPITPersistence,
    adapt_unified_pitcher_snapshot,
)
from modules.baseball_module.context_engine.pitcher_engine import adjust_for_pitchers


class _FakeAPI:
    def __init__(self):
        self.pitcher_full_fallback_calls = []
        self.pitcher_game_log_calls = []

    def get_team_offensive_stats(self, team_id, season):
        return {"woba": 0.330, "ops": 0.740, "wrc_plus": 105}

    def get_team_pitching_stats(self, team_id, season):
        return {
            "team_era": 4.20,
            "team_whip": 1.25,
            "runs_allowed_per_game": 4.4,
            "der": 0.700,
            "bip": 4000,
        }

    def get_bullpen_era(self, team_id, season):
        return {"era": 4.00}

    def get_pitcher_stats_full_fallback(self, pitcher_id, season, **kwargs):
        self.pitcher_full_fallback_calls.append(
            {"pitcher_id": pitcher_id, "season": season, **kwargs}
        )
        return (
            {
                "name": f"Pitcher {pitcher_id}",
                "era": 4.50,
                "fip": 4.40,
                "whip": 1.30,
                "days_rest": 5,
                "last_pitch_count": 91,
            },
            "mlb_current",
        )

    def get_pitcher_game_log(self, pitcher_id, season, **kwargs):
        self.pitcher_game_log_calls.append(
            {"pitcher_id": pitcher_id, "season": season, **kwargs}
        )
        return {"era_last_5": 3.90, "quality_start_pct": 0.55}

    def get_pitcher_f5_stats(self, pitcher_id, season):
        return {"f5_era": 3.80}


class _FakeIntegrator:
    def _fetch_team_rpg(self, team_id, season):
        return 4.5

    def get_team_lambda(self, team_name, rpg, team_id=None):
        return rpg


def _seed_daily_pit(cache, *, entity_id, as_of_date, siera=3.21, est_woba=0.301):
    cache.save_record(
        namespace=FanGraphsDailyPITPersistence.NAMESPACE,
        entity_id=entity_id,
        season=2024,
        as_of_date=as_of_date,
        source=FanGraphsDailyPITPersistence.SOURCE,
        source_fingerprint=f"fg-{entity_id}-{as_of_date}",
        data={
            "siera": siera,
            "xfip": 3.44,
            "xera": 3.12,
            "fip": 3.60,
            "k_pct": 28.5,
            "bb_pct": 7.1,
            "ip": 12.2,
            "player_name": f"Pitcher {entity_id}",
            "fg_playerid": "12345",
            "mlbam_id": entity_id,
            "source_window_start_date": "2024-03-20",
            "source_window_end_date": as_of_date[:10],
            "metric_version": "fangraphs_daily_cutoff_v1",
        },
        fetched_at="2026-06-13T12:00:00Z",
    )
    cache.save_record(
        namespace=SavantRollingPITPersistence.NAMESPACE,
        entity_id=entity_id,
        season=2024,
        as_of_date=as_of_date,
        source=SavantRollingPITPersistence.SOURCE,
        source_fingerprint=f"sv-{entity_id}-{as_of_date}",
        data={
            "est_woba": est_woba,
            "woba": 0.290,
            "brl_percent": 6.5,
            "barrel_count": 2,
            "batted_ball_count": 31,
            "ev95percent": 34.2,
            "sweet_spot_pct": 38.7,
            "pa": 50,
            "bip": 31,
            "source_window_start_date": "2024-04-01",
            "source_window_end_date": as_of_date[:10],
            "metric_version": "savant_rolling_v1",
        },
        fetched_at="2026-06-13T12:00:00Z",
    )


def _base_game_data():
    return {
        "pitcher_home": {
            "name": "Home fallback",
            "era": 4.50,
            "fip": 4.40,
            "days_rest": 5,
            "last_pitch_count": 91,
        },
        "pitcher_away": {
            "name": "Away fallback",
            "era": 4.80,
            "fip": 4.70,
            "days_rest": 4,
            "last_pitch_count": 88,
        },
    }


def test_flag_off_build_game_data_keeps_existing_full_season_enrichment_behavior():
    api = _FakeAPI()
    game_data, _, _ = build_game_data(
        1,
        "2024-04-02",
        "Boston Red Sox",
        "New York Yankees",
        2024,
        100,
        200,
        api,
        _FakeIntegrator(),
        None,
        savant_stats={100: {"est_woba": 0.301, "brl_percent": 6.5}},
        fg_stats={100: {"siera": 3.21, "xfip": 3.44, "fip": 3.60, "k_pct": 0.285, "bb_pct": 0.071, "ip": 12.2}},
    )

    home = game_data["pitcher_home"]
    assert home["siera"] == 3.21
    assert home["xfip"] == 3.44
    assert home["est_woba"] == 0.301
    assert home["brl_percent"] == 6.5
    assert home["days_rest"] == 5
    assert api.pitcher_game_log_calls == [
        {"pitcher_id": 100, "season": 2024},
        {"pitcher_id": 200, "season": 2024},
    ]
    assert [c["pitcher_id"] for c in api.pitcher_full_fallback_calls] == [100, 200]


def test_experimental_game_log_fallback_uses_prediction_cutoff():
    api = _FakeAPI()

    build_game_data(
        1,
        "2024-04-02",
        "Boston Red Sox",
        "New York Yankees",
        2024,
        100,
        200,
        api,
        _FakeIntegrator(),
        None,
        pitcher_game_log_as_of_date="2024-04-01T18:00:00Z",
    )

    assert api.pitcher_game_log_calls == [
        {"pitcher_id": 100, "season": 2024, "as_of_date": "2024-04-01T18:00:00Z"},
        {"pitcher_id": 200, "season": 2024, "as_of_date": "2024-04-01T18:00:00Z"},
    ]


def test_experimental_mode_blocks_full_season_pitcher_fallback():
    api = _FakeAPI()

    game_data, _, _ = build_game_data(
        1,
        "2024-04-02",
        "Boston Red Sox",
        "New York Yankees",
        2024,
        100,
        200,
        api,
        _FakeIntegrator(),
        None,
        pitcher_game_log_as_of_date="2024-04-01T18:00:00Z",
        use_pitcher_full_season_fallback=False,
    )

    assert api.pitcher_full_fallback_calls == []
    assert api.pitcher_game_log_calls == []
    assert game_data["pitcher_home"]["era"] == LEAGUE_AVG_ERA
    assert game_data["pitcher_away"]["era"] == LEAGUE_AVG_ERA


def test_flag_on_uses_pit_snapshot_when_available(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_daily_pit(cache, entity_id=100, as_of_date="2024-04-02T00:00:00Z", siera=2.95)
    game_data = _base_game_data()

    meta = apply_experimental_pitcher_pit_mode(
        game_data=game_data,
        home_pitcher_id=100,
        away_pitcher_id=None,
        season=2024,
        requested_as_of_date="2024-04-02T23:59:59Z",
        snapshot_builder=AdvancedPitcherDailySnapshotBuilder(cache),
        adapter=adapt_unified_pitcher_snapshot,
    )

    assert meta["home"]["pit_found"] is True
    assert meta["home_pitcher_pit_found"] is True
    assert meta["away_pitcher_pit_found"] is False
    assert meta["home_fangraphs_as_of_date"] == "2024-04-02T00:00:00+00:00"
    assert meta["home_savant_as_of_date"] == "2024-04-02T00:00:00+00:00"
    assert meta["home"]["fangraphs_as_of_date"] == "2024-04-02T00:00:00+00:00"
    assert game_data["pitcher_home"]["siera"] == 2.95
    assert game_data["pitcher_home"]["est_woba"] == 0.301
    assert game_data["pitcher_home"]["days_rest"] == 5


def test_flag_on_does_not_use_future_pit_snapshot(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_daily_pit(cache, entity_id=100, as_of_date="2024-04-03T00:00:00Z", siera=1.00)
    game_data = _base_game_data()

    meta = apply_experimental_pitcher_pit_mode(
        game_data=game_data,
        home_pitcher_id=100,
        away_pitcher_id=None,
        season=2024,
        requested_as_of_date="2024-04-02T23:59:59Z",
        snapshot_builder=AdvancedPitcherDailySnapshotBuilder(cache),
        adapter=adapt_unified_pitcher_snapshot,
    )

    assert meta["home"]["pit_found"] is False
    assert game_data["pitcher_home"]["siera"] is None if "siera" in game_data["pitcher_home"] else True


def test_missing_pit_snapshot_falls_back_safely(tmp_path):
    game_data = _base_game_data()

    meta = apply_experimental_pitcher_pit_mode(
        game_data=game_data,
        home_pitcher_id=100,
        away_pitcher_id=200,
        season=2024,
        requested_as_of_date="2024-04-02T23:59:59Z",
        snapshot_builder=AdvancedPitcherDailySnapshotBuilder(cache_db=tmp_path / "pit.db"),
        adapter=adapt_unified_pitcher_snapshot,
    )

    assert meta["home"]["pit_found"] is False
    assert meta["away"]["pit_found"] is False
    assert game_data["pitcher_home"]["era"] == 4.50
    assert game_data["pitcher_away"]["days_rest"] == 4


def test_adapted_pitcher_dict_reaches_adjust_for_pitchers_without_keyerror(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_daily_pit(cache, entity_id=100, as_of_date="2024-04-02T00:00:00Z", siera=2.95)
    _seed_daily_pit(cache, entity_id=200, as_of_date="2024-04-02T00:00:00Z", siera=4.95)
    game_data = _base_game_data()

    apply_experimental_pitcher_pit_mode(
        game_data=game_data,
        home_pitcher_id=100,
        away_pitcher_id=200,
        season=2024,
        requested_as_of_date="2024-04-02T23:59:59Z",
        snapshot_builder=AdvancedPitcherDailySnapshotBuilder(cache),
        adapter=adapt_unified_pitcher_snapshot,
    )

    lh, la, metadata = adjust_for_pitchers(4.5, 4.2, game_data)

    assert lh > 0
    assert la > 0
    assert metadata["pitcher_home"]["quality_mult"] > 0
    assert metadata["pitcher_away"]["quality_mult"] > 0


def test_prediction_cutoff_prefers_explicit_cutoff():
    assert (
        _prediction_cutoff_for_row(
            {"game_date": "2024-04-02", "prediction_cutoff_utc": "2024-04-01T18:00:00Z"}
        )
        == "2024-04-01T18:00:00Z"
    )
    assert _prediction_cutoff_for_row({"game_date": "2024-04-02"}) == "2024-04-02T23:59:59Z"


def test_experimental_pitcher_pit_cutoff_defaults_to_previous_day():
    assert (
        _experimental_pitcher_pit_cutoff_for_row({"game_date": "2024-04-02"})
        == "2024-04-01T23:59:59Z"
    )
    assert (
        _experimental_pitcher_pit_cutoff_for_row(
            {"game_date": "2024-04-02", "prediction_cutoff_utc": "2024-04-01T18:00:00Z"}
        )
        == "2024-04-01T18:00:00Z"
    )


def test_backtest_module_does_not_import_live_app():
    import backtest_and_retrain

    source = inspect.getsource(backtest_and_retrain)
    assert "import app" not in source
    assert "from app" not in source
