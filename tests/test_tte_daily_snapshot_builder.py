import inspect

from modules.baseball_module.advanced_pit_enrichment import (
    PITCache,
    TTEDailySnapshotBuilder,
    TTEPITNamespaces,
    TTEPITSources,
    previous_day_cutoff_for_game_date,
)


def test_previous_day_cutoff_works():
    assert (
        previous_day_cutoff_for_game_date("2024-04-09")
        == "2024-04-08T23:59:59Z"
    )


def test_build_for_game_uses_previous_day_not_same_day(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_rolling(cache, as_of_date="2024-04-08T00:00:00Z", team_woba=0.321)
    _seed_team_rolling(cache, as_of_date="2024-04-09T00:00:00Z", team_woba=0.999)

    snapshot = TTEDailySnapshotBuilder(cache).build_for_game(
        team_id=147,
        team_name="New York Yankees",
        season=2024,
        game_date="2024-04-09",
    )

    assert snapshot["found"] is True
    assert snapshot["requested_as_of_date"] == "2024-04-08T23:59:59Z"
    assert snapshot["team_offense_as_of_date"] == "2024-04-08T00:00:00+00:00"
    assert snapshot["team_woba"] == 0.321
    assert snapshot["lambda_offense"] is None


def test_missing_team_snapshot_returns_found_false(tmp_path):
    snapshot = TTEDailySnapshotBuilder(cache_db=tmp_path / "pit.db").build_team_snapshot(
        team_id=147,
        team_name="New York Yankees",
        season=2024,
        requested_as_of_date="2024-04-08T23:59:59Z",
    )

    assert snapshot == {
        "found": False,
        "team_id": 147,
        "team_name": "New York Yankees",
        "season": 2024,
        "prior_season": None,
        "requested_as_of_date": "2024-04-08T23:59:59Z",
        "team_offense_as_of_date": None,
        "prior_baseline_as_of_date": None,
        "batter_rolling_found": False,
        "team_rolling_found": False,
        "prior_baseline_found": False,
        "lambda_offense": None,
        "runs_per_game": None,
        "team_est_woba": None,
        "team_woba": None,
        "team_brl_percent": None,
        "team_ev95percent": None,
        "barrel_pa": None,
        "bb_pct": None,
        "k_pct": None,
        "pa": None,
        "bip": None,
        "team_est_woba_prior": None,
        "team_woba_prior": None,
        "bb_pct_prior": None,
        "k_pct_prior": None,
        "barrel_pa_prior": None,
        "brl_percent_prior": None,
        "ev95percent_prior": None,
        "pa_prior": None,
        "bip_prior": None,
        "source_fingerprints": {
            "tte_team_daily": None,
            "savant_team_offense_rolling": None,
            "savant_batter_rolling": None,
            "savant_team_offense_prior_baseline": None,
        },
        "snapshot_version": "tte_daily_snapshot_v1",
    }


def test_tte_record_without_team_offense_returns_found_false(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_tte(cache, as_of_date="2024-04-08T00:00:00Z", lambda_offense=9.99)

    snapshot = TTEDailySnapshotBuilder(cache).build_team_snapshot(
        team_id=147,
        season=2024,
        requested_as_of_date="2024-04-08T23:59:59Z",
    )

    assert snapshot["found"] is False
    assert snapshot["team_rolling_found"] is False
    assert snapshot["team_offense_as_of_date"] is None
    assert snapshot["lambda_offense"] is None
    assert snapshot["team_woba"] is None


def test_retrieves_latest_lte_requested_as_of_date(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_rolling(cache, as_of_date="2024-04-07T00:00:00Z", team_woba=0.310)
    _seed_team_rolling(cache, as_of_date="2024-04-08T00:00:00Z", team_woba=0.330)
    _seed_team_rolling(cache, as_of_date="2024-04-09T00:00:00Z", team_woba=0.990)

    snapshot = TTEDailySnapshotBuilder(cache).build_team_snapshot(
        team_id=147,
        season=2024,
        requested_as_of_date="2024-04-08T23:59:59Z",
    )

    assert snapshot["found"] is True
    assert snapshot["team_offense_as_of_date"] == "2024-04-08T00:00:00+00:00"
    assert snapshot["team_woba"] == 0.330


def test_does_not_retrieve_future_snapshot(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_rolling(cache, as_of_date="2024-04-09T00:00:00Z", team_woba=0.990)

    snapshot = TTEDailySnapshotBuilder(cache).build_team_snapshot(
        team_id=147,
        team_name="New York Yankees",
        season=2024,
        requested_as_of_date="2024-04-08T23:59:59Z",
    )

    assert snapshot["found"] is False
    assert snapshot["team_rolling_found"] is False
    assert snapshot["team_offense_as_of_date"] is None
    assert snapshot["team_woba"] is None


def test_existing_team_offense_returns_found_true(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_rolling(
        cache,
        as_of_date="2024-04-08T00:00:00Z",
        team_name="New York Yankees",
        runs_per_game=5.1,
        team_est_woba=0.341,
        team_woba=0.330,
        team_brl_percent=9.2,
        team_ev95percent=41.5,
        pa=321,
        bip=211,
    )

    snapshot = TTEDailySnapshotBuilder(cache).build_team_snapshot(
        team_id="147",
        season=2024,
        requested_as_of_date="2024-04-08T23:59:59Z",
    )

    assert snapshot["found"] is True
    assert snapshot["team_rolling_found"] is True
    assert snapshot["team_id"] == 147
    assert snapshot["team_name"] == "New York Yankees"
    assert snapshot["lambda_offense"] is None
    assert snapshot["runs_per_game"] == 5.1
    assert snapshot["team_est_woba"] == 0.341
    assert snapshot["team_woba"] == 0.330
    assert snapshot["team_brl_percent"] == 9.2
    assert snapshot["team_ev95percent"] == 41.5
    assert snapshot["barrel_pa"] == 0.091
    assert snapshot["bb_pct"] == 0.085
    assert snapshot["k_pct"] == 0.221
    assert snapshot["pa"] == 321
    assert snapshot["bip"] == 211


def test_source_provenance_preserved(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_tte(cache, as_of_date="2024-04-08T00:00:00Z", fingerprint="tte-fp")
    _seed_team_rolling(cache, as_of_date="2024-04-08T00:00:00Z", fingerprint="team-fp")
    _seed_batter_rolling(cache, as_of_date="2024-04-08T00:00:00Z", fingerprint="batter-fp")
    _seed_prior_baseline(cache, as_of_date="2023-10-01T23:59:59Z", fingerprint="prior-fp")

    snapshot = TTEDailySnapshotBuilder(cache).build_team_snapshot(
        team_id=147,
        season=2024,
        requested_as_of_date="2024-04-08T23:59:59Z",
    )

    assert snapshot["batter_rolling_found"] is True
    assert snapshot["team_rolling_found"] is True
    assert snapshot["source_fingerprints"] == {
        "tte_team_daily": "tte-fp",
        "savant_team_offense_rolling": "team-fp",
        "savant_batter_rolling": "batter-fp",
        "savant_team_offense_prior_baseline": "prior-fp",
    }


def test_none_values_preserved_without_fake_zeroes(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_rolling(
        cache,
        as_of_date="2024-04-08T00:00:00Z",
        runs_per_game=None,
        team_est_woba=None,
        team_woba=None,
        team_brl_percent=None,
        team_ev95percent=None,
        barrel_pa=None,
        bb_pct=None,
        k_pct=None,
        pa=None,
        bip=None,
    )

    snapshot = TTEDailySnapshotBuilder(cache).build_team_snapshot(
        team_id=147,
        season=2024,
        requested_as_of_date="2024-04-08T23:59:59Z",
    )

    for key in (
        "lambda_offense",
        "runs_per_game",
        "team_est_woba",
        "team_woba",
        "team_brl_percent",
        "team_ev95percent",
        "barrel_pa",
        "bb_pct",
        "k_pct",
        "pa",
        "bip",
    ):
        assert snapshot[key] is None


def test_missing_prior_baseline_does_not_fake_values(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_rolling(cache, as_of_date="2024-04-08T00:00:00Z")

    snapshot = TTEDailySnapshotBuilder(cache).build_team_snapshot(
        team_id=147,
        season=2024,
        requested_as_of_date="2024-04-08T23:59:59Z",
    )

    assert snapshot["prior_baseline_found"] is False
    assert snapshot["team_est_woba_prior"] is None
    assert snapshot["bb_pct_prior"] is None
    assert snapshot["k_pct_prior"] is None
    assert snapshot["barrel_pa_prior"] is None


def test_retrieves_prior_baseline_for_current_season(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_rolling(cache, as_of_date="2024-04-08T00:00:00Z")
    _seed_prior_baseline(
        cache,
        as_of_date="2023-10-01T23:59:59Z",
        team_est_woba_prior=0.333,
        team_woba_prior=0.321,
        bb_pct_prior=0.091,
        k_pct_prior=0.203,
        barrel_pa_prior=0.088,
        pa_prior=6100,
    )

    snapshot = TTEDailySnapshotBuilder(cache).build_team_snapshot(
        team_id=147,
        season=2024,
        requested_as_of_date="2024-04-08T23:59:59Z",
    )

    assert snapshot["prior_baseline_found"] is True
    assert snapshot["prior_baseline_as_of_date"] == "2023-10-01T23:59:59+00:00"
    assert snapshot["prior_season"] == 2023
    assert snapshot["team_est_woba_prior"] == 0.333
    assert snapshot["team_woba_prior"] == 0.321
    assert snapshot["bb_pct_prior"] == 0.091
    assert snapshot["k_pct_prior"] == 0.203
    assert snapshot["barrel_pa_prior"] == 0.088
    assert snapshot["pa_prior"] == 6100


def test_prior_baseline_zero_rates_preserved(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_rolling(cache, as_of_date="2024-04-08T00:00:00Z")
    _seed_prior_baseline(
        cache,
        as_of_date="2023-10-01T23:59:59Z",
        bb_pct_prior=0.0,
        k_pct_prior=0.0,
        barrel_pa_prior=0.0,
    )

    snapshot = TTEDailySnapshotBuilder(cache).build_team_snapshot(
        team_id=147,
        season=2024,
        requested_as_of_date="2024-04-08T23:59:59Z",
    )

    assert snapshot["bb_pct_prior"] == 0.0
    assert snapshot["k_pct_prior"] == 0.0
    assert snapshot["barrel_pa_prior"] == 0.0


def test_exposes_tte_adapter_inputs_from_team_offense(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_rolling(
        cache,
        as_of_date="2024-04-08T00:00:00Z",
        barrel_pa=0.073,
        bb_pct=0.101,
        k_pct=0.199,
    )

    snapshot = TTEDailySnapshotBuilder(cache).build_team_snapshot(
        team_id=147,
        season=2024,
        requested_as_of_date="2024-04-08T23:59:59Z",
    )

    assert snapshot["barrel_pa"] == 0.073
    assert snapshot["bb_pct"] == 0.101
    assert snapshot["k_pct"] == 0.199


def test_lambda_offense_remains_none_without_real_formula(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_tte(cache, as_of_date="2024-04-08T00:00:00Z", lambda_offense=9.99)
    _seed_team_rolling(cache, as_of_date="2024-04-08T00:00:00Z", team_woba=0.330)

    snapshot = TTEDailySnapshotBuilder(cache).build_team_snapshot(
        team_id=147,
        season=2024,
        requested_as_of_date="2024-04-08T23:59:59Z",
    )

    assert snapshot["found"] is True
    assert snapshot["lambda_offense"] is None


def test_module_does_not_import_live_app_backtest_or_run_module():
    import modules.baseball_module.advanced_pit_enrichment.tte_daily_snapshot_builder as module

    source = inspect.getsource(module)
    assert "import app" not in source
    assert "from app" not in source
    assert "backtest_and_retrain" not in source
    assert "run_module" not in source
    assert "true_talent_engine" not in source


def test_default_behavior_untouched():
    import backtest_and_retrain
    import modules.baseball_module.offense.true_talent_engine as tte

    assert hasattr(tte, "get_true_talent_lambda")
    assert "TTEDailySnapshotBuilder" not in inspect.getsource(backtest_and_retrain)


def _seed_tte(
    cache,
    *,
    as_of_date,
    team_id=147,
    fingerprint="tte-fingerprint",
    lambda_offense=4.5,
    runs_per_game=4.8,
    team_est_woba=0.320,
    team_woba=0.318,
    team_brl_percent=8.1,
    team_ev95percent=38.2,
    barrel_pa=0.091,
    bb_pct=0.085,
    k_pct=0.221,
    pa=100,
    bip=70,
):
    cache.save_record(
        namespace=TTEPITNamespaces.TTE_TEAM_DAILY,
        entity_id=team_id,
        season=2024,
        as_of_date=as_of_date,
        source=TTEPITSources.TTE_TEAM_DAILY,
        source_fingerprint=fingerprint,
        data={
            "team_name": "New York Yankees",
            "lambda_offense": lambda_offense,
            "runs_per_game": runs_per_game,
            "team_est_woba": team_est_woba,
            "team_woba": team_woba,
            "team_brl_percent": team_brl_percent,
            "team_ev95percent": team_ev95percent,
            "pa": pa,
            "bip": bip,
        },
    )


def _seed_team_rolling(
    cache,
    *,
    as_of_date,
    team_id=147,
    fingerprint="team-fingerprint",
    team_name=None,
    runs_per_game=None,
    team_est_woba=0.321,
    team_woba=0.318,
    team_brl_percent=8.1,
    team_ev95percent=38.2,
    barrel_pa=0.091,
    bb_pct=0.085,
    k_pct=0.221,
    pa=100,
    bip=70,
):
    data = {
        "team_name": team_name,
        "runs_per_game": runs_per_game,
        "est_woba": team_est_woba,
        "woba": team_woba,
        "brl_percent": team_brl_percent,
        "ev95percent": team_ev95percent,
        "barrel_pa": barrel_pa,
        "bb_pct": bb_pct,
        "k_pct": k_pct,
        "pa": pa,
        "bip": bip,
    }
    if team_name is None:
        data.pop("team_name")
    cache.save_record(
        namespace=TTEPITNamespaces.TEAM_OFFENSE_ROLLING,
        entity_id=team_id,
        season=2024,
        as_of_date=as_of_date,
        source=TTEPITSources.TEAM_OFFENSE_ROLLING,
        source_fingerprint=fingerprint,
        data=data,
    )


def _seed_batter_rolling(cache, *, as_of_date, team_id=147, fingerprint="batter-fingerprint"):
    cache.save_record(
        namespace=TTEPITNamespaces.BATTER_ROLLING,
        entity_id=team_id,
        season=2024,
        as_of_date=as_of_date,
        source=TTEPITSources.BATTER_ROLLING,
        source_fingerprint=fingerprint,
        data={"n_batters": 9},
    )


def _seed_prior_baseline(
    cache,
    *,
    as_of_date,
    team_id=147,
    fingerprint="prior-fingerprint",
    prior_season=2023,
    team_est_woba_prior=0.315,
    team_woba_prior=0.312,
    bb_pct_prior=0.081,
    k_pct_prior=0.225,
    barrel_pa_prior=0.084,
    brl_percent_prior=7.8,
    ev95percent_prior=37.1,
    pa_prior=6000,
    bip_prior=4100,
):
    cache.save_record(
        namespace=TTEPITNamespaces.TEAM_OFFENSE_PRIOR_BASELINE,
        entity_id=team_id,
        season=2024,
        as_of_date=as_of_date,
        source=TTEPITSources.TEAM_OFFENSE_PRIOR_BASELINE,
        source_fingerprint=fingerprint,
        data={
            "team_id": team_id,
            "season": 2024,
            "prior_season": prior_season,
            "team_est_woba_prior": team_est_woba_prior,
            "team_woba_prior": team_woba_prior,
            "bb_pct_prior": bb_pct_prior,
            "k_pct_prior": k_pct_prior,
            "barrel_pa_prior": barrel_pa_prior,
            "brl_percent_prior": brl_percent_prior,
            "ev95percent_prior": ev95percent_prior,
            "pa_prior": pa_prior,
            "bip_prior": bip_prior,
            "source_fingerprint": fingerprint,
            "baseline_version": "tte_prior_baseline_v1",
        },
    )
