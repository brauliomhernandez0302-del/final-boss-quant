import inspect
import json

import pytest

import backtest_and_retrain as backtest
from modules.baseball_module.advanced_pit_enrichment import PITCache
from modules.baseball_module.advanced_pit_enrichment.tte_daily_snapshot_builder import (
    TTEDailySnapshotBuilder,
    TTEPITNamespaces,
    TTEPITSources,
    previous_day_cutoff_for_game_date,
)
from modules.baseball_module.advanced_pit_enrichment.tte_pit_adapter import (
    adapt_tte_pit_snapshot_to_lambda,
)


class _FakeLearning:
    def get_kalman_lambda_adjustment(self, team, role, season, model_lambda, prediction_source="live"):
        return model_lambda

    def get_pipeline_weights(self, season, prediction_source="live"):
        return {}

    def compute_team_bias_kalman_adjusted(self, team, season, role, month=None, before_date=None,
                                           prediction_source="live"):
        return 1.0

    def get_platt_params(self, season, prediction_source="live"):
        return 1.0, 0.0


class _FakeAPI:
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


class _TrackingIntegrator:
    def __init__(self):
        self.fetch_team_rpg_calls = []
        self.get_team_lambda_calls = []

    def _fetch_team_rpg(self, team_id, season):
        self.fetch_team_rpg_calls.append((team_id, season))
        return 4.5

    def get_team_lambda(self, team_name, rpg, team_id=None):
        self.get_team_lambda_calls.append((team_name, rpg, team_id))
        return rpg


def test_team_tte_pit_mode_uses_previous_day_cutoff_and_lambdas(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_tte_pit(cache, team_id="BOS", as_of_date="2024-04-01T00:00:00Z", est_woba=0.350)
    _seed_team_tte_pit(cache, team_id="NYY", as_of_date="2024-04-01T00:00:00Z", est_woba=0.300)
    game_data = _game_data()

    meta = backtest.apply_experimental_team_tte_pit_mode(
        game_data=game_data,
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TTEDailySnapshotBuilder(cache),
        adapter=adapt_tte_pit_snapshot_to_lambda,
    )

    assert meta["requested_as_of_date"] == previous_day_cutoff_for_game_date("2024-04-02")
    assert meta["home"]["pit_found"] is True
    assert meta["away"]["pit_found"] is True
    assert meta["home"]["team_offense_as_of_date"] == "2024-04-01T00:00:00+00:00"
    assert game_data["home_team"]["tte_pit_lambda_offense"] is not None
    assert game_data["away_team"]["tte_pit_lambda_offense"] is not None


def test_team_tte_pit_skip_reason_none_when_both_lambdas_exist(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_tte_pit(cache, team_id="BOS", as_of_date="2024-04-01T00:00:00Z", est_woba=0.350)
    _seed_team_tte_pit(cache, team_id="NYY", as_of_date="2024-04-01T00:00:00Z", est_woba=0.300)

    meta = backtest.apply_experimental_team_tte_pit_mode(
        game_data=_game_data(),
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TTEDailySnapshotBuilder(cache),
        adapter=adapt_tte_pit_snapshot_to_lambda,
    )

    assert backtest.team_tte_pit_skip_reason(meta) is None


def test_team_tte_pit_mode_does_not_use_future_snapshot(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_tte_pit(cache, team_id="BOS", as_of_date="2024-04-02T00:00:00Z", est_woba=0.410)
    game_data = _game_data()

    meta = backtest.apply_experimental_team_tte_pit_mode(
        game_data=game_data,
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TTEDailySnapshotBuilder(cache),
        adapter=adapt_tte_pit_snapshot_to_lambda,
    )

    assert meta["home"]["pit_found"] is False
    assert meta["home"]["fallback_used"] == "snapshot_not_found"
    assert "tte_pit_lambda_offense" not in game_data["home_team"]


def test_home_missing_team_tte_pit_skips_before_prediction(tmp_path, monkeypatch):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_tte_pit(cache, team_id="NYY", as_of_date="2024-04-01T00:00:00Z", est_woba=0.300)
    game_data = _game_data()

    meta = backtest.apply_experimental_team_tte_pit_mode(
        game_data=game_data,
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TTEDailySnapshotBuilder(cache),
        adapter=adapt_tte_pit_snapshot_to_lambda,
    )

    assert backtest.team_tte_pit_skip_reason(meta) == "missing_home_team_tte_pit"
    _assert_run_pipeline_rejects_missing_team_tte_pit(monkeypatch, game_data, cache)


def test_away_missing_team_tte_pit_skips_before_prediction(tmp_path, monkeypatch):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_tte_pit(cache, team_id="BOS", as_of_date="2024-04-01T00:00:00Z", est_woba=0.350)
    game_data = _game_data()

    meta = backtest.apply_experimental_team_tte_pit_mode(
        game_data=game_data,
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TTEDailySnapshotBuilder(cache),
        adapter=adapt_tte_pit_snapshot_to_lambda,
    )

    assert backtest.team_tte_pit_skip_reason(meta) == "missing_away_team_tte_pit"
    _assert_run_pipeline_rejects_missing_team_tte_pit(monkeypatch, game_data, cache)


def test_both_missing_team_tte_pit_skips_before_prediction(tmp_path, monkeypatch):
    game_data = _game_data()

    meta = backtest.apply_experimental_team_tte_pit_mode(
        game_data=game_data,
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TTEDailySnapshotBuilder(cache_db=tmp_path / "pit.db"),
        adapter=adapt_tte_pit_snapshot_to_lambda,
    )

    assert meta["home"]["pit_found"] is False
    assert meta["away"]["pit_found"] is False
    assert meta["home"]["fallback_used"] == "snapshot_not_found"
    assert meta["away"]["fallback_used"] == "snapshot_not_found"
    assert backtest.team_tte_pit_skip_reason(meta) == "missing_both_team_tte_pit"
    _assert_run_pipeline_rejects_missing_team_tte_pit(
        monkeypatch,
        game_data,
        PITCache(tmp_path / "pit.db"),
    )


def test_team_tte_pit_cutoff_always_defaults_to_previous_day():
    assert (
        backtest._team_tte_pit_cutoff_for_row({"game_date": "2024-04-02"})
        == "2024-04-01T23:59:59Z"
    )
    assert (
        backtest._team_tte_pit_cutoff_for_row(
            {"game_date": "2024-04-02", "prediction_cutoff_utc": "2024-04-02T18:00:00Z"}
        )
        == "2024-04-01T23:59:59Z"
    )


def test_team_tte_pit_run_pipeline_skips_legacy_true_talent(monkeypatch, tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_team_tte_pit(cache, team_id="BOS", as_of_date="2024-04-01T00:00:00Z", est_woba=0.350)
    _seed_team_tte_pit(cache, team_id="NYY", as_of_date="2024-04-01T00:00:00Z", est_woba=0.300)
    game_data = _game_data()

    monkeypatch.setattr(backtest, "_TTE_AVAILABLE", True)

    def legacy_tte_must_not_run(*args, **kwargs):
        raise AssertionError("legacy TTE should not run in Team/TTE PIT mode")

    monkeypatch.setattr(backtest, "_get_tte_lambda", legacy_tte_must_not_run)
    _stub_pipeline_dependencies(monkeypatch)

    pred = backtest.run_pipeline(
        game_data,
        4.5,
        4.5,
        _FakeLearning(),
        2024,
        n_mc=100,
        use_team_tte_pit=True,
        team_tte_pit_snapshot_builder=TTEDailySnapshotBuilder(cache),
        team_tte_pit_adapter=adapt_tte_pit_snapshot_to_lambda,
    )

    assert pred["team_tte_pit"]["home"]["pit_found"] is True
    assert pred["team_tte_pit"]["away"]["pit_found"] is True
    assert pred["lh"] != 4.5
    assert pred["la"] != 4.5


def test_team_tte_pit_build_game_data_does_not_call_full_season_lambda_sources():
    integrator = _TrackingIntegrator()

    game_data, lh, la = backtest.build_game_data(
        1,
        "2024-04-02",
        "Boston Red Sox",
        "New York Yankees",
        2024,
        None,
        None,
        _FakeAPI(),
        integrator,
        None,
        use_team_full_season_offense_base=False,
    )

    assert integrator.fetch_team_rpg_calls == []
    assert integrator.get_team_lambda_calls == []
    assert lh == backtest.LEAGUE_AVG_RUNS
    assert la == backtest.LEAGUE_AVG_RUNS
    assert game_data["home_team"]["runs_per_game"] == backtest.LEAGUE_AVG_RUNS
    assert game_data["away_team"]["runs_per_game"] == backtest.LEAGUE_AVG_RUNS


def test_team_tte_pit_report_counters_track_processed_and_skipped():
    usage = _empty_usage()
    both_found = {
        "home": {"pit_found": True, "lambda_offense": 4.4},
        "away": {"pit_found": True, "lambda_offense": 4.1},
    }
    home_missing = {
        "home": {"pit_found": False, "lambda_offense": None},
        "away": {"pit_found": True, "lambda_offense": 4.1},
    }
    away_missing = {
        "home": {"pit_found": True, "lambda_offense": 4.4},
        "away": {"pit_found": False, "lambda_offense": None},
    }
    both_missing = {
        "home": {"pit_found": False, "lambda_offense": None},
        "away": {"pit_found": False, "lambda_offense": None},
    }

    backtest._update_team_tte_pit_usage(usage, both_found, skipped=False, skip_reason=None)
    backtest._update_team_tte_pit_usage(
        usage,
        home_missing,
        skipped=True,
        skip_reason="missing_home_team_tte_pit",
    )
    backtest._update_team_tte_pit_usage(
        usage,
        away_missing,
        skipped=True,
        skip_reason="missing_away_team_tte_pit",
    )
    backtest._update_team_tte_pit_usage(
        usage,
        both_missing,
        skipped=True,
        skip_reason="missing_both_team_tte_pit",
    )

    summary = backtest._team_tte_pit_usage_summary(usage, games_processed=1, failures=0)

    assert summary["games_attempted"] == 4
    assert summary["games_processed_with_both_team_tte_pit"] == 1
    assert summary["games_skipped_missing_team_tte_pit"] == 3
    assert summary["home_missing"] == 2
    assert summary["away_missing"] == 2
    assert summary["both_missing"] == 1


def test_team_tte_pit_report_is_written_when_all_games_skip(tmp_path):
    summary = backtest._team_tte_pit_usage_summary(
        {
            **_empty_usage(),
            "games_attempted": 2,
            "games_skipped_missing_team_tte_pit": 2,
            "home_team_tte_pit_missing": 2,
            "away_team_tte_pit_missing": 2,
            "both_missing": 2,
        },
        games_processed=0,
        failures=0,
    )

    backtest.generate_report([], tmp_path, team_tte_pit_summary=summary)

    reports = list(tmp_path.glob("backtest_report_*.json"))
    assert len(reports) == 1
    data = json.loads(reports[0].read_text())
    assert data["total_games"] == 0
    assert data["team_tte_pit"]["games_attempted"] == 2
    assert data["team_tte_pit"]["games_skipped_missing_team_tte_pit"] == 2


def test_default_build_game_data_keeps_full_season_lambda_sources():
    integrator = _TrackingIntegrator()

    backtest.build_game_data(
        1,
        "2024-04-02",
        "Boston Red Sox",
        "New York Yankees",
        2024,
        None,
        None,
        _FakeAPI(),
        integrator,
        None,
    )

    assert integrator.fetch_team_rpg_calls == [(111, 2024), (147, 2024)]
    assert [call[0] for call in integrator.get_team_lambda_calls] == [
        "Boston Red Sox",
        "New York Yankees",
    ]


def test_default_run_pipeline_does_not_touch_team_tte_pit(monkeypatch):
    game_data = _game_data()

    def pit_mode_must_not_run(*args, **kwargs):
        raise AssertionError("Team/TTE PIT should be opt-in only")

    monkeypatch.setattr(backtest, "_TTE_AVAILABLE", False)
    monkeypatch.setattr(backtest, "apply_experimental_team_tte_pit_mode", pit_mode_must_not_run)
    _stub_pipeline_dependencies(monkeypatch)

    pred = backtest.run_pipeline(game_data, 4.5, 4.5, _FakeLearning(), 2024, n_mc=100)

    assert pred["team_tte_pit"] is None
    assert "experimental_team_tte_pit" not in game_data


def test_team_tte_pit_missing_builder_is_configuration_error():
    with pytest.raises(ValueError, match="Team/TTE PIT mode requires"):
        backtest.run_pipeline(
            _game_data(),
            4.5,
            4.5,
            _FakeLearning(),
            2024,
            n_mc=100,
            use_team_tte_pit=True,
        )


def test_backtest_module_does_not_import_live_run_module():
    source = inspect.getsource(backtest)

    assert "from modules.baseball_module.core.run_module" not in source
    assert "import modules.baseball_module.core.run_module" not in source
    assert "import app" not in source
    assert "from app" not in source


def _stub_pipeline_dependencies(monkeypatch):
    monkeypatch.setattr(
        backtest,
        "adjust_for_park_and_weather",
        lambda lh, la, game_data: (lh, la, {}),
    )
    monkeypatch.setattr(backtest, "get_adjusted_lambdas", lambda lh, la, game_data: (lh, la, {}))
    monkeypatch.setattr(backtest, "adjust_for_defense", lambda lh, la, game_data: (lh, la, {}))
    monkeypatch.setattr(backtest, "adjust_for_pitchers", lambda lh, la, game_data: (lh, la, {}))
    monkeypatch.setattr(backtest, "adjust_for_bullpen", lambda lh, la, game_data: (lh, la, {}))
    monkeypatch.setattr(backtest, "adjust_for_context", lambda lh, la, game_data: (lh, la, {}))
    monkeypatch.setattr(
        backtest,
        "monte_carlo_advanced",
        lambda **kwargs: {"p_home": 0.55, "p_away": 0.45, "n": kwargs["n_max"]},
    )


def _assert_run_pipeline_rejects_missing_team_tte_pit(monkeypatch, game_data, cache):
    def monte_carlo_must_not_run(**kwargs):
        raise AssertionError("prediction should not be scored when Team/TTE PIT is missing")

    monkeypatch.setattr(backtest, "monte_carlo_advanced", monte_carlo_must_not_run)
    with pytest.raises(ValueError, match="Team/TTE PIT strict coverage failed"):
        backtest.run_pipeline(
            game_data,
            4.5,
            4.5,
            _FakeLearning(),
            2024,
            n_mc=100,
            use_team_tte_pit=True,
            team_tte_pit_snapshot_builder=TTEDailySnapshotBuilder(cache),
            team_tte_pit_adapter=adapt_tte_pit_snapshot_to_lambda,
        )


def _empty_usage():
    return {
        "games_attempted": 0,
        "games_processed_with_both_team_tte_pit": 0,
        "games_skipped_missing_team_tte_pit": 0,
        "home_team_tte_pit_found": 0,
        "away_team_tte_pit_found": 0,
        "home_team_tte_pit_missing": 0,
        "away_team_tte_pit_missing": 0,
        "both_missing": 0,
        "home_missing_only": 0,
        "away_missing_only": 0,
        "samples": [],
    }


def _game_data():
    return {
        "game_date": "2024-04-02",
        "home_team_id": 111,
        "away_team_id": 147,
        "home_team": {"name": "Boston Red Sox"},
        "away_team": {"name": "New York Yankees"},
        "pitcher_home": {},
        "pitcher_away": {},
        "bullpen_home": {},
        "bullpen_away": {},
    }


def _seed_team_tte_pit(cache, *, team_id, as_of_date, est_woba):
    cache.save_record(
        namespace=TTEPITNamespaces.TEAM_OFFENSE_ROLLING,
        entity_id=team_id,
        season=2024,
        as_of_date=as_of_date,
        source=TTEPITSources.TEAM_OFFENSE_ROLLING,
        source_fingerprint=f"team-{team_id}-{as_of_date}",
        data={
            "est_woba": est_woba,
            "woba": est_woba - 0.010,
            "barrel_pa": 0.095,
            "brl_percent": 14.25,
            "bb_pct": 0.090,
            "k_pct": 0.210,
            "pa": 240,
            "bip": 160,
            "batted_ball_count": 160,
            "source_window_start_date": "2024-03-28",
            "source_window_end_date": as_of_date[:10],
        },
        fetched_at="2026-06-26T12:00:00Z",
    )
    cache.save_record(
        namespace=TTEPITNamespaces.TEAM_OFFENSE_PRIOR_BASELINE,
        entity_id=team_id,
        season=2024,
        as_of_date="2024-03-27T00:00:00Z",
        source=TTEPITSources.TEAM_OFFENSE_PRIOR_BASELINE,
        source_fingerprint=f"prior-{team_id}",
        data={
            "prior_season": 2023,
            "team_est_woba_prior": 0.320,
            "team_woba_prior": 0.315,
            "barrel_pa_prior": 0.088,
            "bb_pct_prior": 0.086,
            "k_pct_prior": 0.224,
            "brl_percent_prior": 8.8,
            "ev95percent_prior": 36.0,
            "pa_prior": 5600,
            "bip_prior": 3800,
            "source_window_start_date": "2023-03-30",
            "source_window_end_date": "2023-10-01",
        },
        fetched_at="2026-06-26T12:00:00Z",
    )
