import inspect

from modules.baseball_module.advanced_pit_enrichment import adapt_tte_pit_snapshot_to_lambda


def test_complete_snapshot_produces_lambda_when_formula_requirements_are_met():
    result = adapt_tte_pit_snapshot_to_lambda(
        _snapshot(
            team_est_woba=0.330,
            team_woba=0.320,
            team_brl_percent=10.0,
            team_ev95percent=42.0,
            team_barrel_pa=0.095,
            bb_pct=0.090,
            k_pct=0.210,
            pa=300,
            bip=210,
        ),
        league_baseline={"prior_lambda_offense": 4.5},
    )

    assert result["found"] is True
    assert result["sample_size_status"] == "ok"
    assert result["fallback_used"] is None
    assert result["lambda_offense"] is not None
    assert 3.0 <= result["lambda_offense"] <= 7.0
    assert result["formula_version"] == "tte_pit_adapter_v1"
    assert result["provenance"]["formula_inputs"]["team_est_woba"] == 0.330
    assert result["provenance"]["legacy_constants"]["lambda_min"] == 3.0


def test_missing_est_woba_returns_none_with_fallback_used():
    result = adapt_tte_pit_snapshot_to_lambda(
        _snapshot(team_est_woba=None, team_barrel_pa=0.095, bb_pct=0.09, k_pct=0.21, pa=300),
        league_baseline={"prior_lambda_offense": 4.5},
    )

    assert result["lambda_offense"] is None
    assert result["fallback_used"] == "missing_inputs:team_est_woba"
    assert result["provenance"]["missing_inputs"] == ["team_est_woba"]


def test_thin_pa_sample_is_marked_thin():
    result = adapt_tte_pit_snapshot_to_lambda(
        _snapshot(team_est_woba=0.330, team_barrel_pa=0.095, bb_pct=0.09, k_pct=0.21, pa=80),
        league_baseline={"prior_lambda_offense": 4.5},
    )

    assert result["sample_size_status"] == "thin"
    assert result["lambda_offense"] is None
    assert result["fallback_used"] == "thin_sample:pa<150"


def test_none_values_preserved_without_fake_zeroes():
    result = adapt_tte_pit_snapshot_to_lambda(
        _snapshot(
            team_est_woba=None,
            team_woba=None,
            team_brl_percent=None,
            team_ev95percent=None,
            pa=None,
            bip=None,
        )
    )

    assert result["team_est_woba"] is None
    assert result["team_woba"] is None
    assert result["team_brl_percent"] is None
    assert result["team_ev95percent"] is None
    assert result["pa"] is None
    assert result["bip"] is None
    assert result["lambda_offense"] is None
    assert result["sample_size_status"] == "missing"


def test_true_zero_barrel_rate_with_valid_denominator_is_not_missing():
    result = adapt_tte_pit_snapshot_to_lambda(
        _snapshot(
            team_est_woba=0.310,
            team_woba=0.300,
            team_brl_percent=0.0,
            team_barrel_pa=0.0,
            bb_pct=0.08,
            k_pct=0.22,
            pa=250,
            bip=180,
        ),
        league_baseline={"prior_lambda_offense": 4.5},
    )

    assert result["team_brl_percent"] == 0.0
    assert result["fallback_used"] is None
    assert "barrel_pa" not in result["provenance"]["missing_inputs"]
    assert result["lambda_offense"] is not None


def test_no_fake_zero_for_missing_barrel_pa():
    result = adapt_tte_pit_snapshot_to_lambda(
        _snapshot(team_est_woba=0.330, team_brl_percent=None, bb_pct=0.09, k_pct=0.21, pa=300),
        league_baseline={"prior_lambda_offense": 4.5},
    )

    assert result["team_brl_percent"] is None
    assert result["lambda_offense"] is None
    assert result["fallback_used"] == "missing_inputs:barrel_pa"


def test_provenance_preserved():
    snapshot = _snapshot(
        team_est_woba=0.330,
        team_barrel_pa=0.095,
        bb_pct=0.09,
        k_pct=0.21,
        pa=300,
        source_fingerprints={"savant_team_offense_rolling": "team-fp"},
    )

    result = adapt_tte_pit_snapshot_to_lambda(
        snapshot,
        league_baseline={"prior_lambda_offense": 4.5},
    )

    assert result["provenance"]["snapshot_version"] == "tte_daily_snapshot_v1"
    assert result["provenance"]["requested_as_of_date"] == "2024-04-08T23:59:59Z"
    assert result["provenance"]["team_offense_as_of_date"] == "2024-04-08T00:00:00+00:00"
    assert result["provenance"]["source_fingerprints"] == {
        "savant_team_offense_rolling": "team-fp"
    }


def test_formula_version_present_for_missing_snapshot():
    result = adapt_tte_pit_snapshot_to_lambda({"found": False})

    assert result["found"] is False
    assert result["formula_version"] == "tte_pit_adapter_v1"
    assert result["lambda_offense"] is None
    assert result["fallback_used"] == "snapshot_not_found"


def test_no_live_backtest_or_run_module_imports():
    import modules.baseball_module.advanced_pit_enrichment.tte_pit_adapter as module

    source = inspect.getsource(module)
    assert "import app" not in source
    assert "from app" not in source
    assert "backtest_and_retrain" not in source
    assert "run_module" not in source
    assert "true_talent_engine" not in source


def _snapshot(**overrides):
    snapshot = {
        "found": True,
        "runs_per_game": None,
        "team_est_woba": 0.320,
        "team_woba": 0.318,
        "team_brl_percent": 8.1,
        "team_ev95percent": 38.2,
        "pa": 200,
        "bip": 140,
        "requested_as_of_date": "2024-04-08T23:59:59Z",
        "team_offense_as_of_date": "2024-04-08T00:00:00+00:00",
        "source_fingerprints": {"savant_team_offense_rolling": "team-fingerprint"},
        "snapshot_version": "tte_daily_snapshot_v1",
    }
    snapshot.update(overrides)
    return snapshot
