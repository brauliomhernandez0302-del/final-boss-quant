import inspect

import backtest_and_retrain as backtest
from modules.baseball_module.advanced_pit_enrichment import (
    PITCache,
    RawSavantTeamDefenseEvent,
    TeamDefenseDailySnapshotBuilder,
    TeamDefensePITNamespaces,
    TeamDefensePITSources,
    adapt_defense_pit_snapshot,
    fielding_team_for_event,
)


def test_default_mode_does_not_invoke_defense_pit(monkeypatch):
    game_data = _game_data()

    def pit_must_not_run(*args, **kwargs):
        raise AssertionError("Defense PIT must remain opt-in")

    monkeypatch.setattr(backtest, "apply_experimental_defense_pit_mode", pit_must_not_run)
    lh, la, meta = backtest._apply_defense_stage(
        4.5, 4.2, game_data, use_defense_pit=False
    )

    assert (lh, la) == (4.5, 4.2)
    assert meta["mode"] == "none"
    assert "experimental_defense_pit" not in game_data


def test_default_mode_keeps_legacy_defense_path(monkeypatch):
    game_data = _game_data()
    game_data["defense_home"] = {"der": 0.72, "bip": 100}
    calls = []

    def legacy(lh, la, data):
        calls.append(data)
        return lh * 0.99, la * 1.01, {}

    monkeypatch.setattr(backtest, "adjust_for_defense", legacy)
    lh, la, meta = backtest._apply_defense_stage(
        4.5, 4.2, game_data, use_defense_pit=False
    )

    assert calls == [game_data]
    assert lh == 4.5 * 0.99
    assert la == 4.2 * 1.01
    assert meta["legacy_defense_called"] is True


def test_experimental_mode_blocks_legacy_defense_call(monkeypatch):
    game_data = _game_data()
    game_data["defense_home"] = {"der": 0.99, "bip": 9999}
    game_data["experimental_defense_pit"] = _defense_metadata()

    def legacy_must_not_run(*args, **kwargs):
        raise AssertionError("legacy DER/BIP was called")

    monkeypatch.setattr(backtest, "adjust_for_defense", legacy_must_not_run)
    _, _, meta = backtest._apply_defense_stage(
        4.5, 4.2, game_data, use_defense_pit=True
    )

    assert meta["legacy_defense_called"] is False
    assert meta["mode"] == "experimental_defense_pit"


def test_positive_home_defense_proxy_lowers_away_lambda():
    game_data = _game_data()
    game_data["experimental_defense_pit"] = _defense_metadata(
        home_multiplier=0.97,
        away_multiplier=1.0,
    )

    lh, la, _ = backtest._apply_defense_stage(
        4.5, 4.2, game_data, use_defense_pit=True
    )

    assert lh == 4.5
    assert la == 4.2 * 0.97


def test_negative_away_defense_proxy_raises_home_lambda():
    game_data = _game_data()
    game_data["experimental_defense_pit"] = _defense_metadata(
        home_multiplier=1.0,
        away_multiplier=1.04,
    )

    lh, la, _ = backtest._apply_defense_stage(
        4.5, 4.2, game_data, use_defense_pit=True
    )

    assert lh == 4.5 * 1.04
    assert la == 4.2


def test_positive_proxy_from_adapter_reduces_opponent_lambda(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_prior(cache, "BOS", proxy=0.03)
    game_data = _game_data()
    backtest.apply_experimental_defense_pit_mode(
        game_data=game_data,
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TeamDefenseDailySnapshotBuilder(cache),
        adapter=adapt_defense_pit_snapshot,
    )

    _, away_lambda, _ = backtest._apply_defense_stage(
        4.5, 4.2, game_data, use_defense_pit=True
    )

    assert game_data["experimental_defense_pit"]["home"][
        "contact_adjusted_defense_proxy"
    ] == 0.03
    assert away_lambda < 4.2


def test_negative_proxy_from_adapter_increases_opponent_lambda(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_prior(cache, "NYY", proxy=-0.03)
    game_data = _game_data()
    backtest.apply_experimental_defense_pit_mode(
        game_data=game_data,
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TeamDefenseDailySnapshotBuilder(cache),
        adapter=adapt_defense_pit_snapshot,
    )

    home_lambda, _, _ = backtest._apply_defense_stage(
        4.5, 4.2, game_data, use_defense_pit=True
    )

    assert game_data["experimental_defense_pit"]["away"][
        "contact_adjusted_defense_proxy"
    ] == -0.03
    assert home_lambda > 4.5


def test_apply_mode_uses_d_minus_one_cutoff_and_clears_legacy_payload(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_current(cache, "BOS", "2024-04-01T23:59:59Z", proxy=0.02)
    _seed_current(cache, "NYY", "2024-04-01T23:59:59Z", proxy=-0.01)
    game_data = _game_data()
    game_data["defense_home"] = {"der": 0.99, "bip": 9999}
    game_data["defense_away"] = {"der": 0.50, "bip": 9999}

    meta = backtest.apply_experimental_defense_pit_mode(
        game_data=game_data,
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TeamDefenseDailySnapshotBuilder(cache),
        adapter=adapt_defense_pit_snapshot,
    )

    assert meta["requested_as_of_date"] == "2024-04-01T23:59:59Z"
    assert meta["home"]["current_as_of_date"] == "2024-04-01T23:59:59+00:00"
    assert meta["away"]["current_as_of_date"] == "2024-04-01T23:59:59+00:00"
    assert game_data["defense_home"] == {}
    assert game_data["defense_away"] == {}
    assert meta["legacy_full_season_der_bip_blocked"] is True


def test_current_pit_overrides_prior_baseline_in_integration(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_current(cache, "BOS", "2024-04-01T23:59:59Z", proxy=0.04)
    _seed_prior(cache, "BOS", proxy=-0.03)

    meta = backtest.apply_experimental_defense_pit_mode(
        game_data=_game_data(),
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TeamDefenseDailySnapshotBuilder(cache),
        adapter=adapt_defense_pit_snapshot,
    )

    assert meta["home"]["provenance_source"] == "current_defense_pit"
    assert meta["home"]["current_defense_found"] is True
    assert meta["home"]["prior_baseline_found"] is True


def test_prior_baseline_overrides_neutral_in_integration(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_prior(cache, "BOS", proxy=0.025)

    meta = backtest.apply_experimental_defense_pit_mode(
        game_data=_game_data(),
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TeamDefenseDailySnapshotBuilder(cache),
        adapter=adapt_defense_pit_snapshot,
    )

    assert meta["home"]["provenance_source"] == "prior_season_defense_baseline"
    assert meta["home"]["applied_multiplier"] == 0.975
    assert meta["home"]["neutral_fallback"] is False
    assert meta["home"]["bip_count"] == 4000
    assert meta["home"]["xba_bip_count"] == 3900
    assert meta["home"]["sample_size_status"] == "prior_baseline"


def test_neutral_fallback_is_exactly_one_in_integration(tmp_path):
    meta = backtest.apply_experimental_defense_pit_mode(
        game_data=_game_data(),
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TeamDefenseDailySnapshotBuilder(cache_db=tmp_path / "pit.db"),
        adapter=adapt_defense_pit_snapshot,
    )

    for side in ("home", "away"):
        assert meta[side]["provenance_source"] == "neutral_defense_adjustment"
        assert meta[side]["applied_multiplier"] == 1.0
        assert meta[side]["neutral_fallback"] is True


def test_future_and_same_day_current_record_is_rejected(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_current(cache, "BOS", "2024-04-02T00:00:00Z", proxy=0.50)

    meta = backtest.apply_experimental_defense_pit_mode(
        game_data=_game_data(),
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TeamDefenseDailySnapshotBuilder(cache),
        adapter=adapt_defense_pit_snapshot,
    )

    assert meta["home"]["current_defense_found"] is False
    assert meta["home"]["current_as_of_date"] is None
    assert meta["home"]["provenance_source"] == "neutral_defense_adjustment"


def test_build_game_data_defense_pit_mode_omits_legacy_der_bip():
    game_data, _, _ = backtest.build_game_data(
        1,
        "2024-04-02",
        "Boston Red Sox",
        "New York Yankees",
        2024,
        None,
        None,
        _FakeAPI(),
        _FakeIntegrator(),
        None,
        use_team_full_season_defense=False,
    )

    assert game_data["defense_home"] == {}
    assert game_data["defense_away"] == {}


def test_default_build_game_data_still_populates_legacy_der_bip():
    game_data, _, _ = backtest.build_game_data(
        1,
        "2024-04-02",
        "Boston Red Sox",
        "New York Yankees",
        2024,
        None,
        None,
        _FakeAPI(),
        _FakeIntegrator(),
        None,
    )

    assert game_data["defense_home"] == {
        "team_name": "Boston Red Sox",
        "der": 0.7,
        "bip": 4000,
        "oaa": None,
    }
    assert game_data["defense_away"]["der"] == 0.7


def test_usage_counters_and_three_processed_samples():
    usage = _empty_usage()
    meta = _defense_metadata()

    backtest._update_defense_pit_usage(usage, meta, game_date="2024-04-02")
    for game_pk in range(1, 6):
        backtest._append_defense_pit_sample(
            usage,
            game_pk=game_pk,
            game_date="2024-04-02",
            home_team="Boston Red Sox",
            away_team="New York Yankees",
            metadata=meta,
        )
    summary = backtest._defense_pit_usage_summary(
        usage, games_processed=1, failures=0
    )

    assert summary["current_defense_pit_used"] == 2
    assert summary["legacy_defense_calls_blocked"] == 1
    assert summary["future_snapshot_violations"] == 0
    assert summary["same_day_violations"] == 0
    assert summary["missing_fingerprints"] == 0
    assert len(summary["sample_games"]) == 3


def test_top_bottom_fielding_attribution_contract_remains_correct():
    top = _raw_event("Top")
    bottom = _raw_event("Bot")

    assert fielding_team_for_event(top) == "NYY"
    assert fielding_team_for_event(bottom) == "BOS"


def test_backtest_does_not_import_live_or_run_module_for_defense_pit():
    source = inspect.getsource(backtest)

    assert "from modules.baseball_module.core.run_module" not in source
    assert "import modules.baseball_module.core.run_module" not in source
    assert "import app" not in source
    assert "from app" not in source


def _game_data():
    return {
        "game_date": "2024-04-02",
        "season": 2024,
        "home_team_id": 111,
        "away_team_id": 147,
        "home_team": {"name": "Boston Red Sox"},
        "away_team": {"name": "New York Yankees"},
        "defense_home": {},
        "defense_away": {},
    }


def _defense_metadata(*, home_multiplier=0.98, away_multiplier=0.99):
    def side(name, multiplier):
        return {
            "team_name": name,
            "provenance_source": "current_defense_pit",
            "applied_multiplier": multiplier,
            "current_as_of_date": "2024-04-01T23:59:59+00:00",
            "prior_baseline_found": True,
            "prior_weight": 0.5,
            "mapping_failure": False,
            "source_fingerprints": {
                "current_defense_pit": "current-fp",
                "prior_season_defense_baseline": "prior-fp",
            },
        }

    return {
        "requested_as_of_date": "2024-04-01T23:59:59Z",
        "home": side("Boston Red Sox", home_multiplier),
        "away": side("New York Yankees", away_multiplier),
        "legacy_full_season_der_bip_blocked": True,
    }


def _empty_usage():
    return {
        "games_attempted": 0,
        "current_defense_pit_used": 0,
        "prior_season_baseline_used": 0,
        "neutral_defense_adjustment_used": 0,
        "legacy_defense_calls_blocked": 0,
        "future_snapshot_violations": 0,
        "same_day_violations": 0,
        "missing_fingerprints": 0,
        "duplicate_pit_keys": 0,
        "mapping_failures": 0,
        "samples": [],
    }


def _seed_current(cache, team, as_of_date, *, proxy):
    cache.save_record(
        namespace=TeamDefensePITNamespaces.TEAM_DEFENSE_ROLLING,
        entity_id=team,
        season=2024,
        as_of_date=as_of_date,
        source=TeamDefensePITSources.TEAM_DEFENSE_ROLLING,
        source_fingerprint=f"current-{team}",
        data={
            "contact_adjusted_defense_proxy": proxy,
            "bip_count": 500,
            "xba_bip_count": 500,
            "sample_size_status": "ok",
            "source_window_start_date": "2024-03-28",
            "source_window_end_date": as_of_date[:10],
        },
    )


def _seed_prior(cache, team, *, proxy):
    cache.save_record(
        namespace=TeamDefensePITNamespaces.TEAM_DEFENSE_PRIOR_BASELINE,
        entity_id=team,
        season=2024,
        as_of_date="2023-10-01T23:59:59Z",
        source=TeamDefensePITSources.TEAM_DEFENSE_PRIOR_BASELINE,
        source_fingerprint=f"prior-{team}",
        data={
            "contact_adjusted_defense_proxy": proxy,
            "bip_count": 4000,
            "xba_bip_count": 3900,
            "source_window_start_date": "2023-03-30",
            "source_window_end_date": "2023-10-01",
        },
    )


def _raw_event(half):
    return RawSavantTeamDefenseEvent(
        game_date="2024-04-01",
        game_pk=1,
        at_bat_number=1,
        pitch_number=1,
        events="field_out",
        home_team="NYY",
        away_team="BOS",
        inning_topbot=half,
        batting_team=None,
        estimated_ba_using_speedangle=0.2,
        bb_type="ground_ball",
    )


class _FakeAPI:
    def get_team_offensive_stats(self, team_id, season):
        return {}

    def get_team_pitching_stats(self, team_id, season):
        return {
            "team_era": 4.2,
            "team_whip": 1.3,
            "runs_allowed_per_game": 4.4,
            "der": 0.7,
            "bip": 4000,
        }

    def get_bullpen_era(self, team_id, season):
        return {}


class _FakeIntegrator:
    def _fetch_team_rpg(self, team_id, season):
        return 4.5

    def get_team_lambda(self, team_name, rpg, team_id=None):
        return rpg
