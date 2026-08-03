import inspect

import backtest_and_retrain as backtest
from modules.baseball_module.advanced_pit_enrichment import (
    BullpenPITNamespaces,
    BullpenPITSources,
    PITCache,
    TeamBullpenDailySnapshotBuilder,
    adapt_bullpen_pit_snapshot,
)
from modules.baseball_module.context_engine import bullpen_engine


def test_default_mode_does_not_invoke_bullpen_pit(monkeypatch):
    game_data = _game_data()

    def pit_must_not_run(*args, **kwargs):
        raise AssertionError("Bullpen PIT must remain opt-in")

    monkeypatch.setattr(backtest, "apply_experimental_bullpen_pit_mode", pit_must_not_run)
    lh, la, meta = backtest._apply_bullpen_stage(
        4.5, 4.2, game_data, use_bullpen_pit=False
    )

    assert (lh, la) == (4.5, 4.2)
    assert meta["mode"] == "none"
    assert "experimental_bullpen_pit" not in game_data


def test_default_mode_keeps_legacy_bullpen_path(monkeypatch):
    game_data = _game_data()
    game_data["bullpen_home"] = {"era": 4.0}
    calls = []

    def legacy(lh, la, data):
        calls.append(data)
        return lh * 1.01, la * 0.99, {}

    monkeypatch.setattr(backtest, "adjust_for_bullpen", legacy)
    lh, la, meta = backtest._apply_bullpen_stage(
        4.5, 4.2, game_data, use_bullpen_pit=False
    )

    assert calls == [game_data]
    assert lh == 4.5 * 1.01
    assert la == 4.2 * 0.99
    assert meta["legacy_bullpen_called"] is True


def test_experimental_mode_blocks_every_legacy_bullpen_path(monkeypatch):
    game_data = _game_data()
    game_data["bullpen_home"] = {"era": 1.0}
    game_data["experimental_bullpen_pit"] = _metadata()

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy bullpen path was called")

    monkeypatch.setattr(backtest, "adjust_for_bullpen", forbidden)
    monkeypatch.setattr(bullpen_engine, "_fetch_savant_pitcher_expected", forbidden)
    monkeypatch.setattr(bullpen_engine, "_fetch_savant_pitcher_exitvelo", forbidden)
    monkeypatch.setattr(bullpen_engine, "_fetch_team_roster", forbidden)

    _, _, meta = backtest._apply_bullpen_stage(
        4.5, 4.2, game_data, use_bullpen_pit=True
    )

    assert meta["mode"] == "experimental_bullpen_pit"
    assert meta["legacy_bullpen_called"] is False


def test_build_game_data_experimental_mode_blocks_get_bullpen_era():
    api = _FakeAPI()
    game_data, _, _ = backtest.build_game_data(
        1,
        "2024-04-02",
        "Boston Red Sox",
        "New York Yankees",
        2024,
        None,
        None,
        api,
        _FakeIntegrator(),
        None,
        use_legacy_full_season_bullpen=False,
    )

    assert api.bullpen_calls == []
    assert game_data["bullpen_home"] == {}
    assert game_data["bullpen_away"] == {}


def test_prefetch_experimental_mode_blocks_get_bullpen_era():
    api = _FakeAPI()

    backtest.prefetch_team_stats(api, [2024], include_legacy_bullpen=False)

    assert api.bullpen_calls == []


def test_home_bullpen_multiplier_changes_only_away_lambda():
    game_data = _game_data()
    game_data["experimental_bullpen_pit"] = _metadata(
        home_multiplier=0.96,
        away_multiplier=1.0,
    )

    lh, la, _ = backtest._apply_bullpen_stage(
        4.5, 4.2, game_data, use_bullpen_pit=True
    )

    assert lh == 4.5
    assert la == 4.2 * 0.96


def test_away_bullpen_multiplier_changes_only_home_lambda():
    game_data = _game_data()
    game_data["experimental_bullpen_pit"] = _metadata(
        home_multiplier=1.0,
        away_multiplier=1.04,
    )

    lh, la, _ = backtest._apply_bullpen_stage(
        4.5, 4.2, game_data, use_bullpen_pit=True
    )

    assert lh == 4.5 * 1.04
    assert la == 4.2


def test_current_sufficient_sample_overrides_prior(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_current(cache, bf=200, xwoba=0.28)
    _seed_prior(cache, bf=1000, xwoba=0.36)

    adapted = _adapt(cache)

    assert adapted["provenance_source"] == "current_bullpen_pit"
    assert adapted["current_weight"] == 1.0
    assert adapted["prior_weight"] == 0.0
    assert adapted["quality_metrics"]["xwoba_against"] == 0.28


def test_thin_current_blends_prior_with_weights_summing_to_one(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_current(cache, bf=50, xwoba=0.28)
    _seed_prior(cache, bf=1000, xwoba=0.36)

    adapted = _adapt(cache)

    assert adapted["provenance_source"] == "current_prior_bullpen_blend"
    assert adapted["current_weight"] == 0.25
    assert adapted["prior_weight"] == 0.75
    assert adapted["current_weight"] + adapted["prior_weight"] == 1.0
    assert adapted["quality_metrics"]["xwoba_against"] == 0.34
    assert "current_bf/200" in adapted["blend_formula"]
    assert adapted["selected_as_of_date"] == "2024-04-08T23:59:59+00:00"


def test_prior_baseline_used_when_current_absent(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_prior(cache, bf=1000, xwoba=0.31)

    adapted = _adapt(cache)

    assert adapted["provenance_source"] == "prior_season_bullpen_baseline"
    assert adapted["current_weight"] == 0.0
    assert adapted["prior_weight"] == 1.0


def test_neutral_fallback_is_exactly_one(tmp_path):
    adapted = _adapt(PITCache(tmp_path / "pit.db"))

    assert adapted["provenance_source"] == "neutral_bullpen_adjustment"
    assert adapted["neutral_fallback"] is True
    assert adapted["applied_multiplier"] == 1.0


def test_pit_adjustment_formula_is_active():
    # tbf=0 → Bayesian regression pulls xwOBA to league mean (0.312) so
    # xwoba_factor ≈ 1.0; heavy workload (250 pitches) drives multiplier > 1.
    decision = backtest.calculate_pit_bullpen_adjustment(
        {
            "provenance_source": "current_bullpen_pit",
            "neutral_fallback": False,
            "quality_metrics": {
                "xwoba_against": 0.20,
                "relief_batters_faced": 0,
            },
            "workload_facts": {"pitches_last_3_days": 250},
            "sample_size_status": "sufficient",
            "source_fingerprints": {"current_bullpen_pit": "fp"},
        }
    )

    assert decision["adjustment_formula"] == "pit_native_xwoba_kbb_barrel"
    assert decision["mapping_status"] == "active"
    assert decision["applied_multiplier"] > 1.0  # heavy workload increases scoring


def test_pit_adjustment_neutral_fallback_is_exactly_one():
    decision = backtest.calculate_pit_bullpen_adjustment(
        {
            "provenance_source": "neutral_bullpen_adjustment",
            "neutral_fallback": True,
            "quality_metrics": {},
            "workload_facts": {},
            "sample_size_status": "neutral",
            "source_fingerprints": {},
        }
    )

    assert decision["applied_multiplier"] == 1.0
    assert decision["adjustment_formula"] == "neutral_only_no_data"


def test_apply_mode_enforces_d_minus_one_and_clears_legacy_payload(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_current(cache, bf=200, xwoba=0.28, as_of="2024-04-01T23:59:59Z")
    game_data = _game_data()
    game_data["bullpen_home"] = {"era": 1.0}
    game_data["bullpen_away"] = {"era": 9.0}

    meta = backtest.apply_experimental_bullpen_pit_mode(
        game_data=game_data,
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TeamBullpenDailySnapshotBuilder(cache),
        adapter=adapt_bullpen_pit_snapshot,
    )

    assert meta["requested_as_of_date"] == "2024-04-01T23:59:59Z"
    assert meta["home"]["current_as_of_date"] == "2024-04-01T23:59:59+00:00"
    assert game_data["bullpen_home"] == {}
    assert game_data["bullpen_away"] == {}
    assert meta["legacy_bullpen_blocked"] is True


def test_future_and_same_day_snapshot_is_rejected_in_integration(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_current(cache, bf=999, xwoba=0.10, as_of="2024-04-02T00:00:00Z")

    meta = backtest.apply_experimental_bullpen_pit_mode(
        game_data=_game_data(),
        season=2024,
        game_date="2024-04-02",
        snapshot_builder=TeamBullpenDailySnapshotBuilder(cache),
        adapter=adapt_bullpen_pit_snapshot,
    )

    assert meta["home"]["current_bullpen_found"] is False
    assert meta["home"]["provenance_source"] == "neutral_bullpen_adjustment"


def test_usage_counters_and_three_processed_samples():
    usage = _empty_usage()
    meta = _metadata(source="current_prior_bullpen_blend", current_bf=50)

    backtest._update_bullpen_pit_usage(usage, meta, game_date="2024-04-02")
    for game_pk in range(1, 6):
        backtest._append_bullpen_pit_sample(
            usage,
            game_pk=game_pk,
            game_date="2024-04-02",
            home_team="Boston Red Sox",
            away_team="New York Yankees",
            metadata=meta,
        )
    summary = backtest._bullpen_pit_usage_summary(
        usage, games_processed=1, failures=0
    )

    assert summary["current_prior_blended_uses"] == 2
    assert summary["current_thin_bullpen"] == 2
    assert summary["legacy_bullpen_calls_blocked"] == 1
    assert summary["starter_contamination_violations"] == 0
    assert len(summary["sample_games"]) == 3


def test_backtest_does_not_import_live_app_or_run_module_for_bullpen_pit():
    source = inspect.getsource(backtest)

    assert "from modules.baseball_module.core.run_module" not in source
    assert "import modules.baseball_module.core.run_module" not in source
    assert "import app" not in source
    assert "from app" not in source


def _adapt(cache):
    snapshot = TeamBullpenDailySnapshotBuilder(cache).build_for_game(
        team_id="BOS", season=2024, game_date="2024-04-09"
    )
    return adapt_bullpen_pit_snapshot(snapshot)


def _game_data():
    return {
        "game_date": "2024-04-02",
        "season": 2024,
        "home_team_id": 111,
        "away_team_id": 147,
        "home_team": {"name": "Boston Red Sox"},
        "away_team": {"name": "New York Yankees"},
        "bullpen_home": {},
        "bullpen_away": {},
    }


def _metadata(
    *,
    home_multiplier=1.0,
    away_multiplier=1.0,
    source="current_bullpen_pit",
    current_bf=200,
):
    def side(name, multiplier):
        return {
            "team_name": name,
            "provenance_source": source,
            "applied_multiplier": multiplier,
            "current_bullpen_found": True,
            "current_bf": current_bf,
            "current_as_of_date": "2024-04-01T23:59:59+00:00",
            "current_starter_pitches_included": 0,
            "prior_starter_pitches_included": 0,
            "source_fingerprints": {
                "current_bullpen_pit": "current-fp",
                "prior_season_bullpen_baseline": "prior-fp",
            },
        }

    return {
        "requested_as_of_date": "2024-04-01T23:59:59Z",
        "home": side("Boston Red Sox", home_multiplier),
        "away": side("New York Yankees", away_multiplier),
        "legacy_bullpen_blocked": True,
    }


def _seed_current(cache, *, bf, xwoba, as_of="2024-04-08T23:59:59Z"):
    cache.save_record(
        namespace=BullpenPITNamespaces.TEAM_BULLPEN_ROLLING,
        entity_id="BOS",
        season=2024,
        as_of_date=as_of,
        source=BullpenPITSources.TEAM_BULLPEN_ROLLING,
        source_fingerprint="current-fp",
        data={
            "relief_batters_faced": bf,
            "xwoba_against": xwoba,
            "woba_against": xwoba,
            "k_pct": 0.25,
            "bb_pct": 0.08,
            "k_minus_bb_pct": 0.17,
            "barrel_per_contact": 0.08,
            "sample_size_status": "thin" if bf < 200 else "sufficient",
            "source_window_start_date": "2024-03-28",
            "source_window_end_date": as_of[:10],
            "starter_pitches_included": 0,
        },
    )


def _seed_prior(cache, *, bf, xwoba):
    cache.save_record(
        namespace=BullpenPITNamespaces.TEAM_BULLPEN_PRIOR_BASELINE,
        entity_id="BOS",
        season=2024,
        as_of_date="2023-10-01T23:59:59Z",
        source=BullpenPITSources.TEAM_BULLPEN_PRIOR_BASELINE,
        source_fingerprint="prior-fp",
        data={
            "relief_batters_faced": bf,
            "xwoba_against": xwoba,
            "woba_against": xwoba,
            "k_pct": 0.23,
            "bb_pct": 0.09,
            "k_minus_bb_pct": 0.14,
            "barrel_per_contact": 0.09,
            "sample_size_status": "sufficient",
            "baseline_available": True,
            "minimum_bf_required": 200,
            "source_window_start_date": "2023-03-30",
            "source_window_end_date": "2023-10-01",
            "starter_pitches_included": 0,
        },
    )


def _empty_usage():
    return {
        "games_attempted": 0,
        "current_only_bullpen_pit_uses": 0,
        "current_prior_blended_uses": 0,
        "prior_only_baseline_uses": 0,
        "neutral_bullpen_uses": 0,
        "missing_current_bullpen": 0,
        "current_thin_bullpen": 0,
        "legacy_bullpen_calls_blocked": 0,
        "future_snapshot_violations": 0,
        "same_day_violations": 0,
        "missing_fingerprints": 0,
        "duplicate_pit_keys": 0,
        "starter_contamination_violations": 0,
        "samples": [],
    }


class _FakeAPI:
    def __init__(self):
        self.bullpen_calls = []

    def get_team_offensive_stats(self, team_id, season):
        return {}

    def get_team_pitching_stats(self, team_id, season):
        return {}

    def get_bullpen_era(self, team_id, season):
        self.bullpen_calls.append((team_id, season))
        raise AssertionError("get_bullpen_era must be blocked")

    def get_bullpen_era_if_called(self, team_id, season):
        return None


class _FakeIntegrator:
    def _fetch_team_rpg(self, team_id, season):
        return 4.5

    def get_team_lambda(self, team_name, rpg, team_id=None):
        return rpg
