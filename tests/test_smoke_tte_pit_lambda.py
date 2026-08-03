import inspect
import json
import sqlite3

from modules.baseball_module.advanced_pit_enrichment import PITCache, RawSavantEventsCache
from modules.baseball_module.advanced_pit_enrichment.tte_daily_snapshot_builder import (
    TTEPITNamespaces,
    TTEPITSources,
    previous_day_cutoff_for_game_date,
)
from scripts import smoke_tte_pit_lambda as smoke


def test_script_helper_produces_report_successfully(tmp_path):
    args = _smoke_args(tmp_path)

    report = smoke.run_smoke(args)

    assert args.report.exists()
    written = json.loads(args.report.read_text())
    assert written["final_classification"] == "PASS"
    assert report["raw_event_counts"]["prior_window"] == 6
    assert report["raw_event_counts"]["current_window"] == 12
    assert report["prior_baseline_rows"]["count"] == 2
    assert report["sample_rows"][0]["lambda_offense"] is not None
    assert report["lambda_min_median_max"]["min"] >= 3.0
    assert report["lambda_min_median_max"]["max"] <= 7.0


def test_previous_day_cutoff_is_respected(tmp_path):
    report = smoke.run_smoke(_smoke_args(tmp_path))

    sample = report["sample_rows"][0]
    assert sample["requested_as_of_date"] == previous_day_cutoff_for_game_date("2024-04-03")
    assert sample["team_offense_as_of_date"] == "2024-04-02T00:00:00+00:00"
    assert report["cutoff_policy_violations"] == []


def test_future_pit_row_is_detected(tmp_path):
    args = _smoke_args(tmp_path)
    smoke.run_smoke(args)
    cache = PITCache(args.pit_db)
    _seed_team_rolling(cache, as_of_date="2024-04-04T00:00:00Z")

    report = smoke.run_smoke(args)

    assert report["future_violations"]
    assert report["final_classification"] == "FAIL"


def test_same_day_row_is_detected(tmp_path):
    args = _smoke_args(tmp_path)
    smoke.run_smoke(args)
    cache = PITCache(args.pit_db)
    _seed_team_rolling(cache, as_of_date="2024-04-03T00:00:00Z")

    report = smoke.run_smoke(args)

    assert report["same_day_or_later_violations"]
    assert report["final_classification"] == "FAIL"


def test_missing_fingerprint_is_detected(tmp_path):
    args = _smoke_args(tmp_path)
    initial_report = smoke.run_smoke(args)
    with sqlite3.connect(args.pit_db) as conn:
        conn.execute(
            """
            UPDATE pit_metric_cache
            SET source_fingerprint = ''
            WHERE namespace = ?
            LIMIT 1
            """,
            (TTEPITNamespaces.TEAM_OFFENSE_ROLLING,),
        )

    violations = smoke._collect_violations(
        pit_db=args.pit_db,
        target_season=args.target_season,
        prior_season=args.prior_season,
        sample_rows=initial_report["sample_rows"],
    )

    assert violations["missing_fingerprint_violations"]


def test_weight_sum_violation_is_detected(monkeypatch, tmp_path):
    original = smoke.adapt_tte_pit_snapshot_to_lambda

    def bad_adapter(snapshot):
        result = original(snapshot)
        if result["lambda_offense"] is not None:
            result["blend_current_weight"] = 0.7
            result["blend_prior_weight"] = 0.7
        return result

    monkeypatch.setattr(smoke, "adapt_tte_pit_snapshot_to_lambda", bad_adapter)

    report = smoke.run_smoke(_smoke_args(tmp_path))

    assert report["weight_sum_violations"]
    assert report["final_classification"] == "FAIL"


def test_out_of_clamp_lambda_is_detected(monkeypatch, tmp_path):
    original = smoke.adapt_tte_pit_snapshot_to_lambda

    def bad_adapter(snapshot):
        result = original(snapshot)
        if result["lambda_offense"] is not None:
            result["lambda_offense"] = 8.25
        return result

    monkeypatch.setattr(smoke, "adapt_tte_pit_snapshot_to_lambda", bad_adapter)

    report = smoke.run_smoke(_smoke_args(tmp_path))

    assert report["lambda_clamp_violations"]
    assert report["final_classification"] == "FAIL"


def test_all_star_pseudo_team_is_excluded(tmp_path):
    args = _smoke_args(tmp_path, include_all_star=True)

    report = smoke.run_smoke(args)

    assert report["all_star_pseudo_team_violations"] == []
    assert "AL" not in _pit_entity_ids(args.pit_db)
    assert "NL" not in _pit_entity_ids(args.pit_db)


def test_no_live_backtest_run_module_or_true_talent_imports():
    source = inspect.getsource(smoke)

    assert "import app" not in source
    assert "from app" not in source
    assert "backtest_and_retrain" not in source
    assert "run_module" not in source
    assert "true_talent_engine" not in source


def _smoke_args(tmp_path, *, include_all_star=False):
    prior_raw_db = tmp_path / "prior_raw.db"
    current_raw_db = tmp_path / "current_raw.db"
    pit_db = tmp_path / "pit.db"
    report = tmp_path / "report.json"
    _cache_with_events(
        prior_raw_db,
        _team_events("147", "2023-04-01", start_pk=100)
        + _team_events("158", "2023-04-01", start_pk=200)
        + (_team_events("AL", "2023-07-11", start_pk=300) if include_all_star else []),
    )
    _cache_with_events(
        current_raw_db,
        _team_events("147", "2024-04-01", start_pk=400)
        + _team_events("158", "2024-04-01", start_pk=500)
        + _team_events("147", "2024-04-02", start_pk=600)
        + _team_events("158", "2024-04-02", start_pk=700)
        + (_team_events("NL", "2024-04-02", start_pk=800) if include_all_star else []),
    )
    return smoke.SmokeArgs(
        prior_raw_db=prior_raw_db,
        current_raw_db=current_raw_db,
        pit_db=pit_db,
        report=report,
        target_season=2024,
        prior_season=2023,
        current_start_date="2024-04-01",
        current_end_date="2024-04-02",
        sample_game_dates=["2024-04-03"],
        sample_team_ids=["147", "158"],
        season_start_date="2024-04-01",
    )


def _cache_with_events(db_path, events):
    cache = RawSavantEventsCache(db_path)
    cache.save_events(events, source_fingerprint="test-source", fetched_at="2026-06-24T12:00:00Z")
    return cache


def _team_events(team_id, game_date, *, start_pk):
    return [
        _event(
            game_date=game_date,
            game_pk=start_pk,
            at_bat_number=1,
            pitch_number=1,
            batting_team=team_id,
            events="single",
            launch_speed=100,
            launch_angle=20,
            launch_speed_angle=6,
            estimated_woba_using_speedangle=0.4,
            woba_value=0.9,
            woba_denom=1,
        ),
        _event(
            game_date=game_date,
            game_pk=start_pk,
            at_bat_number=2,
            pitch_number=1,
            batting_team=team_id,
            events="walk",
            launch_speed=None,
            launch_angle=None,
            launch_speed_angle=None,
            estimated_woba_using_speedangle=None,
            woba_value=0.7,
            woba_denom=1,
        ),
        _event(
            game_date=game_date,
            game_pk=start_pk,
            at_bat_number=3,
            pitch_number=1,
            batting_team=team_id,
            events="strikeout",
            launch_speed=None,
            launch_angle=None,
            launch_speed_angle=None,
            estimated_woba_using_speedangle=None,
            woba_value=0.0,
            woba_denom=1,
        ),
    ]


def _event(
    *,
    game_date,
    game_pk,
    at_bat_number,
    pitch_number,
    batting_team,
    events,
    launch_speed,
    launch_angle,
    launch_speed_angle,
    estimated_woba_using_speedangle,
    woba_value,
    woba_denom,
):
    return {
        "game_date": game_date,
        "game_pk": game_pk,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "pitcher": 900,
        "batter": 101 + at_bat_number,
        "events": events,
        "launch_speed": launch_speed,
        "launch_angle": launch_angle,
        "estimated_woba_using_speedangle": estimated_woba_using_speedangle,
        "woba_value": woba_value,
        "woba_denom": woba_denom,
        "launch_speed_angle": launch_speed_angle,
        "batting_team": batting_team,
    }


def _seed_team_rolling(cache, *, as_of_date, team_id=147):
    cache.save_record(
        namespace=TTEPITNamespaces.TEAM_OFFENSE_ROLLING,
        entity_id=team_id,
        season=2024,
        as_of_date=as_of_date,
        source=TTEPITSources.TEAM_OFFENSE_ROLLING,
        source_fingerprint="manual-future-fp",
        data={
            "est_woba": 0.999,
            "woba": 0.999,
            "barrel_pa": 0.2,
            "bb_pct": 0.2,
            "k_pct": 0.1,
            "pa": 99,
            "bip": 50,
            "batted_ball_count": 50,
            "source_window_start_date": "2024-04-01",
            "source_window_end_date": as_of_date[:10],
        },
    )


def _pit_entity_ids(pit_db):
    with sqlite3.connect(pit_db) as conn:
        return {row[0] for row in conn.execute("SELECT DISTINCT entity_id FROM pit_metric_cache")}
