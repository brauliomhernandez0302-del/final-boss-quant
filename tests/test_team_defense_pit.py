import inspect
import sqlite3
from pathlib import Path

import backtest_and_retrain
from modules.baseball_module.advanced_pit_enrichment import (
    PITCache,
    RawSavantEventsCache,
    RawSavantTeamDefenseEvent,
    TeamDefenseDailySnapshotBuilder,
    TeamDefensePITBuilder,
    TeamDefensePITNamespaces,
    TeamDefensePITSources,
    TeamDefensePriorBaseline,
    adapt_defense_pit_snapshot,
    fielding_team_for_event,
)


def test_fielding_team_attribution_top_and_bottom():
    top = _projected_event(inning_topbot="Top", batting_team="BOS")
    bottom = _projected_event(inning_topbot="Bot", batting_team="NYY")

    assert fielding_team_for_event(top) == "NYY"
    assert fielding_team_for_event(bottom) == "BOS"


def test_fielding_team_attribution_rejects_invalid_half_or_batting_team():
    assert fielding_team_for_event(_projected_event(inning_topbot="Middle")) is None
    assert (
        fielding_team_for_event(_projected_event(inning_topbot="Top", batting_team="NYY"))
        is None
    )


def test_contact_adjusted_proxy_inclusions_exclusions_and_missing_xba(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _save(
        raw_db,
        [
            _event(1, events="field_out", xba=0.20),
            _event(2, events="single", xba=0.70),
            _event(3, events="field_out", xba=None),
            _event(4, events="home_run", xba=0.95),
            _event(5, events="strikeout", xba=None),
            _event(6, events="walk", xba=None),
            _event(7, events="hit_by_pitch", xba=None),
            _event(8, events="intent_walk", xba=None),
            _event(9, events="catcher_interf", xba=None),
        ],
    )

    result = TeamDefensePITBuilder(
        raw_cache_db=raw_db, pit_cache_db=pit_db
    ).aggregate_window(start_date="2024-03-28", end_date="2024-03-28")

    row = result.rows["NYY"]
    assert row.bip_count == 3
    assert row.xba_bip_count == 2
    assert row.missing_xba_bip_count == 1
    assert row.outs_on_bip == 2
    assert row.expected_outs == 1.1
    assert row.contact_adjusted_outs == -0.1
    assert row.contact_adjusted_defense_proxy == -0.05
    assert result.excluded_home_runs == 1
    assert result.excluded_non_bip_events == 5


def test_duplicate_plate_appearance_contributes_once_using_terminal_pitch(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _save(
        raw_db,
        [
            _event(1, pitch_number=1, events="single", xba=0.80),
            _event(1, pitch_number=2, events="field_out", xba=0.20),
        ],
    )

    result = TeamDefensePITBuilder(
        raw_cache_db=raw_db, pit_cache_db=pit_db
    ).aggregate_window(start_date="2024-03-28", end_date="2024-03-28")

    assert result.duplicate_pa_rows == 1
    assert result.rows["NYY"].xba_bip_count == 1
    assert result.rows["NYY"].contact_adjusted_defense_proxy == 0.2


def test_current_pit_persistence_has_windows_fingerprint_and_status(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _save(raw_db, [_event(i, events="field_out", xba=0.25) for i in range(1, 102)])

    rows = TeamDefensePITBuilder(
        raw_cache_db=raw_db, pit_cache_db=pit_db
    ).persist_cutoff(
        season=2024,
        season_start_date="2024-03-28",
        as_of_date="2024-04-08",
        fetched_at="2026-06-27T12:00:00Z",
    )

    row = rows["NYY"]
    assert row["source_window_start_date"] == "2024-03-28"
    assert row["source_window_end_date"] == "2024-04-08"
    assert row["sample_size_status"] == "ok"
    assert row["source_fingerprint"].startswith("savant:raw:team_defense:")


def test_future_and_same_day_snapshot_rejected_for_game_cutoff(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_current(cache, "2024-04-08T23:59:59Z", proxy=0.01)
    _seed_current(cache, "2024-04-09T00:00:00Z", proxy=0.99)

    snapshot = TeamDefenseDailySnapshotBuilder(cache).build_for_game(
        team_id="NYY", season=2024, game_date="2024-04-09"
    )

    assert snapshot["requested_as_of_date"] == "2024-04-08T23:59:59Z"
    assert snapshot["current_as_of_date"] == "2024-04-08T23:59:59+00:00"
    assert snapshot["contact_adjusted_defense_proxy"] == 0.01


def test_only_same_day_snapshot_returns_no_current_snapshot(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_current(cache, "2024-04-09T00:00:00Z", proxy=0.99)

    snapshot = TeamDefenseDailySnapshotBuilder(cache).build_for_game(
        team_id="NYY", season=2024, game_date="2024-04-09"
    )

    assert snapshot["current_defense_found"] is False
    assert snapshot["current_as_of_date"] is None


def test_prior_baseline_uses_only_2023_window_for_target_2024(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _save(
        raw_db,
        [
            _event(1, game_date="2023-04-01", events="field_out", xba=0.20),
            _event(2, game_date="2024-04-01", events="single", xba=0.90),
        ],
    )

    result = TeamDefensePriorBaseline(
        raw_cache_db=raw_db, pit_cache_db=pit_db
    ).persist_prior_baseline(
        season=2024,
        prior_season=2023,
        prior_season_start_date="2023-03-30",
        prior_season_end_date="2023-10-01",
    )

    row = result.rows["NYY"]
    assert row["source_window_start_date"] == "2023-03-30"
    assert row["source_window_end_date"] == "2023-10-01"
    assert row["bip_count"] == 1
    assert row["contact_adjusted_defense_proxy"] == 0.2


def test_prior_baseline_rejects_window_outside_2023(tmp_path):
    builder = TeamDefensePriorBaseline(
        raw_cache_db=tmp_path / "raw.db", pit_cache_db=tmp_path / "pit.db"
    )

    try:
        builder.persist_prior_baseline(
            season=2024,
            prior_season=2023,
            prior_season_start_date="2023-03-30",
            prior_season_end_date="2024-04-01",
        )
    except ValueError as exc:
        assert "inside prior_season" in str(exc)
    else:
        raise AssertionError("cross-season prior window was accepted")


def test_prior_baseline_uses_projected_one_pass_not_full_raw_decoding(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _save(raw_db, [_event(1, game_date="2023-04-01", events="field_out", xba=0.2)])
    spy = _SpyRawCache(raw_db)
    builder = TeamDefensePriorBaseline(
        raw_cache_db=raw_db,
        pit_cache_db=pit_db,
        raw_cache=spy,
    )

    builder.persist_prior_baseline(
        season=2024,
        prior_season=2023,
        prior_season_start_date="2023-03-30",
        prior_season_end_date="2023-10-01",
    )

    assert spy.projected_queries == 1
    assert spy.full_queries == 0


def test_current_overrides_prior_and_blends_toward_it(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_current(cache, "2024-04-08T23:59:59Z", proxy=0.04, xba_bip_count=500)
    _seed_prior(cache, proxy=-0.02)
    snapshot = TeamDefenseDailySnapshotBuilder(cache).build_for_game(
        team_id="NYY", season=2024, game_date="2024-04-09"
    )

    adapted = adapt_defense_pit_snapshot(snapshot)

    assert adapted["provenance_source"] == "current_defense_pit"
    assert adapted["contact_adjusted_defense_proxy"] == 0.01
    assert adapted["defense_multiplier"] == 0.99


def test_prior_overrides_neutral_fallback(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_prior(cache, proxy=0.025)
    snapshot = TeamDefenseDailySnapshotBuilder(cache).build_for_game(
        team_id="NYY", season=2024, game_date="2024-04-09"
    )

    adapted = adapt_defense_pit_snapshot(snapshot)

    assert adapted["provenance_source"] == "prior_season_defense_baseline"
    assert adapted["contact_adjusted_defense_proxy"] == 0.025
    assert adapted["defense_multiplier"] == 0.975
    assert adapted["bip_count"] == 4000
    assert adapted["xba_bip_count"] == 3900
    assert adapted["sample_size_status"] == "prior_baseline"


def test_neutral_fallback_is_exactly_no_adjustment(tmp_path):
    snapshot = TeamDefenseDailySnapshotBuilder(cache_db=tmp_path / "pit.db").build_for_game(
        team_id="NYY", season=2024, game_date="2024-04-09"
    )

    adapted = adapt_defense_pit_snapshot(snapshot)

    assert adapted["provenance_source"] == "neutral_defense_adjustment"
    assert adapted["contact_adjusted_defense_proxy"] == 0.0
    assert adapted["defense_multiplier"] == 1.0
    assert adapted["sample_size_status"] == "neutral"
    assert adapted["provenance"]["full_season_der_fallback"] is False


def test_all_star_pseudo_teams_excluded(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _save(
        raw_db,
        [
            _event(1, home_team="AL", away_team="NL", inning_topbot="Top"),
            _event(2, home_team="AL", away_team="NL", inning_topbot="Bot"),
        ],
    )

    result = TeamDefensePITBuilder(
        raw_cache_db=raw_db, pit_cache_db=pit_db
    ).aggregate_window(start_date="2024-03-28", end_date="2024-03-28")

    assert result.rows == {}
    assert result.excluded_non_mlb_teams == ("AL", "NL")


def test_no_full_season_der_call_and_default_backtest_untouched():
    builder_source = inspect.getsource(TeamDefensePITBuilder)
    adapter_source = inspect.getsource(adapt_defense_pit_snapshot)

    assert "get_team_pitching_stats" not in builder_source
    assert "get_team_pitching_stats" not in adapter_source
    assert inspect.signature(backtest_and_retrain.run_pipeline).parameters[
        "use_defense_pit"
    ].default is False
    assert inspect.signature(backtest_and_retrain.build_game_data).parameters[
        "use_team_full_season_defense"
    ].default is True


def test_persistence_is_idempotent_without_duplicate_pit_keys(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _save(raw_db, [_event(1)])
    builder = TeamDefensePITBuilder(raw_cache_db=raw_db, pit_cache_db=pit_db)
    kwargs = dict(
        season=2024,
        season_start_date="2024-03-28",
        as_of_date="2024-03-28",
        fetched_at="2026-06-27T12:00:00Z",
    )

    builder.persist_cutoff(**kwargs)
    builder.persist_cutoff(**kwargs)

    with sqlite3.connect(pit_db) as conn:
        assert conn.execute("select count(*) from pit_metric_cache").fetchone()[0] == 1


class _SpyRawCache(RawSavantEventsCache):
    def __init__(self, db_path):
        super().__init__(db_path)
        self.projected_queries = 0
        self.full_queries = 0

    def iter_team_defense_events_by_date_range(self, **kwargs):
        self.projected_queries += 1
        yield from super().iter_team_defense_events_by_date_range(**kwargs)

    def get_events_by_date_range(self, **kwargs):
        self.full_queries += 1
        return super().get_events_by_date_range(**kwargs)


def _projected_event(**overrides):
    values = {
        "game_date": "2024-03-28",
        "game_pk": 1,
        "at_bat_number": 1,
        "pitch_number": 1,
        "events": "field_out",
        "home_team": "NYY",
        "away_team": "BOS",
        "inning_topbot": "Top",
        "batting_team": None,
        "estimated_ba_using_speedangle": 0.2,
        "bb_type": "ground_ball",
    }
    values.update(overrides)
    return RawSavantTeamDefenseEvent(**values)


def _event(
    at_bat_number,
    *,
    game_date="2024-03-28",
    game_pk=1,
    pitch_number=1,
    events="field_out",
    xba=0.20,
    home_team="NYY",
    away_team="BOS",
    inning_topbot="Top",
):
    return {
        "game_date": game_date,
        "game_pk": game_pk,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "pitcher": 100,
        "batter": 200,
        "events": events,
        "launch_speed": 95.0 if xba is not None else None,
        "launch_angle": 10.0 if xba is not None else None,
        "estimated_ba_using_speedangle": xba,
        "estimated_woba_using_speedangle": xba,
        "woba_value": 0.0,
        "woba_denom": 1.0,
        "launch_speed_angle": 4,
        "home_team": home_team,
        "away_team": away_team,
        "inning_topbot": inning_topbot,
        "bb_type": "ground_ball" if events not in {"walk", "strikeout"} else "",
    }


def _save(raw_db: Path, events):
    RawSavantEventsCache(raw_db).save_events(
        events,
        source_fingerprint="raw-test-fingerprint",
        fetched_at="2026-06-27T12:00:00Z",
    )


def _seed_current(cache, as_of_date, *, proxy, xba_bip_count=100):
    cache.save_record(
        namespace=TeamDefensePITNamespaces.TEAM_DEFENSE_ROLLING,
        entity_id="NYY",
        season=2024,
        as_of_date=as_of_date,
        source=TeamDefensePITSources.TEAM_DEFENSE_ROLLING,
        source_fingerprint="current-fingerprint",
        data={
            "contact_adjusted_defense_proxy": proxy,
            "bip_count": xba_bip_count,
            "xba_bip_count": xba_bip_count,
            "sample_size_status": "ok",
            "source_window_start_date": "2024-03-28",
            "source_window_end_date": as_of_date[:10],
        },
    )


def _seed_prior(cache, *, proxy):
    cache.save_record(
        namespace=TeamDefensePITNamespaces.TEAM_DEFENSE_PRIOR_BASELINE,
        entity_id="NYY",
        season=2024,
        as_of_date="2023-10-01T23:59:59Z",
        source=TeamDefensePITSources.TEAM_DEFENSE_PRIOR_BASELINE,
        source_fingerprint="prior-fingerprint",
        data={
            "contact_adjusted_defense_proxy": proxy,
            "bip_count": 4000,
            "xba_bip_count": 3900,
            "source_window_start_date": "2023-03-30",
            "source_window_end_date": "2023-10-01",
        },
    )
