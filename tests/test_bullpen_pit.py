import inspect
import sqlite3
from pathlib import Path

import backtest_and_retrain
from modules.baseball_module.advanced_pit_enrichment import (
    BullpenPITNamespaces,
    BullpenPITSources,
    BullpenPriorBaselineBuilder,
    BullpenReliefAppearanceBuilder,
    PITCache,
    RawSavantBullpenPitch,
    RawSavantEventsCache,
    TeamBullpenDailySnapshotBuilder,
    TeamBullpenPITBuilder,
    adapt_bullpen_pit_snapshot,
    fielding_team_for_bullpen_pitch,
)


def test_top_bottom_fielding_team_attribution():
    top = _projected_pitch(inning_topbot="Top")
    bottom = _projected_pitch(inning_topbot="Bot")

    assert fielding_team_for_bullpen_pitch(top) == "NYY"
    assert fielding_team_for_bullpen_pitch(bottom) == "BOS"


def test_first_pitcher_excluded_and_subsequent_pitchers_included(tmp_path):
    raw_db = tmp_path / "raw.db"
    _save(
        raw_db,
        [
            _pitch(1, pitcher=10),
            _pitch(2, pitcher=10),
            _pitch(3, pitcher=20, events="strikeout"),
            _pitch(4, pitcher=30, events="walk"),
        ],
    )

    result = BullpenReliefAppearanceBuilder(raw_cache_db=raw_db).build(
        start_date="2024-04-01", end_date="2024-04-01"
    )

    assert [fact.pitcher_id for fact in result.facts] == [20, 30]
    assert [fact.appearance_order for fact in result.facts] == [1, 2]
    assert all(fact.starter_opener_id_excluded == 10 for fact in result.facts)
    assert result.starter_pitches_excluded == 2
    assert result.starter_pitches_included == 0


def test_opener_and_bullpen_game_follow_first_pitcher_rule(tmp_path):
    raw_db = tmp_path / "raw.db"
    _save(
        raw_db,
        [
            _pitch(1, pitcher=10),
            _pitch(2, pitcher=20),
            _pitch(3, pitcher=20),
            _pitch(4, pitcher=30),
            _pitch(5, pitcher=40),
        ],
    )

    result = BullpenReliefAppearanceBuilder(raw_cache_db=raw_db).build(
        start_date="2024-04-01", end_date="2024-04-01"
    )

    assert [(fact.pitcher_id, fact.appearance_order) for fact in result.facts] == [
        (20, 1),
        (30, 2),
        (40, 3),
    ]
    assert sum(fact.pitch_count for fact in result.facts) == 4


def test_bullpen_game_never_promotes_later_pitcher_to_starter(tmp_path):
    raw_db = tmp_path / "raw.db"
    _save(
        raw_db,
        [
            _pitch(1, pitcher=101),
            _pitch(2, pitcher=102),
            _pitch(3, pitcher=103),
            _pitch(4, pitcher=104),
            _pitch(5, pitcher=105),
        ],
    )

    facts = BullpenReliefAppearanceBuilder(raw_cache_db=raw_db).build(
        start_date="2024-04-01", end_date="2024-04-01"
    ).facts

    assert [fact.pitcher_id for fact in facts] == [102, 103, 104, 105]
    assert {fact.starter_opener_id_excluded for fact in facts} == {101}


def test_pitcher_can_start_one_game_and_relieve_another(tmp_path):
    raw_db = tmp_path / "raw.db"
    _save(
        raw_db,
        [
            _pitch(1, game_pk=1, pitcher=20),
            _pitch(2, game_pk=1, pitcher=30),
            _pitch(1, game_pk=2, game_date="2024-04-02", pitcher=10),
            _pitch(2, game_pk=2, game_date="2024-04-02", pitcher=20),
        ],
    )

    facts = BullpenReliefAppearanceBuilder(raw_cache_db=raw_db).build(
        start_date="2024-04-01", end_date="2024-04-02"
    ).facts

    assert (1, 20) not in {(fact.game_pk, fact.pitcher_id) for fact in facts}
    assert (2, 20) in {(fact.game_pk, fact.pitcher_id) for fact in facts}


def test_trade_attribution_uses_team_of_each_appearance(tmp_path):
    raw_db = tmp_path / "raw.db"
    _save(
        raw_db,
        [
            _pitch(1, game_pk=1, pitcher=10, home_team="NYY", away_team="BOS"),
            _pitch(2, game_pk=1, pitcher=50, home_team="NYY", away_team="BOS"),
            _pitch(
                1,
                game_pk=2,
                game_date="2024-04-02",
                pitcher=20,
                home_team="BOS",
                away_team="TB",
            ),
            _pitch(
                2,
                game_pk=2,
                game_date="2024-04-02",
                pitcher=50,
                home_team="BOS",
                away_team="TB",
            ),
        ],
    )

    facts = BullpenReliefAppearanceBuilder(raw_cache_db=raw_db).build(
        start_date="2024-04-01", end_date="2024-04-02"
    ).facts

    assert {(fact.team_id, fact.pitcher_id) for fact in facts} == {
        ("NYY", 50),
        ("BOS", 50),
    }


def test_missing_or_inconsistent_role_order_is_rejected(tmp_path):
    raw_db = tmp_path / "raw.db"
    _save(raw_db, [_pitch(1, pitcher=None), _pitch(2, pitcher=20)])

    result = BullpenReliefAppearanceBuilder(raw_cache_db=raw_db).build(
        start_date="2024-04-01", end_date="2024-04-01"
    )

    assert result.facts == ()
    assert result.rejected_reason_counts == {"missing_pitcher_id": 1}

    raw_db_2 = tmp_path / "raw2.db"
    _save(
        raw_db_2,
        [_pitch(1, pitcher=10), _pitch(2, pitcher=20), _pitch(3, pitcher=10)],
    )
    result_2 = BullpenReliefAppearanceBuilder(raw_cache_db=raw_db_2).build(
        start_date="2024-04-01", end_date="2024-04-01"
    )
    assert result_2.facts == ()
    assert result_2.rejected_reason_counts == {
        "noncontiguous_pitcher_reappearance": 1
    }


def test_duplicate_pitch_and_pa_contributions_are_prevented(tmp_path):
    raw_db = tmp_path / "raw.db"
    _save(
        raw_db,
        [
            _pitch(1, pitcher=10),
            _pitch(2, pitcher=20, pitch_number=1, events="single"),
            _pitch(2, pitcher=20, pitch_number=2, events="field_out"),
        ],
    )

    result = BullpenReliefAppearanceBuilder(raw_cache_db=raw_db).build(
        start_date="2024-04-01", end_date="2024-04-01"
    )

    assert len(result.facts) == 1
    assert result.facts[0].batters_faced == 1
    assert result.duplicate_pa_contributions_prevented == 1
    assert result.duplicate_pitch_contributions_prevented == 0


def test_current_snapshot_workload_status_and_namespaces(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    events = [_pitch(1, pitcher=10)]
    events.extend(_pitch(index + 2, pitcher=20) for index in range(199))
    _save(raw_db, events)

    rows = TeamBullpenPITBuilder(
        raw_cache_db=raw_db, pit_cache_db=pit_db
    ).persist_cutoff(
        season=2024,
        season_start_date="2024-04-01",
        as_of_date="2024-04-01",
    )

    row = rows["NYY"]
    assert row["relief_batters_faced"] == 199
    assert row["sample_size_status"] == "thin"
    assert row["pitches_last_1_day"] == 199
    assert row["pitches_last_3_days"] == 199
    assert row["pitches_last_7_days"] == 199
    assert row["last_used_date"] == "2024-04-01"
    with sqlite3.connect(pit_db) as conn:
        namespace, source = conn.execute(
            "select namespace, source from pit_metric_cache"
        ).fetchone()
    assert namespace == BullpenPITNamespaces.TEAM_BULLPEN_ROLLING
    assert source == BullpenPITSources.TEAM_BULLPEN_ROLLING


def test_d_minus_one_cutoff_rejects_same_day_and_future(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_current(cache, "2024-04-08T23:59:59Z", bf=50)
    _seed_current(cache, "2024-04-09T00:00:00Z", bf=999)

    snapshot = TeamBullpenDailySnapshotBuilder(cache).build_for_game(
        team_id="NYY", season=2024, game_date="2024-04-09"
    )

    assert snapshot["requested_as_of_date"] == "2024-04-08T23:59:59Z"
    assert snapshot["current_as_of_date"] == "2024-04-08T23:59:59+00:00"
    assert snapshot["current"]["relief_batters_faced"] == 50


def test_only_same_day_snapshot_is_not_selected(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_current(cache, "2024-04-09T00:00:00Z", bf=999)

    snapshot = TeamBullpenDailySnapshotBuilder(cache).build_for_game(
        team_id="NYY", season=2024, game_date="2024-04-09"
    )

    assert snapshot["current_bullpen_found"] is False


def test_prior_baseline_uses_only_2023_and_requires_200_bf(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    events = [_pitch(1, game_date="2023-04-01", pitcher=10)]
    events.extend(
        _pitch(index + 2, game_date="2023-04-01", pitcher=20)
        for index in range(200)
    )
    events.extend(
        [
            _pitch(1, game_pk=2, game_date="2024-04-01", pitcher=30),
            _pitch(2, game_pk=2, game_date="2024-04-01", pitcher=40),
        ]
    )
    _save(raw_db, events)

    result = BullpenPriorBaselineBuilder(
        raw_cache_db=raw_db, pit_cache_db=pit_db
    ).persist_prior_baseline(
        season=2024,
        prior_season=2023,
        prior_season_start_date="2023-03-30",
        prior_season_end_date="2023-10-01",
    )

    row = result.rows["NYY"]
    assert row["prior_season"] == 2023
    assert row["relief_batters_faced"] == 200
    assert row["baseline_available"] is True
    assert row["source_window_end_date"] == "2023-10-01"


def test_current_then_prior_then_neutral_hierarchy(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    _seed_prior(cache, bf=500, available=True)

    prior_snapshot = TeamBullpenDailySnapshotBuilder(cache).build_for_game(
        team_id="NYY", season=2024, game_date="2024-04-09"
    )
    prior = adapt_bullpen_pit_snapshot(prior_snapshot)
    assert prior["provenance_source"] == "prior_season_bullpen_baseline"
    assert prior["applied_multiplier"] == 1.0

    _seed_current(cache, "2024-04-08T23:59:59Z", bf=10)
    current_snapshot = TeamBullpenDailySnapshotBuilder(cache).build_for_game(
        team_id="NYY", season=2024, game_date="2024-04-09"
    )
    current = adapt_bullpen_pit_snapshot(current_snapshot)
    assert current["provenance_source"] == "current_bullpen_pit"
    assert current["sample_size_status"] == "thin"
    assert current["applied_multiplier"] == 1.0

    neutral_snapshot = TeamBullpenDailySnapshotBuilder(
        cache_db=tmp_path / "empty.db"
    ).build_for_game(team_id="NYY", season=2024, game_date="2024-04-09")
    neutral = adapt_bullpen_pit_snapshot(neutral_snapshot)
    assert neutral["provenance_source"] == "neutral_bullpen_adjustment"
    assert neutral["neutral_fallback"] is True
    assert neutral["applied_multiplier"] == 1.0


def test_fact_persistence_is_idempotent(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _save(raw_db, [_pitch(1, pitcher=10), _pitch(2, pitcher=20)])
    builder = BullpenReliefAppearanceBuilder(
        raw_cache_db=raw_db, pit_cache_db=pit_db
    )
    kwargs = dict(
        season=2024,
        start_date="2024-04-01",
        end_date="2024-04-01",
        fetched_at="2026-06-28T12:00:00Z",
    )

    builder.persist_daily_facts(**kwargs)
    builder.persist_daily_facts(**kwargs)

    with sqlite3.connect(pit_db) as conn:
        assert conn.execute("select count(*) from pit_metric_cache").fetchone()[0] == 1


def test_isolated_from_live_legacy_and_default_backtest():
    from modules.baseball_module.advanced_pit_enrichment import (
        bullpen_pit_adapter,
        bullpen_pit_builder,
        bullpen_prior_baseline,
        bullpen_relief_appearance_builder,
    )

    sources = "\n".join(
        inspect.getsource(module)
        for module in (
            bullpen_pit_adapter,
            bullpen_pit_builder,
            bullpen_prior_baseline,
            bullpen_relief_appearance_builder,
        )
    )
    assert "get_bullpen_era" not in sources
    assert "BullpenEngine" not in sources
    assert "core.run_module" not in sources
    assert "import app" not in sources
    assert "use_bullpen_pit" not in inspect.getsource(backtest_and_retrain)


def _projected_pitch(**overrides):
    values = {
        "game_date": "2024-04-01",
        "game_pk": 1,
        "at_bat_number": 1,
        "pitch_number": 1,
        "pitcher": 10,
        "events": "field_out",
        "launch_speed": 90.0,
        "estimated_woba_using_speedangle": 0.2,
        "woba_value": 0.0,
        "woba_denom": 1.0,
        "launch_speed_angle": 4,
        "home_team": "NYY",
        "away_team": "BOS",
        "inning_topbot": "Top",
        "source_fingerprint": "raw-test-fingerprint",
    }
    values.update(overrides)
    return RawSavantBullpenPitch(**values)


def _pitch(
    at_bat_number,
    *,
    game_pk=1,
    game_date="2024-04-01",
    pitcher=10,
    pitch_number=1,
    events="field_out",
    home_team="NYY",
    away_team="BOS",
    inning_topbot="Top",
):
    return {
        "game_date": game_date,
        "game_pk": game_pk,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "pitcher": pitcher,
        "batter": 1000 + at_bat_number,
        "events": events,
        "launch_speed": 95.0 if events not in {"walk", "strikeout"} else None,
        "launch_angle": 15.0 if events not in {"walk", "strikeout"} else None,
        "estimated_woba_using_speedangle": (
            0.3 if events not in {"walk", "strikeout"} else None
        ),
        "woba_value": 0.0 if events != "walk" else 0.69,
        "woba_denom": 1.0,
        "launch_speed_angle": 6 if events == "home_run" else 4,
        "home_team": home_team,
        "away_team": away_team,
        "inning_topbot": inning_topbot,
    }


def _save(path: Path, events):
    RawSavantEventsCache(path).save_events(
        events,
        source_fingerprint="raw-test-fingerprint",
        fetched_at="2026-06-28T12:00:00Z",
    )


def _seed_current(cache: PITCache, as_of_date: str, *, bf: int):
    cache.save_record(
        namespace=BullpenPITNamespaces.TEAM_BULLPEN_ROLLING,
        entity_id="NYY",
        season=2024,
        as_of_date=as_of_date,
        source=BullpenPITSources.TEAM_BULLPEN_ROLLING,
        source_fingerprint="current-bullpen-fingerprint",
        data={
            "relief_batters_faced": bf,
            "sample_size_status": "thin" if bf < 200 else "sufficient",
            "source_window_start_date": "2024-03-28",
            "source_window_end_date": as_of_date[:10],
            "pitches_last_1_day": 10,
            "appearances_last_1_day": 1,
            "pitches_last_3_days": 30,
            "appearances_last_3_days": 3,
            "pitches_last_7_days": 70,
            "appearances_last_7_days": 7,
            "consecutive_days": 2,
            "last_used_date": as_of_date[:10],
        },
    )


def _seed_prior(cache: PITCache, *, bf: int, available: bool):
    cache.save_record(
        namespace=BullpenPITNamespaces.TEAM_BULLPEN_PRIOR_BASELINE,
        entity_id="NYY",
        season=2024,
        as_of_date="2023-10-01T23:59:59Z",
        source=BullpenPITSources.TEAM_BULLPEN_PRIOR_BASELINE,
        source_fingerprint="prior-bullpen-fingerprint",
        data={
            "relief_batters_faced": bf,
            "baseline_available": available,
            "minimum_bf_required": 200,
            "sample_size_status": "sufficient" if bf >= 200 else "thin",
            "source_window_start_date": "2023-03-30",
            "source_window_end_date": "2023-10-01",
        },
    )
