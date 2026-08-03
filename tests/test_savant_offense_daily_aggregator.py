import inspect
import sqlite3

from modules.baseball_module.advanced_pit_enrichment import (
    PITCache,
    RawSavantEventsCache,
    SavantBatterRollingPITPersistence,
    SavantOffenseDailyAggregator,
    SavantOffenseRollingBuilder,
    SavantTeamOffenseRollingPITPersistence,
    TTEPITNamespaces,
    TTEPITSources,
    previous_day_cutoff_for_game_date,
)


def test_daily_batter_aggregation_works(tmp_path):
    cache = _cache_with_events(
        tmp_path,
        [
            _event(
                batter=101,
                launch_speed=100,
                launch_angle=20,
                estimated_woba_using_speedangle=0.5,
                woba_value=0.9,
                woba_denom=1,
                launch_speed_angle=6,
            ),
            _event(
                at_bat_number=2,
                batter=101,
                launch_speed=80,
                launch_angle=40,
                estimated_woba_using_speedangle=0.1,
                woba_value=0.0,
                woba_denom=1,
                launch_speed_angle=3,
            ),
        ],
    )

    rows = SavantOffenseDailyAggregator(cache=cache).aggregate_batters_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.game_date == "2024-04-01"
    assert row.batter == 101
    assert row.plate_appearances == 2
    assert row.batted_ball_count == 2
    assert row.est_woba == 0.3
    assert row.est_woba_count == 2
    assert row.woba == 0.45
    assert row.woba_numerator == 0.9
    assert row.woba_denominator == 2
    assert row.barrel_count == 1
    assert row.brl_percent == 50.0
    assert row.barrel_pa == 0.5
    assert row.bb_count == 0
    assert row.k_count == 0
    assert row.bb_pct == 0.0
    assert row.k_pct == 0.0
    assert row.avg_hit_speed == 90.0
    assert row.ev95plus == 1
    assert row.ev95percent == 50.0
    assert row.sweet_spot_count == 1
    assert row.sweet_spot_denominator == 2
    assert row.sweet_spot_pct == 50.0


def test_daily_team_aggregation_works(tmp_path):
    cache = _cache_with_events(
        tmp_path,
        [
            _event(
                game_pk=1,
                batter=101,
                home_team="NYY",
                away_team="BOS",
                inning_topbot="Top",
                launch_speed=100,
                launch_angle=20,
                woba_value=1,
                woba_denom=1,
                launch_speed_angle=6,
            ),
            _event(
                game_pk=1,
                at_bat_number=2,
                batter=102,
                home_team="NYY",
                away_team="BOS",
                inning_topbot="Bot",
                launch_speed=90,
                launch_angle=10,
                woba_value=0,
                woba_denom=1,
                launch_speed_angle=3,
            ),
            _event(
                game_pk=2,
                at_bat_number=1,
                pitch_number=1,
                batter=103,
                batting_team="LAD",
                launch_speed=95,
                launch_angle=20,
                woba_value=0.7,
                woba_denom=1,
                launch_speed_angle=6,
            ),
        ],
    )

    result = SavantOffenseDailyAggregator(cache=cache).aggregate_teams_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )
    rows = {row.batting_team: row for row in result.rows}

    assert result.missing_team_rows == 0
    assert set(rows) == {"BOS", "NYY", "LAD"}
    assert rows["BOS"].plate_appearances == 1
    assert rows["BOS"].woba == 1.0
    assert rows["NYY"].woba == 0.0
    assert rows["LAD"].brl_percent == 100.0
    assert rows["LAD"].barrel_pa == 1.0


def test_daily_bb_k_and_barrel_pa_aggregation(tmp_path):
    cache = _cache_with_events(
        tmp_path,
        [
            _event(at_bat_number=1, events="walk", woba_value=0.7, woba_denom=1),
            _event(at_bat_number=2, events="strikeout", woba_value=0.0, woba_denom=1),
            _event(
                at_bat_number=3,
                events="single",
                launch_speed=102,
                launch_angle=20,
                launch_speed_angle=6,
                woba_value=0.9,
                woba_denom=1,
            ),
        ],
    )

    row = SavantOffenseDailyAggregator(cache=cache).aggregate_batters_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )[0]

    assert row.plate_appearances == 3
    assert row.bb_count == 1
    assert row.k_count == 1
    assert row.barrel_count == 1
    assert row.bb_pct == 1 / 3
    assert row.k_pct == 1 / 3
    assert row.barrel_pa == 1 / 3


def test_pa_is_not_overcounted_by_pitch_level_rows(tmp_path):
    cache = _cache_with_events(
        tmp_path,
        [
            _event(at_bat_number=1, pitch_number=1, events=None, woba_value=None, woba_denom=None),
            _event(at_bat_number=1, pitch_number=2, events=None, woba_value=None, woba_denom=None),
            _event(at_bat_number=1, pitch_number=3, events="walk", woba_value=0.7, woba_denom=1),
        ],
    )

    row = SavantOffenseDailyAggregator(cache=cache).aggregate_batters_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )[0]

    assert row.plate_appearances == 1
    assert row.bb_count == 1
    assert row.bb_pct == 1.0


def test_strikeout_double_play_counted_as_k(tmp_path):
    cache = _cache_with_events(
        tmp_path,
        [_event(events="strikeout_double_play", woba_value=0, woba_denom=1)],
    )

    row = SavantOffenseDailyAggregator(cache=cache).aggregate_batters_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )[0]

    assert row.plate_appearances == 1
    assert row.k_count == 1
    assert row.k_pct == 1.0


def test_walks_counted_as_bb_and_intentional_walk_policy_explicit(tmp_path):
    cache = _cache_with_events(
        tmp_path,
        [
            _event(at_bat_number=1, events="walk", woba_value=0.7, woba_denom=1),
            _event(at_bat_number=2, events="intent_walk", woba_value=0.0, woba_denom=0),
        ],
    )

    row = SavantOffenseDailyAggregator(cache=cache).aggregate_batters_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )[0]

    assert row.plate_appearances == 2
    assert row.bb_count == 1
    assert row.bb_pct == 0.5


def test_rolling_aggregation_excludes_future_dates(tmp_path):
    cache = _cache_with_events(
        tmp_path,
        [
            _event(game_date="2024-04-01", batter=101, woba_value=1, woba_denom=1),
            _event(game_date="2024-04-02", game_pk=2, batter=101, woba_value=0, woba_denom=1),
            _event(game_date="2024-04-03", game_pk=3, batter=101, woba_value=1, woba_denom=1),
        ],
    )

    rows = SavantOffenseRollingBuilder(cache=cache).build_batters_for_as_of_date(
        season_start_date="2024-04-01",
        as_of_date="2024-04-02",
    )

    assert rows[101].plate_appearances == 2
    assert rows[101].woba == 0.5


def test_previous_day_cutoff_compatibility(tmp_path):
    cache = _cache_with_events(
        tmp_path,
        [
            _event(game_date="2024-04-08", batter=101, woba_value=1, woba_denom=1),
            _event(game_date="2024-04-09", game_pk=2, batter=101, woba_value=0, woba_denom=1),
        ],
    )
    cutoff = previous_day_cutoff_for_game_date("2024-04-09")[:10]

    rows = SavantOffenseRollingBuilder(cache=cache).build_batters_for_as_of_date(
        season_start_date="2024-04-08",
        as_of_date=cutoff,
    )

    assert cutoff == "2024-04-08"
    assert rows[101].plate_appearances == 1
    assert rows[101].woba == 1.0


def test_missing_batting_team_is_reported_not_guessed(tmp_path):
    cache = _cache_with_events(
        tmp_path,
        [
            _event(batter=101, home_team=None, away_team=None, inning_topbot=None),
            _event(at_bat_number=2, batter=102, batting_team="NYY"),
        ],
    )

    result = SavantOffenseDailyAggregator(cache=cache).aggregate_teams_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )

    assert result.missing_team_rows == 1
    assert [row.batting_team for row in result.rows] == ["NYY"]


def test_zero_denominators_produce_none(tmp_path):
    cache = _cache_with_events(
        tmp_path,
        [
            _event(
                batter=101,
                events=None,
                launch_speed=None,
                launch_angle=None,
                estimated_woba_using_speedangle=None,
                woba_value=0,
                woba_denom=0,
                launch_speed_angle=None,
            )
        ],
    )

    row = SavantOffenseDailyAggregator(cache=cache).aggregate_batters_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )[0]

    assert row.plate_appearances == 0
    assert row.woba is None
    assert row.woba_numerator is None
    assert row.woba_denominator is None
    assert row.est_woba is None
    assert row.brl_percent is None
    assert row.ev95percent is None
    assert row.sweet_spot_pct is None
    assert row.bb_pct is None
    assert row.k_pct is None
    assert row.barrel_pa is None


def test_true_zero_barrel_rate_with_valid_bip_denominator_returns_zero(tmp_path):
    cache = _cache_with_events(
        tmp_path,
        [
            _event(
                batter=101,
                launch_speed=88,
                launch_angle=15,
                woba_value=0,
                woba_denom=1,
                launch_speed_angle=3,
            )
        ],
    )

    row = SavantOffenseDailyAggregator(cache=cache).aggregate_batters_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )[0]

    assert row.batted_ball_count == 1
    assert row.barrel_count == 0
    assert row.brl_percent == 0.0
    assert row.barrel_pa == 0.0


def test_persistence_writes_canonical_namespaces(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _cache_with_events(
        tmp_path,
        [
            _event(batter=101, batting_team="NYY", woba_value=1, woba_denom=1),
        ],
        db_name="raw.db",
    )

    batter_rows = SavantBatterRollingPITPersistence(
        raw_cache_db=raw_db,
        pit_cache_db=pit_db,
    ).persist_cutoff(
        season=2024,
        season_start_date="2024-04-01",
        as_of_date="2024-04-01",
    )
    team_rows = SavantTeamOffenseRollingPITPersistence(
        raw_cache_db=raw_db,
        pit_cache_db=pit_db,
    ).persist_cutoff(
        season=2024,
        season_start_date="2024-04-01",
        as_of_date="2024-04-01",
    )

    assert set(batter_rows) == {101}
    assert set(team_rows) == {"NYY"}

    con = sqlite3.connect(pit_db)
    rows = con.execute(
        "select namespace, source, count(*) from pit_metric_cache group by namespace, source"
    ).fetchall()
    con.close()
    assert rows == [
        (TTEPITNamespaces.BATTER_ROLLING, TTEPITSources.BATTER_ROLLING, 1),
        (TTEPITNamespaces.TEAM_OFFENSE_ROLLING, TTEPITSources.TEAM_OFFENSE_ROLLING, 1),
    ]


def test_upsert_idempotency_works(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _cache_with_events(
        tmp_path,
        [_event(batter=101, batting_team="NYY", woba_value=1, woba_denom=1)],
        db_name="raw.db",
    )
    persistence = SavantBatterRollingPITPersistence(raw_cache_db=raw_db, pit_cache_db=pit_db)

    for _ in range(2):
        persistence.persist_cutoff(
            season=2024,
            season_start_date="2024-04-01",
            as_of_date="2024-04-01",
        )

    con = sqlite3.connect(pit_db)
    count = con.execute("select count(*) from pit_metric_cache").fetchone()[0]
    con.close()
    assert count == 1


def test_retrieval_latest_lte_requested_as_of_date_works(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _cache_with_events(
        tmp_path,
        [
            _event(game_date="2024-04-01", batter=101, batting_team="NYY", woba_value=1, woba_denom=1),
            _event(game_date="2024-04-02", game_pk=2, batter=101, batting_team="NYY", woba_value=0, woba_denom=1),
        ],
        db_name="raw.db",
    )
    persistence = SavantTeamOffenseRollingPITPersistence(raw_cache_db=raw_db, pit_cache_db=pit_db)
    persistence.persist_cutoff(season=2024, season_start_date="2024-04-01", as_of_date="2024-04-01")
    persistence.persist_cutoff(season=2024, season_start_date="2024-04-01", as_of_date="2024-04-02")

    before_future = persistence.get_latest_team_snapshot(
        team="NYY",
        season=2024,
        requested_as_of_date="2024-04-01T23:59:59Z",
    )

    assert before_future is not None
    assert before_future.as_of_date == "2024-04-01T00:00:00+00:00"
    assert before_future.data["woba"] == 1.0


def test_rolling_bb_k_aggregation(tmp_path):
    cache = _cache_with_events(
        tmp_path,
        [
            _event(game_date="2024-04-01", game_pk=1, events="walk", woba_value=0.7, woba_denom=1),
            _event(game_date="2024-04-02", game_pk=2, events="strikeout", woba_value=0, woba_denom=1),
            _event(game_date="2024-04-03", game_pk=3, events="walk", woba_value=0.7, woba_denom=1),
        ],
    )

    rows = SavantOffenseRollingBuilder(cache=cache).build_batters_for_as_of_date(
        season_start_date="2024-04-01",
        as_of_date="2024-04-02",
    )

    assert rows[101].plate_appearances == 2
    assert rows[101].bb_count == 1
    assert rows[101].k_count == 1
    assert rows[101].bb_pct == 0.5
    assert rows[101].k_pct == 0.5


def test_team_metrics_persisted_and_retrieved_with_tte_inputs(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _cache_with_events(
        tmp_path,
        [
            _event(at_bat_number=1, batting_team="NYY", events="walk", woba_value=0.7, woba_denom=1),
            _event(
                at_bat_number=2,
                batting_team="NYY",
                events="single",
                launch_speed=101,
                launch_angle=20,
                launch_speed_angle=6,
                woba_value=0.9,
                woba_denom=1,
            ),
        ],
        db_name="raw.db",
    )
    persistence = SavantTeamOffenseRollingPITPersistence(raw_cache_db=raw_db, pit_cache_db=pit_db)
    persistence.persist_cutoff(season=2024, season_start_date="2024-04-01", as_of_date="2024-04-01")

    record = persistence.get_latest_team_snapshot(
        team="NYY",
        season=2024,
        requested_as_of_date="2024-04-01T23:59:59Z",
    )

    assert record is not None
    assert record.data["pa"] == 2
    assert record.data["bb_pct"] == 0.5
    assert record.data["k_pct"] == 0.0
    assert record.data["barrel_pa"] == 0.5


def test_module_does_not_import_live_app_backtest_or_run_module():
    import modules.baseball_module.advanced_pit_enrichment.savant_offense_daily_aggregator as module

    source = inspect.getsource(module)
    assert "import app" not in source
    assert "from app" not in source
    assert "backtest_and_retrain" not in source
    assert "run_module" not in source
    assert "true_talent_engine" not in source


def _cache_with_events(tmp_path, events, *, db_name="raw.db"):
    cache = RawSavantEventsCache(tmp_path / db_name)
    cache.save_events(events, source_fingerprint="test-source")
    return cache


def _event(
    *,
    game_date="2024-04-01",
    game_pk=1,
    at_bat_number=1,
    pitch_number=1,
    pitcher=900,
    batter=101,
    events="single",
    launch_speed=90,
    launch_angle=20,
    estimated_woba_using_speedangle=0.4,
    woba_value=0.5,
    woba_denom=1,
    launch_speed_angle=3,
    batting_team=None,
    home_team="NYY",
    away_team="BOS",
    inning_topbot="Bot",
):
    event = {
        "game_date": game_date,
        "game_pk": game_pk,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "pitcher": pitcher,
        "batter": batter,
        "events": events,
        "launch_speed": launch_speed,
        "launch_angle": launch_angle,
        "estimated_woba_using_speedangle": estimated_woba_using_speedangle,
        "woba_value": woba_value,
        "woba_denom": woba_denom,
        "launch_speed_angle": launch_speed_angle,
    }
    if batting_team is not None:
        event["batting_team"] = batting_team
    if home_team is not None:
        event["home_team"] = home_team
    if away_team is not None:
        event["away_team"] = away_team
    if inning_topbot is not None:
        event["inning_topbot"] = inning_topbot
    return event
