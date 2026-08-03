import inspect
import sqlite3

from modules.baseball_module.advanced_pit_enrichment import (
    PITCache,
    RawSavantEventsCache,
    SavantOffenseRollingBuilder,
    TTEPITNamespaces,
    TTEPITSources,
    TTEPriorBaselineBuilder,
)


def test_prior_baseline_persisted_in_canonical_namespace(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _cache_with_events(
        raw_db,
        [
            _event(game_date="2023-04-01", batting_team="147", events="walk", woba_value=0.7, woba_denom=1),
            _event(
                game_date="2023-04-01",
                game_pk=2,
                batting_team="147",
                events="single",
                launch_speed=101,
                launch_angle=20,
                launch_speed_angle=6,
                woba_value=0.9,
                woba_denom=1,
            ),
        ],
    )

    result = TTEPriorBaselineBuilder(raw_cache_db=raw_db, pit_cache_db=pit_db).persist_prior_baseline(
        season=2024,
        prior_season_start_date="2023-04-01",
        prior_season_end_date="2023-10-01",
        prior_season=2023,
    )

    assert set(result.rows) == {"147"}
    row = result.rows["147"]
    assert row["season"] == 2024
    assert row["prior_season"] == 2023
    assert row["team_est_woba_prior"] == 0.4
    assert row["team_woba_prior"] == 0.8
    assert row["bb_pct_prior"] == 0.5
    assert row["k_pct_prior"] == 0.0
    assert row["barrel_pa_prior"] == 0.5
    assert row["pa_prior"] == 2
    assert row["baseline_version"] == "tte_prior_baseline_v1"

    con = sqlite3.connect(pit_db)
    namespace, source, count = con.execute(
        "select namespace, source, count(*) from pit_metric_cache group by namespace, source"
    ).fetchone()
    con.close()
    assert namespace == TTEPITNamespaces.TEAM_OFFENSE_PRIOR_BASELINE
    assert source == TTEPITSources.TEAM_OFFENSE_PRIOR_BASELINE
    assert count == 1


def test_prior_baseline_retrieval_latest_lte_requested_date(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _cache_with_events(raw_db, [_event(game_date="2023-04-01", batting_team="147")])
    builder = TTEPriorBaselineBuilder(raw_cache_db=raw_db, pit_cache_db=pit_db)
    builder.persist_prior_baseline(
        season=2024,
        prior_season_start_date="2023-04-01",
        prior_season_end_date="2023-10-01",
    )

    record = builder.get_latest_prior_baseline(
        team_id=147,
        season=2024,
        requested_as_of_date="2024-04-01T23:59:59Z",
    )

    assert record is not None
    assert record.as_of_date == "2023-10-01T23:59:59+00:00"
    assert record.data["prior_season"] == 2023


def test_prior_season_allowed_but_current_future_rows_not_used(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _cache_with_events(
        raw_db,
        [
            _event(game_date="2023-04-01", batting_team="147", woba_value=1.0, woba_denom=1),
            _event(game_date="2024-04-01", game_pk=2, batting_team="147", woba_value=0.0, woba_denom=1),
        ],
    )

    result = TTEPriorBaselineBuilder(raw_cache_db=raw_db, pit_cache_db=pit_db).persist_prior_baseline(
        season=2024,
        prior_season_start_date="2023-04-01",
        prior_season_end_date="2023-10-01",
    )

    assert result.rows["147"]["team_woba_prior"] == 1.0
    assert result.rows["147"]["pa_prior"] == 1


def test_prior_baseline_uses_bulk_team_projection_without_full_raw_decoding(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    events = []
    game_pk = 1
    for team_id in ("147", "158"):
        for idx in range(75):
            events.append(
                _event(
                    game_date="2023-04-01" if idx < 40 else "2023-04-02",
                    game_pk=game_pk,
                    at_bat_number=idx + 1,
                    pitch_number=1,
                    batting_team=team_id,
                    events="walk" if idx % 3 == 0 else "single",
                    launch_speed=100 if idx % 2 == 0 else 88,
                    launch_angle=20,
                    launch_speed_angle=6 if idx % 2 == 0 else 3,
                    woba_value=0.7 if idx % 3 == 0 else 0.9,
                    woba_denom=1,
                )
            )
            game_pk += 1
    _cache_with_events(raw_db, events)
    spy_cache = _SpyRawSavantEventsCache(raw_db)
    rolling_builder = SavantOffenseRollingBuilder(cache=spy_cache)
    builder = TTEPriorBaselineBuilder(
        raw_cache_db=raw_db,
        pit_cache_db=pit_db,
        rolling_builder=rolling_builder,
    )
    builder.raw_cache = spy_cache

    result = builder.persist_prior_baseline(
        season=2024,
        prior_season_start_date="2023-04-01",
        prior_season_end_date="2023-10-01",
        prior_season=2023,
        fetched_at="2026-06-20T12:00:00Z",
    )

    assert spy_cache.full_event_queries == 0
    assert spy_cache.team_projection_queries == 1
    assert spy_cache.count_queries == 1
    assert spy_cache.distinct_date_queries == 1
    assert result.build_report is not None
    assert result.build_report["raw_event_count"] == 150
    assert result.build_report["distinct_game_dates"] == 2
    assert result.build_report["distinct_teams"] == 2
    assert result.build_report["rows_processed"] == 150
    assert set(result.rows) == {"147", "158"}
    assert result.rows["147"]["pa_prior"] == 75
    assert result.rows["158"]["pa_prior"] == 75
    assert result.rows["147"]["source_fingerprint"].startswith(
        "savant:raw:team_offense_prior:tte_prior_baseline_v1:2024:2023:"
    )

    builder.persist_prior_baseline(
        season=2024,
        prior_season_start_date="2023-04-01",
        prior_season_end_date="2023-10-01",
        prior_season=2023,
        fetched_at="2026-06-20T12:00:00Z",
    )
    con = sqlite3.connect(pit_db)
    count = con.execute("select count(*) from pit_metric_cache").fetchone()[0]
    con.close()
    assert count == 2


def test_prior_baseline_excludes_all_star_pseudo_teams(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    events = [
        _event(
            game_date="2023-07-11",
            game_pk=717421,
            at_bat_number=index,
            pitch_number=1,
            batting_team=team,
            events="single",
            launch_speed=100,
            launch_angle=20,
            launch_speed_angle=6,
            woba_value=0.9,
            woba_denom=1,
        )
        for index, team in enumerate(("AL", "NL", "ATL"), start=1)
    ]
    _cache_with_events(raw_db, events)

    result = TTEPriorBaselineBuilder(
        raw_cache_db=raw_db,
        pit_cache_db=pit_db,
    ).persist_prior_baseline(
        season=2024,
        prior_season_start_date="2023-03-30",
        prior_season_end_date="2023-10-01",
        prior_season=2023,
        fetched_at="2026-06-24T12:00:00Z",
    )

    assert set(result.rows) == {"ATL"}
    assert result.build_report is not None
    assert result.build_report["distinct_teams"] == 1
    assert result.build_report["excluded_non_mlb_teams"] == ["AL", "NL"]


def test_none_values_preserved_for_missing_denominators(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _cache_with_events(
        raw_db,
        [
            _event(
                game_date="2023-04-01",
                batting_team="147",
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

    result = TTEPriorBaselineBuilder(raw_cache_db=raw_db, pit_cache_db=pit_db).persist_prior_baseline(
        season=2024,
        prior_season_start_date="2023-04-01",
        prior_season_end_date="2023-10-01",
    )
    row = result.rows["147"]

    assert row["team_est_woba_prior"] is None
    assert row["team_woba_prior"] is None
    assert row["bb_pct_prior"] is None
    assert row["k_pct_prior"] is None
    assert row["barrel_pa_prior"] is None
    assert row["pa_prior"] == 0


def test_true_zero_rates_preserved(tmp_path):
    raw_db = tmp_path / "raw.db"
    pit_db = tmp_path / "pit.db"
    _cache_with_events(
        raw_db,
        [
            _event(
                game_date="2023-04-01",
                batting_team="147",
                events="single",
                launch_speed=88,
                launch_angle=20,
                launch_speed_angle=3,
                woba_value=0.9,
                woba_denom=1,
            )
        ],
    )

    result = TTEPriorBaselineBuilder(raw_cache_db=raw_db, pit_cache_db=pit_db).persist_prior_baseline(
        season=2024,
        prior_season_start_date="2023-04-01",
        prior_season_end_date="2023-10-01",
    )
    row = result.rows["147"]

    assert row["bb_pct_prior"] == 0.0
    assert row["k_pct_prior"] == 0.0
    assert row["barrel_pa_prior"] == 0.0


def test_module_does_not_import_live_app_backtest_or_run_module():
    import modules.baseball_module.advanced_pit_enrichment.tte_prior_baseline_builder as module

    source = inspect.getsource(module)
    assert "import app" not in source
    assert "from app" not in source
    assert "backtest_and_retrain" not in source
    assert "run_module" not in source
    assert "true_talent_engine" not in source


def _cache_with_events(db_path, events):
    cache = RawSavantEventsCache(db_path)
    cache.save_events(events, source_fingerprint="test-source")
    return cache


class _SpyRawSavantEventsCache(RawSavantEventsCache):
    def __init__(self, db_path):
        super().__init__(db_path)
        self.full_event_queries = 0
        self.team_projection_queries = 0
        self.count_queries = 0
        self.distinct_date_queries = 0

    def get_events_by_date_range(self, *, start_date, end_date):
        self.full_event_queries += 1
        return super().get_events_by_date_range(start_date=start_date, end_date=end_date)

    def iter_team_offense_events_by_date_range(self, *, start_date, end_date):
        self.team_projection_queries += 1
        return super().iter_team_offense_events_by_date_range(
            start_date=start_date,
            end_date=end_date,
        )

    def count_events_by_date_range(self, *, start_date, end_date):
        self.count_queries += 1
        return super().count_events_by_date_range(start_date=start_date, end_date=end_date)

    def count_distinct_dates_by_date_range(self, *, start_date, end_date):
        self.distinct_date_queries += 1
        return super().count_distinct_dates_by_date_range(
            start_date=start_date,
            end_date=end_date,
        )


def _event(
    *,
    game_date,
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
    batting_team="147",
):
    return {
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
        "batting_team": batting_team,
    }
