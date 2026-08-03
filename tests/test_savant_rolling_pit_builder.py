import pytest

from modules.baseball_module.advanced_pit_enrichment import (
    RawSavantEventsCache,
    SavantRollingPITBuilder,
)


def _event(**overrides):
    event = {
        "game_date": "2024-04-01",
        "game_pk": "746001",
        "at_bat_number": "1",
        "pitch_number": "1",
        "pitcher": "605400",
        "batter": "592450",
        "events": "field_out",
        "launch_speed": "95.0",
        "launch_angle": "20",
        "estimated_woba_using_speedangle": "0.300",
        "woba_value": "0",
        "woba_denom": "1",
        "launch_speed_angle": "6",
    }
    event.update(overrides)
    return event


def _cache(tmp_path, events):
    cache = RawSavantEventsCache(tmp_path / "raw.db")
    cache.save_events(events, source_fingerprint="test", fetched_at="2026-06-09T12:00:00Z")
    return cache


def test_rolling_metrics_accumulate_across_multiple_days(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(game_date="2024-04-01", game_pk="1", pitch_number="1", launch_speed="95", estimated_woba_using_speedangle="0.300", woba_value="0", woba_denom="1", launch_speed_angle="6"),
            _event(game_date="2024-04-01", game_pk="1", pitch_number="2", launch_speed="", launch_angle="", estimated_woba_using_speedangle="", woba_value="0", woba_denom="1", launch_speed_angle=""),
            _event(game_date="2024-04-02", game_pk="2", pitch_number="1", launch_speed="100", estimated_woba_using_speedangle="0.700", woba_value="0.9", woba_denom="1", launch_speed_angle="5"),
        ],
    )

    metrics = SavantRollingPITBuilder(cache=cache).build_pitcher_for_as_of_date(
        pitcher=605400,
        season_start_date="2024-04-01",
        as_of_date="2024-04-02",
    )

    assert metrics is not None
    assert metrics.as_of_date == "2024-04-02"
    assert metrics.pa == 3
    assert metrics.bip == 2
    assert metrics.batted_ball_count == 2
    assert metrics.barrel_count == 1
    assert metrics.est_woba == pytest.approx(0.5)
    assert metrics.woba == pytest.approx(0.3)
    assert metrics.brl_percent == pytest.approx(50.0)
    assert metrics.avg_hit_speed == pytest.approx(97.5)
    assert metrics.ev95percent == pytest.approx(100.0)


def test_no_future_day_included(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(game_date="2024-04-01", game_pk="1", estimated_woba_using_speedangle="0.100"),
            _event(game_date="2024-04-02", game_pk="2", estimated_woba_using_speedangle="0.900"),
        ],
    )

    metrics = SavantRollingPITBuilder(cache=cache).build_pitcher_for_as_of_date(
        pitcher=605400,
        season_start_date="2024-04-01",
        as_of_date="2024-04-01",
    )

    assert metrics is not None
    assert metrics.est_woba == pytest.approx(0.100)
    assert metrics.pa == 1


def test_weighted_barrel_percent_uses_total_barrels_over_total_batted_balls(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(game_date="2024-04-01", game_pk="1", pitch_number="1", launch_speed="90", launch_speed_angle="6"),
            _event(game_date="2024-04-01", game_pk="1", pitch_number="2", launch_speed="91", launch_speed_angle="6"),
            _event(game_date="2024-04-02", game_pk="2", pitch_number="1", launch_speed="80", launch_speed_angle="4"),
            _event(game_date="2024-04-02", game_pk="2", pitch_number="2", launch_speed="81", launch_speed_angle="4"),
            _event(game_date="2024-04-02", game_pk="2", pitch_number="3", launch_speed="82", launch_speed_angle="4"),
            _event(game_date="2024-04-02", game_pk="2", pitch_number="4", launch_speed="83", launch_speed_angle="4"),
            _event(game_date="2024-04-02", game_pk="2", pitch_number="5", launch_speed="84", launch_speed_angle="4"),
            _event(game_date="2024-04-02", game_pk="2", pitch_number="6", launch_speed="85", launch_speed_angle="4"),
        ],
    )

    metrics = SavantRollingPITBuilder(cache=cache).build_pitcher_for_as_of_date(
        pitcher=605400,
        season_start_date="2024-04-01",
        as_of_date="2024-04-02",
    )

    assert metrics is not None
    assert metrics.barrel_count == 2
    assert metrics.batted_ball_count == 8
    assert metrics.brl_percent == pytest.approx(25.0)


def test_weighted_est_woba_uses_event_counts_not_daily_average(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(game_date="2024-04-01", game_pk="1", pitch_number="1", estimated_woba_using_speedangle="0.900"),
            _event(game_date="2024-04-02", game_pk="2", pitch_number="1", estimated_woba_using_speedangle="0.100"),
            _event(game_date="2024-04-02", game_pk="2", pitch_number="2", estimated_woba_using_speedangle="0.100"),
            _event(game_date="2024-04-02", game_pk="2", pitch_number="3", estimated_woba_using_speedangle="0.100"),
        ],
    )

    metrics = SavantRollingPITBuilder(cache=cache).build_pitcher_for_as_of_date(
        pitcher=605400,
        season_start_date="2024-04-01",
        as_of_date="2024-04-02",
    )

    assert metrics is not None
    assert metrics.est_woba == pytest.approx(0.3)


def test_weighted_est_woba_ignores_batted_balls_with_missing_estimate(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(game_date="2024-04-01", game_pk="1", pitch_number="1", estimated_woba_using_speedangle="0.900"),
            _event(game_date="2024-04-01", game_pk="1", pitch_number="2", estimated_woba_using_speedangle=""),
            _event(game_date="2024-04-02", game_pk="2", pitch_number="1", estimated_woba_using_speedangle="0.100"),
        ],
    )

    metrics = SavantRollingPITBuilder(cache=cache).build_pitcher_for_as_of_date(
        pitcher=605400,
        season_start_date="2024-04-01",
        as_of_date="2024-04-02",
    )

    assert metrics is not None
    assert metrics.batted_ball_count == 3
    assert metrics.est_woba == pytest.approx(0.5)


def test_missing_denominator_remains_none(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(
                launch_speed="",
                launch_angle="",
                estimated_woba_using_speedangle="",
                woba_value="",
                woba_denom="",
                launch_speed_angle="",
            )
        ],
    )

    metrics = SavantRollingPITBuilder(cache=cache).build_pitcher_for_as_of_date(
        pitcher=605400,
        season_start_date="2024-04-01",
        as_of_date="2024-04-01",
    )

    assert metrics is not None
    assert metrics.est_woba is None
    assert metrics.woba is None
    assert metrics.brl_percent is None
    assert metrics.avg_hit_speed is None
    assert metrics.ev95percent is None
    assert metrics.sweet_spot_pct is None
    assert metrics.pa == 0
    assert metrics.bip == 0


def test_real_zero_barrel_percent_remains_zero(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(launch_speed="88", launch_speed_angle="4"),
            _event(game_pk="2", launch_speed="89", launch_speed_angle="5"),
        ],
    )

    metrics = SavantRollingPITBuilder(cache=cache).build_pitcher_for_as_of_date(
        pitcher=605400,
        season_start_date="2024-04-01",
        as_of_date="2024-04-01",
    )

    assert metrics is not None
    assert metrics.barrel_count == 0
    assert metrics.batted_ball_count == 2
    assert metrics.brl_percent == 0.0


def test_multiple_pitchers_remain_independent(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(pitcher="100", game_pk="1", estimated_woba_using_speedangle="0.100"),
            _event(pitcher="200", game_pk="2", estimated_woba_using_speedangle="0.900"),
        ],
    )

    metrics = SavantRollingPITBuilder(cache=cache).build_for_as_of_date(
        season_start_date="2024-04-01",
        as_of_date="2024-04-01",
    )

    assert sorted(metrics) == [100, 200]
    assert metrics[100].est_woba == pytest.approx(0.100)
    assert metrics[200].est_woba == pytest.approx(0.900)


def test_query_before_first_event_returns_no_metrics(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(game_date="2024-04-02", game_pk="1"),
        ],
    )

    metrics = SavantRollingPITBuilder(cache=cache).build_for_as_of_date(
        season_start_date="2024-04-01",
        as_of_date="2024-04-01",
    )

    assert metrics == {}
