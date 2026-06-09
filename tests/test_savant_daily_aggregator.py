import pytest

from modules.baseball_module.advanced_pit_enrichment import (
    RawSavantEventsCache,
    SavantDailyAggregator,
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
        "estimated_woba_using_speedangle": "0.320",
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


def test_savant_daily_aggregator_computes_pitcher_math(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(
                pitch_number="1",
                launch_speed="95.0",
                launch_angle="20",
                estimated_woba_using_speedangle="0.320",
                woba_value="0",
                woba_denom="1",
                launch_speed_angle="6",
            ),
            _event(
                pitch_number="2",
                launch_speed="100.0",
                launch_angle="10",
                estimated_woba_using_speedangle="0.700",
                woba_value="0.9",
                woba_denom="1",
                launch_speed_angle="5",
            ),
            _event(
                pitch_number="3",
                events="strikeout",
                launch_speed="",
                launch_angle="",
                estimated_woba_using_speedangle="",
                woba_value="0",
                woba_denom="1",
                launch_speed_angle="",
            ),
        ],
    )

    metrics = SavantDailyAggregator(cache=cache).aggregate_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )

    assert len(metrics) == 1
    row = metrics[0]
    assert row.game_date == "2024-04-01"
    assert row.pitcher == 605400
    assert row.pitches == 3
    assert row.pa == 3
    assert row.bip == 2
    assert row.batted_ball_count == 2
    assert row.est_woba == pytest.approx(0.51)
    assert row.woba_numerator == pytest.approx(0.9)
    assert row.woba_denominator == pytest.approx(3.0)
    assert row.woba == pytest.approx(0.3)
    assert row.barrel_count == 1
    assert row.brl_percent == pytest.approx(50.0)
    assert row.avg_hit_speed == pytest.approx(97.5)
    assert row.ev95plus == 2
    assert row.ev95percent == pytest.approx(100.0)
    assert row.sweet_spot_pct == pytest.approx(100.0)


def test_barrel_percent_real_zero_vs_null_denominator(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(pitcher="1", game_pk="1", launch_speed="88.0", launch_speed_angle="4"),
            _event(
                pitcher="2",
                game_pk="2",
                launch_speed="",
                launch_angle="",
                launch_speed_angle="",
                estimated_woba_using_speedangle="",
                woba_value="",
                woba_denom="",
            ),
        ],
    )

    metrics = SavantDailyAggregator(cache=cache).aggregate_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )
    by_pitcher = {row.pitcher: row for row in metrics}

    assert by_pitcher[1].batted_ball_count == 1
    assert by_pitcher[1].barrel_count == 0
    assert by_pitcher[1].brl_percent == 0.0
    assert by_pitcher[2].batted_ball_count == 0
    assert by_pitcher[2].barrel_count == 0
    assert by_pitcher[2].brl_percent is None


def test_missing_woba_denominator_remains_none(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(
                woba_value="",
                woba_denom="",
                estimated_woba_using_speedangle="",
            )
        ],
    )

    row = SavantDailyAggregator(cache=cache).aggregate_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )[0]

    assert row.pa == 0
    assert row.woba_numerator is None
    assert row.woba_denominator is None
    assert row.woba is None


def test_ev95_percent_uses_batted_ball_denominator(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(pitch_number="1", launch_speed="94.9", launch_angle="20"),
            _event(pitch_number="2", launch_speed="95.0", launch_angle="25"),
            _event(pitch_number="3", launch_speed="100.0", launch_angle="40"),
            _event(pitch_number="4", launch_speed="", launch_angle=""),
        ],
    )

    row = SavantDailyAggregator(cache=cache).aggregate_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-01",
    )[0]

    assert row.batted_ball_count == 3
    assert row.ev95plus == 2
    assert row.ev95percent == pytest.approx(200.0 / 3.0)
    assert row.sweet_spot_pct == pytest.approx(200.0 / 3.0)


def test_multiple_pitchers_and_multiple_dates_are_grouped_separately(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(game_date="2024-04-01", game_pk="1", pitcher="100", pitch_number="1"),
            _event(game_date="2024-04-01", game_pk="2", pitcher="200", pitch_number="1"),
            _event(game_date="2024-04-02", game_pk="3", pitcher="100", pitch_number="1"),
        ],
    )

    metrics = SavantDailyAggregator(cache=cache).aggregate_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-02",
    )

    assert [(row.game_date, row.pitcher, row.pitches) for row in metrics] == [
        ("2024-04-01", 100, 1),
        ("2024-04-01", 200, 1),
        ("2024-04-02", 100, 1),
    ]


def test_no_future_date_included(tmp_path):
    cache = _cache(
        tmp_path,
        [
            _event(game_date="2024-04-01", game_pk="1", pitcher="100"),
            _event(game_date="2024-04-02", game_pk="2", pitcher="100"),
            _event(game_date="2024-04-03", game_pk="3", pitcher="100"),
        ],
    )

    metrics = SavantDailyAggregator(cache=cache).aggregate_by_date_range(
        start_date="2024-04-01",
        end_date="2024-04-02",
    )

    assert {row.game_date for row in metrics} == {"2024-04-01", "2024-04-02"}
    assert all(row.game_date <= "2024-04-02" for row in metrics)
