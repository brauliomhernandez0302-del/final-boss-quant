import pytest

from modules.baseball_module.advanced_pit_enrichment import (
    PITCache,
    PitcherPriorBaselinePersistence,
    RawSavantEventsCache,
    SavantRollingPITBuilder,
    SavantRollingPitcherMetrics,
)


class _FakeBuilder:
    def build_for_as_of_date(self, *, season_start_date, as_of_date):
        return {
            605400: SavantRollingPitcherMetrics(
                as_of_date=as_of_date,
                pitcher=605400,
                est_woba=0.301,
                woba=0.295,
                woba_numerator=120.0,
                woba_denominator=400.0,
                brl_percent=7.5,
                barrel_count=20,
                batted_ball_count=267,
                pa=410,
                bip=267,
                avg_hit_speed=88.4,
                ev95plus=95,
                ev95percent=35.6,
                sweet_spot_pct=33.2,
            )
        }


def test_persists_prior_baseline_with_strict_2023_window_for_2024(tmp_path):
    cache = PITCache(tmp_path / "pit.db")
    persistence = PitcherPriorBaselinePersistence(
        pit_cache=cache,
        builder=_FakeBuilder(),
    )

    rows = persistence.persist_prior_baseline(
        target_season=2024,
        prior_season=2023,
        prior_season_start_date="2023-03-30",
        prior_season_end_date="2023-10-01",
        fetched_at="2026-06-26T12:00:00Z",
    )

    record = cache.get_latest(
        namespace=PitcherPriorBaselinePersistence.NAMESPACE,
        entity_id=605400,
        season=2024,
        as_of_date="2024-03-28T23:59:59Z",
        source=PitcherPriorBaselinePersistence.SOURCE,
    )
    assert len(rows) == 1
    assert record is not None
    assert record.data["prior_season"] == 2023
    assert record.data["source_window_start_date"] == "2023-03-30"
    assert record.data["source_window_end_date"] == "2023-10-01"
    assert record.data["est_woba"] == 0.301
    assert record.source_fingerprint


@pytest.mark.parametrize(
    ("prior_season", "start_date", "end_date"),
    [
        (2022, "2022-04-01", "2022-10-01"),
        (2023, "2024-03-28", "2024-10-01"),
        (2023, "2023-03-30", "2024-03-28"),
    ],
)
def test_rejects_non_prior_season_windows(
    tmp_path,
    prior_season,
    start_date,
    end_date,
):
    persistence = PitcherPriorBaselinePersistence(
        pit_cache=PITCache(tmp_path / "pit.db"),
        builder=_FakeBuilder(),
    )

    with pytest.raises(ValueError):
        persistence.persist_prior_baseline(
            target_season=2024,
            prior_season=prior_season,
            prior_season_start_date=start_date,
            prior_season_end_date=end_date,
        )


def _event(**overrides):
    row = {
        "game_date": "2023-04-01",
        "game_pk": 1,
        "at_bat_number": 1,
        "pitch_number": 1,
        "pitcher": 100,
        "batter": 200,
        "events": "field_out",
        "launch_speed": 96.0,
        "launch_angle": 20.0,
        "estimated_woba_using_speedangle": 0.3,
        "woba_value": 0.0,
        "woba_denom": 1.0,
        "launch_speed_angle": 6,
    }
    row.update(overrides)
    return row


def test_one_pass_prior_matches_rolling_builder_metric_semantics(tmp_path):
    raw = RawSavantEventsCache(tmp_path / "raw.db")
    raw.save_events(
        [
            _event(),
            _event(
                game_date="2023-04-02",
                game_pk=2,
                at_bat_number=2,
                launch_speed=94.0,
                launch_angle=40.0,
                estimated_woba_using_speedangle=0.7,
                woba_value=0.9,
                launch_speed_angle=5,
            ),
            _event(
                game_date="2023-04-02",
                game_pk=3,
                at_bat_number=3,
                pitcher=101,
                launch_speed=None,
                launch_angle=None,
                estimated_woba_using_speedangle=None,
                woba_value=0.9,
                woba_denom=None,
                launch_speed_angle=None,
            ),
            _event(
                game_date="2023-04-02",
                game_pk=4,
                at_bat_number=4,
                pitcher=100,
                launch_speed=None,
                launch_angle=None,
                estimated_woba_using_speedangle=None,
                woba_value=0.2,
                woba_denom=None,
                launch_speed_angle=None,
            ),
        ],
        source_fingerprint="fixture",
    )
    expected = SavantRollingPITBuilder(cache=raw).build_for_as_of_date(
        season_start_date="2023-04-01",
        as_of_date="2023-04-02",
    )
    persistence = PitcherPriorBaselinePersistence(
        raw_cache=raw,
        pit_cache=PITCache(tmp_path / "pit.db"),
    )
    actual, report = persistence._build_one_pass(
        start_date="2023-04-01",
        end_date="2023-04-02",
    )

    for pitcher, legacy in expected.items():
        optimized = actual[pitcher]
        for field in (
            "est_woba",
            "woba",
            "woba_numerator",
            "woba_denominator",
            "brl_percent",
            "avg_hit_speed",
            "ev95percent",
            "sweet_spot_pct",
        ):
            legacy_value = getattr(legacy, field)
            optimized_value = getattr(optimized, field)
            if legacy_value is None:
                assert optimized_value is None
            else:
                assert optimized_value == pytest.approx(legacy_value)
        for field in (
            "barrel_count",
            "batted_ball_count",
            "pa",
            "bip",
            "ev95plus",
        ):
            assert getattr(optimized, field) == getattr(legacy, field)
    assert report == {
        "events_scanned": 4,
        "duplicate_plate_appearances": 0,
    }


def test_one_pass_deduplicates_plate_appearance_contributions(tmp_path):
    raw = RawSavantEventsCache(tmp_path / "raw.db")
    raw.save_events(
        [
            _event(pitch_number=1, woba_value=0.7, woba_denom=1),
            _event(pitch_number=2, woba_value=0.9, woba_denom=1),
        ],
        source_fingerprint="fixture",
    )
    persistence = PitcherPriorBaselinePersistence(
        raw_cache=raw,
        pit_cache=PITCache(tmp_path / "pit.db"),
    )

    rows = persistence.persist_prior_baseline(
        target_season=2024,
        prior_season=2023,
        prior_season_start_date="2023-04-01",
        prior_season_end_date="2023-04-01",
    )

    assert rows[100]["pa"] == 1
    assert rows[100]["woba"] == pytest.approx(0.9)
    assert persistence.last_build_report["duplicate_plate_appearances"] == 1


def test_one_pass_uses_exactly_one_projected_range_read(tmp_path):
    class CountingCache(RawSavantEventsCache):
        def __init__(self, db_path):
            super().__init__(db_path)
            self.calls = 0

        def iter_pitcher_metric_events_by_date_range(self, *, start_date, end_date):
            self.calls += 1
            yield from super().iter_pitcher_metric_events_by_date_range(
                start_date=start_date,
                end_date=end_date,
            )

        def get_events_by_date_range(self, *, start_date, end_date):
            raise AssertionError("full-row/raw_json read must not be used")

    raw = CountingCache(tmp_path / "raw.db")
    raw.save_events([_event()], source_fingerprint="fixture")
    persistence = PitcherPriorBaselinePersistence(
        raw_cache=raw,
        pit_cache=PITCache(tmp_path / "pit.db"),
    )

    persistence.persist_prior_baseline(
        target_season=2024,
        prior_season=2023,
        prior_season_start_date="2023-04-01",
        prior_season_end_date="2023-04-01",
    )

    assert raw.calls == 1
    assert persistence.last_build_report["events_scanned"] == 1


def test_one_pass_enforces_exact_prior_season_source_window(tmp_path):
    raw = RawSavantEventsCache(tmp_path / "raw.db")
    raw.save_events(
        [
            _event(game_date="2022-10-05", game_pk=2022),
            _event(game_date="2023-04-01", game_pk=2023),
            _event(game_date="2024-03-28", game_pk=2024),
        ],
        source_fingerprint="fixture",
    )
    persistence = PitcherPriorBaselinePersistence(
        raw_cache=raw,
        pit_cache=PITCache(tmp_path / "pit.db"),
    )

    rows = persistence.persist_prior_baseline(
        target_season=2024,
        prior_season=2023,
        prior_season_start_date="2023-01-01",
        prior_season_end_date="2023-12-31",
    )

    assert rows[100]["pa"] == 1
    assert rows[100]["bip"] == 1
    assert persistence.last_build_report["events_scanned"] == 1
