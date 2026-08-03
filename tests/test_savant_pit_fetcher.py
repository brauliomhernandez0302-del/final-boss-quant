import pytest

import modules.baseball_module.advanced_pit_enrichment.savant_pit_fetcher as savant_pit
from modules.baseball_module.advanced_pit_enrichment import PITCache
from modules.baseball_module.advanced_pit_enrichment.savant_pit_fetcher import (
    SavantPITFetcher,
)


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self, by_day):
        self.headers = {}
        self.by_day = by_day
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        day = params["game_date_gt"]
        assert params["game_date_lt"] == day
        return _FakeResponse(self.by_day[day])


def _csv(*rows):
    header = (
        "game_date,pitcher,player_name,events,launch_speed,launch_angle,"
        "estimated_woba_using_speedangle,woba_value,woba_denom,launch_speed_angle\n"
    )
    return header + "".join(rows)


def test_savant_pit_fetcher_aggregates_daily_statcast_and_caches_window(tmp_path):
    session = _FakeSession(
        {
            "2025-04-01": _csv(
                "2025-04-01,605400,\"Nola, Aaron\",field_out,95.0,20,0.320,0,1,6\n",
                "2025-04-01,605400,\"Nola, Aaron\",strikeout,,,,0,1,\n",
            ),
            "2025-04-02": _csv(
                "2025-04-02,605400,\"Nola, Aaron\",single,100.0,10,0.700,0.9,1,5\n",
                "2025-04-02,999999,\"Other, Pitcher\",field_out,80.0,40,0.100,0,1,2\n",
            ),
        }
    )
    fetcher = SavantPITFetcher(tmp_path / "pit.db", session=session)

    result = fetcher.fetch_pitcher_metrics_by_date_range(
        season=2025,
        start_date="2025-04-01",
        end_date="2025-04-02",
    )

    assert len(session.calls) == 2
    assert session.calls[0]["params"]["player_type"] == "pitcher"
    assert session.calls[0]["params"]["game_date_gt"] == "2025-04-01"
    assert session.calls[1]["params"]["game_date_gt"] == "2025-04-02"

    nola = result[605400]
    assert nola["player_name"] == "Nola, Aaron"
    assert nola["pitches"] == 3
    assert nola["pa"] == 3
    assert nola["bip"] == 2
    assert nola["est_woba"] == pytest.approx(0.51)
    assert nola["woba"] == pytest.approx(0.3)
    assert nola["avg_hit_speed"] == pytest.approx(97.5)
    assert nola["max_hit_speed"] == 100.0
    assert nola["ev95plus"] == 2
    assert nola["ev95percent"] == pytest.approx(100.0)
    assert nola["brl_count"] == 1
    assert nola["brl_percent"] == pytest.approx(50.0)
    assert nola["brl_pa"] == pytest.approx(100.0 / 3.0)
    assert nola["sweet_spot_pct"] == pytest.approx(100.0)
    assert nola["first_game_date"] == "2025-04-01"
    assert nola["last_game_date"] == "2025-04-02"

    before_cutoff = fetcher.get_cached_pitcher_metrics(
        mlbam_id=605400,
        season=2025,
        as_of_date="2025-04-01T23:59:59Z",
    )
    at_cutoff = fetcher.get_cached_pitcher_metrics(
        mlbam_id=605400,
        season=2025,
        as_of_date="2025-04-02T00:00:00Z",
    )

    assert before_cutoff == {}
    assert at_cutoff == nola


def test_savant_pit_fetcher_keeps_savant_source_isolated(tmp_path):
    db_path = tmp_path / "pit.db"
    cache = PITCache(db_path)
    cache.save_record(
        namespace=SavantPITFetcher.PITCHER_NAMESPACE,
        entity_id=42,
        season=2025,
        as_of_date="2025-04-05T00:00:00Z",
        source="fangraphs",
        source_fingerprint="fg",
        data={"siera": 2.0},
        fetched_at="2026-06-06T12:00:00Z",
    )
    fetcher = SavantPITFetcher(db_path, session=_FakeSession({}))

    assert (
        fetcher.get_cached_pitcher_metrics(
            mlbam_id=42,
            season=2025,
            as_of_date="2025-04-06T00:00:00Z",
        )
        == {}
    )


def test_savant_pit_fetcher_rejects_rows_outside_requested_day(tmp_path):
    session = _FakeSession(
        {
            "2025-04-01": _csv(
                "2025-04-02,605400,\"Nola, Aaron\",field_out,95.0,20,0.320,0,1,6\n",
            )
        }
    )
    fetcher = SavantPITFetcher(tmp_path / "pit.db", session=session)

    with pytest.raises(RuntimeError, match="expected '2025-04-01'"):
        fetcher.fetch_pitcher_metrics_by_date_range(
            season=2025,
            start_date="2025-04-01",
            end_date="2025-04-01",
        )


def test_savant_pit_fetcher_rejects_possible_daily_truncation(monkeypatch, tmp_path):
    monkeypatch.setattr(savant_pit, "_MAX_ROWS_PER_DAY", 1)
    session = _FakeSession(
        {
            "2025-04-01": _csv(
                "2025-04-01,605400,\"Nola, Aaron\",field_out,95.0,20,0.320,0,1,6\n",
            )
        }
    )
    fetcher = SavantPITFetcher(tmp_path / "pit.db", session=session)

    with pytest.raises(RuntimeError, match="may be truncated"):
        fetcher.fetch_pitcher_metrics_by_date_range(
            season=2025,
            start_date="2025-04-01",
            end_date="2025-04-01",
        )
