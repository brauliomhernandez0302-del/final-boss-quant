from modules.baseball_module.advanced_pit_enrichment import PITCache
from modules.baseball_module.advanced_pit_enrichment.fangraphs_pit_fetcher import (
    FanGraphsPITFetcher,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.headers = {}
        self.payload = payload
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return _FakeResponse(self.payload)


def test_fangraphs_pit_fetcher_fetches_parses_and_caches_date_window(tmp_path):
    payload = {
        "dateRange": "2025-03-27 and 2025-04-05",
        "data": [
            {
                "xMLBAMID": "669923",
                "xFIP": "3.44",
                "SIERA": "3.21",
                "FIP": "3.60",
                "xERA": "3.12",
                "ERA": "3.80",
                "WAR": "1.2",
                "K%": "28.5",
                "BB%": "7.1",
                "K-BB%": "21.4",
                "SwStr%": "13.2",
                "BABIP": ".290",
                "LOB%": "74.0",
                "HR/FB": "11.5",
                "GB%": "45.0",
                "FB%": "34.0",
                "LD%": "21.0",
                "IP": "33.2",
                "WHIP": "1.08",
                "AvgBatSpeed": "70.1",
                "Hard%": "34.8",
                "playerid": "12345",
                "Name": '<a href="/players/example">Example Pitcher</a>',
                "Team": "<span>NYY</span>",
                "Season": "2025",
            },
            {"xMLBAMID": "", "SIERA": "2.0"},
        ]
    }
    session = _FakeSession(payload)
    fetcher = FanGraphsPITFetcher(tmp_path / "pit.db", session=session)

    result = fetcher.fetch_pitcher_metrics_by_date_range(
        season=2025,
        start_date="2025-03-27",
        end_date="2025-04-05",
    )

    assert list(result) == [669923]
    assert result[669923]["siera"] == 3.21
    assert result[669923]["xfip"] == 3.44
    assert result[669923]["fg_name"] == "Example Pitcher"
    assert result[669923]["fg_team"] == "NYY"
    assert session.calls[0]["params"]["month"] == "1000"
    assert session.calls[0]["params"]["startdate"] == "2025-03-27"
    assert session.calls[0]["params"]["enddate"] == "2025-04-05"

    cached = fetcher.get_cached_pitcher_metrics(
        mlbam_id=669923,
        season=2025,
        as_of_date="2025-04-05T12:00:00Z",
    )
    assert cached == result[669923]

    before_cutoff = fetcher.get_cached_pitcher_metrics(
        mlbam_id=669923,
        season=2025,
        as_of_date="2025-04-04T23:59:59Z",
    )
    assert before_cutoff == {}


def test_fangraphs_pit_fetcher_keeps_fangraphs_source_isolated(tmp_path):
    db_path = tmp_path / "pit.db"
    cache = PITCache(db_path)
    cache.save_record(
        namespace=FanGraphsPITFetcher.PITCHER_NAMESPACE,
        entity_id=42,
        season=2025,
        as_of_date="2025-04-05T00:00:00Z",
        source="baseball_savant",
        source_fingerprint="savant",
        data={"xera": 2.0},
        fetched_at="2026-06-06T12:00:00Z",
    )
    fetcher = FanGraphsPITFetcher(db_path, session=_FakeSession({"data": []}))

    assert (
        fetcher.get_cached_pitcher_metrics(
            mlbam_id=42,
            season=2025,
            as_of_date="2025-04-06T00:00:00Z",
        )
        == {}
    )


def test_fangraphs_pit_fetcher_rejects_mismatched_date_range(tmp_path):
    session = _FakeSession({"dateRange": "2025-01-01 and 2025-12-31", "data": []})
    fetcher = FanGraphsPITFetcher(tmp_path / "pit.db", session=session)

    try:
        fetcher.fetch_pitcher_metrics_by_date_range(
            season=2025,
            start_date="2025-03-27",
            end_date="2025-04-05",
        )
    except RuntimeError as exc:
        assert "expected '2025-03-27 and 2025-04-05'" in str(exc)
    else:
        raise AssertionError("mismatched FanGraphs date range should be rejected")
