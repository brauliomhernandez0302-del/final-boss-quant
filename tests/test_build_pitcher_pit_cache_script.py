import importlib
import sys


SCRIPT_MODULE = "scripts.build_pitcher_pit_cache"
PIPELINE_MODULES = {
    "app",
    "backtest_and_retrain",
    "data_fetchers",
    "odds_fetcher",
    "run_daily_picks",
}


class _FakeFetcher:
    calls = []
    rows = {}

    def __init__(self, cache_db):
        self.cache_db = cache_db

    def fetch_pitcher_metrics_by_date_range(self, *, season, start_date, end_date):
        self.calls.append(
            {
                "cache_db": self.cache_db,
                "season": season,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        return self.rows


class _FakeFanGraphsFetcher(_FakeFetcher):
    calls = []
    rows = {1: {"fg_name": "One"}, 2: {"fg_name": "Two"}}


class _FakeSavantFetcher(_FakeFetcher):
    calls = []
    rows = {3: {"player_name": "Three"}}


def _load_script(monkeypatch):
    # monkeypatch.delitem (not a raw sys.modules.pop()) restores these after
    # the test — a raw pop() permanently evicts them from sys.modules for
    # the rest of the pytest session, silently breaking a later test's
    # monkeypatch.setattr() on one of these already-imported modules (found
    # 2026-07-19; see test_build_raw_savant_events_script.py's identical fix).
    for module_name in PIPELINE_MODULES:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    script = importlib.import_module(SCRIPT_MODULE)
    monkeypatch.setattr(script, "FanGraphsPITFetcher", _FakeFanGraphsFetcher)
    monkeypatch.setattr(script, "SavantPITFetcher", _FakeSavantFetcher)
    _FakeFanGraphsFetcher.calls = []
    _FakeSavantFetcher.calls = []
    return script


def _argv(tmp_path, provider):
    return [
        "--cache-db",
        str(tmp_path / "pit.db"),
        "--season",
        "2025",
        "--start-date",
        "2025-03-27",
        "--end-date",
        "2025-04-05",
        "--provider",
        provider,
    ]


def test_cli_parses_args(tmp_path, monkeypatch):
    script = _load_script(monkeypatch)

    args = script.parse_args(_argv(tmp_path, "fangraphs"))

    assert args.cache_db == tmp_path / "pit.db"
    assert args.season == 2025
    assert args.start_date == "2025-03-27"
    assert args.end_date == "2025-04-05"
    assert args.provider == "fangraphs"


def test_fangraphs_provider_calls_only_fangraphs(tmp_path, monkeypatch, capsys):
    script = _load_script(monkeypatch)

    assert script.main(_argv(tmp_path, "fangraphs")) == 0

    assert len(_FakeFanGraphsFetcher.calls) == 1
    assert _FakeFanGraphsFetcher.calls[0]["cache_db"] == tmp_path / "pit.db"
    assert _FakeFanGraphsFetcher.calls[0]["season"] == 2025
    assert _FakeFanGraphsFetcher.calls[0]["start_date"] == "2025-03-27"
    assert _FakeFanGraphsFetcher.calls[0]["end_date"] == "2025-04-05"
    assert _FakeSavantFetcher.calls == []
    output = capsys.readouterr().out
    assert "provider=fangraphs" in output
    assert "pitcher_count=2" in output


def test_savant_provider_calls_only_savant(tmp_path, monkeypatch, capsys):
    script = _load_script(monkeypatch)

    assert script.main(_argv(tmp_path, "savant")) == 0

    assert _FakeFanGraphsFetcher.calls == []
    assert len(_FakeSavantFetcher.calls) == 1
    output = capsys.readouterr().out
    assert "provider=savant" in output
    assert "pitcher_count=1" in output


def test_all_provider_calls_both(tmp_path, monkeypatch, capsys):
    script = _load_script(monkeypatch)

    assert script.main(_argv(tmp_path, "all")) == 0

    assert len(_FakeFanGraphsFetcher.calls) == 1
    assert len(_FakeSavantFetcher.calls) == 1
    output = capsys.readouterr().out
    assert "provider=fangraphs" in output
    assert "provider=savant" in output
    assert "pitcher_count=2" in output
    assert "pitcher_count=1" in output


def test_script_does_not_import_live_or_backtest_modules(tmp_path, monkeypatch):
    script = _load_script(monkeypatch)

    script.main(_argv(tmp_path, "all"))

    assert PIPELINE_MODULES.isdisjoint(sys.modules)
