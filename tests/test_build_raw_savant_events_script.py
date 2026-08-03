import ast
import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_raw_savant_events.py"
FORBIDDEN_IMPORTS = {
    "app",
    "backtest_and_retrain",
    "data_fetchers",
    "modules.baseball_module.core.run_module",
    "run_daily_picks",
}


def _load_script():
    spec = importlib.util.spec_from_file_location("build_raw_savant_events_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_paths_are_durable_and_season_specific():
    script = _load_script()
    args = script.parse_args(
        [
            "--season",
            "2023",
            "--start-date",
            "2023-03-30",
            "--end-date",
            "2023-04-05",
        ]
    )

    assert args.season == 2023
    assert script.DEFAULT_CACHE_ROOT == REPO_ROOT / "data" / "pit_raw"


def test_script_has_no_live_backtest_or_run_module_imports():
    tree = ast.parse(SCRIPT_PATH.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert FORBIDDEN_IMPORTS.isdisjoint(imported)


def test_pit_raw_directory_and_database_are_gitignored(tmp_path):
    ignored_db = REPO_ROOT / "data" / "pit_raw" / "raw_savant_2023.db"
    ignored_manifest = REPO_ROOT / "data" / "pit_raw" / "raw_savant_2023.manifest.json"

    for ignored_path in (ignored_db, ignored_manifest):
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(ignored_path)],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(ignored_db.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode != 0


def test_importing_script_does_not_import_pipeline_modules(monkeypatch):
    # monkeypatch.delitem (not a raw sys.modules.pop()) so these 5 modules
    # are restored to their pre-test cached state after this test — a raw
    # pop() here permanently evicted them from sys.modules for the rest of
    # the pytest session (found 2026-07-19: it made a later test's
    # monkeypatch.setattr() on an already-imported module silently
    # ineffective, since the next `import` after the eviction rebuilds a
    # fresh, unpatched module object instead of reusing the patched one).
    for module_name in FORBIDDEN_IMPORTS:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    _load_script()

    assert FORBIDDEN_IMPORTS.isdisjoint(sys.modules)
