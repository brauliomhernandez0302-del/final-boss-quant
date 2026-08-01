"""
Roadmap Step 3 regression tests (audit_20260714/14_remediation_roadmap.md) —
ODDS-001 + MATH-001, the deduplication bundle.

ODDS-001: backtest_and_retrain.py used to carry its own `_devig()`,
algebraically identical to but independent of
core/value_detector.py::remove_vig_multiplicative — the 5th confirmed
instance of this codebase's duplicated-constant/logic-drift pattern
(LG_XWOBA, stadium-name dictionaries, F5 naming, the TTE formula). Fixed
by importing the shared function and deleting the local copy.

MATH-001: learning_engine.py::_l0_ratio() accepted a `stage_factors_json`
parameter it never read — a leftover from two reverted denominator-change
attempts (see its postmortem docstring, preserved verbatim). Removed.
"""
import subprocess
import sys
from pathlib import Path

import pytest

from core.value_detector import remove_vig_multiplicative

REPO_ROOT = Path(__file__).resolve().parents[1]


def _devig_reference(o1: float, o2: float):
    """The exact formula _devig() used to implement, kept here as a
    historical reference for the equality test — NOT reintroduced as a
    real code path."""
    t = 1.0 / o1 + 1.0 / o2
    return (1.0 / o1) / t, (1.0 / o2) / t


ODDS_PAIRS = [
    pytest.param(1.91, 1.91, id="equal-odds"),
    pytest.param(1.05, 9.00, id="strong-favorite"),
    pytest.param(1.10, 8.00, id="strong-favorite-2"),
    pytest.param(3.75, 1.28, id="high-vig"),
    pytest.param(1.9091, 1.9091, id="low-vig-sharp"),
    pytest.param(1.01, 50.0, id="extreme-favorite"),
    pytest.param(10.0, 1.02, id="extreme-underdog"),
    pytest.param(2.50, 1.50, id="mid-range"),
]


@pytest.mark.parametrize("o1,o2", ODDS_PAIRS)
def test_remove_vig_multiplicative_matches_old_devig_exactly(o1, o2):
    """Exact equality (`==`), no tolerance — both implementations do the
    same operations (1/o per outcome, sum, divide by total) in the same
    order, so the result must be bit-identical, not merely close."""
    old = _devig_reference(o1, o2)
    new = tuple(remove_vig_multiplicative([o1, o2]))
    assert old == new


def test_devig_no_longer_exists_in_backtest_and_retrain():
    """Programmatic check that _devig was fully removed, not just
    unused — greps the actual source file rather than trusting an import
    check alone (a stale, unreferenced definition would still "pass" an
    import-based check)."""
    src = (REPO_ROOT / "backtest_and_retrain.py").read_text()
    # Se busca la DEFINICIÓN y la LLAMADA, no la subcadena pelada: el nombre
    # `_devig` es prefijo de otros identificadores legítimos de otros módulos
    # (p.ej. `track_record.db::_clv_devigged`, citado en un comentario acá), y
    # un grep de subcadena los tomaba como si el helper duplicado hubiera vuelto.
    # Esto sigue atrapando exactamente lo que el test vigila —que exista o se
    # invoque un `_devig` local— sin prohibir mencionar el nombre en prosa.
    assert "def _devig" not in src, "_devig redefinido en backtest_and_retrain.py"
    assert "_devig(" not in src, "_devig sigue siendo invocado en backtest_and_retrain.py"


def test_backtest_and_retrain_imports_remove_vig_multiplicative():
    """Confirms the single-source-of-truth import is actually present,
    not just that the old name is gone."""
    src = (REPO_ROOT / "backtest_and_retrain.py").read_text()
    assert "from core.value_detector import remove_vig_multiplicative" in src


def test_no_circular_import_between_value_detector_and_backtest():
    """core/value_detector.py must not import anything from
    backtest_and_retrain.py — checks for an actual import statement, not
    just any textual mention (the file legitimately references
    backtest_and_retrain.py by name in a few analysis comments, e.g. about
    Monte Carlo sample-size parity — those aren't imports and must not
    fail this check)."""
    src = (REPO_ROOT / "core" / "value_detector.py").read_text()
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("import backtest_and_retrain") or \
           stripped.startswith("from backtest_and_retrain"):
            pytest.fail(f"circular import found: {stripped!r}")


def test_backtest_and_retrain_module_imports_cleanly():
    """Real, no-mock import of the module — the actual test that a
    circular import or a broken call site would fail. Run in a
    subprocess so backtest_and_retrain.py's module-level argparse/CLI
    setup can't interfere with the current pytest process."""
    result = subprocess.run(
        [sys.executable, "-c", "import backtest_and_retrain"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
