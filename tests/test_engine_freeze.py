"""El congelamiento del motor del protocolo de CLV, verificado en cada corrida.

`docs/PROTOCOLO_CLV_V1.md` fija un `engine_commit` y prohíbe tocar el motor de
predicción de moneyline mientras la ventana esté abierta. El stamp que cada
pick guarda en `picks.engine_commit` NO alcanza para certificar eso: es el HEAD
del repo (`track_record/publisher.py::_get_engine_commit()`), así que avanza con
cualquier commit — de UI, de docs, de tests — sin que el motor haya cambiado.

Lo que sí certifica el congelamiento es esto: el diff entre el `engine_commit`
registrado y HEAD, restringido a los paths del motor, tiene que estar VACÍO.
Este test lo hace ejecutable en vez de dejarlo como verificación manual de una
sesión.
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PROTOCOL = ROOT / "docs" / "PROTOCOLO_CLV_V1.md"

# El camino de predicción de moneyline, tal como lo enumera CLAUDE.md.
FROZEN_PATHS = [
    "modules/baseball_module/montecarlo/",
    "modules/baseball_module/offense/",
    "modules/baseball_module/context_engine/",
    "modules/baseball_module/hfa/",
    "modules/baseball_module/core/run_module.py",
    "core/value_detector.py",
    "calibration/learning_engine.py",
    "backtest_and_retrain.py",
    "config.py",
    "scripts/promote_calibration.py",
]


def _registered_engine_commit() -> str:
    text = PROTOCOL.read_text(encoding="utf-8")
    match = re.search(r"engine_commit congelado:\s*`([0-9a-f]{7,40})`", text)
    assert match, "no se encontró el engine_commit en el Registro del protocolo"
    return match.group(1)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def test_engine_paths_unchanged_since_registered_commit():
    commit = _registered_engine_commit()
    try:
        _git("cat-file", "-e", f"{commit}^{{commit}}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"commit {commit} no disponible en este checkout")

    changed = _git("diff", "--name-only", f"{commit}..HEAD", "--", *FROZEN_PATHS).split()
    assert changed == [], (
        f"El motor cambió desde el engine_commit registrado ({commit[:7]}): {changed}. "
        "Si el cambio es intencional, el protocolo exige re-apuntar el engine_commit "
        "y, si la ventana ya arrancó, reiniciar la muestra primaria."
    )


def test_working_tree_engine_paths_are_clean():
    """Un cambio sin commitear en el motor también rompe el congelamiento —
    el cron corre desde el working tree, no desde HEAD."""
    dirty = _git("status", "--porcelain", "--", *FROZEN_PATHS).strip()
    assert dirty == "", f"cambios sin commitear en el camino congelado:\n{dirty}"
