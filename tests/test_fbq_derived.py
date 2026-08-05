"""fbq.evaluator.derived — la balanza de total y runline.

El test central es el del PUNTO. Escribiendo este módulo el 2026-08-04 cometí
exactamente PURP-1: emparejé la probabilidad de "el local cubre −1.5" con el
mejor precio disponible, que en 343 juegos estaba cotizado a +1.5. El resultado
fue un ROI ciego de +4.76% que desapareció por completo al exigir que el punto
coincidiera. La regla ya existía en `market.pair_before`; no aplicarla acá costó
una hora y casi un hallazgo falso.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbq.evaluator.derived import load_derived, roi_derivado


def _db(tmp_path, filas):
    p = tmp_path / "h.db"
    con = sqlite3.connect(p)
    con.execute("""CREATE TABLE game_outcomes (game_pk INT, season INT,
        official_date TEXT, home_team TEXT, actual_home_runs INT, actual_away_runs INT)""")
    con.execute("""CREATE TABLE historical_odds (game_pk INT,
        total_point_pin REAL, total_point_best REAL, total_over_pin REAL,
        total_under_pin REAL, total_over_best REAL, total_under_best REAL,
        rl_home_point_pin REAL, rl_home_point_best REAL, rl_home_pin REAL,
        rl_away_pin REAL, rl_home_best REAL, rl_away_best REAL)""")
    for i, (hr, ar, pt_pin, pt_best) in enumerate(filas):
        con.execute("INSERT INTO game_outcomes VALUES (?,?,?,?,?,?)",
                    (i, 2024, "2024-05-01", "Reds", hr, ar))
        con.execute("INSERT INTO historical_odds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (i, 8.5, 8.5, 1.95, 1.95, 2.00, 2.00,
                     pt_pin, pt_best, 2.40, 1.62, 2.45, 1.65))
    con.commit(); con.close()
    return p


def test_el_precio_tiene_que_estar_en_el_mismo_punto_que_la_probabilidad(tmp_path):
    """Sin esta regla se empareja el evento difícil con el precio del fácil."""
    db = _db(tmp_path, [(5, 3, -1.5, -1.5), (5, 3, -1.5, +1.5), (2, 4, -1.5, -1.5)])
    f = load_derived("runline", (2024,), db_path=db)
    assert len(f) == 2                      # el del punto cambiado queda fuera


def test_el_evento_del_runline_se_deriva_del_punto_firmado(tmp_path):
    """Local FAVORITO a −1.5 cubre ganando por 2+; local a +1.5 cubre incluso
    perdiendo por 1. Son eventos distintos, no el mismo con otro signo."""
    db = _db(tmp_path, [(5, 3, -1.5, -1.5),    # gana por 2 → cubre −1.5
                        (4, 3, -1.5, -1.5),    # gana por 1 → NO cubre −1.5
                        (3, 4, +1.5, +1.5)])   # pierde por 1 → SÍ cubre +1.5
    f = load_derived("runline", (2024,), db_path=db)
    assert list(f.y) == [1.0, 0.0, 1.0]


def test_el_evento_del_total_es_estrictamente_mayor_que_el_punto(tmp_path):
    db = _db(tmp_path, [(5, 4, -1.5, -1.5), (4, 4, -1.5, -1.5)])
    f = load_derived("total", (2024,), db_path=db)
    assert list(f.y) == [1.0, 0.0]           # 9 > 8.5 ; 8 < 8.5


def test_un_empuje_no_es_ni_acierto_ni_error(tmp_path):
    """Meterlo en un Brier como 0 o como 1 inventa un resultado que no ocurrió;
    en el ROI cuenta como apuesta con ganancia cero, que es lo que pasa."""
    db = _db(tmp_path, [(5, 3, -2.0, -2.0)])   # margen +2 == umbral +2 → empuje
    f = load_derived("runline", (2024,), db_path=db)
    assert f.empuje[0] == 1
    assert len(f.sin_empuje) == 0
    r = roi_derivado(np.array([0.99]), f, umbrales=(0.0,))
    assert r[0]["n"] == 1 and r[0]["roi_pct"] == 0.0


def test_el_empuje_no_se_excluye_del_roi(tmp_path):
    """Excluirlo inflaría el ROI: los empujes son más frecuentes justo en las
    líneas enteras donde más se apuesta."""
    db = _db(tmp_path, [(5, 3, -2.0, -2.0), (7, 3, -2.0, -2.0)])
    f = load_derived("runline", (2024,), db_path=db)
    r = roi_derivado(np.array([0.99, 0.99]), f, umbrales=(0.0,))
    assert r[0]["n"] == 2
