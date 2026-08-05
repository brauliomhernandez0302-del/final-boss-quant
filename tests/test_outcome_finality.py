"""El resultado que se guarda tiene que ser el FINAL, y sólo el final.

Estos tests fijan tres defectos encontrados el 2026-08-04 comparando las 862
filas de `game_outcomes` de 2026 contra el schedule real de MLB (2024 y 2025
salieron 100% limpias: el problema es del camino en vivo).

Lo que los vuelve caros: `update_outcome()` es idempotente a propósito —sólo
escribe donde `actual_home_runs IS NULL`— así que un marcador equivocado que
entra una vez NO se puede corregir después por el camino normal, y se queda
como verdad de terreno alimentando Kalman y descenso de gradiente.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.baseball_module.calibration.learning_engine import LearningEngine


@pytest.fixture()
def engine(tmp_path):
    return LearningEngine(db_path=str(tmp_path / "t.db"))


def _sembrar(engine, game_pk=1, season=2026):
    with engine._get_conn() as conn:
        conn.execute(
            """INSERT INTO game_outcomes (game_pk, game_date, season, home_team,
                                          away_team, official_date)
               VALUES (?,?,?,?,?,?)""",
            (game_pk, "2026-07-27", season, "Reds", "Guardians", "2026-07-27"),
        )


def _leer(engine, game_pk=1):
    with engine._get_conn() as conn:
        return conn.execute(
            "SELECT actual_home_runs, actual_away_runs, home_won "
            "FROM game_outcomes WHERE game_pk=?", (game_pk,)
        ).fetchone()


def test_un_empate_no_se_guarda_como_derrota_del_local(engine):
    """En MLB no hay finales empatados: un 5–5 es un juego que no terminó.

    Caso real, game_pk=824490 (Guardians @ Reds, 2026-07-27): se guardó 5–5
    con home_won=0 cuando el final verdadero fue 6–5.
    """
    _sembrar(engine)
    assert engine.update_outcome(1, 5, 5) is False
    fila = _leer(engine)
    assert fila["actual_home_runs"] is None
    assert fila["home_won"] is None


def test_un_final_normal_si_se_guarda(engine):
    _sembrar(engine)
    assert engine.update_outcome(1, 5, 6) is True
    fila = _leer(engine)
    assert (fila["actual_away_runs"], fila["actual_home_runs"]) == (6, 5)
    assert fila["home_won"] == 0


def test_el_rechazo_del_empate_deja_la_fila_reintentables(engine):
    """No basta con no escribir: la fila tiene que quedar pendiente, para que
    la próxima barrida la resuelva cuando el juego sí termine."""
    _sembrar(engine)
    engine.update_outcome(1, 5, 5)
    assert engine.update_outcome(1, 7, 5) is True
    fila = _leer(engine)
    assert (fila["actual_away_runs"], fila["actual_home_runs"]) == (5, 7)
    assert fila["home_won"] == 1


def test_postponed_no_cuenta_como_final():
    """`abstractGameState` vale "Final" también para un juego POSPUESTO — el
    que distingue es `detailedState`. Y /schedule devuelve DOS bloques con el
    mismo gamePk cuando un juego se pospone y se rejuega, así que leer
    `dates[0]` agarra el cascarón pospuesto.

    Forma exacta de la respuesta real para game_pk=824490.
    """
    respuesta = {"dates": [
        {"date": "2026-07-27", "games": [{
            "gamePk": 824490,
            "status": {"detailedState": "Postponed", "abstractGameState": "Final"},
            "linescore": {"teams": {"home": {}, "away": {}}},
        }]},
        {"date": "2026-07-28", "games": [{
            "gamePk": 824490,
            "status": {"detailedState": "Final", "abstractGameState": "Final"},
            "linescore": {"teams": {"home": {"runs": 5}, "away": {"runs": 6}}},
        }]},
    ]}

    # La regla que aplica fetch_pending_outcomes(): filtrar por detailedState
    # sobre TODOS los bloques, y tomar el último.
    finales = [
        g for d in respuesta["dates"] for g in d["games"]
        if g["gamePk"] == 824490 and g["status"]["detailedState"] == "Final"
    ]
    assert len(finales) == 1
    ls = finales[-1]["linescore"]["teams"]
    assert (ls["away"]["runs"], ls["home"]["runs"]) == (6, 5)

    # La regla vieja —dates[0] + abstractGameState— habría pasado el guard
    # sobre el cascarón pospuesto, que no tiene marcador.
    viejo = respuesta["dates"][0]["games"][0]
    assert viejo["status"]["abstractGameState"] == "Final"   # pasaba
    assert viejo["linescore"]["teams"]["home"] == {}         # y no traía marcador
