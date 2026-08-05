"""fbq.results — el almacén de hechos (paso 2).

Las tres primeras reglas se validan en el CONSTRUCTOR de `Final`: un estado que
no es final, o un empate, no llegan a existir como objeto. Eso deja el fetcher
sin ninguna condición sutil que revisar y hace imposible escribir un no-final
por descuido desde cualquier llamador futuro.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbq.results.store import Final, ResultsStore


def _final(game_pk=1, home=5, away=3, estado="Final"):
    return Final(game_pk=game_pk, official_date="2026-07-28", season=2026,
                 home_team="Reds", away_team="Guardians",
                 home_runs=home, away_runs=away, detailed_state=estado)


@pytest.fixture()
def store(tmp_path):
    return ResultsStore(db_path=tmp_path / "results.db")


# ── Lo que no puede llegar a existir ─────────────────────────────────────


def test_un_empate_no_se_puede_construir():
    """En MLB no hay finales empatados: un 5–5 significa que lo que llegó no
    es un final. Caso real: game_pk=824490, guardado 5–5 con home_won=0 cuando
    el final verdadero fue 6–5."""
    with pytest.raises(ValueError, match="empatado"):
        _final(home=5, away=5)


@pytest.mark.parametrize("estado", ["Postponed", "In Progress", "Suspended", "Scheduled"])
def test_un_estado_que_no_es_final_no_se_puede_construir(estado):
    """`abstractGameState` vale "Final" también para un POSPUESTO. El que
    distingue es `detailedState`, y acá es lista de permitidos, no de
    prohibidos: un estado nuevo del proveedor entra como no-final."""
    with pytest.raises(ValueError, match="no es"):
        _final(estado=estado)


# ── Append-only y correcciones ───────────────────────────────────────────


def test_no_se_puede_actualizar_ni_borrar_una_observacion(store):
    store.registrar([_final()])
    for sql in ("UPDATE observacion SET home_runs = 9", "DELETE FROM observacion"):
        with pytest.raises(sqlite3.IntegrityError):
            with store._conn() as conn:
                conn.execute(sql)


def test_reobservar_el_mismo_marcador_no_agrega_fila(store):
    store.registrar([_final()])
    r = store.registrar([_final()])
    assert r == {"nuevos": 0, "corregidos": 0, "sin_cambio": 1}
    assert store.resumen()["observaciones"] == 1


def test_una_correccion_agrega_y_conserva_la_anterior(store):
    """El marcador oficial puede cambiar. Sin conservar la anterior, 'el
    marcador cambió' es indistinguible de 'lo escribimos mal', y la diferencia
    importa cuando algo ya entrenó con el viejo."""
    store.registrar([_final(home=5, away=3)])
    r = store.registrar([_final(home=6, away=3)])
    assert r["corregidos"] == 1
    assert store.resumen()["observaciones"] == 2
    assert store.resultado(1)["home_runs"] == 6          # la vista da la vigente
    assert len(store.correcciones()) == 1                 # y queda el rastro


def test_home_won_se_deriva_del_marcador_no_se_guarda(store):
    """Guardar `home_won` como columna propia permite que contradiga al
    marcador. En la tabla anterior había una fila con 5–5 y home_won=0."""
    store.registrar([_final(home=3, away=7)])
    assert store.resultado(1)["home_won"] == 0
    store.registrar([_final(game_pk=2, home=7, away=3)])
    assert store.resultado(2)["home_won"] == 1


def test_la_vista_devuelve_un_juego_una_fila(store):
    store.registrar([_final(home=5, away=3)])
    store.registrar([_final(home=6, away=3)])
    store.registrar([_final(home=7, away=3)])
    assert len(store.por_temporada([2026])) == 1
    assert store.resultado(1)["home_runs"] == 7


def test_solo_hechos_ninguna_columna_de_modelo(store):
    """Un hecho y una opinión tienen ciclos de vida distintos. Compartir tabla
    es garantizar que tarde o temprano una pise a la otra — eso destruyó 563
    filas de predicciones en vivo sin backup."""
    with store._conn() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(observacion)")}
    prohibidas = {"lambda_home", "lambda_away", "p_home", "p_away",
                  "backtest_p_home", "stage_factors_json"}
    assert not (cols & prohibidas)
