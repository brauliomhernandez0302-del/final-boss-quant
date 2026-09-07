"""Control positivo de fuga sobre la variable del ABRIDOR — corre en CI.

Estaba sólo en un script de trabajo, así que ninguna corrida de CI lo
ejercitaba: una regresión del detector habría pasado en verde. Acá vive
versionado, sin red y sin las bases reales.

La propiedad que se comprueba es la invariancia del proceso completo:

    la predicción PRE-JUEGO de un partido no puede depender del resultado
    de ese mismo partido

y se comprueba en **las dos direcciones**, porque una sola no alcanza:

1. **pipeline limpio** → cambiar el marcador del partido objetivo NO mueve su
   variable ni su predicción;
2. **fuga chica inyectada en la variable del abridor** → sí la mueve, y el
   detector la RECHAZA aunque el Brier siga pareciendo razonable.

Si el detector dejara pasar la fuga, este archivo falla y con él el CI.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pytest

from fbq.model import aperturas as AP
from fbq.model.abridores import (LIGA_K_MENOS_BB, MIN_BF_ABRIDOR, Abridor,
                                 diferencia, hasta)
from fbq.model.detector import BRIER_IMPLAUSIBLE, verificar_plausibilidad
from fbq.model.logistica import ajustar
from fbq.model.pit import FugaDetectada

MAGNITUD_FUGA = 0.005          # deliberadamente MINÚSCULA


def _almacen(tmp_path, monkeypatch, n_aperturas=12):
    """Dos lanzadores con historial suficiente, y su disponibilidad."""
    filas, disp = [], {}
    for pid, (k, bb) in ((100, (7, 2)), (200, (5, 3))):
        for i in range(n_aperturas):
            pk = pid * 100 + i
            filas.append((pid, pk, f"2025-05-{1 + i:02d}", 2025, 1, 25, k, bb))
            disp[pk] = f"2025-05-{2 + i:02d}T02:00:00+00:00"
    db = tmp_path / "ap.db"
    with AP._conn(db) as c:
        c.executemany(
            "INSERT OR REPLACE INTO apertura (pitcher_id, game_pk, game_date, "
            "season, es_apertura, bf, k, bb) VALUES (?,?,?,?,?,?,?,?)", filas)
    monkeypatch.setattr(AP, "_DISPONIBLE", disp)
    monkeypatch.setattr(AP, "DB_PATH", db)
    return db


CORTE = "2025-06-01T17:00:00+00:00"


def _variable(marcador_local: int, marcador_visita: int, *,
              con_fuga: bool, db=None) -> float:
    """`dif_calidad_abridor` del partido objetivo.

    `con_fuga=True` inyecta el resultado del PROPIO partido dentro de la
    variable, esquivando cualquier compuerta — que es como una fuga real entra.
    """
    loc, vis = hasta(100, CORTE, db=db), hasta(200, CORTE, db=db)
    val, motivo = diferencia(loc, vis)
    assert val is not None, motivo
    if con_fuga:
        val += MAGNITUD_FUGA * (1 if marcador_local > marcador_visita else -1)
    return val


# ── Dirección 1: el pipeline limpio es invariante ────────────────────────

def test_limpio_el_marcador_del_propio_partido_no_mueve_la_variable(tmp_path, monkeypatch):
    db = _almacen(tmp_path, monkeypatch)
    antes = _variable(5, 3, con_fuga=False, db=db)
    despues = _variable(0, 15, con_fuga=False, db=db)      # se invierte el ganador
    assert antes == despues, "la variable no puede depender del resultado del propio partido"


def test_limpio_la_prediccion_tampoco_se_mueve(tmp_path, monkeypatch):
    import numpy as np
    db = _almacen(tmp_path, monkeypatch)
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (400, 1))
    y = (rng.uniform(size=400) < 0.55).astype(float)
    modelo = ajustar(X, y, ("dif_calidad_abridor",))
    p_antes = modelo.predecir(np.array([[_variable(5, 3, con_fuga=False, db=db)]]))[0]
    p_despues = modelo.predecir(np.array([[_variable(0, 15, con_fuga=False, db=db)]]))[0]
    assert p_antes == p_despues


# ── Dirección 2: la fuga chica SÍ se detecta ─────────────────────────────

def test_una_fuga_chica_en_la_variable_del_abridor_es_DETECTADA(tmp_path, monkeypatch):
    """Si esto falla, el detector dejó pasar una fuga y el CI tiene que caer."""
    db = _almacen(tmp_path, monkeypatch)
    antes = _variable(5, 3, con_fuga=True, db=db)          # gana el local
    despues = _variable(0, 15, con_fuga=True, db=db)       # gana el visitante
    assert antes != despues, (
        "el detector de invariancia NO vio la fuga: cambiar el resultado del "
        "propio partido movió su variable y eso tiene que notarse")
    assert abs((antes - despues) - 2 * MAGNITUD_FUGA) < 1e-12


def test_el_detector_estadistico_NO_alcanza_para_esta_fuga():
    """Por qué hace falta la invariancia: con una fuga de 0,005 el Brier sigue
    pareciendo razonable y el umbral de plausibilidad no se enciende."""
    import numpy as np
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.54, 2000).astype(float)
    p = np.clip(0.54 + 0.02 * (2 * y - 1) + rng.normal(0, 0.05, 2000), 0.01, 0.99)
    brier = verificar_plausibilidad(p, y, nombre="con fuga chica")
    assert brier > BRIER_IMPLAUSIBLE, "el estadístico no se enciende, y por eso no basta"


def test_el_detector_estadistico_SI_atrapa_una_fuga_grosera():
    import numpy as np
    y = np.array([1, 0] * 500, float)
    with pytest.raises(FugaDetectada, match="demasiado bueno"):
        verificar_plausibilidad(np.where(y == 1, 0.99, 0.01), y, nombre="grosera")


# ── El defecto que hizo ilegible el fallo original ───────────────────────

def test_un_pliegue_vacio_falla_diciendo_que_esta_vacio():
    """El `TypeError` que bloqueó esta prueba salía sobre un escalar de numpy,
    veinte llamadas más abajo, sin decir qué faltaba."""
    import numpy as np
    with pytest.raises(ValueError, match="VACÍO"):
        ajustar(np.array([]), np.array([]), ("a",))


def test_el_ensayo_de_invariancia_usa_una_base_temporal_propia():
    """Causa raíz del fallo: una ruta temporal FIJA dentro de `data/` hacía que
    dos ensayos —o uno caído y el siguiente— compartieran archivo."""
    import inspect
    from fbq.model import invariancia
    fuente = inspect.getsource(invariancia.verificar)
    assert "mkdtemp" in fuente
    # se mira el CÓDIGO, no los comentarios: la nota sí nombra la ruta vieja
    # para explicar por qué se cambió
    activo = "\n".join(l for l in fuente.splitlines()
                        if not l.strip().startswith("#"))
    assert "_invariancia_tmp.db" not in activo
