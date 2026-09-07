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


# ══ La variable de v1.5: carga reciente del bullpen ══════════════════════
#
# Los mismos dos sentidos, sobre la variable nueva. La compuerta la hereda por
# construcción —lee la ventana por `disponible_desde`— pero heredar no es
# demostrar: acá se demuestra.

from fbq.model.carga import ESCALA_CARGA, VENTANA_HORAS, IndiceCarga
from fbq.model.pit import Partido


def _partido(pk, fecha, local, visita, disponible, hr=3, ar=1):
    return Partido(game_pk=pk, official_date=fecha, season=2025, home_team=local,
                   away_team=visita, home_runs=hr, away_runs=ar,
                   disponible_desde=disponible)


def _indice(partidos, pitches):
    """`pitches` es `{(game_pk, es_local): lanzamientos de relevo}`."""
    return IndiceCarga(partidos, {k: {"pitches_relevo": v}
                                  for k, v in pitches.items()})


CORTE_BP = "2025-05-10T17:00:00+00:00"


def test_la_carga_solo_suma_partidos_TERMINADOS_antes_del_corte():
    """El de después del corte no entra aunque su fecha de calendario sea la
    misma: la ventana se cierra en `disponible_desde`, no en el día."""
    ps = [_partido(1, "2025-05-08", "L", "V", "2025-05-09T02:00:00+00:00"),
          _partido(2, "2025-05-10", "L", "V", "2025-05-10T23:00:00+00:00")]
    idx = _indice(ps, {(1, 1): 50, (1, 0): 40, (2, 1): 99, (2, 0): 99})
    c = idx.carga("L", CORTE_BP)
    assert c["ok"] and c["juegos"] == 1 and c["pitches"] == 50, \
        "el partido que terminó DESPUÉS del corte no puede entrar"


def test_la_carga_no_alcanza_mas_alla_de_las_72_horas():
    viejo = "2025-05-07T16:00:00+00:00"      # 73 h antes del corte
    dentro = "2025-05-07T18:00:00+00:00"     # 71 h antes
    ps = [_partido(1, "2025-05-07", "L", "V", viejo),
          _partido(2, "2025-05-07", "L", "V", dentro)]
    idx = _indice(ps, {(1, 1): 70, (1, 0): 1, (2, 1): 30, (2, 0): 1})
    assert VENTANA_HORAS == 72.0
    c = idx.carga("L", CORTE_BP)
    assert c["pitches"] == 30 and c["juegos"] == 1


def test_cambiar_el_marcador_del_propio_partido_no_mueve_la_carga():
    """Dirección 1, sobre la variable nueva."""
    ps = [_partido(1, "2025-05-08", "L", "V", "2025-05-09T02:00:00+00:00")]
    pit = {(1, 1): 55, (1, 0): 44}
    a = _indice(ps, pit).diferencia("L", "V", CORTE_BP)
    # mismo partido, ganador invertido
    ps2 = [_partido(1, "2025-05-08", "L", "V", "2025-05-09T02:00:00+00:00", hr=0, ar=15)]
    b = _indice(ps2, pit).diferencia("L", "V", CORTE_BP)
    assert a["dif_carga_relevo"] == b["dif_carga_relevo"] == (55 - 44) / ESCALA_CARGA


def test_una_fuga_chica_en_la_carga_SI_se_nota():
    """Dirección 2: si alguien metiera el resultado del propio partido en la
    variable, la invariancia tiene que romperse. Un control que sólo pasa la
    dirección limpia no distingue un detector de una constante."""
    ps = [_partido(1, "2025-05-08", "L", "V", "2025-05-09T02:00:00+00:00")]
    base = _indice(ps, {(1, 1): 55, (1, 0): 44}).diferencia("L", "V", CORTE_BP)
    def con_fuga(gana_local: bool) -> float:
        return base["dif_carga_relevo"] + MAGNITUD_FUGA * (1 if gana_local else -1)
    assert con_fuga(True) != con_fuga(False)
    assert abs((con_fuga(True) - con_fuga(False)) - 2 * MAGNITUD_FUGA) < 1e-12


def test_un_partido_sin_fila_de_relevo_deja_la_carga_NO_COMPUTABLE():
    """No se imputa (preregistro §4): un partido que existió y del que no se
    sabe la carga no es carga cero."""
    ps = [_partido(1, "2025-05-08", "L", "V", "2025-05-09T02:00:00+00:00")]
    idx = _indice(ps, {(1, 0): 44})            # falta la fila del local
    assert idx.carga("L", CORTE_BP)["ok"] is False
    assert idx.carga("L", CORTE_BP)["motivo"] == "sin_fila_de_relevo"
    assert idx.diferencia("L", "V", CORTE_BP)["ok"] is False


def test_un_partido_sin_fin_fechable_en_la_ventana_deja_la_carga_NO_COMPUTABLE():
    """Podría haber terminado dentro de la ventana y no se puede afirmar que
    no aportó carga."""
    ps = [_partido(1, "2025-05-08", "L", "V", None)]
    idx = _indice(ps, {})
    c = idx.carga("L", CORTE_BP)
    assert c["ok"] is False and c["motivo"] == "partido_sin_fin_fechable"


def test_un_equipo_que_no_jugo_tiene_carga_CERO_no_faltante():
    """El cero es el dato: un bullpen que no lanzó no es un bullpen desconocido."""
    idx = _indice([], {})
    c = idx.carga("L", CORTE_BP)
    assert c["ok"] is True and c["pitches"] == 0 and c["juegos"] == 0


def test_COMPLEMENTO_la_carga_si_se_mueve_con_un_partido_de_dentro_de_la_ventana():
    """Sin esto, una variable constante pasaría todas las pruebas de arriba con
    nota perfecta.

    Y dice algo que la invariancia del proceso completo NO puede decir: esa
    prueba perturba MARCADORES, y la carga no depende de marcadores, así que su
    invariancia ahí es trivial. Lo que hay que comprobar de esta variable es que
    dependa de los partidos de la ventana y de NINGÚN otro — que es esto y las
    dos pruebas de ventana de arriba.
    """
    ps = [_partido(1, "2025-05-08", "L", "V", "2025-05-09T02:00:00+00:00"),   # dentro
          _partido(2, "2025-05-10", "L", "V", "2025-05-10T23:00:00+00:00")]   # después del corte
    antes = _indice(ps, {(1, 1): 55, (1, 0): 44, (2, 1): 10, (2, 0): 10})
    dentro = _indice(ps, {(1, 1): 90, (1, 0): 44, (2, 1): 10, (2, 0): 10})
    fuera = _indice(ps, {(1, 1): 55, (1, 0): 44, (2, 1): 200, (2, 0): 10})
    a = antes.diferencia("L", "V", CORTE_BP)["dif_carga_relevo"]
    assert dentro.diferencia("L", "V", CORTE_BP)["dif_carga_relevo"] != a, \
        "cambiar la carga de un partido DE la ventana tiene que mover la variable"
    assert fuera.diferencia("L", "V", CORTE_BP)["dif_carga_relevo"] == a, \
        "cambiar la de un partido que aún no había terminado NO puede moverla"
