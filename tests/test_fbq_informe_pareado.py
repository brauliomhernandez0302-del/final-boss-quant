"""La regla de un par por partido, y que los repetidos no inflen la muestra.

`docs/REGLA_SELECCION_PAREJA_2026-09-07.md`, declarada con 0 partidos evaluados.
Un partido con tres emisiones aporta UN par: contar las tres haría caer el error
estándar por √3 sin una sola observación nueva.
"""

from __future__ import annotations

import sqlite3

import pytest

from fbq.model.informe_pareado import informe


def _bases(tmp_path, filas, resultados):
    pred = tmp_path / "prospectiva.db"
    con = sqlite3.connect(pred)
    con.execute("CREATE TABLE prediccion (id INTEGER PRIMARY KEY, game_pk INT, "
                "corte TEXT, version TEXT, p_home REAL, cohorte TEXT, origen TEXT, "
                "modelo_sha TEXT, commence_time TEXT)")
    con.executemany("INSERT INTO prediccion (game_pk, corte, version, p_home, "
                    "cohorte, origen, modelo_sha, commence_time) "
                    "VALUES (?,?,?,?,?,?,?,?)", filas)
    con.commit(); con.close()

    res = tmp_path / "results.db"
    con = sqlite3.connect(res)
    con.execute("CREATE TABLE resultado (game_pk INT, home_won INT)")
    con.executemany("INSERT INTO resultado VALUES (?,?)", resultados)
    con.commit(); con.close()
    return pred, res


def _par(pk, corte, p12, p14, origen="prospectiva_verificada"):
    return [(pk, corte, "v1.2", p12, "prospectiva", origen, "SHA12", "2099-01-01T00:00:00+00:00"),
            (pk, corte, "v1.4", p14, "prospectiva", origen, "SHA14", "2099-01-01T00:00:00+00:00")]


@pytest.fixture(autouse=True)
def _shas(monkeypatch):
    import fbq.model.informe_pareado as M
    monkeypatch.setattr(M, "_shas_vigentes", lambda: {"v1.2": "SHA12", "v1.4": "SHA14"})


def test_un_partido_con_tres_emisiones_aporta_UN_par(tmp_path):
    filas = (_par(1, "2026-09-07T02:00:00+00:00", 0.60, 0.62)
             + _par(1, "2026-09-07T03:00:00+00:00", 0.61, 0.63)
             + _par(1, "2026-09-07T04:00:00+00:00", 0.62, 0.64))
    pred, res = _bases(tmp_path, filas, [(1, 1)])
    r = informe(db_pred=pred, db_res=res)
    p = r["poblaciones"]["prospectiva_verificada"]
    assert p["pares"] == 1 and p["partidos_unicos"] == 1
    post = r["emisiones_posteriores"]["prospectiva_verificada"]
    assert post["n"] == 2, "las otras dos se conservan, aparte"


def test_se_elige_el_PRIMER_corte_no_el_mejor(tmp_path):
    """El criterio no puede depender del resultado: el segundo par acierta más,
    y aun así se usa el primero."""
    filas = (_par(1, "2026-09-07T02:00:00+00:00", 0.50, 0.50)
             + _par(1, "2026-09-07T05:00:00+00:00", 0.99, 0.99))
    pred, res = _bases(tmp_path, filas, [(1, 1)])   # ganó el local
    p = informe(db_pred=pred, db_res=res)["poblaciones"]["prospectiva_verificada"]
    assert p["brier_v1_2"] == pytest.approx(0.25), "usó el par de las 02:00"


def test_una_emision_sin_pareja_no_entra_ni_desplaza_a_la_completa(tmp_path):
    filas = [(1, "2026-09-07T01:00:00+00:00", "v1.4", 0.9, "prospectiva",
              "prospectiva_verificada", "SHA14", "2099-01-01T00:00:00+00:00")]
    filas += _par(1, "2026-09-07T02:00:00+00:00", 0.50, 0.50)
    pred, res = _bases(tmp_path, filas, [(1, 1)])
    r = informe(db_pred=pred, db_res=res)
    p = r["poblaciones"]["prospectiva_verificada"]
    assert p["pares"] == 1
    assert p["brier_v1_2"] == pytest.approx(0.25), "la suelta de las 01:00 no cuenta"
    assert r["pendientes"]["sin_las_dos_versiones"] == 1


def test_los_partidos_repetidos_no_inflan_la_muestra(tmp_path):
    """Diez partidos × tres emisiones = 10 pares, no 30."""
    filas = []
    for pk in range(1, 11):
        for h in (2, 3, 4):
            filas += _par(pk, f"2026-09-07T0{h}:00:00+00:00", 0.55, 0.56)
    pred, res = _bases(tmp_path, filas, [(pk, pk % 2) for pk in range(1, 11)])
    r = informe(db_pred=pred, db_res=res)
    p = r["poblaciones"]["prospectiva_verificada"]
    assert p["pares"] == 10 and p["partidos_unicos"] == 10
    assert r["emisiones_posteriores"]["prospectiva_verificada"]["n"] == 20


def test_un_partido_sin_resultado_queda_pendiente_no_se_cuenta(tmp_path):
    filas = _par(1, "2026-09-07T02:00:00+00:00", 0.55, 0.56)
    pred, res = _bases(tmp_path, filas, [])
    r = informe(db_pred=pred, db_res=res)
    assert "prospectiva_verificada" not in r["poblaciones"]
    assert r["pendientes"]["prospectiva_verificada:sin_resultado_todavia"] == 1


def test_las_cohortes_no_se_mezclan(tmp_path):
    filas = _par(1, "2026-09-07T02:00:00+00:00", 0.55, 0.56)
    filas += _par(2, "2026-08-02T17:00:00+00:00", 0.55, 0.56, origen="reconstruccion")
    pred, res = _bases(tmp_path, filas, [(1, 1), (2, 0)])
    r = informe(db_pred=pred, db_res=res)
    assert set(r["poblaciones"]) == {"prospectiva_verificada", "reconstruccion"}
    assert all(p["pares"] == 1 for p in r["poblaciones"].values())


def test_un_par_vale_lo_que_su_pierna_mas_debil(tmp_path):
    """Una v1.4 escrita antes del partido no sirve si su v1.2 no lo está."""
    filas = [(1, "2026-09-07T02:00:00+00:00", "v1.2", 0.55, "prospectiva",
              "no_verificable", "SHA12", "2099-01-01T00:00:00+00:00"),
             (1, "2026-09-07T02:00:00+00:00", "v1.4", 0.56, "prospectiva",
              "prospectiva_verificada", "SHA14", "2099-01-01T00:00:00+00:00")]
    pred, res = _bases(tmp_path, filas, [(1, 1)])
    r = informe(db_pred=pred, db_res=res)
    assert set(r["poblaciones"]) == {"no_verificable"}
