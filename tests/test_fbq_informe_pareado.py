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
    p = r["poblaciones"]["prospectiva_verificada|sin_anotar"]
    assert p["pares"] == 1 and p["partidos_unicos"] == 1
    post = r["emisiones_posteriores"]["prospectiva_verificada|sin_anotar"]
    assert post["n"] == 2, "las otras dos se conservan, aparte"


def test_se_elige_el_PRIMER_corte_no_el_mejor(tmp_path):
    """El criterio no puede depender del resultado: el segundo par acierta más,
    y aun así se usa el primero."""
    filas = (_par(1, "2026-09-07T02:00:00+00:00", 0.50, 0.50)
             + _par(1, "2026-09-07T05:00:00+00:00", 0.99, 0.99))
    pred, res = _bases(tmp_path, filas, [(1, 1)])   # ganó el local
    p = informe(db_pred=pred, db_res=res)["poblaciones"]["prospectiva_verificada|sin_anotar"]
    assert p["brier_v1_2"] == pytest.approx(0.25), "usó el par de las 02:00"


def test_una_emision_sin_pareja_no_entra_ni_desplaza_a_la_completa(tmp_path):
    filas = [(1, "2026-09-07T01:00:00+00:00", "v1.4", 0.9, "prospectiva",
              "prospectiva_verificada", "SHA14", "2099-01-01T00:00:00+00:00")]
    filas += _par(1, "2026-09-07T02:00:00+00:00", 0.50, 0.50)
    pred, res = _bases(tmp_path, filas, [(1, 1)])
    r = informe(db_pred=pred, db_res=res)
    p = r["poblaciones"]["prospectiva_verificada|sin_anotar"]
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
    p = r["poblaciones"]["prospectiva_verificada|sin_anotar"]
    assert p["pares"] == 10 and p["partidos_unicos"] == 10
    assert r["emisiones_posteriores"]["prospectiva_verificada|sin_anotar"]["n"] == 20


def test_un_partido_sin_resultado_queda_pendiente_no_se_cuenta(tmp_path):
    filas = _par(1, "2026-09-07T02:00:00+00:00", 0.55, 0.56)
    pred, res = _bases(tmp_path, filas, [])
    r = informe(db_pred=pred, db_res=res)
    assert "prospectiva_verificada|sin_anotar" not in r["poblaciones"]
    assert r["pendientes"]["prospectiva_verificada|sin_anotar:sin_resultado_todavia"] == 1


def test_las_cohortes_no_se_mezclan(tmp_path):
    filas = _par(1, "2026-09-07T02:00:00+00:00", 0.55, 0.56)
    filas += _par(2, "2026-08-02T17:00:00+00:00", 0.55, 0.56, origen="reconstruccion")
    pred, res = _bases(tmp_path, filas, [(1, 1), (2, 0)])
    r = informe(db_pred=pred, db_res=res)
    assert set(r["poblaciones"]) == {"prospectiva_verificada|sin_anotar", "reconstruccion|sin_anotar"}
    assert all(p["pares"] == 1 for p in r["poblaciones"].values())


def test_un_par_vale_lo_que_su_pierna_mas_debil(tmp_path):
    """Una v1.4 escrita antes del partido no sirve si su v1.2 no lo está."""
    filas = [(1, "2026-09-07T02:00:00+00:00", "v1.2", 0.55, "prospectiva",
              "no_verificable", "SHA12", "2099-01-01T00:00:00+00:00"),
             (1, "2026-09-07T02:00:00+00:00", "v1.4", 0.56, "prospectiva",
              "prospectiva_verificada", "SHA14", "2099-01-01T00:00:00+00:00")]
    pred, res = _bases(tmp_path, filas, [(1, 1)])
    r = informe(db_pred=pred, db_res=res)
    assert set(r["poblaciones"]) == {"no_verificable|sin_anotar"}


# ── Calidad de datos: dimensión APARTE de la condición prospectiva ───────

def _con_calidad(tmp_path, filas, resultados, anotaciones):
    pred, res = _bases(tmp_path, filas, resultados)
    con = sqlite3.connect(pred)
    con.execute("CREATE TABLE calidad_datos (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "anotado_en TEXT, game_pk INT, version TEXT, corte TEXT, "
                "modelo_sha TEXT, calidad TEXT, motivo TEXT, historial_hasta TEXT, "
                "historial_filas INT, dias_sin_datos INT)")
    con.executemany("INSERT INTO calidad_datos (anotado_en, game_pk, version, corte, "
                    "modelo_sha, calidad) VALUES (?,?,?,?,?,?)", anotaciones)
    con.commit(); con.close()
    return pred, res


def test_una_emision_puede_ser_prospectiva_Y_tener_entradas_incompletas(tmp_path):
    """Las dos cosas son ciertas a la vez y ninguna anula a la otra: degradar
    la primera escondería que la predicción sí se hizo antes."""
    filas = _par(1, "2026-09-07T02:00:00+00:00", 0.55, 0.56)
    anot = [("2026-09-07T13:00:00+00:00", 1, v, "2026-09-07T02:00:00+00:00",
             s, "historial_incompleto")
            for v, s in (("v1.2", "SHA12"), ("v1.4", "SHA14"))]
    pred, res = _con_calidad(tmp_path, filas, [(1, 1)], anot)
    r = informe(db_pred=pred, db_res=res)
    clave = "prospectiva_verificada|historial_incompleto"
    assert clave in r["poblaciones"]
    p = r["poblaciones"][clave]
    assert p["origen"] == "prospectiva_verificada"
    assert p["calidad_datos"] == "historial_incompleto"


def test_calidades_distintas_NO_se_suman_en_la_misma_poblacion(tmp_path):
    filas = _par(1, "2026-09-07T02:00:00+00:00", 0.55, 0.56)
    filas += _par(2, "2026-09-09T02:00:00+00:00", 0.55, 0.56)
    anot = [("x", 1, v, "2026-09-07T02:00:00+00:00", s, "historial_incompleto")
            for v, s in (("v1.2", "SHA12"), ("v1.4", "SHA14"))]
    anot += [("x", 2, v, "2026-09-09T02:00:00+00:00", s, "completo")
             for v, s in (("v1.2", "SHA12"), ("v1.4", "SHA14"))]
    pred, res = _con_calidad(tmp_path, filas, [(1, 1), (2, 0)], anot)
    r = informe(db_pred=pred, db_res=res)
    assert set(r["poblaciones"]) == {
        "prospectiva_verificada|historial_incompleto",
        "prospectiva_verificada|completo"}
    assert all(p["pares"] == 1 for p in r["poblaciones"].values())


def test_vale_la_ULTIMA_anotacion_de_calidad(tmp_path):
    """Una anotación equivocada se corrige agregando otra fila, no pisándola —
    y el informe tiene que leer la corrección."""
    filas = _par(1, "2026-08-02T17:00:00+00:00", 0.55, 0.56)
    anot = []
    for v, s in (("v1.2", "SHA12"), ("v1.4", "SHA14")):
        anot.append(("2026-09-07T13:00:00+00:00", 1, v, "2026-08-02T17:00:00+00:00",
                     s, "historial_incompleto"))          # anotación de más
        anot.append(("2026-09-07T14:00:00+00:00", 1, v, "2026-08-02T17:00:00+00:00",
                     s, "completo"))                      # corrección posterior
    pred, res = _con_calidad(tmp_path, filas, [(1, 1)], anot)
    r = informe(db_pred=pred, db_res=res)
    assert list(r["poblaciones"]) == ["prospectiva_verificada|completo"]


def test_un_par_hereda_la_PEOR_calidad_de_sus_dos_piernas(tmp_path):
    filas = _par(1, "2026-09-07T02:00:00+00:00", 0.55, 0.56)
    anot = [("x", 1, "v1.2", "2026-09-07T02:00:00+00:00", "SHA12", "completo"),
            ("x", 1, "v1.4", "2026-09-07T02:00:00+00:00", "SHA14", "historial_incompleto")]
    pred, res = _con_calidad(tmp_path, filas, [(1, 1)], anot)
    r = informe(db_pred=pred, db_res=res)
    assert list(r["poblaciones"]) == ["prospectiva_verificada|historial_incompleto"]


# ── El desglose por partido, que se publica en CADA corrida ──────────────

def _bases_completas(tmp_path, filas, resultados_full):
    """Como `_bases`, pero con el `resultado` COMPLETO que el desglose lee."""
    pred = tmp_path / "prospectiva.db"
    con = sqlite3.connect(pred)
    con.execute("CREATE TABLE prediccion (id INTEGER PRIMARY KEY, game_pk INT, "
                "corte TEXT, version TEXT, p_home REAL, cohorte TEXT, origen TEXT, "
                "modelo_sha TEXT, commence_time TEXT)")
    con.executemany("INSERT INTO prediccion (game_pk, corte, version, p_home, "
                    "cohorte, origen, modelo_sha, commence_time) VALUES (?,?,?,?,?,?,?,?)",
                    filas)
    con.commit(); con.close()
    res = tmp_path / "results.db"
    con = sqlite3.connect(res)
    con.execute("CREATE TABLE resultado (game_pk INT, official_date TEXT, "
                "home_team TEXT, away_team TEXT, home_runs INT, away_runs INT, "
                "home_won INT)")
    con.executemany("INSERT INTO resultado VALUES (?,?,?,?,?,?,?)", resultados_full)
    con.commit(); con.close()
    return pred, res


def test_el_desglose_trae_una_fila_por_par_con_lo_que_paso(tmp_path):
    filas = _par(1, "2026-09-07T02:00:00+00:00", 0.60, 0.55)
    pred, res = _bases_completas(
        tmp_path, filas,
        [(1, "2026-09-07", "Locales", "Visitas", 5, 3, 1)])
    r = informe(db_pred=pred, db_res=res)
    d = r["desglose_por_partido"]
    assert len(d) == 1
    f = d[0]
    assert (f["official_date"], f["home_team"], f["away_team"]) == (
        "2026-09-07", "Locales", "Visitas")
    assert f["marcador"] == "5-3" and f["gano"] == "local"
    assert f["p12"] == 0.60 and f["p14"] == 0.55
    assert f["b12"] == pytest.approx(0.16) and f["b14"] == pytest.approx(0.2025)
    assert f["delta_brier"] == pytest.approx(0.0425)
    assert f["poblacion"].startswith("prospectiva_verificada")


def test_el_desglose_NO_incluye_partidos_sin_resultado(tmp_path):
    """Lo que no terminó no tiene nada que mostrar, y contarlo insinuaría que sí."""
    filas = _par(1, "2026-09-07T02:00:00+00:00", 0.60, 0.55)
    filas += _par(2, "2026-09-08T02:00:00+00:00", 0.60, 0.55)
    pred, res = _bases_completas(
        tmp_path, filas, [(1, "2026-09-07", "L", "V", 5, 3, 1)])
    r = informe(db_pred=pred, db_res=res)
    assert [f["game_pk"] for f in r["desglose_por_partido"]] == [1]
    assert r["pendientes"]["prospectiva_verificada|sin_anotar:sin_resultado_todavia"] == 1


def test_con_y_igual_a_uno_el_signo_de_la_diferencia_lo_fija_quien_predijo_mas_alto():
    """Aritmética, no interpretación: si el local gana SIEMPRE,
    b14 − b12 = (p14 − p12)(p14 + p12 − 2) y el segundo factor es negativo.

    O sea que en una muestra donde todos los partidos los gana el local, la
    comparación pareada no mide qué modelo predice mejor: mide cuál fue más
    optimista con el local. Esta prueba existe para que esa propiedad quede
    escrita y nadie lea un ranking donde no lo hay.
    """
    for p12, p14 in ((0.60, 0.55), (0.44, 0.51), (0.50, 0.50)):
        d = (p14 - 1) ** 2 - (p12 - 1) ** 2
        assert (d > 0) == (p14 < p12)
        assert (d == 0) == (p14 == p12)


def test_el_cotejo_de_abridores_distingue_cambio_de_falta_de_anuncio():
    """Tres respuestas, ninguna disfrazada de otra."""
    from fbq.model.informe_pareado import _cotejo_abridores

    class _A:
        def __init__(self, pid): self._d = {"pitcher_id": pid,
                                            "pitcher_nombre": "X",
                                            "observado_en": "2026-09-07T01:00:00+00:00"}
        def __getitem__(self, k): return self._d[k]

    anunciados = {(1, "home"): _A(100), (1, "away"): _A(200), (2, "home"): None,
                  (2, "away"): _A(400)}
    reales = {(1, "home"): 100, (1, "away"): 999, (2, "home"): 300}
    c1 = _cotejo_abridores(1, "c", anunciados, reales)
    assert c1["home"]["coincide"] == "coincidio"
    assert c1["away"]["coincide"] == "cambio"
    c2 = _cotejo_abridores(2, "c", anunciados, reales)
    assert c2["home"]["coincide"] == "sin_anuncio_al_corte"
    assert c2["away"]["coincide"] == "sin_dato", \
        "sin abridor real no se puede afirmar ni coincidencia ni cambio"
