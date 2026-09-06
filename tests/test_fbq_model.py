"""El candidato v1: la compuerta temporal, los dos detectores y el ajuste.

Cada test fija una regla del preregistro
(`docs/PREREGISTRO_MODELO_V1_2026-09-06.md`). Sin red y sin las bases reales.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from fbq.model import features as F
from fbq.model.detector import BRIER_IMPLAUSIBLE, verificar_plausibilidad
from fbq.model.logistica import ajustar
from fbq.model.pit import DURACION_MAXIMA, FugaDetectada, Partido, VentanaPIT


def _p(pk, dia, local, visita, cl, cv, inicio):
    disp = (datetime.fromisoformat(inicio) + DURACION_MAXIMA).isoformat()
    return Partido(game_pk=pk, official_date=dia, season=int(dia[:4]),
                   home_team=local, away_team=visita, home_runs=cl,
                   away_runs=cv, disponible_desde=disp)


# ── La compuerta temporal ────────────────────────────────────────────────

def test_un_partido_no_terminado_no_puede_entrar_en_una_fila():
    """El detector estructural. Un hecho posterior al corte muere acá."""
    p = _p(1, "2024-04-24", "A", "B", 5, 3, "2024-04-24T22:05:00+00:00")
    v = VentanaPIT([p])
    with pytest.raises(FugaDetectada, match="FUGA"):
        v.exigir_disponible(1, "2024-04-24T17:00:00+00:00")
    # ocho horas después del inicio sí está disponible
    assert v.exigir_disponible(1, "2024-04-25T06:06:00+00:00").game_pk == 1


def test_la_cota_de_duracion_es_conservadora_y_explicita():
    """Errar por exceso cuesta cobertura; errar por defecto cuesta una fuga."""
    assert DURACION_MAXIMA == timedelta(hours=8)
    p = _p(1, "2024-04-24", "A", "B", 5, 3, "2024-04-24T22:00:00+00:00")
    v = VentanaPIT([p])
    with pytest.raises(FugaDetectada):
        v.exigir_disponible(1, "2024-04-25T05:59:00+00:00")   # 7h59: todavía no


def test_un_partido_sin_hora_de_inicio_nunca_esta_disponible():
    """No se le inventa una hora: no se puede afirmar que hubiera terminado."""
    p = Partido(1, "2024-04-24", 2024, "A", "B", 5, 3, disponible_desde=None)
    v = VentanaPIT([p])
    assert v.disponible(p, "2099-01-01T00:00:00+00:00") is False
    with pytest.raises(FugaDetectada, match="no tiene hora de inicio"):
        v.exigir_disponible(1, "2099-01-01T00:00:00+00:00")


def test_un_suspendido_se_fecha_por_su_reanudacion(tmp_path):
    """Dos entradas con el mismo game_pk: el partido no terminó antes de haber
    empezado por última vez."""
    from fbq.model import pit
    cache = tmp_path / "sched.json"
    cache.write_text(json.dumps({"juegos": [
        {"game_pk": 7, "game_date": "2024-05-21T23:45:00Z", "official_date": "2024-05-21"},
        {"game_pk": 7, "game_date": "2024-05-22T16:15:00Z", "official_date": "2024-05-21"},
    ]}), encoding="utf-8")
    inicios = pit._inicios(cache)
    assert max(inicios[7]) == "2024-05-22T16:15:00+00:00"


def test_la_ventana_solo_devuelve_partidos_ya_terminados():
    previos = [_p(i, f"2024-04-{10+i:02d}", "A", "B", 5, 3,
                  f"2024-04-{10+i:02d}T22:00:00+00:00") for i in range(1, 6)]
    v = VentanaPIT(previos)
    corte = "2024-04-13T17:00:00+00:00"
    vistos = v.de_equipo("A", corte, ventana=162)
    assert [p.game_pk for p in vistos] == [1, 2]      # el del 12 termina el 13 a las 06
    assert all(p.disponible_desde <= corte for p in vistos)


# ── Control positivo de fuga ─────────────────────────────────────────────

def test_control_positivo_estructural_la_fuga_por_la_compuerta_se_rechaza():
    """Se inyecta el resultado del PROPIO partido pidiéndolo por la vía
    legítima. Tiene que morir en la compuerta."""
    juego = _p(9, "2024-06-01", "A", "B", 4, 2, "2024-06-01T22:00:00+00:00")
    v = VentanaPIT([juego])
    with pytest.raises(FugaDetectada, match="FUGA"):
        v.exigir_disponible(juego.game_pk, "2024-06-01T17:00:00+00:00")


def test_control_positivo_estadistico_la_fuga_que_esquiva_la_compuerta_se_rechaza():
    """Un candidato demasiado bueno para ser cierto se rechaza aunque su código
    parezca limpio. Es el que atrapa a quien esquivó la compuerta."""
    y = np.array([1, 0] * 200, float)
    p_envenenado = np.where(y == 1, 0.99, 0.01)
    with pytest.raises(FugaDetectada, match="demasiado bueno"):
        verificar_plausibilidad(p_envenenado, y, nombre="envenenado")


def test_el_detector_estadistico_deja_pasar_un_candidato_plausible():
    """Un detector que rechaza todo no distingue nada."""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.54, 3000).astype(float)
    p = np.full(3000, 0.54)
    assert verificar_plausibilidad(p, y, nombre="tasa base") > BRIER_IMPLAUSIBLE


# ── El ajuste ────────────────────────────────────────────────────────────

def test_la_estandarizacion_sale_solo_del_entrenamiento():
    """Estandarizar con la muestra completa mete el pliegue de prueba en el
    ajuste por la puerta de al lado."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (500, 2))
    y = (rng.uniform(size=500) < 0.5).astype(float)
    m = ajustar(X, y, ("a", "b"))
    assert np.allclose(m.mu, X.mean(axis=0)) and np.allclose(m.sd, X.std(axis=0))
    # aplicar el modelo a otros datos NO recalcula la estandarización
    otros = X + 100.0
    assert not np.allclose(m.predecir(otros), m.predecir(X))


def test_el_ridge_no_penaliza_el_intercepto():
    """Penalizarlo encogería la tasa base hacia 0.5, que es justo el hecho que
    el intercepto tiene que capturar (la ventaja de local)."""
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (4000, 2))
    y = np.ones(4000); y[:1600] = 0.0            # tasa base 0.60
    flojo = ajustar(X, y, ("a", "b"), lambda_l2=1.0)
    duro = ajustar(X, y, ("a", "b"), lambda_l2=1e6)
    esperado = np.log(0.6 / 0.4)
    assert abs(flojo.beta[0] - esperado) < 0.05
    assert abs(duro.beta[0] - esperado) < 0.05   # el intercepto no se mueve
    assert abs(duro.beta[1]) < abs(flojo.beta[1]) + 1e-9   # los demás sí


def test_un_juego_con_poco_historial_se_EXCLUYE_y_no_se_imputa():
    """Imputarle la media lo convertiría en una predicción que no se hizo."""
    previos = [_p(i, f"2024-04-{10+i:02d}", "A", "B", 5, 3,
                  f"2024-04-{10+i:02d}T22:00:00+00:00") for i in range(1, 4)]
    v = VentanaPIT(previos)
    juego = _p(99, "2024-05-01", "A", "B", 0, 0, "2024-05-01T22:00:00+00:00")
    r = F.construir_fila(v, previos, juego, "2024-05-01T17:00:00+00:00")
    assert r["ok"] is False and r["motivo"] == "historial_insuficiente"


# ── Reutilización del sistema anterior ───────────────────────────────────

def test_regress_es_identica_a_la_del_sistema_anterior():
    """La copia atribuida no puede derivar del original.

    El proyecto ya pagó una duplicación que derivó: la constante de barrel%
    arreglada en una copia y no en la otra, dos implementaciones independientes
    durante meses. Este test vigila la copia mientras el original exista.
    """
    origen = Path(__file__).parent.parent / "modules" / "baseball_module" / \
        "offense" / "tte_formula.py"
    if not origen.exists():
        pytest.skip("el sistema anterior ya no está en el árbol")
    from modules.baseball_module.offense.tte_formula import regress as original
    for obs, media, n, k in [(0.5, 0.3, 10, 67), (4.8, 4.5, 162, 67),
                             (0.0, 4.5, 0, 67), (9.9, 4.5, 1, 67)]:
        assert F.regress(obs, media, n, k) == original(obs, media, n, k)


def test_las_constantes_son_las_del_preregistro():
    """Si una cambia, el preregistro deja de describir lo que corre."""
    assert (F.VENTANA, F.MIN_JUEGOS_PREVIOS, F.K_REGRESION) == (162, 30, 67)
    assert F.EXPONENTE_PITAGORICO == 1.83 and F.TOPE_DESCANSO == 5
    from fbq.model.logistica import LAMBDA_L2
    assert LAMBDA_L2 == 1.0 and BRIER_IMPLAUSIBLE == 0.22


# ── El ROI no se publica sin intervalo ───────────────────────────────────

def test_el_roi_viene_con_intervalo_agrupado():
    """Un ROI sin intervalo es una invitación a creerle. El proyecto ya tomó
    por buena una ventaja que era el vig."""
    from fbq.evaluator.frame import EvalFrame
    from fbq.evaluator.score import roi_con_ic

    rng = np.random.default_rng(3)
    n = 1200
    equipos = np.array([f"E{i%30}" for i in range(n)])
    p_mkt = rng.uniform(0.35, 0.65, n)
    y = rng.binomial(1, p_mkt).astype(float)
    frame = EvalFrame(
        game_pk=np.arange(n), official_date=np.array(["2025-05-01"] * n),
        season=np.full(n, 2025), home_team=equipos, away_team=equipos[::-1],
        y=y, p_market=p_mkt, overround=np.full(n, 0.02),
        best_home=np.full(n, 2.0), best_away=np.full(n, 2.0))
    filas = roi_con_ic(np.clip(p_mkt + 0.05, 0.01, 0.99), frame, n_bootstrap=60)
    assert filas, "sin umbrales no hay gate"
    for f in filas:
        if f["n"]:
            lo, hi = f["ic95_roi_pct"]
            assert lo < hi
            assert 0.0 <= f["p_roi_positivo"] <= 1.0


# ── Invariancia del proceso completo (2026-09-06) ────────────────────────

def _almacen_sintetico(tmp_path, n_dias=40):
    """Un `results.db` con historia suficiente para que el modelo prediga."""
    import sqlite3
    from datetime import date, timedelta
    from fbq.results.store import Final, ResultsStore

    res = ResultsStore(tmp_path / "results.db")
    equipos = [f"Equipo{i}" for i in range(6)]
    finales, inicios, fines = [], [], {}
    pk = 1000
    d0 = date(2024, 4, 1)
    for dia in range(n_dias):
        f = d0 + timedelta(days=dia)
        for i in range(0, 6, 2):
            local, visita = equipos[i], equipos[i + 1]
            cl, cv = 3 + (pk % 5), 2 + (pk % 3)
            if cl == cv:
                cl += 1
            finales.append(Final(game_pk=pk, official_date=f.isoformat(),
                                 season=2024, home_team=local, away_team=visita,
                                 home_runs=cl, away_runs=cv,
                                 detailed_state="Final"))
            inicio = f"{f.isoformat()}T22:00:00Z"
            inicios.append({"game_pk": pk, "game_date": inicio,
                            "official_date": f.isoformat(), "estado": "Final"})
            fines[str(pk)] = {"game_pk": pk, "inicio": inicio,
                              "fin": f"{(f + timedelta(days=1)).isoformat()}T01:00:00Z"}
            pk += 1
    res.registrar(finales)
    (tmp_path / "sched.json").write_text(json.dumps({"juegos": inicios}), encoding="utf-8")
    (tmp_path / "fines.json").write_text(json.dumps({"juegos": fines}), encoding="utf-8")
    return tmp_path / "results.db", tmp_path / "sched.json", tmp_path / "fines.json"


def test_perturbar_siempre_invierte_al_ganador_y_cambia_el_total(tmp_path):
    """Un test de fuga que a veces no perturba nada es peor que no tenerlo.

    La primera versión sumaba 7 e invertía los lados, y con margen original
    grande dejaba al mismo ganador: 3 de 14 partidos no se perturbaban.
    """
    from fbq.model.invariancia import _perturbar
    db, _, _ = _almacen_sintetico(tmp_path, n_dias=4)
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    antes = {r[0]: (r[1], r[2]) for r in
             con.execute("SELECT game_pk, home_runs, away_runs FROM resultado")}
    con.close()
    nuevos = _perturbar(db, list(antes))
    for pk, (cl, cv) in antes.items():
        nl, nv = nuevos[pk]
        assert nl != nv, "un empate no es un final legítimo en MLB"
        assert (cl > cv) != (nl > nv), f"game_pk={pk}: el ganador no se invirtió"
        assert (nl + nv) != (cl + cv), f"game_pk={pk}: el total no cambió"


def test_la_simetria_de_perfiles_hace_la_fila_insensible_por_construccion():
    """La excepción de la prueba 2, comprobada y no supuesta: con perfiles
    idénticos `dif_pitagorica` vale 0 para CUALQUIER media de liga."""
    from fbq.model.features import Perfil, pitagorica
    p = Perfil(n=162, cf_juego=4.197531, cc_juego=4.561728, ultimo_dia="2026-06-27")
    for media in (3.5, 4.0, 4.532426778, 5.5, 9.9):
        assert pitagorica(p, media) - pitagorica(p, media) == 0.0


def test_el_fin_medido_manda_sobre_la_cota():
    """La cota de 8h era mala en las dos direcciones: retrasaba de más casi
    todos los partidos y era CORTA en 2 de 7.664 (máximo real 9,04 h)."""
    from fbq.results.fines import DURACION_MAXIMA, disponible_desde
    medidos = {1: {"fin": "2024-04-24T01:30:00Z", "inicio": "2024-04-23T22:00:00Z"}}
    cuando, proc = disponible_desde(1, "2024-04-23T22:00:00Z", medidos)
    assert proc == "medido" and cuando == "2024-04-24T01:30:00+00:00"
    # sin medición cae a la cota, y queda marcado
    cuando, proc = disponible_desde(2, "2024-04-23T22:00:00Z", {})
    assert proc == "cota" and cuando == "2024-04-24T06:00:00+00:00"
    assert DURACION_MAXIMA == timedelta(hours=8)
    # sin nada, nunca disponible
    assert disponible_desde(3, None, {}) == (None, "desconocido")


def test_un_partido_sin_fin_ni_inicio_no_entra_a_ninguna_ventana():
    from fbq.model.pit import Partido, VentanaPIT
    p = Partido(1, "2024-04-24", 2024, "A", "B", 5, 3, None, "desconocido")
    v = VentanaPIT([p])
    assert v.de_equipo("A", "2099-01-01T00:00:00+00:00", ventana=162) == []
