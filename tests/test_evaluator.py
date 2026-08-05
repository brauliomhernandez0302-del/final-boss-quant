"""El evaluador: que mida lo que dice medir, y que no favorezca a nadie.

La autoprueba del instrumento es el primer test: puntuar el propio mercado como
candidato tiene que reproducir la barra EXACTAMENTE. Si eso no se cumple, todo
número que salga de acá es sospechoso antes de mirarlo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbq.evaluator.frame import EvalFrame, _devig_multiplicativo
from fbq.evaluator.score import (brier, calibracion, evaluate, log_loss,
                             roi_por_umbral, ventaja_sobre_azar)


def _frame(n=400, seed=3) -> EvalFrame:
    """Marco sintético con un mercado bien calibrado."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.30, 0.70, n)
    y = (rng.uniform(size=n) < p).astype(float)
    # Precios de un libro con 2% de vig sobre la probabilidad verdadera
    over = 1.02
    ph, pa = 1 / (p * over), 1 / ((1 - p) * over)
    p_market, overround = _devig_multiplicativo(ph, pa)
    equipos = np.array([f"T{i%30:02d}" for i in range(n)])
    return EvalFrame(
        game_pk=np.arange(n), official_date=np.array(["2026-01-01"] * n),
        season=np.full(n, 2026), home_team=equipos, away_team=equipos[::-1],
        y=y, p_market=p_market, overround=overround,
        best_home=ph * 1.01, best_away=pa * 1.01,
    )


# ── La autoprueba ────────────────────────────────────────────────────────


def test_puntuar_el_mercado_reproduce_la_barra_exactamente():
    f = _frame()
    rep = evaluate(f, f.p_market.copy(), nombre="mercado", n_bootstrap=30)
    assert rep.brier_candidato == pytest.approx(rep.brier_mercado)
    assert rep.brecha == pytest.approx(0.0, abs=1e-12)
    assert rep.logloss_candidato == pytest.approx(rep.logloss_mercado)


def test_el_mercado_como_candidato_no_genera_ninguna_apuesta():
    """Edge cero en todos los juegos: cero apuestas en todos los umbrales."""
    f = _frame()
    rep = evaluate(f, f.p_market.copy(), n_bootstrap=30)
    assert all(r["n"] == 0 for r in rep.roi)


# ── Que no favorezca a nadie ─────────────────────────────────────────────


def test_un_candidato_de_ruido_puro_no_aporta_sobre_el_precio():
    f = _frame()
    rng = np.random.default_rng(11)
    rep = evaluate(f, rng.uniform(0.3, 0.7, len(f)), n_bootstrap=200)
    assert rep.brecha > 0                      # peor que el mercado
    assert not rep.aporta_sobre_el_precio      # y el IC incluye el cero


def test_un_candidato_que_ve_el_resultado_si_aporta():
    """Control positivo: si el instrumento no detecta un candidato que hace
    trampa, no detectaría tampoco una señal real."""
    f = _frame()
    tramposo = np.clip(f.p_market + 0.25 * (f.y - 0.5), 0.02, 0.98)
    rep = evaluate(f, tramposo, n_bootstrap=200)
    assert rep.brecha < 0
    assert rep.aporta_sobre_el_precio


# ── Los juegos sin candidato se excluyen, no se rellenan ─────────────────


def test_un_juego_sin_prediccion_se_excluye_y_no_se_inventa_media():
    """Rellenar con 0.5 parecería una predicción y no lo es."""
    f = _frame(n=100)
    parcial = {int(pk): 0.55 for pk in f.game_pk[:60]}
    rep = evaluate(f, parcial, n_bootstrap=20)
    assert rep.n == 60


def test_sin_cobertura_falla_ruidoso():
    f = _frame(n=50)
    with pytest.raises(ValueError):
        evaluate(f, {999999: 0.5}, n_bootstrap=10)


# ── Las métricas ─────────────────────────────────────────────────────────


def test_brier_y_ventaja_sobre_azar_son_coherentes():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier(np.full(4, 0.5), y) == pytest.approx(0.25)
    assert ventaja_sobre_azar(np.full(4, 0.5), y) == pytest.approx(0.0)


def test_el_devig_deja_las_probabilidades_sumando_uno():
    ph, pa = np.array([1.80]), np.array([2.10])
    p, over = _devig_multiplicativo(ph, pa)
    p_op, _ = _devig_multiplicativo(pa, ph)
    assert float(p[0] + p_op[0]) == pytest.approx(1.0)
    assert float(over[0]) == pytest.approx(1 / 1.80 + 1 / 2.10 - 1)


def test_el_roi_apuesta_al_mejor_precio_no_al_de_pinnacle():
    """La línea justa se calcula contra el libro sharp; se apuesta al mejor
    precio disponible. Son dos preguntas distintas."""
    f = _frame(n=50)
    p = np.clip(f.p_market + 0.20, 0, 1)          # edge grande y uniforme
    barato = roi_por_umbral(p, f)
    f_mejor = f.subset(np.ones(len(f), bool))
    f_mejor.best_home = f.best_home * 1.5
    caro = roi_por_umbral(p, f_mejor)
    assert caro[0]["pnl_u"] > barato[0]["pnl_u"]


def test_la_calibracion_parte_en_grupos_de_igual_tamano():
    f = _frame(n=100)
    grupos = calibracion(f.p_market, f.y, 10)
    assert len(grupos) == 10
    assert sum(g["n"] for g in grupos) == 100
    predichos = [g["predicho"] for g in grupos]
    assert predichos == sorted(predichos)


# ── Paso 4: v0 = el mercado, y la mezcla fuera de muestra ────────────────


def test_la_mezcla_no_ayuda_cuando_el_candidato_es_ruido():
    """Fuera de muestra, un candidato sin señal no mejora a v0.

    En muestra siempre encontraría un peso que no daña — por eso la pregunta
    del paso 4 sólo tiene sentido fuera de muestra.
    """
    from fbq.evaluator.score import mezcla_fuera_de_muestra

    f = _frame(n=1200, seed=5)
    f.season = np.where(np.arange(len(f)) < 600, 2024, 2025)
    ruido = np.random.default_rng(2).uniform(0.3, 0.7, len(f))
    pliegues = mezcla_fuera_de_muestra(f, ruido)
    assert len(pliegues) == 2
    assert all(r["brier_mezcla"] >= r["brier_v0"] - 1e-4 for r in pliegues)


def test_la_mezcla_si_ayuda_con_un_candidato_que_aporta():
    """Control positivo del paso 4: si una señal real no aparece acá, el
    instrumento no serviría para aceptar ninguna."""
    from fbq.evaluator.score import mezcla_fuera_de_muestra

    f = _frame(n=1200, seed=6)
    f.season = np.where(np.arange(len(f)) < 600, 2024, 2025)
    util = np.clip(f.p_market + 0.20 * (f.y - 0.5), 0.02, 0.98)
    pliegues = mezcla_fuera_de_muestra(f, util)
    assert all(r["brier_mezcla"] < r["brier_v0"] for r in pliegues)
    assert all(r["b_candidato_ajustado"] > 0 for r in pliegues)


def test_los_pliegues_son_temporadas_enteras_no_filas_al_azar():
    """Un corte aleatorio pondría juegos del mismo día a ambos lados y el
    ajuste aprendería del futuro por la puerta de al lado."""
    from fbq.evaluator.score import mezcla_fuera_de_muestra

    f = _frame(n=800)
    f.season = np.where(np.arange(len(f)) < 400, 2024, 2025)
    pliegues = mezcla_fuera_de_muestra(f, f.p_market.copy())
    assert [r["pliegue"] for r in pliegues] == [2024, 2025]
    assert all(r["n_test"] == 400 for r in pliegues)
