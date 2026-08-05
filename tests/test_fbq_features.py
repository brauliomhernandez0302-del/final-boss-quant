"""fbq.features — el portón (pasos 5-6).

El portón es lo que decide qué entra al sistema, así que sus tres criterios
tienen que fallar por las razones correctas. Cada test acá corresponde a una
forma en que una versión anterior del proyecto se equivocó midiendo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbq.evaluator.frame import EvalFrame, _devig_multiplicativo
from fbq.features.base import Feature, obtener, registrar, todas
from fbq.features.gate import evaluar


def _frame(n=6000, seed=4) -> EvalFrame:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.30, 0.70, n)
    y = (rng.uniform(size=n) < p).astype(float)
    ph, pa = 1 / (p * 1.02), 1 / ((1 - p) * 1.02)
    pm, over = _devig_multiplicativo(ph, pa)
    return EvalFrame(
        game_pk=np.arange(n), official_date=np.array(["2026-01-01"] * n),
        season=np.where(np.arange(n) < n // 2, 2024, 2025),
        home_team=np.array([f"T{i%30:02d}" for i in range(n)]),
        away_team=np.array([f"T{i%30:02d}" for i in range(n)]),
        y=y, p_market=pm, overround=over,
        # El mejor precio es 1% mejor que el de Pinnacle pero SIGUE llevando
        # vig: con `* 1.02` se cancelaría exactamente el overround y quedaría
        # el precio justo, con el que apostar al azar da EV cero y el criterio
        # de ROI dejaría pasar ruido. Un mercado sin vig no existe.
        best_home=ph * 1.01, best_away=pa * 1.01,
    )


def _feature(nombre, fn):
    return Feature(nombre=nombre, descripcion="", calcular=fn)


def test_una_senal_de_ruido_no_cruza():
    f = _frame()
    rng = np.random.default_rng(9)
    ruido = {int(pk): float(v) for pk, v in zip(f.game_pk, rng.normal(size=len(f)))}
    r = evaluar(_feature("ruido", lambda pks: ruido), frame=f)
    assert not r.cruza


def test_una_senal_que_ve_el_resultado_si_cruza():
    """Control positivo. Si el portón no deja pasar una señal plantada, no
    serviría para aceptar ninguna real."""
    f = _frame()
    tramposa = {int(pk): float(v) for pk, v in zip(f.game_pk, f.y - 0.5)}
    r = evaluar(_feature("tramposa", lambda pks: tramposa), frame=f)
    assert r.signo_estable and r.mejora_brier
    assert r.cruza


def test_el_signo_inestable_reprueba_aunque_el_brier_mejore():
    """El único motor que salía significativo EN MUESTRA en el sistema
    anterior (`defense`, P(b>0)=99.0%) resultó el peor al probarlo fuera, con
    el coeficiente saltando de +0.1192 a +0.0218."""
    f = _frame()
    # Señal que apunta al resultado en 2024 y en contra en 2025.
    signo = np.where(f.season == 2024, 1.0, -1.0)
    s = {int(pk): float(v) for pk, v in zip(f.game_pk, signo * (f.y - 0.5))}
    r = evaluar(_feature("voltea", lambda pks: s), frame=f)
    assert not r.signo_estable
    assert not r.cruza


def test_un_juego_sin_valor_se_excluye_y_no_se_rellena():
    """Rellenar con la media parecería una observación y no lo es."""
    f = _frame()
    parcial = {int(pk): 0.5 for pk in f.game_pk[:3200]}
    r = evaluar(_feature("parcial", lambda pks: parcial), frame=f)
    assert r.n == 3200


def test_las_features_del_sistema_estan_registradas_con_veredicto():
    """Una feature medida y descartada se queda con su veredicto: borrarla
    garantizaría que alguien la reintente sin saber que ya se midió."""
    nombres = {f.nombre for f in todas()}
    assert {"desacuerdo_cons_pin", "prima_mejor_precio",
            "profundidad_mercado"} <= nombres
    for n in nombres:
        f = obtener(n)
        assert f.veredicto, f"{n} sin veredicto registrado"
        assert f.hipotesis, f"{n} sin hipótesis escrita antes de medir"


def test_no_se_puede_registrar_dos_veces_el_mismo_nombre():
    with pytest.raises(ValueError):
        registrar(_feature("desacuerdo_cons_pin", lambda pks: {}))
