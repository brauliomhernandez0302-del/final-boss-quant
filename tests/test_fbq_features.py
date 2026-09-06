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


# ── Veredictos versionados (2026-09-06) ──────────────────────────────────

def test_cada_feature_conserva_su_veredicto_v1_con_su_referencia_original():
    """Un veredicto sin la referencia contra la que se midió no se puede ni
    reproducir ni comparar con el siguiente. Y los veredictos no se pisan: la
    v1 se midió contra otro nulo y sigue siendo cierta sobre esa referencia."""
    from fbq.features import todas
    for f in todas():
        versiones = [v.version for v in f.veredictos]
        assert versiones == ["v1", "v2"], f"{f.nombre}: {versiones}"
        v1, v2 = f.veredictos
        assert v1.almacen == "legado" and v1.fecha == "2026-08-04"
        assert v1.referencia == "n=5.429, Brier del nulo 0,241560"
        assert v1.codigo, f"{f.nombre}: la v1 no dice de qué commit salió"
        assert v2.almacen == "propio" and v2.fecha == "2026-09-06"
        assert v2.referencia == "n=6.106, Brier del nulo 0,242124"
        # el vigente es el último, no el primero
        assert f.veredicto == v2.resultado


def test_profundidad_queda_como_no_medible_y_no_como_refutada():
    """"No cruza" y "no se pudo medir" son cosas distintas: confundirlas
    archiva como refutada una señal que nadie refutó. `n_bookmakers` no existe
    en el almacén propio y no se inventa."""
    from fbq.features import obtener
    f = obtener("profundidad_mercado")
    assert f.veredictos[0].resultado == "NO CRUZA"      # v1, sobre la señal
    assert f.veredictos[-1].resultado == "NO MEDIBLE"   # v2, sobre el dato


def test_el_porton_distingue_no_medible_de_no_cruza():
    from fbq.features.gate import Resultado
    vacio = Resultado(feature="x", n=25, pliegues=[])
    assert vacio.medible is False and vacio.cruza is False
    uno = Resultado(feature="x", n=900, pliegues=[
        {"coef": 0.1, "mejora": 0.001, "roi": [{"n": 600, "roi_pct": 1.0}]}])
    assert uno.medible is False, "un solo pliegue no permite hablar de estabilidad"


def test_las_features_ya_no_leen_el_almacen_del_sistema_anterior():
    """La migración, fijada como test: ningún módulo de `features/` abre
    `predictions_history.db`."""
    from pathlib import Path
    raiz = Path(__file__).parent.parent / "fbq" / "features"
    for py in raiz.glob("*.py"):
        codigo = "\n".join(l for l in py.read_text(encoding="utf-8").splitlines()
                           if not l.strip().startswith("#"))
        assert "predictions_history" not in codigo, py.name
