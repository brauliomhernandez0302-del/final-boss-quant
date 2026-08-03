"""Paso 8 — la "forma reciente" del abridor no predice nada, y predecía al revés.

`_adjust_pitcher_form` pesaba 0.256 (2º factor del motor) y multiplicaba tres
señales sacadas de la MISMA lista de ~5 arranques: nivel de ERA reciente,
pendiente de la ERA, y % de quality starts. Contar un dato tres veces y
multiplicarlo entre sí.

Medido sobre 1709 pares (arranque previo → siguiente) de 148 abridores reales,
con la ERA acumulada como CONTROL y errores agrupados por pitcher:

    ERA acumulada (control)   +0.4080   t = +2.25   ← lo único con señal
    ERA últimos 5             -0.0062   t = -0.04
    tendencia                 +0.0042   t = +0.06
    quality start %           +0.4601   t = +0.74

Y sin control, el factor completo daba r = -0.0226 contra el residual: medía AL
REVÉS. Las señales captan regresión a la media; el motor las leía como
persistencia.

Estos tests fijan la neutralización y, sobre todo, las CONDICIONES que tendría
que cumplir cualquier intento futuro de reinstaurar un ajuste de forma.
"""
import sys

import pytest

sys.path.insert(0, "modules/baseball_module")
from context_engine.pitcher_engine import PitcherEngine  # noqa: E402


@pytest.fixture
def motor():
    return PitcherEngine()


# ── la neutralización ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("pitcher", [
    {},
    {"era": 4.15, "era_last_5": 1.20, "era_trend": -2.0, "quality_start_pct": 1.00},
    {"era": 4.15, "era_last_5": 9.00, "era_trend": +2.0, "quality_start_pct": 0.00},
    {"era": None, "era_last_5": None, "era_trend": None, "quality_start_pct": None},
])
def test_la_forma_es_neutra_pase_lo_que_pase(motor, pitcher):
    """Ni una racha perfecta ni una desastrosa mueven λ: no hay señal que mover."""
    assert motor._adjust_pitcher_form(pitcher) == 1.0


def test_un_factor_neutro_no_aporta_aunque_pese_lo_que_pese(motor):
    """La combinación es aditiva sobre deltas: `total = 1 + Σ wᵢ·(fᵢ-1)`.

    Es la razón por la que NO se redistribuye el peso — un factor en 1.0 aporta
    exactamente cero, así que no queda ningún hueco. Mover ese peso a los otros
    factores los amplificaría, que es un cambio distinto y sin evidencia.
    """
    base = {"era": 4.15, "whip": 1.30, "innings_pitched": 150.0}
    con_racha = dict(base, era_last_5=1.00, quality_start_pct=1.0, era_trend=-3.0)
    sin_racha = dict(base, era_last_5=9.00, quality_start_pct=0.0, era_trend=+3.0)
    a = motor._calculate_pitcher_adjustment(con_racha, {}, is_home=True)
    b = motor._calculate_pitcher_adjustment(sin_racha, {}, is_home=True)
    assert a["total_multiplier"] == b["total_multiplier"]


def test_el_peso_sigue_declarado_y_no_se_movio():
    """Si alguien redistribuye ese peso tiene que ser una decisión explícita, no
    un efecto colateral de este paso."""
    from config import PITCHER_ENGINE_WEIGHTS as W
    assert W["pitcher_form"] == 0.256
    assert sum(W.values()) == pytest.approx(1.0)


# ── las condiciones para reinstaurar algo acá ─────────────────────────────────

def test_la_evidencia_queda_en_el_sitio(motor):
    """La función se conserva (en vez de sacar el término de la suma) justamente
    para que quien vaya a reinstaurar un ajuste de forma encuentre primero por
    qué se quitó, con los números y con los dos falsos positivos que dieron las
    especificaciones intermedias."""
    doc = motor._adjust_pitcher_form.__doc__ or ""
    for pista in ("t = +0.74", "regresión a la media", "agrupados", "residual"):
        assert pista in doc, f"la evidencia perdió la referencia a {pista!r}"


def test_las_tres_señales_venian_de_la_misma_fuente():
    """El defecto estructural, además del de signo: `era_trend` y
    `quality_start_pct` se derivan de la misma lista `_metrics` de la que sale
    `era_last_5`, y las tres se multiplicaban entre sí."""
    fuente = open("data_fetchers.py", encoding="utf-8").read()
    desde = fuente.index("def get_pitcher_game_log")
    hasta = fuente.index("def get_pitcher_f5_stats")
    bloque = fuente[desde:hasta]
    # Las tres se construyen dentro de la MISMA función, de la misma lista.
    assert '_metrics' in bloque
    for señal in ('"era_trend"', '"quality_start_pct"', '"era_last_5"'):
        assert señal in bloque, f"{señal} ya no sale de get_pitcher_game_log"
