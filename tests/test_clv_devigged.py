"""El CLV se mide contra el precio JUSTO de cierre, no contra el crudo.

Hasta el 2026-07-31, `capture_closing_line` calculaba
`clv_pct = odds_tomadas / precio_crudo_de_cierre − 1`. El precio crudo de un lado
lleva el margen de la casa adentro, así que esa cuenta **regalaba el vig entero
como si fuera habilidad**.

Medido sobre los picks reales del ledger:

    ANTES (crudo)     media +1.514%   61.4% positivos
    AHORA (devigged)  media −0.497%   40.9% positivos

La diferencia es exactamente el overround de cierre de Pinnacle (~2%). El único
indicador de habilidad positivo del proyecto se vuelve negativo al medirlo bien.

Importa especialmente porque el CLV es la moneda de credibilidad de un producto
de venta de picks: es lo que un comprador informado pide, y necesita ~10x menos
muestra que el ROI para alcanzar significancia.
"""
import sys

import pytest

sys.path.insert(0, ".")
from track_record.db import _clv_devigged  # noqa: E402


# ── la aritmética ─────────────────────────────────────────────────────────────

def test_tomar_el_precio_justo_da_clv_cero():
    """Si tomaste exactamente el precio justo de cierre, no ganaste nada al
    mercado. Con el cálculo viejo esto daba +vig y parecía habilidad."""
    lado, opuesto = 2.00, 2.00          # overround 1.0 — sin vig
    assert _clv_devigged(2.00, "ML_HOME", pin_side=lado, pin_opposite=opuesto) == 0.0


def test_el_vig_ya_no_se_cuenta_como_habilidad():
    """Par 1.95/1.95 → overround 1.0256. Tomar 1.95 (el precio crudo) NO es CLV
    positivo: es exactamente el precio de mercado, con su margen."""
    crudo = (1.95 / 1.95 - 1) * 100          # lo que reportaba el cálculo viejo
    justo = _clv_devigged(1.95, "ML_HOME", pin_side=1.95, pin_opposite=1.95)
    assert crudo == 0.0
    assert justo < -2.0, f"tiene que cobrar el vig, dio {justo}"


def test_un_precio_mejor_que_el_justo_sigue_dando_positivo():
    """El arreglo no vuelve todo negativo — un CLV real sigue siendo real."""
    assert _clv_devigged(2.30, "ML_HOME", pin_side=1.95, pin_opposite=1.95) > 0


def test_la_diferencia_es_el_overround():
    """Comprobación directa de la magnitud: crudo menos devigged ≈ el margen."""
    lado, opuesto, tomado = 1.91, 2.05, 2.10
    crudo = (tomado / lado - 1) * 100
    justo = _clv_devigged(tomado, "ML_HOME", pin_side=lado, pin_opposite=opuesto)
    overround_pct = (1 / lado + 1 / opuesto - 1) * 100
    assert crudo - justo == pytest.approx(overround_pct, abs=0.15)


# ── de dónde saca los dos lados ───────────────────────────────────────────────

def test_prefiere_el_par_del_propio_mercado():
    """`pin_side`/`pin_opposite` es el par del PROPIO lado del pick y le gana a
    home/away, que obliga a deducir cuál lado corresponde."""
    con_par = _clv_devigged(2.10, "ML_HOME", pin_home=1.50, pin_away=2.60,
                            pin_side=1.91, pin_opposite=2.05)
    solo_par = _clv_devigged(2.10, "ML_HOME", pin_side=1.91, pin_opposite=2.05)
    assert con_par == solo_par


@pytest.mark.parametrize("market,esperado_lado", [("ML_HOME", 1.80), ("ML_AWAY", 2.10)])
def test_moneyline_toma_el_lado_correcto(market, esperado_lado):
    """El lado del pick, no siempre el local."""
    r = _clv_devigged(esperado_lado, market, pin_home=1.80, pin_away=2.10)
    directo = _clv_devigged(esperado_lado, market, pin_side=esperado_lado,
                            pin_opposite=2.10 if market == "ML_HOME" else 1.80)
    assert r == directo


# ── sin datos, sin número inventado ───────────────────────────────────────────

@pytest.mark.parametrize("kwargs", [
    {},                                             # nada
    {"pin_side": 1.91},                             # un solo lado
    {"pin_home": 1.80},                             # un solo lado, moneyline
    {"pin_side": 1.91, "pin_opposite": 0.5},        # precio imposible
])
def test_sin_ambos_lados_no_hay_clv(kwargs):
    """Sin los dos precios no se puede devigar. Devolver None es preferible a
    devolver un número inflado — es el error que se acaba de corregir."""
    assert _clv_devigged(2.00, "ML_HOME", **kwargs) is None


@pytest.mark.parametrize("market", ["OVER", "UNDER", "RL_HOME", "RL_AWAY",
                                    "F5_OVER", "F5_HOME"])
def test_los_derivados_siguen_sin_clv_a_proposito(market):
    """Decisión del 2026-07-26 que este arreglo NO toca: para un derivado la
    LÍNEA puede moverse (un total tomado a 8.5 que cierra en 9.0), y comparar
    precios en puntos distintos no significa nada. Qué es CLV para un derivado
    lo define el protocolo V2.

    Lo que sí está resuelto es que el dato ya no falta: `closing_pin_side` y
    `closing_pin_opposite` se capturan para todos los mercados desde b629f55,
    así que cuando V2 lo defina se calcula quitando este gate y exigiendo que
    `closing_point_moved` sea falso. Este test existe para que ampliar el
    alcance sea una decisión explícita y no un efecto colateral — que es
    exactamente lo que estuvo a punto de pasar al arreglar el devig.
    """
    assert _clv_devigged(2.00, market, pin_side=1.91, pin_opposite=2.05) is None


@pytest.mark.parametrize("odds", [None, 0, 1.0, -2.0])
def test_odds_invalidas_devuelven_none(odds):
    assert _clv_devigged(odds, "ML_HOME", pin_side=1.91, pin_opposite=2.05) is None


# ── que nadie vuelva a medir contra el crudo ──────────────────────────────────

def test_el_calculo_viejo_no_sobrevive_en_el_codigo():
    fuente = open("track_record/db.py", encoding="utf-8").read()
    assert "odds_decimal / closing_ref" not in fuente
    assert "_clv_devigged" in fuente


def test_la_evidencia_queda_documentada():
    doc = _clv_devigged.__doc__ or ""
    for pista in ("crudo", "overround", "None"):
        assert pista in doc
