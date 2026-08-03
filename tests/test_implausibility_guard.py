"""Un edge imposible contra el precio desvigorizado se bloquea, no se publica.

POR QUÉ EXISTE. Hasta el 2026-07-31 `analyze_runline` calculaba siempre
`p_home_cover = P(diff > 1.5)` — "el local gana por 2 o más" — sin mirar quién
era el favorito. Cuando el local era el NO-favorito, el mercado traía los
precios de HOME +1.5 y AWAY −1.5, pero el motor los apareaba con las
probabilidades de HOME −1.5 y AWAY +1.5: la probabilidad del evento FÁCIL con
el precio del evento DIFÍCIL.

Medido sobre los 134 picks de runline publicados antes del arreglo:

    con EV inflado                        74  (55 %)
    FALSOS (EV publicado > 0, real ≤ 0)   55  (41 %)
    EV medio publicado                +34.47 %
    EV medio real                      −2.27 %
    probabilidad media publicada        0.6419
    probabilidad media correcta         0.4871

Caso concreto, reproducido abajo: RL_AWAY a cuota 2.60 publicado con p=0.650 y
EV +68.9 %, cuando la probabilidad real era 0.271 y el EV −29.6 %.

El arreglo de raíz (pasar el punto FIRMADO) ya está. Esto es el detector de
humo para la CLASE de fallo, no para este bug: cualquier futuro emparejamiento
de una probabilidad con el precio de otro mercado produce la misma firma —un
desacuerdo enorme contra un precio ya desvigorizado— y se bloquea aunque nadie
haya previsto ese camino concreto.

Honestidad sobre su alcance: a 20pp habría atrapado el 62 % de aquellos falsos,
no todos. Es un backstop.
"""
import logging

import pytest

import config as _cfg
from core.value_detector import CONFIG, ValueTier, analyze_market_generic


def _analizar(model_prob, odds, true_implied, confidence=0.93):
    return analyze_market_generic(
        model_prob=model_prob,
        odds=odds,
        prob_ci=(model_prob - 0.01, model_prob + 0.01),
        overround=4.5,
        true_implied=true_implied,
        fractional_kelly=0.25,
        market_name="RUNLINE AWAY -1.5",
        confidence=confidence,
    )


def test_el_caso_real_de_purp1_queda_bloqueado():
    """p=0.650 contra un mercado que la precia en 0.385: 26.5pp de desacuerdo.

    Son las cifras de uno de los 55 picks falsos. Antes salía publicado con
    EV +68.9 %; ahora tiene que quedar bloqueado.
    """
    r = _analizar(model_prob=0.650, odds=2.60, true_implied=0.385)
    assert r["edge"] > CONFIG.IMPLAUSIBLE_EDGE_PP
    assert r["implausible"] is True
    assert r["tier_enum"] is ValueTier.NEGATIVE, "un edge imposible no puede publicarse"
    assert r["ev"] > 0, "el EV crudo sigue siendo positivo — por eso hacía falta el guardarraíl"


def test_un_pick_sano_no_se_toca():
    """Las cifras del pick real de Padres/Diamondbacks: edge 5.5pp, EV +10.5 %."""
    r = _analizar(model_prob=0.5471, odds=2.02, true_implied=0.4925)
    assert r["implausible"] is False
    assert r["edge_suspicious"] is False
    assert r["tier_enum"] is not ValueTier.NEGATIVE
    assert r["ev"] > 0


def test_la_zona_de_aviso_marca_pero_deja_pasar():
    """Entre 12pp y 20pp se anota y se loguea, pero no se bloquea: ahí todavía
    puede haber ventaja real, y bloquearla costaría picks legítimos."""
    r = _analizar(model_prob=0.560, odds=2.10, true_implied=0.420)
    assert CONFIG.SUSPICIOUS_EDGE_PP <= r["edge"] < CONFIG.IMPLAUSIBLE_EDGE_PP
    assert r["edge_suspicious"] is True
    assert r["implausible"] is False
    assert r["tier_enum"] is not ValueTier.NEGATIVE


def test_el_bloqueo_deja_rastro_en_los_logs(caplog):
    """Bloquear en silencio esconde el bug que lo causó. Tiene que gritar."""
    with caplog.at_level(logging.ERROR, logger="core.value_detector"):
        _analizar(model_prob=0.650, odds=2.60, true_implied=0.385)
    assert any("IMPLAUSIBLE" in rec.message for rec in caplog.records)


def test_un_edge_negativo_enorme_no_se_marca():
    """El guardarraíl es asimétrico a propósito: un desacuerdo grande EN CONTRA
    no llega a apostarse nunca (EV negativo), así que marcarlo sólo añadiría
    ruido a los logs."""
    r = _analizar(model_prob=0.271, odds=2.60, true_implied=0.650)
    assert r["implausible"] is False
    assert r["edge_suspicious"] is False


def test_los_umbrales_salen_de_config():
    """Que se puedan mover sin tocar el motor, y que el orden tenga sentido."""
    assert CONFIG.SUSPICIOUS_EDGE_PP < CONFIG.IMPLAUSIBLE_EDGE_PP
    assert CONFIG.IMPLAUSIBLE_EDGE_PP == _cfg.IMPLAUSIBLE_EDGE_PP
    assert CONFIG.MIN_EDGE < CONFIG.SUSPICIOUS_EDGE_PP


def test_las_banderas_estan_siempre_presentes():
    """Se exponen aunque no salten, para poder contar los que rozan el umbral."""
    r = _analizar(model_prob=0.5471, odds=2.02, true_implied=0.4925)
    assert "implausible" in r and "edge_suspicious" in r
