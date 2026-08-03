"""Paso 10 — el Kalman de ofensa ya no corrige λ, y por qué.

`get_kalman_lambda_adjustment` mezclaba `0.65·λ_modelo + 0.35·λ_kalman`, donde el
Kalman observa CARRERAS REALES ANOTADAS. La λ que corregía viene del TTE y es una
estimación MERECIDA (xwOBA/barrel/disciplina) que por construcción filtra la
suerte en pelotas en juego — así que la corrección le devolvía justo eso.

Medido por dos caminos (detalle en `audit_20260714/paso10/reporte.md`):

  Datos vivos, 236 obs, resto de temporada, recursión de Kalman real:
      w=0.00 r=+0.5303 | 0.15 +0.5267 | 0.35 +0.4713 | 1.00 +0.3455

  Backtest completo, tres corridas de 4.825 juegos:
      w=0.35  Brier 0.24675  vs azar 1.300%  accuracy 55.05%
      w=0.15  Brier 0.24642  vs azar 1.430%  accuracy 54.86%
      w=0.00  Brier 0.24624  vs azar 1.510%  accuracy 54.84%

Decisión del dueño con esa evidencia. La salvedad —la ganancia de Brier está
concentrada en 2024— queda escrita en la constante y no se borra.
"""
import sys

import pytest

# `core` existe en la raíz Y bajo modules/baseball_module. Importar el de la
# raíz PRIMERO evita que el otro lo tape en sys.modules.
from core.value_detector import compute_data_quality_confidence  # noqa: E402

sys.path.insert(0, "modules/baseball_module")
from calibration.learning_engine import _KALMAN_BLEND  # noqa: E402


def test_el_kalman_ya_no_corrige_lambda():
    assert _KALMAN_BLEND == 0.0


def test_la_evidencia_y_la_salvedad_quedan_en_el_sitio():
    """La constante tiene que llevar encima por qué se bajó Y qué la pondría en
    duda. Sin la salvedad, un tercer año que no reproduzca la ganancia se leería
    como un problema nuevo en vez de como el riesgo que ya estaba anotado."""
    fuente = open("modules/baseball_module/calibration/learning_engine.py",
                  encoding="utf-8").read()
    bloque = fuente[:fuente.index("_KALMAN_BLEND = 0.0")]
    bloque = bloque[bloque.rindex("# Fracción de la λ"):]
    for pista in ("0.24624", "0.24675", "concentrada en 2024", "neutra de parque"):
        assert pista.lower() in bloque.lower(), f"la constante perdió la referencia a {pista!r}"


def test_el_amortiguamiento_del_sesgo_degenera_limpio():
    """`_KALMAN_BLEND` gobierna además el amortiguamiento del sesgo de equipo:
    `dampened = raw_bias / ((1-B) + B·raw_bias)`. Con B=0 el denominador es 1.0 y
    devuelve el sesgo crudo — lo correcto, porque sin Kalman no hay solapamiento
    que remover. Es la razón por la que bajar el blend NO toca la parte de esa
    fórmula que este proyecto ya intentó arreglar dos veces y revirtió.
    """
    for raw in (0.85, 1.0, 1.15, 1.30):
        denom = (1.0 - _KALMAN_BLEND) + _KALMAN_BLEND * raw
        assert denom == 1.0
        assert raw / denom == raw


def test_la_confianza_sigue_usando_el_conteo_del_kalman():
    """El conteo alimenta el 25% de la confianza y NO se tocó: sigue midiendo
    volumen de datos de temporada en curso, que es real y pertinente aunque el
    Kalman ya no corrija λ. Lo que se corrigió es la justificación escrita."""
    conf = compute_data_quality_confidence
    base = dict(home_sp_ip=200, away_sp_ip=200, home_prior_weight=0.2, away_prior_weight=0.2)
    poco = conf(dict(base, home_kalman_n_obs=1, away_kalman_n_obs=1))
    mucho = conf(dict(base, home_kalman_n_obs=50, away_kalman_n_obs=50))
    assert mucho > poco, "más juegos observados ⇒ más confianza"
    doc = conf.__doc__ or ""
    assert "2026-07-28" in doc, "la justificación vieja tiene que estar corregida"


def test_el_estado_del_kalman_se_sigue_manteniendo():
    """`update_kalman` no se desactivó: es barato, deja el camino abierto para
    revertir, y su n_obs es lo que alimenta la confianza."""
    import inspect
    from calibration.learning_engine import LearningEngine
    src = inspect.getsource(LearningEngine.update_kalman)
    assert "_save_kalman_state" in src
