"""El scorer de ROI resuelve bien el evento de cobertura de cada mercado.

Lo que se fija acá es la lógica que PURP-1 tuvo mal durante semanas: con un
precio de runline y sin el punto FIRMADO no se sabe qué evento se está
preciando. Si el local es el favorito (−1.5), cubrir es ganar por 2 o más; si
es el no-favorito (+1.5), cubrir es perder por menos de 2, o ganar. Asumir
siempre lo primero publicó 55 picks con EV falso.

El scorer usa `umbral = -punto`, así que:

    punto −1.5  →  umbral +1.5  →  el local cubre con margen > 1.5  (gana por 2+)
    punto +1.5  →  umbral −1.5  →  el local cubre con margen > −1.5 (pierde por 1, o gana)

Estos tests recorren los dos casos con marcadores concretos, para que la
inversión de signo no pueda volver a colarse.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "roi_por_mercado", RAIZ / "scripts" / "roi_por_mercado.py"
)
roi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(roi)


def _muestras(margenes: list[int], totales: list[int]):
    """Construye home/away con los márgenes y totales pedidos."""
    home = np.array([(t + m) / 2 for m, t in zip(margenes, totales)], dtype=float)
    away = np.array([(t - m) / 2 for m, t in zip(margenes, totales)], dtype=float)
    return home, away


def _p_sim(home, away) -> float:
    """Aproximación del `p_home` del simulador para los fixtures: reparte los
    empates a la mitad. En producción el simulador los reparte según el ruido
    de λ, pero acá sólo hace falta que NO se cuenten como derrota."""
    import numpy as _np
    m = _np.asarray(home) - _np.asarray(away)
    return float((_np.count_nonzero(m > 0) + 0.5 * _np.count_nonzero(m == 0)) / len(m))


def _juego(**kw):
    base = dict(
        game_pk=1, season=2025, game_date="2025-06-15", home_won=1,
        actual_home_runs=5, actual_away_runs=3,
        lh=4.6, la=4.3,
        ml_home_pin=1.90, ml_away_pin=2.00, ml_home_best=1.95, ml_away_best=2.05,
        total_point_pin=None, total_over_pin=None, total_under_pin=None,
        total_point_best=None, total_over_best=None, total_under_best=None,
        rl_home_point_pin=None, rl_home_pin=None, rl_away_pin=None,
        rl_home_point_best=None, rl_home_best=None, rl_away_best=None,
    )
    base.update(kw)
    return base


# ── Runline: el corazón de PURP-1 ──────────────────────────────────────────

def test_local_favorito_cubre_solo_ganando_por_dos():
    """punto −1.5: el local pone la carrera y media."""
    g = _juego(actual_home_runs=5, actual_away_runs=3,          # margen real +2
               rl_home_point_pin=-1.5, rl_home_pin=2.40, rl_away_pin=1.62,
               rl_home_point_best=-1.5, rl_home_best=2.45, rl_away_best=1.66)
    # márgenes simulados: +3, +2, +1, 0, −1  → cubre con margen>1.5 → 2 de 5
    home, away = _muestras([3, 2, 1, 0, -1], [9, 8, 7, 6, 5])
    apuestas = {a["lado"]: a for a in roi.apuestas_del_juego(g, home, away, _p_sim(home, away))
                if a["mercado"] == "RUNLINE"}
    assert apuestas["HOME -1.5"]["p"] == pytest.approx(0.4)
    assert apuestas["AWAY +1.5"]["p"] == pytest.approx(0.6)
    # margen real +2 > 1.5 → el local cubrió
    assert apuestas["HOME -1.5"]["gano"] is True
    assert apuestas["AWAY +1.5"]["gano"] is False


def test_local_no_favorito_cubre_perdiendo_por_uno():
    """punto +1.5: el local RECIBE la carrera y media. Éste es el caso que la
    lógica vieja calculaba como si fuera el anterior."""
    g = _juego(actual_home_runs=3, actual_away_runs=4,          # margen real −1
               rl_home_point_pin=1.5, rl_home_pin=1.64, rl_away_pin=2.40,
               rl_home_point_best=1.5, rl_home_best=1.68, rl_away_best=2.45)
    home, away = _muestras([3, 2, 1, 0, -1], [9, 8, 7, 6, 5])
    apuestas = {a["lado"]: a for a in roi.apuestas_del_juego(g, home, away, _p_sim(home, away))
                if a["mercado"] == "RUNLINE"}
    # cubre con margen > −1.5 → los cinco menos ninguno = 5 de 5
    assert apuestas["HOME +1.5"]["p"] == pytest.approx(1.0)
    # margen real −1 > −1.5 → el local CUBRIÓ aunque perdió el juego
    assert apuestas["HOME +1.5"]["gano"] is True
    assert apuestas["AWAY -1.5"]["gano"] is False


def test_los_dos_puntos_dan_probabilidades_distintas():
    """La prueba de que el signo importa: mismo juego, mismas simulaciones,
    distinto punto → distinta probabilidad. Si alguien vuelve a ignorar el
    punto, estas dos coincidirían."""
    home, away = _muestras([3, 2, 1, 0, -1], [9, 8, 7, 6, 5])
    fav = roi.apuestas_del_juego(
        _juego(rl_home_point_pin=-1.5, rl_home_pin=2.4, rl_away_pin=1.6,
               rl_home_point_best=-1.5, rl_home_best=2.4, rl_away_best=1.6),
        home, away, _p_sim(home, away))
    nofav = roi.apuestas_del_juego(
        _juego(rl_home_point_pin=1.5, rl_home_pin=1.6, rl_away_pin=2.4,
               rl_home_point_best=1.5, rl_home_best=1.6, rl_away_best=2.4),
        home, away, _p_sim(home, away))
    p_fav = next(a["p"] for a in fav if a["mercado"] == "RUNLINE" and "HOME" in a["lado"])
    p_nofav = next(a["p"] for a in nofav if a["mercado"] == "RUNLINE" and "HOME" in a["lado"])
    assert p_fav != p_nofav


def test_sin_punto_firmado_no_se_evalua_runline():
    """Adivinar el punto es lo que causó PURP-1. Sin él, el mercado se salta."""
    g = _juego(rl_home_pin=2.4, rl_away_pin=1.6, rl_home_best=2.4, rl_away_best=1.6)
    assert not [a for a in roi.apuestas_del_juego(g, *_muestras([1], [7]), 0.5)
                if a["mercado"] == "RUNLINE"]


def test_punto_distinto_entre_pinnacle_y_mejor_precio_se_descarta():
    """Un fair calculado en 1.5 contra un precio pagado en 2.5 compara dos
    mercados distintos."""
    g = _juego(rl_home_point_pin=-1.5, rl_home_pin=2.4, rl_away_pin=1.6,
               rl_home_point_best=-2.5, rl_home_best=3.1, rl_away_best=1.4)
    assert not [a for a in roi.apuestas_del_juego(g, *_muestras([1], [7]), 0.5)
                if a["mercado"] == "RUNLINE"]


# ── Total ──────────────────────────────────────────────────────────────────

def test_total_resuelve_over_y_under():
    g = _juego(actual_home_runs=6, actual_away_runs=4,           # total real 10
               total_point_pin=8.5, total_over_pin=1.90, total_under_pin=1.98,
               total_point_best=8.5, total_over_best=1.95, total_under_best=2.02)
    home, away = _muestras([1, 1, 1, 1], [7, 8, 9, 10])          # >8.5 → 2 de 4
    ap = {a["lado"]: a for a in roi.apuestas_del_juego(g, home, away, _p_sim(home, away))
          if a["mercado"] == "TOTAL"}
    assert ap["OVER"]["p"] == pytest.approx(0.5)
    assert ap["OVER"]["gano"] is True and ap["UNDER"]["gano"] is False


def test_push_de_total_se_excluye():
    """Total real exactamente en la línea: no gana ni pierde, y contarlo como
    derrota falsearía el ROI hacia abajo."""
    g = _juego(actual_home_runs=5, actual_away_runs=4,           # total 9.0
               total_point_pin=9.0, total_over_pin=1.90, total_under_pin=1.98,
               total_point_best=9.0, total_over_best=1.95, total_under_best=2.02)
    assert not [a for a in roi.apuestas_del_juego(g, *_muestras([1], [9]), 0.5)
                if a["mercado"] == "TOTAL"]


# ── El fair ────────────────────────────────────────────────────────────────

def test_el_fair_sale_de_pinnacle_y_el_pago_del_mejor_precio():
    """Confundirlos inventa ventaja: el mejor par tiene overround artificialmente
    bajo porque mezcla casas, así que usarlo como referencia justa infla el edge."""
    g = _juego(ml_home_pin=1.90, ml_away_pin=2.00, ml_home_best=1.99, ml_away_best=2.10)
    ml = {a["lado"]: a for a in roi.apuestas_del_juego(g, *_muestras([1], [7]), 0.5)
          if a["mercado"] == "MONEYLINE"}
    esperado_h, esperado_a = roi._devig(1.90, 2.00)
    assert ml["HOME"]["fair"] == pytest.approx(esperado_h)
    assert ml["HOME"]["cuota"] == 1.99          # se paga al mejor
    assert ml["AWAY"]["fair"] == pytest.approx(esperado_a)
    assert sum(a["fair"] for a in ml.values()) == pytest.approx(1.0)
