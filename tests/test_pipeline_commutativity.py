"""Los motores de λ son multiplicadores puros — y por eso vivo y backtest coinciden.

EL HALLAZGO (2026-08-04)
=======================
`run_module.run_module()` (producción) y `backtest_and_retrain.run_pipeline()`
(el instrumento que mide el modelo) corren los MISMOS seis motores en órdenes
DISTINTOS, y hasta numeran los pasos diferente:

    VIVO       pitcher → contextual → bullpen → park+weather → defense → hfa
    BACKTEST   park+weather → hfa → defense → pitcher → bullpen → contextual

Medido, no deducido: la λ final es idéntica en los dos órdenes hasta el último
bit (diferencia 8.88e-16, un ULP). Los seis motores devuelven un ratio que NO
depende de la λ que reciben —verificado a λ=2.5, 4.0 y 6.0— así que la cadena
es conmutativa y el orden es numéricamente irrelevante.

POR QUÉ ESTO ES UN TEST Y NO UNA ANÉCDOTA
=========================================
1. El comentario del PASO 5 en `run_module.py` afirma que el orden importa:
   "Aplicar después del pitcheo evita inflar métricas park-neutrales (xFIP,
   SIERA) antes de aplicar el ajuste del pitcher". Tal como está implementado
   eso es FALSO — y el backtest hace justo lo contrario sin consecuencia. Una
   justificación falsa es peor que ninguna: la próxima persona la respeta y no
   entiende por qué el otro camino la viola.

2. Es una trampa latente. El día que alguien meta un recorte de λ, un término
   no lineal o un piso dentro de cualquiera de los seis motores, la cadena deja
   de conmutar y **los dos caminos empiezan a calcular λ distintas en
   silencio** — con el agravante de que el backtest dejaría de medir lo que
   producción ejecuta, que es justo para lo que existe.

Este test fija la invariante. Si se rompe, no es necesariamente un error: puede
ser un motor nuevo legítimamente no lineal. Pero entonces hay que unificar el
orden de los dos caminos antes de seguir, y el fallo obliga a esa conversación.
"""
from copy import deepcopy

import pytest

from modules.baseball_module.context_engine.bullpen_engine import adjust_for_bullpen
from modules.baseball_module.context_engine.contextual_engine import adjust_for_context
from modules.baseball_module.context_engine.defensive_efficiency_engine import adjust_for_defense
from modules.baseball_module.context_engine.pitcher_engine import adjust_for_pitchers
from modules.baseball_module.hfa.hfa_engine import get_adjusted_lambdas
from modules.baseball_module.hfa.park_weather_engine import adjust_for_park_and_weather

MOTORES = {
    "pitcher":      adjust_for_pitchers,
    "contextual":   adjust_for_context,
    "bullpen":      adjust_for_bullpen,
    "park_weather": adjust_for_park_and_weather,
    "defense":      adjust_for_defense,
    "hfa":          get_adjusted_lambdas,
}

# El orden REAL de cada camino, leído del código (no de la documentación, que
# en ambos sitios numera los pasos de forma distinta).
ORDEN_VIVO = ["pitcher", "contextual", "bullpen", "park_weather", "defense", "hfa"]
ORDEN_BACKTEST = ["park_weather", "hfa", "defense", "pitcher", "bullpen", "contextual"]


def _game_data() -> dict:
    """Un juego sintético que los seis motores aceptan, sin red.

    Los valores están elegidos para que NINGÚN motor quede en identidad: un
    test que pasara porque todos devuelven 1.0 no probaría nada. El assert de
    `test_el_escenario_no_es_trivial` lo vigila.
    """
    return {
        "game_pk": 999999,
        "season": 2026,
        "venue": "Coors Field",
        "park": {"name": "Coors Field"},
        "home_team": {"name": "Colorado Rockies", "rest_days": 0},
        "away_team": {"name": "San Diego Padres", "rest_days": 0},
        "home_team_id": 115,
        "away_team_id": 135,
        "back_to_back_home": True,
        "back_to_back_away": True,
        "miles_traveled_away": 1200,
        "time_zones_crossed_away": 1,
        "travel_source_away": "test",
        "pitcher_home": {
            "name": "A", "era": 3.10, "fip": 3.20, "whip": 1.05, "k_per_9": 9.5,
            "innings_pitched": 140, "ip_mlb_equivalent": 140,
            "siera": 3.30, "xfip": 3.25, "est_woba": 0.290, "brl_percent": 6.0,
        },
        "pitcher_away": {
            "name": "B", "era": 5.40, "fip": 5.10, "whip": 1.48, "k_per_9": 6.5,
            "innings_pitched": 120, "ip_mlb_equivalent": 120,
            "siera": 5.00, "xfip": 4.95, "est_woba": 0.345, "brl_percent": 10.5,
        },
        "bullpen_home": {"era": 3.20, "siera": 3.30, "xfip": 3.25,
                         "innings_last_3d": 4.0, "n_pitchers": 8},
        "bullpen_away": {"era": 5.10, "siera": 4.90, "xfip": 4.85,
                         "innings_last_3d": 9.0, "n_pitchers": 8},
        "defense_home": {"der": 0.720, "oaa": 25},
        "defense_away": {"der": 0.680, "oaa": -20},
        "weather": {
            "temp_f": 92.0, "wind_speed_mph": 14.0, "wind_direction": 45,
            "conditions": "Clear", "rain_mm": 0.0, "precip_probability": 0,
            "postponement_risk": False,
        },
        "roof_closed": False,
    }


def _ratio(nombre: str, lam: float) -> float:
    lh, _la, _meta = MOTORES[nombre](lam, lam, deepcopy(_game_data()))
    return lh / lam


def _correr(orden: list[str], lh: float, la: float) -> tuple[float, float]:
    for n in orden:
        lh, la, _ = MOTORES[n](lh, la, deepcopy(_game_data()))
    return lh, la


# ── La invariante ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("motor", sorted(MOTORES))
def test_cada_motor_es_multiplicador_puro(motor: str):
    """El ratio no puede depender de la λ que entra.

    Si este test falla, ese motor dejó de ser conmutativo y vivo y backtest
    empiezan a calcular λ distintas. Antes de seguir hay que unificar el orden
    de las dos cadenas.
    """
    ratios = [_ratio(motor, lam) for lam in (2.5, 4.0, 6.0, 9.0)]
    dispersion = max(ratios) - min(ratios)
    assert dispersion < 1e-9, (
        f"'{motor}' devuelve un ratio que cambia con λ "
        f"(de {min(ratios):.6f} a {max(ratios):.6f}). La cadena deja de conmutar, "
        f"y vivo y backtest la aplican en órdenes distintos — ver el docstring."
    )


@pytest.mark.parametrize("lh,la", [(4.60, 4.35), (3.10, 5.90), (6.20, 2.80)])
def test_los_dos_ordenes_dan_la_misma_lambda(lh: float, la: float):
    """La prueba directa: las dos secuencias reales, mismo resultado."""
    a = _correr(ORDEN_VIVO, lh, la)
    b = _correr(ORDEN_BACKTEST, lh, la)
    assert a[0] == pytest.approx(b[0], abs=1e-12)
    assert a[1] == pytest.approx(b[1], abs=1e-12)


def test_el_escenario_no_es_trivial():
    """Sin esto, el test de arriba pasaría aunque todos devolvieran 1.0.

    Se exige que al menos cuatro de los seis muevan λ de verdad. El contextual
    puede quedar neutro del lado local por diseño (auditoría del paso 3), así
    que el umbral no es seis.
    """
    activos = [n for n in MOTORES if abs(_ratio(n, 4.0) - 1.0) > 1e-6]
    assert len(activos) >= 4, (
        f"sólo {len(activos)} motores mueven λ con este escenario ({activos}); "
        "el fixture dejó de ejercitar la cadena y los demás tests no prueban nada"
    )


def test_los_dos_ordenes_cubren_los_mismos_motores():
    """Una permutación, no dos listas distintas. Si un camino gana o pierde una
    etapa, la conmutatividad ya no alcanza para que coincidan."""
    assert sorted(ORDEN_VIVO) == sorted(ORDEN_BACKTEST) == sorted(MOTORES)
