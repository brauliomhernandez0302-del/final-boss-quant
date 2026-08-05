"""Contrato de la costura odds → GameOdds: ningún campo se pierde por el camino.

POR QUÉ EXISTE
==============
Los precios llegan al detector de valor por DOS caminos independientes:

    A) cron / pipeline   get_best_odds_for_teams()  ──────────────┐
                                                                  ├─→ GameOdds
    B) selector de la UI  _normalize_event() → build_game_selector()
                          → GameData → build_market_odds()  ──────┘

Esa bifurcación se rompió dos veces, siempre igual: alguien añade un campo,
lo enchufa en el camino A, y el B se queda sin él durante semanas.

    2026-07-06  las cinco claves de F5 no existían en el camino B, así que
                el análisis lanzado desde el selector nunca podía sacar
                mercados de F5 por más que el fetcher los trajera.
    2026-08-02  `runline_home_point` se perdía en DOS sitios del camino B:
                `build_game_selector()` no lo copiaba, y una vez copiado
                `_safe_float()` lo anulaba por descartar todo `<= 0` —
                justo cuando el local FAVORITO cotiza −1.5, el caso
                mayoritario. Sin ese campo `analyze_runline` asume local
                favorito: es PURP-1, y publicó 55 picks con EV falso.

Las dos se descubrieron por sus consecuencias, nunca por un test. Los tests
que existen cubren un campo cada uno y se escribieron DESPUÉS del bug
correspondiente. Este es el genérico: recorre `GameOdds` entero y exige que
los dos caminos lo cubran, así que el próximo campo que falte falla en CI
antes de publicar nada.

QUÉ **NO** GARANTIZA
====================
Que el valor sea correcto — sólo que llega. Un campo bien transportado con el
signo invertido pasaría estos tests; para eso están
`test_runline_signed_point.py` y `test_odds_fetcher_runline_sign.py`.
"""
from dataclasses import fields as dataclass_fields

import pandas as pd
import pytest

from core.value_detector import GameOdds
from ui.mlb import MARKET_ODDS_FIELD_MAP, build_market_odds
from ui.odds_loader import build_game_selector


# Campos de GameOdds que el selector de la UI no puede poblar por diseño, con
# la razón. Cualquier otro campo ausente es un fallo, no una excepción: la
# lista obliga a justificar cada hueco en vez de dejarlo pasar en silencio.
SIN_ORIGEN_EN_LA_UI: dict[str, str] = {}


def _fila_cruda() -> dict:
    """Una fila como la emite `_normalize_event()`, con los tres mercados.

    Valores realistas de un juego con el LOCAL DE NO-FAVORITO — el caso que
    rompía PURP-1 y el que un helper de precios anula por ser negativo.
    """
    return {
        "home_team": "Arizona Diamondbacks",
        "away_team": "San Diego Padres",
        "commence_time": "2026-08-03T01:40:00Z",
        "home_odds": 2.02, "away_odds": 1.96,
        "pin_home": 1.98, "pin_away": 1.93,
        "total_line": 8.5, "over_odds": 1.94, "under_odds": 1.98,
        # Par propio de Pinnacle de cada derivado: es contra lo que se
        # desvigoriza para la línea justa. El punto del runline va NEGATIVO
        # a propósito, que es el caso que un helper de precios anula.
        "pin_total_over": 1.91, "pin_total_under": 1.95, "pin_total_point": 8.5,
        "pin_runline_home": 1.62, "pin_runline_away": 2.46,
        "pin_runline_home_point": -1.5,
        "runline_home": 1.64, "runline_away": 2.51,
        "runline_line": 1.5, "runline_home_point": 1.5,
        "f5_home_odds": 1.90, "f5_away_odds": 1.95,
        "f5_total_line": 4.5, "f5_over_odds": 1.88, "f5_under_odds": 1.98,
    }


def _game_data_desde_fila(fila: dict):
    opciones, mapping = build_game_selector(pd.DataFrame([fila]))
    return mapping[opciones[0]]


def _campos_de_gameodds() -> list[str]:
    return [f.name for f in dataclass_fields(GameOdds)]


# ── El contrato ────────────────────────────────────────────────────────────

def test_el_mapa_cubre_todo_gameodds():
    """Cada campo de GameOdds tiene un origen en el camino de la UI.

    Éste es el test que habría atajado los dos bugs históricos EN EL MOMENTO
    DE AÑADIR EL CAMPO, sin necesidad de que a nadie se le ocurriera escribir
    un test específico para él.
    """
    faltan = [
        c for c in _campos_de_gameodds()
        if c not in MARKET_ODDS_FIELD_MAP and c not in SIN_ORIGEN_EN_LA_UI
    ]
    assert not faltan, (
        f"GameOdds declara {faltan} y el camino del selector no los puebla. "
        "Añádelos a MARKET_ODDS_FIELD_MAP en ui/mlb.py, o a SIN_ORIGEN_EN_LA_UI "
        "en este test con la razón por la que la UI no puede tenerlos."
    )


def test_el_mapa_no_inventa_campos():
    """Al revés: nada en el mapa que GameOdds no acepte (typo o campo muerto)."""
    validos = set(_campos_de_gameodds())
    sobran = [c for c in MARKET_ODDS_FIELD_MAP if c not in validos]
    assert not sobran, f"MARKET_ODDS_FIELD_MAP apunta a campos que GameOdds no tiene: {sobran}"


def test_todo_campo_del_mapa_llega_con_valor():
    """De la fila cruda a GameOdds, sin perder nada por el camino.

    No basta con que la clave exista: tiene que llegar un valor. El bug de
    `_safe_float()` producía exactamente esto — clave presente, valor None.
    """
    game_data = _game_data_desde_fila(_fila_cruda())
    market_odds = build_market_odds(game_data)
    assert market_odds is not None

    vacios = [k for k, v in market_odds.items() if v is None]
    assert not vacios, (
        f"estos campos llegaron vacíos pese a venir con valor en la fila cruda: "
        f"{vacios} — se pierden entre _normalize_event, build_game_selector y "
        f"build_market_odds"
    )


def test_gameodds_se_construye_con_lo_que_da_la_ui():
    """El dict del selector entra en GameOdds sin sobrantes ni faltantes."""
    game_data = _game_data_desde_fila(_fila_cruda())
    odds = GameOdds(**build_market_odds(game_data))
    assert odds.ml_home == 2.02
    assert odds.runline_home_point == 1.5
    assert odds.f5_total_over == 1.88


def test_el_punto_firmado_negativo_sobrevive():
    """El caso mayoritario y el que un helper de precios anula.

    Local FAVORITO cotiza −1.5. Un `if v > 0` en cualquier punto del camino
    lo convierte en None y `analyze_runline` vuelve a asumir local favorito.
    """
    fila = _fila_cruda()
    fila["runline_home_point"] = -1.5
    odds = GameOdds(**build_market_odds(_game_data_desde_fila(fila)))
    assert odds.runline_home_point == -1.5


def test_sin_moneyline_no_hay_market_odds():
    """Sin las dos patas de ML el selector no tiene evento utilizable, y
    fabricar un precio par sería indistinguible de uno real aguas abajo."""
    fila = _fila_cruda()
    fila["home_odds"] = None
    assert build_market_odds(_game_data_desde_fila(fila)) is None


@pytest.mark.parametrize("campo", sorted(MARKET_ODDS_FIELD_MAP))
def test_cada_campo_por_separado(campo: str):
    """Un test por campo, generado del mapa: cuando uno se rompe, el nombre
    del que falla lo dice sin tener que leer un assert compuesto."""
    market_odds = build_market_odds(_game_data_desde_fila(_fila_cruda()))
    assert market_odds[campo] is not None, f"{campo} no sobrevive la costura"
