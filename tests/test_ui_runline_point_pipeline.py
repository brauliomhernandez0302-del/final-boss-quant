"""El punto FIRMADO del runline sobrevive el camino del selector de la UI.

Contexto (2026-08-03). `analyze_runline` necesita `runline_home_point` para
saber quién es el favorito y por lo tanto cuál es el evento de cobertura de
cada lado; sin él asume local favorito, que es la causa raíz de PURP-1
(arreglada el 2026-07-31, pero sólo en el camino de `get_best_odds_for_teams`).

El camino de la UI lo perdía en DOS puntos independientes, y hacía falta
arreglar los dos para que el dato llegara:

  1. `build_game_selector()` no copiaba el campo a `GameData` en absoluto,
     pese a que `_normalize_event()` ya lo emitía y a que su propio docstring
     promete poblar el GameData completo.
  2. Una vez copiado, pasaba por `_safe_float()`, que descarta todo `<= 0` —
     correcto para precios, fatal acá: el punto del local FAVORITO es −1.5,
     el caso mayoritario. El arreglo del punto 1 por sí solo habría quedado
     silenciosamente sin efecto justo donde más importa.

Estos tests fijan los dos, con el caso del favorito primero porque es el que
un helper de precios rompe.
"""
import pandas as pd
import pytest

from ui.odds_loader import build_game_selector, _safe_float, _safe_signed_float


def _row(point):
    return {
        "home_team": "Arizona Diamondbacks",
        "away_team": "San Diego Padres",
        "commence_time": "2026-08-03T01:40:00Z",
        "home_odds": 2.02, "away_odds": 1.96,
        "runline_home": 1.65, "runline_away": 2.25,
        "runline_line": 1.5,
        "runline_home_point": point,
    }


def _point_from_selector(point):
    options, mapping = build_game_selector(pd.DataFrame([_row(point)]))
    return mapping[options[0]].get("runline_home_point")


@pytest.mark.parametrize("point", [-1.5, -2.5])
def test_home_favorite_negative_point_survives(point):
    """El caso que `_safe_float` borraba: local favorito, punto negativo."""
    assert _point_from_selector(point) == point


def test_home_underdog_positive_point_survives():
    assert _point_from_selector(1.5) == 1.5


def test_missing_point_stays_none():
    """Sin dato debe quedar None — que `analyze_runline` sepa que no lo tiene
    es mejor que un 0 o un 1.5 fabricado, que serían indistinguibles de reales."""
    assert _point_from_selector(None) is None


def test_safe_float_still_rejects_non_positive_prices():
    """No aflojar el helper de PRECIOS al agregar el de magnitudes firmadas:
    una cuota decimal negativa o cero sigue siendo inválida."""
    assert _safe_float(-1.5) is None
    assert _safe_float(0) is None
    assert _safe_float(2.02) == 2.02


def test_safe_signed_float_keeps_sign_and_drops_zero():
    assert _safe_signed_float(-1.5) == -1.5
    assert _safe_signed_float(1.5) == 1.5
    assert _safe_signed_float(0) is None
    assert _safe_signed_float(None) is None
    assert _safe_signed_float("no-es-un-numero") is None


def test_gamedata_declares_the_field():
    """El TypedDict debe declararlo: es lo que hace que un `.get()` mal escrito
    en el futuro se note en el type-check en vez de devolver None en silencio."""
    from db.predictions_db import GameData
    assert "runline_home_point" in GameData.__annotations__
