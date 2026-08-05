"""Línea justa de mercados de dos lados, y canonización de nombres de equipo.

Los dos bloques cubren defectos encontrados el 2026-08-04 recorriendo el paso 1
(precios) de la reconstrucción: la línea justa de los derivados salía de un par
sintético que no cotiza ninguna casa, y el emparejador de nombres traducía en
un solo sentido, dejando 276 juegos sin precio atribuidos a "team name
mismatch".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.value_detector import _fair_two_way, remove_vig_multiplicative
from fetch_historical_odds import match_games


# ── Línea justa ──────────────────────────────────────────────────────────


def test_prefiere_el_par_de_pinnacle_sobre_el_par_tomado():
    justo, _, fuente = _fair_two_way(
        2.10, 2.05,                      # par tomado (mejor precio de cada lado)
        pin_side=1.95, pin_opposite=1.95,
        pin_point=8.5, target_point=8.5,
    )
    assert fuente == "pinnacle"
    assert justo == pytest.approx(0.5)   # 1.95/1.95 devigado es exactamente 50/50


def test_cae_al_par_tomado_sin_pinnacle():
    justo, opuesto, fuente = _fair_two_way(2.10, 2.05, vig_method="multiplicative")
    assert fuente == "multiplicative"
    assert justo + opuesto == pytest.approx(1.0)
    assert (justo, opuesto) == pytest.approx(tuple(remove_vig_multiplicative([2.10, 2.05])))


def test_descarta_el_par_de_pinnacle_si_cotiza_otro_punto():
    """Un total de 8.5 y uno de 9.0 no son dos lados del mismo mercado — usar
    el precio de uno como referencia justa del otro es la misma clase de error
    de emparejamiento que PURP-1."""
    _, _, fuente = _fair_two_way(
        2.10, 2.05,
        pin_side=1.95, pin_opposite=1.95,
        pin_point=9.0, target_point=8.5,
    )
    assert fuente == "multiplicative"


def test_el_punto_del_runline_se_compara_con_signo():
    """Local a −1.5 y local a +1.5 son apuestas distintas, no la misma con
    otro signo."""
    _, _, fuente = _fair_two_way(
        2.40, 1.62,
        pin_side=2.38, pin_opposite=1.65,
        pin_point=-1.5, target_point=+1.5,
    )
    assert fuente == "multiplicative"

    _, _, fuente_ok = _fair_two_way(
        2.40, 1.62,
        pin_side=2.38, pin_opposite=1.65,
        pin_point=-1.5, target_point=-1.5,
    )
    assert fuente_ok == "pinnacle"


def test_medio_par_de_pinnacle_no_alcanza():
    _, _, fuente = _fair_two_way(2.10, 2.05, pin_side=1.95, pin_opposite=None)
    assert fuente == "multiplicative"


# ── Canonización de nombres ──────────────────────────────────────────────


def _odds_game(home: str, away: str):
    return {"home_team": home, "away_team": away,
            "commence_time": "2026-08-04T23:05:00Z"}


@pytest.mark.parametrize("nombre_api", [
    "Athletics",            # como los llama la Odds API en 2026
    "Oakland Athletics",    # como los llamaba en 2024-2025
    "Sacramento Athletics",
])
def test_athletics_empareja_con_cualquier_alias(nombre_api):
    """El equipo cambió de nombre dos veces y cada fuente lo hizo en su
    momento. Canonizar los dos lados vuelve el emparejamiento inmune a eso."""
    resultado = match_games(
        [_odds_game("Cincinnati Reds", nombre_api)],
        [(824484, "Cincinnati Reds", "Athletics")],
    )
    assert [game_pk for _, game_pk in resultado] == [824484]


def test_equipos_distintos_no_se_emparejan():
    """La canonización no puede volverse un fuzzy match que acerque equipos
    que no son el mismo."""
    resultado = match_games(
        [_odds_game("Cincinnati Reds", "Los Angeles Angels")],
        [(824484, "Cincinnati Reds", "Athletics")],
    )
    assert resultado == []
