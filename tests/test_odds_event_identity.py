"""Identidad juego ↔ evento de odds: ante dos candidatos plausibles, abstenerse.

`get_best_odds_for_teams` resuelve una pregunta de IDENTIDAD ("¿cuál evento del
mercado es ESTE juego?") con una heurística de PARECIDO: substring de los nombres
de ambos equipos + ventana de ±6h. Funciona — medido el 2026-07-26 sobre los 27
juegos de la ventana: 12 de 12 emparejados entre los jugables, 0 ambigüedades de
nombre — pero el desempate tenía un borde peligroso.

La regla anterior sólo se abstenía ante un empate EXACTO de distancia temporal,
cosa que no pasa nunca. En un doubleheader (dos juegos del mismo par de equipos
dentro de la ventana) bastaba con estar un minuto más cerca para quedarse con el
precio del OTRO partido, sin ningún aviso.

Ahora el mejor candidato tiene que ganarle al segundo por `_ODDS_MATCH_MIN_MARGIN`
(90 min). Un doubleheader real separa sus juegos por 3.5-7h, así que el candidato
correcto sigue ganando por horas; por debajo de 90 min los dos eventos son
igual de plausibles y el sistema se abstiene en vez de adivinar — mismo criterio
que la lista de estados publicables (paso 1) y la garantía pre-juego (paso 2).
"""
from datetime import datetime, timedelta, timezone

import pytest

import odds_fetcher as of


def _evento(home, away, commence, ml_home=1.80, ml_away=2.10):
    return {
        "sport_key": "baseball_mlb", "home_team": home, "away_team": away,
        "commence_time": commence,
        "bookmakers": [{
            "key": "pinnacle", "title": "Pinnacle",
            "markets": [{"key": "h2h", "outcomes": [
                {"name": home, "price": ml_home}, {"name": away, "price": ml_away},
            ]}],
        }],
    }


def _t(h, m=0):
    return f"2026-08-01T{h:02d}:{m:02d}:00Z"


@pytest.fixture
def sin_red(monkeypatch):
    """Aísla del disco y de la red: el matcher lee sólo lo que le damos."""
    def _instalar(eventos):
        monkeypatch.setattr(of, "_get_raw_events", lambda: eventos)
    return _instalar


def test_un_solo_candidato_se_empareja(sin_red):
    sin_red([_evento("New York Yankees", "Pittsburgh Pirates", _t(23))])
    r = of.get_best_odds_for_teams("New York Yankees", "Pittsburgh Pirates", _t(23))
    assert r.get("ml_home") == 1.80


def test_doubleheader_lejano_se_resuelve_bien(sin_red):
    """Dos juegos a 4h: el correcto gana por horas, el margen no estorba."""
    sin_red([
        _evento("New York Yankees", "Pittsburgh Pirates", _t(17), ml_home=1.50),
        _evento("New York Yankees", "Pittsburgh Pirates", _t(21), ml_home=1.90),
    ])
    primero = of.get_best_odds_for_teams("New York Yankees", "Pittsburgh Pirates", _t(17))
    segundo = of.get_best_odds_for_teams("New York Yankees", "Pittsburgh Pirates", _t(21))
    assert primero.get("ml_home") == 1.50, "debe tomar el precio de SU juego"
    assert segundo.get("ml_home") == 1.90


def test_dos_candidatos_casi_igual_de_cerca_se_abstiene(sin_red):
    """El caso que la regla vieja resolvía a cara o cruz: 30 min de diferencia."""
    sin_red([
        _evento("New York Yankees", "Pittsburgh Pirates", _t(19, 30)),
        _evento("New York Yankees", "Pittsburgh Pirates", _t(20, 30)),
    ])
    # Objetivo justo en el medio: 30 min a cada lado.
    r = of.get_best_odds_for_teams("New York Yankees", "Pittsburgh Pirates", _t(20))
    assert r == {}, "dos eventos igual de plausibles ⇒ no adivinar"


def test_empate_exacto_se_abstiene(sin_red):
    sin_red([
        _evento("New York Yankees", "Pittsburgh Pirates", _t(19)),
        _evento("New York Yankees", "Pittsburgh Pirates", _t(21)),
    ])
    assert of.get_best_odds_for_teams("New York Yankees", "Pittsburgh Pirates", _t(20)) == {}


def test_fuera_de_la_ventana_no_empareja(sin_red):
    sin_red([_evento("New York Yankees", "Pittsburgh Pirates", _t(23))])
    # 7h de diferencia, fuera de los ±6h
    assert of.get_best_odds_for_teams("New York Yankees", "Pittsburgh Pirates", _t(16)) == {}


def test_commence_time_ilegible_no_empareja(sin_red):
    sin_red([_evento("New York Yankees", "Pittsburgh Pirates", _t(23))])
    assert of.get_best_odds_for_teams("New York Yankees", "Pittsburgh Pirates", "no-es-fecha") == {}


def test_el_margen_es_mayor_que_cualquier_ruido_de_reloj():
    """90 min: holgado frente a un doubleheader real (3.5-7h de separación) y
    muy por encima de cualquier discrepancia de horario entre feeds."""
    assert of._ODDS_MATCH_MIN_MARGIN == timedelta(minutes=90)
    assert of._ODDS_MATCH_MIN_MARGIN < of._ODDS_MATCH_WINDOW
