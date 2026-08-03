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
(90 min). El umbral no es arbitrario: la separación real entre los dos juegos de
un doubleheader, medida sobre `gameDate` del schedule de MLB en las 431 fechas
con datos (2024-2026, el 2026-07-27), es bimodal con el hueco vacío —

    partido      (doubleHeader='S', n=43): 275 a 405 min
    tradicional  (doubleHeader='Y', n=24): 5 min los 24

— y los dos modos piden respuestas opuestas: el partido se empareja bien, el
tradicional NO se puede desambiguar por hora de inicio y hay que abstenerse.
90 min es el único umbral que da las dos, con 3x y 18x de holgura.

Los tests de abajo usan esos dos regímenes medidos como casos, no números
inventados — mismo criterio que la lista de estados publicables (paso 1) y la
garantía pre-juego (paso 2): ante la duda, abstenerse.
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


@pytest.mark.parametrize("separacion_min", [275, 330, 405])
def test_doubleheader_partido_se_resuelve_bien(sin_red, separacion_min):
    """DH partido: los tres percentiles medidos (mín, mediana, máx) se emparejan.

    Con 275+ min de separación el candidato correcto gana por mucho más que el
    margen, así que cada juego se lleva SU propio precio.
    """
    h1, m1 = 16, 35
    t1 = _t(h1, m1)
    t2 = _t(h1 + (m1 + separacion_min) // 60, (m1 + separacion_min) % 60)
    sin_red([
        _evento("New York Yankees", "Pittsburgh Pirates", t1, ml_home=1.50),
        _evento("New York Yankees", "Pittsburgh Pirates", t2, ml_home=1.90),
    ])
    primero = of.get_best_odds_for_teams("New York Yankees", "Pittsburgh Pirates", t1)
    assert primero.get("ml_home") == 1.50, "debe tomar el precio de SU juego"
    # El segundo juego sólo entra si está dentro de la ventana de ±6h; a 405 min
    # el primer evento ya quedó fuera y es el único candidato de todos modos.
    segundo = of.get_best_odds_for_teams("New York Yankees", "Pittsburgh Pirates", t2)
    assert segundo.get("ml_home") == 1.90


def test_doubleheader_tradicional_se_abstiene(sin_red):
    """El caso REAL, y el que la regla vieja resolvía a cara o cruz.

    Los 24 doubleheaders tradicionales medidos tienen sus dos juegos a exactamente
    5 min (el schedule le pone un marcador al segundo, que arranca "a continuación"
    del primero). A esa distancia no hay forma de saber cuál evento es cuál: con la
    regla anterior el juego 2 se llevaba el precio del juego 1 en silencio.
    """
    sin_red([
        _evento("New York Yankees", "Pittsburgh Pirates", _t(16, 35), ml_home=1.50),
        _evento("New York Yankees", "Pittsburgh Pirates", _t(16, 40), ml_home=1.90),
    ])
    for objetivo in (_t(16, 35), _t(16, 40)):
        assert of.get_best_odds_for_teams(
            "New York Yankees", "Pittsburgh Pirates", objetivo
        ) == {}, "5 min de separación no distinguen nada: los dos quedan sin precio"


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


def test_el_umbral_cae_en_el_hueco_entre_los_dos_regimenes_medidos():
    """90 min separa los dos modos medidos sin rozar ninguno.

    Si alguien mueve la constante, este test dice exactamente qué se rompe: por
    debajo de 5 min deja de abstenerse en los tradicionales (vuelve el bug), por
    encima de 275 min deja de emparejar los partidos (pierde picks buenos).
    """
    DH_TRADICIONAL_MAX = timedelta(minutes=5)      # n=24, sin dispersión
    DH_PARTIDO_MIN = timedelta(minutes=275)        # n=43, el mínimo observado

    assert of._ODDS_MATCH_MIN_MARGIN == timedelta(minutes=90)
    assert DH_TRADICIONAL_MAX < of._ODDS_MATCH_MIN_MARGIN < DH_PARTIDO_MIN
    # Holgura de sobra a cada lado, no un ajuste al borde de los datos.
    assert of._ODDS_MATCH_MIN_MARGIN > DH_TRADICIONAL_MAX * 3
    assert of._ODDS_MATCH_MIN_MARGIN < DH_PARTIDO_MIN / 3
    # Y el margen tiene que caber dentro de la ventana, o nunca podría cumplirse.
    assert of._ODDS_MATCH_MIN_MARGIN < of._ODDS_MATCH_WINDOW
