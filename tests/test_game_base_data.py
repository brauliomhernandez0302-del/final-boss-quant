"""Paso 4 — los hechos base del juego: el día, el doubleheader, y qué se sabe.

Tres defectos reales, encontrados auditando `data_fetchers.py` el 2026-07-27:

1. El DÍA del juego se derivaba de `game_date[:10]`, que es el timestamp UTC de
   primer pitcheo truncado — para cualquier nocturno que cruce medianoche UTC (la
   norma en la costa oeste) eso es el día SIGUIENTE. `get_team_days_rest` resta
   de ahí la fecha del último juego, que la API devuelve como día oficial: mezcla
   de unidades, y el resultado era exactamente +1 día de descanso. Medido sobre la
   ventana real: 8 de 27 juegos (30%) difieren, y en ellos 16 de 16 equipos se
   movían. Sesgo direccional, siempre a favor de quien juega de noche en el oeste.

   Misma clase que el leak V4 que la Fase 2B arregló del lado del backtest; del
   lado vivo nadie lo había mirado. El campo correcto (`official_date`) ya estaba
   en el mismo diccionario, con un comentario que pedía justamente usarlo.

2. `doubleHeader`/`gameNumber` venían en el payload y se descartaban — son la
   única forma de saber que dos juegos del mismo par de equipos el mismo día son
   dos juegos distintos.

3. Una excepción a mitad del enriquecimiento devolvía el juego crudo sin ninguna
   marca, indistinguible de uno normal: aguas abajo, bullpen/defensa/descanso
   caían a promedio de liga en silencio y el pick se hacía igual.
"""
import pytest

import data_fetchers as df


# ── 1. el día del juego ───────────────────────────────────────────────────────

@pytest.mark.parametrize("game_date,official,esperado", [
    # Nocturno del oeste: UTC ya es el día siguiente. El caso que rompía.
    ("2026-07-27T02:10:00Z", "2026-07-26", "2026-07-26"),
    # Diurno del este: coinciden, y tienen que seguir coincidiendo.
    ("2026-07-27T17:05:00Z", "2026-07-27", "2026-07-27"),
])
def test_el_dia_sale_del_campo_oficial(game_date, official, esperado):
    assert df._official_day(
        {"game_date": game_date, "official_date": official}
    ) == esperado


def test_sin_official_date_avisa_y_no_falla_callado(caplog):
    """El fallback es el comportamiento viejo (sesgado): tiene que verse."""
    with caplog.at_level("WARNING"):
        dia = df._official_day({
            "game_date": "2026-07-27T02:10:00Z", "official_date": None,
            "game_pk": 1, "home_team": "H", "away_team": "A",
        })
    assert dia == "2026-07-27"
    assert "official_date ausente" in caplog.text


def test_official_date_se_normaliza_a_diez_caracteres():
    assert df._official_day(
        {"game_date": "x", "official_date": "2026-07-26T00:00:00Z"}
    ) == "2026-07-26"


def test_los_dos_consumidores_del_dia_usan_el_campo_oficial():
    """Ambos sitios que preguntan "¿qué día es este juego?" pasan por el helper.

    Fijado como texto porque el bug era precisamente que UNO de los dos usaba
    otra cosa, y nada lo delataba.
    """
    fuente = (df.__file__).replace(".pyc", ".py")
    texto = open(fuente, encoding="utf-8").read()
    cuerpo = texto[texto.index("def get_complete_game_data"):]
    assert cuerpo.count("game_date_str = _official_day(game)") == 2
    assert 'game_date_str = (game.get("game_date") or "")[:10]' not in cuerpo


# ── 2. doubleheader ───────────────────────────────────────────────────────────

def _payload(**extra):
    base = {
        "gamePk": 1, "gameDate": "2026-04-30T16:35:00Z", "officialDate": "2026-04-30",
        "teams": {
            "home": {"team": {"id": 110, "name": "Baltimore Orioles"}},
            "away": {"team": {"id": 117, "name": "Houston Astros"}},
        },
        "status": {"detailedState": "Scheduled"},
        "venue": {"name": "Oriole Park"},
    }
    base.update(extra)
    return base


@pytest.mark.parametrize("dh,num", [("Y", 2), ("S", 1), ("N", 1)])
def test_el_doubleheader_se_conserva(dh, num):
    g = df.MLBStatsAPI()._parse_game(_payload(doubleHeader=dh, gameNumber=num))
    assert g["doubleheader"] == dh
    assert g["game_number"] == num


def test_sin_campos_de_doubleheader_asume_juego_unico():
    g = df.MLBStatsAPI()._parse_game(_payload())
    assert g["doubleheader"] == "N"
    assert g["game_number"] == 1


def test_el_dia_oficial_sobrevive_al_parseo():
    """`_official_day` depende de que `_parse_game` lo haya guardado."""
    g = df.MLBStatsAPI()._parse_game(_payload())
    assert df._official_day(g) == "2026-04-30"


# ── 3. fallo de enriquecimiento visible ───────────────────────────────────────

def test_un_fallo_de_enriquecimiento_queda_marcado(monkeypatch):
    integrador = df.MLBDataIntegrator()
    crudo = {"game_pk": 1, "home_team": "H", "away_team": "A",
             "home_team_id": 1, "away_team_id": 2, "venue": "V",
             "game_date": "2026-07-27T23:05:00Z", "official_date": "2026-07-27"}
    monkeypatch.setattr(integrador.mlb_api, "get_todays_games", lambda d=None: [crudo])
    monkeypatch.setattr(integrador, "_enrich_pitchers_concurrent",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    salida = integrador.get_complete_game_data(date="2026-07-27", season=2026)
    assert len(salida) == 1
    g = salida[0]
    assert g["enrichment_failed"] is True
    assert "RuntimeError: boom" in g["enrichment_error"]
    assert g["pitchers_valid"] is False, "no se sabe si son válidos: la respuesta honesta es False"


def test_el_fallo_no_contamina_el_dict_de_entrada(monkeypatch):
    """El juego crudo lo comparte el llamador — marcarlo in-place lo mutaría."""
    integrador = df.MLBDataIntegrator()
    crudo = {"game_pk": 1, "home_team": "H", "away_team": "A",
             "game_date": "2026-07-27T23:05:00Z", "official_date": "2026-07-27"}
    monkeypatch.setattr(integrador.mlb_api, "get_todays_games", lambda d=None: [crudo])
    monkeypatch.setattr(integrador, "_enrich_pitchers_concurrent",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    integrador.get_complete_game_data(date="2026-07-27", season=2026)
    assert "enrichment_failed" not in crudo
