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


# ── 1b. la ventana de viaje ───────────────────────────────────────────────────

def _params_de_la_consulta(monkeypatch, **kwargs):
    """Captura los params con que get_travel_fatigue consulta el schedule."""
    capturado = {}

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"dates": []}

    api = df.MLBStatsAPI()
    monkeypatch.setattr(api.session, "get",
                        lambda url, **kw: (capturado.update(kw.get("params", {})), _Resp())[1])
    # Sin caché: el fixture escribe en CACHE_DIR y contaminaría la siguiente corrida.
    monkeypatch.setattr(df.Path, "exists", lambda self: False)
    api.get_travel_fatigue(**kwargs)
    return capturado


def test_la_ventana_de_viaje_se_ancla_en_el_dia_oficial(monkeypatch):
    """Nocturno del oeste: el UTC es 07-28, el día del juego es 07-27.

    La ventana pretendida son 3 días hacia atrás desde el DÍA del juego. Con el
    UTC quedaba [07-26, 07-29] — perdía el 07-24 y llegaba a un día futuro.
    """
    p = _params_de_la_consulta(
        monkeypatch, team_id=1, game_date="2026-07-28T02:10:00Z",
        current_venue="Angel Stadium", official_date="2026-07-27",
    )
    assert p["startDate"] == "2026-07-24"
    assert p["endDate"] == "2026-07-27"


def test_sin_dia_oficial_cae_al_comportamiento_viejo(monkeypatch):
    """Fallback explícito: sin el dato no se puede anclar, y no se inventa."""
    p = _params_de_la_consulta(
        monkeypatch, team_id=1, game_date="2026-07-28T02:10:00Z",
        current_venue="Angel Stadium",
    )
    assert p["startDate"] == "2026-07-25"
    assert p["endDate"] == "2026-07-28"


def test_la_ventana_sigue_siendo_de_tres_dias(monkeypatch):
    """NO se ensanchó, y no debe ensancharse.

    `hfa_engine._travel_penalty` castiga por millas y husos sin mirar recencia
    (nunca lee `hours_since_last_game`), así que un día extra de ventana le
    daría penalización de viaje a un equipo que lleva 3-4 días en la ciudad.
    Anclar corrige el sesgo; ensanchar lo empeoraría.
    """
    from datetime import date, datetime as _dt
    p = _params_de_la_consulta(
        monkeypatch, team_id=1, game_date="2026-07-27T23:05:00Z",
        current_venue="X", official_date="2026-07-27",
    )
    ini = _dt.strptime(p["startDate"], "%Y-%m-%d").date()
    fin = _dt.strptime(p["endDate"], "%Y-%m-%d").date()
    assert (fin - ini).days == 3


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
