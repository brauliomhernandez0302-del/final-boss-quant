"""Paso 6 — la mano de la alineación: qué se usa cuando el lineup no existe.

Al publicar casi nunca hay lineup confirmado. Medido el 2026-07-27 sobre el log
real: la corrida de las 07:00 tuvo 0 de 27 juegos con lineup; la de las 13:00,
12 de 31. Histórico: 42 contra 199, o sea que el 83% de las corridas predice sin
alineación. Eso es inherente a publicar temprano y no es el defecto.

El defecto era qué se usaba en su lugar. `get_complete_game_data` rellenaba
`*_lineup_lhb_pct` con 0.45 cuando no había lineup, y `park_weather_engine` tiene
una cadena `_first_present(lineup, roster, liga)` que por eso NUNCA podía llegar
al segundo escalón — el roster real, que el propio pipeline mide 40 líneas más
arriba. Misma clase que FALL-002: un valor fabricado que aguas abajo no se
distingue de una medición, y que además tapa la medición. Error medio de ese 0.45
contra el roster real: 0.076, casi la desviación completa entre equipos (0.085).

Y las dos rutas ni siquiera medían lo mismo: el roster contaba al ambidiestro
como 0.5 y el lineup como 1.0, así que caer de un escalón al otro movía el número
0.056 en silencio.
"""
import pytest

import data_fetchers as df


# ── el ambidiestro se resuelve por la mano del rival ──────────────────────────

MANOS = {1: "L", 2: "R", 3: "S", 4: "S", 5: "R", 6: "R", 7: "L", 8: "R", 9: "R"}
NUEVE = list(range(1, 10))


@pytest.mark.parametrize("rival,esperado", [
    # 2 zurdos + 2 ambidiestros. Contra derecho el ambidiestro batea zurdo.
    ("R", round(4 / 9, 3)),
    # Contra zurdo batea derecho: sólo cuentan los 2 zurdos puros.
    ("L", round(2 / 9, 3)),
    # Sin saber contra quién batea, 0.5 reparte el error en vez de apostar.
    (None, round(3 / 9, 3)),
])
def test_el_ambidiestro_pesa_segun_la_mano_del_rival(rival, esperado):
    assert df.lhb_pct_efectivo(MANOS, NUEVE, rival) == esperado


def test_ni_cero_ni_uno_es_universal():
    """El punto entero: 1.0 y 0.0 son ambos correctos, cada uno en su caso.
    Fijar una convención única es equivocarse en el otro."""
    vs_r = df.lhb_pct_efectivo(MANOS, NUEVE, "R")
    vs_l = df.lhb_pct_efectivo(MANOS, NUEVE, "L")
    assert vs_r != vs_l
    assert vs_r > df.lhb_pct_efectivo(MANOS, NUEVE, None) > vs_l


def test_sin_bateadores_devuelve_none_no_una_media():
    """Fabricar 0.45 acá es exactamente lo que tapaba el dato real."""
    assert df.lhb_pct_efectivo(MANOS, [], "R") is None


def test_un_bateador_desconocido_cuenta_como_derecho():
    assert df.lhb_pct_efectivo({}, [99], "R") == 0.0


# ── sin lineup, la clave queda AUSENTE ────────────────────────────────────────

def _integrador(monkeypatch, con_lineup: bool):
    integ = df.MLBDataIntegrator()
    juego = {
        "game_pk": 1, "home_team": "H", "away_team": "A",
        "home_team_id": 10, "away_team_id": 20, "venue": "V",
        "game_date": "2026-07-27T23:05:00Z", "official_date": "2026-07-27",
        "home_pitcher_id": 100, "away_pitcher_id": 200,
        "home_lineup": [{"id": i, "name": f"b{i}"} for i in NUEVE] if con_lineup else [],
        "away_lineup": [],
    }
    monkeypatch.setattr(integ.mlb_api, "get_todays_games", lambda d=None: [juego])
    monkeypatch.setattr(integ, "_enrich_pitchers_concurrent", lambda *a, **k: {})
    for nombre, valor in [
        ("get_team_runs_trend", None), ("get_bullpen_workload", None),
        ("get_bullpen_era", None), ("get_roof_status", None),
        ("get_team_pitching_stats", None), ("get_travel_fatigue", None),
    ]:
        monkeypatch.setattr(integ.mlb_api, nombre, lambda *a, **k: valor)
    monkeypatch.setattr(integ.mlb_api, "get_team_days_rest", lambda *a, **k: 2)
    monkeypatch.setattr(integ.mlb_api, "get_pitcher_throws", lambda pid: "R")
    monkeypatch.setattr(integ.mlb_api, "get_batter_handedness_batch", lambda pids: MANOS)
    monkeypatch.setattr(integ.mlb_api, "get_team_batting_handedness_pct",
                        lambda tid, d, vs_hand=None: 0.30)
    monkeypatch.setattr(integ.mlb_api, "get_prev_lineup_lhb_pct",
                        lambda tid, d, vs_hand=None: 0.50)
    monkeypatch.setattr(integ.weather_api, "get_weather_for_stadium", lambda *a, **k: None)
    return integ


def test_sin_lineup_la_clave_no_se_inventa(monkeypatch):
    g = _integrador(monkeypatch, con_lineup=False).get_complete_game_data(
        date="2026-07-27", season=2026)[0]
    assert "away_lineup_lhb_pct" not in g, "rellenarla con 0.45 tapaba el roster real"
    assert "home_lineup_lhb_pct" not in g


def test_con_lineup_la_clave_aparece_resuelta(monkeypatch):
    g = _integrador(monkeypatch, con_lineup=True).get_complete_game_data(
        date="2026-07-27", season=2026)[0]
    assert g["home_lineup_lhb_pct"] == round(4 / 9, 3)   # vs RHP
    assert "away_lineup_lhb_pct" not in g                # ese lado no tiene lineup


def test_la_estimacion_del_equipo_es_la_mezcla_medida(monkeypatch):
    """0.5·(alineación previa) + 0.5·(roster) — mitad y mitad ganó las dos
    métricas sobre 250 casos reales (MAE 0.0835 vs 0.0936 y 0.0986 solos)."""
    g = _integrador(monkeypatch, con_lineup=False).get_complete_game_data(
        date="2026-07-27", season=2026)[0]
    assert g["home_lhb_pct"] == pytest.approx(0.4)   # 0.5*0.50 + 0.5*0.30


# ── la cadena de respaldo, ahora sí completa ──────────────────────────────────

def test_el_pitcher_engine_usa_la_cadena_y_no_solo_el_lineup():
    """Antes leía SÓLO `*_lineup_lhb_pct`, así que en el 83% de las corridas
    usaba una constante teniendo la estimación del equipo en el mismo dict."""
    import sys
    sys.path.insert(0, "modules/baseball_module")
    from context_engine.pitcher_engine import PitcherEngine
    from config import LEAGUE_AVG_LHB_PCT

    e = PitcherEngine()
    pitcher = {"whip": 1.20, "platoon_splits": {
        "vs_lhb": {"whip": 1.60}, "vs_rhb": {"whip": 1.00}}}

    solo_equipo = e._adjust_pitcher_platoon(pitcher, {"away_lhb_pct": 0.70}, is_home=True)
    con_lineup = e._adjust_pitcher_platoon(
        pitcher, {"away_lhb_pct": 0.70, "away_lineup_lhb_pct": 0.20}, is_home=True)
    sin_nada = e._adjust_pitcher_platoon(pitcher, {}, is_home=True)

    assert solo_equipo != sin_nada, "el dato del equipo tiene que llegar al motor"
    assert con_lineup != solo_equipo, "el lineup confirmado tiene que ganarle al equipo"
    # Sin ningún dato cae a la constante de liga, no a un literal suelto.
    assert sin_nada == e._adjust_pitcher_platoon(
        pitcher, {"away_lhb_pct": LEAGUE_AVG_LHB_PCT}, is_home=True)


def test_la_constante_de_liga_tiene_una_sola_fuente():
    """Estaba duplicada como 0.45 en park_weather_engine y como literal suelto
    en pitcher_engine — mismo patrón que motivó centralizar LEAGUE_AVG_XWOBA."""
    from config import LEAGUE_AVG_LHB_PCT
    import modules.baseball_module.hfa.park_weather_engine as pw
    import modules.baseball_module.context_engine.pitcher_engine as pe
    # Los dos motores tienen que apuntar al MISMO objeto, no a copias iguales:
    # dos literales que hoy coinciden es exactamente cómo empezó este bug.
    assert pw._AVG_LHB_PCT is LEAGUE_AVG_LHB_PCT
    assert pe._LG_LHB_PCT is LEAGUE_AVG_LHB_PCT


def test_la_constante_esta_calibrada_contra_el_dato_real():
    """Media real de las 24 alineaciones de la cartelera: 0.406. El 0.45
    anterior estaba ~4pp sesgado a la izquierda."""
    from config import LEAGUE_AVG_LHB_PCT
    assert 0.39 <= LEAGUE_AVG_LHB_PCT <= 0.42
