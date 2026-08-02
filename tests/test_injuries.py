"""Las lesiones entran al módulo de béisbol. Hasta el 2026-08-01 no existían.

`app.py` tenía `"home_injuries": [], "away_injuries": []` HARDCODEADO en vacío, y
nada más en todo el módulo de béisbol miraba lesiones. El contraste con básquet
es directo: ahí se modelan con peso 0.20 y función de análisis propia.

Por qué importa: la ofensa de cada equipo se calcula del Statcast ACUMULADO de la
temporada, así que incluye entera la producción de quien hoy está en la lista de
lesionados. Casos reales del día en que se construyó esto:

    Boston Red Sox      5 bateadores fuera   22.1% de sus turnos ofensivos
    Baltimore Orioles   4 fuera              19.8%   (incluido Adley Rutschman)
    Toronto Blue Jays   3 fuera               2.1%

La dispersión entre equipos va de 1.6% a 22.1%. El modelo no distinguía ninguno.

ALCANCE DE ESTE PASO: el dato ENTRA y queda registrado por pick. El λ todavía NO
lo descuenta — primero el dato, después medir cuánto importa, después decidir cómo
usarlo. Los tests fijan esa frontera para que usarlo sea una decisión explícita.
"""
import sys

import pytest

sys.path.insert(0, ".")
import data_fetchers as df  # noqa: E402


def _roster(entradas):
    return {"roster": [
        {"person": {"id": i, "fullName": n,
                    "stats": [{"splits": [{"stat": {"plateAppearances": pa}}]}]},
         "position": {"abbreviation": pos},
         "status": {"description": est}}
        for i, (n, pa, pos, est) in enumerate(entradas, 1)
    ]}


@pytest.fixture
def api(monkeypatch):
    a = df.MLBStatsAPI()
    monkeypatch.setattr(df.Path, "exists", lambda self: False)   # sin caché
    return a


def _instalar(monkeypatch, api, entradas):
    class _R:
        def raise_for_status(self): pass
        def json(self): return _roster(entradas)
    monkeypatch.setattr(api.session, "get", lambda *a, **k: _R())


# ── el cálculo ────────────────────────────────────────────────────────────────

def test_cuenta_los_turnos_de_los_lesionados(monkeypatch, api):
    _instalar(monkeypatch, api, [
        ("Sano A", 500, "LF", "Active"),
        ("Sano B", 300, "SS", "Active"),
        ("Lesionado", 200, "RF", "Injured 10-Day"),
    ])
    r = api.get_team_injuries(1, 2026)
    assert r["n_out"] == 1
    assert r["pa_share_out"] == pytest.approx(200 / 1000)
    assert r["players"][0]["name"] == "Lesionado"


def test_los_lanzadores_no_cuentan(monkeypatch, api):
    """Un abridor lesionado ya lo cubre el probable pitcher; contarlo acá sería
    doble conteo, y además contamina una métrica que es de OFENSA."""
    _instalar(monkeypatch, api, [
        ("Bateador", 400, "CF", "Active"),
        ("Abridor", 5, "P", "Injured 60-Day"),
    ])
    r = api.get_team_injuries(1, 2026)
    assert r["n_out"] == 0
    assert r["pa_share_out"] == 0.0


def test_ordena_por_importancia_ofensiva(monkeypatch, api):
    """El que más turnos se lleva primero: no es lo mismo perder al 3 que al 9."""
    _instalar(monkeypatch, api, [
        ("Suplente", 40, "C", "Injured 10-Day"),
        ("Titular",  500, "RF", "Injured 60-Day"),
        ("Sano", 400, "SS", "Active"),
    ])
    r = api.get_team_injuries(1, 2026)
    assert [p["name"] for p in r["players"]] == ["Titular", "Suplente"]


@pytest.mark.parametrize("estado,fuera", [
    ("Injured 10-Day", True), ("Injured 15-Day", True), ("Injured 60-Day", True),
    ("Active", False), ("Reassigned to Minors", False), ("Free Agent", False),
])
def test_reconoce_los_estados_reales_de_la_api(monkeypatch, api, estado, fuera):
    _instalar(monkeypatch, api, [("Sano", 400, "SS", "Active"),
                                 ("X", 100, "LF", estado)])
    assert (api.get_team_injuries(1, 2026)["n_out"] == 1) is fuera


def test_sin_turnos_no_inventa_una_fraccion(monkeypatch, api):
    _instalar(monkeypatch, api, [("Nadie", 0, "SS", "Active")])
    assert api.get_team_injuries(1, 2026) is None


def test_un_fallo_de_la_api_no_rompe_el_juego(monkeypatch, api):
    def _boom(*a, **k): raise RuntimeError("caído")
    monkeypatch.setattr(api.session, "get", _boom)
    assert api.get_team_injuries(1, 2026) is None


# ── que el dato ENTRE de verdad ───────────────────────────────────────────────

def test_el_pipeline_lo_incorpora():
    fuente = open("data_fetchers.py", encoding="utf-8").read()
    bloque = fuente[fuente.index("def get_complete_game_data"):]
    assert "get_team_injuries" in bloque, "no alcanza con tener el fetcher"
    assert '"home_injuries"' in bloque and '"away_injuries"' in bloque
    # La clave del share se arma dinámicamente por lado, así que el literal
    # completo no aparece en la fuente — se verifica el patrón que la construye.
    assert '_injured_pa_share' in bloque


def _dict_de_la_instantanea() -> str:
    """Devuelve el literal COMPLETO de `results['inputs_snapshot']`.

    Antes esto era un corte fijo de 2.400 caracteres desde el inicio del dict,
    que rompía por motivos ajenos al test en cuanto alguien documentaba un
    campo: el 2026-08-03 el arreglo de `weather_source` añadió un comentario y
    empujó las claves de lesiones al carácter 2.4xx, con las claves intactas.
    Emparejar llaves mide lo que el test dice medir — que estén en el dict —
    en vez de dónde caen dentro de una ventana arbitraria.
    """
    fuente = open("modules/baseball_module/core/run_module.py", encoding="utf-8").read()
    ini = fuente.index("results['inputs_snapshot'] = {")
    apertura = fuente.index("{", ini)
    profundidad = 0
    for pos in range(apertura, len(fuente)):
        if fuente[pos] == "{":
            profundidad += 1
        elif fuente[pos] == "}":
            profundidad -= 1
            if profundidad == 0:
                return fuente[ini:pos + 1]
    raise AssertionError("el dict de inputs_snapshot no cierra")


def test_queda_registrado_en_cada_pick():
    """Sin esto no se puede medir después si los picks de equipos diezmados
    rinden distinto — que es la pregunta que decide si el λ debe descontarlo."""
    bloque = _dict_de_la_instantanea()
    assert "home_injured_pa_share" in bloque and "away_injured_pa_share" in bloque


def test_el_placeholder_muerto_quedo_explicado():
    """El `[]` de app.py sigue ahí por compatibilidad de forma, pero ya no puede
    hacerle creer a nadie que el módulo contemplaba lesiones."""
    fuente = open("app.py", encoding="utf-8").read()
    i = fuente.index('"home_injuries"')
    assert "get_team_injuries" in fuente[max(0, i - 700):i]


def test_el_lambda_todavia_NO_descuenta_lesiones():
    """Frontera explícita de este paso: el dato entra y se registra, el λ no lo
    usa. Cuando se decida usarlo tiene que ser un cambio deliberado, medido con
    su propio gate — no un efecto colateral de haber traído el dato."""
    tte = open("modules/baseball_module/offense/true_talent_engine.py", encoding="utf-8").read()
    formula = open("modules/baseball_module/offense/tte_formula.py", encoding="utf-8").read()
    for fuente in (tte, formula):
        assert "injur" not in fuente.lower() and "lesion" not in fuente.lower()
