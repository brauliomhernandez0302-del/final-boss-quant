"""PURP-1 — el run line se precia con el punto FIRMADO del lado, no con la magnitud.

`analyze_runline` recibía sólo la magnitud de la línea (`odds_fetcher` la construye
con `abs(...)`) y calculaba:

    p_home_cover = P(diff >  runline_line)
    p_away_cover = P(diff <  runline_line)

Eso da el evento correcto SÓLO si el local es el favorito. Cuando el favorito es el
visitante, `RL_AWAY` no es "+1.5" sino "−1.5" —ganar por 2 o más— pero el código
seguía devolviendo P(diff < 1.5), que es el evento del UNDERDOG (~65-70%), y lo
multiplicaba por el precio del FAVORITO. De ahí salían EV de tres dígitos.

DAÑO MEDIDO en el ledger antes del arreglo, sobre 281 picks resueltos:

    RUNLINE        105 picks   -51.55u   ROI  -49.09%
    TODO LO DEMÁS  176 picks    -9.98u   ROI   -5.67%

El 37% de los picks causaba el 84% de la pérdida. El EV declarado en runline tenía
mediana 20.6% y máximo 110.2%, con 38 de 105 por encima del 40%; el resto de los
mercados tenía mediana 8.6% y máximo 43.1%.

LA MATEMÁTICA. Con `h` = punto firmado del local, el umbral es `L = −h`:
el local cubre si `diff > L`, el visitante si `diff < L`. Son complementarios y con
medio-enteros no hay empate posible.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")
from core.value_detector import analyze_runline  # noqa: E402


MC = {"mean_home": 4.6, "mean_away": 4.3}


@pytest.fixture(scope="module")
def muestras():
    rng = np.random.default_rng(1)
    return rng.poisson(4.6, 200_000), rng.poisson(4.3, 200_000)


def _correr(muestras, punto, precio_local=2.00, precio_visita=1.80):
    h, a = muestras
    return analyze_runline(
        MC, precio_local, precio_visita, 1.5, h, a, 200_000,
        0.25, "multiplicative", False, 0.8, runline_home_point=punto,
    )


# ── el evento correcto de cada lado ───────────────────────────────────────────

def test_local_favorito_pide_ganar_por_dos(muestras):
    """Punto local −1.5: el local cubre sólo si gana por 2 o más — evento poco
    probable, así que su probabilidad tiene que ser BAJA."""
    r = _correr(muestras, -1.5)
    assert r["home"]["probability"] < 0.45
    assert r["away"]["probability"] > 0.55


def test_visitante_favorito_invierte_los_eventos(muestras):
    """Punto local +1.5: el local sólo necesita NO perder por 2 o más — evento
    probable. Éste es exactamente el caso que el código viejo calculaba al revés.
    """
    r = _correr(muestras, +1.5)
    assert r["home"]["probability"] > 0.55, "el local underdog cubre seguido"
    assert r["away"]["probability"] < 0.45, "el visitante favorito tiene que ganar por 2+"


def test_los_dos_lados_suman_uno(muestras):
    """Con líneas de medio-entero no hay empate: son complementarios exactos."""
    for punto in (-1.5, +1.5, -2.5, +2.5):
        r = _correr(muestras, punto)
        assert r["home"]["probability"] + r["away"]["probability"] == pytest.approx(1.0, abs=1e-9)


def test_el_signo_del_punto_cambia_la_probabilidad(muestras):
    """La prueba directa del bug: mismo juego, mismos precios, sólo cambia quién
    es favorito — y la probabilidad del local tiene que MOVERSE mucho.

    Ojo con la relación: P(diff > 1.5) y P(diff < −1.5) son dos COLAS, no
    complementarias entre sí (cada una lo es de su propio lado). Lo que sí vale
    siempre es que el local underdog cubre más seguido que el local favorito,
    porque le alcanza con no perder por 2 en vez de tener que ganar por 2.
    """
    fav_local = _correr(muestras, -1.5)["home"]["probability"]
    fav_visita = _correr(muestras, +1.5)["home"]["probability"]
    assert fav_visita > fav_local + 0.30, (
        f"el bug era justamente que no se movía: {fav_local:.4f} vs {fav_visita:.4f}")
    # Y cada una es la cola correcta de su propio umbral (la probabilidad
    # publicada viene redondeada a 4 decimales, de ahí la tolerancia).
    h, a = muestras
    diff = h - a
    assert fav_local == pytest.approx(float(np.mean(diff > 1.5)), abs=1e-4)
    assert fav_visita == pytest.approx(float(np.mean(diff > -1.5)), abs=1e-4)


# ── compatibilidad y honestidad del fallback ──────────────────────────────────

def test_sin_punto_reproduce_el_comportamiento_viejo(muestras):
    """Sin el dato no se puede saber quién es favorito. Se cae a asumir local
    favorito —que es el 72% de los juegos— y se avisa por log. El resultado tiene
    que ser idéntico al caso explícito de local favorito, para que este fallback
    no introduzca un tercer comportamiento."""
    h, a = muestras
    sin = analyze_runline(MC, 2.00, 1.80, 1.5, h, a, 200_000, 0.25,
                          "multiplicative", False, 0.8)
    con = _correr(muestras, -1.5)
    assert sin["home"]["probability"] == con["home"]["probability"]


def test_el_fallback_avisa(muestras, caplog):
    h, a = muestras
    with caplog.at_level("WARNING"):
        analyze_runline(MC, 2.00, 1.80, 1.5, h, a, 200_000, 0.25,
                        "multiplicative", False, 0.8)
    assert "punto firmado" in caplog.text


# ── las etiquetas no pueden volver a asumir el signo ──────────────────────────

@pytest.mark.parametrize("punto,etq_local,etq_visita", [
    (-1.5, "-1.5", "+1.5"),
    (+1.5, "+1.5", "-1.5"),
])
def test_las_etiquetas_llevan_el_signo_real(muestras, punto, etq_local, etq_visita):
    """Aguas abajo nadie debe tener que re-deducir quién era favorito a partir de
    una magnitud. El pick publicado dice el punto que realmente se compró."""
    r = _correr(muestras, punto)
    assert r["home_point_label"] == etq_local
    assert r["away_point_label"] == etq_visita


def test_el_umbral_queda_expuesto_para_auditar(muestras):
    r = _correr(muestras, +1.5)
    assert r["market_info"]["runline_threshold"] == -1.5
    assert r["market_info"]["runline_home_point"] == +1.5


# ── el punto firmado tiene que llegar desde el fetcher ────────────────────────

def test_el_fetcher_expone_el_punto_firmado():
    """La magnitud sola no alcanza, y el comentario que decía que sí era la causa
    raíz. Las DOS rutas del fetcher tienen que exponer el punto con signo."""
    fuente = open("odds_fetcher.py", encoding="utf-8").read()
    assert fuente.count('"runline_home_point"') >= 2
    assert fuente.count('"runline_away_point"') >= 2


def test_game_odds_lleva_el_punto():
    from core.value_detector import GameOdds
    assert "runline_home_point" in GameOdds.__dataclass_fields__
