"""Captura activada y predicciones pareadas guardadas antes del partido."""

from __future__ import annotations

import json
import sqlite3

import pytest

from fbq.anuncios.capturar import Ocupado, _Cerrojo
from fbq.model.prospectiva import Prospectiva, aplicar


# ── El cerrojo ───────────────────────────────────────────────────────────

def test_dos_capturas_simultaneas_no_se_pisan(tmp_path):
    """Dos corridas a la vez insertarían la misma barrida dos veces y cada una
    vería un estado distinto al deduplicar por cambio."""
    ruta = tmp_path / "c.lock"
    with _Cerrojo(ruta):
        with pytest.raises(Ocupado):
            with _Cerrojo(ruta):
                pass


def test_el_cerrojo_se_suelta_aunque_el_bloque_falle(tmp_path):
    """Un cerrojo por existencia de archivo deja el sistema trabado si el
    proceso muere; `flock` lo suelta el kernel pase lo que pase."""
    ruta = tmp_path / "c.lock"
    with pytest.raises(RuntimeError):
        with _Cerrojo(ruta):
            raise RuntimeError("boom")
    with _Cerrojo(ruta):
        pass   # se puede volver a tomar


# ── El almacén de predicciones ───────────────────────────────────────────

def _fila(**kw):
    base = {"corte": "2026-09-06T18:00:00+00:00", "game_pk": 1,
            "official_date": "2026-09-06", "commence_time": "2026-09-06T23:05:00+00:00",
            "home_team": "L", "away_team": "V", "version": "v1.2", "p_home": 0.55,
            "variables_json": "{}", "cohorte": "prospectiva", "modelo_sha": "abc"}
    base.update(kw)
    return base


def test_una_prediccion_posterior_al_inicio_se_rechaza(tmp_path):
    """No es una predicción: es una observación disfrazada."""
    s = Prospectiva(tmp_path / "p.db")
    with pytest.raises(sqlite3.IntegrityError, match="posterior al primer lanzamiento"):
        s.guardar([_fila(corte="2026-09-07T00:00:00+00:00")])


def test_es_append_only(tmp_path):
    s = Prospectiva(tmp_path / "p.db")
    s.guardar([_fila()])
    with s._conn() as c:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            c.execute("UPDATE prediccion SET p_home = 0.9")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            c.execute("DELETE FROM prediccion")


def test_repetir_la_corrida_no_duplica(tmp_path):
    """El cron dispara cada hora; repetir tiene que salir en cero."""
    s = Prospectiva(tmp_path / "p.db")
    assert s.guardar([_fila()]) == 1
    assert s.guardar([_fila()]) == 0
    assert s.resumen()["predicciones"] == 1


def test_las_dos_versiones_quedan_pareadas_al_mismo_corte(tmp_path):
    s = Prospectiva(tmp_path / "p.db")
    s.guardar([_fila(version="v1.2"), _fila(version="v1.4", p_home=0.58)])
    r = s.resumen()
    assert r["por_version"] == {"v1.2": 1, "v1.4": 1}
    assert r["pareados"] == 1


def test_la_cohorte_historica_se_guarda_APARTE(tmp_path):
    """Mezclarla con la prospectiva sería juntar dos muestras distintas: otro
    capturador de anuncios, otro corte, otra fuente de estadísticas."""
    s = Prospectiva(tmp_path / "p.db")
    s.guardar([_fila(), _fila(game_pk=2, cohorte="historica_42")])
    assert s.resumen()["por_cohorte"] == {"prospectiva": 1, "historica_42": 1}


# ── El ajuste congelado ──────────────────────────────────────────────────

def test_aplicar_falla_si_falta_una_variable(tmp_path):
    """Rellenarla con la media convertiría una predicción imposible en una
    plausible."""
    modelo = {"version": "v1.4", "variables": ["a", "b"], "beta": [0.1, 0.2, 0.3],
              "mu": [0.0, 0.0], "sd": [1.0, 1.0]}
    assert 0.0 < aplicar(modelo, {"a": 1.0, "b": 2.0}) < 1.0
    with pytest.raises(KeyError, match="faltan variables"):
        aplicar(modelo, {"a": 1.0})


def test_el_congelado_publicado_tiene_las_dos_versiones_y_su_huella():
    """Un ajuste sin huella no se puede verificar contra la predicción que dice
    haber producido."""
    from pathlib import Path
    ruta = Path(__file__).parent.parent / "docs" / "modelos_congelados_2026-09-06.json"
    if not ruta.exists():
        pytest.skip("todavía no se congeló")
    d = json.loads(ruta.read_text(encoding="utf-8"))
    assert set(d["modelos"]) == {"v1.2", "v1.4"}
    assert d["entrenado_con"] == [2024, 2025]
    for v, m in d["modelos"].items():
        assert m["sha"] and len(m["beta"]) == len(m["variables"]) + 1
    assert "dif_calidad_abridor" in d["modelos"]["v1.4"]["variables"]
    assert "dif_calidad_abridor" not in d["modelos"]["v1.2"]["variables"]
    assert "concesion_declarada" in d and "7,1%" in d["concesion_declarada"]


# ── La variable de abridor ───────────────────────────────────────────────

def test_la_calidad_se_encoge_hacia_la_liga_por_bateadores_enfrentados():
    from fbq.model.abridores import K_BF, Abridor
    liga = 0.135
    poco = Abridor(1, k_pct=0.40, bb_pct=0.05, bf=10, fuente="t")
    mucho = Abridor(2, k_pct=0.40, bb_pct=0.05, bf=5000, fuente="t")
    assert abs(poco.calidad(liga) - liga) < abs(mucho.calidad(liga) - liga)
    medio = Abridor(3, k_pct=0.40, bb_pct=0.05, bf=K_BF, fuente="t")
    assert medio.calidad(liga) == pytest.approx((0.35 + liga) / 2, abs=1e-12)


def test_un_abridor_con_poca_muestra_EXCLUYE_el_juego(tmp_path):
    from fbq.model.abridores import MIN_BF_ABRIDOR, Abridor, diferencia
    bueno = Abridor(1, 0.25, 0.07, MIN_BF_ABRIDOR, "t")
    flaco = Abridor(2, 0.25, 0.07, MIN_BF_ABRIDOR - 1, "t")
    assert diferencia(bueno, bueno)[0] is not None
    assert diferencia(bueno, flaco) == (None, "abridor_con_muestra_insuficiente")
    assert diferencia(None, bueno) == (None, "sin_estadistica_de_abridor")
