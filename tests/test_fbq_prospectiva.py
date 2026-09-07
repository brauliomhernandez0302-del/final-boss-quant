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

# ── Compatibilidad de fuentes y ventana de aperturas (2026-09-06) ────────

def _almacen_aperturas(tmp_path, filas):
    from fbq.model import aperturas as AP
    db = tmp_path / "ap.db"
    with AP._conn(db) as c:
        c.executemany(
            """INSERT OR REPLACE INTO apertura
               (pitcher_id, game_pk, game_date, season, es_apertura, bf, k, bb)
               VALUES (?,?,?,?,?,?,?,?)""", filas)
    return db


def test_la_ventana_es_de_40_aperturas_y_el_corte_es_ESTRICTO(tmp_path, monkeypatch):
    """El preregistro fija 40 aperturas; ninguna de las dos fuentes del almacén
    PIT la ofrecía —las dos son acumuladas de temporada—, y por eso se
    reconstruye desde las aperturas individuales."""
    from fbq.model import aperturas as AP
    filas = [(1, 1000 + i, f"2025-0{4 + i // 28}-{1 + i % 28:02d}", 2025, 1, 25, 6, 2)
             for i in range(50)]
    filas.append((1, 9999, "2025-09-01", 2025, 1, 25, 99, 0))   # el día del juego
    db = _almacen_aperturas(tmp_path, filas)
    disp = {1000 + i: "2025-01-01T00:00:00+00:00" for i in range(50)}
    disp[9999] = "2025-09-01T23:00:00+00:00"      # termina esa misma noche
    monkeypatch.setattr(AP, "_DISPONIBLE", disp)
    k, bb, bf, n = AP.ventana(1, "2025-09-01T17:00:00+00:00", n=40, db=db)
    assert n == 40, "toma exactamente las 40 últimas"
    assert k == 40 * 6 and bf == 40 * 25, "el K=99 del propio día NO entra"


def test_los_tres_conteos_se_suman_sobre_las_MISMAS_aperturas(tmp_path, monkeypatch):
    """El defecto que obligó a reconstruir: numerador y denominador venían de
    fuentes que no contaban la misma población de turnos (129/300 de acuerdo)."""
    from fbq.model import aperturas as AP
    filas = [(7, 1, "2025-05-01", 2025, 1, 20, 5, 1),
             (7, 2, "2025-05-07", 2025, 1, 30, 9, 3),
             (7, 3, "2025-05-13", 2025, 0, 4, 1, 0)]   # relevo: no es apertura
    db = _almacen_aperturas(tmp_path, filas)
    monkeypatch.setattr(AP, "_DISPONIBLE",
                        {1: "2025-05-02T00:00:00+00:00", 2: "2025-05-08T00:00:00+00:00",
                         3: "2025-05-14T00:00:00+00:00"})
    k, bb, bf, n = AP.ventana(7, "2025-06-01T00:00:00+00:00", n=40, db=db)
    assert (k, bb, bf, n) == (14, 4, 50, 2), "el relevo queda fuera de las tres sumas"


def test_solo_se_cuentan_aperturas_no_relevos(tmp_path, monkeypatch):
    from fbq.model import aperturas as AP
    db = _almacen_aperturas(tmp_path, [(7, 1, "2025-05-01", 2025, 0, 4, 2, 0)])
    monkeypatch.setattr(AP, "_DISPONIBLE", {1: "2025-05-02T00:00:00+00:00"})
    assert AP.ventana(7, "2025-06-01T00:00:00+00:00", db=db) == (0, 0, 0, 0)


def test_un_abridor_bajo_el_minimo_de_BF_excluye_el_juego():
    from fbq.model.abridores import MIN_BF_ABRIDOR, Abridor, diferencia
    bueno = Abridor(1, k=40, bb=10, bf=MIN_BF_ABRIDOR, n_aperturas=6)
    flaco = Abridor(2, k=40, bb=10, bf=MIN_BF_ABRIDOR - 1, n_aperturas=5)
    assert diferencia(bueno, bueno)[0] == pytest.approx(0.0)
    assert diferencia(bueno, flaco) == (None, "abridor_con_muestra_insuficiente")
    assert diferencia(None, bueno) == (None, "sin_aperturas_previas")


def test_la_calidad_encoge_hacia_la_liga_por_bateadores_enfrentados():
    from fbq.model.abridores import K_BF, Abridor
    liga = 0.135
    poco = Abridor(1, k=5, bb=0, bf=10, n_aperturas=1)      # K−BB% = 0.50
    mucho = Abridor(2, k=2500, bb=0, bf=5000, n_aperturas=99)
    assert abs(poco.calidad(liga) - liga) < abs(mucho.calidad(liga) - liga)
    medio = Abridor(3, k=K_BF // 2, bb=0, bf=K_BF, n_aperturas=40)
    assert medio.calidad(liga) == pytest.approx((0.5 + liga) / 2, abs=1e-12)


def test_las_constantes_del_preregistro_de_v1_4_no_se_movieron():
    from fbq.model import abridores as AB
    assert (AB.VENTANA_ABRIDOR, AB.K_BF, AB.MIN_BF_ABRIDOR) == (40, 300, 150)


# ── Evidencia de emisión (2026-09-07) ────────────────────────────────────

def test_registrado_utc_lo_pone_el_motor_no_quien_llama(tmp_path):
    """Un corte anterior al partido no demuestra nada por sí solo: se puede
    declarar cualquier corte en cualquier momento. Lo que demuestra que la
    predicción existía antes del primer lanzamiento es el instante de
    ESCRITURA, y ése lo sella el almacén."""
    s = Prospectiva(tmp_path / "p.db")
    s.guardar([_fila(corte="2020-01-01T00:00:00+00:00",
                     commence_time="2020-01-02T00:00:00+00:00")])
    with s._conn() as c:
        r = c.execute("SELECT registrado_utc, origen FROM prediccion").fetchone()
    assert r["registrado_utc"] is not None and r["registrado_utc"].endswith("+00:00")
    # el partido de 2020 ya pasó: se escribió DESPUÉS, así que es reconstrucción
    assert r["origen"] == "reconstruccion"


def test_una_fila_escrita_antes_del_inicio_queda_como_prospectiva_verificada(tmp_path):
    from datetime import datetime, timedelta, timezone
    futuro = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    s = Prospectiva(tmp_path / "p.db")
    s.guardar([_fila(corte="2020-01-01T00:00:00+00:00", commence_time=futuro)])
    with s._conn() as c:
        assert c.execute("SELECT origen FROM prediccion").fetchone()[0] == "prospectiva_verificada"


def test_no_se_puede_declarar_prospectiva_sin_escribirla_antes(tmp_path):
    """El trigger impide la etiqueta, no la disciplina de quien la escribe."""
    s = Prospectiva(tmp_path / "p.db")
    with pytest.raises(sqlite3.IntegrityError, match="prospectiva_verificada"):
        s.guardar([_fila(corte="2020-01-01T00:00:00+00:00",
                         commence_time="2020-01-02T00:00:00+00:00",
                         origen="prospectiva_verificada")])


def test_dos_ajustes_distintos_conviven_sobre_el_mismo_juego_y_corte(tmp_path):
    """La llave única incluye `modelo_sha`. Sin eso, un ajuste nuevo colisionaba
    con el anterior y la regeneración reportaba 'guardadas: 0' — pasó."""
    s = Prospectiva(tmp_path / "p.db")
    assert s.guardar([_fila(modelo_sha="aaa")]) == 1
    assert s.guardar([_fila(modelo_sha="bbb")]) == 1
    assert s.guardar([_fila(modelo_sha="aaa")]) == 0
    assert s.resumen()["predicciones"] == 2


# ── La compuerta temporal sobre las aperturas ────────────────────────────

def test_una_apertura_que_no_habia_terminado_al_corte_NO_entra(tmp_path, monkeypatch):
    """La misma compuerta del resto del proyecto: fin medido + margen ≤ corte.
    Comparar sólo fechas era más débil — una apertura de anoche que terminó a
    las 02:10 no está disponible para un corte de la 01:00 del mismo día."""
    from fbq.model import aperturas as AP
    filas = [(1, 10, "2025-05-01", 2025, 1, 25, 7, 2),
             (1, 11, "2025-05-07", 2025, 1, 25, 7, 2)]
    db = _almacen_aperturas(tmp_path, filas)
    monkeypatch.setattr(AP, "_DISPONIBLE", {
        10: "2025-05-02T02:30:00+00:00",     # terminó de madrugada
        11: "2025-05-08T02:30:00+00:00"})
    # corte a la 01:00 del 2025-05-08: la del 05-07 todavía no estaba disponible
    k, bb, bf, n = AP.ventana(1, "2025-05-08T01:00:00+00:00", db=db)
    assert n == 1 and bf == 25
    # tres horas más tarde sí
    assert AP.ventana(1, "2025-05-08T04:00:00+00:00", db=db)[3] == 2


def test_una_apertura_sin_fin_fechable_no_entra(tmp_path, monkeypatch):
    from fbq.model import aperturas as AP
    db = _almacen_aperturas(tmp_path, [(1, 10, "2025-05-01", 2025, 1, 25, 7, 2)])
    monkeypatch.setattr(AP, "_DISPONIBLE", {10: None})
    assert AP.ventana(1, "2099-01-01T00:00:00+00:00", db=db) == (0, 0, 0, 0)


def test_la_compuerta_va_ANTES_de_recortar_a_40(tmp_path, monkeypatch):
    """Recortar y después filtrar dejaría ventanas de menos de 40 sin avisar."""
    from fbq.model import aperturas as AP
    filas = [(1, 100 + i, f"2025-06-{1 + i:02d}", 2025, 1, 25, 6, 2) for i in range(45)]
    db = _almacen_aperturas(tmp_path, filas)
    # las 5 más recientes no están disponibles al corte
    disp = {100 + i: ("2099-01-01T00:00:00+00:00" if i >= 40
                      else "2025-01-01T00:00:00+00:00") for i in range(45)}
    monkeypatch.setattr(AP, "_DISPONIBLE", disp)
    assert AP.ventana(1, "2025-08-01T00:00:00+00:00", n=40, db=db)[3] == 40
