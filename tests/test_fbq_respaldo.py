"""Respaldos consistentes, versionados y fuera del proyecto.

Este proyecto perdió datos irrecuperables dos veces por el mismo mecanismo: los
triggers de append-only protegen las FILAS de un UPDATE o un DELETE, y no
protegen el ARCHIVO. Estos tests fijan las tres decisiones que salieron de ahí.
"""

from __future__ import annotations

import sqlite3

import pytest

from fbq.respaldo import respaldar, restaurar, verificar_restauracion


def _base(tmp_path, nombre="prospectiva.db", filas=3):
    origen = tmp_path / "data"
    origen.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(origen / nombre)
    con.execute("CREATE TABLE prediccion (game_pk INT, version TEXT, corte TEXT, "
                "modelo_sha TEXT, p_home REAL, generado_utc TEXT, "
                "registrado_utc TEXT, origen TEXT)")
    con.executemany("INSERT INTO prediccion VALUES (?,?,?,?,?,?,?,?)",
                    [(i, "v1.4", "2026-09-07T02:00:00+00:00", "abc123",
                      0.5 + i / 100, "2026-09-07T02:00:01+00:00",
                      "2026-09-07T02:00:02+00:00", "prospectiva_verificada")
                     for i in range(filas)])
    con.commit(); con.close()
    return origen


def test_el_respaldo_usa_la_API_de_sqlite_y_verifica_integridad(tmp_path):
    """`cp` sobre una base con un escritor en curso produce un archivo corrupto
    que se ve sano hasta que alguien lo abre."""
    origen = _base(tmp_path)
    destino = tmp_path / "fuera"
    entradas = respaldar(["prospectiva.db"], motivo="test",
                         origen=origen, destino=destino)
    assert len(entradas) == 1
    e = entradas[0]
    assert e["integridad"] == "ok" and e["coincide"] is True
    assert e["filas_origen"] == e["filas_copia"] == {"prediccion": 3}


def test_los_respaldos_se_versionan_y_no_se_pisan(tmp_path):
    origen = _base(tmp_path)
    destino = tmp_path / "fuera"
    respaldar(["prospectiva.db"], origen=origen, destino=destino, motivo="uno")
    import time; time.sleep(1.1)          # el sello tiene resolución de segundo
    respaldar(["prospectiva.db"], origen=origen, destino=destino, motivo="dos")
    assert len(list(destino.glob("prospectiva.*.db"))) == 2
    lineas = (destino / "manifiesto.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 2


def test_el_destino_esta_FUERA_del_arbol_del_proyecto():
    """Un respaldo que vive en `data/` desaparece con el mismo `rm -rf` que
    borra lo que respalda."""
    from fbq.respaldo import DESTINO, RAIZ
    assert RAIZ not in DESTINO.parents and DESTINO != RAIZ


def test_restaurar_sobre_produccion_esta_PROHIBIDO(tmp_path):
    """Un ensayo que puede destruir lo que verifica no es un ensayo."""
    from fbq.respaldo import RAIZ
    origen = _base(tmp_path)
    destino = tmp_path / "fuera"
    respaldar(["prospectiva.db"], origen=origen, destino=destino)
    with pytest.raises(ValueError, match="producción"):
        restaurar("prospectiva.db", RAIZ / "data" / "prospectiva.db", destino=destino)


def test_la_restauracion_conserva_probabilidades_sellos_y_SHA(tmp_path):
    """No alcanza con que el archivo abra."""
    origen = _base(tmp_path, filas=5)
    destino = tmp_path / "fuera"
    respaldar(["prospectiva.db"], origen=origen, destino=destino)
    inf = verificar_restauracion("prospectiva.db", destino=destino, origen=origen)
    assert inf["integridad"] == "ok"
    assert inf["filas_coinciden"] and inf["esquema_coincide"]
    assert inf["predicciones_identicas"] is True
    assert inf["n_predicciones"] == 5
    assert inf["shas"] == ["abc123"]
    assert inf["origenes"] == ["prospectiva_verificada"]
    assert inf["con_sello_de_emision"] == 5


def test_una_copia_alterada_se_detecta(tmp_path):
    """Si el respaldo no coincidiera, la verificación tiene que decirlo."""
    origen = _base(tmp_path, filas=4)
    destino = tmp_path / "fuera"
    respaldar(["prospectiva.db"], origen=origen, destino=destino)
    copia = sorted(destino.glob("prospectiva.*.db"))[-1]
    con = sqlite3.connect(copia)
    con.execute("UPDATE prediccion SET p_home = 0.99 WHERE game_pk = 1")
    con.commit(); con.close()
    inf = verificar_restauracion("prospectiva.db", destino=destino, origen=origen)
    assert inf["filas_coinciden"] is True         # el conteo no cambia
    assert inf["predicciones_identicas"] is False  # el contenido sí
