"""El detector de campos mudos dispara cuando un campo nunca llega.

`scripts/inputs_coverage.py` existe por el bug de `weather_source` del
2026-08-02: la instantánea de entradas leía `game_data['weather']['source']`,
una clave que el fetcher no emite, así que el campo era None en 52 de 52 picks
mientras el clima SÍ llegaba al motor y SÍ movía λ.

La firma es visible sin saber nada del dominio: un campo en 0% mientras sus
vecinos están al 100%. Estos tests fijan que el script la reconozca — un
detector que no detecta es peor que ninguno, porque da falsa tranquilidad.
"""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ / "scripts" / "inputs_coverage.py"


def _db_con_picks(tmp_path: Path, inputs_por_pick: list[dict]) -> Path:
    db = tmp_path / "track_record.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE picks (game_date TEXT, pipeline_json TEXT)")
    for i, inputs in enumerate(inputs_por_pick):
        con.execute(
            "INSERT INTO picks VALUES (?, ?)",
            (f"2026-08-0{(i % 4) + 1}", json.dumps({"inputs": inputs})),
        )
    con.commit()
    con.close()
    return db


def _correr(db: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(db)],
        capture_output=True, text=True, cwd=RAIZ,
    )


def test_dispara_con_un_campo_siempre_vacio(tmp_path):
    """El caso real: `weather_source` None en todos, el resto con valor."""
    picks = [
        {"pitchers_valid": True, "home_days_rest": 2, "weather_source": None}
        for _ in range(20)
    ]
    r = _correr(_db_con_picks(tmp_path, picks))
    assert r.returncode == 1, "un campo en 0% tiene que salir con código de error"
    assert "weather_source" in r.stdout
    assert "SIEMPRE VACÍOS" in r.stdout


def test_no_dispara_si_todos_traen_algo(tmp_path):
    picks = [
        {"pitchers_valid": True, "home_days_rest": 2, "weather_source": "live"}
        for _ in range(20)
    ]
    r = _correr(_db_con_picks(tmp_path, picks))
    assert r.returncode == 0
    assert "Ningún campo en 0%" in r.stdout


def test_cobertura_baja_no_es_cobertura_cero(tmp_path):
    """`lineup_confirmed` bajo es el mercado, no un fallo de plomería: las
    alineaciones se publican después de que corre el cron. Un solo pick con
    valor ya saca al campo de la alarma."""
    picks = [{"lineup_confirmed": None} for _ in range(49)] + [{"lineup_confirmed": True}]
    r = _correr(_db_con_picks(tmp_path, picks))
    assert r.returncode == 0, "2% de cobertura no es 0% — no debe alarmar"


def test_false_y_cero_cuentan_como_valor(tmp_path):
    """La ausencia es None. `False` y `0.0` son respuestas legítimas: si
    contaran como vacío, `enrichment_failed` (False en el 100% de los picks
    sanos) y `injured_pa_share=0.0` darían una alarma permanente."""
    picks = [{"enrichment_failed": False, "injured_pa_share": 0.0} for _ in range(10)]
    r = _correr(_db_con_picks(tmp_path, picks))
    assert r.returncode == 0, "False y 0.0 son valores, no ausencias"


def test_sin_bloque_inputs_no_falla(tmp_path):
    """Los picks anteriores al 2026-08-01 no tienen el bloque. No hay nada que
    medir, y eso no es un error."""
    db = tmp_path / "track_record.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE picks (game_date TEXT, pipeline_json TEXT)")
    con.execute("INSERT INTO picks VALUES ('2026-07-20', ?)",
                (json.dumps({"lambdas": {}, "bet": {}}),))
    con.commit(); con.close()
    r = _correr(db)
    assert r.returncode == 0
    assert "127bac6" in r.stdout


def test_db_ausente_sale_con_codigo_2(tmp_path):
    r = _correr(tmp_path / "no-existe.db")
    assert r.returncode == 2
