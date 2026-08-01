"""Cada pick guarda lo que el modelo VIO, no sólo lo que concluyó.

`pipeline_json` —el snapshot de auditoría de cada pick— guardaba tres cosas:
`lambdas`, `mc_probs` y `bet`. Los RESULTADOS. Nunca las ENTRADAS.

El costo se descubrió al chocarlo (2026-08-01): con una anticipación mediana de
26.6 horas, el abridor anunciado cambia a veces entre que se publica el pick y
que se juega — y cuando pasa, todo el análisis del pitcher de ese pick fue sobre
la persona equivocada. Se intentó medir cuántas veces ocurre y **fue imposible**:
no quedaba registro de a qué abridor habíamos visto, y la API de MLB sólo
devuelve el probable ACTUAL, no el de hace 26 horas.

No es un dato de entrada que falte al modelo: es que no se podían auditar las
propias decisiones. Para un producto cuyo producto ES el track record, eso pesa
más que una feature.
"""
import json
import sys

import pytest

sys.path.insert(0, ".")


CLAVES_MINIMAS = {
    # identidad y procedencia de los abridores — el motivo principal
    "home_pitcher", "home_pitcher_id", "home_pitcher_source",
    "away_pitcher", "away_pitcher_id", "away_pitcher_source",
    # banderas de calidad del dato que el pipeline ya calcula y descartaba
    "pitchers_valid", "lineup_confirmed", "home_lhb_source",
    "home_days_rest", "away_days_rest", "travel_source_away",
    "enrichment_failed", "official_date", "doubleheader",
}


def test_el_pipeline_expone_la_instantanea():
    """`run_module` tiene que producirla; el publisher sólo la copia."""
    fuente = open("modules/baseball_module/core/run_module.py", encoding="utf-8").read()
    assert "results['inputs_snapshot']" in fuente
    for clave in CLAVES_MINIMAS:
        assert f"'{clave}'" in fuente, f"la instantánea perdió {clave!r}"


def test_el_publisher_la_guarda_en_el_snapshot():
    fuente = open("track_record/publisher.py", encoding="utf-8").read()
    assert '"inputs": result.get("inputs_snapshot"' in fuente
    assert '"market": {' in fuente, "también el precio y el libro que se vieron"


def test_la_procedencia_del_abridor_queda_registrada():
    """No alcanza con el nombre: `home_pitcher_source` distingue un abridor con
    temporada de MLB de uno resuelto por AAA o por ERA del staff. Es la misma
    jerarquía del paso 5, ahora visible pick por pick en vez de sólo en el log.
    """
    fuente = open("modules/baseball_module/core/run_module.py", encoding="utf-8").read()
    i = fuente.index("results['inputs_snapshot']")
    bloque = fuente[i:i + 1800]
    assert "home_pitcher_source" in bloque and "away_pitcher_source" in bloque


def test_registra_de_donde_salio_la_lateralidad():
    """`home_lhb_source` dice si el LHB% vino del lineup confirmado, de la
    estimación del equipo, o de la media de liga — los tres escalones de la
    cadena del paso 6. Sin esto no se puede distinguir un pick hecho con
    alineación real de uno hecho con una constante."""
    fuente = open("modules/baseball_module/core/run_module.py", encoding="utf-8").read()
    i = fuente.index("results['inputs_snapshot']")
    bloque = fuente[i:i + 1800]
    for escalon in ("'lineup'", "'equipo'", "'liga'"):
        assert escalon in bloque


def test_el_snapshot_sigue_siendo_serializable():
    """El `pipeline_json` ya rompió la publicación una vez por un tipo de numpy
    (`kelly_floor_applied` como np.bool_, f4a4bf4): tres corridas de cron
    hicieron todo el trabajo y murieron al guardar. Agregar campos nuevos es
    exactamente el momento de que vuelva a pasar."""
    from track_record.publisher import _json_default
    ejemplo = {
        "inputs": {
            "home_pitcher": "X", "home_pitcher_id": 1, "home_pitcher_source": "home_split",
            "pitchers_valid": True, "lineup_confirmed": False, "home_days_rest": 2,
            "weather_source": None, "doubleheader": "N",
        },
        "market": {"ml_home": 1.9, "pin_home": None},
    }
    json.dumps(ejemplo, default=_json_default)


def test_los_resultados_siguen_estando():
    """La instantánea se AGREGA, no reemplaza — lo que ya se guardaba tiene que
    seguir ahí o se rompe la auditoría de los picks viejos."""
    fuente = open("track_record/publisher.py", encoding="utf-8").read()
    i = fuente.index("pipeline_snap = {")
    bloque = fuente[i:i + 1600]
    for previo in ('"lambdas"', '"mc_probs"', '"bet"'):
        assert previo in bloque
