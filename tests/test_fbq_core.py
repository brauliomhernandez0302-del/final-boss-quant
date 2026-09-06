"""fbq.core — el contrato de tiempo e identidad (paso 0).

Cada test acá fija una regla que en el proyecto anterior vivía repartida en los
llamadores y falló al menos una vez por eso.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbq.core.clock import (cutoff_para_dia, dia_oficial, es_anterior,
                            instantanea_vigente)
from fbq.core.identity import canonico, elegir_unico, mismo_equipo


# ── El día del juego ─────────────────────────────────────────────────────


def test_el_dia_del_juego_es_el_oficial_no_el_utc():
    """Un nocturno de la costa oeste empieza el 18 y su timestamp UTC dice 19.

    Sobre datos reales de 2024-2026 esto pasa en el 22-24% de los juegos.
    Caso concreto: game_pk=745199, oficial 2024-09-18, UTC 2024-09-19.
    """
    juego = {"official_date": "2024-09-18", "game_date": "2024-09-19T02:10:00Z"}
    assert dia_oficial(juego) == "2024-09-18"


def test_sin_dia_oficial_cae_al_timestamp_pero_no_inventa():
    assert dia_oficial({"game_date": "2024-09-19T02:10:00Z"}) == "2024-09-19"
    with pytest.raises(KeyError):
        dia_oficial({"home_team": "Dodgers"})


# ── El corte ─────────────────────────────────────────────────────────────


def test_el_corte_es_el_fin_del_dia_anterior():
    """Predecir un juego del día D usa datos hasta el fin de D−1.

    Un solo sitio lo calcula. Antes vivía repetido en cinco funciones que
    restaban un segundo a mano, y una de las cinco no lo restaba: devolvía el
    fin del día DEL JUEGO.
    """
    assert cutoff_para_dia("2024-09-18").startswith("2024-09-17T23:59:59")


def test_el_corte_cruza_el_cambio_de_mes():
    assert cutoff_para_dia("2024-10-01").startswith("2024-09-30T23:59:59")


# ── La comparación estricta ──────────────────────────────────────────────


def test_la_comparacion_contra_el_corte_es_estricta():
    """Un `<=` sobre el corte incluye el propio instante. Ésa es la definición
    del leak que la Fase 2B tuvo que remediar."""
    corte = "2024-09-17T23:59:59+00:00"
    assert es_anterior("2024-09-17T23:59:58+00:00", corte)
    assert not es_anterior(corte, corte)


def test_compara_instantes_y_no_cadenas():
    """`2024-05-01` y `2024-05-01T00:00:00+00:00` son el mismo instante y como
    TEXTO ordenan distinto. El cache anterior guardaba dos convenciones a la vez
    (`T00:00:00` y `T23:59:59`) y sólo funcionaba por aritmética afortunada."""
    assert not es_anterior("2024-05-01", "2024-05-01T00:00:00+00:00")
    assert es_anterior("2024-05-01", "2024-05-01T00:00:01+00:00")


def test_la_instantanea_vigente_nunca_devuelve_una_posterior():
    """Que no haya dato es un estado legítimo. Devolver el más cercano
    'para no quedarse sin nada' es el leak."""
    candidatas = [("2024-09-16T23:59:59+00:00", "vieja"),
                  ("2024-09-17T23:59:59+00:00", "justa"),
                  ("2024-09-18T23:59:59+00:00", "futura")]
    corte = cutoff_para_dia("2024-09-18")           # 2024-09-17T23:59:59
    assert instantanea_vigente(candidatas, corte) == "vieja"
    assert instantanea_vigente(candidatas[:1], "2024-01-01") is None


# ── Identidad ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("nombre", [
    "Athletics", "Oakland Athletics", "Sacramento Athletics", "Las Vegas Athletics",
])
def test_los_alias_de_equipo_canonizan_en_ambos_sentidos(nombre):
    """El equipo cambió de ciudad dos veces. Un mapa de un solo sentido lleva
    un nombre LEJOS del otro en cuanto una fuente cambia — eso costó 276 juegos
    sin precio, atribuidos a 'team name mismatch' sin que nadie mirara cuál."""
    assert canonico(nombre) == "athletics"
    assert mismo_equipo(nombre, "Athletics")
    assert mismo_equipo("Oakland Athletics", nombre)


def test_equipos_distintos_no_se_acercan():
    assert not mismo_equipo("Los Angeles Angels", "Los Angeles Dodgers")
    assert not mismo_equipo("Chicago Cubs", "Chicago White Sox")


def _juego(inicio):
    return {"gameDate": inicio}


def test_elige_el_candidato_inequivoco():
    elegido, motivo = elegir_unico(
        [_juego("2026-08-04T23:05:00Z"), _juego("2026-08-05T02:40:00Z")],
        "2026-08-04T23:05:00Z", inicio_de=lambda g: g["gameDate"])
    assert motivo == "ok"
    assert elegido["gameDate"] == "2026-08-04T23:05:00Z"


def test_se_abstiene_ante_un_doubleheader_tradicional():
    """Los dos juegos separados por 5 minutos: quedarse con el que está un
    minuto más cerca es quedarse con el precio del otro partido, sin rastro."""
    _, motivo = elegir_unico(
        [_juego("2026-08-04T23:05:00Z"), _juego("2026-08-04T23:10:00Z")],
        "2026-08-04T23:06:00Z", inicio_de=lambda g: g["gameDate"])
    assert motivo == "ambiguo"


def test_un_doubleheader_partido_si_se_resuelve():
    """275-405 min de separación real: el correcto gana por mucho más que el
    margen y se empareja bien. Los dos regímenes exigen respuestas opuestas."""
    elegido, motivo = elegir_unico(
        [_juego("2026-08-04T17:10:00Z"), _juego("2026-08-04T23:10:00Z")],
        "2026-08-04T23:05:00Z", inicio_de=lambda g: g["gameDate"])
    assert motivo == "ok"
    assert elegido["gameDate"] == "2026-08-04T23:10:00Z"


def test_los_tres_motivos_de_no_elegir_son_distinguibles():
    """Significan cosas distintas para quien llama: reintentar, ampliar la
    búsqueda, o no insistir nunca."""
    assert elegir_unico([], "2026-08-04T23:05:00Z",
                        inicio_de=lambda g: g["gameDate"])[1] == "sin_candidato"
    assert elegir_unico([_juego("x")], "no-es-fecha",
                        inicio_de=lambda g: g["gameDate"])[1] == "sin_inicio"
    assert elegir_unico([_juego("2026-01-01T00:00:00Z")], "2026-08-04T23:05:00Z",
                        inicio_de=lambda g: g["gameDate"])[1] == "sin_candidato"


# ── Normalización de instantes (2026-09-06) ──────────────────────────────

def test_normalizar_utc_da_una_sola_forma_canonica():
    """El almacén compara instantes como CADENAS dentro de SQL. Dos formas del
    mismo instante ordenan distinto, así que la frontera tiene que dejar una."""
    from fbq.core.clock import normalizar_utc
    esperado = "2024-04-24T22:05:00+00:00"
    assert normalizar_utc("2024-04-24T22:05:00Z") == esperado
    assert normalizar_utc("2024-04-24T22:05:00+00:00") == esperado
    assert normalizar_utc("2024-04-24T18:05:00-04:00") == esperado


def test_normalizar_utc_rechaza_una_fecha_sin_hora():
    """Completarla con medianoche convertiría "no sé a qué hora empezó" en
    "empezó a la medianoche", que es falso y encima creíble. Es exactamente lo
    que hacía `importar_historico` al tomar `game_outcomes.game_date`."""
    import pytest
    from fbq.core.clock import normalizar_utc, tiene_hora
    assert tiene_hora("2024-04-24T22:05:00Z") is True
    assert tiene_hora("2024-04-24") is False
    with pytest.raises(ValueError, match="fecha sin hora"):
        normalizar_utc("2024-04-24")


def test_la_forma_canonica_hace_correcta_la_comparacion_de_cadenas():
    """La prueba de que la normalización sirve para lo que existe: el filtro
    pre-juego del almacén es `captured_at < commence_time` en SQL puro."""
    from fbq.core.clock import normalizar_utc
    captura = normalizar_utc("2024-04-24T17:00:00Z")
    inicio = normalizar_utc("2024-04-24T22:05:00Z")
    assert captura < inicio                      # pre-juego
    assert not (normalizar_utc("2024-04-24T23:00:00Z") < inicio)   # en vivo


def test_estados_finales_vive_en_core_y_es_la_misma_en_todo_el_sistema():
    """La regla que distingue el cascarón pospuesto del partido jugado la usan
    `results/` (para no escribir un cascarón como hecho) y `market/` (para no
    fechar una cotización con la hora del partido suspendido). Duplicarla sería
    garantizar que algún día digan cosas distintas."""
    from fbq.core.identity import ESTADOS_FINALES
    from fbq.results.store import ESTADOS_FINALES as desde_results
    from fbq.results.fetch import ESTADOS_FINALES as desde_fetch
    assert ESTADOS_FINALES is desde_results is desde_fetch
    assert "Postponed" not in ESTADOS_FINALES
