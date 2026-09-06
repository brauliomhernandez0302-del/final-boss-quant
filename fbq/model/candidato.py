"""fbq/model/candidato.py — arma las filas, entrena expansivo y predice.

Todo lo que decide está en `docs/PREREGISTRO_MODELO_V1_2026-09-06.md`,
commiteado antes de medir. Este archivo lo ejecuta; no lo reinterpreta.

⚠️ **La evaluación de 2025 y 2026 es HISTÓRICA, no fuera de muestra.** Esos años
ya fueron explorados exhaustivamente por este proyecto —siete baselines, una
auditoría de dieciséis informes, nueve motores medidos sobre ellos—. El diseño
temporal impide que el MODELO vea el futuro; no impide que lo haya visto quien
eligió las variables. La única prueba limpia posible es hacia adelante.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fbq.evaluator.frame import (CONSENSO, DB_MERCADO, DB_RESULTADOS,
                                 _devig_multiplicativo)
from fbq.model import features as F
from fbq.model.detector import verificar_plausibilidad
from fbq.model.logistica import LAMBDA_L2, Logistica, ajustar
from fbq.core.clock import normalizar_utc
from fbq.model.pit import VentanaPIT, cargar_partidos, _inicios
from fbq.results import fines as _fines

CONFIG = {
    "preregistro": "docs/PREREGISTRO_MODELO_V1_2026-09-06.md",
    "variables": list(F.NOMBRES),
    "ventana": F.VENTANA,
    "min_juegos_previos": F.MIN_JUEGOS_PREVIOS,
    "k_regresion": F.K_REGRESION,
    "exponente_pitagorico": F.EXPONENTE_PITAGORICO,
    "tope_descanso": F.TOPE_DESCANSO,
    "lambda_l2": LAMBDA_L2,
    "corte": "captured_at del precio de referencia (último par de Pinnacle pre-juego)",
    "evaluacion": "HISTÓRICA — 2025 y 2026 ya fueron explorados por el proyecto",
}


@dataclass
class Fila:
    game_pk: int
    official_date: str
    season: int
    home_team: str
    away_team: str
    corte: str
    inicio_utc: str
    y: int
    p_mercado: float
    x: Tuple[float, ...]
    extra: Dict[str, Any] = field(default_factory=dict)


def _precios_de_referencia(
    seasons: Sequence[int], *, db_mercado: Path, db_resultados: Path,
    book: str = "pinnacle", sport_key: str = "baseball_mlb",
) -> Dict[int, Dict[str, Any]]:
    """`{game_pk: {corte, inicio, p_mercado}}` — el último par PRE-JUEGO.

    Mismo criterio y mismo devig que `evaluator.frame.load_frame_propio`: el
    candidato tiene que medirse contra el mismo nulo que ya está publicado, no
    contra una variante.
    """
    marcas = ",".join("?" * len(seasons))
    con = sqlite3.connect(f"file:{db_resultados}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        con.execute("ATTACH DATABASE ? AS mkt", (f"file:{db_mercado}?mode=ro",))
        filas = con.execute(
            f"""
            WITH ultima AS (
                SELECT s.event_id, s.side, s.price_dec, s.captured_at,
                       s.commence_time,
                       ROW_NUMBER() OVER (PARTITION BY s.event_id, s.side
                                          ORDER BY s.captured_at DESC, s.id DESC) AS rn
                FROM mkt.odds_snapshot s
                WHERE s.sport_key = ? AND s.market = 'h2h' AND s.book = ?
                  AND s.captured_at < s.commence_time
            )
            SELECT r.game_pk, r.season,
                   MAX(CASE WHEN u.side='home' THEN u.price_dec END)   AS pin_home,
                   MAX(CASE WHEN u.side='away' THEN u.price_dec END)   AS pin_away,
                   MAX(u.captured_at)                                  AS corte,
                   MAX(u.commence_time)                                AS inicio
            FROM resultado r
            JOIN mkt.event_link l ON l.game_pk = r.game_pk
            JOIN ultima u ON u.event_id = l.event_id AND u.rn = 1
            WHERE r.season IN ({marcas})
            GROUP BY r.game_pk
            """, (sport_key, book, *seasons)).fetchall()
    finally:
        con.close()

    out = {}
    for r in filas:
        if not (r["pin_home"] and r["pin_away"]):
            continue
        p, _ = _devig_multiplicativo(1.0 * r["pin_home"], 1.0 * r["pin_away"])
        out[int(r["game_pk"])] = {"corte": r["corte"], "inicio": r["inicio"],
                                  "p_mercado": float(p)}
    return out


def construir(
    seasons: Sequence[int] = (2024, 2025, 2026),
    *,
    db_mercado: Path = DB_MERCADO,
    db_resultados: Path = DB_RESULTADOS,
    constructor=None,
    precios: Optional[Dict[int, Dict[str, Any]]] = None,
    cache_fines: Optional[Path] = None,
) -> Tuple[List[Fila], Counter]:
    """Las filas predecibles y el conteo de exclusiones por motivo.

    `constructor` sólo se cambia en los tests de fuga: es el punto por el que se
    inyecta una variable envenenada para comprobar que los detectores la
    rechazan.
    """
    partidos = cargar_partidos(seasons, db_resultados=db_resultados,
                               cache_fines=cache_fines)
    ventana = VentanaPIT(partidos)
    # Los precios no dependen de los marcadores, así que el test de invariancia
    # los calcula UNA vez y los reusa en cada reconstrucción. Es la diferencia
    # entre dos minutos y un segundo por perturbación.
    if precios is None:
        precios = _precios_de_referencia(seasons, db_mercado=db_mercado,
                                         db_resultados=db_resultados)
    construir_fila = constructor or F.construir_fila
    indice_liga = F.IndiceLiga(partidos)
    # El inicio de cada partido, para el componente recuperado.
    #
    # La fuente primaria es la caché de FINES, que trae `inicio` para los 7.664
    # partidos de `results.db`. La caché del schedule se armó con el rango de
    # `historical_odds`, que termina el 2026-08-03, así que le faltan los 25
    # juegos de la captura propia en vivo — usarla sola los excluía.
    #
    # Para un suspendido se toma la REANUDACIÓN, igual que en todo el resto del
    # contrato temporal: un partido no empezó por última vez antes de reanudarse.
    inicios: Dict[int, str] = {pk: max(v) for pk, v in _inicios().items()}
    for pk, m in _fines.cargar().items():
        cuando = m.get("reanudacion") or m.get("inicio")
        if cuando:
            inicios[int(pk)] = normalizar_utc(cuando)

    filas: List[Fila] = []
    excl: Counter = Counter()
    for juego in partidos:
        ref = precios.get(juego.game_pk)
        if ref is None:
            excl["sin_precio_pinnacle_pre_juego"] += 1
            continue
        corte = ref["corte"]
        v = construir_fila(ventana, partidos, juego, corte, indice_liga, inicios)
        if not v.get("ok"):
            excl[str(v.get("motivo", "desconocido"))] += 1
            continue
        filas.append(Fila(
            game_pk=juego.game_pk, official_date=juego.official_date,
            season=juego.season, home_team=juego.home_team,
            away_team=juego.away_team, corte=corte, inicio_utc=ref["inicio"],
            y=juego.home_won, p_mercado=ref["p_mercado"],
            x=tuple(float(v[n]) for n in F.TODAS),
            extra={k: v[k] for k in ("n_local", "n_visita", "media_carreras_liga")
                   if k in v}))
    return filas, excl


def _matriz(filas: Sequence[Fila],
            nombres: Sequence[str] = F.NOMBRES) -> Tuple[np.ndarray, np.ndarray]:
    """Las columnas que pide `nombres`, en ese orden.

    Las filas llevan SIEMPRE todas las variables; la versión del modelo decide
    cuáles usa. Así v1.2 y v1.3 se comparan sobre filas idénticas y la única
    diferencia entre las dos es qué columnas entran al ajuste.
    """
    idx = [F.TODAS.index(n) for n in nombres]
    return (np.array([[f.x[i] for i in idx] for f in filas], float),
            np.array([f.y for f in filas], float))


@dataclass
class Candidato:
    """El resultado de evaluar una temporada con lo aprendido en las anteriores."""

    temporada: int
    temporadas_entrenamiento: List[int]
    modelo: Logistica
    filas: List[Fila]
    p_v1: np.ndarray
    p_tasa_base: np.ndarray
    p_mercado: np.ndarray
    y: np.ndarray
    tasa_base_entrenamiento: float


def evaluar_expansivo(
    filas: Sequence[Fila], temporadas_evaluacion: Sequence[int] = (2025, 2026),
    nombres: Sequence[str] = F.NOMBRES,
) -> List[Candidato]:
    """Entrena con las temporadas ANTERIORES y evalúa cada una por separado.

    Nunca un corte aleatorio: pondría juegos del mismo día a ambos lados del
    corte y el ajuste aprendería del futuro por la puerta de al lado.
    """
    salida = []
    for temporada in temporadas_evaluacion:
        tr = [f for f in filas if f.season < temporada]
        te = [f for f in filas if f.season == temporada]
        if not tr or not te:
            continue

        # La etiqueta de todo juego de entrenamiento es de una temporada
        # estrictamente anterior. Se verifica, no se asume.
        assert max(f.season for f in tr) < temporada

        Xtr, ytr = _matriz(tr, nombres)
        Xte, yte = _matriz(te, nombres)
        modelo = ajustar(Xtr, ytr, tuple(nombres))
        p = modelo.predecir(Xte)

        # El detector estadístico corre SIEMPRE, no sólo cuando se sospecha.
        verificar_plausibilidad(p, yte, nombre=f"modelo {temporada}")

        tasa = float(ytr.mean())
        salida.append(Candidato(
            temporada=temporada,
            temporadas_entrenamiento=sorted({f.season for f in tr}),
            modelo=modelo, filas=list(te), p_v1=p,
            p_tasa_base=np.full(len(te), tasa),
            p_mercado=np.array([f.p_mercado for f in te], float),
            y=yte, tasa_base_entrenamiento=tasa))
    return salida
