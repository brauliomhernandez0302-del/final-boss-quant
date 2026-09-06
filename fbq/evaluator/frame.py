"""evaluator/frame.py — el marco de evaluación: un juego, una fila, solo hechos.

Lo que entra acá es exclusivamente lo que OCURRIÓ (quién ganó) y lo que el
MERCADO cotizaba (el par de precios). Ninguna probabilidad del modelo: ésas
entran después, como candidato, por parámetro. La separación es deliberada —
un marco que ya trae adentro la predicción de la casa invita a compararla
contra sí misma.

Solo lectura. Ninguna función de este archivo escribe.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

RAIZ = Path(__file__).parent.parent.parent
# El almacén del sistema ANTERIOR. `load_frame()` sigue leyendo de acá y se
# conserva a propósito: es la referencia contra la cual se valida el marco
# propio. Cuando `load_frame_propio()` reproduzca sus números sobre las mismas
# claves, esta ruta deja de ser necesaria — pero no antes, y borrarla antes
# sería cambiar de instrumento y de resultado en el mismo paso.
DB_PATH = RAIZ / "data" / "predictions_history.db"

DB_MERCADO = RAIZ / "data" / "market.db"
DB_RESULTADOS = RAIZ / "data" / "results.db"

# Agregado derivado, no una casa. Nunca entra al "mejor precio disponible":
# nadie puede apostar contra un promedio.
CONSENSO = "_consensus"


@dataclass
class EvalFrame:
    """Un juego por posición en cada array. Todos los arrays van alineados."""

    game_pk: np.ndarray
    official_date: np.ndarray
    season: np.ndarray
    home_team: np.ndarray
    away_team: np.ndarray
    y: np.ndarray            # 1 si ganó el local
    p_market: np.ndarray     # Pinnacle desvigorizado — el NULO
    overround: np.ndarray    # el margen del par de Pinnacle, por juego
    best_home: np.ndarray    # mejor precio disponible, para el ROI
    best_away: np.ndarray

    def __len__(self) -> int:
        return len(self.game_pk)

    @property
    def base_rate(self) -> float:
        return float(self.y.mean())

    def candidate_from_column(self, valores: Dict[int, float]) -> np.ndarray:
        """Alinea un dict {game_pk: p} al orden del marco.

        Un juego sin valor queda NaN y las métricas lo excluyen — nunca se
        rellena con 0.5, que parecería una predicción y no lo es.
        """
        return np.array([valores.get(int(pk), np.nan) for pk in self.game_pk], float)

    def subset(self, mask: np.ndarray) -> "EvalFrame":
        return EvalFrame(**{k: v[mask] for k, v in self.__dict__.items()})


def _devig_multiplicativo(precio_a: np.ndarray, precio_b: np.ndarray):
    """(prob_justa_a, overround). Mismo método que usa el pipeline.

    Se desvigoriza el par de UN MISMO libro (Pinnacle). El precio crudo lleva el
    margen de la casa adentro: medir contra él regala el vig entero como si
    fuera habilidad — sobre los picks reales del ledger eso invirtió el signo
    del CLV, de +1.514% a −0.497%.
    """
    imp_a, imp_b = 1.0 / precio_a, 1.0 / precio_b
    total = imp_a + imp_b
    return imp_a / total, total - 1.0


def load_frame(
    seasons: Sequence[int] = (2024, 2025),
    *,
    db_path: Path = DB_PATH,
    require_market: bool = True,
) -> EvalFrame:
    """Carga el marco desde `game_outcomes` ⋈ `historical_odds`.

    `require_market=True` (default) deja fuera los juegos sin par de Pinnacle:
    sin el nulo no hay nada contra qué comparar, y arrastrarlos sólo cambiaría
    la muestra entre métricas sin avisar.

    El precio del ROI es el MEJOR disponible (`ml_*_best`), no el de Pinnacle:
    la línea justa se calcula contra el libro sharp, pero se apuesta al mejor
    precio que haya. Son dos preguntas distintas y usan números distintos.
    """
    marcadores = ",".join("?" * len(seasons))
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        filas = con.execute(
            f"""
            SELECT g.game_pk, g.official_date, g.season, g.home_team, g.away_team,
                   g.home_won,
                   h.ml_home_pin, h.ml_away_pin,
                   h.ml_home_best, h.ml_away_best
            FROM game_outcomes g
            LEFT JOIN historical_odds h ON h.game_pk = g.game_pk
            WHERE g.season IN ({marcadores})
              AND g.home_won IS NOT NULL
            ORDER BY g.official_date, g.game_pk
            """,
            tuple(seasons),
        ).fetchall()
    finally:
        con.close()

    if require_market:
        filas = [r for r in filas if r["ml_home_pin"] and r["ml_away_pin"]]
    if not filas:
        raise ValueError(f"sin juegos utilizables para las temporadas {list(seasons)}")

    pin_h = np.array([r["ml_home_pin"] or np.nan for r in filas], float)
    pin_a = np.array([r["ml_away_pin"] or np.nan for r in filas], float)
    p_market, overround = _devig_multiplicativo(pin_h, pin_a)

    return EvalFrame(
        game_pk=np.array([r["game_pk"] for r in filas]),
        official_date=np.array([r["official_date"] for r in filas]),
        season=np.array([r["season"] for r in filas]),
        home_team=np.array([r["home_team"] for r in filas]),
        away_team=np.array([r["away_team"] for r in filas]),
        y=np.array([r["home_won"] for r in filas], float),
        p_market=p_market,
        overround=overround,
        best_home=np.array([r["ml_home_best"] or np.nan for r in filas], float),
        best_away=np.array([r["ml_away_best"] or np.nan for r in filas], float),
    )


def load_frame_propio(
    seasons: Sequence[int] = (2024, 2025),
    *,
    db_mercado: Path = DB_MERCADO,
    db_resultados: Path = DB_RESULTADOS,
    solo_pregame: bool = True,
    book: str = "pinnacle",
    sport_key: str = "baseball_mlb",
) -> EvalFrame:
    """El mismo marco, armado desde los almacenes PROPIOS de `fbq`.

    Hechos de `results.db`, precios de `market.db`, unidos por `event_link`.
    Ninguna lectura del sistema anterior.

    `solo_pregame=True` (default) descarta las cotizaciones observadas después
    del primer lanzamiento. No es un detalle de higiene: sobre esta misma foto
    histórica son 139 juegos cuyo "precio de mercado" es en realidad una
    cotización EN VIVO, con mediana de 45 minutos de juego ya disputado. Un
    precio en vivo sabe cosas que un modelo pre-juego no puede saber, así que
    usarlo como nulo es medirse contra un rival que vio parte del partido.

    El precio del ROI es el MEJOR disponible entre casas reales —el consenso
    queda fuera: nadie puede apostar contra un promedio— mientras que la línea
    justa se calcula con el par del libro sharp. Son dos preguntas distintas y
    usan números distintos, igual que en `load_frame()`.
    """
    marcadores = ",".join("?" * len(seasons))
    con = sqlite3.connect(f"file:{db_resultados}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        con.execute("ATTACH DATABASE ? AS mkt", (f"file:{db_mercado}?mode=ro",))
        filtro_pregame = "AND s.captured_at < s.commence_time" if solo_pregame else ""
        filas = con.execute(
            f"""
            WITH ultima AS (
                SELECT s.event_id, s.book, s.side, s.price_dec,
                       ROW_NUMBER() OVER (
                           PARTITION BY s.event_id, s.side, s.book
                           ORDER BY s.captured_at DESC, s.id DESC) AS rn
                FROM mkt.odds_snapshot s
                WHERE s.sport_key = ? AND s.market = 'h2h'
                  {filtro_pregame}
            ),
            precios AS (
                SELECT l.game_pk,
                       MAX(CASE WHEN u.book = ? AND u.side='home' THEN u.price_dec END) AS pin_home,
                       MAX(CASE WHEN u.book = ? AND u.side='away' THEN u.price_dec END) AS pin_away,
                       MAX(CASE WHEN u.book <> ? AND u.side='home' THEN u.price_dec END) AS best_home,
                       MAX(CASE WHEN u.book <> ? AND u.side='away' THEN u.price_dec END) AS best_away
                FROM ultima u
                JOIN mkt.event_link l ON l.event_id = u.event_id
                WHERE u.rn = 1 AND l.game_pk IS NOT NULL
                GROUP BY l.game_pk
            )
            SELECT r.game_pk, r.official_date, r.season, r.home_team, r.away_team,
                   r.home_won, p.pin_home, p.pin_away, p.best_home, p.best_away
            FROM resultado r
            JOIN precios p ON p.game_pk = r.game_pk
            WHERE r.season IN ({marcadores})
            ORDER BY r.official_date, r.game_pk
            """,
            (sport_key, book, book, CONSENSO, CONSENSO, *seasons),
        ).fetchall()
    finally:
        con.close()

    filas = [f for f in filas if f["pin_home"] and f["pin_away"]]
    if not filas:
        raise ValueError(f"sin juegos utilizables para las temporadas {list(seasons)}")

    pin_h = np.array([f["pin_home"] for f in filas], float)
    pin_a = np.array([f["pin_away"] for f in filas], float)
    p_market, overround = _devig_multiplicativo(pin_h, pin_a)

    return EvalFrame(
        game_pk=np.array([f["game_pk"] for f in filas]),
        official_date=np.array([f["official_date"] for f in filas]),
        season=np.array([f["season"] for f in filas]),
        home_team=np.array([f["home_team"] for f in filas]),
        away_team=np.array([f["away_team"] for f in filas]),
        y=np.array([f["home_won"] for f in filas], float),
        p_market=p_market,
        overround=overround,
        best_home=np.array([f["best_home"] or np.nan for f in filas], float),
        best_away=np.array([f["best_away"] or np.nan for f in filas], float),
    )


def load_candidate(
    columna: str,
    seasons: Sequence[int],
    *,
    db_path: Path = DB_PATH,
    run_date: Optional[str] = None,
) -> Dict[int, float]:
    """Un candidato guardado en `game_outcomes`, como {game_pk: p_home}.

    Atajo de conveniencia para los dos candidatos que ya viven en la DB
    (`p_home` en vivo, `backtest_p_home`). Cualquier otro candidato se pasa como
    dict o callable sin tocar este archivo — que es el punto del módulo.

    `run_date` filtra por `backtest_run_at` para no mezclar corridas: la DB
    tiene tres, y una de ellas (2026-06-28) es anterior a la remediación de
    CHRON-001.
    """
    if columna not in ("p_home", "backtest_p_home"):
        raise ValueError(f"columna no permitida: {columna!r}")
    marcadores = ",".join("?" * len(seasons))
    sql = (f"SELECT game_pk, {columna} AS p FROM game_outcomes "
           f"WHERE season IN ({marcadores}) AND {columna} IS NOT NULL")
    params: List = list(seasons)
    if run_date:
        sql += " AND substr(backtest_run_at,1,10) = ?"
        params.append(run_date)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {int(pk): float(p) for pk, p in con.execute(sql, params)}
    finally:
        con.close()
