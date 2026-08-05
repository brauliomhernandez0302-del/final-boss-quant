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

DB_PATH = Path(__file__).parent.parent.parent / "data" / "predictions_history.db"


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
