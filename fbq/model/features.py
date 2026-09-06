"""fbq/model/features.py — los hechos deportivos, calculados sólo con lo disponible.

Dos variables, del preregistro §3. Ninguna toca nada que no venga de
`VentanaPIT`, que es lo que hace que el corte temporal sea una propiedad del
mecanismo y no de la disciplina de quien las escribe.

## Qué se reutiliza del sistema anterior

`regress()` es una **copia atribuida** de
`modules/baseball_module/offense/tte_formula.py`. Su validez temporal es
trivial: es matemática pura, recibe números, no accede a ningún dato y no
conoce fechas — no hay nada que pueda filtrarse.

Se copia en vez de importarse por dos razones que tiran en la misma dirección:
`fbq/` es autónomo del sistema anterior por diseño, y ese árbol está declarado
inválido y puede borrarse en cualquier momento. Como el propio proyecto ya pagó
una duplicación que derivó —la constante de barrel% arreglada en una copia y no
en la otra, dos implementaciones independientes durante meses—,
`tests/test_fbq_model.py` compara esta copia contra el original mientras el
original exista.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence

from fbq.model.pit import Partido, VentanaPIT
from fbq.model.recuperado import en_back_to_back

# ── Constantes del preregistro §9 ────────────────────────────────────────
VENTANA = 162
MIN_JUEGOS_PREVIOS = 30
K_REGRESION = 67
EXPONENTE_PITAGORICO = 1.83
TOPE_DESCANSO = 5

# Las dos variables de la v1.2, congeladas.
NOMBRES = ("dif_pitagorica", "dif_descanso")

# La v1.3 agrega el componente recuperado. Los nombres van por separado para que
# la v1.2 se siga pudiendo reproducir exactamente sobre las MISMAS filas: lo que
# cambia entre versiones es qué columnas usa el ajuste, no cómo se arma la fila.
NOMBRES_V13 = NOMBRES + ("b2b_visita",)

# Se calcula siempre pero NO entra a ningún modelo: es el diagnóstico que dice
# si la neutralización del lado local que hizo el sistema anterior se sostiene
# sobre datos propios.
DIAGNOSTICOS = ("b2b_local",)

TODAS = NOMBRES_V13 + DIAGNOSTICOS


def regress(observed: float, mean: float, n: float, k: float) -> float:
    """Encogimiento bayesiano hacia `mean`. En n=0 → mean; en n=k → 50/50;
    en n=∞ → observed.

    COPIA ATRIBUIDA de `modules/baseball_module/offense/tte_formula.py`. Ver la
    nota del módulo sobre por qué se copia y qué test vigila la copia.
    """
    if n <= 0:
        return mean
    return (observed * n + mean * k) / (n + k)


@dataclass(frozen=True)
class Perfil:
    """Lo que se sabe de un equipo en un corte, y con cuánta evidencia."""

    n: int
    cf_juego: float      # carreras a favor por partido
    cc_juego: float      # carreras en contra por partido
    ultimo_dia: Optional[str]


def perfil(ventana: VentanaPIT, equipo: str, corte: str) -> Perfil:
    previos = ventana.de_equipo(equipo, corte, ventana=VENTANA)
    if not previos:
        return Perfil(0, 0.0, 0.0, None)
    cf = cc = 0
    for p in previos:
        if p.home_team == equipo:
            cf += p.home_runs; cc += p.away_runs
        else:
            cf += p.away_runs; cc += p.home_runs
    n = len(previos)
    return Perfil(n, cf / n, cc / n, previos[-1].official_date)


def pitagorica(p: Perfil, media_liga: float) -> float:
    """`CF^e / (CF^e + CC^e)` con las tasas regresadas hacia la liga.

    El curso del proyecto registra el orden de calidad de los estimadores de
    talento: récord < pitagórica < BaseRuns. BaseRuns necesita hits, bases por
    bolas y bases totales, y ninguna vive en los almacenes propios; queda como
    la mejora obvia de la v2.
    """
    cf = regress(p.cf_juego, media_liga, p.n, K_REGRESION)
    cc = regress(p.cc_juego, media_liga, p.n, K_REGRESION)
    if cf <= 0 and cc <= 0:
        return 0.5
    a, b = cf ** EXPONENTE_PITAGORICO, cc ** EXPONENTE_PITAGORICO
    return a / (a + b)


def descanso(p: Perfil, dia_juego: str) -> Optional[float]:
    """Días entre el último partido disponible del equipo y el día del juego.

    Se cuenta sobre `official_date` y no sobre el timestamp UTC: el paso 4 de la
    auditoría del sistema anterior encontró que derivarlo del UTC truncado daba
    `days_rest` +1 para todo nocturno de la costa oeste, el 22-24% de los
    juegos.
    """
    if p.ultimo_dia is None:
        return None
    d = (date.fromisoformat(dia_juego) - date.fromisoformat(p.ultimo_dia)).days
    return float(max(0, min(d, TOPE_DESCANSO)))


class IndiceLiga:
    """Carreras por equipo y por partido en la liga, en cualquier corte.

    También pasa por la compuerta: la media de la liga es un hecho tan sujeto al
    corte como el récord de un equipo. Se precalcula con sumas acumuladas sobre
    los partidos ordenados por disponibilidad — el resultado es idéntico al del
    barrido, sólo cambia el costo, y ese costo es lo que hace practicable
    reconstruir el marco entero por cada partido perturbado.
    """

    def __init__(self, partidos: Sequence[Partido]) -> None:
        ordenados = sorted((p for p in partidos if p.disponible_desde is not None),
                           key=lambda p: p.disponible_desde)
        self._claves = [p.disponible_desde for p in ordenados]
        self._acum = [0]
        for p in ordenados:
            self._acum.append(self._acum[-1] + p.home_runs + p.away_runs)

    def media(self, corte: str) -> float:
        n = bisect_right(self._claves, corte)
        if n == 0:
            return 4.5   # sólo alcanzable si no hay NINGÚN partido previo
        return self._acum[n] / (2 * n)


def media_carreras_liga(ventana: VentanaPIT, partidos: Sequence[Partido],
                        corte: str) -> float:
    """Compatibilidad: la versión directa, sin índice. Misma cuenta."""
    disponibles = [p for p in partidos if ventana.disponible(p, corte)]
    if not disponibles:
        return 4.5
    return sum(p.home_runs + p.away_runs for p in disponibles) / (2 * len(disponibles))


def construir_fila(
    ventana: VentanaPIT,
    partidos: Sequence[Partido],
    juego: Partido,
    corte: str,
    indice_liga: Optional["IndiceLiga"] = None,
    inicios: Optional[Dict[int, str]] = None,
) -> Dict[str, object]:
    """Las variables de un juego, al corte de su precio de referencia.

    Devuelve `{"ok": False, "motivo": ...}` cuando el juego no es predecible con
    lo disponible. Nunca imputa: un juego que no se puede predecir se excluye, y
    la exclusión se cuenta.
    """
    pl = perfil(ventana, juego.home_team, corte)
    pv = perfil(ventana, juego.away_team, corte)
    if pl.n < MIN_JUEGOS_PREVIOS or pv.n < MIN_JUEGOS_PREVIOS:
        return {"ok": False, "motivo": "historial_insuficiente",
                "n_local": pl.n, "n_visita": pv.n}

    media = (indice_liga.media(corte) if indice_liga is not None
             else media_carreras_liga(ventana, partidos, corte))
    dl, dv = descanso(pl, juego.official_date), descanso(pv, juego.official_date)
    if dl is None or dv is None:
        return {"ok": False, "motivo": "sin_descanso_calculable"}

    # Componente recuperado (v1.3). El partido anterior es el último DISPONIBLE
    # al corte, no el último del calendario.
    inicios = inicios or {}
    inicio_juego = inicios.get(juego.game_pk)
    b2b_v = en_back_to_back(ventana.de_equipo(juego.away_team, corte, ventana=VENTANA),
                            inicio_juego, inicios)
    b2b_l = en_back_to_back(ventana.de_equipo(juego.home_team, corte, ventana=VENTANA),
                            inicio_juego, inicios)
    if b2b_v is None or b2b_l is None:
        return {"ok": False, "motivo": "sin_b2b_calculable"}

    return {
        "ok": True,
        "dif_pitagorica": pitagorica(pl, media) - pitagorica(pv, media),
        "dif_descanso": dl - dv,
        "b2b_visita": b2b_v,
        "b2b_local": b2b_l,
        "n_local": pl.n, "n_visita": pv.n,
        "media_carreras_liga": media,
    }
