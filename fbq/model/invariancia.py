"""fbq/model/invariancia.py — el detector que atrapa una fuga de cualquier tamaño.

Los dos detectores anteriores tienen huecos conocidos y complementarios:

- el **estructural** (`pit.py`) impide la fuga por construcción, pero sólo si
  quien arma las variables usa la compuerta; quien la esquiva no lo despierta;
- el **estadístico** (`detector.py`) mira el resultado, pero sólo se enciende
  con una fuga grosera: una que dejara el Brier en 0,235 pasaría bajo el umbral
  de 0,22 sin hacer ruido.

Éste cierra los dos huecos porque no mira ni el código ni la calidad: mira la
**invariancia del proceso completo**. Una predicción pre-juego no puede depender
del resultado del propio partido. Si cambiar ese resultado en los datos de
origen mueve la predicción, hay fuga — sea de 0,005 o de 0,5, esquive o no la
compuerta.

## Las tres pruebas, y por qué hacen falta las tres

1. **Invariancia**: se cambia el marcador del partido objetivo, manteniéndolo
   un final legítimo, y sus variables y su predicción tienen que quedar
   **idénticas**.
2. **Complemento**: se cambia un partido ANTERIOR que sí estaba disponible y sí
   se usó, y las variables que dependen de él tienen que **moverse**. Sin esta,
   una construcción que ignore los datos —una constante— pasaría la primera
   con nota perfecta.
3. **Fuga chica**: se inyecta a propósito una fuga pequeña esquivando la
   compuerta, se comprueba que el Brier sigue pareciendo razonable, y que aun
   así la prueba 1 la rechaza.

## Cómo se perturba

Insertando una **corrección** en `results.observacion`, que es el mecanismo que
el propio almacén tiene para eso: es append-only por trigger, así que un UPDATE
aborta y la vista `resultado` resuelve al último registro. La perturbación usa
la puerta legítima del almacén, no un atajo.

Se perturban únicamente partidos de la temporada de EVALUACIÓN. Así el modelo
se entrena sobre temporadas intactas y queda fijo: si las variables no se
mueven, la predicción tampoco, y la igualdad es exacta en vez de aproximada.
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fbq.evaluator.frame import DB_MERCADO, DB_RESULTADOS
from fbq.model import features as F
from fbq.model.candidato import Fila, _matriz, _precios_de_referencia, construir
from fbq.model.logistica import Logistica, ajustar
from fbq.model.pit import FugaDetectada


@dataclass
class Informe:
    dias_perturbados: List[str]
    n_perturbados: int
    n_comprobables: int
    n_variables_movidas: int
    n_predicciones_movidas: int
    n_excluidos_por_contaminacion: int
    n_posteriores_que_usan_perturbados: int
    n_posteriores_que_se_movieron: int
    n_posteriores_exentos_por_simetria: int = 0
    exentos: List[dict] = field(default_factory=list)
    detalle: List[dict] = field(default_factory=list)

    @property
    def invariante(self) -> bool:
        """Prueba 1: el objetivo no se movió."""
        return (self.n_comprobables > 0
                and self.n_variables_movidas == 0
                and self.n_predicciones_movidas == 0)

    @property
    def reacciona(self) -> bool:
        """Prueba 2: los posteriores se movieron, salvo los exentos.

        Un partido queda EXENTO cuando sus dos equipos tienen perfiles
        idénticos al corte: ahí `dif_pitagorica` vale exactamente 0 para
        cualquier media de liga, así que la fila es insensible a la
        perturbación **por construcción**, no por ignorar los datos. Es raro y
        se cuenta: sobre 5.736 filas hay una sola, dos equipos cuyas ventanas de
        162 partidos —con sólo 12 en común— suman las mismas 680 carreras a
        favor y 739 en contra. Contarla como falla sería confundir una
        coincidencia verificada con un defecto.
        """
        deben = (self.n_posteriores_que_usan_perturbados
                 - self.n_posteriores_exentos_por_simetria)
        return deben > 0 and self.n_posteriores_que_se_movieron == deben

    @property
    def aprueba(self) -> bool:
        return self.invariante and self.reacciona


def _perturbar(db: Path, game_pks: Sequence[int]) -> Dict[int, Tuple[int, int]]:
    """Corrige el marcador de cada partido, dejándolo un final legítimo.

    El nuevo marcador **siempre invierte al ganador y siempre cambia el total**,
    y las dos cosas hacen falta: el ganador mueve todo lo que dependa del
    resultado, el total mueve las carreras y la media de la liga. Una primera
    versión sumaba 7 e invertía los lados, y eso NO invertía al ganador cuando
    el margen original era grande — tres de catorce partidos quedaban con el
    mismo ganador y una fuga inyectada sobre ellos habría pasado inadvertida.
    Un test de fuga que a veces no perturba nada es peor que no tenerlo.

    Nunca produce un empate: en MLB no existe un final empatado y `Final` lo
    rechaza por construcción.
    """
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        filas = con.execute(
            f"""SELECT * FROM resultado WHERE game_pk IN
                ({','.join('?' * len(game_pks))})""",
            tuple(int(g) for g in game_pks)).fetchall()
        nuevos: Dict[int, Tuple[int, int]] = {}
        for r in filas:
            cl, cv = int(r["home_runs"]), int(r["away_runs"])
            # el que perdía gana por paliza; el total cambia siempre
            nl, nv = ((0, cl + cv + 7) if cl > cv else (cl + cv + 7, 0))
            nuevos[int(r["game_pk"])] = (nl, nv)
            con.execute(
                """INSERT INTO observacion
                   (observado_en, game_pk, official_date, season, home_team,
                    away_team, home_runs, away_runs, detailed_state, innings)
                   VALUES ('2099-01-01T00:00:00+00:00',?,?,?,?,?,?,?,'Final',9)""",
                (r["game_pk"], r["official_date"], r["season"], r["home_team"],
                 r["away_team"], nl, nv))
        con.commit()
        return nuevos
    finally:
        con.close()


def _modelo_fijo(filas: Sequence[Fila], temporada: int) -> Logistica:
    """El modelo entrenado con las temporadas anteriores, que no se perturban."""
    tr = [f for f in filas if f.season < temporada]
    X, y = _matriz(tr)
    return ajustar(X, y, F.NOMBRES)


def verificar(
    *,
    dias: Sequence[str],
    temporada: int,
    seasons: Sequence[int] = (2024, 2025, 2026),
    db_mercado: Path = DB_MERCADO,
    db_resultados: Path = DB_RESULTADOS,
    tmp: Optional[Path] = None,
    constructor: Optional[Callable] = None,
) -> Informe:
    """Corre las pruebas 1 y 2 sobre los partidos de `dias`.

    `constructor` se usa sólo para inyectar una fuga y comprobar que la prueba
    la rechaza; en uso normal va en None.
    """
    tmp = Path(tmp or (db_resultados.parent / "_invariancia_tmp.db"))
    _CACHE_PARTIDOS.clear()
    shutil.copy2(db_resultados, tmp)
    try:
        # Los precios no dependen de los marcadores: se calculan una vez.
        precios = _precios_de_referencia(seasons, db_mercado=db_mercado,
                                         db_resultados=tmp)
        base, _ = construir(seasons, db_mercado=db_mercado, db_resultados=tmp,
                            precios=precios, constructor=constructor)
        modelo = _modelo_fijo(base, temporada)
        b_por_pk = {f.game_pk: f for f in base}

        objetivo = [f.game_pk for f in base
                    if f.season == temporada and f.official_date in set(dias)]
        if not objetivo:
            raise ValueError(f"ningún partido evaluable en {list(dias)}")
        _perturbar(tmp, objetivo)

        post, _ = construir(seasons, db_mercado=db_mercado, db_resultados=tmp,
                            precios=precios, constructor=constructor)
        p_por_pk = {f.game_pk: f for f in post}

        # ── Prueba 1: el objetivo no se mueve ────────────────────────────
        perturbados = set(objetivo)
        movidas = predicciones = contaminados = comprobables = 0
        detalle: List[dict] = []
        for pk in objetivo:
            a, b = b_por_pk[pk], p_por_pk.get(pk)
            if b is None:
                detalle.append({"game_pk": pk, "nota": "desapareció del marco"})
                movidas += 1
                continue
            # Si OTRO partido perturbado es previo suyo, su cambio es legítimo
            # y no dice nada sobre la fuga: se excluye y se cuenta.
            if _usa_alguno(tmp, pk, perturbados - {pk}, a.corte, seasons):
                contaminados += 1
                continue
            comprobables += 1
            if a.x != b.x:
                movidas += 1
                detalle.append({"game_pk": pk, "x_antes": a.x, "x_despues": b.x})
            pa = float(modelo.predecir(np.array([a.x]))[0])
            pb = float(modelo.predecir(np.array([b.x]))[0])
            if pa != pb:
                predicciones += 1
                detalle.append({"game_pk": pk, "p_antes": pa, "p_despues": pb})

        # ── Prueba 2: los posteriores que los usan SÍ se mueven ──────────
        usan, se_movieron, exentos_n = 0, 0, 0
        exentos: List[dict] = []
        for pk, a in b_por_pk.items():
            if pk in perturbados or pk not in p_por_pk:
                continue
            if not _usa_alguno(tmp, pk, perturbados, a.corte, seasons):
                continue
            usan += 1
            if a.x != p_por_pk[pk].x:
                se_movieron += 1
            elif _simetrico(tmp, pk, a.corte, seasons):
                exentos_n += 1
                if len(exentos) < 20:
                    exentos.append({"game_pk": pk, "official_date": a.official_date,
                                    "x": a.x, "motivo": "perfiles idénticos: "
                                    "dif_pitagorica ≡ 0 para cualquier media"})

        return Informe(
            dias_perturbados=list(dias), n_perturbados=len(objetivo),
            n_comprobables=comprobables, n_variables_movidas=movidas,
            n_predicciones_movidas=predicciones,
            n_excluidos_por_contaminacion=contaminados,
            n_posteriores_que_usan_perturbados=usan,
            n_posteriores_que_se_movieron=se_movieron,
            n_posteriores_exentos_por_simetria=exentos_n,
            exentos=exentos, detalle=detalle[:20])
    finally:
        tmp.unlink(missing_ok=True)


_CACHE_PARTIDOS: Dict[Tuple, Any] = {}


def _usa_alguno(db: Path, pk: int, candidatos: set, corte: str,
                seasons: Sequence[int]) -> bool:
    """¿Alguno de `candidatos` estaba disponible en el corte de `pk`?

    Es la pregunta de si el cambio de una fila es atribuible a la perturbación
    de otra. Basta con que el candidato estuviera DISPONIBLE en ese corte:
    aunque no sea de ninguno de los dos equipos, entra igual por la media de
    carreras de la liga, que también es un hecho sujeto al corte. Mirar sólo la
    ventana de los dos equipos dejaría ese canal sin cubrir.
    """
    from fbq.model.pit import VentanaPIT, cargar_partidos
    clave = (str(db), tuple(seasons))
    if clave not in _CACHE_PARTIDOS:
        ps = cargar_partidos(seasons, db_resultados=db)
        _CACHE_PARTIDOS[clave] = (ps, VentanaPIT(ps), {p.game_pk: p for p in ps})
    ps, ventana, por_pk = _CACHE_PARTIDOS[clave]
    for otro in candidatos:
        p_otro = por_pk.get(otro)
        if p_otro is not None and ventana.disponible(p_otro, corte):
            return True
    return False


def _simetrico(db: Path, pk: int, corte: str, seasons: Sequence[int]) -> bool:
    """¿Los dos equipos tienen el MISMO perfil en este corte?

    Si lo tienen, `pitagorica(local) − pitagorica(visita)` es exactamente 0 para
    cualquier media de liga —la media entra igual en los dos lados— y la fila es
    insensible a la perturbación por construcción. Distinguirlo de una
    construcción que ignora los datos es justo lo que la prueba 2 existe para
    hacer, así que la excepción se comprueba, no se supone.
    """
    from fbq.model.pit import VentanaPIT, cargar_partidos
    clave = (str(db), tuple(seasons))
    if clave not in _CACHE_PARTIDOS:
        ps = cargar_partidos(seasons, db_resultados=db)
        _CACHE_PARTIDOS[clave] = (ps, VentanaPIT(ps), {p.game_pk: p for p in ps})
    _, ventana, por_pk = _CACHE_PARTIDOS[clave]
    juego = por_pk.get(pk)
    if juego is None:
        return False
    return (F.perfil(ventana, juego.home_team, corte)
            == F.perfil(ventana, juego.away_team, corte))
