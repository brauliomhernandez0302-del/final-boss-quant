"""fbq/model/carga.py — la variable de v1.5: diferencia de carga del bullpen.

Definición congelada en `docs/PREREGISTRO_V1_5_BULLPEN_2026-09-07.md` §3, escrito
antes de medir. Este archivo la ejecuta; no la reinterpreta.

    dif_carga_relevo = (P_local(corte) − P_visita(corte)) / ESCALA_CARGA

    P_E(corte) = Σ lanzamientos de relevo de E en los partidos p con
                 corte − 72 h ≤ disponible_desde(p) ≤ corte

## Por qué `disponible_desde` y no la fecha del partido

«Finalizados durante las 72 h anteriores al corte» es una afirmación sobre el
FIN, y las dos definiciones del sistema anterior no podían sostenerla:

- `data_fetchers.get_bullpen_workload` cerraba la ventana en `utcnow()`, que en
  una reconstrucción histórica es el futuro;
- `bullpen_pit_builder.workload_facts` la cerraba en el DÍA del corte, y un día
  no distingue un partido que terminó a las 02:10 de uno que sigue jugándose.

`disponible_desde` es el fin medido de la última jugada más el margen de 20 min
ya preregistrado. Con él la ventana es una comparación exacta, y la compuerta de
`VentanaPIT` se hereda por construcción: un partido que al corte no había
terminado ni siquiera está indexado.

## Faltantes

No se imputa nunca (preregistro §4). Un partido dentro de la ventana sin fila en
el almacén, o sin fin fechable, deja la fila **no computable** — y una fila no
computable se cae de v1.5, de v1.2 y del mercado a la vez, para que las tres
columnas se midan sobre exactamente las mismas filas.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from fbq.core.clock import normalizar_utc
from fbq.model.pit import Partido

# ── Constantes del preregistro §3. No se ajustan. ────────────────────────
VENTANA_HORAS = 72.0
#: Divide para que la variable entre a la ridge (λ=1,0) en un orden comparable
#: al de `dif_descanso`. Constante FIJA: estandarizar con media y desvío de la
#: muestra metería el pliegue de prueba en el ajuste.
ESCALA_CARGA = 100.0


def _t(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(normalizar_utc(s))


@dataclass(frozen=True)
class _Entrada:
    disponible_desde: str
    pitches: Optional[int]        # None = el almacén no tiene la fila
    game_pk: int


class IndiceCarga:
    """Los lanzamientos de relevo de cada equipo, ordenados por disponibilidad.

    Se construye una vez y se consulta miles de veces: la ventana es una
    búsqueda lineal sobre la cola de la lista, que en 72 h son 3 o 4 partidos.
    """

    def __init__(self, partidos: Iterable[Partido],
                 relevo: Dict[Tuple[int, int], Dict[str, object]]) -> None:
        self._por_equipo: Dict[str, List[_Entrada]] = defaultdict(list)
        #: Partidos sin fin fechable, por equipo y día. No pueden entrar a la
        #: ventana, pero su EXISTENCIA impide afirmar que el equipo no lanzó.
        self._sin_fecha: Dict[str, List[str]] = defaultdict(list)
        for p in partidos:
            for equipo, es_local in ((p.home_team, 1), (p.away_team, 0)):
                if p.disponible_desde is None:
                    self._sin_fecha[equipo].append(p.official_date)
                    continue
                fila = relevo.get((int(p.game_pk), es_local))
                self._por_equipo[equipo].append(_Entrada(
                    disponible_desde=p.disponible_desde,
                    pitches=(None if fila is None
                             else int(fila["pitches_relevo"])),
                    game_pk=int(p.game_pk)))
        for lista in self._por_equipo.values():
            lista.sort(key=lambda e: (e.disponible_desde, e.game_pk))

    def carga(self, equipo: str, corte: str) -> Dict[str, object]:
        """`{ok, pitches, juegos, motivo}` para un equipo en un corte."""
        hasta = _t(corte)
        desde = hasta - dt.timedelta(hours=VENTANA_HORAS)
        desde_s, hasta_s = desde.isoformat(), hasta.isoformat()

        en_ventana = [e for e in self._por_equipo.get(equipo, [])
                      if desde_s <= e.disponible_desde <= hasta_s]
        if any(e.pitches is None for e in en_ventana):
            return {"ok": False, "motivo": "sin_fila_de_relevo",
                    "juegos": len(en_ventana), "pitches": None}

        # Un partido sin fin fechable cuyo DÍA cae en la ventana podría haber
        # terminado dentro de ella. No se puede afirmar que no aportó carga, así
        # que la fila no es computable. Se mira por día, con un día de holgura a
        # cada lado, porque es lo único que se sabe de esos partidos.
        dias = {(desde.date() + dt.timedelta(days=i)).isoformat()
                for i in range(-1, int(VENTANA_HORAS // 24) + 2)}
        if dias.intersection(self._sin_fecha.get(equipo, ())):
            return {"ok": False, "motivo": "partido_sin_fin_fechable",
                    "juegos": len(en_ventana), "pitches": None}

        return {"ok": True, "motivo": "",
                "juegos": len(en_ventana),
                "pitches": sum(int(e.pitches or 0) for e in en_ventana)}

    def diferencia(self, local: str, visita: str, corte: str) -> Dict[str, object]:
        """La variable del preregistro, ya escalada, o el motivo de que no."""
        cl = self.carga(local, corte)
        cv = self.carga(visita, corte)
        if not (cl["ok"] and cv["ok"]):
            return {"ok": False,
                    "motivo": cl["motivo"] or cv["motivo"],
                    "dif_carga_relevo": None}
        return {
            "ok": True, "motivo": "",
            "dif_carga_relevo": (int(cl["pitches"]) - int(cv["pitches"])) / ESCALA_CARGA,
            "pitches_relevo_local": int(cl["pitches"]),
            "pitches_relevo_visita": int(cv["pitches"]),
            "juegos_ventana_local": int(cl["juegos"]),
            "juegos_ventana_visita": int(cv["juegos"]),
        }


def cargar_indice(partidos: Sequence[Partido], db=None) -> IndiceCarga:
    from fbq.model.bullpen import DB_PATH, cargar
    return IndiceCarga(partidos, cargar(db or DB_PATH))


# ══ Concentración de la carga entre relevistas (candidato v1.6) ══════════
#
# Definición congelada en `docs/PREREGISTRO_V1_6_CONCENTRACION_2026-09-08.md`.
#
# Misma ventana de 72 h y misma compuerta `disponible_desde` que la carga total:
# se REUTILIZAN, no se vuelven a elegir. Lo único nuevo es el estadístico.

class IndiceConcentracion:
    """Herfindahl de los lanzamientos de relevo entre los brazos de un equipo.

        HHI_E(corte) = Σ_i (p_i / P)²    sobre los relevistas i con p_i > 0
                                          en la ventana; P = Σ p_i

    En 1 cuando un solo brazo cargó con todo; en 1/k cuando k brazos cargaron
    por igual. Es una proporción, así que no depende de cuántos lanzamientos se
    hicieron sino de **entre cuántos brazos se repartieron**.

    ⚠️ Esto es carga OBSERVADA, no disponibilidad. Que la carga se concentrara
    en dos brazos no demuestra que los demás no estuvieran disponibles: pudo ser
    rol, marcador, o que el partido no pidiera más. Ver el encabezado de
    `fbq/model/relevistas.py`.
    """

    def __init__(self, apariciones: Iterable[Dict[str, object]]) -> None:
        #: {equipo_id: [(disponible_desde, pitcher_id, pitches)]}, ordenado
        self._por_equipo: Dict[int, List[tuple]] = defaultdict(list)
        #: {(game_pk, es_local): team_id} — el puente entre `Partido`, que
        #: identifica equipos por NOMBRE, y este índice, que los identifica por
        #: id. Se arma del mismo documento, así que no hay emparejamiento de
        #: nombres que pueda fallar en silencio.
        self._equipo_de_juego: Dict[tuple, int] = {}
        for a in apariciones:
            if a.get("game_pk") is not None and a.get("es_local") is not None:
                self._equipo_de_juego[(int(a["game_pk"]), int(a["es_local"]))] = \
                    int(a["team_id"])
            if a["rol"] != "relevo" or a["disponible_desde"] is None:
                continue
            self._por_equipo[int(a["team_id"])].append(
                (str(a["disponible_desde"]), int(a["pitcher_id"]), int(a["pitches"])))
        for lista in self._por_equipo.values():
            lista.sort()

    def hhi(self, team_id: int, corte: str) -> Dict[str, object]:
        hasta = _t(corte)
        desde = (hasta - dt.timedelta(hours=VENTANA_HORAS)).isoformat()
        hasta_s = hasta.isoformat()
        por_brazo: Dict[int, int] = defaultdict(int)
        for disp, pid, pitches in self._por_equipo.get(int(team_id), ()):
            if desde <= disp <= hasta_s:
                por_brazo[pid] += pitches
        total = sum(v for v in por_brazo.values() if v > 0)
        if total <= 0:
            # Sin lanzamientos de relevo en la ventana el índice es 0/0. No se
            # inventa un valor: la fila no es computable (preregistro §4).
            return {"ok": False, "motivo": "sin_relevo_en_la_ventana",
                    "hhi": None, "brazos": 0, "pitches": 0}
        hhi = sum((v / total) ** 2 for v in por_brazo.values() if v > 0)
        return {"ok": True, "motivo": "", "hhi": hhi,
                "brazos": sum(1 for v in por_brazo.values() if v > 0),
                "pitches": total}

    def diferencia_de_juego(self, game_pk: int, corte: str) -> Dict[str, object]:
        """La diferencia del partido, resolviendo los ids por `game_pk`."""
        local = self._equipo_de_juego.get((int(game_pk), 1))
        visita = self._equipo_de_juego.get((int(game_pk), 0))
        if local is None or visita is None:
            return {"ok": False, "motivo": "sin_identidad_de_equipo",
                    "dif_concentracion": None}
        return self.diferencia(local, visita, corte)

    def diferencia(self, local_id: int, visita_id: int, corte: str) -> Dict[str, object]:
        hl = self.hhi(local_id, corte)
        hv = self.hhi(visita_id, corte)
        if not (hl["ok"] and hv["ok"]):
            return {"ok": False, "motivo": hl["motivo"] or hv["motivo"],
                    "dif_concentracion": None}
        return {
            "ok": True, "motivo": "",
            "dif_concentracion": float(hl["hhi"]) - float(hv["hhi"]),
            "hhi_local": float(hl["hhi"]), "hhi_visita": float(hv["hhi"]),
            "brazos_local": int(hl["brazos"]), "brazos_visita": int(hv["brazos"]),
        }


def cargar_concentracion(db=None) -> IndiceConcentracion:
    import sqlite3
    from fbq.model.relevistas import DB_PATH as DB_REL
    con = sqlite3.connect(f"file:{db or DB_REL}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return IndiceConcentracion(
            dict(r) for r in con.execute(
                # TODAS las filas, no sólo las de relevo: el puente
                # juego→equipo necesita también los equipos cuyo abridor lanzó
                # el partido completo. El filtro por rol vive en el índice.
                "SELECT game_pk, es_local, team_id, pitcher_id, pitches, rol, "
                "disponible_desde FROM aparicion"))
    finally:
        con.close()
