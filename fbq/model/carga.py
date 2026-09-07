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
