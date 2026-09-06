"""fbq/model/abridores.py — la calidad del abridor, `K% − BB%`, del almacén PIT.

Preregistro: `docs/PREREGISTRO_V1_4_ABRIDORES_2026-09-06.md`.

## De dónde sale cada número

| dato | fuente | por qué ésa |
|---|---|---|
| `k_pct`, `bb_pct` | `pit_cache_pitcher.db`, namespace `fangraphs.pitcher.daily` | son **instantáneas PIT diarias** por `as_of_date`: el sistema anterior las construyó justamente para poder preguntar "qué se sabía el día D" |
| bateadores enfrentados | mismo almacén, namespace `savant.pitcher.rolling`, campo `pa` | los turnos enfrentados por el lanzador hasta esa fecha |

Cobertura del almacén: **2024-03-28 → 2025-09-27**. Fuera de ese rango no hay
instantánea y el juego se excluye; no se imputa un abridor promedio.

## Para predecir hacia adelante

Un partido de mañana no necesita una instantánea histórica: necesita lo que se
sabe HOY, que es trivialmente anterior al corte. Ahí la fuente es la API de MLB
(gratis, sin clave) con las estadísticas de la temporada en curso.

**La sustitución está medida, no supuesta.** Sobre 29 lanzadores con ≥100
bateadores enfrentados en 2025, comparando el `k_pct`/`bb_pct` de FanGraphs al
cierre de la temporada contra el derivado de la API de MLB:

    K%   diferencia media +0,00075   |dif| máx 0,01522
    BB%  diferencia media −0,00038   |dif| máx 0,01238

Frente a una dispersión de K% entre lanzadores de ~0,15 a 0,30, esas magnitudes
son ruido de definición, no un cambio de métrica. La paridad se declara y se
vigila; el proyecto ya pagó una vez por dos implementaciones que derivaron.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from fbq.model.features import regress

RAIZ = Path(__file__).parent.parent.parent
DB_PITCHERS = RAIZ / "data" / "pit_cache_pitcher.db"

# ── Constantes del preregistro §3, fijadas antes de medir ────────────────
K_BF = 300              # encogimiento hacia la liga, en bateadores enfrentados
MIN_BF_ABRIDOR = 150    # debajo de esto el juego se EXCLUYE, no se imputa
LIGA_K_MENOS_BB = 0.135 # K% − BB% de liga; ver la nota de `media_liga()`

TIMEOUT = (5, 20)


@dataclass(frozen=True)
class Abridor:
    pitcher_id: int
    k_pct: float
    bb_pct: float
    bf: float
    fuente: str
    as_of: Optional[str] = None

    @property
    def bruto(self) -> float:
        return self.k_pct - self.bb_pct

    def calidad(self, media_liga: float = LIGA_K_MENOS_BB) -> float:
        """`K% − BB%` encogido hacia la liga por bateadores enfrentados."""
        return regress(self.bruto, media_liga, self.bf, K_BF)


# ── Camino histórico: el almacén PIT ─────────────────────────────────────

def desde_pit(
    pitcher_id: int, corte_dia: str, *, db: Path = DB_PITCHERS,
) -> Optional[Abridor]:
    """La instantánea más reciente ESTRICTAMENTE anterior a `corte_dia`.

    `corte_dia` es una fecha (`YYYY-MM-DD`); las instantáneas se etiquetan por
    día. La comparación es estricta, como todo corte de este proyecto: una
    instantánea etiquetada el propio día del juego incluiría el juego.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        fg = con.execute(
            """SELECT as_of_date, data_json FROM pit_metric_cache
               WHERE namespace='fangraphs.pitcher.daily' AND entity_id=?
                 AND substr(as_of_date,1,10) < ?
               ORDER BY as_of_date DESC LIMIT 1""",
            (str(pitcher_id), corte_dia)).fetchone()
        if fg is None:
            return None
        sv = con.execute(
            """SELECT data_json FROM pit_metric_cache
               WHERE namespace='savant.pitcher.rolling' AND entity_id=?
                 AND substr(as_of_date,1,10) < ?
               ORDER BY as_of_date DESC LIMIT 1""",
            (str(pitcher_id), corte_dia)).fetchone()
    finally:
        con.close()

    d = json.loads(fg["data_json"])
    if d.get("k_pct") is None or d.get("bb_pct") is None:
        return None
    bf = float((json.loads(sv["data_json"]).get("pa") or 0) if sv else 0)
    return Abridor(int(pitcher_id), float(d["k_pct"]), float(d["bb_pct"]), bf,
                   fuente="pit_fangraphs_daily", as_of=fg["as_of_date"][:10])


# ── Camino prospectivo: la API gratuita de MLB ───────────────────────────

def desde_mlb(pitcher_id: int, season: int) -> Optional[Abridor]:
    """Estadísticas de la temporada en curso. Gratis, sin clave.

    Para un partido futuro no hay problema de corte: lo que se sabe hoy es, por
    definición, anterior al primer lanzamiento de mañana.
    """
    try:
        r = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{int(pitcher_id)}/stats",
            params={"stats": "season", "season": int(season), "group": "pitching"},
            timeout=TIMEOUT)
        r.raise_for_status()
        sp = r.json()["stats"][0]["splits"][0]["stat"]
    except Exception:                                     # noqa: BLE001
        return None
    bf = float(sp.get("battersFaced") or 0)
    if bf <= 0:
        return None
    return Abridor(int(pitcher_id), float(sp["strikeOuts"]) / bf,
                   float(sp["baseOnBalls"]) / bf, bf, fuente=f"mlb_api_{season}")


def diferencia(
    local: Optional[Abridor], visita: Optional[Abridor],
    *, media_liga: float = LIGA_K_MENOS_BB,
) -> Tuple[Optional[float], str]:
    """`calidad(local) − calidad(visitante)`, o `(None, motivo)`.

    El abridor local somete a la ofensa visitante, así que un local mejor sube
    P(gana el local): positivo favorece al local, igual que `dif_pitagorica`.
    """
    if local is None or visita is None:
        return None, "sin_estadistica_de_abridor"
    if local.bf < MIN_BF_ABRIDOR or visita.bf < MIN_BF_ABRIDOR:
        return None, "abridor_con_muestra_insuficiente"
    return local.calidad(media_liga) - visita.calidad(media_liga), "ok"
