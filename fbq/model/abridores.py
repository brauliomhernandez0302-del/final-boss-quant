"""fbq/model/abridores.py — la calidad del abridor, `K% − BB%`, del preregistro.

Preregistro: `docs/PREREGISTRO_V1_4_ABRIDORES_2026-09-06.md`.

## Por qué NO se combinan las dos fuentes del almacén PIT

Se verificó antes de combinarlas, sobre 300 instantáneas al azar:

| comprobación | resultado |
|---|---|
| `K%`/`BB%` de `fangraphs.pitcher.daily` == reconstruido del Statcast crudo | 221 / 300 |
| `pa` de `savant.pitcher.rolling` == bateadores enfrentados del crudo | **129 / 300** |
| instantáneas que incluyen juegos POSTERIORES a su `as_of` | **0 / 300** |

El numerador y el denominador venían de fuentes que **no cuentan la misma
población de turnos**. Y ninguna de las dos ventanas era la del preregistro: las
dos son acumuladas de temporada, y el preregistro fija **40 aperturas** que
cruzan el borde de temporada.

## Lo que sí se usa

`fbq/model/aperturas.py`: una fila por (lanzador, apertura) del `gameLog`
oficial de MLB —gratis, sin clave—, con `k`, `bb` y `bf` del **mismo registro de
boxscore**, así que numerador y denominador son consistentes por construcción.
La misma fuente para entrenar y para predecir: no hay paridad que vigilar entre
dos caminos.

Validado contra el Statcast crudo sobre 300 aperturas: **K 300/300**, **BB
300/300**, **BF 287/300** (Statcast cuenta un turno de más en el 4,3%, por
turnos truncados o repartidos entre lanzadores).

## Disponibilidad temporal

La ventana se arma con `game_date < día del juego`, **estricto**: una apertura
del propio día no entra. Y lo que demuestra la disponibilidad no es el nombre de
la tabla sino la comprobación de arriba: ninguna instantánea contiene un juego
posterior a su fecha, verificado contra el registro de lanzamientos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from fbq.model import aperturas as AP
from fbq.model.features import regress

# ── Constantes del preregistro §3, sin cambios ───────────────────────────
VENTANA_ABRIDOR = 40    # aperturas anteriores
K_BF = 300              # encogimiento hacia la liga, en bateadores enfrentados
MIN_BF_ABRIDOR = 150    # debajo de esto el juego se EXCLUYE, no se imputa
LIGA_K_MENOS_BB = 0.135 # K% − BB% de liga


@dataclass(frozen=True)
class Abridor:
    pitcher_id: int
    k: int
    bb: int
    bf: int
    n_aperturas: int

    @property
    def k_pct(self) -> float:
        return self.k / self.bf if self.bf else 0.0

    @property
    def bb_pct(self) -> float:
        return self.bb / self.bf if self.bf else 0.0

    @property
    def bruto(self) -> float:
        return self.k_pct - self.bb_pct

    def calidad(self, media_liga: float = LIGA_K_MENOS_BB) -> float:
        """`K% − BB%` encogido hacia la liga por bateadores enfrentados.

        Las tres cantidades se suman sobre EXACTAMENTE el mismo conjunto de
        aperturas, que es lo que evita mezclar poblaciones.
        """
        return regress(self.bruto, media_liga, self.bf, K_BF)


def hasta(pitcher_id: Optional[int], corte: str, *,
          ventana: int = VENTANA_ABRIDOR, db=None) -> Optional[Abridor]:
    """Las últimas `ventana` aperturas DISPONIBLES en `corte`.

    `corte` es un instante, no un día: la compuerta compara el fin medido de
    cada apertura más el margen preregistrado contra ese instante.
    """
    if pitcher_id is None:
        return None
    k, bb, bf, n = (AP.ventana(int(pitcher_id), corte, n=ventana, db=db)
                    if db is not None else
                    AP.ventana(int(pitcher_id), corte, n=ventana))
    if bf <= 0:
        return None
    return Abridor(int(pitcher_id), k, bb, bf, n)


def diferencia(
    local: Optional[Abridor], visita: Optional[Abridor],
    *, media_liga: float = LIGA_K_MENOS_BB,
) -> Tuple[Optional[float], str]:
    """`calidad(local) − calidad(visitante)`, o `(None, motivo)`.

    El abridor local somete a la ofensa visitante, así que un local mejor sube
    P(gana el local): positivo favorece al local, igual que `dif_pitagorica`.
    """
    if local is None or visita is None:
        return None, "sin_aperturas_previas"
    if local.bf < MIN_BF_ABRIDOR or visita.bf < MIN_BF_ABRIDOR:
        return None, "abridor_con_muestra_insuficiente"
    return local.calidad(media_liga) - visita.calidad(media_liga), "ok"
