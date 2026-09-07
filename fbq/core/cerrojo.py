"""fbq/core/cerrojo.py — exclusión mutua entre corridas de cron.

Vive acá, y no dentro de un módulo de captura, porque el problema no es de
ningún módulo: **cualquier almacén que deduplique comparando contra lo que ya
tiene** se rompe con dos escritores simultáneos. Los dos leen el mismo estado
"todavía no está", y los dos insertan.

Es el modo de falla que costó una calibración entera en el sistema anterior
(`ml_state`, 2026-07-11): escritura concurrente sin protección, en silencio.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


class Ocupado(Exception):
    """Ya hay una corrida en curso sobre el mismo recurso."""


class Cerrojo:
    """Exclusión mutua por `flock`, no por existencia de archivo.

    Un cerrojo basado en "existe el archivo" deja el sistema trabado para
    siempre si el proceso muere; `flock` lo suelta el kernel cuando el proceso
    termina, pase lo que pase.
    """

    def __init__(self, ruta: Path, que: str = "corrida") -> None:
        self.ruta = Path(ruta)
        self.que = que
        self._fh = None

    def __enter__(self):
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.ruta, "w")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._fh.close()
            self._fh = None
            raise Ocupado(f"ya hay una {self.que} corriendo ({self.ruta})") from exc
        self._fh.write(str(os.getpid()))
        self._fh.flush()
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
        return False
