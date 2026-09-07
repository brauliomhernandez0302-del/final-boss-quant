"""fbq/respaldo.py — respaldos consistentes, versionados y fuera del proyecto.

Existe porque este proyecto ya perdió datos irrecuperables **dos veces**, y las
dos por el mismo mecanismo: los triggers de append-only protegen las FILAS de un
UPDATE o un DELETE, y no protegen el ARCHIVO.

    2026-08-28  GANICUS borrado entero, sin copia
    2026-09-07  `rm -f data/prospectiva.db` antes de re-congelar; las emisiones
                originales (sha 277364404636b308) no existen en ningún respaldo,
                artefacto ni log — sólo sobrevive su metadato

## Tres decisiones, y las tres por eso

**Se usa la API de respaldo de SQLite**, no `cp`. `sqlite3.Connection.backup()`
toma una instantánea **consistente** aunque haya un escritor en curso; copiar el
archivo con el cron a mitad de una transacción produce una base corrupta que se
ve sana hasta que alguien la abre.

**Fuera del directorio de trabajo del proyecto.** Un respaldo que vive en
`data/` desaparece con el mismo `rm -rf` que borra lo que respalda.

**Versionados por instante**, nunca sobrescritos, con un manifiesto que anota
qué se copió, cuántas filas tenía y el resultado de `PRAGMA integrity_check`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

RAIZ = Path(__file__).parent.parent
# FUERA del árbol del proyecto, a propósito.
DESTINO = Path.home() / "respaldos-fbq"
MANIFIESTO = DESTINO / "manifiesto.jsonl"

BASES = ("prospectiva.db", "anuncios.db", "aperturas.db", "market.db", "results.db")


def _filas_por_tabla(con: sqlite3.Connection) -> Dict[str, int]:
    tablas = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tablas}


def respaldar(nombres: Iterable[str] = BASES, *, motivo: str = "manual",
              origen: Path = RAIZ / "data", destino: Path = DESTINO,
              ) -> List[Dict[str, Any]]:
    """Una instantánea consistente de cada base, con su manifiesto."""
    destino.mkdir(parents=True, exist_ok=True)
    sello = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    salida: List[Dict[str, Any]] = []
    for nombre in nombres:
        src = origen / nombre
        if not src.exists():
            continue
        dst = destino / f"{src.stem}.{sello}.db"
        con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        try:
            filas = _filas_por_tabla(con)
            with sqlite3.connect(dst) as copia:
                con.backup(copia)        # API de respaldo: instantánea consistente
        finally:
            con.close()
        rev = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
        try:
            integridad = rev.execute("PRAGMA integrity_check").fetchone()[0]
            filas_copia = _filas_por_tabla(rev)
        finally:
            rev.close()
        entrada = {
            "sello": sello, "motivo": motivo, "base": nombre,
            "archivo": str(dst), "bytes": dst.stat().st_size,
            "integridad": integridad,
            "filas_origen": filas, "filas_copia": filas_copia,
            "coincide": filas == filas_copia,
        }
        if integridad != "ok" or not entrada["coincide"]:
            raise RuntimeError(f"respaldo inválido de {nombre}: {entrada}")
        salida.append(entrada)
        # El manifiesto vive JUNTO a las copias, no en una ruta fija: si no, un
        # respaldo a otro destino queda sin registro y el manifiesto miente por
        # omisión. Lo encontró `test_los_respaldos_se_versionan_y_no_se_pisan`.
        manifiesto = destino / MANIFIESTO.name
        with manifiesto.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    return salida


def restaurar(base: str, hacia: Path, *, destino: Path = DESTINO,
              sello: Optional[str] = None) -> Path:
    """Copia un respaldo a `hacia`. **Nunca escribe sobre producción.**

    Un ensayo de restauración que sobrescribiera el archivo vivo sería un
    ensayo que puede destruir lo que verifica.
    """
    hacia = Path(hacia)
    if hacia.resolve() == (RAIZ / "data" / base).resolve():
        raise ValueError("restaurar sobre la base de producción está prohibido")
    copias = sorted(destino.glob(f"{Path(base).stem}.{sello or '*'}.db"))
    if not copias:
        raise FileNotFoundError(f"sin respaldos de {base} en {destino}")
    hacia.parent.mkdir(parents=True, exist_ok=True)
    origen = sqlite3.connect(f"file:{copias[-1]}?mode=ro", uri=True)
    try:
        with sqlite3.connect(hacia) as destino_con:
            origen.backup(destino_con)
    finally:
        origen.close()
    return hacia


def verificar_restauracion(base: str, *, destino: Path = DESTINO,
                           origen: Path = RAIZ / "data") -> Dict[str, Any]:
    """Restaura a un directorio temporal y compara contra producción.

    No alcanza con que el archivo abra: se comparan integridad, conteos, y —lo
    que de verdad importa acá— las probabilidades, los sellos de emisión y los
    SHA de los ajustes, fila por fila.
    """
    import tempfile
    carpeta = Path(tempfile.mkdtemp(prefix="fbq_restauracion_"))
    try:
        copia = restaurar(base, carpeta / base, destino=destino)
        viva = origen / base
        a = sqlite3.connect(f"file:{viva}?mode=ro", uri=True)
        b = sqlite3.connect(f"file:{copia}?mode=ro", uri=True)
        try:
            informe: Dict[str, Any] = {
                "base": base, "respaldo": str(copia),
                "integridad": b.execute("PRAGMA integrity_check").fetchone()[0],
                "filas": {"produccion": _filas_por_tabla(a),
                          "restaurada": _filas_por_tabla(b)},
            }
            informe["filas_coinciden"] = (
                informe["filas"]["produccion"] == informe["filas"]["restaurada"])
            tablas = set(informe["filas"]["produccion"])
            informe["esquema_coincide"] = tablas == set(informe["filas"]["restaurada"])
            if "prediccion" in tablas:
                cols = ("game_pk, version, corte, modelo_sha, p_home, "
                        "generado_utc, registrado_utc, origen")
                orden = " ORDER BY game_pk, version, corte, modelo_sha"
                fa = a.execute(f"SELECT {cols} FROM prediccion{orden}").fetchall()
                fb = b.execute(f"SELECT {cols} FROM prediccion{orden}").fetchall()
                informe["predicciones_identicas"] = fa == fb
                informe["n_predicciones"] = len(fa)
                informe["shas"] = sorted({r[3] for r in fa})
                informe["origenes"] = sorted({r[7] for r in fa})
                informe["con_sello_de_emision"] = sum(1 for r in fa if r[6])
            return informe
        finally:
            a.close(); b.close()
    finally:
        import shutil as _sh
        _sh.rmtree(carpeta, ignore_errors=True)


def ultimo(base: str, *, destino: Path = DESTINO) -> Optional[Path]:
    copias = sorted(destino.glob(f"{Path(base).stem}.*.db"))
    return copias[-1] if copias else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motivo", default="manual",
                    help="antes_de_migracion | tras_predicciones | manual")
    ap.add_argument("--bases", nargs="*", default=list(BASES))
    ap.add_argument("--verificar", metavar="BASE", default=None,
                    help="restaura esa base en un temporal y la compara")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.verificar:
        log.info("%s", json.dumps(verificar_restauracion(args.verificar),
                                  indent=1, ensure_ascii=False))
        return
    for e in respaldar(args.bases, motivo=args.motivo):
        log.info("  %s → %s · %s · %s bytes", e["base"], Path(e["archivo"]).name,
                 e["integridad"], e["bytes"])


if __name__ == "__main__":
    main()
