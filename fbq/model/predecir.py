"""Genera y guarda las predicciones pareadas de v1.2 y v1.4, antes del partido.

    python3 -m fbq.model.predecir --dias 3

Las dos versiones se calculan **en la misma pasada, con un solo reloj y una sola
lectura del almacén**. Dos procesos separados no garantizan eso: cada uno
llamaría a `ahora()` por su cuenta y leería los anuncios en un instante distinto,
y ahí las dos versiones dejan de ser comparables sin que nadie lo note. Es la
misma lección que GANICUS dejó escrita en su cron para v2/v3.

Si a un juego le falta cualquier insumo de cualquiera de las dos versiones,
**no se guarda ninguna de las dos**: una serie pareada con huecos de un lado
deja de ser pareada.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fbq.anuncios.store import AnunciosStore
from fbq.core.clock import ahora, normalizar_utc
from fbq.model import abridores as AB
from fbq.model import features as F
from fbq.model.pit import VentanaPIT, Partido, cargar_partidos, _inicios
from fbq.model.prospectiva import Prospectiva, aplicar, cargar_congelado
from fbq.results import fines as _fines
from fbq.sources import mlb_stats

log = logging.getLogger(__name__)
LOG = Path(__file__).parent.parent.parent / "logs" / "prospectiva.log"


def _juegos_futuros(dias: int) -> List[Dict[str, Any]]:
    hoy = date.today()
    fuera = []
    for i in range(dias):
        f = (hoy + timedelta(days=i)).isoformat()
        for g in mlb_stats.schedule(fecha=f, hidratar=""):
            eq = g.get("teams") or {}
            fuera.append({
                "game_pk": int(g["gamePk"]),
                "official_date": g.get("officialDate"),
                "commence_time": normalizar_utc(g["gameDate"]),
                "home_team": ((eq.get("home") or {}).get("team") or {}).get("name"),
                "away_team": ((eq.get("away") or {}).get("team") or {}).get("name"),
            })
    return fuera


def generar(dias: int = 3, *, store: Optional[Prospectiva] = None) -> Dict[str, Any]:
    store = store or Prospectiva()
    congelado = cargar_congelado()
    m12, m14 = congelado["modelos"]["v1.2"], congelado["modelos"]["v1.4"]
    anuncios = AnunciosStore()

    corte = normalizar_utc(ahora())
    temporada = int(corte[:4])
    partidos = cargar_partidos((temporada - 2, temporada - 1, temporada))
    ventana = VentanaPIT(partidos)
    indice = F.IndiceLiga(partidos)
    inicios = {pk: max(v) for pk, v in _inicios().items()}
    for pk, m in _fines.cargar().items():
        c = m.get("reanudacion") or m.get("inicio")
        if c:
            inicios[int(pk)] = normalizar_utc(c)

    filas: List[Dict[str, Any]] = []
    motivos: Counter = Counter()
    for g in _juegos_futuros(dias):
        if g["commence_time"] <= corte:
            motivos["ya_empezo"] += 1
            continue
        # Variables de equipo, al MISMO corte
        juego = Partido(g["game_pk"], g["official_date"], temporada,
                        g["home_team"], g["away_team"], 0, 0, None, "n/a")
        inicios[g["game_pk"]] = g["commence_time"]
        v = F.construir_fila(ventana, partidos, juego, corte, indice, inicios)
        if not v.get("ok"):
            motivos[str(v.get("motivo"))] += 1
            continue
        # Identidad del abridor: SÓLO lo anunciado antes del corte
        ident = {lado: anuncios.vigente_antes(g["game_pk"], lado, corte)
                 for lado in ("home", "away")}
        if any(r is None or r["pitcher_id"] is None for r in ident.values()):
            motivos["sin_abridor_anunciado_antes_del_corte"] += 1
            continue
        loc = AB.hasta(ident["home"]["pitcher_id"], corte)
        vis = AB.hasta(ident["away"]["pitcher_id"], corte)
        dif, motivo = AB.diferencia(loc, vis)
        if dif is None:
            motivos[motivo] += 1
            continue

        x = {"dif_pitagorica": v["dif_pitagorica"], "dif_descanso": v["dif_descanso"],
             "dif_calidad_abridor": dif}
        base = {"corte": corte, "game_pk": g["game_pk"],
                "official_date": g["official_date"],
                "commence_time": g["commence_time"], "home_team": g["home_team"],
                "away_team": g["away_team"], "cohorte": "prospectiva",
                "generado_utc": normalizar_utc(ahora())}
        for version, modelo in (("v1.2", m12), ("v1.4", m14)):
            filas.append({**base, "version": version,
                          "p_home": aplicar(modelo, x),
                          "variables_json": json.dumps(
                              {n: x[n] for n in modelo["variables"]}
                              | {"abridor_local": ident["home"]["pitcher_id"],
                                 "abridor_visita": ident["away"]["pitcher_id"],
                                 "anuncio_local_observado": ident["home"]["observado_en"],
                                 "anuncio_visita_observado": ident["away"]["observado_en"]}),
                          "modelo_sha": modelo["sha"]})

    nuevas = store.guardar(filas)
    return {"corte": corte, "candidatos": len(filas) // 2, "guardadas": nuevas,
            "exclusiones": dict(motivos), "almacen": store.resumen()}


def generar_historicas(
    *, cobertura: Path = Path("docs/cobertura_anuncios_2026-09-06.json"),
    store: Optional[Prospectiva] = None, precios=None,
) -> Dict[str, Any]:
    """La cohorte histórica de 42 juegos, guardada APARTE.

    Son los únicos juegos ya jugados con anuncio previo al corte verificable.
    Van con `cohorte='historica_42'` y **no se mezclan** con la serie
    prospectiva: su anuncio viene de otro capturador, su corte es un precio de
    hace un mes y sus estadísticas de abridor salen del almacén PIT en vez de la
    API. Mezclarlas sería juntar dos muestras distintas y llamarlas una.
    """
    import pickle as _pickle
    from fbq.model.candidato import _precios_de_referencia
    store = store or Prospectiva()
    congelado = cargar_congelado()
    m12, m14 = congelado["modelos"]["v1.2"], congelado["modelos"]["v1.4"]
    anuncios = AnunciosStore()
    det = json.loads(Path(cobertura).read_text(encoding="utf-8"))["detalle"]

    partidos = cargar_partidos((2024, 2025, 2026))
    ventana = VentanaPIT(partidos)
    indice = F.IndiceLiga(partidos)
    por_pk = {p.game_pk: p for p in partidos}
    inicios = {pk: max(v) for pk, v in _inicios().items()}
    for pk, m in _fines.cargar().items():
        c = m.get("reanudacion") or m.get("inicio")
        if c:
            inicios[int(pk)] = normalizar_utc(c)

    filas: List[Dict[str, Any]] = []
    motivos: Counter = Counter()
    for d in det:
        pk, corte = int(d["game_pk"]), d["corte"]
        juego = por_pk.get(pk)
        inicio = inicios.get(pk)
        if juego is None or inicio is None:
            motivos["sin_juego_o_inicio"] += 1
            continue
        # El reloj OFICIAL manda sobre el del proveedor de cuotas. Medido:
        # el `commence_time` de The Odds API va hasta un minuto después del
        # primer lanzamiento oficial, y por eso el filtro pre-juego del almacén
        # de precios dejó pasar 2 de estos 42 —823429 (+11 min) y 823516
        # (+46 min)—. Un precio observado con el partido ya empezado no es un
        # precio pre-juego, lo diga quien lo diga.
        if corte >= inicio:
            motivos["precio_posterior_al_inicio_oficial"] += 1
            continue
        v = F.construir_fila(ventana, partidos, juego, corte, indice, inicios)
        if not v.get("ok"):
            motivos[str(v.get("motivo"))] += 1
            continue
        ident = {lado: anuncios.vigente_antes(pk, lado, corte) for lado in ("home", "away")}
        if any(r is None or r["pitcher_id"] is None for r in ident.values()):
            motivos["sin_abridor_anunciado_antes_del_corte"] += 1
            continue
        loc = AB.hasta(ident["home"]["pitcher_id"], corte)
        vis = AB.hasta(ident["away"]["pitcher_id"], corte)
        dif, motivo = AB.diferencia(loc, vis)
        if dif is None:
            motivos[motivo] += 1
            continue
        x = {"dif_pitagorica": v["dif_pitagorica"], "dif_descanso": v["dif_descanso"],
             "dif_calidad_abridor": dif}
        base = {"corte": corte, "game_pk": pk, "official_date": juego.official_date,
                "commence_time": inicio, "home_team": juego.home_team,
                "away_team": juego.away_team, "cohorte": "historica_42",
                "generado_utc": normalizar_utc(ahora()),
                # La cohorte histórica se RECONSTRUYE hoy sobre cortes de hace
                # un mes. No es prospectiva y no puede contarse como tal.
                "origen": "reconstruccion"}
        for version, modelo in (("v1.2", m12), ("v1.4", m14)):
            filas.append({**base, "version": version, "p_home": aplicar(modelo, x),
                          "variables_json": json.dumps(
                              {n: x[n] for n in modelo["variables"]}
                              | {"abridor_local": ident["home"]["pitcher_id"],
                                 "abridor_visita": ident["away"]["pitcher_id"]}),
                          "modelo_sha": modelo["sha"]})
    return {"candidatos": len(filas) // 2, "guardadas": store.guardar(filas),
            "exclusiones": dict(motivos)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dias", type=int, default=3)
    ap.add_argument("--historicas", action="store_true",
                    help="genera además la cohorte histórica de 42, aparte")
    args = ap.parse_args()
    # El log en ARCHIVO es parte de la evidencia: una predicción cuya emisión
    # sólo consta en la salida de una terminal no es demostrable después.
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG, encoding="utf-8"),
                  logging.StreamHandler()], force=True)
    try:
        if args.historicas:
            log.info("cohorte histórica: %s",
                     json.dumps(generar_historicas(), ensure_ascii=False))
        log.info("%s", json.dumps(generar(args.dias), indent=1, ensure_ascii=False))
        # Respaldo DESPUÉS de escribir: una tanda de emisiones que no se
        # respalda es una tanda que un `rm` se lleva entera.
        from fbq.respaldo import respaldar
        for e in respaldar(["prospectiva.db"], motivo="tras_predicciones"):
            log.info("respaldo: %s · %s", Path(e["archivo"]).name, e["integridad"])
        return 0
    except Exception:                                     # noqa: BLE001
        log.exception("FALLÓ la generación prospectiva")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
