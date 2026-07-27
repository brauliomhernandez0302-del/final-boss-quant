#!/usr/bin/env python3
"""¿Corrió hoy lo que tenía que correr, y produjo lo que tenía que producir?

El paso 0 de la cadena (el disparador) no tenía forma de saber que algo dejó de
pasar. Evidencia del 2026-07-26: entre el 20 y el 26 de julio, de ~13 corridas
de publicación programadas, **3 no dejaron una sola línea en el log** (la
máquina estaba abajo) y **3 murieron con traceback** (el crash de `np.bool_` en
`pipeline_json`, arreglado en f4a4bf4). Siete días, tres con cero picks, y nadie
se enteró hasta que alguien fue a mirar por qué faltaban picks.

Este script contesta esa pregunta y **falla ruidoso** cuando la respuesta es no.
No arregla nada, no publica nada, no escribe en ninguna DB: solo mira y avisa.

Tres cosas que distingue, porque no son lo mismo:
  AUSENCIA  — el cron no corrió (no hay banner en el log). Falla de máquina.
  CRASH     — corrió y murió (traceback). Falla de código.
  VACÍO     — corrió bien y publicó 0 picks. Puede ser legítimo (toda la
              cartelera ya empezada), así que es WARN, no FAIL.

Uso:
    python3 scripts/cron_health_check.py            # el día de hoy
    python3 scripts/cron_health_check.py --date 2026-07-24
    python3 scripts/cron_health_check.py --quiet    # solo si hay problemas

Código de salida: 0 todo bien · 1 hay al menos un FAIL · 0 con WARN solamente.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOG = ROOT / "logs" / "daily_picks.log"
TR_DB = ROOT / "data" / "track_record.db"

# Las dos corridas de publicación del crontab, en hora local del servidor.
CORRIDAS_ESPERADAS = ("07:00", "13:00")

# Margen antes de dar por ausente una corrida: la publicación tarda ~5 min, así
# que a los 10 ya debería haber dejado su banner. Sin este margen el chequeo de
# las 07:15 reportaría FAIL todos los días por la corrida de las 13:00 que
# todavía no ocurrió — una alerta que grita a diario deja de ser una alerta.
MARGEN_MIN = 10

# Un pick cuyo juego arranca dentro de esta ventana ya debería tener su cierre
# capturado: si no lo tiene, la próxima barrida probablemente llegue tarde.
VENTANA_CIERRE_MIN = 90


def _lineas_del_dia(fecha: str) -> tuple[list[str], bool]:
    """Las líneas del log que pertenecen a `fecha`, y si la atribución es exacta.

    Desde 2026-07-26 `run_daily_picks.py` estampa la fecha en cada línea
    (`datefmt="%Y-%m-%d %H:%M:%S"`), así que filtrar es exacto. Para líneas
    anteriores a ese cambio solo hay hora, y una corrida del día D analiza juegos
    de D y D+1 — o sea que ni las fechas de juego del texto desambiguan. En ese
    caso se cae a una heurística y **se avisa**, en vez de dar un veredicto que
    no se puede sostener.

    Devuelve (líneas, exacto).
    """
    if not LOG.exists():
        return [], False
    texto = LOG.read_text(errors="replace").splitlines()

    # Camino exacto: líneas con prefijo de fecha.
    conf = [l for l in texto if l.startswith(fecha)]
    if conf:
        return conf, True
    # Ancla: la primera aparición de una fecha de juego == fecha o fecha+1.
    manana = (datetime.strptime(fecha, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    inicio = None
    for i, l in enumerate(texto):
        if f"juegos encontrados para {fecha}" in l or f"juegos encontrados para {manana}" in l:
            inicio = i
            break
    if inicio is None:
        return [], False
    # Fin: la primera ancla de un día posterior.
    fin = len(texto)
    for i in range(inicio, len(texto)):
        m = re.search(r"juegos encontrados para (\d{4}-\d{2}-\d{2})", texto[i])
        if m and m.group(1) > manana:
            fin = i
            break
    return texto[inicio:fin], False


def _matchups(bloque: list[str], patron: str) -> set[str]:
    """Matchups DISTINTOS ("AWAY @ HOME") que aparecen en las líneas que casan.

    El emparejador de odds se llama varias veces por juego y por día, así que
    la unidad honesta para reportarle a un operador es el juego, no la línea.
    """
    rx = re.compile(patron)
    return {m.group(1).strip() for l in bloque for m in [rx.search(l)] if m}


def _juegos_incompletos_con_pick(bloque, conn, fecha: str):
    """(matchups marcados NO APOSTAR, picks publicados tras esa marca).

    La atribución es por CORRIDA, no por día: un juego puede estar incompleto a
    las 07:00 (abridor sin anunciar) y completo a las 13:00, y publicar entonces
    es correcto. Sólo cuenta el estado más reciente ANTERIOR a cada pick.

    Requiere que las líneas del log tengan fecha (formato desde 2026-07-26); con
    el formato viejo no se puede fechar un estado y se devuelve vacío en vez de
    inventar una atribución.
    """
    rx = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?[⚠✅]️?\s+"
        r"(.+?) @ (.+?): Data (incompleta|completa)"
    )
    eventos = []
    for linea in bloque:
        m = rx.match(linea)
        if m:
            eventos.append((
                datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"),
                (m.group(2).strip(), m.group(3).strip()),
                m.group(4),
            ))
    incompletos = {e[1] for e in eventos if e[2] == "incompleta"}
    if not eventos:
        return incompletos, set()

    con_pick = set()
    filas = conn.execute(
        "SELECT pick_uid, home_team, away_team, published_at FROM picks "
        "WHERE date(published_at) = ?", (fecha,)
    ).fetchall()
    for fila in filas:
        try:
            t = datetime.fromisoformat(fila["published_at"].replace("Z", "")).replace(tzinfo=None)
        except Exception:
            continue
        clave = (fila["away_team"], fila["home_team"])
        previos = [e for e in eventos if e[1] == clave and e[0] <= t]
        if previos and max(previos, key=lambda e: e[0])[2] == "incompleta":
            con_pick.add(fila["pick_uid"])
    return incompletos, con_pick


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--quiet", action="store_true", help="callar si todo está bien")
    ap.add_argument("--at", default=None, metavar="HH:MM",
                    help="simular que son estas horas (para probar el chequeo)")
    args = ap.parse_args()
    fecha = args.date

    problemas: list[str] = []
    avisos: list[str] = []
    lineas: list[str] = []

    # ── 1. ¿Corrió? ────────────────────────────────────────────────────────
    bloque, exacto = _lineas_del_dia(fecha)
    if bloque and not exacto:
        avisos.append(
            "atribución APROXIMADA: estas líneas del log no tienen fecha (anteriores "
            "al cambio de formato del 2026-07-26), así que el bloque se delimitó por "
            "heurística y puede incluir líneas del día anterior."
        )
    banners = [l for l in bloque if "PUBLISH DAILY PICKS" in l]
    horas = {re.search(r"(\d{2}:\d{2}):\d{2}", l).group(1)
             for l in banners if re.search(r"(\d{2}:\d{2}):\d{2}", l)}
    # Solo se exige una corrida cuya hora programada ya pasó (con margen). Para
    # un día pasado se exigen todas.
    hoy = datetime.now().strftime("%Y-%m-%d")
    if fecha == hoy:
        ahora_local = datetime.strptime(args.at, "%H:%M") if args.at else datetime.now()
        ref = ahora_local.hour * 60 + ahora_local.minute
        exigibles = [h for h in CORRIDAS_ESPERADAS
                     if int(h[:2]) * 60 + int(h[3:]) + MARGEN_MIN <= ref]
    else:
        exigibles = list(CORRIDAS_ESPERADAS)
    faltantes = [h for h in exigibles if h not in horas]
    pendientes = [h for h in CORRIDAS_ESPERADAS if h not in exigibles]

    if not bloque:
        avisos.append(
            f"no pude segmentar el log para {fecha} (¿rotado, o el día aún no arrancó?) "
            "— este chequeo no puede afirmar nada sobre las corridas"
        )
    elif faltantes:
        problemas.append(
            f"AUSENCIA: no hay banner de publicación para {', '.join(faltantes)} "
            f"(encontradas: {sorted(horas) or 'ninguna'}). El cron no arrancó — "
            "revisar si la instancia estaba viva (keepalive de Windows)."
        )
    lineas.append(f"corridas de publicación halladas: {sorted(horas) or 'ninguna'}"
                  + (f"  (aún no exigibles: {', '.join(pendientes)})" if pendientes else ""))

    # ── 2. ¿Murió? ─────────────────────────────────────────────────────────
    crashes = sum(1 for l in bloque if "Traceback (most recent call last)" in l)
    if crashes:
        ultimo = next((l for l in reversed(bloque)
                       if re.match(r"^\w*Error|^\w*Exception", l.strip())), "")
        msg = (f"CRASH: {crashes} traceback(s) en el log de {fecha}"
               + (f" — último: {ultimo.strip()[:120]}" if ultimo else ""))
        # Con atribución aproximada, un traceback puede ser de la corrida del día
        # anterior (una corrida del día D analiza juegos de D y D+1, así que el
        # ancla heurística los solapa). No se afirma un fallo que la evidencia no
        # sostiene: se avisa. Con el formato de log fechado esto pasa a FAIL real.
        (problemas if exacto else avisos).append(
            msg + ("" if exacto else " — ATRIBUCIÓN APROXIMADA, puede ser del día anterior")
        )
    lineas.append(f"tracebacks: {crashes}")

    # ── 2b. ¿Se abstuvo de emparejar algún juego con su mercado? ───────────
    # `get_best_odds_for_teams` resuelve identidad con una heurística de parecido
    # (nombres + ventana de ±6h) y devuelve {} tanto cuando no hay mercado todavía
    # —normal a las 07:00 para los juegos de mañana— como cuando no supo cuál de
    # dos eventos es el correcto. Los dos casos son indistinguibles para quien
    # llama, así que se cuentan acá desde el log, que sí los separa.
    #
    # Se cuentan MATCHUPS DISTINTOS, no líneas: el emparejador se invoca dos
    # veces por juego y por corrida (run_module.py cuando el selector no le pasó
    # odds, y publisher.py para las columnas de mercado), y el día tiene dos
    # corridas — así que contar líneas infla el número ~4x y le haría buscar al
    # operador cuatro juegos donde hay uno. Verificado sobre logs/daily_picks.log.
    ambiguos = _matchups(bloque, r"ambiguous match for (.+?) —")
    sin_mercado = _matchups(bloque, r"no odds event for (.+?) within")
    lineas.append(
        f"juegos sin mercado (normal a esta hora para los de mañana): {len(sin_mercado)}"
    )
    if ambiguos:
        avisos.append(
            f"IDENTIDAD AMBIGUA: {len(ambiguos)} juego(s) tenían 2 eventos de odds "
            "casi igual de cerca y el sistema se abstuvo de elegir — casi siempre un "
            "doubleheader tradicional, cuyos dos juegos el schedule pone a 5 min uno "
            "del otro. Quedaron sin precio y sin pick, que es lo correcto, pero son "
            f"picks que NO se hicieron: {', '.join(sorted(ambiguos))}"
        )
    lineas.append(f"emparejamientos ambiguos (abstenciones): {len(ambiguos)}")

    conn = sqlite3.connect(f"file:{TR_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # ── 2c. ¿Se apostó sobre un juego que el propio pipeline marcó NO APOSTAR? ──
    # `get_complete_game_data` calcula `pitchers_valid` y, cuando es falso, loguea
    # "Data incompleta - NO APOSTAR" — pero NADIE lee ese flag: no existe ningún
    # gate que lo consuma (auditado el 2026-07-27, paso 4). El caso dominante es
    # benigno (abridor todavía sin anunciar, y la corrida de las 13:00 ya lo tiene),
    # así que en vez de suprimir el 20% de los juegos se mide lo que de verdad
    # importa: si algún pick salió de una corrida que había marcado ESE juego.
    incompletos, con_pick = _juegos_incompletos_con_pick(bloque, conn, fecha)
    lineas.append(f"juegos marcados 'NO APOSTAR' (abridor sin anunciar o sin stats): {len(incompletos)}")
    if con_pick:
        problemas.append(
            f"APUESTA SOBRE DATO INCOMPLETO: {len(con_pick)} pick(s) se publicaron sobre "
            "juegos que la MISMA corrida marcó 'Data incompleta - NO APOSTAR'. El flag "
            "`pitchers_valid` existe y nadie lo consume, así que el pipeline apostó con "
            "el abridor caído a fallback. Picks: " + ", ".join(sorted(con_pick))
        )

    # ── 3. ¿Produjo? ───────────────────────────────────────────────────────
    n_picks = conn.execute(
        "SELECT COUNT(*) FROM picks WHERE date(published_at) = ?", (fecha,)
    ).fetchone()[0]
    lineas.append(f"picks publicados: {n_picks}")
    if n_picks == 0 and not faltantes and not crashes and bloque and not pendientes:
        avisos.append(
            "VACÍO: el cron corrió sin error y publicó 0 picks. Puede ser legítimo "
            "(cartelera ya empezada, o sin odds), pero conviene mirarlo."
        )

    # ── 4. ¿Se le va a escapar un cierre? ──────────────────────────────────
    ahora = datetime.now(timezone.utc)
    limite = (ahora + timedelta(minutes=VENTANA_CIERRE_MIN)).isoformat()
    porvencer = conn.execute(
        "SELECT COUNT(*) FROM picks WHERE closing_captured_at IS NULL "
        "AND commence_time IS NOT NULL AND commence_time > ? AND commence_time <= ?",
        (ahora.isoformat(), limite),
    ).fetchone()[0]
    lineas.append(f"picks sin cierre capturado que arrancan en <{VENTANA_CIERRE_MIN}min: {porvencer}")
    if porvencer:
        problemas.append(
            f"CIERRE EN RIESGO: {porvencer} pick(s) arrancan en menos de "
            f"{VENTANA_CIERRE_MIN} minutos y todavía no tienen precio de cierre. "
            "Una vez que empieza el juego ese dato es irrecuperable — correr "
            "`python3 -m track_record.capture_closing_lines` ahora."
        )
    conn.close()

    # ── Veredicto ──────────────────────────────────────────────────────────
    if args.quiet and not problemas and not avisos:
        return 0

    print(f"=== salud del cron — {fecha} ===")
    for l in lineas:
        print(f"  {l}")
    for a in avisos:
        print(f"\n  ⚠ WARN  {a}")
    for p in problemas:
        print(f"\n  ✖ FAIL  {p}")
    if not problemas and not avisos:
        print("\n  ✓ todo en orden")
    return 1 if problemas else 0


if __name__ == "__main__":
    raise SystemExit(main())
