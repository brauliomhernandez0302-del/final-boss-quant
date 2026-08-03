# Fase 2A — Commit 4: cuarentena + cron instalado + primera corrida real

**Fecha**: 2026-07-19.

## 1. Flag de cuarentena

- `config.QUARANTINE_MODE` (default `True`, controlado por `QUARANTINE_MODE=false` en `.env`).
- `track_record/db.py`: nueva columna `publish_mode TEXT NOT NULL DEFAULT 'quarantine'` (migración idempotente).
- `track_record/publisher.py`: el filtro `MIN_TIER`/`_tier_ok()` se salta por completo mientras
  `QUARANTINE_MODE` es verdadero — se publica TODO lo que el pipeline genere, sin banda de EV
  (esa banda es de la Fase 2C). `publish_pick(publish_mode=...)` etiqueta cada fila.
- `track_record/ui.py`: banner de advertencia inequívoco (🔬 MODO CUARENTENA) al inicio de la
  página, más una columna "Modo" con badge por pick (🔬 Cuarentena / ✅ Publico) en la tabla.
- Tests: `tests/test_track_record_quarantine_mode.py` (bet de tier vacío SÍ publica en
  cuarentena, NO publica con `QUARANTINE_MODE=False`; `publish_pick()` por defecto etiqueta
  `quarantine`).

## 2. Hallazgo real durante la verificación previa al cron: `dry_run` nunca llegaba a `persist`

Antes de instalar el cron, verifiqué que la primera corrida real no reintrodujera la
contaminación del ledger (V1). Al revisar `game_outcomes` tras un dry-run reciente, until que
**`publish_mlb_picks()` nunca pasaba `persist` a `run_mlb()`** — sin importar `dry_run`, la
llamada real al pipeline siempre usaba el default `persist=True`. Es decir: el Commit 1 agregó
el parámetro `persist` a `run_module()`, pero nunca se conectó en el único call site real que
un `--dry-run` de `run_daily_picks.py` atraviesa. Un dry-run seguía escribiendo filas reales a
`game_outcomes` (source='live') exactamente como el batch de 46 filas que motivó el Commit 1.

**Arreglado**: `publish_mlb_picks()` ahora llama `run_mlb(..., persist=not dry_run)`. Test de
regresión nuevo: `tests/test_publisher_persist_wiring.py` (verifica el kwarg real que llega a
`run_mlb()`, no un mock que ignora sus propios argumentos — el gap anterior era exactamente
ese: mis tests previos sí mockeaban `run_module` pero nunca afirmaban sobre el valor de
`persist` recibido).

## 3. Crontab instalado

Servidor confirmado en `America/Los_Angeles` (`timedatectl`). Cron ya estaba activo (corriendo
23h antes de esta sesión) — no hizo falta `systemctl enable --now`.

```cron
0 7 * * *   cd /home/raulio && . mi_entorno/bin/activate && python run_daily_picks.py >> logs/daily_picks.log 2>&1
0 13 * * *  cd /home/raulio && . mi_entorno/bin/activate && python run_daily_picks.py --publish-only >> logs/daily_picks.log 2>&1
15 9 * * *  cd /home/raulio && . mi_entorno/bin/activate && python -m track_record.capture_closing_lines --game-date $(date +%F) >> logs/closing_lines.log 2>&1
30 15 * * * cd /home/raulio && . mi_entorno/bin/activate && python -m track_record.capture_closing_lines --game-date $(date +%F) >> logs/closing_lines.log 2>&1
30 18 * * * cd /home/raulio && . mi_entorno/bin/activate && python -m track_record.capture_closing_lines --game-date $(date +%F) >> logs/closing_lines.log 2>&1
```

Instalado con `crontab`, verificado con `crontab -l`. `logs/` ya existía.

**Costo de quota por sweep**: cada `capture_closing_lines` sweep hace como máximo 1 llamada a
`_get_raw_events()` por pick pendiente distinto (con cache de archivo, TTL propio de
`odds_fetcher.py` — no dispara una llamada nueva a la API por cada pick si el cache sigue
vigente). Con "última-pre-inicio gana", los 3 sweeps/día son redundantes por diseño (no
dependen de acertar el momento exacto) — si la cuota de la API empieza a ser un problema, el
primer candidato a quitar es el sweep de las 18:30 (el más tarde, con menos margen para que
valga la pena otra pasada antes de que la mayoría de los juegos nocturnos ya hayan empezado).

## 4. Primera corrida real (sin dry-run, en cuarentena)

Ejecutada 2026-07-19 ~13:12-13:14 PDT. Resultado: **13 picks publicados**, todos
`publish_mode='quarantine'`, todos con `commence_time` poblado correctamente (juegos de
2026-07-20 — todos los de HOY ya habían empezado o estaban a menos de `MIN_LEAD_MINUTES`,
15 juegos saltados por esa razón). EVs entre 1.73% y 105.09% (el más alto, `RL_AWAY` en
`MLB:824006`) — se dejaron todos publicados sin filtrar, tal como pide el diseño de cuarentena.

### Veredicto de V1 — NO CERRABLE HOY, evidencia parcial a favor de (a)

El objetivo era confirmar si `ml_home_pin`/`ml_away_pin` se pueblan en `game_outcomes` para los
juegos analizados en esta corrida real. Resultado: **`ml_home`/`ml_away` (y por tanto los
pins) salieron `None` para los 13 juegos** — pero esto NO es evidencia de un bug (b):

- Los 15 juegos de HOY (que sí tendrían mercado de moneyline maduro) fueron todos saltados por
  el gate de `MIN_LEAD_MINUTES` — ya habían empezado o estaban a minutos de empezar.
- Los únicos juegos analizables eran los de MAÑANA (2026-07-20), 20-30+ horas en el futuro —
  para esa distancia, ningún libro (ninguno de los ~30 en el feed) había posteado línea de
  moneyline todavía (`get_best_odds_for_teams` sí encontró el evento — "✅ Odds fetched" en el
  log, no "sin odds" — pero `ml_home`/`ml_away` quedaron en `None` porque el mercado `h2h` de
  ESE evento específico aún no tenía precios). Consistente con el propio razonamiento del
  Commit 1: "el batch de mañana suele predecirse antes de que Pinnacle postee la línea".
- **Evidencia real de que la tubería SÍ funciona** viene de horas antes en esta misma sesión
  (la corrida de `calibration_health()` del sweep de verificación): 19/19 y 12/12 filas de
  `game_outcomes` con pins poblados correctamente para juegos analizados cerca de su horario
  real (ver `audit_20260714/verificacion_operativa/reporte.md`, V1.2).
- **Veredicto comprometido**: (a) con evidencia fuerte pero indirecta (de antes en la sesión,
  no de esta corrida específica). La corrida de las 13:00 (`--publish-only`) o el sweep de
  cierre de mañana en la mañana (07:00, cuando los juegos de "mañana" ya sean los de "hoy" y
  estén más cerca) es la primera oportunidad real de confirmarlo con un dato fresco de esta
  cadena de cron. **No detengo el cron** — no hay evidencia de (b), solo ausencia de una
  oportunidad de prueba a esta hora del día.

## 5. Windows keepalive — comandos para que el dueño los ejecute (NO ejecutados aquí)

Los jobs de cron solo disparan mientras la instancia de WSL esté arriba. Reemplaza
`<NombreDistro>` por el nombre real (verificar en PowerShell con `wsl -l -v`).

```powershell
schtasks /create /tn "WSL Keepalive FBQ 0655" /tr "wsl.exe -d <NombreDistro> -- true" /sc daily /st 06:55 /f
schtasks /create /tn "WSL Keepalive FBQ 0910" /tr "wsl.exe -d <NombreDistro> -- true" /sc daily /st 09:10 /f
schtasks /create /tn "WSL Keepalive FBQ 1255" /tr "wsl.exe -d <NombreDistro> -- true" /sc daily /st 12:55 /f
schtasks /create /tn "WSL Keepalive FBQ 1525" /tr "wsl.exe -d <NombreDistro> -- true" /sc daily /st 15:25 /f
schtasks /create /tn "WSL Keepalive FBQ 1825" /tr "wsl.exe -d <NombreDistro> -- true" /sc daily /st 18:25 /f
```

Cada una dispara 5 minutos antes del cron correspondiente. Verificar con
`schtasks /query /tn "WSL Keepalive FBQ 0655"`. Esto es una decisión y acción del dueño — no
se ejecutó nada de esto desde esta sesión (no hay acceso al lado Windows de todos modos).

## 6. Test de convención de unidades (rider V3)

`tests/test_ev_unit_convention.py` — congela `calculate_ev()` en puntos porcentuales
(`calculate_ev(0.6, 2.0) == 20.0`, no `0.2`), y que `DEFAULT_MIN_EV`/los cortes de `ValueTier`
están en la misma escala.

## Suite final

551/551 tests verdes (incluyendo los 3 marcados `integration` del Commit 1).

## Estado final: reloj de CLV corriendo en cuarentena

- Cron instalado y verificado, cron.service activo.
- 13 picks reales publicados en cuarentena, con `commence_time` correcto.
- Closing-line sweeps corriendo 3x/día, "última-pre-inicio gana".
- V1 no cerrado con evidencia fresca de HOY (no había juegos cercanos analizables a esta hora),
  pero sin señal de bug — evidencia indirecta fuerte a favor de (a) de más temprano en la
  sesión. Primera oportunidad real de re-confirmarlo: la corrida de mañana 07:00.
