"""
Sprint 3 — Gradient Descent Diagnostic
=======================================
Responde las tres preguntas del audit:

  P1: ¿Se llama _gradient_step en el backtest actual?
  P2: ¿Los gradientes son cero o no?
  P3: ¿Los pesos se guardan correctamente?

Methodology:
  - Lee N juegos completados de game_outcomes (con actual_home_runs, stage_factors_json, lambda)
  - Simula _gradient_step manualmente con logging detallado
  - Reporta: n_calls, n_skipped, grad_by_stage, delta_weight, before/after weights

Usage:
    python3 scripts/diagnose_gradient.py [--games 100] [--season 2024]
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "predictions_history.db"

# Mirror constants from learning_engine.py
_STAGE_KEYS = ["park", "hfa", "defense", "pitcher", "bullpen", "context"]
_LR         = 0.01
_MIN_WEIGHT = 0.30
_MAX_WEIGHT = 1.50


def _gradient_step_verbose(
    game_pk: int,
    actual_home: int,
    actual_away: int,
    season: int,
    weights: dict,
    conn: sqlite3.Connection,
) -> dict:
    """
    Mirrors learning_engine._gradient_step() with full logging.
    Returns dict with: gradients, deltas, skip_reason, updated_weights.
    """
    row = conn.execute(
        "SELECT lambda_home, lambda_away, stage_factors_json "
        "FROM game_outcomes WHERE game_pk = ?",
        (game_pk,),
    ).fetchone()

    if not row or not row["stage_factors_json"]:
        return {"skip": True, "reason": "no stage_factors_json in DB"}

    try:
        factors = json.loads(row["stage_factors_json"])
    except Exception:
        return {"skip": True, "reason": "invalid JSON in stage_factors_json"}

    lam_h, lam_a = row["lambda_home"], row["lambda_away"]
    if not lam_h or not lam_a or lam_h <= 0 or lam_a <= 0:
        return {"skip": True, "reason": f"invalid lambda: lh={lam_h} la={lam_a}"}

    gradients   = defaultdict(float)   # stage → total gradient (both roles)
    deltas      = defaultdict(float)   # stage → delta_weight
    adj_counts  = defaultdict(int)     # stage → n_times adj != 1.0
    skip_counts = defaultdict(int)     # stage → n_times skipped (adj==1.0)
    updated     = dict(weights)

    for role, lam_final, actual in [
        ("home", lam_h, actual_home),
        ("away", lam_a, actual_away),
    ]:
        for stage in _STAGE_KEYS:
            adj = factors.get(f"{role}_{stage}", 1.0)
            if adj == 1.0:
                skip_counts[stage] += 1
                continue
            adj_counts[stage] += 1
            w = weights[stage]
            denom = 1.0 + w * (adj - 1.0)
            if abs(denom) < 1e-6:
                skip_counts[stage] += 1
                continue
            grad = (lam_final - actual) * (adj - 1.0) / denom
            gradients[stage] += grad
            delta = -_LR * grad
            deltas[stage] += delta
            updated[stage] = updated.get(stage, 1.0) + delta

    # Clamp
    for stage in _STAGE_KEYS:
        updated[stage] = max(_MIN_WEIGHT, min(_MAX_WEIGHT, updated.get(stage, 1.0)))

    return {
        "skip":        False,
        "gradients":   dict(gradients),
        "deltas":      dict(deltas),
        "adj_counts":  dict(adj_counts),
        "skip_counts": dict(skip_counts),
        "updated":     updated,
    }


def main():
    parser = argparse.ArgumentParser(description="Gradient descent diagnostic")
    parser.add_argument("--games",  type=int, default=100, help="Number of games to diagnose")
    parser.add_argument("--season", type=int, default=2024, help="Season to analyse")
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # ── Check if _gradient_step was ever called in the backtest ──────────────
    # Proxy: if weights ever moved from 1.0, gradient_step was active.
    weights_in_db = conn.execute(
        "SELECT value_json FROM ml_state WHERE key='pipeline_weights' AND season=?",
        (args.season,),
    ).fetchone()

    print("=" * 68)
    print("  SPRINT 3 — GRADIENT DESCENT DIAGNOSTIC")
    print("=" * 68)
    print()

    if weights_in_db:
        w = json.loads(weights_in_db["value_json"])
        all_ones = all(abs(v - 1.0) < 1e-6 for v in w.values())
        print(f"P1 — ¿Se llamó _gradient_step en el backtest?")
        print(f"     Pesos en DB para season {args.season}:")
        for k, v in w.items():
            moved = "" if abs(v - 1.0) < 1e-6 else f"  ← MOVED from 1.0"
            print(f"       {k:<12} {v:.6f}{moved}")
        if all_ones:
            print(f"\n     DIAGNÓSTICO: Todos los pesos = 1.0 → _gradient_step NUNCA se llamó ❌")
        else:
            print(f"\n     DIAGNÓSTICO: Pesos divergieron de 1.0 → _gradient_step se llamó ✅")
    else:
        print(f"P1 — No hay pesos en DB para season {args.season} (no se pudo inicializar)")

    print()

    # ── Load N completed games ───────────────────────────────────────────────
    rows = conn.execute(
        """
        SELECT game_pk, season, actual_home_runs, actual_away_runs,
               lambda_home, lambda_away, stage_factors_json
        FROM game_outcomes
        WHERE season = ?
          AND actual_home_runs IS NOT NULL
          AND stage_factors_json IS NOT NULL
          AND lambda_home IS NOT NULL
        ORDER BY game_pk ASC
        LIMIT ?
        """,
        (args.season, args.games),
    ).fetchall()

    print(f"P2+P3 — Simulando _gradient_step en {len(rows)} juegos de temporada {args.season}")
    print(f"         (empezando desde pesos=1.0 — walk-forward simulado)")
    print()

    weights = {k: 1.0 for k in _STAGE_KEYS}
    n_calls   = 0
    n_skipped = 0
    all_grads     = defaultdict(list)
    all_deltas    = defaultdict(list)
    all_adj_counts = defaultdict(int)
    skip_reasons  = defaultdict(int)
    weight_history = []

    for row in rows:
        n_calls += 1
        result = _gradient_step_verbose(
            row["game_pk"],
            int(row["actual_home_runs"]),
            int(row["actual_away_runs"]),
            row["season"],
            weights,
            conn,
        )

        if result.get("skip"):
            n_skipped += 1
            skip_reasons[result["reason"]] += 1
            continue

        for stage in _STAGE_KEYS:
            if stage in result["gradients"]:
                all_grads[stage].append(result["gradients"][stage])
                all_deltas[stage].append(result["deltas"][stage])
            for stage in result["adj_counts"]:
                all_adj_counts[stage] += result["adj_counts"][stage]

        weights = result["updated"]
        weight_history.append(dict(weights))

    print(f"  n_games_processed : {n_calls}")
    print(f"  n_skipped         : {n_skipped}")
    if skip_reasons:
        for reason, count in skip_reasons.items():
            print(f"    skip reason: {reason!r} × {count}")
    print(f"  n_effective       : {n_calls - n_skipped}")
    print()

    print("P2 — Gradientes por stage:")
    print(f"  {'Stage':<12} {'N_adj':>6} {'Mean_grad':>12} {'Max|grad|':>12} {'Mean_Δw':>12} {'Δw_sign':>10}")
    print("  " + "-" * 64)
    for stage in _STAGE_KEYS:
        grads  = all_grads.get(stage, [])
        deltas = all_deltas.get(stage, [])
        n_adj  = all_adj_counts.get(stage, 0)
        if grads:
            mean_g  = sum(grads) / len(grads)
            max_g   = max(abs(g) for g in grads)
            mean_dw = sum(deltas) / len(deltas)
            sign    = "↑weight" if mean_dw > 0 else "↓weight"
            zero    = " ← ZERO" if max_g < 1e-8 else ""
            print(f"  {stage:<12} {n_adj:>6} {mean_g:>12.6f} {max_g:>12.6f} {mean_dw:>12.6f} {sign:>10}{zero}")
        else:
            print(f"  {stage:<12} {n_adj:>6}    (no adj≠1.0 — always skipped)")
    print()

    print("P3 — Evolución de pesos (walk-forward simulado):")
    print(f"  {'Stage':<12} {'Start':>8} {'After 50':>10} {'After N':>10} {'Total_Δ':>10} {'Moved?':>8}")
    print("  " + "-" * 56)
    start_w = {k: 1.0 for k in _STAGE_KEYS}
    mid_w   = weight_history[49] if len(weight_history) >= 50 else weights
    end_w   = weights
    for stage in _STAGE_KEYS:
        s = start_w.get(stage, 1.0)
        m = mid_w.get(stage, 1.0)
        e = end_w.get(stage, 1.0)
        delta = e - s
        moved = "✅ YES" if abs(delta) > 1e-4 else "❌ NO"
        print(f"  {stage:<12} {s:>8.4f} {m:>10.4f} {e:>10.4f} {delta:>10.4f} {moved:>8}")
    print()

    print("VEREDICTO:")
    effective = n_calls - n_skipped
    if effective == 0:
        print("  ❌ _gradient_step habría saltado TODOS los juegos — bug en DB/data")
    elif all(abs(end_w.get(s, 1.0) - 1.0) < 1e-4 for s in _STAGE_KEYS):
        print("  ⚠  Gradientes existen pero pesos no se movieron — posible bug de LR muy pequeño o gradientes cancelados")
    else:
        print(f"  ✅ Gradientes ≠ 0 y pesos se moverían si _gradient_step fuera llamado")
        print(f"     Bug confirmado: el backtest loop bypasea _gradient_step completamente.")
        print(f"     Fix: añadir learning._gradient_step(game_pk, actual_h, actual_a, season)")
        print(f"           en el loop del backtest, después de update_kalman (línea ~1336)")
    print()
    print("=" * 68)

    conn.close()


if __name__ == "__main__":
    main()
