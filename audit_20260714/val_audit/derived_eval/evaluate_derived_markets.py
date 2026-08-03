#!/usr/bin/env python3
"""
Evaluador distribucional de mercados derivados — runline y total.

Qué mide
--------
`backtest_and_retrain.py` evalúa EXCLUSIVAMENTE moneyline (Brier/accuracy/ROI
sobre `home_won`). El fix del simulador de b3325a5 (truncamiento de walk-off,
rho_game real, empates proporcionales) corrige un sesgo que vive en la COLA de
la distribución de carreras — margen y total — no en el signo de quién gana.
Este script es el instrumento que sí mide eso.

Cómo
----
Toma las λ ya guardadas por la corrida canónica del backtest
(`game_outcomes.backtest_lambda_home/away`), re-simula cada juego DOS veces —
con el simulador viejo (pre-b3325a5, vendorizado en `_vendor/`) y con el actual —
usando el MISMO seed por `game_pk` (comparación pareada, cero ruido de MC en el
delta), y compara ambas distribuciones contra `actual_home_runs`/`actual_away_runs`.

Las λ son entrada del simulador, no salida: los tres cambios de b3325a5 viven
enteramente dentro del Monte Carlo, así que re-simular sobre las λ guardadas
aísla el efecto del fix de forma exacta. Ninguna λ se recalcula acá.

CALIBRACIÓN, NO ROI
-------------------
No hay líneas históricas de runline/total en esta DB (`game_outcomes` solo tiene
moneyline: `ml_*_open/cons/pin`). Sin líneas y sin precios no existe ni "cubrir
el spread del mercado" ni EV ni ROI — cualquier número de rentabilidad acá sería
inventado. Este evaluador mide EXCLUSIVAMENTE qué tan bien la distribución
simulada describe los resultados reales: sesgo, Brier, ECE, pendiente de
calibración, RPS y PIT. Un fix puede mejorar la calibración y aun así no ser
rentable; esa pregunta necesita líneas históricas y este instrumento no la toca.

Walk-forward
------------
Las λ evaluadas vienen de la corrida walk-forward del backtest (cada juego
predicho solo con datos anteriores a su `official_date`, post-Fase 2B), así que
la propiedad se hereda. Además, acá NO se ajusta ningún parámetro contra los
resultados: no hay Platt, ni bias, ni fit de ninguna clase — todas las
probabilidades salen de la simulación cruda. No hay superficie de sobreajuste
que un split temporal pudiera esconder; los cortes por temporada y por mes se
reportan igual, para que una mejora agregada no pueda tapar deriva local.

Uso
---
    python3 audit_20260714/val_audit/derived_eval/evaluate_derived_markets.py
    python3 ... --n-sims 50000 --seasons 2024,2025 --limit 200   # smoke test

Salidas (en `results/`): `per_game_<ts>.csv` (una fila por juego, ambas
variantes) y `metrics_<ts>.json` (todas las agregaciones).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

# El simulador viejo es el del commit padre del fix. Está vendorizado para que
# esta medición sea repetible sin depender del estado del working tree.
OLD_SIM_COMMIT = "b3325a5"          # el fix; el viejo es su padre (b3325a5^)
OLD_SIM_PATH = "modules/baseball_module/montecarlo/simulator.py"
VENDORED_OLD_SIM = HERE / "_vendor" / "simulator_pre_b3325a5.py"

# Líneas de total evaluadas. Solo medias líneas: sin push, cada una es un
# binario limpio. Cubren el rango realista de totales de MLB.
TOTAL_LINES: Tuple[float, ...] = (6.5, 7.5, 8.5, 9.5, 10.5, 11.5)

# Runline estándar de MLB: ±1.5 carreras.
RUNLINE = 1.5


# ── Carga del simulador viejo ──────────────────────────────────────────────────

def load_old_simulator(verify_against_git: bool = True):
    """Importa el simulador pre-fix desde el archivo vendorizado.

    Si git está disponible, verifica que el vendor sea byte-idéntico a
    `b3325a5^:...` — si alguien lo editó a mano, esta corrida no mide lo que
    dice medir, y es mejor fallar ruidoso que reportar un delta falso.
    """
    if not VENDORED_OLD_SIM.exists():
        raise SystemExit(
            f"Falta {VENDORED_OLD_SIM}. Reconstruilo con:\n"
            f"  git show {OLD_SIM_COMMIT}^:{OLD_SIM_PATH} > {VENDORED_OLD_SIM}"
        )

    vendored_bytes = VENDORED_OLD_SIM.read_bytes()
    vendor_sha = hashlib.sha256(vendored_bytes).hexdigest()

    if verify_against_git:
        try:
            from_git = subprocess.run(
                ["git", "show", f"{OLD_SIM_COMMIT}^:{OLD_SIM_PATH}"],
                cwd=REPO, capture_output=True, check=True,
            ).stdout
            if from_git != vendored_bytes:
                raise SystemExit(
                    f"{VENDORED_OLD_SIM} NO coincide con {OLD_SIM_COMMIT}^:{OLD_SIM_PATH}. "
                    "El 'simulador viejo' de esta corrida no sería el real — abortando."
                )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  [aviso] no se pudo verificar el vendor contra git; sigo con el archivo local")

    spec = importlib.util.spec_from_file_location("_sim_pre_b3325a5", VENDORED_OLD_SIM)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.monte_carlo_advanced, vendor_sha


# ── Carga de juegos ────────────────────────────────────────────────────────────

def load_games(
    db_path: Path,
    seasons: Sequence[int],
    run_date: Optional[str],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    """Juegos con λ de backtest guardadas + resultado real.

    `run_date` filtra por el día de la corrida canónica (`backtest_run_at`), para
    no mezclar λ de corridas viejas del mismo juego — las columnas `backtest_*`
    se sobrescriben corrida a corrida por diseño (CHRON-001).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    where = [
        "backtest_lambda_home IS NOT NULL",
        "backtest_lambda_away IS NOT NULL",
        "actual_home_runs IS NOT NULL",
        "actual_away_runs IS NOT NULL",
        f"season IN ({','.join('?' * len(seasons))})",
    ]
    params: List[Any] = list(seasons)
    if run_date:
        where.append("substr(backtest_run_at, 1, 10) = ?")
        params.append(run_date)

    sql = (
        "SELECT game_pk, season, official_date, game_date, home_team, away_team, "
        "       backtest_lambda_home, backtest_lambda_away, "
        "       actual_home_runs, actual_away_runs, home_won "
        f"FROM game_outcomes WHERE {' AND '.join(where)} "
        "ORDER BY COALESCE(official_date, date(game_date)), game_pk"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"

    rows = [dict(r) for r in conn.execute(sql, params)]
    conn.close()
    return rows


# ── Métricas por juego ─────────────────────────────────────────────────────────

def _rps(pmf: np.ndarray, observed: int) -> float:
    """Ranked Probability Score sobre soporte entero.

    Para una variable de conteo, el CRPS colapsa exactamente a
    Σ_k (F(k) − 1{y ≤ k})². Es una regla de puntuación propia sobre la
    distribución COMPLETA (no solo un umbral), así que castiga tanto el sesgo
    como la dispersión mal calibrada.
    """
    k_max = max(len(pmf) - 1, observed)
    cdf = np.zeros(k_max + 1)
    cdf[: len(pmf)] = np.cumsum(pmf)
    cdf[len(pmf):] = 1.0
    step = (np.arange(k_max + 1) >= observed).astype(float)
    return float(np.sum((cdf - step) ** 2))


def _pit(pmf: np.ndarray, observed: int, u: float) -> float:
    """PIT aleatorizado: F(y−1) + u·p(y). Uniforme(0,1) sii está calibrado.

    La aleatorización es la corrección estándar para variables discretas (sin
    ella el PIT es escalonado y no puede ser uniforme ni con el modelo perfecto).
    `u` viene de un RNG con seed fijo, así que la corrida es reproducible.
    """
    if observed >= len(pmf):
        return 1.0
    below = float(np.sum(pmf[:observed]))
    return below + u * float(pmf[observed])


def variant_metrics(home: np.ndarray, away: np.ndarray, p_home_sim: float,
                    actual_h: int, actual_a: int, u_draws: Dict[str, float]) -> Dict[str, float]:
    """Todo lo que se necesita de una variante del simulador para un juego."""
    n = home.size
    margin = home - away          # >0 = gana local
    total = home + away

    out: Dict[str, float] = {
        "p_ml_home": float(p_home_sim),
        "p_rl_home_cover": float(np.count_nonzero(margin >= 2) / n),   # local −1.5
        "p_rl_away_cover": float(np.count_nonzero(margin <= -2) / n),  # visita −1.5
        "p_home_win_nontie": float(np.count_nonzero(margin >= 1) / n),
        "p_tie_regulation": float(np.count_nonzero(margin == 0) / n),
        "mean_total": float(total.mean()),
        "std_total": float(total.std()),
        "mean_margin": float(margin.mean()),
        "std_margin": float(margin.std()),
    }
    for line in TOTAL_LINES:
        out[f"p_over_{line}"] = float(np.count_nonzero(total > line) / n)

    total_pmf = np.bincount(total, minlength=1) / n
    m_off = int(margin.min())
    margin_pmf = np.bincount(margin - m_off, minlength=1) / n

    out["rps_total"] = _rps(total_pmf, actual_h + actual_a)
    out["rps_margin"] = _rps(margin_pmf, max(0, (actual_h - actual_a) - m_off))
    out["pit_total"] = _pit(total_pmf, actual_h + actual_a, u_draws["total"])
    out["pit_margin"] = _pit(margin_pmf, max(0, (actual_h - actual_a) - m_off), u_draws["margin"])

    home_pmf = np.bincount(home, minlength=1) / n
    away_pmf = np.bincount(away, minlength=1) / n
    out["pit_home_runs"] = _pit(home_pmf, actual_h, u_draws["home"])
    out["pit_away_runs"] = _pit(away_pmf, actual_a, u_draws["away"])
    out["rps_home_runs"] = _rps(home_pmf, actual_h)
    out["rps_away_runs"] = _rps(away_pmf, actual_a)
    return out


# ── Agregación ─────────────────────────────────────────────────────────────────

def _logistic_calibration(p: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    """Pendiente/intercepto de y ~ 1 + logit(p) por IRLS.

    Calibrado perfecto = pendiente 1.0, intercepto 0.0. Pendiente <1 significa
    probabilidades demasiado extremas (sobre-confianza); intercepto ≠0 es sesgo
    direccional que sobrevive al reescalado.
    """
    if y.size < 30 or len(np.unique(y)) < 2:
        return {"slope": None, "intercept": None}
    eps = 1e-6
    x = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(100):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        try:
            beta_new = np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ z)
        except np.linalg.LinAlgError:
            return {"slope": None, "intercept": None}
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


def binary_market_metrics(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    eps = 1e-9
    pc = np.clip(p, eps, 1 - eps)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins, ece, mce = [], 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        cnt = int(np.count_nonzero(m))
        if not cnt:
            continue
        mp, obs = float(p[m].mean()), float(y[m].mean())
        gap = abs(mp - obs)
        ece += cnt / p.size * gap
        mce = max(mce, gap)
        bins.append({"bin": f"[{edges[b]:.1f},{edges[b+1]:.1f})", "n": cnt,
                     "mean_pred": mp, "observed": obs, "gap_pp": (mp - obs) * 100})

    return {
        "n": int(p.size),
        "mean_pred": float(p.mean()),
        "observed_rate": float(y.mean()),
        "bias_pp": float((p.mean() - y.mean()) * 100),
        "brier": brier,
        "logloss": logloss,
        "ece": float(ece),
        "mce": float(mce),
        **_logistic_calibration(p, y),
        "reliability": bins,
    }


def paired_brier_test(p_old: np.ndarray, p_new: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Test pareado sobre la contribución por juego al Brier (viejo − nuevo).

    Pareado porque ambas variantes corrieron con el mismo seed sobre las mismas
    λ: la diferencia por juego no arrastra ruido de MC ni de selección.
    """
    d = (p_old - y) ** 2 - (p_new - y) ** 2      # >0 = el nuevo es mejor
    res = stats.ttest_rel((p_old - y) ** 2, (p_new - y) ** 2)
    return {
        "mean_diff_old_minus_new": float(d.mean()),
        "t_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "n_games_new_better": int(np.count_nonzero(d > 0)),
        "n_games_old_better": int(np.count_nonzero(d < 0)),
    }


def pit_uniformity(u: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    ks = stats.kstest(u, "uniform")
    counts, _ = np.histogram(u, bins=n_bins, range=(0.0, 1.0))
    expected = np.full(n_bins, u.size / n_bins)
    chi2 = stats.chisquare(counts, expected)
    return {
        "n": int(u.size),
        "mean": float(u.mean()),          # 0.5 si está calibrado
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "chi2_stat": float(chi2.statistic),
        "chi2_pvalue": float(chi2.pvalue),
        "bin_counts": counts.tolist(),
        "bin_share": (counts / u.size).tolist(),
    }


def aggregate(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Todas las métricas de un subconjunto de juegos, para ambas variantes."""
    if not rows:
        return {"slice": label, "n": 0}

    a_h = np.array([r["actual_home_runs"] for r in rows], dtype=int)
    a_a = np.array([r["actual_away_runs"] for r in rows], dtype=int)
    a_margin = a_h - a_a
    a_total = a_h + a_a

    markets: Dict[str, np.ndarray] = {
        "RL_HOME_-1.5": (a_margin >= 2).astype(float),
        "RL_AWAY_-1.5": (a_margin <= -2).astype(float),
        "ML_HOME (control)": (a_margin > 0).astype(float),
    }
    for line in TOTAL_LINES:
        markets[f"TOTAL_OVER_{line}"] = (a_total > line).astype(float)

    pred_key = {
        "RL_HOME_-1.5": "p_rl_home_cover",
        "RL_AWAY_-1.5": "p_rl_away_cover",
        "ML_HOME (control)": "p_ml_home",
        **{f"TOTAL_OVER_{L}": f"p_over_{L}" for L in TOTAL_LINES},
    }

    out: Dict[str, Any] = {
        "slice": label,
        "n": len(rows),
        "actual": {
            "mean_total": float(a_total.mean()),
            "std_total": float(a_total.std()),
            "mean_margin": float(a_margin.mean()),
            "std_margin": float(a_margin.std()),
            "home_win_rate": float(np.mean(a_margin > 0)),
            "rate_margin_ge2": float(np.mean(a_margin >= 2)),
        },
        "markets": {},
        "distributional": {},
        "moments": {},
    }

    for variant in ("old", "new"):
        out["moments"][variant] = {
            "mean_total": float(np.mean([r[f"{variant}_mean_total"] for r in rows])),
            "mean_margin": float(np.mean([r[f"{variant}_mean_margin"] for r in rows])),
            "mean_std_total": float(np.mean([r[f"{variant}_std_total"] for r in rows])),
            "mean_p_tie_regulation": float(np.mean([r[f"{variant}_p_tie_regulation"] for r in rows])),
            "rps_total": float(np.mean([r[f"{variant}_rps_total"] for r in rows])),
            "rps_margin": float(np.mean([r[f"{variant}_rps_margin"] for r in rows])),
            "rps_home_runs": float(np.mean([r[f"{variant}_rps_home_runs"] for r in rows])),
            "rps_away_runs": float(np.mean([r[f"{variant}_rps_away_runs"] for r in rows])),
        }
        out["distributional"][variant] = {
            field: pit_uniformity(np.array([r[f"{variant}_pit_{field}"] for r in rows]))
            for field in ("total", "margin", "home_runs", "away_runs")
        }

    for market, y in markets.items():
        p_old = np.array([r[f"old_{pred_key[market]}"] for r in rows])
        p_new = np.array([r[f"new_{pred_key[market]}"] for r in rows])
        out["markets"][market] = {
            "old": binary_market_metrics(p_old, y),
            "new": binary_market_metrics(p_new, y),
            "paired_brier_test": paired_brier_test(p_old, p_new, y),
        }

    # VAL-1.3 replicado: P(margen local ≥2 | el local ganó). Pooled — el modelo
    # aporta Σ P(m≥2) / Σ P(m≥1); el real, la fracción de victorias locales por 2+.
    for variant in ("old", "new"):
        num = float(np.sum([r[f"{variant}_p_rl_home_cover"] for r in rows]))
        den = float(np.sum([r[f"{variant}_p_home_win_nontie"] for r in rows]))
        out.setdefault("val13_margin_ge2_given_home_won", {})[variant] = num / den if den else None
    won = a_margin > 0
    out["val13_margin_ge2_given_home_won"]["actual"] = (
        float(np.mean(a_margin[won] >= 2)) if np.any(won) else None
    )
    return out


# ── Corrida ────────────────────────────────────────────────────────────────────

def simulate_all(games: List[Dict[str, Any]], n_sims: int, old_mc, new_mc,
                 progress_every: int = 250) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    block = min(10_000, n_sims)
    t0 = time.time()

    for i, g in enumerate(games, start=1):
        pk = int(g["game_pk"])
        seed = pk % (2 ** 32)      # mismo esquema de seed que backtest_and_retrain.py
        # Los u del PIT se sortean de un stream aparte, fijado por game_pk, para
        # que no dependan de cuántas variantes corrieron ni en qué orden.
        rng_u = np.random.default_rng(seed ^ 0x5F3759DF)
        u_draws = {k: float(rng_u.random()) for k in ("total", "margin", "home", "away")}

        lh = float(g["backtest_lambda_home"])
        la = float(g["backtest_lambda_away"])

        row: Dict[str, Any] = {
            "game_pk": pk,
            "season": g["season"],
            "date": g["official_date"] or str(g["game_date"])[:10],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "lambda_home": lh,
            "lambda_away": la,
            "actual_home_runs": int(g["actual_home_runs"]),
            "actual_away_runs": int(g["actual_away_runs"]),
        }

        for variant, mc in (("old", old_mc), ("new", new_mc)):
            res = mc(lh=lh, la=la, n_max=n_sims, block=block, rng_seed=seed,
                     analyze_f5=False, store_samples=True)
            m = variant_metrics(
                np.asarray(res["home_samples"], dtype=int),
                np.asarray(res["away_samples"], dtype=int),
                res["p_home"],
                row["actual_home_runs"], row["actual_away_runs"], u_draws,
            )
            for k, v in m.items():
                row[f"{variant}_{k}"] = v

        rows.append(row)
        if i % progress_every == 0 or i == len(games):
            el = time.time() - t0
            print(f"  {i}/{len(games)} juegos  ({el:.0f}s, {el / i * 1000:.0f} ms/juego)", flush=True)

    return rows


def walkoff_sweep(games: List[Dict[str, Any]], shares: Sequence[float], n_sims: int,
                  new_sim_mod) -> List[Dict[str, Any]]:
    """Sensibilidad de `WALKOFF_9TH_SHARE` — MEDICIÓN, no un cambio.

    `WALKOFF_9TH_SHARE=1/9` es una constante ASUMIDA (no existe un split real por
    inning en este repo; así está documentado en el docstring del simulador).
    Este barrido reporta, sobre los mismos juegos y con el mismo seed, qué le
    pasa al sesgo de runline/total con otros valores — para que la decisión de
    tocarla (o no) tenga evidencia en vez de intuición. Igual que MATH-003: un
    residual que implica otro valor se REPORTA, no se cambia acá. El parche del
    módulo es in-process y se revierte al terminar; nada se persiste.
    """
    original = new_sim_mod.WALKOFF_9TH_SHARE
    mc = new_sim_mod.monte_carlo_advanced
    block = min(10_000, n_sims)
    out: List[Dict[str, Any]] = []

    a_h = np.array([int(g["actual_home_runs"]) for g in games])
    a_a = np.array([int(g["actual_away_runs"]) for g in games])
    a_margin = a_h - a_a
    a_total = a_h + a_a
    obs_rl_home = float(np.mean(a_margin >= 2))
    obs_total = float(a_total.mean())
    won = a_margin > 0
    obs_val13 = float(np.mean(a_margin[won] >= 2))

    try:
        for share in shares:
            new_sim_mod.WALKOFF_9TH_SHARE = float(share)
            t0 = time.time()
            p_rl, p_win, totals, margins = [], [], [], []
            for g in games:
                res = mc(lh=float(g["backtest_lambda_home"]), la=float(g["backtest_lambda_away"]),
                         n_max=n_sims, block=block, rng_seed=int(g["game_pk"]) % (2 ** 32),
                         analyze_f5=False, store_samples=True)
                h = np.asarray(res["home_samples"], dtype=int)
                a = np.asarray(res["away_samples"], dtype=int)
                m = h - a
                p_rl.append(np.count_nonzero(m >= 2) / m.size)
                p_win.append(np.count_nonzero(m >= 1) / m.size)
                totals.append(float((h + a).mean()))
                margins.append(float(m.mean()))
            p_rl_arr = np.array(p_rl)
            row = {
                "walkoff_9th_share": float(share),
                "is_current_default": abs(float(share) - original) < 1e-12,
                "rl_home_mean_pred": float(p_rl_arr.mean()),
                "rl_home_bias_pp": float((p_rl_arr.mean() - obs_rl_home) * 100),
                "rl_home_brier": float(np.mean((p_rl_arr - (a_margin >= 2)) ** 2)),
                "val13_model": float(p_rl_arr.sum() / np.sum(p_win)),
                "val13_gap_pp": float((p_rl_arr.sum() / np.sum(p_win) - obs_val13) * 100),
                "mean_total": float(np.mean(totals)),
                "mean_total_bias": float(np.mean(totals) - obs_total),
                "mean_margin": float(np.mean(margins)),
                "mean_margin_bias": float(np.mean(margins) - float(a_margin.mean())),
                "seconds": round(time.time() - t0, 1),
            }
            out.append(row)
            print(f"  share={share:.4f}  RL_HOME sesgo {row['rl_home_bias_pp']:+.2f}pp  "
                  f"VAL-1.3 gap {row['val13_gap_pp']:+.2f}pp  total {row['mean_total']:.3f} "
                  f"({row['mean_total_bias']:+.3f})  margen {row['mean_margin']:+.3f} "
                  f"({row['mean_margin_bias']:+.3f})  [{row['seconds']}s]", flush=True)
    finally:
        new_sim_mod.WALKOFF_9TH_SHARE = original

    return out


def _rel(path: Path) -> str:
    """Ruta relativa al repo cuando se pueda (p. ej. --out-dir fuera del árbol)."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(REPO / "data" / "predictions_history.db"))
    ap.add_argument("--seasons", default="2024,2025")
    ap.add_argument("--run-date", default="2026-07-25",
                    help="día de backtest_run_at de la corrida canónica ('' para no filtrar)")
    ap.add_argument("--n-sims", type=int, default=50_000,
                    help="simulaciones por juego y por variante (default: N_MC del backtest)")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: solo los primeros N juegos")
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--no-git-verify", action="store_true")
    ap.add_argument("--walkoff-sweep", default=None,
                    help="lista de valores de WALKOFF_9TH_SHARE a medir (in-process, no persiste). "
                         "Corre SOLO el barrido, no la evaluación completa.")
    args = ap.parse_args()

    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    print("Evaluador distribucional runline/total — CALIBRACIÓN, NO ROI")
    print("(no hay líneas históricas de estos mercados en la DB; ningún número de acá es rentabilidad)\n")

    old_mc, vendor_sha = load_old_simulator(verify_against_git=not args.no_git_verify)
    from modules.baseball_module.montecarlo.simulator import monte_carlo_advanced as new_mc
    from modules.baseball_module.montecarlo import simulator as new_sim_mod

    games = load_games(Path(args.db), seasons, args.run_date or None, args.limit)
    if not games:
        print("Sin juegos que evaluar con esos filtros.", file=sys.stderr)
        return 1
    if args.walkoff_sweep:
        shares = [float(s) for s in args.walkoff_sweep.split(",") if s.strip()]
        print(f"Barrido de WALKOFF_9TH_SHARE sobre {len(games)} juegos "
              f"({args.n_sims:,} sims/juego) — reporte, no cambio\n")
        sweep = walkoff_sweep(games, shares, args.n_sims, new_sim_mod)
        sweep_path = out_dir / f"walkoff_sweep_{ts}.json"
        sweep_path.write_text(json.dumps({
            "run_at": datetime.now(timezone.utc).isoformat(),
            "note": ("WALKOFF_9TH_SHARE es una constante asumida (1/9). Este barrido la mide, "
                     "no la cambia — el valor en el código sigue intacto."),
            "current_default": new_sim_mod.WALKOFF_9TH_SHARE,
            "config": {"seasons": seasons, "n_games": len(games), "n_sims": args.n_sims},
            "sweep": sweep,
        }, indent=2))
        print(f"\nBarrido: {_rel(sweep_path)}")
        return 0

    print(f"Juegos: {len(games)}  |  temporadas {seasons}  |  {args.n_sims:,} sims × 2 variantes\n")

    rows = simulate_all(games, args.n_sims, old_mc, new_mc)

    per_game = out_dir / f"per_game_{ts}.csv"
    with per_game.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    slices = {"overall": aggregate(rows, "overall")}
    for season in seasons:
        sub = [r for r in rows if r["season"] == season]
        slices[f"season_{season}"] = aggregate(sub, f"season_{season}")
    for month in sorted({r["date"][:7] for r in rows}):
        sub = [r for r in rows if r["date"].startswith(month)]
        slices[f"month_{month}"] = aggregate(sub, f"month_{month}")

    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "what_this_measures": (
            "Calibración distribucional de runline y total: qué tan bien la distribución "
            "simulada de carreras describe los resultados reales. NO mide ROI ni EV — no "
            "existen líneas históricas de runline/total en esta DB."
        ),
        "config": {
            "db": str(args.db),
            "seasons": seasons,
            "backtest_run_date_filter": args.run_date or None,
            "n_sims_per_variant": args.n_sims,
            "n_games": len(rows),
            "total_lines": list(TOTAL_LINES),
            "runline": RUNLINE,
            "seed_scheme": "game_pk % 2**32 (idéntico en ambas variantes — comparación pareada)",
        },
        "simulators": {
            "old": {"source": f"{OLD_SIM_COMMIT}^:{OLD_SIM_PATH}",
                    "vendored": _rel(VENDORED_OLD_SIM),
                    "sha256": vendor_sha},
            "new": {"source": _rel(Path(new_sim_mod.__file__)),
                    "walkoff_9th_share": getattr(new_sim_mod, "WALKOFF_9TH_SHARE", None),
                    "nb_dispersion": getattr(new_sim_mod, "NB_DISPERSION", None)},
        },
        "slices": slices,
    }
    metrics_path = out_dir / f"metrics_{ts}.json"
    metrics_path.write_text(json.dumps(payload, indent=2))

    ov = slices["overall"]
    print("\n" + "=" * 78)
    print(f"RESUMEN — {ov['n']} juegos")
    print("=" * 78)
    print(f"{'mercado':<22} {'real':>7} {'viejo':>8} {'nuevo':>8} {'sesgo_v':>9} {'sesgo_n':>9} "
          f"{'Brier_v':>9} {'Brier_n':>9}")
    for market, m in ov["markets"].items():
        print(f"{market:<22} {m['new']['observed_rate']:>7.4f} {m['old']['mean_pred']:>8.4f} "
              f"{m['new']['mean_pred']:>8.4f} {m['old']['bias_pp']:>8.2f}p {m['new']['bias_pp']:>8.2f}p "
              f"{m['old']['brier']:>9.5f} {m['new']['brier']:>9.5f}")

    v13 = ov["val13_margin_ge2_given_home_won"]
    print(f"\nVAL-1.3  P(margen local ≥2 | ganó el local):  real {v13['actual']:.4f}  |  "
          f"viejo {v13['old']:.4f} ({(v13['old']-v13['actual'])*100:+.2f}pp)  |  "
          f"nuevo {v13['new']:.4f} ({(v13['new']-v13['actual'])*100:+.2f}pp)")
    print(f"Total medio:  real {ov['actual']['mean_total']:.3f}  |  "
          f"viejo {ov['moments']['old']['mean_total']:.3f}  |  nuevo {ov['moments']['new']['mean_total']:.3f}")
    print(f"RPS total   :  viejo {ov['moments']['old']['rps_total']:.5f}  |  "
          f"nuevo {ov['moments']['new']['rps_total']:.5f}")
    print(f"RPS margen  :  viejo {ov['moments']['old']['rps_margin']:.5f}  |  "
          f"nuevo {ov['moments']['new']['rps_margin']:.5f}")
    print(f"\nPor juego: {_rel(per_game)}")
    print(f"Métricas  : {_rel(metrics_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
