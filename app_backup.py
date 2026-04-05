# SISTEMA DE APUESTAS INTELIGENTE - APP PRINCIPAL (FINAL BOSS QUANT G6)
# Autor: Braulio & GPT-5 (correcciones)
import os
import sqlite3
import math
import random
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ---------- Módulos propios (deben existir) ----------
from odds_fetcher import get_odds_data  # -> list[dict]
from score_predictor import ScorePredictor  # -> predict_match(...)
from value_detector import ValueDetector  # -> analyze(...), show_table(...)

# ---------- Config general ----------
st.set_page_config(page_title="Sistema de Apuestas (FINAL BOSS QUANT G6)", layout="wide")
PRIMARY = "#00C853"  # verde
DANGER = "#FF5252"  # rojo
WARN = "#FDD835"  # amarillo
NEUTRAL = "#90A4AE"  # gris

st.markdown(
    f"""
<h2 style="text-align:center; color:{PRIMARY}">
⚽ Sistema de Apuestas Inteligente — FINAL BOSS QUANT (Ultra G6)
</h2>
<p style="text-align:center; opacity:.8">
Poisson · EV · AutoPred · Ranking · SQLite AutoLog · Kelly · Monte Carlo · Backtesting ML · Bias · Portfolio · Optimización
</p>
""",
    unsafe_allow_html=True,
)
st.markdown("---")


# ---------- Barra de estado ----------
def barra_estado_fuente():
    fuente = "❌ Sin datos"
    color = DANGER
    if os.path.exists("odds_last.json"):
        fuente, color = "♻️ Cache Local", WARN
    if os.getenv("ODDS_API_KEY"):
        fuente, color = "🌐 API en Línea", PRIMARY
    st.markdown(
        f"<div style='text-align:center;background:{color};padding:6px;border-radius:6px;'><b>{fuente}</b></div>",
        unsafe_allow_html=True,
    )


barra_estado_fuente()

# ---------- Instancias core ----------
predictor = ScorePredictor()
detector = ValueDetector(threshold=0.05)  # EV ≥ 5% por defecto
DB_FILE = "predictions_history.sqlite"

# ---------- Utilidades DB / EV / Kelly ----------
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            league TEXT,
            home TEXT,
            away TEXT,
            p_home REAL,
            p_draw REAL,
            p_away REAL,
            odd1 REAL,
            draw REAL,
            odd2 REAL,
            pick TEXT,
            ev_best REAL,
            kelly_frac REAL,
            source TEXT
        )
        """
        )
        conn.commit()


def save_pred_row(row: dict):
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        cur.execute(
            """
        INSERT INTO predictions (ts, league, home, away, p_home, p_draw, p_away,
                                 odd1, draw, odd2, pick, ev_best, kelly_frac, source)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
            (
                row.get("ts"),
                row.get("league"),
                row.get("home"),
                row.get("away"),
                row.get("p_home"),
                row.get("p_draw"),
                row.get("p_away"),
                row.get("odd1"),
                row.get("draw"),
                row.get("odd2"),
                row.get("pick"),
                row.get("ev_best"),
                row.get("kelly_frac"),
                row.get("source"),
            ),
        )
        conn.commit()


def read_history(limit=1000) -> pd.DataFrame:
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
    with sqlite3.connect(DB_FILE) as conn:
        return pd.read_sql_query(f"SELECT * FROM predictions ORDER BY id DESC LIMIT {int(limit)}", conn)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["league", "home", "away", "odd1", "draw", "odd2"])
    df = pd.DataFrame(df).copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    mapping = {
        "league": pick("league", "liga", "sport key", "sport_key", "tournament"),
        "home": pick("home", "home_team", "equipo 1", "local", "team1"),
        "away": pick("away", "away_team", "equipo 2", "visitante", "team2"),
        "odd1": pick("odd1", "cuota 1", "cuota local", "home odd", "home_odd", "odd_home"),
        "draw": pick("draw", "cuota x", "empate", "odd_x", "draw_odd"),
        "odd2": pick("odd2", "cuota 2", "cuota visitante", "away odd", "away_odd", "odd_away"),
    }

    out = pd.DataFrame()
    for k, orig in mapping.items():
        if orig is None:
            out[k] = pd.Series([np.nan] * len(df))
        elif k in ["odd1", "draw", "odd2"]:
            out[k] = pd.to_numeric(df[orig], errors="coerce")
        else:
            out[k] = df.get(orig, pd.Series([None] * len(df)))

    # fill missing draw with a reasonable default if full column missing
    if "draw" not in out.columns or out["draw"].isna().all():
        out["draw"] = 3.2

    # require essential columns
    out = out.dropna(subset=["home", "away", "odd1", "odd2"]).reset_index(drop=True)
    return out


def calc_ev(odd: float, prob_pct: float) -> float:
    # odd: decimal odd (e.g. 2.5), prob_pct: probability in percent (0..100)
    if odd is None or prob_pct is None or np.isnan(odd) or np.isnan(prob_pct):
        return float("nan")
    p = max(min(prob_pct / 100.0, 1.0), 0.0)
    return (odd * p) - 1.0


def kelly_fraction(odd: float, prob_pct: float) -> float:
    # classic fractional Kelly: f* = (b*p - q)/b   where b = odd-1, p in [0,1], q = 1-p
    try:
        if odd is None or np.isnan(odd):
            return 0.0
        b = odd - 1.0
        if b <= 0:
            return 0.0
        p = max(min(prob_pct / 100.0, 1.0), 0.0)
        q = 1.0 - p
        f = (b * p - q) / b
        return float(max(0.0, f))
    except Exception:
        return 0.0


def probs_for_detector(res: dict) -> dict:
    return {
        "home_win_prob": res.get("prob_home", 0.0),
        "draw_prob": res.get("prob_draw", 0.0),
        "away_win_prob": res.get("prob_away", 0.0),
    }


# ---------- Sidebar ----------
init_db()
st.sidebar.header("⚙️ Controles Globales")
predictor.base_avg_home = st.sidebar.slider("λ base LOCAL (Poisson)", 0.5, 2.8, 1.45, 0.05)
predictor.base_avg_away = st.sidebar.slider("λ base VISITANTE (Poisson)", 0.5, 2.8, 1.25, 0.05)
detector.threshold = st.sidebar.slider("Umbral EV (✅ mínimo)", 0.00, 0.30, 0.05, 0.01)
bankroll = st.sidebar.number_input("💰 Bankroll (moneda)", min_value=0.0, value=1000.0, step=50.0)
auto_rank_show = st.sidebar.checkbox("Mostrar Ranking Top 10 EV", value=True)
mc_trials = int(st.sidebar.number_input("🔁 Monte Carlo: nº simulaciones por pick", min_value=100, value=5000, step=100))
st.sidebar.caption("Kelly usa el bankroll para sugerir stake en picks con EV > 0.")


# ---------- Panel de botones ----------
st.markdown("### ⚙️ Panel de Predicciones Inteligentes")
b1, b2, b3 = st.columns(3)
with b1:
    if st.button("🔄 Refrescar Datos"):
        st.session_state.pop("data", None)
        st.success("✅ Datos refrescados.")
with b2:
    if st.button("🧹 Limpiar Pantalla"):
        st.session_state.clear()
        st.info("Pantalla limpia.")
        st.stop()
with b3:
    if st.button("📡 Cargar Cuotas"):
        try:
            st.session_state["data"] = get_odds_data()
            st.success("📊 Cuotas cargadas.")
        except Exception as e:
            st.error(f"❌ Error al cargar cuotas: {e}")
st.markdown("---")


# ============================================================
# DATOS CARGADOS
# ============================================================
if "data" in st.session_state and st.session_state["data"]:
    raw_df = pd.DataFrame(st.session_state["data"])
    df = normalize_df(raw_df)
    st.markdown("### 📋 Eventos Cargados")
    st.dataframe(df, use_container_width=True)

    # --------------- Predicciones automáticas + autolog ---------------
    st.markdown("## 🤖 Predicciones Automáticas (Ultra + AutoLog)")
    auto_rows = []
    alerts = []
    for _, r in df.iterrows():
        try:
            home, away = str(r["home"]), str(r["away"])
            league = str(r.get("league", ""))
            o1, ox, o2 = float(r["odd1"]), float(r["draw"]), float(r["odd2"])

            # predictor.predict_match devuelve probabilidades en porcentaje (0..100)
            res = predictor.predict_match(home, away, 1.0, 1.0, 1.0, 1.0)

            ev_home = calc_ev(o1, res["prob_home"])
            ev_draw = calc_ev(ox, res["prob_draw"])
            ev_away = calc_ev(o2, res["prob_away"])
            evs = [ev_home, ev_draw, ev_away]

            best_idx = int(np.nanargmax(evs))
            pick = ["Local", "Empate", "Visitante"][best_idx] if not np.isnan(evs[best_idx]) else "N/A"
            best_ev = evs[best_idx] if not np.isnan(evs[best_idx]) else float("nan")
            k_frac = kelly_fraction([o1, ox, o2][best_idx], [res["prob_home"], res["prob_draw"], res["prob_away"]][best_idx])

            save_pred_row(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "league": league,
                    "home": home,
                    "away": away,
                    "p_home": res["prob_home"],
                    "p_draw": res["prob_draw"],
                    "p_away": res["prob_away"],
                    "odd1": o1,
                    "draw": ox,
                    "odd2": o2,
                    "pick": pick,
                    "ev_best": best_ev,
                    "kelly_frac": k_frac,
                    "source": "auto",
                }
            )

            auto_rows.append(
                {
                    "Partido": f"{home} vs {away}",
                    "Liga": league,
                    "Prob Local (%)": res["prob_home"],
                    "Prob Empate (%)": res["prob_draw"],
                    "Prob Visitante (%)": res["prob_away"],
                    "EV Local (%)": round(ev_home * 100, 2) if not np.isnan(ev_home) else None,
                    "EV Empate (%)": round(ev_draw * 100, 2) if not np.isnan(ev_draw) else None,
                    "EV Visitante (%)": round(ev_away * 100, 2) if not np.isnan(ev_away) else None,
                    "Pick Recomendado": pick,
                    "Mejor EV (%)": round(best_ev * 100, 2) if not np.isnan(best_ev) else None,
                    "Kelly (%)": round(k_frac * 100, 2),
                    "Stake Sugerido": round(k_frac * bankroll, 2),
                }
            )

            if not np.isnan(best_ev) and best_ev >= detector.threshold:
                alerts.append(f"🔥 {home} vs {away} → {pick} · EV {best_ev*100:.1f}% · Kelly {k_frac*100:.1f}%")
        except Exception as e:
            st.warning(f"Error en {r.get('home')} vs {r.get('away')}: {e}")

    if auto_rows:
        auto_df = pd.DataFrame(auto_rows).sort_values("Mejor EV (%)", ascending=False, na_position="last")
        st.dataframe(auto_df, use_container_width=True)
        if alerts:
            st.success(" · ".join(alerts))

        if auto_rank_show:
            st.markdown("### 🏆 Ranking: Top 10 Picks por EV")
            top10 = auto_df.head(10)[["Partido", "Pick Recomendado", "Mejor EV (%)", "Kelly (%)", "Stake Sugerido"]].fillna("N/A")
            if len(top10):
                chart = (
                    alt.Chart(top10)
                    .mark_bar()
                    .encode(
                        x=alt.X("Mejor EV (%):Q", title="EV (%)"),
                        y=alt.Y("Partido:N", sort="-x", title=""),
                        color=alt.condition(
                            alt.datum["Mejor EV (%)"] >= detector.threshold * 100,
                            alt.value(PRIMARY),
                            alt.value(DANGER),
                        ),
                        tooltip=["Partido", "Pick Recomendado", "Mejor EV (%)", "Kelly (%)", "Stake Sugerido"],
                    )
                    .properties(height=380)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No hay Top 10 disponible.")
    else:
        st.info("No se generaron predicciones automáticas (verifica datos).")

    # --------------- Predicción manual + autolog ---------------
    st.markdown("---")
    st.markdown("## 🎯 Predicción Manual de Partido")

    partidos = [f"{str(r['home'])} vs {str(r['away'])}" for _, r in df.iterrows()]
    if partidos:
        sel = st.selectbox("Selecciona un partido:", partidos, key="manual_select")

        c1, c2, c3, c4 = st.columns(4)
        att_h = c1.slider("Ataque Local", 0.5, 2.0, 1.0, 0.05)
        att_a = c2.slider("Ataque Visitante", 0.5, 2.0, 1.0, 0.05)
        def_h = c3.slider("Defensa Local", 0.5, 2.0, 1.0, 0.05)
        def_a = c4.slider("Defensa Visitante", 0.5, 2.0, 1.0, 0.05)

        if st.button("🎲 Calcular Predicción Manual"):
            try:
                home, away = sel.split(" vs ")
                row = df.loc[(df["home"] == home) & (df["away"] == away)].iloc[0]
                o1, ox, o2 = float(row["odd1"]), float(row["draw"]), float(row["odd2"])
                league = str(row.get("league", ""))

                res = predictor.predict_match(home, away, att_h, att_a, def_h, def_a)

                st.success(f"Resultados para {home} vs {away}")
                probs = pd.DataFrame(
                    {
                        "Resultado": ["Local", "Empate", "Visitante"],
                        "Probabilidad (%)": [res["prob_home"], res["prob_draw"], res["prob_away"]],
                    }
                )
                st.bar_chart(probs.set_index("Resultado"))

                ev_home = calc_ev(o1, res["prob_home"])
                ev_draw = calc_ev(ox, res["prob_draw"])
                ev_away = calc_ev(o2, res["prob_away"])

                det_input = probs_for_detector(res)
                odds = {"home_win": o1, "draw": ox, "away_win": o2}
                try:
                    ev_df = detector.analyze(f"{home} vs {away}", det_input, odds)
                    if hasattr(detector, "show_table"):
                        detector.show_table(ev_df)
                    else:
                        st.dataframe(ev_df, use_container_width=True)
                except Exception:
                    st.write(
                        "EV Local:",
                        round(ev_home * 100, 2),
                        "% | EV Empate:",
                        round(ev_draw * 100, 2),
                        "% | EV Visitante:",
                        round(ev_away * 100, 2),
                        "%",
                    )

                evs = [ev_home, ev_draw, ev_away]
                best_idx = int(np.nanargmax(evs))
                pick = ["Local", "Empate", "Visitante"][best_idx]
                best_ev = evs[best_idx]
                k_frac = kelly_fraction([o1, ox, o2][best_idx], [res["prob_home"], res["prob_draw"], res["prob_away"]][best_idx])

                st.markdown(
                    f"### 🏆 Pick recomendado: **{pick}** · EV **{best_ev*100:.2f}%** · Kelly **{k_frac*100:.2f}%** · Stake sugerido: **{k_frac*bankroll:.2f}**"
                )

                save_pred_row(
                    {
                        "ts": datetime.utcnow().isoformat(),
                        "league": league,
                        "home": home,
                        "away": away,
                        "p_home": res["prob_home"],
                        "p_draw": res["prob_draw"],
                        "p_away": res["prob_away"],
                        "odd1": o1,
                        "draw": ox,
                        "odd2": o2,
                        "pick": pick,
                        "ev_best": best_ev,
                        "kelly_frac": k_frac,
                        "source": "manual",
                    }
                )
            except Exception as e:
                st.error(f"Error al calcular manual: {e}")
    else:
        st.info("No hay partidos cargados para predicción manual.")

    # --------------- Monte Carlo (por pick) ---------------
    st.markdown("---")
    st.markdown("## 🧮 Simulador Monte Carlo (por pick recomendado)")
    st.caption("Simula el pick recomendado para estimar distribución de retorno, VaR y Sharpe.")

    def mc_simulate_pick(prob_pct, odd, trials, stake):
        p = max(min(prob_pct / 100.0, 1.0), 0.0)
        gains = []
        for _ in range(int(trials)):
            win = (np.random.rand() < p)
            gains.append((odd - 1.0) * stake if win else -stake)
        arr = np.array(gains, dtype=float)
        mean = arr.mean()
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        sharpe = (mean / std) if std > 1e-9 else (float("inf") if mean > 0 else 0.0)
        var_p95 = np.percentile(arr, 5)
        return {"mean": mean, "std": std, "sharpe": sharpe, "VaR(95%)": var_p95}

    if "auto_rows" in locals() and auto_rows:
        sample = pd.DataFrame(auto_rows).head(10)
        if not sample.empty:
            choice = st.selectbox("Pick a simular (Top 10 por EV)", list(sample["Partido"]))
            stake_sim = st.number_input("Stake simulado", min_value=1.0, value=10.0, step=1.0)
            if st.button("▶️ Ejecutar Monte Carlo"):
                home_sim, away_sim = choice.split(" vs ")
                base = df.loc[(df["home"] == home_sim) & (df["away"] == away_sim)].iloc[0]
                res = predictor.predict_match(home_sim, away_sim, 1.0, 1.0, 1.0, 1.0)
                pick_sim = sample[sample["Partido"] == choice]["Pick Recomendado"].iloc[0]
                if pick_sim == "Local":
                    prob, odd = res["prob_home"], float(base["odd1"])
                elif pick_sim == "Empate":
                    prob, odd = res["prob_draw"], float(base["draw"])
                else:
                    prob, odd = res["prob_away"], float(base["odd2"])
                out = mc_simulate_pick(prob, odd, mc_trials, stake_sim)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Media retorno", f"{out['mean']:.2f}")
                c2.metric("Volatilidad", f"{out['std']:.2f}")
                c3.metric("Sharpe (≈)", f"{out['sharpe']:.2f}")
                c4.metric("VaR 95%", f"{out['VaR(95%)']:.2f}")
        else:
            st.caption("No hay picks para simular.")
    else:
        st.caption("Carga cuotas para habilitar Monte Carlo.")

    # --------------- Backtesting / ML Calibrator ---------------
    st.markdown("---")
    st.markdown("## 🤖 Backtesting & ML Calibrator (historial)")
    st.caption("Sube CSV de resultados reales (home, away, result ∈ {Local,Empate,Visitante}) para calibrar.")

    hist = read_history(limit=2000)
    uploaded = st.file_uploader("CSV resultados reales", type=["csv"])
    results_df = None
    if uploaded is not None:
        try:
            results_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"CSV inválido: {e}")

    if not hist.empty and results_df is not None:
        h = hist.copy()
        h["match"] = h["home"].astype(str) + " vs " + h["away"].astype(str)
        results_df["match"] = results_df["home"].astype(str) + " vs " + results_df["away"].astype(str)
        merged = pd.merge(h, results_df[["match", "result"]], on="match", how="inner")
        st.write("Registros emparejados:", len(merged))

        if len(merged):
            def row_brier(r):
                probs = np.array([r["p_home"], r["p_draw"], r["p_away"]], dtype=float) / 100.0
                outcome = {"Local": 0, "Empate": 1, "Visitante": 2}.get(r["result"], 0)
                y = np.zeros(3)
                y[outcome] = 1.0
                return float(((probs - y) ** 2).sum())

            merged["brier"] = merged.apply(row_brier, axis=1)
            st.metric("Brier Score (↓ mejor)", f"{merged['brier'].mean():.3f}")

            scales = np.linspace(0.9, 1.1, 9)
            best_s = (1.0, 1.0, 1.0)
            best_brier = float("inf")
            for s_h in scales:
                for s_d in scales:
                    for s_a in scales:
                        probs = np.stack(
                            [
                                merged["p_home"].values / 100.0 * s_h,
                                merged["p_draw"].values / 100.0 * s_d,
                                merged["p_away"].values / 100.0 * s_a,
                            ],
                            axis=1,
                        )
                        probs = probs / probs.sum(axis=1, keepdims=True)
                        y = np.zeros((len(merged), 3))
                        idx = merged["result"].map({"Local": 0, "Empate": 1, "Visitante": 2}).values
                        y[np.arange(len(merged)), idx] = 1.0
                        brier = float(((probs - y) ** 2).sum(axis=1).mean())
                        if brier < best_brier:
                            best_brier = brier
                            best_s = (s_h, s_d, s_a)
            st.success(f"Escalas óptimas (Home, Draw, Away): {best_s} · Brier={best_brier:.3f}")
        else:
            st.caption("No se encontraron registros emparejados entre historial y CSV.")
    else:
        if hist.empty:
            st.caption("Aún no hay historial suficiente.")
        else:
            st.caption("Sube el CSV para calibrar.")

    # --------------- Market Bias Analyzer ---------------
    st.markdown("---")
    st.markdown("## 📊 Market Bias Analyzer")
    st.caption("Compara probas del modelo vs probas implícitas por cuotas para detectar sesgos.")
    bias_means = {"Local": 0.0, "Empate": 0.0, "Visitante": 0.0}
    if "data" in st.session_state and st.session_state["data"]:
        bias_rows = []
        for _, r in df.iterrows():
            try:
                home, away = str(r["home"]), str(r["away"])
                o1, ox, o2 = float(r["odd1"]), float(r["draw"]), float(r["odd2"])
                res = predictor.predict_match(home, away, 1.0, 1.0, 1.0, 1.0)
                imp = np.array([1.0 / o1, 1.0 / ox, 1.0 / o2], dtype=float)
                imp = imp / imp.sum()
                mod = np.array([res["prob_home"], res["prob_draw"], res["prob_away"]], dtype=float) / 100.0
                diff = (mod - imp) * 100.0  # puntos %
                bias_rows.append(
                    {
                        "Partido": f"{home} vs {away}",
                        "Δ Local": round(diff[0], 2),
                        "Δ Empate": round(diff[1], 2),
                        "Δ Visitante": round(diff[2], 2),
                    }
                )
            except Exception as e:
                # no romper todo por un partido mal formado
                continue
        bd = pd.DataFrame(bias_rows)
        st.dataframe(bd, use_container_width=True)
        if not bd.empty:
            bias_means = {
                "Local": bd["Δ Local"].mean(),
                "Empate": bd["Δ Empate"].mean(),
                "Visitante": bd["Δ Visitante"].mean(),
            }
            ch = (
                alt.Chart(pd.DataFrame({"Lado": list(bias_means.keys()), "Δ (%)": list(bias_means.values())}))
                .mark_bar()
                .encode(
                    x=alt.X("Δ (%):Q", title="Modelo - Mercado (puntos %)"),
                    y=alt.Y("Lado:N", sort="-x"),
                    color=alt.condition(alt.datum["Δ (%)"] > 0, alt.value(PRIMARY), alt.value(DANGER)),
                )
                .properties(height=200)
            )
            st.altair_chart(ch, use_container_width=True)

    # --------------- Auto-Calibrador simple ---------------
    st.markdown("---")
    st.markdown("## 🔁 Auto-Calibrador (simple)")
    st.caption("Ajuste sugerido de λ base según sesgo medio observado (heurística).")
    try:
        adj = float(bias_means["Local"]) / 100.0
    except Exception:
        adj = 0.0
    colA, colB = st.columns(2)
    new_lh = predictor.base_avg_home * (1.0 - 0.2 * adj)
    new_la = predictor.base_avg_away * (1.0 + 0.2 * adj)
    colA.metric("λ Local sugerido", f"{new_lh:.2f}", f"{(new_lh - predictor.base_avg_home):+.2f}")
    colB.metric("λ Visitante sugerido", f"{new_la:.2f}", f"{(new_la - predictor.base_avg_away):+.2f}")
    if st.button("Aplicar λ sugeridos"):
        predictor.base_avg_home, predictor.base_avg_away = new_lh, new_la
        st.success("Aplicado. Recalcula predicciones para ver efecto.")

    # --------------- Gestor de Portafolio ---------------
    st.markdown("---")
    st.markdown("## 🏦 Gestor de Portafolio (Kelly simplificado)")
    st.caption("Distribuye stake entre picks con EV>0 usando fracciones Kelly normalizadas.")
    if "auto_rows" in locals() and auto_rows:
        picks = pd.DataFrame(auto_rows)
        picks_pos = picks[picks["Mejor EV (%)"] > 0].copy()
        if not picks_pos.empty:
            kellys = picks_pos["Kelly (%)"].values / 100.0
            denom = kellys.sum() if kellys.sum() > 1e-9 else 1.0
            weights = kellys / denom
            picks_pos["Peso"] = weights
            picks_pos["Stake Portafolio"] = (weights * bankroll).round(2)
            st.dataframe(picks_pos[["Partido", "Pick Recomendado", "Mejor EV (%)", "Kelly (%)", "Stake Portafolio"]], use_container_width=True)
            st.success(f"Stake total asignado: {picks_pos['Stake Portafolio'].sum():.2f} / Bankroll {bankroll:.2f}")
        else:
            st.info("No hay picks con EV>0 para portafolio.")

    # --------------- Optimizador multivariable ---------------
    st.markdown("---")
    st.markdown("## 🧪 Optimizador Multivariable (Random + Bayes simple)")
    st.caption("Busca λ base y umbral EV que maximizan EV medio en los partidos cargados.")
    iters = int(st.number_input("Iteraciones de búsqueda", min_value=10, value=60, step=10))
    if st.button("⚡ Ejecutar Optimización"):
        space = []
        for _ in range(int(iters)):
            lh = float(np.clip(np.random.normal(predictor.base_avg_home, 0.2), 0.6, 2.6))
            la = float(np.clip(np.random.normal(predictor.base_avg_away, 0.2), 0.6, 2.6))
            th = float(np.clip(np.random.normal(detector.threshold, 0.03), 0.0, 0.30))
            space.append((lh, la, th))

        def evaluate(lh, la, thr):
            tot = []
            old_h, old_a = predictor.base_avg_home, predictor.base_avg_away
            for _, r in df.iterrows():
                try:
                    home, away = str(r["home"]), str(r["away"])
                    o1, ox, o2 = float(r["odd1"]), float(r["draw"]), float(r["odd2"])
                    predictor.base_avg_home, predictor.base_avg_away = lh, la
                    res = predictor.predict_match(home, away, 1.0, 1.0, 1.0, 1.0)
                    evs = [calc_ev(o1, res["prob_home"]), calc_ev(ox, res["prob_draw"]), calc_ev(o2, res["prob_away"])]
                    m = max(evs)
                    if not np.isnan(m) and m >= thr:
                        tot.append(m)
                except Exception:
                    continue
            predictor.base_avg_home, predictor.base_avg_away = old_h, old_a
            return float(np.mean(tot)) if tot else -1.0

        scored = []
        for (lh, la, thr) in space:
            sc = evaluate(lh, la, thr)
            scored.append((sc, lh, la, thr))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[0]
        st.success(f"Mejor configuración → λH={top[1]:.2f} · λA={top[2]:.2f} · Umbral EV={top[3]*100:.1f}% · Score={top[0]*100:.2f}%")
        if st.button("Aplicar mejor configuración"):
            predictor.base_avg_home, predictor.base_avg_away = top[1], top[2]
            detector.threshold = top[3]
            st.success("Aplicado. Recalcula predicciones para ver efecto.")

    # --------------- Estadísticas (SQLite) ---------------
    st.markdown("---")
    st.markdown("## 📈 Estadísticas y Gestión (SQLite)")
    hist2 = read_history(limit=2000)
    if hist2.empty:
        st.caption("Sin registros aún. Calcula predicciones para poblar la DB.")
    else:
        ev_mean = pd.to_numeric(hist2["ev_best"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        kelly = pd.to_numeric(hist2["kelly_frac"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        total = len(hist2)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicciones logueadas", f"{total}")
        c2.metric("EV medio", f"{(ev_mean.mean() * 100 if len(ev_mean) else 0):.2f}%")
        c3.metric("Kelly medio", f"{(kelly.mean() * 100 if len(kelly) else 0):.2f}%")
        bank_sim = bankroll + (((kelly.fillna(0) * bankroll) * (ev_mean.fillna(0))).sum() if len(kelly) and len(ev_mean) else 0)
        c4.metric("Banca simulada (EV)", f"{bank_sim:.2f}")
        st.markdown("### 🔎 Últimos 200 registros")
        st.dataframe(hist2.head(200), use_container_width=True)

else:
    st.info("Presiona **📡 Cargar Cuotas** para iniciar el análisis.")

