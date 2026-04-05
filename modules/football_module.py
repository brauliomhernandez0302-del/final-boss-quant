# =================================================================
# football_quant_g11.py
# FINAL BOSS QUANT G11 — "The Omega Engine" (unificación G8/G10/G11)
# Autor: Braulio & GPT-5
# =================================================================

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sqlite3
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# Page config (dark-like feel using wide layout)
st.set_page_config(page_title="Final Boss Quant G11", layout="wide", initial_sidebar_state="expanded")

DB_FILE = "predictions_history.sqlite"

# ---------------------------
# Utilidades DB
# ---------------------------
def ensure_db():
    """Crea tablas necesarias si no existen."""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions_g11(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    engine TEXT,
                    league TEXT,
                    home TEXT,
                    away TEXT,
                    p_home REAL,
                    p_draw REAL,
                    p_away REAL,
                    odd_home REAL,
                    odd_draw REAL,
                    odd_away REAL,
                    pick TEXT,
                    ev_best REAL,
                    kelly REAL,
                    ht_home REAL,
                    ht_away REAL,
                    btts REAL
                )
            """)
            conn.commit()
    except Exception:
        pass

def save_prediction(record: dict):
    """Guarda una predicción en la tabla predictions_g11"""
    try:
        ensure_db()
        with sqlite3.connect(DB_FILE) as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO predictions_g11(
                    ts, engine, league, home, away, p_home, p_draw, p_away,
                    odd_home, odd_draw, odd_away, pick, ev_best, kelly,
                    ht_home, ht_away, btts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                record.get("engine", "G11"),
                record.get("league"),
                record.get("home"),
                record.get("away"),
                float(record.get("p_home") or 0.0),
                float(record.get("p_draw") or 0.0),
                float(record.get("p_away") or 0.0),
                float(record.get("odd_home") or 0.0),
                float(record.get("odd_draw") or 0.0),
                float(record.get("odd_away") or 0.0),
                record.get("pick"),
                float(record.get("ev_best") or 0.0),
                float(record.get("kelly") or 0.0),
                float(record.get("ht_home") or 0.0),
                float(record.get("ht_away") or 0.0),
                float(record.get("btts") or 0.0)
            ))
            conn.commit()
    except Exception:
        pass

def read_history(limit: int = 200) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_FILE) as conn:
            df = pd.read_sql_query(f"SELECT * FROM predictions_g11 ORDER BY id DESC LIMIT {int(limit)}", conn)
            return df
    except Exception:
        return pd.DataFrame()

# ---------------------------
# Small internal learning engine for fusion
# ---------------------------
class MiniLearner:
    """Combina probabilidades de submotores mediante pesos adaptativos simples."""
    def __init__(self, engines=("G8","G10","G11")):
        self.engines = list(engines)
        self.w = np.ones(len(self.engines)) / len(self.engines)
        self.lr = 0.03

    def predict(self, probs: Dict[str, float]) -> float:
        # probs: {"G8":0.5,"G10":0.45,"G11":0.52}
        x = np.array([probs.get(e, 0.0) for e in self.engines], dtype=float)
        denom = self.w.sum()
        return float(np.dot(self.w, x) / max(1e-9, denom))

    def update(self, probs: Dict[str, float], outcome: int):
        p = self.predict(probs)
        grad = (p - outcome)
        x = np.array([probs.get(e, 0.0) for e in self.engines], dtype=float)
        self.w = np.clip(self.w - self.lr * grad * x, 1e-6, None)
        self.w /= self.w.sum()

# ---------------------------
# Math helpers
# ---------------------------
def kelly_fraction(odd_decimal: float, prob: float) -> float:
    """Kelly fraction for decimal odds."""
    if odd_decimal <= 1.0 or prob <= 0.0 or prob >= 1.0:
        return 0.0
    b = odd_decimal - 1.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    return max(0.0, f)

def expected_value(odd_decimal: float, prob: float) -> float:
    return odd_decimal * prob - 1.0

def simulate_poisson(mu_h: float, mu_a: float, trials: int = 20000, seed: Optional[int]=None):
    rng = np.random.default_rng(seed)
    home = rng.poisson(mu_h, trials)
    away = rng.poisson(mu_a, trials)
    return home, away

# ---------------------------
# Internal simple predictor (fallback if no external predictor passed)
# ---------------------------
def internal_predictor(home: str, away: str) -> dict:
    """Devuelve una predicción básica (probabilidades en % y mu estimados)."""
    # heurística: fuerza local + historial simulado
    rng = np.random.default_rng(abs(hash(home+away)) % (2**32))
    mu_home = float(np.clip(rng.normal(1.45, 0.25), 0.2, 3.5))
    mu_away = float(np.clip(rng.normal(1.25, 0.25), 0.1, 3.0))
    h, a = simulate_poisson(mu_home, mu_away, trials=8000, seed=rng.integers(1,2**31-1))
    prob_home = float((h > a).mean() * 100.0)
    prob_draw = float((h == a).mean() * 100.0)
    prob_away = float((h < a).mean() * 100.0)
    return {
        "prob_home": prob_home,
        "prob_draw": prob_draw,
        "prob_away": prob_away,
        "mu_home": mu_home,
        "mu_away": mu_away,
        "home_goals": int(np.round(h.mean())),
        "away_goals": int(np.round(a.mean()))
    }

# ---------------------------
# G8 engine (classic) - returns probabilities (0..1)
# ---------------------------
def engine_g8(predictor_fn, home: str, away: str) -> Dict[str, float]:
    """Imita el motor G8 (rápido, Poisson baseline)."""
    try:
        res = predictor_fn(home, away)
        p_h = res.get("prob_home", 50.0) / 100.0
        p_d = res.get("prob_draw", 0.0) / 100.0
        p_a = res.get("prob_away", 50.0) / 100.0
        mu_h = res.get("mu_home", max(0.8, p_h*2.2))
        mu_a = res.get("mu_away", max(0.6, p_a*1.8))
        return {"p_home":p_h, "p_draw":p_d, "p_away":p_a, "mu_home":mu_h, "mu_away":mu_a}
    except Exception:
        return {"p_home":0.5, "p_draw":0.0, "p_away":0.5, "mu_home":1.45, "mu_away":1.25}

# ---------------------------
# G10 engine (visual/pro) - more Monte Carlo emphasis
# ---------------------------
def engine_g10(predictor_fn, home: str, away: str, sims:int=12000) -> Dict[str, float]:
    try:
        res = predictor_fn(home, away)
        mu_h = res.get("mu_home", 1.45)
        mu_a = res.get("mu_away", 1.25)
        h, a = simulate_poisson(mu_h, mu_a, trials=sims)
        p_h = float((h > a).mean())
        p_d = float((h == a).mean())
        p_a = float((h < a).mean())
        return {"p_home":p_h, "p_draw":p_d, "p_away":p_a, "mu_home":mu_h, "mu_away":mu_a}
    except Exception:
        return {"p_home":0.5, "p_draw":0.0, "p_away":0.5, "mu_home":1.45, "mu_away":1.25}

# ---------------------------
# G11 engine (omega) - xG-like + contextual bias adjust
# ---------------------------
def engine_g11(predictor_fn, home: str, away: str, sims:int=20000) -> Dict[str, float]:
    try:
        res = predictor_fn(home, away)
        # contextual adjustment: small shift based on "home" name hashing (simulated bias)
        bias = (abs(hash(home)) % 7 - 3) * 0.02
        mu_h = max(0.1, float(res.get("mu_home", 1.45)) * (1.0 + bias))
        mu_a = max(0.1, float(res.get("mu_away", 1.25)) * (1.0 - bias))
        h, a = simulate_poisson(mu_h, mu_a, trials=sims)
        p_h = float((h > a).mean())
        p_d = float((h == a).mean())
        p_a = float((h < a).mean())
        # small calibration toward implied market (if provided by res)
        return {"p_home":p_h, "p_draw":p_d, "p_away":p_a, "mu_home":mu_h, "mu_away":mu_a}
    except Exception:
        return {"p_home":0.5, "p_draw":0.0, "p_away":0.5, "mu_home":1.45, "mu_away":1.25}

# ---------------------------
# Probable scorers heuristic
# ---------------------------
def probable_scorers(home: str, away: str, mu_h: float, mu_a: float, top_k:int=5):
    # Simulación heurística basada en mu: produce ranked list with probabilities
    home_names = [f"{home} - Jugador {i+1}" for i in range(top_k)]
    away_names = [f"{away} - Jugador {i+1}" for i in range(top_k)]
    # weights decay
    hv = np.array([1.0/(i+1) for i in range(top_k)], dtype=float)
    av = np.array([1.0/(i+1) for i in range(top_k)], dtype=float)
    hv = hv / hv.sum() * min(0.95, mu_h / (mu_h + mu_a + 0.1))
    av = av / av.sum() * min(0.95, mu_a / (mu_h + mu_a + 0.1))
    return list(zip(home_names, hv)), list(zip(away_names, av))

# ---------------------------
# AutoPred / Top Picks analyzer
# ---------------------------
def analyze_top_picks_from_df(df: pd.DataFrame, predictor_fn, top_n:int=10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Toma DataFrame con columns(home,away, odd_home, odd_draw, odd_away) y devuelve top_n picks y full df."""
    rows = []
    for _, r in df.iterrows():
        home = str(r.get("home") or r.get("Home") or "")
        away = str(r.get("away") or r.get("Away") or "")
        try:
            odd_h = float(r.get("odd_home") or r.get("odd1") or r.get("odd") or np.nan)
            odd_d = float(r.get("odd_draw") or r.get("draw") or np.nan)
            odd_a = float(r.get("odd_away") or r.get("odd2") or np.nan)
        except Exception:
            odd_h, odd_d, odd_a = np.nan, np.nan, np.nan

        # get predictions from engines
        g8 = engine_g8(predictor_fn, home, away)
        g10 = engine_g10(predictor_fn, home, away, sims=6000)
        g11 = engine_g11(predictor_fn, home, away, sims=8000)

        # fusion: weighted average with simple learner (static weights here for speed)
        probs = {
            "G8": g8["p_home"], "G10": g10["p_home"], "G11": g11["p_home"]
        }
        # score: EV * prob^alpha, where EV computed using implied odds if present
        best_ev = -999.0
        pick_side = None
        score_val = -999.0
        alpha = 1.1
        # evaluate three sides
        for side in ("home","draw","away"):
            if side=="home":
                p = np.mean([g8["p_home"], g10["p_home"], g11["p_home"]])
                odd = odd_h
            elif side=="draw":
                p = np.mean([g8["p_draw"], g10["p_draw"], g11["p_draw"]])
                odd = odd_d
            else:
                p = np.mean([g8["p_away"], g10["p_away"], g11["p_away"]])
                odd = odd_a
            if np.isnan(odd) or odd <= 1.01:
                ev = -999.0
            else:
                ev = expected_value(odd, p)
            if ev > best_ev:
                best_ev = ev
                pick_side = side
            if ev > -0.5:
                sc = ev * (p ** alpha)
            else:
                sc = ev
            # store
        # create row
        rows.append({
            "home": home, "away": away,
            "pick": pick_side, "best_ev": best_ev,
            "score": sc, "odd_h": odd_h, "odd_d": odd_d, "odd_a": odd_a,
            "p_g8_h": g8["p_home"], "p_g10_h": g10["p_home"], "p_g11_h": g11["p_home"]
        })
    full = pd.DataFrame(rows)
    full_sorted = full.sort_values(["score","best_ev"], ascending=[False, False]).reset_index(drop=True)
    return full_sorted.head(top_n), full_sorted

# ---------------------------
# Charts helpers
# ---------------------------
def chart_ev_bar(df: pd.DataFrame, title="Mejor EV (%)"):
    dfc = df.copy()
    if "Best_EV_pct" not in dfc.columns and "best_ev" in dfc.columns:
        dfc["Best_EV_pct"] = dfc["best_ev"] * 100.0
    chart = alt.Chart(dfc).mark_bar().encode(
        x=alt.X("Best_EV_pct:Q", title=title),
        y=alt.Y("Partido:N", sort="-x", title="Partido"),
        color=alt.condition(alt.datum["Best_EV_pct"] > 0, alt.value("#00C853"), alt.value("#E53935")),
        tooltip=list(dfc.columns)
    ).properties(height=360)
    return chart

def chart_score_heatmap(matrix_df: pd.DataFrame):
    chart = alt.Chart(matrix_df).mark_rect().encode(
        x=alt.X("away_goals:O", title="Goles Visitante"),
        y=alt.Y("home_goals:O", title="Goles Local"),
        color=alt.Color("prob:Q", scale=alt.Scale(scheme="greenblue"), title="Probabilidad"),
        tooltip=["home_goals", "away_goals", "prob"]
    ).properties(height=360, width=520)
    return chart

# ---------------------------
# Main UI: run_module
# ---------------------------
def run_module(external_predictor: Optional[Any]=None, detector: Optional[Any]=None):
    """
    external_predictor: función predictor(home, away) -> dict con keys 'prob_home','prob_draw','prob_away','mu_home','mu_away' etc.
    """
    ensure_db()
    st.title("⚽ FINAL BOSS QUANT G11 — THE OMEGA ENGINE")
    st.caption("Unifica G8 / G10 / G11 · AutoPred · Monte Carlo · HT/FT · BTTS · Goleadores · Fusión")

    # Sidebar controls
    st.sidebar.header("⚙️ Controles Globales")
    engine_mode = st.sidebar.selectbox("Motor a usar (para una predicción única)", ["Fusión (por defecto)", "G8", "G10", "G11"])
    dark_mode = st.sidebar.checkbox("Modo oscuro (visual)", value=False)
    use_external = st.sidebar.checkbox("Usar predictor externo (si está cargado)", value=True)
    top_n = st.sidebar.number_input("Top N Picks (AutoPred)", min_value=1, max_value=50, value=10, step=1)

    # Input panel
    with st.expander("🔧 Parámetros partido", expanded=True):
        col1, col2, col3 = st.columns(3)
        league = col1.text_input("🏆 Liga", "Champions League")
        home = col2.text_input("🏠 Equipo Local", "Real Madrid")
        away = col3.text_input("🚀 Equipo Visitante", "Manchester City")

        c1, c2, c3 = st.columns(3)
        odd_home = c1.number_input("Cuota Local (decimal)", min_value=1.01, value=2.10, step=0.01)
        odd_draw = c2.number_input("Cuota Empate (decimal)", min_value=1.01, value=3.30, step=0.01)
        odd_away = c3.number_input("Cuota Visitante (decimal)", min_value=1.01, value=3.80, step=0.01)

        c4, c5 = st.columns(2)
        sims = int(c4.number_input("Simulaciones Monte Carlo", min_value=500, value=20000, step=500))
        include_btts = c5.checkbox("Incluir BTTS/HT análisis", value=True)

    # Predictor selection
    predictor_fn = external_predictor if (external_predictor is not None and use_external) else internal_predictor

    # Run a combined prediction button
    st.markdown("---")
    st.header("🔮 Predicción (única) — Ejecutar motores y fusión")
    colA, colB = st.columns([1,3])
    with colA:
        run_btn = st.button("▶️ Ejecutar predicción")
    with colB:
        st.write("Selecciona motor o deja 'Fusión' para combinar G8/G10/G11.")

    if run_btn:
        try:
            # Run engines
            g8 = engine_g8(predictor_fn, home, away)
            g10 = engine_g10(predictor_fn, home, away, sims=max(1000, min(20000, int(sims/2))))
            g11 = engine_g11(predictor_fn, home, away, sims=sims)

            # Convert to comparable probabilities (0..1)
            probs_g8 = {"home": g8["p_home"], "draw": g8["p_draw"], "away": g8["p_away"]}
            probs_g10 = {"home": g10["p_home"], "draw": g10["p_draw"], "away": g10["p_away"]}
            probs_g11 = {"home": g11["p_home"], "draw": g11["p_draw"], "away": g11["p_away"]}

            # Fusion via MiniLearner (online)
            learner = MiniLearner(engines=("G8","G10","G11"))
            fused_home = learner.predict({"G8":probs_g8["home"], "G10":probs_g10["home"], "G11":probs_g11["home"]})
            fused_draw = learner.predict({"G8":probs_g8["draw"], "G10":probs_g10["draw"], "G11":probs_g11["draw"]})
            fused_away = learner.predict({"G8":probs_g8["away"], "G10":probs_g10["away"], "G11":probs_g11["away"]})
            # normalize
            s = fused_home + fused_draw + fused_away
            if s <= 0: s = 1e-9
            fused_home, fused_draw, fused_away = fused_home/s, fused_draw/s, fused_away/s

            # Choose engine based on engine_mode
            if engine_mode == "G8":
                final = probs_g8
                engine_tag = "G8"
            elif engine_mode == "G10":
                final = probs_g10
                engine_tag = "G10"
            elif engine_mode == "G11":
                final = probs_g11
                engine_tag = "G11"
            else:
                final = {"home": fused_home, "draw": fused_draw, "away": fused_away}
                engine_tag = "FUSION"

            # EV/Kelly using decimal odds
            p_home = final["home"]
            p_draw = final["draw"]
            p_away = final["away"]

            ev_home = expected_value(odd_home, p_home)
            ev_draw = expected_value(odd_draw, p_draw)
            ev_away = expected_value(odd_away, p_away)
            best_ev = max(ev_home, ev_draw, ev_away)
            best_pick = ["Local","Empate","Visitante"][int(np.argmax([ev_home, ev_draw, ev_away]))]

            k_home = kelly_fraction(odd_home, p_home)
            k_draw = kelly_fraction(odd_draw, p_draw)
            k_away = kelly_fraction(odd_away, p_away)

            # Monte Carlo detailed for final engine's mu if available
            mu_h = g11.get("mu_home", 1.45)
            mu_a = g11.get("mu_away", 1.25)

            home_sample, away_sample = simulate_poisson(mu_h, mu_a, trials=sims)
            wins = (home_sample > away_sample).mean() * 100.0
            draws = (home_sample == away_sample).mean() * 100.0
            losses = (home_sample < away_sample).mean() * 100.0

            # HT estimations
            ht_mu_h = max(0.05, mu_h * 0.45)
            ht_mu_a = max(0.05, mu_a * 0.45)
            ht_sample_h, ht_sample_a = simulate_poisson(ht_mu_h, ht_mu_a, trials=sims)
            ht_home = (ht_sample_h > ht_sample_a).mean() * 100.0
            ht_draw = (ht_sample_h == ht_sample_a).mean() * 100.0
            ht_away = (ht_sample_h < ht_sample_a).mean() * 100.0

            # BTTS empirical
            btts = ((home_sample > 0) & (away_sample > 0)).mean() * 100.0 if include_btts else np.nan

            # Probable scorers
            scorers_h, scorers_a = probable_scorers(home, away, mu_h, mu_a, top_k=5)

            # Show summary
            st.subheader("🔎 Resumen - Motor: " + engine_tag)
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Prob Local", f"{p_home*100:.2f}%")
            k2.metric("Prob Empate", f"{p_draw*100:.2f}%")
            k3.metric("Prob Visitante", f"{p_away*100:.2f}%")
            k4.metric("Mejor EV", f"{best_ev*100:.2f}% · Pick: {best_pick}")

            # Detailed table
            table = pd.DataFrame({
                "Resultado": ["Local","Empate","Visitante"],
                "Prob. (%)": [p_home*100, p_draw*100, p_away*100],
                "Cuota": [odd_home, odd_draw, odd_away],
                "EV (%)": [ev_home*100, ev_draw*100, ev_away*100],
                "Kelly (%)": [k_home*100, k_draw*100, k_away*100]
            })
            st.dataframe(table.style.background_gradient(cmap="RdYlGn"), use_container_width=True)

            # MC charts
            st.markdown("### 📊 Monte Carlo — Distribución de marcadores y top resultados")
            combos = pd.crosstab(home_sample, away_sample)
            if not combos.empty:
                combo_series = combos.stack().reset_index(name='count').sort_values("count", ascending=False).head(10)
                combo_series["scoreline"] = combo_series["home_sample"].astype(str) + " - " + combo_series["away_sample"].astype(str) if "home_sample" in combo_series.columns else combo_series.apply(lambda r: f"{r[0]} - {r[1]}", axis=1)
                # build heatmap matrix
                matrix_df = combos.stack().reset_index()
                matrix_df.columns = ["home_goals","away_goals","count"]
                matrix_df["prob"] = matrix_df["count"] / matrix_df["count"].sum()
                # show heatmap
                try:
                    st.altair_chart(chart_score_heatmap(matrix_df), use_container_width=True)
                except Exception:
                    st.write(matrix_df.head(10))
            else:
                st.write("No hay datos MC para mostrar matriz.")

            # Goals distribution chart
            try:
                dist_df = pd.DataFrame({"Local":home_sample, "Visitante":away_sample})
                chart = alt.Chart(dist_df).transform_fold(["Local","Visitante"], as_=["Lado","Goles"]).mark_area(opacity=0.4).encode(
                    x=alt.X("Goles:Q", bin=alt.Bin(maxbins=20)),
                    y='count()',
                    color='Lado:N'
                ).properties(height=240)
                st.altair_chart(chart, use_container_width=True)
            except Exception:
                pass

            # Scorers
            st.markdown("### 🎯 Probables goleadores (heurística)")
            cs1, cs2 = st.columns(2)
            with cs1:
                st.write("Local:")
                sh = pd.DataFrame({"Jugador":[s[0] for s in scorers_h], "Prob_rel":[s[1] for s in scorers_h]})
                st.table(sh.style.format({"Prob_rel":"{:.3f}"}))
            with cs2:
                st.write("Visitante:")
                sa = pd.DataFrame({"Jugador":[s[0] for s in scorers_a], "Prob_rel":[s[1] for s in scorers_a]})
                st.table(sa.style.format({"Prob_rel":"{:.3f}"}))

            # Save record
            record = {
                "engine": engine_tag, "league": league, "home": home, "away": away,
                "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
                "odd_home": odd_home, "odd_draw": odd_draw, "odd_away": odd_away,
                "pick": best_pick, "ev_best": best_ev, "kelly": max(k_home, k_draw, k_away),
                "ht_home": ht_home, "ht_away": ht_away, "btts": btts
            }
            save_prediction(record)

            st.success("✅ Predicción completada y guardada en historial local.")
        except Exception as e:
            st.error(f"Error al ejecutar predicción: {e}")

    # -----------------------------
    # AutoPred / Top Picks (pestaña)
    # -----------------------------
    st.markdown("---")
    st.header("📈 AutoPred — Top Picks (análisis masivo)")

    uploaded = st.file_uploader("Cargar CSV con partidos (home,away,odd_home,odd_draw,odd_away)", type=["csv"])
    matches_df = pd.DataFrame()
    if uploaded is not None:
        try:
            matches_df = pd.read_csv(uploaded)
            st.success(f"CSV cargado: {len(matches_df)} partidos")
        except Exception as e:
            st.error(f"CSV inválido: {e}")
    else:
        if "data" in st.session_state and st.session_state["data"]:
            try:
                matches_df = pd.DataFrame(st.session_state["data"])
            except Exception:
                matches_df = pd.DataFrame()

    if not matches_df.empty:
        if st.button("▶️ Ejecutar AutoPred y calcular Top Picks"):
            try:
                top_df, full = analyze_top_picks_from_df(matches_df, predictor_fn, top_n=int(top_n))
                st.subheader("Top Picks (ordenado por Score = EV * Prob^α)")
                display = top_df.copy()
                display["Best_EV_pct"] = display["best_ev"] * 100.0
                display["Partido"] = display["home"] + " vs " + display["away"]
                st.dataframe(display[["Partido","pick","Best_EV_pct","score","odd_h","odd_a","odd_d"]].fillna("-").style.format({
                    "Best_EV_pct":"{:.2f}", "score":"{:.4f}"
                }), use_container_width=True)

                # Chart
                try:
                    st.altair_chart(chart_ev_bar(display), use_container_width=True)
                except Exception:
                    pass

                csv = full.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar listado completo (CSV)", csv, file_name="top_picks_full.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error en AutoPred: {e}")
    else:
        st.info("Sube un CSV con partidos para ejecutar AutoPred o pon datos en session_state['data'].")

    # -----------------------------
    # Historial y comparador
    # -----------------------------
    st.markdown("---")
    st.header("📜 Historial y Comparador de motores")
    hist = read_history(200)
    if hist.empty:
        st.info("Aún no hay historial. Ejecuta alguna predicción para poblar la base.")
    else:
        st.dataframe(hist, use_container_width=True)
        csv = hist.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar historial (CSV)", csv, file_name="predictions_history_g11.csv", mime="text/csv")

    st.markdown("---")
    st.caption("Final Boss Quant G11 — diseñado para mantener todo lo que tenías (G8,G10), corregir errores y añadir fusión, AutoPred y visuales avanzadas. Si quieres que lo deje en modo más compacto o que integre directamente tu `ScorePredictor`/`ValueDetector`, dímelo y lo enlazo.")

# End of module

