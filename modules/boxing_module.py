# ============================================================
# MÓDULO BOXEO - FINAL BOSS QUANT G7+ (sin cambiar nada visual)
# Autor: Braulio & GPT-5
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import altair as alt
from datetime import datetime

def _kelly(odd, p):
    b = max(odd - 1, 1e-9)
    q = 1 - p
    return max(0.0, (b * p - q) / b)

def save_boxing_prediction(row: dict):
    DB_FILE = "predictions_history.sqlite"
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS boxing_predictions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                event TEXT,
                boxer_a TEXT,
                boxer_b TEXT,
                rounds INTEGER,
                p_a_model REAL,
                p_b_model REAL,
                odd_a REAL,
                odd_b REAL,
                pick TEXT,
                ev_best REAL,
                kelly_frac REAL
            )
            """)
            cur.execute("""
            INSERT INTO boxing_predictions
            (ts, event, boxer_a, boxer_b, rounds, p_a_model, p_b_model, odd_a, odd_b, pick, ev_best, kelly_frac)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                row.get("event", ""),
                row.get("boxer_a"),
                row.get("boxer_b"),
                row.get("rounds"),
                row.get("p_a_model"),
                row.get("p_b_model"),
                row.get("odd_a"),
                row.get("odd_b"),
                row.get("pick"),
                row.get("ev_best"),
                row.get("kelly_frac"),
            ))
            conn.commit()
    except Exception:
        # no romper la UI por fallo de guardado
        pass

def run_module(predictor, detector):
    st.markdown("## 🥊 Boxeo — Predicción 1v1 (ML · EV · Kelly · Monte Carlo)")
    st.caption("Moneyline + simulación por asaltos (KO/TKO/DEC) con motor G6+")

    c1, c2, c3 = st.columns(3)
    boxer_a = c1.text_input("Boxeador A", "Boxer A")
    boxer_b = c2.text_input("Boxeador B", "Boxer B")
    rounds = c3.selectbox("Asaltos programados", [8, 10, 12], index=2)

    c4, c5 = st.columns(2)
    odd_a = c4.number_input("Cuota A", min_value=1.01, value=1.75, step=0.01)
    odd_b = c5.number_input("Cuota B", min_value=1.01, value=2.15, step=0.01)

    st.markdown("### ⚙️ Supuestos de finalización por asalto (ajustables)")
    c6, c7, c8 = st.columns(3)
    ko_rate_a = c6.slider("p(KO/TKO) A por asalto (≈)", 0.00, 0.30, 0.06, 0.01)
    ko_rate_b = c7.slider("p(KO/TKO) B por asalto (≈)", 0.00, 0.30, 0.05, 0.01)
    sim_n = int(c8.number_input("Simulaciones Monte Carlo", min_value=1000, value=30000, step=1000))

    if st.button("🥊 Calcular Predicción Boxeo"):
        try:
            # Prob. base ML desde tu predictor 1v1
            base = predictor.predict_match(boxer_a, boxer_b, 1.0, 1.0, 1.0, 1.0)
            pA = max(min(base.get("prob_home", 50) / 100.0, 0.99), 0.01)
            pB = max(min(base.get("prob_away", 50) / 100.0, 0.99), 0.01)
            s = pA + pB
            if s <= 0:
                pA, pB = 0.5, 0.5
            else:
                pA, pB = pA / s, pB / s

            ev_a = odd_a * pA - 1
            ev_b = odd_b * pB - 1
            k_a = _kelly(odd_a, pA)
            k_b = _kelly(odd_b, pB)

            st.success(f"**{boxer_a} vs {boxer_b}** · Asaltos: {rounds}")
            df = pd.DataFrame({
                "Lado": [boxer_a, boxer_b],
                "Prob. ML (%)": [pA * 100, pB * 100],
                "Cuota": [odd_a, odd_b],
                "EV (%)": [ev_a * 100, ev_b * 100],
                "Kelly (%)": [k_a * 100, k_b * 100]
            })
            st.dataframe(df, use_container_width=True)

            pick = boxer_a if ev_a > ev_b else boxer_b
            best_ev = max(ev_a, ev_b)
            best_k = k_a if pick == boxer_a else k_b
            st.markdown(f"### 🏆 Pick recomendado: **{pick}** · EV **{best_ev*100:.2f}%** · Kelly **{best_k*100:.2f}%**")

            # --- Sesgo del mercado (si detector provee algo) ---
            try:
                if hasattr(detector, "implied_prob"):
                    market_a = detector.implied_prob(odd_a)
                    market_b = detector.implied_prob(odd_b)
                else:
                    market_a = 100.0 / odd_a
                    market_b = 100.0 / odd_b
                st.markdown("### ⚖️ Sesgo del mercado (Modelo vs Cuotas)")
                st.write(pd.DataFrame({
                    "Boxeador": [boxer_a, boxer_b],
                    "Prob. Modelo (%)": [pA * 100, pB * 100],
                    "Prob. Mercado (%)": [market_a, market_b],
                    "Diferencia (%)": [pA * 100 - market_a, pB * 100 - market_b]
                }))
            except Exception:
                pass

            # Guardado automático
            save_boxing_prediction({
                "event": "",
                "boxer_a": boxer_a,
                "boxer_b": boxer_b,
                "rounds": rounds,
                "p_a_model": pA,
                "p_b_model": pB,
                "odd_a": odd_a,
                "odd_b": odd_b,
                "pick": pick,
                "ev_best": best_ev,
                "kelly_frac": best_k
            })

            # -------- Monte Carlo por asaltos ----------
            rng = np.random.default_rng(7)
            koA = rng.binomial(1, ko_rate_a, size=(sim_n, rounds)).any(axis=1)
            koB = rng.binomial(1, ko_rate_b, size=(sim_n, rounds)).any(axis=1)

            # Vectorizado: outcomes
            outcome_codes = np.full(sim_n, "DRAW", dtype=object)
            mask_A_KO = (koA == 1) & (koB == 0)
            mask_B_KO = (koB == 1) & (koA == 0)
            mask_both = (koA == 1) & (koB == 1)
            mask_none = (koA == 0) & (koB == 0)

            outcome_codes[mask_A_KO] = "A_KO"
            outcome_codes[mask_B_KO] = "B_KO"

            idx_both = np.where(mask_both)[0]
            idx_none = np.where(mask_none)[0]
            if len(idx_both) > 0:
                rand_both = rng.random(len(idx_both))
                outcome_codes[idx_both] = np.where(rand_both < pA, "A_DEC", "B_DEC")
            if len(idx_none) > 0:
                rand_none = rng.random(len(idx_none))
                outcome_codes[idx_none] = np.where(rand_none < pA, "A_DEC", "B_DEC")

            outcomes = pd.Series(outcome_codes)
            pA_total = (outcomes.str.startswith("A")).mean()
            pB_total = (outcomes.str.startswith("B")).mean()
            koA_pct = (outcomes == "A_KO").mean() * 100
            koB_pct = (outcomes == "B_KO").mean() * 100
            decA_pct = (outcomes == "A_DEC").mean() * 100
            decB_pct = (outcomes == "B_DEC").mean() * 100

            st.markdown("### 📊 Resultados Monte Carlo")
            cA, cB, cC, cD = st.columns(4)
            cA.metric("A por KO/TKO", f"{koA_pct:.1f}%")
            cB.metric("A por DEC", f"{decA_pct:.1f}%")
            cC.metric("B por KO/TKO", f"{koB_pct:.1f}%")
            cD.metric("B por DEC", f"{decB_pct:.1f}%")

            ev_a_mc = odd_a * pA_total - 1
            ev_b_mc = odd_b * pB_total - 1
            st.info(f"EV con Monte Carlo → A: {(ev_a_mc*100):.2f}% · B: {(ev_b_mc*100):.2f}%")

            # Gráfico de distribución de resultados
            summary = pd.DataFrame({
                "Outcome": ["A_KO", "A_DEC", "B_KO", "B_DEC"],
                "Pct": [koA_pct, decA_pct, koB_pct, decB_pct]
            })
            chart = alt.Chart(summary).mark_bar().encode(
                x=alt.X("Outcome:N", title="Resultado"),
                y=alt.Y("Pct:Q", title="Porcentaje (%)"),
                tooltip=["Outcome", "Pct"],
                color=alt.condition(alt.datum["Pct"] > 0, alt.value("#00C853"), alt.value("#FF5252"))
            ).properties(height=300, title="Distribución de Resultados (MC)")
            st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.error(f"Error en el módulo Boxeo: {e}")

