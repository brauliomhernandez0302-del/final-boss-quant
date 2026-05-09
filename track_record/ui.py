"""
track_record/ui.py — Streamlit page for the public live track record.

Render with:
    from track_record.ui import render_track_record
    render_track_record()
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def render_track_record() -> None:
    import streamlit as st

    from track_record.db import TrackRecordDB
    from track_record.stats import compute_stats

    st.title("Track Record Publico")
    st.caption(
        "Cada pick tiene un timestamp publicado **antes** del inicio del juego. "
        "Los resultados se resuelven automaticamente via MLB Stats API."
    )

    db = TrackRecordDB()

    # ── controls ─────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        sport_filter = st.selectbox("Deporte", ["Todos", "MLB", "NBA", "UFC"])
    with c2:
        result_filter = st.selectbox("Resultado", ["Todos", "WIN", "LOSS", "PUSH", "Pendiente"])
    with c3:
        if st.button("Actualizar"):
            st.rerun()

    sport_arg = None if sport_filter == "Todos" else sport_filter

    # load stats
    with st.spinner("Cargando estadisticas..."):
        stats = compute_stats(db=db, sport=sport_arg)

    hl = stats["headline"]

    # ── headline metrics ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Resumen General")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Picks Publicados", hl["total_picks"])
    m2.metric(
        "Record (W-L)",
        f"{hl['wins']}-{hl['losses']}",
        delta=f"+{hl['pushes']}P" if hl["pushes"] else None,
    )
    m3.metric(
        "Win Rate",
        f"{hl['win_rate']:.1%}",
        delta=f"vs 50% base",
        delta_color="normal",
    )
    roi_color = "normal" if hl["roi_pct"] >= 0 else "inverse"
    m4.metric(
        "ROI",
        f"{hl['roi_pct']:+.1f}%",
        delta_color=roi_color,
    )
    m5.metric(
        "Unidades P&L",
        f"{hl['total_units']:+.2f}u",
        delta_color="normal" if hl["total_units"] >= 0 else "inverse",
    )
    m6.metric("Sharpe", f"{hl['sharpe']:.2f}")

    # ── bankroll curve ────────────────────────────────────────────────────────
    bk = stats["bankroll_curve"]
    if bk:
        st.markdown("---")
        st.subheader("Curva de Bankroll (unidades)")
        bk_df = pd.DataFrame(bk)
        if "running_total" in bk_df.columns and "date" in bk_df.columns:
            bk_df = bk_df.sort_values("date")
            st.line_chart(bk_df.set_index("date")["running_total"])
    else:
        st.info("Sin picks resueltos todavia — la curva aparecera cuando se completen juegos.")

    # ── breakdown tables ──────────────────────────────────────────────────────
    st.markdown("---")
    tab_sport, tab_market, tab_tier, tab_month = st.tabs(
        ["Por Deporte", "Por Mercado", "Por Tier", "Por Mes"]
    )

    def _breakdown_df(data: Dict[str, Dict], label: str) -> pd.DataFrame:
        rows = []
        for key, d in data.items():
            wl = d["wins"] + d["losses"]
            rows.append({
                label:     key,
                "W":       d["wins"],
                "L":       d["losses"],
                "P":       d.get("pushes", 0),
                "WR":      f"{d['win_rate']:.1%}",
                "ROI":     f"{d['roi_pct']:+.1f}%",
                "P&L (u)": f"{d['pnl']:+.2f}",
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    with tab_sport:
        df = _breakdown_df(stats["by_sport"], "Deporte")
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Sin datos")

    with tab_market:
        df = _breakdown_df(stats["by_market"], "Mercado")
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Sin datos")

    with tab_tier:
        df = _breakdown_df(stats["by_tier"], "Tier")
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Sin datos")

    with tab_month:
        df = _breakdown_df(stats["by_month"], "Mes")
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Sin datos")

    # ── recent picks table ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Picks Recientes")

    recent = stats["recent_picks"]
    if result_filter != "Todos":
        if result_filter == "Pendiente":
            recent = [r for r in recent if r["result"] is None]
        else:
            recent = [r for r in recent if r["result"] == result_filter]

    if recent:
        picks_df = pd.DataFrame(recent)

        # style result column
        def _color_result(val: Any) -> str:
            if val == "WIN":
                return "background-color: #1a3a1a; color: #00C853"
            if val == "LOSS":
                return "background-color: #3a1a1a; color: #FF5252"
            if val == "PUSH":
                return "background-color: #1a2a3a; color: #FDD835"
            return "color: #90A4AE"

        display_cols = [
            "published_at", "game_date", "sport", "matchup",
            "market", "model_prob", "ev_pct", "tier",
            "odds", "stake", "result", "score", "pnl",
        ]
        display_cols = [c for c in display_cols if c in picks_df.columns]
        picks_df = picks_df[display_cols].copy()
        picks_df["published_at"] = picks_df["published_at"].str[:16].str.replace("T", " ")
        picks_df["model_prob"] = picks_df["model_prob"].map(
            lambda x: f"{x:.1%}" if x is not None else "—"
        )
        picks_df["ev_pct"] = picks_df["ev_pct"].map(
            lambda x: f"{x:+.1%}" if x is not None else "—"
        )
        picks_df["pnl"] = picks_df["pnl"].map(
            lambda x: f"{x:+.2f}u" if x is not None else "—"
        )
        picks_df["odds"] = picks_df["odds"].map(
            lambda x: f"{x:.2f}" if x is not None else "—"
        )

        styled = picks_df.style.applymap(_color_result, subset=["result"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("No hay picks que coincidan con el filtro seleccionado.")

    # ── pending picks ─────────────────────────────────────────────────────────
    pending_rows = db.get_pending(sport=sport_arg)
    if pending_rows:
        st.markdown("---")
        st.subheader(f"Picks Pendientes ({len(pending_rows)})")
        pend_df = pd.DataFrame(
            [
                {
                    "Publicado": row["published_at"][:16].replace("T", " "),
                    "Juego":     row["game_date"],
                    "Matchup":  f"{row['away_team']} @ {row['home_team']}",
                    "Mercado":   row["market"],
                    "Prob":      f"{(row['model_prob'] or 0):.1%}",
                    "EV":        f"{(row['ev_pct'] or 0):+.1%}",
                    "Tier":      row["confidence_tier"] or "—",
                    "Odds":      f"{row['odds_decimal']:.2f}" if row["odds_decimal"] else "—",
                    "Stake":     f"{row['stake_units']:.2f}u" if row["stake_units"] else "—",
                }
                for row in pending_rows
            ]
        )
        st.dataframe(pend_df, use_container_width=True, hide_index=True)

    # ── audit note ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Picks generados automaticamente por FINAL BOSS QUANT G8+. "
        "Los timestamps 'published_at' son UTC y se registran antes del primer lanzamiento. "
        "Los resultados se resuelven via MLB Stats API (free tier). "
        "No se editan picks retroactivamente."
    )
