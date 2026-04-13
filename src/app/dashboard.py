# src/app/dashboard.py
import streamlit as st
import plotly.express as px
import pandas as pd

def render(history: pd.DataFrame, metrics: dict):
    """Affiche le graphique de valeur du portefeuille et les KPI."""
    if history.empty:
        st.info("Pas d’historique disponible – lancez d’abord une simulation.")
        return

    # le DataFrame possède déjà la colonne `date` (type datetime.date)
    fig = px.line(
        history,
        x="date",
        y="total_value",
        title="Valeur du portefeuille",
        labels={"date": "Date", "total_value": "Valeur (USDT)"},
    )
    fig.update_layout(
        plot_bgcolor="#111111", paper_bgcolor="#111111", font_color="#eeeeee"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- KPI ---------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("ROI %", f"{metrics.get('roi_pct', 0.0):.2f}%")
    col2.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0.0):.2f}")
    col3.metric("Max‑drawdown", f"{metrics.get('max_drawdown_pct', 0.0):.2f}%")
    col4.metric("Volatilité (ann.)", f"{metrics.get('volatility_pct', 0.0):.2f}%")
    col5.metric("VaR 95 %", f"{metrics.get('var_95_pct', 0.0):.2f}%")
