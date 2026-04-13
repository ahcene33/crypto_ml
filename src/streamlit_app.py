# src/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

# ----------------------------------------------------------------------
# Imports absolus du package « src »
# (le répertoire racine du projet est automatiquement dans le PYTHONPATH
#  quand on lance :  `streamlit run src/streamlit_app.py`)
# ----------------------------------------------------------------------
from src.portfolio import predict_all, run_simulation, get_live_prices

# ----------------------------------------------------------------------
# Configuration globale de la page
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto‑ML Portfolio",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀",
)

# ----------------------------------------------------------------------
# ── Sidebar – paramètres de simulation ───────────────────────────────────
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Paramètres de la simulation")
with st.sidebar.form(key="sim_form"):
    start_date = st.date_input(
        "Date de départ", value=date(2026, 1, 22), min_value=date(2025, 1, 1)
    )
    end_date = st.date_input(
        "Date de fin", value=datetime.today().date()
    )
    capital = st.number_input(
        "Capital initial (USDT)", min_value=0.0, value=200.0, step=10.0
    )
    monthly = st.number_input(
        "Apport mensuel (≈ USDT)", min_value=0.0, value=50.0, step=5.0
    )
    submitted = st.form_submit_button("▶️ Lancer la simulation")

# ----------------------------------------------------------------------
# Gestion du cache des résultats de simulation (évite de tout recalculer
# à chaque rafraîchissement du tableau Live Prices)
# ----------------------------------------------------------------------
if "simulation_result" not in st.session_state:
    st.session_state["simulation_result"] = None

if submitted:
    with st.spinner("🔧 Simulation en cours…"):
        sim = run_simulation(
            start_date=start_date,
            end_date=end_date,
            initial_capital=capital,
            monthly_deposit=monthly,
        )
        st.session_state["simulation_result"] = sim
else:
    sim = st.session_state["simulation_result"]

# ----------------------------------------------------------------------
# ── Mise en page principale – onglets
# ----------------------------------------------------------------------
st.title("🚀 Crypto‑ML Portfolio Dashboard")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Valeur du portefeuille",
        "🔎 Analyse crypto",
        "💼 Portefeuille (prédictions)",
        "💡 Description de la stratégie",
        "💹 Live Prices (Binance)",
    ]
)

# ----------------------------------------------------------------------
# 1️⃣ Valeur du portefeuille + KPI
# ----------------------------------------------------------------------
with tab1:
    if sim is None:
        st.info(
            "⚡️ Lancez la simulation depuis le panneau de gauche pour afficher les KPI "
            "et les graphiques."
        )
    else:
        hist = sim["portfolio_history"]
        mets = sim["metrics"]

        # ----------- Graphique principal (Valeur totale) -------------
        fig_value = px.line(
            hist,
            x="date",
            y="total_value",
            title="Valeur totale du portefeuille",
            labels={"date": "Date", "total_value": "Valeur (USDT)"},
        )
        fig_value.update_traces(mode="lines+markers", line_color="#00cc96")
        fig_value.update_layout(
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font_color="#eeeeee",
            hovermode="x unified",
        )

        # ----------- Cash & positions (stacked area) -----------------
        fig_cash = go.Figure()
        fig_cash.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["cash"],
                name="Cash",
                mode="lines",
                line=dict(color="#ff6b6b"),
                stackgroup="one",
            )
        )
        fig_cash.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["positions_value"],
                name="Positions",
                mode="lines",
                line=dict(color="#4dabf7"),
                stackgroup="one",
            )
        )
        fig_cash.update_layout(
            title="Répartition Cash / Positions",
            yaxis_title="USDT",
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font_color="#eeeeee",
            hovermode="x unified",
        )

        # ----------- Draw‑down (aire en rouge) -----------------------
        drawdown = (
            hist["total_value"] / hist["total_value"].cummax() - 1
        ) * 100
        fig_dd = px.area(
            x=hist["date"],
            y=drawdown,
            title="Draw‑down (en %)",
            labels={"x": "Date", "y": "Draw‑down (%)"},
        )
        fig_dd.update_traces(fillcolor="rgba(255,0,0,0.3)", line_color="#ff4444")
        fig_dd.update_layout(
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font_color="#eeeeee",
            hovermode="x unified",
        )

        # ----------- Affichage des métriques -------------------------
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("ROI %", f"{mets.get('roi_pct', 0):.2f}%")
        col2.metric("Sharpe", f"{mets.get('sharpe_ratio', 0):.2f}")
        col3.metric("Max‑drawdown", f"{mets.get('max_drawdown_pct', 0):.2f}%")
        col4.metric("Volatilité (ann.)", f"{mets.get('volatility_pct', 0):.2f}%")
        col5.metric("VaR 95 %", f"{mets.get('var_95_pct', 0):.2f}%")
        col6.metric("ES 95 %", f"{mets.get('es_95_pct', 0):.2f}%")

        # ----------- Render ----------
        st.plotly_chart(fig_value, use_container_width=True)
        st.plotly_chart(fig_cash, use_container_width=True)
        st.plotly_chart(fig_dd, use_container_width=True)

# ----------------------------------------------------------------------
# 2️⃣ Analyse crypto (tableau + top‑15)
# ----------------------------------------------------------------------
with tab2:
    if sim is None:
        st.info("⚡️ Exécutez la simulation pour visualiser les prédictions.")
    else:
        today = pd.Timestamp(datetime.now().date())
        try:
            df_pred = predict_all(today)
            df_pred = df_pred.sort_values("score", ascending=False)

            # ---- Tableau complet (premières lignes) ----
            with st.expander("🔎 Tableau complet – 20 meilleures prédictions"):
                st.dataframe(df_pred.head(20), use_container_width=True)

            # ---- Top‑15 bar chart (score + signal) ----
            fig_top = px.bar(
                df_pred.head(15),
                x="symbol",
                y="score",
                color="signal",
                color_discrete_map={0: "#ff4444", 1: "#00ff88"},
                title="Top‑15 cryptos – Score (probabilité × (1‑risque))",
                labels={"score": "Score", "symbol": "Crypto"},
                text="score",
            )
            fig_top.update_traces(textposition="outside")
            fig_top.update_layout(
                plot_bgcolor="#111111",
                paper_bgcolor="#111111",
                font_color="#eeeeee",
                xaxis_tickangle=-45,
                uniformtext_minsize=8,
                uniformtext_mode="hide",
            )
            st.plotly_chart(fig_top, use_container_width=True)

        except Exception as exc:
            st.error(f"❗️ Impossible de récupérer les prédictions : {exc}")

# ----------------------------------------------------------------------
# 3️⃣ Portefeuille actuel – visualisation des signaux du jour
# ----------------------------------------------------------------------
with tab3:
    if sim is None:
        st.info("⚡️ Lancer la simulation pour afficher les signaux d’investissement.")
    else:
        # On ré‑utilise les prédictions du dernier jour de la simulation
        last_date = sim["portfolio_history"]["date"].iloc[-1]
        df_last_pred = predict_all(pd.Timestamp(last_date))

        # Filtrer uniquement les actifs avec signal = BUY
        df_buy = df_last_pred[df_last_pred["signal"] == 1]

        # ------------------------------------------------------------
        # Section résumé (cards)
        # ------------------------------------------------------------
        st.subheader(f"📊 Signals d’achat au {last_date}")

        if df_buy.empty:
            st.info("❌ Aucun signal BUY détecté à la dernière date.")
        else:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Nb. actifs BUY", f"{len(df_buy)}")
            col_b.metric(
                "Score moyen", f"{df_buy['score'].mean():.3f}"
            )
            col_c.metric(
                "Probabilité moyenne", f"{df_buy['probability'].mean():.3f}"
            )

            # ------------------------------------------------------------
            # Tableau détaillé
            # ------------------------------------------------------------
            with st.expander("🗂️ Détail des actifs BUY"):
                st.dataframe(
                    df_buy[
                        ["symbol", "price", "probability", "score", "category"]
                    ].reset_index(drop=True),
                    use_container_width=True,
                )

# ----------------------------------------------------------------------
# 4️⃣ Description de la stratégie – texte explicatif
# ----------------------------------------------------------------------
with tab4:
    st.markdown(
        """
        ### 📋 Rappel de la stratégie

        - **Capital initial** : 200 USDT (le 22/01/2026)  
        - **Apport mensuel** : 50 USDT le jour **10** de chaque mois.  
        - **Logique de décision** :
          1. Chaque jour on calcule le **score** = `probabilité × (1‑risque)`.  
          2. **Le 10 du mois** :
             - on vend toutes les positions dont le **signal = SELL**.  
             - on achète les **2 cryptos** avec le **plus haut score** dont le **signal = BUY**, à parts égales.  
        - **KPI affichés** : ROI, Sharpe, Max‑drawdown, Volatilité, VaR 95 % et ES 95 %.  

        Les indicateurs VaR/ES sont calculés **historique‑daily** à partir des
        rendements du portefeuille (méthode non‑paramétrique – percentile 5 %).  
        """
    )
    st.info(
        "✅ Vous pouvez relancer la simulation avec d’autres dates ou paramètres "
        "et observer l’impact immédiatement sur les KPI et les graphiques."
    )

# ----------------------------------------------------------------------
# 5️⃣ Live Prices – cotations en temps réel via Binance
# ----------------------------------------------------------------------
with tab5:
    st.subheader("💹 Cours USDT en temps réel (endpoint public Binance)")
    with st.spinner("🔄 Récupération des prix…"):
        df_live = get_live_prices()

    if df_live.empty:
        st.warning(
            "⚠️ Aucun prix n’a pu être récupéré – vérifiez votre connexion "
            "Internet ou l’accessibilité du service Binance."
        )
    else:
        # ------------------- Tableau -------------------
        st.dataframe(df_live, use_container_width=True)

        # ------------------- Bar chart -----------------
        fig_live = px.bar(
            df_live.sort_values("price", ascending=False),
            x="symbol",
            y="price",
            title="Prix USDT actuels (départ de 0 USDT)",
            labels={"price": "Prix (USDT)", "symbol": "Crypto"},
            color="price",
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        fig_live.update_layout(
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font_color="#eeeeee",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_live, use_container_width=True)

# ----------------------------------------------------------------------
# Footer – mentions légales
# ----------------------------------------------------------------------
st.caption(
    """
    ⚠️ Les cours affichés proviennent du **endpoint public de Binance**
    (pas besoin de clé API). Aucun appel ne conserve d’état ; chaque rafraîchissement
    interroge l’API.  

    📈 Cette application est destinée à un usage de **démo / recherche** et ne
    constitue en aucun cas un conseil d’investissement.  
    """
)
