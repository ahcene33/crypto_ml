# src/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots      # <-- NEW
from datetime import datetime, date
from pathlib import Path                       # <-- NEW

# ----------------------------------------------------------------------
# Imports absolus du package « src »
# ----------------------------------------------------------------------
from portfolio import predict_all, run_simulation, get_live_prices

# ----------------------------------------------------------------------
# Fonctions utilitaires -------------------------------------------------
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_signals(dates):
    """
    Calcule les prédictions (incluant le champ `signal`) pour chaque date fournie.
    """
    frames = []
    for d in dates:
        pred = predict_all(pd.Timestamp(d))
        wanted = ["symbol", "signal", "probability", "score", "category", "price"]
        cols = [c for c in wanted if c in pred.columns]
        pred = pred[cols]
        pred["date"] = pd.Timestamp(d)
        frames.append(pred)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["date", "symbol", "signal", "probability", "score", "category", "price"])

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
# Sidebar – paramètres de simulation
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
# Gestion du cache des résultats de simulation (évite de tout recalculer)
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
# Onglets
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📈 Valeur du portefeuille",
        "🔎 Analyse crypto",
        "💼 Portefeuille (predictions)",
        "💡 Description de la stratégie",
        "💹 Live Prices (Binance)",
        "🔔 Signaux (temps)",
        "📊 Comparaison prix BTC vs crypto low‑price",   # <-- NEW TAB
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

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("ROI %", f"{mets.get('roi_pct', 0):.2f}%")
        col2.metric("Sharpe", f"{mets.get('sharpe_ratio', 0):.2f}")
        col3.metric("Max‑drawdown", f"{mets.get('max_drawdown_pct', 0):.2f}%")
        col4.metric("Volatilité (ann.)", f"{mets.get('volatility_pct', 0):.2f}%")
        col5.metric("VaR 95 %", f"{mets.get('var_95_pct', 0):.2f}%")
        col6.metric("ES 95 %", f"{mets.get('es_95_pct', 0):.2f}%")

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

            with st.expander("🔎 Tableau complet – 20 meilleures prédictions"):
                st.dataframe(df_pred.head(20), use_container_width=True)

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
        last_date = sim["portfolio_history"]["date"].iloc[-1]
        df_last_pred = predict_all(pd.Timestamp(last_date))
        df_buy = df_last_pred[df_last_pred["signal"] == 1]

        st.subheader(f"📊 Signals d’achat au {last_date}")

        if df_buy.empty:
            st.info("❌ Aucun signal BUY détecté à la dernière date.")
        else:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Nb. actifs BUY", f"{len(df_buy)}")
            col_b.metric("Score moyen", f"{df_buy['score'].mean():.3f}")
            col_c.metric("Probabilité moyenne", f"{df_buy['probability'].mean():.3f}")

            with st.expander("🗂️ Détail des actifs BUY"):
                st.dataframe(
                    df_buy[
                        ["symbol", "price", "probability", "score", "category"]
                    ].reset_index(drop=True),
                    use_container_width=True,
                )

# ----------------------------------------------------------------------
# 4️⃣ Description de la stratégie
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
        st.dataframe(df_live, use_container_width=True)

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
# 6️⃣ Signaux (temps) – nombre BUY / SELL par jour
# ----------------------------------------------------------------------
with tab6:
    if sim is None:
        st.info("⚡️ Lancez la simulation pour visualiser les signaux dans le temps.")
    else:
        signals_df = st.session_state.get("signals_df")
        if signals_df is None or signals_df.empty:
            st.warning("Aucun signal n’a pu être calculé.")
        else:
            signals_df["date"] = pd.to_datetime(signals_df["date"])
            counts = (
                signals_df.groupby(["date", "signal"])
                .size()
                .reset_index(name="count")
                .pivot(index="date", columns="signal", values="count")
                .fillna(0)
                .astype(int)
            )
            for col in [0, 1]:
                if col not in counts.columns:
                    counts[col] = 0
            counts = counts[[0, 1]]
            counts.columns = ["SELL", "BUY"]

            fig_sign = go.Figure()
            fig_sign.add_trace(
                go.Scatter(
                    x=counts.index,
                    y=counts["BUY"],
                    name="BUY",
                    mode="lines",
                    line=dict(color="#00cc96"),
                )
            )
            fig_sign.add_trace(
                go.Scatter(
                    x=counts.index,
                    y=counts["SELL"],
                    name="SELL",
                    mode="lines",
                    line=dict(color="#ff4444"),
                )
            )
            rebalance_dates = [d for d in counts.index if d.day == 10]
            for rd in rebalance_dates:
                fig_sign.add_vline(
                    x=rd,
                    line_width=1,
                    line_dash="dot",
                    line_color="gray",
                )
            fig_sign.update_layout(
                title="Nombre de signaux BUY / SELL par jour",
                xaxis_title="Date",
                yaxis_title="Nb. cryptos",
                plot_bgcolor="#111111",
                paper_bgcolor="#111111",
                font_color="#eeeeee",
                hovermode="x unified",
            )
            st.plotly_chart(fig_sign, use_container_width=True)

            min_dt = signals_df["date"].dt.date.min()
            max_dt = signals_df["date"].dt.date.max()
            selected_date = st.date_input(
                "Date à explorer",
                value=min_dt,
                min_value=min_dt,
                max_value=max_dt,
            )
            day_df = signals_df[signals_df["date"].dt.date == selected_date]

            nb_buy = (day_df["signal"] == 1).sum()
            nb_sell = (day_df["signal"] == 0).sum()
            c1, c2 = st.columns(2)
            c1.metric("Signals BUY", nb_buy)
            c2.metric("Signals SELL", nb_sell)

            with st.expander(f"Détails des signaux le {selected_date}"):
                if day_df.empty:
                    st.info("Aucun signal enregistré pour cette date.")
                else:
                    st.dataframe(
                        day_df[
                            ["symbol", "signal", "probability", "score", "category", "price"]
                        ].reset_index(drop=True),
                        use_container_width=True,
                    )

# ----------------------------------------------------------------------
# 7️⃣ Comparaison BTC vs crypto low‑price (dual‑axis)
# ----------------------------------------------------------------------
with tab7:
    st.subheader("🔎 BTC ↔ Crypto à petit prix (axe secondaire)")

    # ------------------------------------------------------------------
    # 1️⃣  Récupérer la liste des symboles disponibles
    # ------------------------------------------------------------------
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    all_symbols = sorted([p.stem.upper() for p in raw_dir.glob("*.parquet")])

    # ------------------------------------------------------------------
    # 2️⃣  Trouver la crypto la moins chère (exclure BTC)
    # ------------------------------------------------------------------
    latest_prices = {}
    for sym in all_symbols:
        df_tmp = pd.read_parquet(raw_dir / f"{sym}.parquet")
        latest_prices[sym] = float(df_tmp["price"].iloc[-1])

    cheap_sym = min(
        (s for s in all_symbols if s != "BTC"),
        key=lambda s: latest_prices[s],
        default=None,
    )

    # ------------------------------------------------------------------
    # 3️⃣  Sélecteur (l’utilisateur peut choisir une autre crypto)
    # ------------------------------------------------------------------
    chosen_sym = st.selectbox(
        "Crypto faible prix (défaut = la plus basse)",
        options=all_symbols,
        index=all_symbols.index(cheap_sym) if cheap_sym else 0,
    )

    # ------------------------------------------------------------------
    # 4️⃣  Charger les deux séries temporelles
    # ------------------------------------------------------------------
    df_btc = pd.read_parquet(raw_dir / "BTC.parquet")
    df_alt = pd.read_parquet(raw_dir / f"{chosen_sym}.parquet")

    # Align dates (intersection)
    common_dates = df_btc["date"].isin(df_alt["date"])
    df_btc = df_btc[common_dates].reset_index(drop=True)
    df_alt = df_alt[df_alt["date"].isin(df_btc["date"])].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5️⃣  Dual‑axis plot (left = crypto low‑price, right = BTC)
    # ------------------------------------------------------------------
    fig_cmp = make_subplots(specs=[[{"secondary_y": True}]])
    # Left axis – crypto low‑price (primary)
    fig_cmp.add_trace(
        go.Scatter(
            x=df_alt["date"],
            y=df_alt["price"],
            name=chosen_sym,
            line=dict(color="#00cc96", dash="dot", width=2),
        ),
        secondary_y=False,
    )
    # Right axis – BTC (secondary)
    fig_cmp.add_trace(
        go.Scatter(
            x=df_btc["date"],
            y=df_btc["price"],
            name="BTC",
            line=dict(color="#ff6b6b", width=2),
        ),
        secondary_y=True,
    )
    fig_cmp.update_layout(
        title=f"BTC vs {chosen_sym} (échelle adaptée)",
        hovermode="x unified",
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font_color="#eeeeee",
    )
    fig_cmp.update_yaxes(title_text=f"{chosen_sym} (USDT)", secondary_y=False, color="#00cc96")
    fig_cmp.update_yaxes(title_text="BTC (USDT)", secondary_y=True, color="#ff6b6b")

    st.plotly_chart(fig_cmp, use_container_width=True)

    # ------------------------------------------------------------------
    # 6️⃣  Derniers prix (mise en évidence)
    # ------------------------------------------------------------------
    col_l, col_r = st.columns(2)
    col_l.metric(f"Dernier prix {chosen_sym}", f"{df_alt['price'].iloc[-1]:,.4f} USDT")
    col_r.metric("Dernier prix BTC", f"{df_btc['price'].iloc[-1]:,.2f} USDT")

# ----------------------------------------------------------------------
# Footer – mentions légales
# ----------------------------------------------------------------------
st.caption(
    """
    ⚠️ Les cours affichés proviennent du **endpoint public de Binance**
    (pas besoin de clé API). Aucun appel ne conserve d’état ; chaque rafraîchissement
    interroge l’API.

    📈 Cette application est destinée à un usage de **demo / recherche** et ne
    constitue en aucun cas un conseil d’investissement.
    """
)
