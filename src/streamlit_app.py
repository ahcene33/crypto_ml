# src/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
from pathlib import Path
import pickle
import logging

# ----------------------------------------------------------------------
# Imports absolus du package « src »
# ----------------------------------------------------------------------
from portfolio import predict_all, run_simulation

# ----------------------------------------------------------------------
# Logging (utile pour le débogage)
# ----------------------------------------------------------------------
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Fonctions utilitaires -------------------------------------------------
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_signals(dates):
    """Calcule les prédictions (incluant le champ `signal`) pour chaque date."""
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
    return pd.DataFrame(
        columns=[
            "date",
            "symbol",
            "signal",
            "probability",
            "score",
            "category",
            "price",
        ]
    )


# ----------------------------------------------------------------------
# Chemin de persistance de la simulation
# ----------------------------------------------------------------------
SIM_PATH = Path(__file__).resolve().parents[1] / "simulation.pkl"


def _save_simulation(sim):
    """Sauvegarde la simulation au format pickle (robuste, simple)."""
    try:
        with open(SIM_PATH, "wb") as f:
            pickle.dump(sim, f)
    except Exception as exc:
        st.error(f"Erreur lors de la sauvegarde de la simulation : {exc}")


def _load_simulation():
    """Charge la simulation depuis le disque, ou renvoie None."""
    try:
        with open(SIM_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


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
# Sidebar – paramètres de simulation + navigation
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
    reset_sim = st.form_submit_button("🔄 Reset simulation")

# Navigation dropdown (remplace les onglets)
page_options = [
    "📈 Valeur du portefeuille",
    "🔎 Analyse crypto",
    "💼 Portefeuille (predictions)",
    "💡 Description de la stratégie",
    "🔔 Signaux (temps)",
    "📊 Comparaison prix BTC vs crypto low‑price",
]
selected_page = st.sidebar.selectbox("📖 Navigation", options=page_options)

# ----------------------------------------------------------------------
# Gestion du cache des résultats de simulation
# ----------------------------------------------------------------------
if "simulation_result" not in st.session_state:
    st.session_state["simulation_result"] = None

# Reset demandé → on purge le cache + le fichier persistant
if reset_sim:
    st.session_state["simulation_result"] = None
    if "signals_df" in st.session_state:
        del st.session_state["signals_df"]
    if SIM_PATH.is_file():
        SIM_PATH.unlink()
    st.experimental_rerun()

# Lancement ou chargement de la simulation
if submitted:
    with st.spinner("🔧 Simulation en cours…"):
        sim = run_simulation(
            start_date=start_date,
            end_date=end_date,
            initial_capital=capital,
            monthly_deposit=monthly,
        )
        # Calcul des signaux (temps) une fois pour éviter de le refaire à chaque affichage
        dates = (
            pd.to_datetime(sim["portfolio_history"]["date"])
            .dt.date
            .unique()
            .tolist()
        )
        st.session_state["signals_df"] = compute_signals(dates)

        st.session_state["simulation_result"] = sim
        _save_simulation(sim)
else:
    # Pas de nouvelle soumission → on tente de charger la précédente
    if st.session_state["simulation_result"] is None:
        st.session_state["simulation_result"] = _load_simulation()
    sim = st.session_state["simulation_result"]
    if sim and "signals_df" not in st.session_state:
        dates = (
            pd.to_datetime(sim["portfolio_history"]["date"])
            .dt.date
            .unique()
            .tolist()
        )
        st.session_state["signals_df"] = compute_signals(dates)

# ----------------------------------------------------------------------
# Fonctions d’affichage pour chaque page
# ----------------------------------------------------------------------
def show_portfolio_value(sim):
    """Valeur du portefeuille + KPI + historique des transactions."""
    if sim is None:
        st.info(
            "⚡️ Lancez la simulation depuis le panneau de gauche pour afficher les KPI "
            "et les graphiques."
        )
        return

    hist = sim["portfolio_history"]
    mets = sim["metrics"]
    txns = sim.get("transactions", pd.DataFrame())

    # Graphiques principaux
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

    drawdown = (hist["total_value"] / hist["total_value"].cummax() - 1) * 100
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

    # KPI
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("ROI %", f"{mets.get('roi_pct', 0):.2f}%")
    col2.metric("Sharpe", f"{mets.get('sharpe_ratio', 0):.2f}")
    col3.metric("Max‑drawdown", f"{mets.get('max_drawdown_pct', 0):.2f}%")
    col4.metric("Volatilité (ann.)", f"{mets.get('volatility_pct', 0):.2f}%")
    col5.metric("VaR 95 %", f"{mets.get('var_95_pct', 0):.2f}%")
    col6.metric("ES 95 %", f"{mets.get('es_95_pct', 0):.2f}%")

    # Affichage
    st.plotly_chart(fig_value, use_container_width=True)
    st.plotly_chart(fig_cash, use_container_width=True)
    st.plotly_chart(fig_dd, use_container_width=True)

    # Tableau des transactions (bonus)
    with st.expander("📄 Historique des transactions"):
        if txns.empty:
            st.info("Aucune transaction enregistrée.")
        else:
            # réordonner les colonnes pour plus de lisibilité
            txns = txns[
                ["date", "symbol", "type", "price", "quantity", "cash_change"]
            ].sort_values("date", ascending=False)
            st.dataframe(txns, use_container_width=True)


def show_crypto_analysis(sim):
    """Analyse crypto – tableau complet + top‑15."""
    if sim is None:
        st.info("⚡️ Exécutez la simulation pour visualiser les prédictions.")
        return

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


def show_portfolio_predictions(sim):
    """Portefeuille actuel – visualisation des signaux du jour."""
    if sim is None:
        st.info("⚡️ Lancer la simulation pour afficher les signaux d’investissement.")
        return

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
                df_buy[["symbol", "price", "probability", "score", "category"]]
                .reset_index(drop=True),
                use_container_width=True,
            )


def show_strategy_description():
    """Bloc explicatif de la stratégie."""
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
        - **KPI affichés** : ROI, Sharpe, Max‑draw‑down, Volatilité, VaR 95 % et ES 95 %.  

        Les indicateurs VaR/ES sont calculés **historique‑daily** à partir des
        rendements du portefeuille (méthode non‑paramétrique – percentile 5 %).  
        """
    )
    st.info(
        "✅ Vous pouvez relancer la simulation avec d’autres dates ou paramètres "
        "et observer l’impact immédiatement sur les KPI et les graphiques."
    )


def show_signals(sim):
    """Visualisation du nombre de signaux BUY / SELL au fil du temps."""
    if sim is None:
        st.info("⚡️ Lancez la simulation pour visualiser les signaux dans le temps.")
        return

    signals_df = st.session_state.get("signals_df")
    if signals_df is None or signals_df.empty:
        st.warning("Aucun signal n’a pu être calculé.")
        return

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

    # Exploration d’une date
    min_dt = signals_df["date"].dt.date.min()
    max_dt = signals_df["date"].dt.date.max()
    selected_date = st.date_input(
        "Date à explorer", value=min_dt, min_value=min_dt, max_value=max_dt
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


def show_btc_vs_lowprice():
    """Comparaison du prix du BTC avec la crypto la moins chère (dual‑axis)."""
    st.subheader("🔎 BTC ↔ Crypto à petit prix (axe secondaire)")

    # 1️⃣  Liste des symboles disponibles
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    all_symbols = sorted([p.stem.upper() for p in raw_dir.glob("*.parquet")])

    # 2️⃣  Crypto la moins chère (exclure BTC)
    latest_prices = {}
    for sym in all_symbols:
        try:
            df_tmp = pd.read_parquet(raw_dir / f"{sym}.parquet")
            latest_prices[sym] = float(df_tmp["price"].iloc[-1])
        except Exception:
            latest_prices[sym] = float("nan")
    cheap_sym = min(
        (s for s in all_symbols if s != "BTC" and not pd.isna(latest_prices[s])),
        key=lambda s: latest_prices[s],
        default=None,
    )

    # 3️⃣  Sélecteur (l’utilisateur peut choisir une autre crypto)
    chosen_sym = st.selectbox(
        "Crypto faible prix (défaut = la plus basse)",
        options=all_symbols,
        index=all_symbols.index(cheap_sym) if cheap_sym else 0,
    )

    # 4️⃣  Charger les deux séries temporelles – reset index → colonne "date"
    df_btc = pd.read_parquet(raw_dir / "BTC.parquet").reset_index()
    df_alt = pd.read_parquet(raw_dir / f"{chosen_sym}.parquet").reset_index()

    # renommer la colonne d’index en "date" si besoin
    if df_btc.columns[0] != "date":
        df_btc.rename(columns={df_btc.columns[0]: "date"}, inplace=True)
    if df_alt.columns[0] != "date":
        df_alt.rename(columns={df_alt.columns[0]: "date"}, inplace=True)

    # 5️⃣  Alignement des dates (intersection)
    common_dates = df_btc["date"].isin(df_alt["date"])
    df_btc = df_btc[common_dates].reset_index(drop=True)
    df_alt = df_alt[df_alt["date"].isin(df_btc["date"])].reset_index(drop=True)

    # 6️⃣  Dual‑axis plot
    fig_cmp = make_subplots(specs=[[{"secondary_y": True}]])

    fig_cmp.add_trace(
        go.Scatter(
            x=df_alt["date"],
            y=df_alt["price"],
            name=chosen_sym,
            line=dict(color="#00cc96", dash="dot", width=2),
        ),
        secondary_y=False,
    )
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
    fig_cmp.update_yaxes(
        title_text=f"{chosen_sym} (USDT)", secondary_y=False, color="#00cc96"
    )
    fig_cmp.update_yaxes(
        title_text="BTC (USDT)", secondary_y=True, color="#ff6b6b"
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Derniers prix
    col_l, col_r = st.columns(2)
    col_l.metric(f"Dernier prix {chosen_sym}", f"{df_alt['price'].iloc[-1]:,.4f} USDT")
    col_r.metric("Dernier prix BTC", f"{df_btc['price'].iloc[-1]:,.2f} USDT")


# ----------------------------------------------------------------------
# Affichage selon le menu sélectionné
# ----------------------------------------------------------------------
if selected_page == "📈 Valeur du portefeuille":
    show_portfolio_value(sim)
elif selected_page == "🔎 Analyse crypto":
    show_crypto_analysis(sim)
elif selected_page == "💼 Portefeuille (predictions)":
    show_portfolio_predictions(sim)
elif selected_page == "💡 Description de la stratégie":
    show_strategy_description()
elif selected_page == "🔔 Signaux (temps)":
    show_signals(sim)
elif selected_page == "📊 Comparaison prix BTC vs crypto low‑price":
    show_btc_vs_lowprice()
else:
    st.error("Page inconnue : sélectionnez une option du menu.")

# ----------------------------------------------------------------------
# Footer – mentions légales
# ----------------------------------------------------------------------
st.caption(
    """
    ⚠️ Les cours affichés proviennent du **endpoint public de Binance** (pour les
    pages restantes, uniquement via les fichiers parquet déjà collectés).
    📈 Cette application est destinée à un usage de **demo / recherche** et ne
    constitue en aucun cas un conseil d’investissement.
    """
)
