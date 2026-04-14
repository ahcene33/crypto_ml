# src/portfolio.py
"""
High‑level façade utilisée par l’application Streamlit.
- charge le modèle / les noms de features **une fois** (cache module‑level)
- fournit `predict_all(date)` et `run_simulation(...)`
- fonction `get_live_prices()` qui interroge Binance en temps réel.
"""

import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from features import add_basic_features, select_features
from prediction_utils import probability_to_category
from portfolio_manager import Portfolio
from config import cfg

# Nouveau helper d’appel à Binance (prices live)
from binance_price import fetch_latest_prices

def get_live_prices() -> pd.DataFrame:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    symbols = [fp.stem.upper() for fp in raw_dir.glob("*.parquet")]
    price_dict = fetch_latest_prices(symbols)

    df = pd.DataFrame(list(price_dict.items()), columns=["symbol", "price"])
    return df.sort_values("symbol").reset_index(drop=True)

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
#   Cache du modèle & des features (chargement à la première importation)
# ----------------------------------------------------------------------
_MODEL, _FEATURES = None, None


def _load_artifacts():
    """Lecture unique du modèle + de la liste de features."""
    global _MODEL, _FEATURES
    if _MODEL is not None and _FEATURES is not None:
        return _MODEL, _FEATURES

    root = Path(__file__).resolve().parents[1]          # crypto_ml/
    model_path = root / "models" / "model.pkl"
    feats_path = root / "data" / "processed" / "features.parquet"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found → {model_path}")
    if not feats_path.is_file():
        raise FileNotFoundError(f"Features not found → {feats_path}")

    _MODEL = joblib.load(model_path)
    _FEATURES = list(pd.read_parquet(feats_path, columns=None).columns)
    log.info("Modèle & features chargés (cache global)")
    return _MODEL, _FEATURES


def predict_all(date_ref: pd.Timestamp) -> pd.DataFrame:
    """
    Prédiction de **toutes** les cryptos disponibles au `date_ref`.
    Retourne : symbol, price, probability, signal, score, category.
    """
    model, feature_names = _load_artifacts()
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"

    rows = []
    for fp in raw_dir.glob("*.parquet"):
        symbol = fp.stem.upper()
        df = pd.read_parquet(fp)

        # garder les lignes ≤ date_ref
        df = df.loc[df.index <= date_ref]
        if df.empty:
            continue

        df = add_basic_features(df)
        last = df.iloc[-1]

        row = {"symbol": symbol, "price": float(last["price"])}
        for f in feature_names:
            row[f] = last.get(f, pd.NA)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"Aucune donnée crypto disponible à la date {date_ref}")

    pred = pd.DataFrame(rows)

    # -----------------------------------------------------------------
    #  Fill‑na → on force le down‑casting maintenant (évite le FutureWarning)
    # -----------------------------------------------------------------
    X = (
        pred[feature_names]
        .fillna(0)
        .infer_objects(copy=False)      # <-- important
        .values
    )
    prob = model.predict(X)                # LightGBM renvoie déjà des probas
    pred["probability"] = prob
    threshold = cfg.get("training", {}).get("threshold", 0.5)
    pred["signal"] = (prob > threshold).astype(int)

    # score = prob × (1 - risk)
    if "std_30" in pred.columns:
        risk = (pred["std_30"] / pred["price"]).fillna(0).clip(0, 1)
    else:
        risk = 0.0
    pred["score"] = pred["probability"] * (1 - risk)

    pred["category"] = pred["probability"].apply(probability_to_category)

    return pred[["symbol", "price", "probability", "signal", "score", "category"]]


def get_live_prices() -> pd.DataFrame:
    """
    Retourne les cotations “live” pour toutes les cryptos présentes
    dans data/raw/*.parquet via Binance (endpoint public).
    """
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    symbols = [fp.stem.upper() for fp in raw_dir.glob("*.parquet")]
    price_dict = fetch_latest_prices(symbols)

    df = pd.DataFrame(list(price_dict.items()), columns=["symbol", "price"])
    df = df.sort_values("symbol").reset_index(drop=True)
    return df


def run_simulation(
    start_date: date = date(2026, 1, 22),
    end_date: date | None = None,
    initial_capital: float = 200.0,
    monthly_deposit: float = 50.0,
) -> dict:
    """Boucle de simulation : mise à jour quotidienne, rebalance le 10 , KPI."""
    if end_date is None:
        end_date = datetime.today().date()

    # 1️⃣  Portefeuille
    portfolio = Portfolio(
        initial_capital=initial_capital,
        monthly_amount=monthly_deposit,
        investment_day=10,
        start_date=datetime.combine(start_date, datetime.min.time()),
    )

    # 2️⃣  Chargement de toutes les séries historiques en mémoire
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    price_series: Dict[str, pd.Series] = {}
    for fp in raw_dir.glob("*.parquet"):
        sym = fp.stem.upper()
        df = pd.read_parquet(fp)               # index = datetime
        price_series[sym] = df["price"]

    # 3️⃣  Boucle jour‑par‑jour
    cur = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())

    while cur <= end_dt:
        # a) dépôt mensuel (le 10)
        portfolio.add_monthly_deposit(cur)

        # b) prix du jour (dernier prix ≤ cur pour chaque crypto)
        day_prices = {
            sym: ser.loc[ser.index <= cur].iloc[-1]
            for sym, ser in price_series.items()
            if not ser.loc[ser.index <= cur].empty
        }

        # c) mise à jour des positions (valorisation)
        portfolio.update_prices(cur, day_prices)

        # d) prédiction du jour (si on a des prix)
        if day_prices:
            df_pred = predict_all(pd.Timestamp(cur))
            # ne garder que les cryptos dont le prix du jour est connu
            df_pred = df_pred[df_pred["symbol"].isin(day_prices.keys())]
            portfolio.rebalance(cur, df_pred)   # rebalance uniquement le jour 10

        cur += timedelta(days=1)

    # 4️⃣  Résultats
    history = pd.DataFrame(portfolio.history)
    metrics = portfolio.metrics()

    return {
        "portfolio_history": history,
        "metrics": metrics,
        "last_predictions": df_pred if "df_pred" in locals() else pd.DataFrame(),
    }
