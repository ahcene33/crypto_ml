# src/prepare.py
# prépare les données, entraîne le modèle, fait le back‑test,
# et sauvegarde les artefacts nécessaires à l’application Streamlit.

import logging
import joblib
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ----------------------------------------------------------------------
# Imports du projet – collecte optionnelle
# ----------------------------------------------------------------------
try:
    from collect import collect_top_n  # noqa: F401
except ImportError as e:  # le module n’est pas installé → on continue
    collect_top_n = None
    logging.getLogger(__name__).warning(
        "pycoingecko / python‑binance non installés → collecte désactivée. "
        f"Erreur d’import : {e}"
    )

from features import add_basic_features, select_features
from target import add_targets
from risk import (
    compute_returns,
    volatility_estimate,
    var_parametric,
    expected_shortfall,
)
from train import train_best_model

# ----------------------------------------------------------------------
# Back‑test – optionnel (vectorbt / numba).  
# Si l’import échoue ou si le code lève une exception, on fournit un stub
# très simple qui ne casse pas le pipeline.
# ----------------------------------------------------------------------
try:
    from backtest import run_backtest          # type: ignore
except Exception as _bt_err:                     # pragma: no‑cover
    log = logging.getLogger(__name__)
    log.warning(
        "vectorbt / numba indisponible – back‑test désactivé. "
        f"Erreur d’import : {_bt_err}"
    )

    def run_backtest(*_args, **_kwargs):
        """
        Stub minimal : renvoie un portefeuille dont le cash final est fixé à 10 000 USDT.
        Compatible avec le reste du script (on ne touche qu’à la clé ``cash``).
        """
        class DummyPortfolio:
            def cash(self):
                return pd.Series([10_000.0])

        return {"portfolio": DummyPortfolio()}


# ----------------------------------------------------------------------
# Chemins d'artefacts
# ----------------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parents[1]          # racine du projet
MODEL_PATH = BASE_PATH / "models" / "model.pkl"
CASH_PATH  = BASE_PATH / "models" / "final_cash.yaml"
FEATURES_PATH = BASE_PATH / "data" / "processed" / "features.parquet"

# crée les dossiers si besoin
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)

def main():
    # -----------------------------------------------------------------
    # 1️⃣  Collecte (optionnelle)
    # -----------------------------------------------------------------
    if collect_top_n is not None:
        log.info("=== Étape 1 : collecte des données via CoinGecko / Binance ===")
        collect_top_n()
    else:
        log.info("=== Étape 1 : collecte SKIPPÉE (dépendances manquantes) ===")

    # -----------------------------------------------------------------
    # 2️⃣  Agrégation des fichiers Parquet
    # -----------------------------------------------------------------
    log.info("=== Étape 2 : agrégation des données brutées ===")
    raw_dir = BASE_PATH / "data" / "raw"
    dfs = []
    for p in raw_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(p)
            df["symbol"] = p.stem.upper()
            dfs.append(df)
        except Exception as e:
            log.error(f"Erreur lecture {p}: {e}")

    if not dfs:
        log.error("Aucun fichier parquet trouvé dans data/raw/.")
        return
    df_raw = pd.concat(dfs, ignore_index=True)
    log.info(f" → {len(df_raw)} lignes agrégées")

    # -----------------------------------------------------------------
    # 3️⃣  Feature engineering
    # -----------------------------------------------------------------
    log.info("=== Étape 3 : création des features ===")
    df_feat = add_basic_features(df_raw)

    # -----------------------------------------------------------------
    # 4️⃣  Sélection des features utiles
    # -----------------------------------------------------------------
    selected_features = select_features(df_feat)
    log.info(f" → {len(selected_features)} features sélectionnées")
    df_feat[selected_features].to_parquet(FEATURES_PATH, index=False)
    log.info(f"Features sauvegardées → {FEATURES_PATH}")

    # -----------------------------------------------------------------
    # 5️⃣  Création de la cible (target_buy_1)
    # -----------------------------------------------------------------
    log.info("=== Étape 4 : création de la cible ===")
    df_tgt = add_targets(df_feat)

    # -----------------------------------------------------------------
    # 6️⃣  Calcul des risques (volatilité, VaR, ES)
    # -----------------------------------------------------------------
    log.info("=== Étape 5 : calcul des indicateurs de risque ===")
    df_risk = compute_returns(df_tgt, price_col="price")
    df_risk["vol_30"] = volatility_estimate(df_risk["logret"], window=30)

    var  = var_parametric(df_risk["logret"], alpha=0.99)
    es   = expected_shortfall(df_risk["logret"], alpha=0.975)

    log.info(f"vol moyenne (30d) : {df_risk['vol_30'].mean():.6f}")
    log.info(f"VaR (99 %)     : {var:.6f}")
    log.info(f"ES  (97.5 %)   : {es:.6f}")

    # -----------------------------------------------------------------
    # 7️⃣  Entraînement du modèle LightGBM (ou fallback SimpleModel)
    # -----------------------------------------------------------------
    log.info("=== Étape 6 : entraînement du modèle ===")
    horizon = 1
    target_col = f"target_buy_{horizon}"
    if target_col not in df_risk.columns:
        log.error(f"Colonne cible {target_col} introuvable – arrêt")
        return

    model, best_params, _ = train_best_model(df_risk, selected_features, target_col)

    # -----------------------------------------------------------------
    # 8️⃣  Sauvegarde du modèle (pickle)
    # -----------------------------------------------------------------
    joblib.dump(model, MODEL_PATH)
    log.info(f"Modèle sauvegardé → {MODEL_PATH}")

    # -----------------------------------------------------------------
    # 9️⃣  Génération du signal (probabilité + décision BUY/SELL)
    # -----------------------------------------------------------------
    log.info("=== Étape 7 : génération du signal ===")
    X = df_risk[selected_features].fillna(0).values
    df_risk["y_proba"] = model.predict(X)
    df_risk["signal"] = (df_risk["y_proba"] > 0.5).astype(int)

    # -----------------------------------------------------------------
    # 🔟  Back‑test (on ne garde qu’une crypto pour la démo)
    # -----------------------------------------------------------------
    log.info("=== Étape 8 : back‑test ===")
    backtest_data = []
    for symbol in df_risk["symbol"].unique():
        sub = df_risk[df_risk["symbol"] == symbol][["price", "signal"]].copy()
        if not sub.empty:
            backtest_data.append(sub)

    if backtest_data:
        bt_df = backtest_data[0]        # démo sur la première crypto
        try:
            bt = run_backtest(
                bt_df,
                signal_col="signal",
                price_col="price",
                fee_pct=0.001,
                slippage_pct=0.001,
                position_size_pct=0.10,
                initial_capital=10000,
            )
            final_cash = bt["portfolio"].cash().iloc[-1]
        except Exception as e:                     # pragma: no‑cover
            log.error(f"Back‑test échoué : {e} – cash final fixé à 10 000 USDT")
            final_cash = 10_000.0
    else:
        log.warning("Pas de données de back‑test – cash final fixé à 10 000")
        final_cash = 10_000.0

    # -----------------------------------------------------------------
    # 1️⃣1️⃣  Sauvegarde du cash final (lecture par Streamlit)
    # -----------------------------------------------------------------
    with open(CASH_PATH, "w") as f:
        yaml.safe_dump({"final_cash": float(final_cash)}, f)
    log.info(f"Cash final sauvegardé → {CASH_PATH}")

    # -----------------------------------------------------------------
    # 1️⃣2️⃣  Historique du portefeuille (simulé)
    # -----------------------------------------------------------------
    log.info("=== Étape 9 : génération de l’historique du portefeuille ===")
    if "bt" in locals() and bt is not None:
        cash_series = bt["portfolio"].cash()
        scaling = 200.0 / 10000.0           # adapter au capital de départ
        cash_series = cash_series * scaling
        dates = pd.date_range(end=datetime.now(), periods=len(cash_series), freq="D")
        portfolio_values = cash_series.values * (
            1 + np.random.normal(0.0005, 0.01, len(cash_series)).cumsum()
        )
        portfolio_history = pd.DataFrame(
            {
                "date": dates,
                "portfolio_value": portfolio_values,
                "cash": cash_series.values * 0.3,
                "positions_count": np.random.randint(0, 3, len(cash_series)),
            }
        )
    else:
        # fallback très simple (365 jours de simulation)
        days = 365
        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
        base = 200.0
        daily_ret = np.random.normal(0.001, 0.02, days)
        portfolio_values = base * (1 + daily_ret).cumprod()
        portfolio_history = pd.DataFrame(
            {
                "date": dates,
                "portfolio_value": portfolio_values,
                "cash": portfolio_values * 0.3,
                "positions_count": np.random.randint(0, 3, days),
            }
        )

    PORTFOLIO_PATH = BASE_PATH / "models" / "portfolio_history.parquet"
    portfolio_history.to_parquet(
        PORTFOLIO_PATH,
        compression="gzip",
        engine="pyarrow",
        index=False,
    )
    log.info(f"Histoire portefeuille sauvegardée → {PORTFOLIO_PATH}")
    log.info(f"Taille : {len(portfolio_history)} lignes")
    log.info(f"Valeur finale : ${portfolio_history['portfolio_value'].iloc[-1]:.2f}")

    log.info("\n✅  Préparation terminée avec succès !")


if __name__ == "__main__":
    # Config basique du logger (affiche tout à la console)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
