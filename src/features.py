# src/features.py
# fonctions d’ingénierie de features pour les données crypto
# on utilise la loi des grands nombres : plus on a d'observations,
# plus la moyenne empirique (retour moyen) converge vers la vraie moyenne.

import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """calcul du RSI (relative strength index) sur la série de prix."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # moyenne mobile simple des gains / pertes (MLE de l’espérance)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-9)           # évite division par zéro
    return 100 - (100 / (1 + rs))


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df doit contenir au minimum les colonnes : price, volume
    la fonction ajoute de nombreuses colonnes de features et renvoie le df enrichi.
    """
    df = df.copy()

    # -----------------------------------------------------------------
    # Garantir que la date soit bien une colonne (certaines dataframes
    # sont indexées par datetime)
    # -----------------------------------------------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' not in df.columns:
            df = df.reset_index()
        else:
            df = df.set_index('date')
    else:
        df = df.reset_index().rename(columns={'index': 'date'})

    # -----------------------------------------------------------------
    # rendements (retours simples) – estimateur de moyenne (MLE, LLN)
    # -----------------------------------------------------------------
    df["ret_1"] = df["price"].pct_change(1)
    df["logret_1"] = np.log1p(df["ret_1"])

    # rendements multi‑horizon (pour capturer différentes échelles)
    for h in [3, 7, 14, 30]:
        df[f"ret_{h}"] = df["price"].pct_change(h)
        df[f"logret_{h}"] = np.log1p(df[f"ret_{h}"])

    # -----------------------------------------------------------------
    # rolling statistics : moyenne, écart‑type, skew, kurtosis
    # -----------------------------------------------------------------
    for w in [7, 14, 30, 60]:
        df[f"ma_{w}"] = df["price"].rolling(w).mean()
        df[f"std_{w}"] = df["price"].rolling(w).std()
        df[f"skew_{w}"] = df["price"].rolling(w).skew()
        df[f"kurt_{w}"] = df["price"].rolling(w).kurt()

    # -----------------------------------------------------------------
    # momentum simple (price / price d’il y a w jours - 1)
    # -----------------------------------------------------------------
    df["mom_7"] = df["price"] / df["price"].shift(7) - 1

    # -----------------------------------------------------------------
    # RSI (indice de force relative)
    # -----------------------------------------------------------------
    df["rsi_14"] = _rsi(df["price"], window=14)

    # -----------------------------------------------------------------
    # Bollinger Bands (ma_20 +/- 2*std_20) – mesure de volatilité
    # -----------------------------------------------------------------
    df["bb_mid_20"] = df["price"].rolling(20).mean()
    df["bb_std_20"] = df["price"].rolling(20).std()
    df["bb_upper_20"] = df["bb_mid_20"] + 2 * df["bb_std_20"]
    df["bb_lower_20"] = df["bb_mid_20"] - 2 * df["bb_std_20"]

    # -----------------------------------------------------------------
    # nettoyage des inf / NaN (ex. division par zéro)
    # -----------------------------------------------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def select_features(df: pd.DataFrame) -> list:
    """
    Retourne la liste des colonnes utilisables comme features.
    - on exclut les colonnes de base price/volume,
    - on retire les colonnes fortement corrélées (>0.95) pour éviter la redondance.
    - **sécurité** : si la suppression laisserait le jeu vide,
      on retourne la liste initiale (au moins une feature disponible).
    """
    exclude = {"price", "volume"}
    # colonnes numériques non exclues
    feat = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feat:
        log.warning("select_features : aucune colonne numérique détectée – retourne tout.")
        return feat

    # Corrélation absolue, NaN remplacés par 0
    corr = df[feat].corr().abs().fillna(0.0)

    to_drop = set()
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            if corr.iloc[i, j] > 0.95:
                to_drop.add(corr.columns[j])

    selected = set(feat) - to_drop

    if not selected:
        log.warning(
            "select_features : toutes les colonnes ont été éliminées par corrélation – "
            "on conserve la liste complète des features disponibles."
        )
        selected = set(feat)

    return sorted(list(selected))

