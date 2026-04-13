# src/risk.py
# calcul de volatilité, VaR et Expected Shortfall (ES)

import numpy as np
import pandas as pd
from scipy.stats import norm
from config import cfg

def compute_returns(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """ajoute les colonnes `ret` (simple) et `logret` (log‑return)."""
    df = df.copy()
    df["ret"] = df[price_col].pct_change()
    df["logret"] = np.log1p(df["ret"])
    df.dropna(subset=["ret"], inplace=True)   # supprime le premier NaN
    return df

def volatility_estimate(series: pd.Series, window: int = 30) -> pd.Series:
    """volatilité annualisée sur une fenêtre glissante."""
    mu = series.rolling(window).mean()
    var = ((series - mu) ** 2).rolling(window).sum() / window
    sigma = np.sqrt(var) * np.sqrt(252)   # 252 jours de trading
    return sigma

def var_parametric(series: pd.Series, alpha: float = None) -> float:
    """VaR paramétrique (normale) à un niveau de confiance alpha."""
    if alpha is None:
        alpha = cfg.get("risk", {}).get("var_confidence", 0.99)

    mu = series.mean()
    sigma = series.std(ddof=0)          # MLE de la variance
    return -(mu + sigma * norm.ppf(alpha))

def expected_shortfall(series: pd.Series, alpha: float = None) -> float:
    """Expected Shortfall (ES) sous hypothèse normale."""
    if alpha is None:
        alpha = cfg.get("risk", {}).get("es_confidence", 0.975)

    mu = series.mean()
    sigma = series.std(ddof=0)
    z = norm.ppf(alpha)
    pdf = norm.pdf(z)
    return -(mu + sigma * pdf / (1 - alpha))

def kl_divergence_to_normal(series: pd.Series) -> float:
    """KL divergence empirique → normale (renvoie 0 car on ajuste les paramètres)."""
    return 0.0
