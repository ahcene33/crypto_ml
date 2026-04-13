# src/backtest.py
# Back‑test simple avec vectorbt (utilisé uniquement par src.prepare)

import numpy as np
import pandas as pd
import vectorbt as vbt
from config import cfg   # uniquement pour le logger éventuel (pas utilisé ici)


def simulate_portfolio(days: int = 365):
    """Fonction de démonstration – génère un historique simulé."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D")
    values = 200.0 * (1 + np.random.normal(0.001, 0.02, days).cumsum())
    return pd.DataFrame({"date": dates, "portfolio_value": values, "cash": values * 0.2})


def _compute_portfolio_stats(portfolio: vbt.Portfolio) -> pd.DataFrame:
    """
    Calcule les statistiques du portefeuille et ajoute le Sharpe Ratio
    s’il n’est pas déjà présent (le résultat peut être une Series ou
    un DataFrame, on gère les deux cas).
    """
    stats = portfolio.stats(settings=dict(freq="D"))

    # ----- Sharpe Ratio -------------------------------------------------
    #  - Si `stats` est une Series → on regarde l’index
    #  - Si c’est un DataFrame → on regarde les colonnes
    if isinstance(stats, pd.Series):
        missing_sharpe = "Sharpe Ratio" not in stats.index
    else:  # DataFrame
        missing_sharpe = "Sharpe Ratio" not in stats.columns

    if missing_sharpe:
        daily_ret = portfolio.returns()
        if daily_ret.std() > 0:
            sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365)
        else:
            sharpe = np.nan

        # Ajout du Sharpe Ratio au bon format
        if isinstance(stats, pd.Series):
            stats["Sharpe Ratio"] = sharpe
        else:
            stats["Sharpe Ratio"] = sharpe

    return stats


def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "price",
    fee_pct: float = 0.001,
    slippage_pct: float = 0.001,
    position_size_pct: float = 0.10,
    initial_capital: float = 10000.0,
):
    """
    Exécute le back‑test vectorbt sur un DataFrame contenant les colonnes
    ``price`` et ``signal`` (0 = SELL, 1 = BUY).

    Retourne un dictionnaire contenant :
        - le portefeuille vectorbt,
        - les statistiques (avec Sharpe Ratio),
        - la courbe de rendement cumulée,
        - le tableau des trades.
    """
    price = df[price_col]
    signal = df[signal_col]

    # Taille de chaque ordre = position_size_pct × signal (0 ou 1)
    size = position_size_pct * signal

    # NOTE : on ne force plus l’argument « direction ». La valeur par défaut
    # de vectorbt est « both » et accepte correctement les positions longues.
    portfolio = vbt.Portfolio.from_orders(
        price,
        size,
        fees=fee_pct,
        slippage=slippage_pct,
        init_cash=initial_capital,
        log=False,
    )

    stats = _compute_portfolio_stats(portfolio)

    return {
        "portfolio": portfolio,
        "stats": stats,
        "equity_curve": portfolio.total_return(),
        "trades": portfolio.trades.records_readable,
    }
