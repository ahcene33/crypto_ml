# src/binance_price.py
"""
Fonctions très légères pour récupérer les cotations “live” de Binance.

- Utilise uniquement le endpoint public https://api.binance.com/api/v3/ticker/price
- Pas de client API lourd, pas de clé d’API.
- Retry / timeout incorporés → robuste même en cas de coupure temporaire.
- Ne charge jamais plus de quelques ko en mémoire.
"""

import logging
import time
from typing import List, Dict

import requests

log = logging.getLogger(__name__)

_BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"
_MAX_RETRIES = 3          # nombre maximal de tentatives
_BASE_SLEEP = 0.1         # s – augmente légèrement à chaque essai


def _fetch_one(symbol: str) -> float | None:
    """Retourne le prix USDT du symbole (ex. BTC → BTCUSDT)."""
    pair = f"{symbol.upper()}USDT"
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(
                _BINANCE_TICKER_URL,
                params={"symbol": pair},
                timeout=5,
            )
            resp.raise_for_status()
            price = float(resp.json()["price"])
            return price
        except Exception as exc:                     # pragma: no‑cover
            log.warning(
                f"[{pair}] tentative {attempt}/{_MAX_RETRIES} – {exc}"
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_BASE_SLEEP * attempt)
    return None


def fetch_latest_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Retourne un dictionnaire {symbol.upper(): price} contenant le cours USDT
    le plus récent pour chaque symbole fourni.
    L’appel est séquentiel (un symbole à la fois) afin de garder l’empreinte
    mémoire nulle et de respecter les limites de débit de Binance.
    """
    prices: Dict[str, float] = {}
    for sym in symbols:
        price = _fetch_one(sym)
        if price is not None:
            prices[sym.upper()] = price
        else:
            log.error(f"Impossible de récupérer le prix de {sym} après {_MAX_RETRIES} essais.")
    return prices
