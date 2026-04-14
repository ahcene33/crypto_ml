# src/binance_price.py
"""
Utility to fetch the latest USDT‑denominated prices from Binance.

- Uses the public endpoint https://api.binance.com/api/v3/ticker/price
  which returns *all* ticker pairs in a single request.
- Includes retry logic with exponential back‑off and basic rate‑limit handling.
- Returns a dict `{symbol.upper(): price}` for the symbols supplied by the caller.
- If the request ultimately fails, a deterministic fallback (price = 0.0) is
  returned so the Streamlit “Live Prices” tab can still display a table.
"""

import logging
import time
import random
import functools
from typing import List, Dict

import requests

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Binance public endpoint – returns a list like:
#   [{"symbol":"BTCUSDT","price":"27345.23"}, {"symbol":"ETHUSDT","price":"1820.12"}, …]
# ----------------------------------------------------------------------
_BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"
_MAX_RETRIES = 3                # how many times we repeat the whole request
_BASE_SLEEP = 0.2               # seconds – multiplied by attempt number


@functools.lru_cache(maxsize=1)
def _fetch_all_tickers() -> List[Dict[str, str]]:
    """
    Retrieve the complete list of ticker objects from Binance.
    Retries up to `_MAX_RETRIES` times with a short back‑off.
    Returns an empty list on permanent failure (caller will handle fallback).
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            # Binance does not require an API key for this public endpoint,
            # but adding a User‑Agent improves reliability.
            headers = {
                "User-Agent": "crypto‑ml/1.0 (https://github.com/your-repo)"
            }
            response = requests.get(_BINANCE_TICKER_URL, timeout=5, headers=headers)
            response.raise_for_status()
            return response.json()  # list of dicts {"symbol": "...", "price": "..."}
        except requests.exceptions.HTTPError as http_err:
            # 429 = Too Many Requests → respect Retry‑After if present
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else _BASE_SLEEP * attempt
                log.warning(f"Rate limit hit, sleeping {delay:.2f}s")
                time.sleep(delay + random.uniform(0, 0.1))
            else:
                log.warning(
                    f"[Binance ticker] HTTP error (attempt {attempt}/{_MAX_RETRIES}) – {http_err}"
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_BASE_SLEEP * attempt + random.uniform(0, 0.1))
        except Exception as exc:
            log.warning(
                f"[Binance ticker] attempt {attempt}/{_MAX_RETRIES} – {exc}"
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_BASE_SLEEP * attempt + random.uniform(0, 0.1))

    # All retries exhausted – log once more and give up
    log.error("Unable to fetch Binance ticker list after multiple attempts.")
    return []


def fetch_latest_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Return a mapping `{symbol.upper(): price}` for the subset of `symbols`
    supplied by the caller.

    The function:
      1. Downloads the complete Binance ticker list once.
      2. Builds an internal map `base_symbol -> price` (only USDT pairs are kept).
      3. Looks up each requested `symbol`; if a price is missing we fall back to `0.0`
         (so the UI never ends up with an empty dataframe).

    Parameters
    ----------
    symbols: List[str]
        Crypto symbols (e.g. ["BTC", "ETH", "DOGE"]) for which a USDT price is wanted.

    Returns
    -------
    Dict[str, float]
        Mapping of the requested symbols to their latest USDT price.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Fetch the full ticker payload once
    # ------------------------------------------------------------------
    all_tickers = _fetch_all_tickers()

    # ------------------------------------------------------------------
    # 2️⃣  Build a fast lookup: base symbol (without trailing "USDT") → price
    # ------------------------------------------------------------------
    price_lookup: Dict[str, float] = {}
    for entry in all_tickers:
        sym = entry.get("symbol", "")
        if sym.endswith("USDT"):
            base = sym[:-4]                     # strip the trailing "USDT"
            try:
                price_lookup[base.upper()] = float(entry["price"])
            except Exception:                 # pragma: no‑cover
                continue

    # ------------------------------------------------------------------
    # 3️⃣  Assemble the result for the symbols requested by the app
    # ------------------------------------------------------------------
    result: Dict[str, float] = {}
    for sym in symbols:
        price = price_lookup.get(sym.upper())
        if price is not None:
            result[sym.upper()] = price
        else:
            # Missing price → use a deterministic fallback (0.0) and log it
            log.warning(
                f"Binance price for '{sym}' not found in ticker list – using 0.0 as fallback."
            )
            result[sym.upper()] = 0.0

    return result
