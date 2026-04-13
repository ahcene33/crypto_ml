# src/collect.py
# récupération d’historique OHLCV via Binance + ranking top‑N via CoinGecko

import logging
import time
from pathlib import Path

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from pycoingecko import CoinGeckoAPI
from config import cfg

# dossiers ------------------------------------------------------------
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)

# -----------------------------------------------------------------
def fetch_binance_ohlcv(symbol: str, days: int) -> pd.DataFrame:
    """
    retourne un DataFrame indexé par datetime contenant :
        price  – prix de clôture quotidien
        volume – volume quotidien (unités crypto)
    le symbole doit être sous la forme SYMBOLUSDT sur Binance.
    """
    client = Client()                         # public client, aucune clef API
    pair = f"{symbol.upper()}USDT"

    end_ts = int(time.time() * 1000)                     # maintenant (ms)
    start_ts = end_ts - days * 24 * 60 * 60 * 1000       # il y a `days` jours

    frames = []
    cur_start = start_ts
    while cur_start < end_ts:
        klines = client.get_klines(
            symbol=pair,
            interval=Client.KLINE_INTERVAL_1DAY,
            startTime=cur_start,
            limit=1000,
        )
        if not klines:
            break

        df_chunk = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_base_vol", "taker_quote_vol", "ignore"
        ])
        df_chunk["date"] = pd.to_datetime(df_chunk["open_time"], unit="ms")
        df_chunk.set_index("date", inplace=True)

        price = df_chunk["close"].astype(float)
        volume = df_chunk["volume"].astype(float)

        frames.append(pd.concat([price, volume], axis=1))

        # avancer d’un jour
        last_open = df_chunk["open_time"].iloc[-1]
        cur_start = last_open + 24 * 60 * 60 * 1000   # +1 jour (ms)

        time.sleep(0.05)   # respecter le rate‑limit

    if not frames:
        return pd.DataFrame(columns=["price", "volume"])

    df = pd.concat(frames)
    df.columns = ["price", "volume"]
    return df.sort_index()


# -----------------------------------------------------------------
def _save_parquet(df: pd.DataFrame, symbol: str):
    out_path = RAW_DIR / f"{symbol.upper()}.parquet"
    try:
        df.to_parquet(out_path, compression="gzip", engine="pyarrow")
        log.info(f"sauvegarde {out_path} ({len(df)} lignes)")
    except Exception as e:
        log.error(f"echec sauvegarde {symbol} : {e}")


# -----------------------------------------------------------------
STABLECOINS = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "USDS"}

def collect_top_n():
    """
    - récupère le top‑N via CoinGecko (ranking market‑cap)
    - ne télécharge que les symboles disposant d’une paire SYMBOLUSDT sur Binance
    - ignore les stablecoins
    """
    # 1️⃣ top‑N depuis CoinGecko
    cg = CoinGeckoAPI()
    top = cg.get_coins_markets(
        vs_currency="usd",
        order="market_cap_desc",
        per_page=cfg["coins"]["top_n"],
        page=1,
        sparkline=False,
    )

    # 2️⃣ paires USDT disponibles sur Binance
    client = Client()
    exchange_info = client.get_exchange_info()
    usdt_pairs = {
        sym["symbol"]
        for sym in exchange_info["symbols"]
        if sym["status"] == "TRADING" and sym["symbol"].endswith("USDT")
    }

    # 3️⃣ téléchargement
    for coin in top:
        symbol = coin["symbol"].upper()

        # ignore stablecoins
        if symbol in STABLECOINS:
            log.warning(f"{symbol} est un stablecoin → ignore")
            continue

        if f"{symbol}USDT" not in usdt_pairs:
            log.warning(f"{symbol} n’a pas de paire USDT sur Binance → ignore")
            continue

        log.info(f"collecte {symbol} via Binance")
        try:
            df = fetch_binance_ohlcv(symbol, days=cfg["coins"]["days_history"])
            if not df.empty:
                _save_parquet(df, symbol)
        except BinanceAPIException as e:
            log.error(f"echec collecte {symbol} (BinanceAPI) : {e}")
        except Exception as e:
            log.error(f"echec collecte {symbol} (autre) : {e}")

if __name__ == "__main__":
    collect_top_n()
