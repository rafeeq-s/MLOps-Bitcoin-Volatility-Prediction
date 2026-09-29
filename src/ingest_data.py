"""
src/ingest_data.py
Modul untuk penarikan data dinamis BTC/USD secara berkala.
"""
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import pandas as pd
import requests

# Konfigurasi Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_bitcoin_data(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 100) -> pd.DataFrame:
    """Mengambil data OHLCV Bitcoin dari Binance Public API."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    logging.info("Mengambil data dari %s dengan parameter %s...", url, params)
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logging.error("Gagal menarik data dari API: %s", exc)
        raise

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(data, columns=columns)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def save_raw_data(df: pd.DataFrame) -> Path:
    """Menyimpan data mentah dengan penamaan file berbasis timestamp (non-destruktif)."""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = RAW_DATA_DIR / f"btc_raw_{current_time}.csv"
    df.to_csv(file_path, index=False)
    logging.info("Data mentah berhasil disimpan di: %s", file_path)
    return file_path


if __name__ == "__main__":
    try:
        raw_df = fetch_bitcoin_data()
        save_raw_data(raw_df)
    except Exception as err:
        logging.critical("Pipeline Ingestion gagal: %s", err)
        sys.exit(1)