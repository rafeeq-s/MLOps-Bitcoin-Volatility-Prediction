"""
src/ingest_data.py
Modul penarikan data dinamis Bitcoin dari CoinGecko Public REST API sesuai LK-03.
"""
from datetime import datetime
import logging
from pathlib import Path
import sys
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_crypto_raw_data(coin_id: str = "bitcoin", vs_currency: str = "usd", days: int = 180) -> pd.DataFrame:
    """
    Mengambil data mentah (Price, Volume, Market Cap) dari CoinGecko Public API
    sesuai spesifikasi LK-03.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": "daily",
    }
    headers = {"accept": "application/json"}

    logging.info("Mengambil data dari CoinGecko (%s, %s hari)...", coin_id, days)
    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as err:
        logging.error("Gagal menarik data CoinGecko: %s", err)
        raise

    # Ekstraksi komponen data
    df_price = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df_vol = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume_24h"])
    df_mc = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])

    # Menggabungkan berdasarkan timestamp
    df = df_price.merge(df_vol, on="timestamp").merge(df_mc, on="timestamp")

    # Konversi timestamp epoch (ms) ke representasi tanggal terbaca
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date

    # Urutkan kolom sesuai skema LK-03
    df = df[["timestamp", "datetime", "price", "volume_24h", "market_cap"]]
    return df


def save_raw_data(df: pd.DataFrame) -> Path:
    """Menyimpan data mentah dengan timestamp untuk mendukung simulasi periodik non-destruktif."""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = RAW_DATA_DIR / f"btc_raw_{current_time}.csv"
    
    df.to_csv(file_path, index=False)
    # Menyimpan salinan btc_raw.csv default sesuai LK-03
    df.to_csv(RAW_DATA_DIR / "btc_raw.csv", index=False)
    
    logging.info("Data mentah berhasil disimpan di: %s", file_path)
    return file_path


if __name__ == "__main__":
    try:
        raw_df = get_crypto_raw_data()
        save_raw_data(raw_df)
    except Exception as exc:
        logging.critical("Pipeline Ingestion gagal: %exc", exc)
        sys.exit(1)