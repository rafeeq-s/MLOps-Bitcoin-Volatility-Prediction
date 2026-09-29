"""
src/preprocess.py
Modul prapemrosesan data mentah BTC/USD untuk persiapan rekayasa fitur & continual learning.
"""
from datetime import datetime
import glob
import logging
from pathlib import Path
import sys
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_latest_raw_file() -> Path:
    """Mencari file data mentah terbaru di folder data/raw/."""
    raw_files = sorted(RAW_DATA_DIR.glob("btc_raw_*.csv"), key=lambda p: p.stat().st_mtime)
    if not raw_files:
        # Fallback jika ada file template bawaan repo
        fallback = RAW_DATA_DIR / "bitcoin_raw_market_data.csv"
        if fallback.exists():
            return fallback
        raise FileNotFoundError("Tidak ditemukan file CSV mentah di folder data/raw/")
    return raw_files[-1]


def clean_and_preprocess(raw_path: Path) -> pd.DataFrame:
    """Membersihkan duplikasi, memvalidasi missing values, dan casting tipe data."""
    logging.info("Memproses file mentah: %s", raw_path)
    df = pd.read_csv(raw_path)

    # Memastikan kolom timestamp terformat benar
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "open_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")

    # Kolom numerik utama yang digunakan untuk prediksi target & drift test
    feature_cols = ["open", "high", "low", "close", "volume"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Pembersihan missing values dan baris duplikat
    df = df.dropna(subset=feature_cols)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Simpan subset kolom yang relevan
    clean_df = df[["timestamp"] + feature_cols]
    return clean_df


def save_processed_data(df: pd.DataFrame) -> Path:
    """Menyimpan data hasil pembersihan ke data/processed/."""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROCESSED_DATA_DIR / f"btc_clean_{current_time}.parquet"
    # Menyimpan dalam format parquet untuk integritas tipe data deret waktu
    df.to_parquet(out_path, index=False)
    # Menyimpan juga versi referensi statis untuk pipeline model
    df.to_parquet(PROCESSED_DATA_DIR / "btc_clean_latest.parquet", index=False)
    logging.info("Data bersih berhasil disimpan di: %s", out_path)
    return out_path


if __name__ == "__main__":
    try:
        latest_file = get_latest_raw_file()
        processed_df = clean_and_preprocess(latest_file)
        save_processed_data(processed_df)
    except Exception as err:
        logging.critical("Pipeline Prapemrosesan gagal: %s", err)
        sys.exit(1)