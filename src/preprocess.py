"""
src/preprocess.py
Modul prapemrosesan data mentah Bitcoin (Transform Tahap 1 sesuai LK-03).
"""
from datetime import datetime
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
    """Mencari file mentah btc_raw terbaru."""
    raw_files = sorted(RAW_DATA_DIR.glob("btc_raw_*.csv"), key=lambda p: p.stat().st_mtime)
    if raw_files:
        return raw_files[-1]
    
    fallback = RAW_DATA_DIR / "btc_raw.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Tidak ditemukan file data mentah di data/raw/")


def clean_and_preprocess(raw_path: Path) -> pd.DataFrame:
    """
    Pembersihan Data sesuai LK-03:
    1. Konversi format timestamp ke UTC datetime
    2. Deduplikasi baris berdasarkan timestamp
    3. Imputasi missing values dengan Forward Fill (ffill) untuk data deret waktu
    """
    logging.info("Membaca dan memproses file: %s", raw_path)
    df = pd.read_csv(raw_path)

    # 1. Konversi Datetime standar ISO UTC
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # 2. Deduplikasi baris berdasarkan keunikan waktu
    initial_len = len(df)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    logging.info("Deduplikasi: %d baris menjadi %d baris", initial_len, len(df))

    # 3. Penanganan Missing Values dengan Forward Fill (ffill) sesuai LK-03
    numeric_cols = ["price", "volume_24h", "market_cap"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[numeric_cols] = df[numeric_cols].ffill()

    # Hapus baris awal jika masih ada NaN yang tidak ter-cover ffill
    df = df.dropna(subset=numeric_cols)
    return df


def save_processed_data(df: pd.DataFrame) -> Path:
    """Menyimpan data hasil prapemrosesan ke data/processed/."""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROCESSED_DATA_DIR / f"btc_clean_{current_time}.csv"
    
    df.to_csv(out_path, index=False)
    # Menyimpan pointer versi terbaru
    df.to_csv(PROCESSED_DATA_DIR / "btc_clean_latest.csv", index=False)
    
    logging.info("Data bersih berhasil disimpan di: %s", out_path)
    return out_path


if __name__ == "__main__":
    try:
        latest_file = get_latest_raw_file()
        clean_df = clean_and_preprocess(latest_file)
        save_processed_data(clean_df)
    except Exception as exc:
        logging.critical("Prapemrosesan gagal: %s", exc)
        sys.exit(1)