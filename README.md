# MLOps: Sistem Prediksi Harga dan Volatilitas Bitcoin

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/rafeeq-s/MLOps-Bitcoin-Volatility-Prediction)

Repositori ini berisi fondasi teknis dan arsitektur MLOps untuk prediksi deret waktu (*time-series regression*) harga penutupan dan estimasi volatilitas Bitcoin (BTC) menggunakan data historis dari CoinGecko API.

---

## 1. Struktur Direktori
Struktur repositori mengadopsi konvensi industri berbasis standardisasi Data Science:

```text
├── .devcontainer/       # Konfigurasi otomatisasi lingkungan Codespaces
├── config/              # File konfigurasi parameter model dan pipeline
├── data/
│   ├── raw/             # Data mentah hasil ingestion API (dikelola via DVC)
│   └── processed/       # Fitur hasil transformasi dan feature engineering
├── models/              # Artefak model terlatih
├── notebooks/           # Jupyter Notebooks untuk eksplorasi (EDA & PoC)
├── src/                 # Source code modular
│   ├── data/            # Skrip penarikan dan ingestion data
│   ├── features/        # Skrip kalkulasi indikator teknikal & volatilitas
│   └── models/          # Logika pelatihan model dan evaluasi
├── tests/               # Unit testing skrip dan pipeline
├── requirements.txt     # Daftar dependensi pustaka Python
├── .gitignore           # File pengabaian Git (data besar/cache)
└── README.md            # Dokumentasi utama proyek

## Pipeline Data Otomatis (LK-04)

Pipeline ini dirancang untuk penarikan data dinamis BTC/USD berkala dari CoinGecko Public REST API guna mendukung deteksi drift dan pelatihan kontinu (Continual Learning).

### 1. Ingestion Data Mentah
Mengambil data pasar mentah terbaru (harga penutupan, volume transaksi 24 jam, dan market cap) secara periodik dan non-destruktif:
```bash
python src/ingest_data.py