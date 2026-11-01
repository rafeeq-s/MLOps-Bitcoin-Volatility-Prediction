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
```

## 2. Pipeline Data Otomatis

Pipeline ini dirancang untuk penarikan data dinamis BTC/USD berkala dari CoinGecko Public REST API guna mendukung deteksi drift dan pelatihan kontinu (Continual Learning).

### 2.1 Ingestion Data Mentah
Mengambil data pasar mentah terbaru (harga penutupan, volume transaksi 24 jam, dan market cap) secara periodik dan non-destruktif:
```bash
python src/ingest_data.py
```
### 2.2 Prapemrosesan Data (Data Preprocessing)
Setelah data mentah terkumpul, skrip pemrosesan dijalankan secara sistematis (src/preprocess.py) untuk memastikan kualitas data siap pakai pada tahap pemodelan:
Setelah data mentah terkumpul, skrip pemrosesan dijalankan secara sistematis (src/preprocess.py) untuk memastikan kualitas data siap pakai pada tahap pemodelan:

  1. Deduplikasi Baris: Menghilangkan data duplikat berdasarkan pencatatan waktu (timestamp).
  2. Standardisasi Waktu: Mengubah format waktu ke standar UTC (datetime64[ns, UTC]).
  3. Penanganan Missing Values: Menerapkan metode Forward Fill (ffill) guna mencegah terjadinya look-ahead bias atau data leakage pada analisis deret waktu (time-series).
  4. Penyimpanan Terstruktur: Menyimpan hasil pembersihan ke dalam direktori data/processed/btc_clean_latest.csv.

Perintah eksekusi pemrosesan data:
```bash
python src/preprocess.py
```

### 3. Manajemen Versi Data dengan DVC
Dalam proyek MLOps, pelacakan versi dataset sangat krusial untuk menjamin reproduksibilitas model (model reproducibility). Tanpa versioning yang tepat, performa model yang berubah di kemudian hari tidak dapat ditelusuri kembali ke asal usul dataset pelatihannya.

Langkah Implementasi:
  1. Inisialisasi DVC
     ```bash
     dvc init
     git add .dvc .dvcignore
     git commit -m "chore: initialize DVC"
     ```
  2. Tracking Dataset Mentah:
     ```bash
     dvc add data/raw/btc_raw_20261013_010602.csv
     git add data/raw/btc_raw_20261013_010602.csv.dvc data/raw/.gitignore
     git commit -m "feat(data): track raw dataset version with DVC"
     ```
  3. Simulasi Continual Learning & Audit Perubahan Versi:
     Setelah melakukan penambahan data baru secara berkala, silsilah data diverifikasi menggunakan:
     ```bash
     dvc diff
     ```

### 4. Experiment Tracking & Model Registry (LK-06)
Sistem menggunakan **MLflow** untuk mencatat parameter, metrik evaluasi ($RMSE, MAE, MAPE$), dan artefak model pelatihan secara terstruktur.

- **Experiment Name**: `Bitcoin_Price_Prediction_XGBoost`
- **Registered Model**: `BitcoinPriceXGBoostModel`
- **Hyperparameter Tuning**: Dijalankan 3 variasi run (XGBoost Regressor) dengan variasi `n_estimators`, `max_depth`, dan `learning_rate`.
- **Model Terpilih**: Run dengan `max_depth=5`, `learning_rate=0.03`, dan `n_estimators=100` menghasilkan performa terbaik ($MAPE \le 3.0\%$) dan dipromosikan ke MLflow Model Registry.

### 5. Experiment Tracking, Model Registry, & Kesiapan Inferensi (LK-07)

Sistem menggunakan **MLflow** dengan backend database SQLite (`mlflow.db`) untuk mencatat parameter, metrik evaluasi ($RMSE, MAE, MAPE$), dan mengelola siklus hidup model (*Model Lifecycle Management*) dari tahap eksperimen hingga *Production*.

- **Experiment Name**: `Bitcoin_Price_Prediction_XGBoost`
- **Registered Model**: `BitcoinPriceXGBoostModel`

### 5.1 Model Aktif & Alasan Pemilihan
Versi model yang saat ini dipromosikan dan diaktifkan di stage **`Production`** (atau alias `Champion`) adalah **Version 6**.

* **Hyperparameter Utama:** `max_depth=5`, `learning_rate=0.03`, `n_estimators=100`
* **Metrik Evaluasi Performa:**
  * **RMSE:** $1,410.25 USD
  * **MAE:** $1,120.40 USD
  * **MAPE:** **1.82%**
* **Alasan Pemilihan Versi:** Version 6 menghasilkan tingkat kesalahan terkecil dibandingkan variasi versi sebelumnya (v1–v5) dan berhasil memenuhi kriteria ambang batas sukses proyek ($MAPE \le 3.0\%$).

### 5.2 Transisi Stage Model (Model Registry Lifecycle)
Pengelolaan versi dilakukan melalui alur kerja *Model Registry*:
1. **Version 1–3:** Eksperimen awal (Archived / None).
2. **Version 4–5:** Pengujian hyperparameter menengah (Staging).
3. **Version 6:** Promosi status ke **Production** setelah verifikasi performa terbaik.

### 5.3 Verifikasi Inferensi Programatik
Kesiapan sistem inferensi diverifikasi dengan memuat model berstatus *Production* secara otomatis dari registry menggunakan skrip `test_inference.py`:

```bash
python test_inference.py
```

Hasil Pengujian Terminal (Output):

```plaintext
--- Memuat Model dari Version 6 (Stage: Production) ---

Hasil Prediksi Harga BTC Berikutnya: $63,714.63

--- Verifikasi Inferensi Berhasil Mulus! ---
```
