import os
import mlflow.pyfunc
import pandas as pd

def test_production_model():
    print("--- Memuat Model dari Version 6 ---")
    
    # Path ke database SQLite utama di root
    db_path = os.path.abspath("mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    
    # Panggil langsung Version 6
    model_uri = "models:/BitcoinPriceXGBoostModel/6"
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Dummy Data Input (6 fitur)
    dummy_data = pd.DataFrame([{
        'price': 65000.0,
        'volume_24h': 35000000000.0,
        'market_cap': 1200000000000.0,
        'price_lag1': 64500.0,
        'vol_lag1': 34000000000.0,
        'volatility_7d': 0.02
    }])
    
    # Uji Inferensi
    prediction = model.predict(dummy_data)
    print(f"\nHasil Prediksi Harga BTC Berikutnya: ${prediction[0]:,.2f}\n")
    print("--- Verifikasi Inferensi Berhasil Mulus! ---")

if __name__ == "__main__":
    test_production_model()
