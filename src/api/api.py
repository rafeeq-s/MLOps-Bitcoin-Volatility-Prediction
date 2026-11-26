import os
import time
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Bitcoin Price & Volatility Prediction API")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "BitcoinPriceXGBoostModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Definisi Metrik Operasional & Model Drift Prometheus
PREDICTION_REQUESTS = Counter(
    "btc_prediction_requests_total",
    "Total permintaan prediksi yang diterima API",
    ["endpoint", "status"]
)
INFERENCE_LATENCY = Histogram(
    "btc_inference_latency_seconds",
    "Latensi eksekusi inferensi model dalam detik",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
PREDICTED_PRICE_GAUGE = Gauge(
    "btc_predicted_price_usd",
    "Skor prediksi harga penutupan Bitcoin terakhir (USD) untuk deteksi drift"
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "model_name": MODEL_NAME,
        "stage": MODEL_STAGE,
    }

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/predict/latest")
def predict_latest():
    start_time = time.time()
    try:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)

        data_path = "data/processed/btc_clean_latest.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path}")

        df = pd.read_csv(data_path)
        exclude_cols = ['price', 'timestamp', 'date', 'datetime']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X_latest = df[feature_cols].select_dtypes(include=[np.number]).tail(1)

        prediction = model.predict(X_latest)
        pred_value = round(float(prediction[0]), 2)

        latency = time.time() - start_time
        INFERENCE_LATENCY.observe(latency)
        PREDICTED_PRICE_GAUGE.set(pred_value)
        PREDICTION_REQUESTS.labels(endpoint="/predict/latest", status="success").inc()

        return {
            "model": MODEL_NAME,
            "stage": MODEL_STAGE,
            "mlflow_server": MLFLOW_TRACKING_URI,
            "predicted_btc_price_usd": pred_value,
            "inference_latency_sec": round(latency, 4)
        }
    except Exception as e:
        PREDICTION_REQUESTS.labels(endpoint="/predict/latest", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
