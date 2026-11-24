import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Bitcoin Price & Volatility Prediction API")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "BitcoinPriceXGBoostModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "model_name": MODEL_NAME,
        "stage": MODEL_STAGE,
    }

@app.get("/predict/latest")
def predict_latest():
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

        return {
            "model": MODEL_NAME,
            "stage": MODEL_STAGE,
            "mlflow_server": MLFLOW_TRACKING_URI,
            "predicted_btc_price_usd": round(float(prediction[0]), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
