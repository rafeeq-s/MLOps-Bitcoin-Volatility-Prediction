import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import mlflow
import mlflow.xgboost

def calculate_mape(y_true, y_pred):
    """Menghitung Mean Absolute Percentage Error (MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def load_and_preprocess_data(data_path):
    """Membaca dataset terbersihkan dan membuat fitur time-series sederhana"""
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Feature Engineering (Lag Features & Target)
    df['price_lag1'] = df['price'].shift(1)
    df['vol_lag1'] = df['volume_24h'].shift(1)
    df['returns'] = df['price'].pct_change()
    df['volatility_7d'] = df['returns'].rolling(window=7).std()
    
    # Target: Harga Penutupan Hoki Berikutnya (P_t+1)
    df['target_price'] = df['price'].shift(-1)
    
    # Drop Missing Values akibat rolling/lag
    df = df.dropna().reset_index(drop=True)
    
    features = ['price', 'volume_24h', 'market_cap', 'price_lag1', 'vol_lag1', 'volatility_7d']
    X = df[features]
    y = df['target_price']
    
    # Split Data (Time-Series Split: 80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test

def train_and_log_model(data_path, n_estimators, max_depth, learning_rate):
    # Set MLflow Experiment
    mlflow.set_experiment("Bitcoin_Price_Prediction_XGBoost")
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    with mlflow.start_run(run_name=f"XGB_depth{max_depth}_lr{learning_rate}_n{n_estimators}"):
        # 1. Log Hyperparameters
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": 42
        }
        mlflow.log_params(params)
        
        # 2. Train Model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # 3. Predict & Evaluate
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = calculate_mape(y_test, predictions)
        
        # 4. Log Metrics
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "mape": mape
        }
        mlflow.log_metrics(metrics)
        
        # 5. Log Model Artifact
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="BitcoinPriceXGBoostModel"
        )
        
        print(f"Run Berhasil! RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost with MLflow Tracking")
    parser.add_argument("--data-path", type=str, default="data/processed/btc_clean_latest.csv")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    
    args = parser.args = parser.parse_args()
    train_and_log_model(args.data_path, args.n_estimators, args.max_depth, args.learning_rate)