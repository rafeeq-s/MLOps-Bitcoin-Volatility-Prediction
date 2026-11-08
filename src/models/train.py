import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.xgboost

def train_and_log_model(data_path, n_estimators, max_depth, learning_rate):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    db_path = os.path.join(project_root, "mlflow.db")
    
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Bitcoin_Price_Prediction_XGBoost")

    # 1. Load Data Terbaru (atau fallback ke data paling baru di data/processed/)
    if not os.path.exists(data_path):
        processed_dir = os.path.join(project_root, "data/processed")
        files = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir) if f.startswith("btc_clean_") and f.endswith(".csv")]
        if files:
            data_path = max(files, key=os.path.getmtime)
        else:
            raise FileNotFoundError(f"Data tidak ditemukan di path: {data_path}")

    df = pd.read_csv(data_path)
    
    # Filter fitur: Ambil HANYA kolom numerik dan kecualikan kolom target 'price' serta kolom waktu/date/datetime
    exclude_cols = ['price', 'timestamp', 'date', 'datetime']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Seleksi hanya tipe data numerik (int, float) untuk XGBoost
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df['price']

    # Split Train/Test (80:20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 2. Train Model
    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate
        })

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 3. Evaluate Model
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

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
            artifact_path="model"
        )

        print(f"Run Berhasil! RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost with MLflow Tracking")
    parser.add_argument("--data-path", type=str, default="data/processed/btc_clean_latest.csv")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)

    args = parser.parse_args()
    train_and_log_model(args.data_path, args.n_estimators, args.max_depth, args.learning_rate)