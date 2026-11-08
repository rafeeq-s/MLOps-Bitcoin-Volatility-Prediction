import os
import mlflow
from mlflow.tracking import MlflowClient

# Dapatkan absolute path direktori root proyek secara dinamis
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
db_path = os.path.join(project_root, "mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{db_path}")
client = MlflowClient()

THRESHOLD_MAPE = 3.0  # Threshold sesuai kriteria LK-01 (MAPE <= 3.0%)

def evaluate_and_register():
    # Ambil run terbaru dari experiment
    experiment = client.get_experiment_by_name("Bitcoin_Price_Prediction_XGBoost")
    if not experiment:
        raise ValueError("Experiment 'Bitcoin_Price_Prediction_XGBoost' tidak ditemukan.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("Tidak ada run MLflow yang ditemukan di mlflow.db.")
        
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    mape = latest_run.data.metrics.get("mape", float("inf"))
    
    print(f"Run ID: {run_id}")
    print(f"Hasil Evaluasi MAPE: {mape:.2f}% (Threshold: {THRESHOLD_MAPE}%)")
    
    # Gatekeeper Validation
    if mape <= THRESHOLD_MAPE:
        print(" Validasi Lolos: Memenuhi kriteria performa!")
        model_name = "BitcoinPriceXGBoostModel"
        model_uri = f"runs:/{run_id}/model"
        
        # Mendaftarkan model dan mempromosikan ke Staging
        mv = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Staging"
        )
        print(f" Model berhasil didaftarkan ke MLflow Registry [{model_name}] versi {mv.version} dengan status STAGING.")
    else:
        raise ValueError(f" Validasi Gagal: MAPE ({mape:.2f}%) melebihi threshold ({THRESHOLD_MAPE}%). Model ditolak!")

if __name__ == "__main__":
    evaluate_and_register()