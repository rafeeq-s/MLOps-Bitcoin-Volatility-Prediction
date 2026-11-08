import os

def test_config_exists():
    assert os.path.exists("requirements.txt"), "requirements.txt harus ada."

def test_mlflow_db_exists():
    assert os.path.exists("mlflow.db"), "mlflow.db harus ada."