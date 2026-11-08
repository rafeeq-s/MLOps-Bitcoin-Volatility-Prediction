import os

def test_config_exists():
    assert os.path.exists("requirements.txt"), "requirements.txt harus ada."

def test_source_directory_exists():
    assert os.path.exists("src"), "Folder src/ harus ada."
    assert os.path.exists("src/models/train.py"), "File train.py harus ada."
    assert os.path.exists("src/models/evaluate.py"), "File evaluate.py harus ada."