import os
import sys
import pytest
import pandas as pd
from pathlib import Path
from src.logger import logger, LOGS_DIR
from src.exception import CustomException
from src.utils import save_object, load_object, save_json, load_json, save_dataframe, load_dataframe

def test_logger_file_creation():
    logger.info("Test log message for verification")
    log_files = list(LOGS_DIR.glob("*.log"))
    assert len(log_files) > 0

def test_custom_exception_capture():
    try:
        raise ValueError("Simulated pipeline failure")
    except Exception as e:
        custom_exc = CustomException(e, sys)
        assert "Simulated pipeline failure" in str(custom_exc)
        assert "test_utils_exception.py" in str(custom_exc)

def test_utils_serialization(tmp_path):
    obj_data = {'model_name': 'XGBoost', 'score': 0.95}
    obj_path = tmp_path / "model.pkl"
    save_object(obj_path, obj_data)
    loaded_obj = load_object(obj_path)
    assert loaded_obj == obj_data

    json_data = {'epochs': 100, 'lr': 0.01}
    json_path = tmp_path / "config.json"
    save_json(json_path, json_data)
    loaded_json = load_json(json_path)
    assert loaded_json == json_data

    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df_path = tmp_path / "data.csv"
    save_dataframe(df_path, df)
    loaded_df = load_dataframe(df_path)
    assert loaded_df.shape == (3, 2)
