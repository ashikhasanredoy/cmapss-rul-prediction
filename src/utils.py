import os
import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = Path(file_path).parent
        dir_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Object file not found at {file_path}")
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)

def save_json(file_path, data):
    try:
        dir_path = Path(file_path).parent
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        raise CustomException(e, sys)

def load_json(file_path):
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"JSON file not found at {file_path}")
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise CustomException(e, sys)

def save_dataframe(file_path, df):
    try:
        dir_path = Path(file_path).parent
        dir_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise CustomException(e, sys)

def load_dataframe(file_path):
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"DataFrame file not found at {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomException(e, sys)
