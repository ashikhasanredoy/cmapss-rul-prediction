import pandas as pd
from pathlib import Path
from typing import Union
from src.config import ALL_COLUMNS, TRAIN_FILE, TEST_FILE, RUL_FILE

def load_train_data(file_path: Union[str, Path]=TRAIN_FILE) -> pd.DataFrame:
    return pd.read_csv(file_path, sep='\\s+', header=None, names=ALL_COLUMNS)

def load_official_test_data(file_path: Union[str, Path]=TEST_FILE) -> pd.DataFrame:
    return pd.read_csv(file_path, sep='\\s+', header=None, names=ALL_COLUMNS)

def load_official_rul_truth(file_path: Union[str, Path]=RUL_FILE) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep='\\s+', header=None, names=['RUL'])
    df['engine_id'] = range(1, len(df) + 1)
    return df[['engine_id', 'RUL']]
