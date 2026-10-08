import numpy as np
import pandas as pd
from typing import Tuple, Dict
from src.config import TRAIN_RATIO, VAL_RATIO, INTERNAL_TEST_RATIO, RANDOM_STATE
from src.logger import logger

def split_engines(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO, test_ratio: float = INTERNAL_TEST_RATIO, random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), 'Split ratios must sum to 1.0'
    engines = np.sort(df['engine_id'].unique())
    num_engines = len(engines)
    np.random.seed(random_state)
    shuffled_engines = np.random.permutation(engines)
    n_train = int(num_engines * train_ratio)
    n_val = int(num_engines * val_ratio)
    train_eng = shuffled_engines[:n_train]
    val_eng = shuffled_engines[n_train:n_train + n_val]
    internal_test_eng = shuffled_engines[n_train + n_val:]
    dev_eng = np.concatenate([train_eng, val_eng])
    train_df = df[df['engine_id'].isin(train_eng)].copy().reset_index(drop=True)
    val_df = df[df['engine_id'].isin(val_eng)].copy().reset_index(drop=True)
    internal_test_df = df[df['engine_id'].isin(internal_test_eng)].copy().reset_index(drop=True)
    dev_df = df[df['engine_id'].isin(dev_eng)].copy().reset_index(drop=True)
    logger.info(f"Engine split: {len(train_eng)} train, {len(val_eng)} val, {len(internal_test_eng)} test engines")
    return (train_df, val_df, internal_test_df, dev_df)
