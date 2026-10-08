import pandas as pd
from typing import Tuple, List
from src.config import SENSOR_COLUMNS, ROLLING_WINDOWS, PIECEWISE_RUL_LIMIT
from src.features.rul import add_rul_target
from src.features.rolling import add_rolling_features
from src.features.trends import add_trend_features

def build_feature_dataframe(df: pd.DataFrame, is_train: bool=True, piecewise_limit: int=PIECEWISE_RUL_LIMIT) -> pd.DataFrame:
    df_feat = df.copy()
    if is_train:
        df_feat = add_rul_target(df_feat, piecewise_limit=piecewise_limit)
    available_sensors = [c for c in SENSOR_COLUMNS if c in df_feat.columns]
    df_feat = add_rolling_features(df_feat, sensor_cols=available_sensors, windows=ROLLING_WINDOWS)
    df_feat = add_trend_features(df_feat, sensor_cols=available_sensors, lags=[1, 5])
    return df_feat

def extract_features_and_target(df_features: pd.DataFrame, target_col: str='RUL_piecewise') -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    drop_cols = ['engine_id', 'RUL', 'RUL_piecewise']
    feature_cols = [c for c in df_features.columns if c not in drop_cols]
    X = df_features[feature_cols].copy()
    y = df_features[target_col].copy() if target_col in df_features.columns else None
    groups = df_features['engine_id'].copy() if 'engine_id' in df_features.columns else None
    return (X, y, groups)

def extract_official_test_features(df_test_features: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    last_cycle_indices = df_test_features.groupby('engine_id')['cycle'].idxmax()
    test_last_records = df_test_features.loc[last_cycle_indices].sort_values('engine_id').reset_index(drop=True)
    engine_ids = test_last_records['engine_id'].tolist()
    drop_cols = ['engine_id', 'RUL', 'RUL_piecewise']
    feature_cols = [c for c in test_last_records.columns if c not in drop_cols]
    return (test_last_records[feature_cols], engine_ids)
