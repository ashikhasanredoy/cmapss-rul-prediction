import pandas as pd
from typing import List

def add_trend_features(df: pd.DataFrame, sensor_cols: List[str], lags: List[int] = [1, 5]) -> pd.DataFrame:
    df_out = df.copy()
    if 'engine_id' not in df_out.columns:
        df_out['engine_id'] = 1
    if 'cycle' not in df_out.columns:
        df_out['cycle'] = list(range(1, len(df_out) + 1))
    target_sensors = [col for col in sensor_cols if col in df_out.columns]
    diff_1 = df_out.groupby('engine_id')[target_sensors].diff(1).fillna(0)
    diff_1.columns = [f'{col}_delta_1' for col in target_sensors]
    early_cycles = df_out[df_out['cycle'] <= 5]
    if early_cycles.empty:
        early_cycles = df_out.groupby('engine_id').head(5)
    initial_baseline = early_cycles.groupby('engine_id')[target_sensors].mean().reset_index()
    initial_baseline.columns = ['engine_id'] + [f'{col}_initial_base' for col in target_sensors]
    df_merged = df_out.merge(initial_baseline, on='engine_id', how='left')
    base_diff_cols = {}
    for col in target_sensors:
        base_diff_cols[f'{col}_dev_from_base'] = (df_merged[col] - df_merged[f'{col}_initial_base']).fillna(0)
    base_diff_df = pd.DataFrame(base_diff_cols, index=df_out.index)
    lag_dfs = []
    for l in lags:
        lag_df = df_out.groupby('engine_id')[target_sensors].shift(l).bfill().fillna(0)
        lag_df.columns = [f'{col}_lag_{l}' for col in target_sensors]
        lag_dfs.append(lag_df)
    df_out = pd.concat([df_out, diff_1, base_diff_df] + lag_dfs, axis=1)
    return df_out
