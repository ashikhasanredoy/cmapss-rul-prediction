import pandas as pd
from typing import List

def add_rolling_features(df: pd.DataFrame, sensor_cols: List[str], windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    df_out = df.copy()
    if 'engine_id' not in df_out.columns:
        df_out['engine_id'] = 1
    if 'cycle' not in df_out.columns:
        df_out['cycle'] = list(range(1, len(df_out) + 1))
    target_sensors = [col for col in sensor_cols if col in df_out.columns]
    for w in windows:
        rolled = df_out.groupby('engine_id')[target_sensors].rolling(window=w, min_periods=1)
        roll_mean = rolled.mean().bfill().fillna(0).reset_index(level=0, drop=True)
        roll_std = rolled.std().fillna(0).reset_index(level=0, drop=True)
        roll_mean.columns = [f'{col}_roll_mean_{w}' for col in target_sensors]
        roll_std.columns = [f'{col}_roll_std_{w}' for col in target_sensors]
        df_out = pd.concat([df_out, roll_mean, roll_std], axis=1)
    return df_out
