import pandas as pd
from typing import Optional
from src.config import PIECEWISE_RUL_LIMIT

def add_rul_target(df: pd.DataFrame, piecewise_limit: Optional[int]=PIECEWISE_RUL_LIMIT) -> pd.DataFrame:
    df_out = df.copy()
    max_cycle = df_out.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']
    df_out = df_out.merge(max_cycle, on='engine_id', how='left')
    df_out['RUL'] = df_out['max_cycle'] - df_out['cycle']
    df_out.drop(columns=['max_cycle'], inplace=True)
    if piecewise_limit is not None and piecewise_limit > 0:
        df_out['RUL_piecewise'] = df_out['RUL'].clip(upper=piecewise_limit)
    else:
        df_out['RUL_piecewise'] = df_out['RUL']
    return df_out
