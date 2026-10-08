import pandas as pd

def clean_dataframe(df: pd.DataFrame, drop_duplicates: bool=True) -> pd.DataFrame:
    df_clean = df.copy()
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    if 'engine_id' in df_clean.columns and 'cycle' in df_clean.columns:
        df_clean = df_clean.sort_values(by=['engine_id', 'cycle']).reset_index(drop=True)
    return df_clean
