import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
from src.config import CONSTANT_SENSORS_FD001, SETTING_DROPS_FD001
from src.preprocessing.cleaning import clean_dataframe

class PreprocessingPipeline:

    def __init__(self, drop_cols: List[str]=CONSTANT_SENSORS_FD001 + SETTING_DROPS_FD001, scale_numeric: bool=False):
        self.drop_cols = drop_cols
        self.scale_numeric = scale_numeric
        self.scaler = StandardScaler() if scale_numeric else None
        self.feature_columns: Optional[List[str]] = None
        self.is_fitted = False

    def clean_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = clean_dataframe(df)
        cols_to_remove = [c for c in self.drop_cols if c in df_clean.columns]
        return df_clean.drop(columns=cols_to_remove)

    def fit(self, X_train: pd.DataFrame):
        self.feature_columns = list(X_train.columns)
        if self.scale_numeric and self.scaler is not None:
            self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError('PreprocessingPipeline must be fitted on training data before transform.')
        X_aligned = X[self.feature_columns].copy()
        if self.scale_numeric and self.scaler is not None:
            scaled_vals = self.scaler.transform(X_aligned)
            return pd.DataFrame(scaled_vals, columns=self.feature_columns, index=X.index)
        return X_aligned

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)
