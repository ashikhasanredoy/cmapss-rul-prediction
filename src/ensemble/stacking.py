import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.base import clone

class EngineGroupedStackingRegressor:

    def __init__(self, base_models: List[Tuple[str, Any]], meta_model: Any = None):
        self.base_models = base_models
        self.meta_model = Ridge(alpha=1.0) if meta_model is None else meta_model
        self.fitted_base_models = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5):
        gkf = GroupKFold(n_splits=n_splits)
        num_samples = len(X)
        num_models = len(self.base_models)
        oof_predictions = np.zeros((num_samples, num_models))
        X_arr = X.values
        y_arr = y.values
        groups_arr = groups.values
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_arr, y_arr, groups=groups_arr)):
            X_fold_train, y_fold_train = X_arr[train_idx], y_arr[train_idx]
            X_fold_val = X_arr[val_idx]
            for model_idx, (name, model) in enumerate(self.base_models):
                cloned_model = clone(model)
                cloned_model.fit(X_fold_train, y_fold_train)
                oof_predictions[val_idx, model_idx] = cloned_model.predict(X_fold_val)
        self.meta_model.fit(oof_predictions, y_arr)
        self.fitted_base_models = []
        for name, model in self.base_models:
            full_model = clone(model)
            full_model.fit(X, y)
            self.fitted_base_models.append((name, full_model))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError('StackingRegressor is not fitted yet.')
        base_preds = np.zeros((len(X), len(self.fitted_base_models)))
        for idx, (name, model) in enumerate(self.fitted_base_models):
            base_preds[:, idx] = model.predict(X)
        return self.meta_model.predict(base_preds)
