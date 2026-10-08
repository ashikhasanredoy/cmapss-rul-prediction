import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from src.config import RANDOM_STATE
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_xgboost_optuna(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int=15) -> Tuple[XGBRegressor, Dict[str, Any]]:

    def objective(trial):
        params = {'n_estimators': trial.suggest_int('n_estimators', 100, 250, step=50), 'max_depth': trial.suggest_int('max_depth', 3, 6), 'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.12, log=True), 'subsample': trial.suggest_float('subsample', 0.7, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0), 'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 5.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 5.0, log=True), 'random_state': RANDOM_STATE, 'n_jobs': -1}
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, preds)))
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = -1
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    return (best_model, best_params)

def tune_lightgbm_optuna(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int=15) -> Tuple[LGBMRegressor, Dict[str, Any]]:

    def objective(trial):
        params = {'n_estimators': trial.suggest_int('n_estimators', 100, 250, step=50), 'max_depth': trial.suggest_int('max_depth', 3, 7), 'num_leaves': trial.suggest_int('num_leaves', 15, 45), 'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.12, log=True), 'subsample': trial.suggest_float('subsample', 0.7, 1.0), 'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 5.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 5.0, log=True), 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1}
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, preds)))
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = -1
    best_params['verbose'] = -1
    best_model = LGBMRegressor(**best_params)
    best_model.fit(X_train, y_train)
    return (best_model, best_params)

def tune_catboost_optuna(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int=12) -> Tuple[CatBoostRegressor, Dict[str, Any]]:

    def objective(trial):
        params = {'iterations': trial.suggest_int('iterations', 100, 200, step=50), 'depth': trial.suggest_int('depth', 4, 7), 'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.12, log=True), 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 8.0), 'random_seed': RANDOM_STATE, 'verbose': 0}
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, preds)))
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params['random_seed'] = RANDOM_STATE
    best_params['verbose'] = 0
    best_model = CatBoostRegressor(**best_params)
    best_model.fit(X_train, y_train)
    return (best_model, best_params)
