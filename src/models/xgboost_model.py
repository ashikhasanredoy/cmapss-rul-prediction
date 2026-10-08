from xgboost import XGBRegressor
from src.config import RANDOM_STATE

def get_xgboost_model(**kwargs):
    params = {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': RANDOM_STATE, 'n_jobs': -1}
    params.update(kwargs)
    return XGBRegressor(**params)
