from lightgbm import LGBMRegressor
from src.config import RANDOM_STATE

def get_lightgbm_model(**kwargs):
    params = {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.05, 'num_leaves': 31, 'subsample': 0.8, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1}
    params.update(kwargs)
    return LGBMRegressor(**params)
