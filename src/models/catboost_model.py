from catboost import CatBoostRegressor
from src.config import RANDOM_STATE

def get_catboost_model(**kwargs):
    params = {'iterations': 150, 'depth': 6, 'learning_rate': 0.05, 'random_seed': RANDOM_STATE, 'verbose': 0}
    params.update(kwargs)
    return CatBoostRegressor(**params)
