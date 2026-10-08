from sklearn.ensemble import RandomForestRegressor
from src.config import RANDOM_STATE

def get_random_forest_model(n_estimators: int=100, max_depth: int=12, random_state: int=RANDOM_STATE):
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=5, min_samples_leaf=2, n_jobs=-1, random_state=random_state)
