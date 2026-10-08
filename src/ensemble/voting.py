from sklearn.ensemble import VotingRegressor
from typing import List, Tuple, Any
import pandas as pd

def create_voting_ensemble(estimators: List[Tuple[str, Any]], weights: list=None) -> VotingRegressor:
    return VotingRegressor(estimators=estimators, weights=weights, n_jobs=-1)
