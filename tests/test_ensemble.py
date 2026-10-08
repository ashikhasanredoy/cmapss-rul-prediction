import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from src.ensemble.voting import create_voting_ensemble
from src.ensemble.stacking import EngineGroupedStackingRegressor

def test_voting_ensemble():
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.randn(100))
    m1 = Ridge(alpha=1.0)
    m2 = LinearRegression()
    ensemble = create_voting_ensemble([('m1', m1), ('m2', m2)])
    ensemble.fit(X, y)
    preds = ensemble.predict(X)
    assert len(preds) == 100

def test_stacking_ensemble_with_engine_groups():
    X = pd.DataFrame(np.random.randn(120, 5), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(np.random.randn(120))
    groups = pd.Series(np.repeat(range(1, 7), 20))
    m1 = Ridge(alpha=1.0)
    m2 = LinearRegression()
    stacker = EngineGroupedStackingRegressor(base_models=[('m1', m1), ('m2', m2)])
    stacker.fit(X, y, groups=groups, n_splits=3)
    preds = stacker.predict(X)
    assert len(preds) == 120
    assert not np.isnan(preds).any()
