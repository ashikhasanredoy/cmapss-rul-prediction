from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

def get_mean_baseline_model():
    return DummyRegressor(strategy='mean')

def get_linear_regression_model():
    return LinearRegression()
