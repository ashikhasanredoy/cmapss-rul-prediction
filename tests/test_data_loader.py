import pytest
import pandas as pd
from src.data.data_loader import load_train_data, load_official_test_data, load_official_rul_truth
from src.data.validation import validate_dataset

def test_load_train_data():
    df = load_train_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'engine_id' in df.columns
    assert 'cycle' in df.columns
    assert 'sensor_1' in df.columns
    assert 'sensor_21' in df.columns

def test_load_official_test_data():
    df = load_official_test_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'engine_id' in df.columns

def test_load_official_rul_truth():
    df = load_official_rul_truth()
    assert isinstance(df, pd.DataFrame)
    assert 'RUL' in df.columns
    assert len(df) == 100

def test_validate_dataset():
    df = load_train_data()
    report = validate_dataset(df, 'Train Set')
    assert report['missing_values'] == 0
    assert report['infinite_values'] == 0
    assert report['is_valid'] is True
