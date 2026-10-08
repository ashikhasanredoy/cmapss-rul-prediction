import pandas as pd
from src.data.data_loader import load_train_data
from src.features.rul import add_rul_target
from src.features.builder import build_feature_dataframe, extract_features_and_target

def test_rul_target_computation():
    df = load_train_data().head(200)
    df_rul = add_rul_target(df, piecewise_limit=125)
    assert 'RUL' in df_rul.columns
    assert 'RUL_piecewise' in df_rul.columns
    assert df_rul['RUL_piecewise'].max() <= 125
    assert df_rul['RUL'].min() >= 0

def test_feature_builder():
    df = load_train_data().head(300)
    df_feat = build_feature_dataframe(df, is_train=True)
    assert any(('_roll_mean_' in c for c in df_feat.columns))
    assert any(('_delta_1' in c for c in df_feat.columns))
    (X, y, groups) = extract_features_and_target(df_feat)
    assert len(X) == len(df)
    assert len(y) == len(df)
    assert len(groups) == len(df)
