import pandas as pd
from src.data.data_loader import load_train_data
from src.preprocessing.cleaning import clean_dataframe
from src.preprocessing.splitting import split_engines
from src.preprocessing.pipeline import PreprocessingPipeline

def test_clean_dataframe():
    df = load_train_data()
    cleaned_df = clean_dataframe(df)
    assert isinstance(cleaned_df, pd.DataFrame)
    assert not cleaned_df.empty

def test_split_engines_leakage_free():
    df = load_train_data()
    (train_df, val_df, test_df, dev_df) = split_engines(df)
    train_eng = set(train_df['engine_id'].unique())
    val_eng = set(val_df['engine_id'].unique())
    test_eng = set(test_df['engine_id'].unique())
    assert len(train_eng.intersection(val_eng)) == 0
    assert len(train_eng.intersection(test_eng)) == 0
    assert len(val_eng.intersection(test_eng)) == 0
    assert len(train_eng) == 70
    assert len(val_eng) == 15
    assert len(test_eng) == 15
