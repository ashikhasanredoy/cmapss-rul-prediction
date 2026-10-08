import numpy as np
import pandas as pd
from src.evaluation.nasa_score import compute_nasa_score
from src.evaluation.metrics import calculate_metrics
from src.inference.predict import RULPredictor, predict_single_engine

def test_nasa_score_properties():
    y_true = np.array([50, 30, 20])
    y_pred = np.array([50, 30, 20])
    score = compute_nasa_score(y_true, y_pred)
    assert score == 0.0
    late_score = compute_nasa_score(np.array([50]), np.array([60]))
    early_score = compute_nasa_score(np.array([50]), np.array([40]))
    assert late_score > early_score

def test_metrics_calculation():
    y_true = [100, 50, 25]
    y_pred = [95, 48, 22]
    metrics = calculate_metrics(y_true, y_pred)
    assert 'MAE' in metrics
    assert 'RMSE' in metrics
    assert 'R2' in metrics
    assert 'NASA_Score' in metrics
    assert metrics['MAE'] > 0

def test_predictor_single_and_sequence():
    predictor = RULPredictor()
    single_input = {
        'engine_id': 1,
        'cycle': 50,
        'setting_1': 0.0023,
        'setting_2': -0.0004,
        'setting_3': 100.0,
        'sensor_1': 518.67,
        'sensor_2': 642.25,
        'sensor_3': 1589.70,
        'sensor_4': 1400.60,
        'sensor_5': 14.62,
        'sensor_6': 21.61,
        'sensor_7': 554.36,
        'sensor_8': 2388.05,
        'sensor_9': 9046.19,
        'sensor_10': 1.30,
        'sensor_11': 47.47,
        'sensor_12': 521.66,
        'sensor_13': 2388.02,
        'sensor_14': 8138.62,
        'sensor_15': 8.4195,
        'sensor_16': 0.03,
        'sensor_17': 392,
        'sensor_18': 2388,
        'sensor_19': 100.0,
        'sensor_20': 39.06,
        'sensor_21': 23.4190
    }
    df_single = pd.DataFrame([single_input])
    res_single = predictor.predict_engine(df_single)
    assert 'predicted_rul_cycles' in res_single
    assert res_single['predicted_rul_cycles'] >= 0

    df_seq = pd.DataFrame([single_input, single_input])
    df_seq['cycle'] = [50, 51]
    res_seq = predictor.predict_engine(df_seq)
    assert res_seq['current_cycle'] == 51
