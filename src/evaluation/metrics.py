import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.evaluation.nasa_score import compute_nasa_score

def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred_clipped = np.clip(y_pred, 0, None)
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clipped)))
    r2 = float(r2_score(y_true, y_pred_clipped))
    nasa_score = float(compute_nasa_score(y_true, y_pred_clipped))
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4), 'NASA_Score': round(nasa_score, 2)}
