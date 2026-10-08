import pandas as pd
from typing import Dict, Any, List
from src.evaluation.metrics import calculate_metrics

def evaluate_model(y_true, y_pred, model_name: str='Model') -> Dict[str, Any]:
    metrics = calculate_metrics(y_true, y_pred)
    metrics['Model'] = model_name
    return metrics

def create_leaderboard(evaluations: List[Dict[str, Any]], sort_by: str='RMSE') -> pd.DataFrame:
    df = pd.DataFrame(evaluations)
    cols = ['Model', 'MAE', 'RMSE', 'R2', 'NASA_Score']
    df = df[cols].sort_values(by=sort_by).reset_index(drop=True)
    return df
