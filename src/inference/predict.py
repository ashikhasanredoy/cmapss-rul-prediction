import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from src.config import MODELS_FINAL_DIR
from src.features.builder import build_feature_dataframe
from src.utils import load_object, load_json
from src.exception import CustomException

class RULPredictor:

    def __init__(self, model_dir: Path = MODELS_FINAL_DIR):
        try:
            self.model = load_object(model_dir / 'model.pkl')
            self.feature_names = load_json(model_dir / 'feature_config.json')['feature_names']
        except Exception as e:
            raise CustomException(e, sys)

    def predict_engine(self, df_engine: pd.DataFrame) -> Dict[str, Any]:
        try:
            df_in = df_engine.copy()
            if 'engine_id' in df_in.columns and df_in['engine_id'].nunique() > 1:
                target_id = df_in['engine_id'].iloc[-1]
                df_in = df_in[df_in['engine_id'] == target_id]
            df_feat = build_feature_dataframe(df_in, is_train=False)
            latest = df_feat.sort_values('cycle').iloc[-1]
            engine_id = int(latest.get('engine_id', 1))
            current_cycle = int(latest.get('cycle', len(df_in)))
            X = pd.DataFrame([latest]).reindex(columns=self.feature_names, fill_value=0.0)
            predicted_rul = float(np.clip(self.model.predict(X)[0], 0, None))
            status = 'CRITICAL (Immediate Maintenance Required)' if predicted_rul <= 15 else 'WARNING (Schedule Inspection)' if predicted_rul <= 40 else 'HEALTHY (Normal Operational Margin)'
            indicator = 'RED' if predicted_rul <= 15 else 'YELLOW' if predicted_rul <= 40 else 'GREEN'
            return {'engine_id': engine_id, 'current_cycle': current_cycle, 'predicted_rul_cycles': round(predicted_rul, 1), 'estimated_failure_cycle': round(current_cycle + predicted_rul, 1), 'health_status': status, 'status_indicator': indicator}
        except Exception as e:
            raise CustomException(e, sys)

def predict_single_engine(engine_df: pd.DataFrame) -> Dict[str, Any]:
    return RULPredictor().predict_engine(engine_df)
