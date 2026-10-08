import pandas as pd
import numpy as np
from typing import Dict, Any
from src.logger import logger

def validate_dataset(df: pd.DataFrame, dataset_name: str = 'Dataset') -> Dict[str, Any]:
    num_rows, num_cols = df.shape
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = int(np.isinf(df[numeric_cols].values).sum())
    unique_engines = int(df['engine_id'].nunique()) if 'engine_id' in df.columns else 0
    max_cycles = int(df.groupby('engine_id')['cycle'].max().max()) if 'engine_id' in df.columns and 'cycle' in df.columns else 0
    return {
        'dataset_name': dataset_name,
        'rows': num_rows,
        'columns': num_cols,
        'missing_values': missing_count,
        'duplicates': duplicate_count,
        'infinite_values': inf_count,
        'unique_engines': unique_engines,
        'max_cycle_observed': max_cycles,
        'is_valid': missing_count == 0 and duplicate_count == 0 and (inf_count == 0)
    }

def print_validation_report(report: Dict[str, Any]) -> None:
    logger.info(f"Dataset '{report['dataset_name']}' validated: {report['rows']} rows, {report['columns']} cols, {report['unique_engines']} engines (Valid: {report['is_valid']})")
