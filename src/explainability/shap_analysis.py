import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import SHAP_DIR
from src.logger import logger

def generate_shap_explanations(model, X_sample: pd.DataFrame, summary_path: Path = SHAP_DIR / 'shap_summary.png', bar_path: Path = SHAP_DIR / 'shap_bar.png', max_display: int = 15):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if hasattr(model, 'predict'):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_sample)
            except Exception:
                explainer = shap.Explainer(model.predict, X_sample)
                shap_values = explainer(X_sample)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
        plt.title('SHAP Feature Attribution (Summary Plot)', fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        plt.title('SHAP Global Feature Importance', fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"SHAP explanation generation note: {e}")
