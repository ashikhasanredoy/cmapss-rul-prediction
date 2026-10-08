import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import FIGURES_DIR

def plot_actual_vs_predicted(y_true, y_pred, title: str='Actual vs Predicted RUL', output_path: Path=FIGURES_DIR / 'actual_vs_predicted.png'):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 0, None)
    plt.scatter(y_true, y_pred, alpha=0.6, color='#235789', edgecolors='w', s=45, label='Predictions')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Agreement (y = x)')
    plt.title(title, fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Actual RUL (Cycles)', fontsize=11)
    plt.ylabel('Predicted RUL (Cycles)', fontsize=11)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_residual_analysis(y_true, y_pred, output_path: Path=FIGURES_DIR / 'residual_plot.png'):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 0, None)
    residuals = y_pred - y_true
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(y_true, residuals, alpha=0.5, color='#e6550d', edgecolors='none', s=35)
    ax1.axhline(0, color='black', linestyle='--', lw=1.5)
    ax1.set_title('Residuals vs Actual RUL', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Actual RUL')
    ax1.set_ylabel('Residual (Predicted - Actual)')
    sns.histplot(residuals, kde=True, ax=ax2, color='#756bb1', edgecolor='black')
    ax2.axvline(0, color='black', linestyle='--', lw=1.5)
    ax2.set_title('Residual Error Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Residual')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_importance(feature_names: list, importances: np.ndarray, top_n: int=15, output_path: Path=FIGURES_DIR / 'feature_importance.png'):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='Blues_r')
    plt.title(f'Top {top_n} Most Important Features', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Importance Score', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_model_comparison_bar(comparison_df: pd.DataFrame, output_path: Path=FIGURES_DIR / 'model_comparison.png'):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_sorted = comparison_df.sort_values(by='RMSE', ascending=True)
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=df_sorted, x='MAE', y='Model', ax=ax1, palette='crest')
    ax1.set_title('Mean Absolute Error (MAE) by Model', fontsize=12, fontweight='bold')
    ax1.set_xlabel('MAE (Lower is Better)')
    sns.barplot(data=df_sorted, x='RMSE', y='Model', ax=ax2, palette='viridis')
    ax2.set_title('Root Mean Squared Error (RMSE) by Model', fontsize=12, fontweight='bold')
    ax2.set_xlabel('RMSE (Lower is Better)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
