import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from src.config import FIGURES_DIR
sns.set_theme(style='whitegrid', palette='muted')

def plot_rul_distribution(df: pd.DataFrame, output_path: Path=FIGURES_DIR / 'rul_distribution.png'):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    sns.histplot(df['RUL'], kde=True, bins=35, color='#2077B4', edgecolor='black')
    plt.title('Distribution of Remaining Useful Life (RUL)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('RUL (Cycles)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_correlation_matrix(df: pd.DataFrame, output_path: Path=FIGURES_DIR / 'correlation_matrix.png'):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr_cols = [c for c in df.columns if c.startswith('sensor_') or c in ['cycle', 'RUL']]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, cmap='coolwarm', annot=False, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Heatmap: Sensors & RUL', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_sensor_trends(df: pd.DataFrame, engine_ids: list=[1, 2, 3], sensors: list=['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_11', 'sensor_12'], output_path: Path=FIGURES_DIR / 'sensor_trends.png'):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    available_sensors = [s for s in sensors if s in df.columns]
    num_sensors = len(available_sensors)
    if num_sensors == 0:
        return
    (fig, axes) = plt.subplots(nrows=(num_sensors + 1) // 2, ncols=2, figsize=(14, 3.5 * ((num_sensors + 1) // 2)), sharex=False)
    axes = axes.flatten()
    sample_df = df[df['engine_id'].isin(engine_ids)]
    for (idx, sensor) in enumerate(available_sensors):
        ax = axes[idx]
        for engine_id in engine_ids:
            eng_data = sample_df[sample_df['engine_id'] == engine_id]
            ax.plot(eng_data['cycle'], eng_data[sensor], label=f'Engine {engine_id}', alpha=0.85, linewidth=1.8)
        ax.set_title(f'Degradation Trend: {sensor}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Operational Cycle')
        ax.set_ylabel('Sensor Reading')
        ax.legend(loc='best', fontsize=9)
    for idx in range(num_sensors, len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
