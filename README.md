# ✈️ NASA C-MAPSS Turbofan Engine RUL Prediction (Gold-Standard ML Pipeline)

An end-to-end, research-grade, and production-ready Machine Learning system for predicting the **Remaining Useful Life (RUL)** of aircraft turbofan engines using the benchmark **NASA C-MAPSS (FD001)** dataset, equipped with an interactive **Streamlit Frontend Web Dashboard**.

---

## 📊 Dataset Description

This project uses the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset**, specifically the **FD001 subset**, for **Predictive Maintenance and Remaining Useful Life (RUL) prediction**.

The dataset contains simulated sensor measurements from turbofan engines operating until failure. The training data contains complete engine life cycles, while the test data contains partially observed engine cycles. The goal is to predict how many operating cycles remain before an engine reaches failure.

### FD001 Dataset

FD001 contains:

* **100 training engines**
* **100 test engines**
* **1 operating condition**
* **1 fault mode**
* **21 sensor measurements**
* **3 operational settings**

Each row represents the condition of an engine at a particular operating cycle.

### Dataset Files

```text
data/
└── raw/
    ├── train_FD001.txt
    ├── test_FD001.txt
    └── RUL_FD001.txt
```

#### `train_FD001.txt`

Contains the complete run-to-failure history of 100 engines. Each engine is observed from its initial operating cycle until failure.

#### `test_FD001.txt`

Contains the partially observed history of another 100 engines. These engines have not been observed until failure, so their remaining useful life must be predicted.

#### `RUL_FD001.txt`

Contains the **true RUL values for the test engines**. These values are used only during final evaluation.

### Dataset Columns

The original dataset contains **26 columns**:

| Column | Description |
| :--- | :--- |
| `engine_id` | Unique identifier of the engine |
| `cycle` | Operating cycle of the engine |
| `setting_1` | Operational setting 1 |
| `setting_2` | Operational setting 2 |
| `setting_3` | Operational setting 3 |
| `sensor_1` – `sensor_21` | Sensor measurements collected from the engine |

The `RUL` column is not originally included in the training file. It is calculated from the training engine's failure cycle:

```text
RUL = Maximum Cycle of Engine - Current Cycle
```

For example, if an engine fails at cycle 200 and the current observation is cycle 150:

```text
RUL = 200 - 150
    = 50 cycles
```

### Prediction Target

The prediction target is:

```text
RUL (Remaining Useful Life)
```

RUL represents the estimated number of operating cycles remaining before an engine reaches failure.

This makes the problem a **supervised regression task**.

### Dataset Characteristics

The FD001 dataset is particularly useful for predictive maintenance because it contains:

* Time-series engine degradation
* Multiple sensor measurements
* Engine operating cycles
* Complete run-to-failure training histories
* Truncated test histories
* Official RUL values for evaluation

The dataset is simulated rather than collected from real aircraft engines, but it is widely used as a benchmark for evaluating predictive maintenance and RUL prediction methods.

### Data Split

To avoid data leakage, the training engines are split by `engine_id` rather than randomly splitting individual rows.

The development dataset is divided into:

* **70% Training**
* **15% Validation**
* **15% Internal Test**

The official NASA test set is kept separate and is used only for final evaluation.

This ensures that observations from the same engine do not appear in both training and evaluation sets.

---

## 🖥️ Streamlit Interactive Frontend Dashboard

Launch the interactive web UI with:

```bash
streamlit run app.py
```

### Dashboard Features:
1. **🚀 Live Engine Telemetry & Degradation Forecaster**:
   - Select any test engine (Engine 1 to 100) from the fleet.
   - Interactive slider to simulate flight cycle progression.
   - Real-time RUL prediction, Estimated Failure Cycle, and Ground Truth Error comparison.
   - Color-coded triage status badges: `HEALTHY (Green)`, `WARNING (Yellow)`, `CRITICAL (Red)`.
   - Dynamic Plotly degradation curves with risk zone threshold bands.
   - Key sensor trajectory subplots (LPC Outlet Temp, HPC Outlet Temp, Static Pressure Ps30, Bypass Ratio).
2. **📊 Model Comparison & Leaderboard**:
   - Benchmark leaderboard comparing Stacking Ensemble, Voting Ensemble, CatBoost, XGBoost, LightGBM, Random Forest, and Linear Regression.
   - Bar chart comparisons across RMSE, MAE, $R^2$, and NASA Score.
   - Official Test Actual vs. Predicted scatter plot with perfect agreement line.
3. **🔍 Explainable AI (Tree-SHAP) Insights**:
   - Beeswarm summary plots and mean $|SHAP|$ global feature importance rankings.
   - Physical domain interpretations of critical engine degradation drivers.
4. **📈 Sensor EDA & Health Trajectories**:
   - Run-to-failure lifespan histogram across all 100 engines.
   - Multi-engine sensor degradation comparisons.
   - Sensor correlation heatmaps and trend plots.
5. **⚙️ Pipeline Architecture & Specs**:
   - Visual flowchart of the 12-step zero-leakage workflow.
   - JSON viewer for model hyperparameters and metadata.

---

## 🔬 Rigorous Zero-Leakage Research Methodology

This project is built following strict industry and academic best practices:

```text
                               NASA C-MAPSS FD001
                                       │
                                       ▼
                              1. Raw Data Validation
                                       │
                                       ▼
                            2. Exploratory Data Analysis
                                       │
                                       ▼
                            3. Engine-Wise Data Split
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
               70% Engines        15% Engines        15% Engines
               Development         Validation        Internal Test
                  Train          (Tuning & Model     (Untouched until
                                    Selection)        Final Freeze)
                    │                  │
                    └────────┬─────────┘
                             ▼
                    Feature Engineering
                  (Strict past windows)
                             │
                             ▼
                     Baseline Models
                  (Mean & Linear Reg)
                             │
                             ▼
               Tree Models: RF, XGB, LGBM, CAT
                             │
                             ▼
                 Validation Leaderboard
                             │
                             ▼
               Optuna Hyperparameter Tuning
                             │
                             ▼
                       Select Best 3
                             │
                             ▼
                 Combined 85% Dev Data
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
         Voting Ensemble          Stacking Ensemble
                                (GroupKFold OOF + Ridge)
                │                         │
                └────────────┬────────────┘
                             ▼
                    Model Comparison on
                    15% Internal Test
                             │
                             ▼
                     Freeze Best Model
                             │
                             ▼
                 Official C-MAPSS Test Set
               (test_FD001.txt + RUL_FD001.txt)
                             │
                             ▼
                    Final Official Metrics
                             │
                             ▼
                      SHAP Attribution
                             │
                             ▼
                 Production Bundle & UI
```

---

## 🏆 Experimental Results & Leaderboards

### 1. Internal Test Set Leaderboard (15% Untouched Engines)

| Model Architecture | MAE (Cycles) | RMSE (Cycles) | $R^2$ Score | NASA Score |
| :--- | :---: | :---: | :---: | :---: |
| 🥇 **Stacking Ensemble (Winner)** | **7.01** | **10.09** | **0.9412** | **4,709.01** |
| 🥈 Voting Ensemble | 7.17 | 10.19 | 0.9400 | 4,622.99 |
| 🥉 Retrained CatBoost | 7.36 | 10.34 | 0.9383 | 4,790.29 |
| 4. Retrained XGBoost | 7.39 | 10.48 | 0.9366 | 4,921.02 |
| 5. Retrained LightGBM | 7.43 | 10.67 | 0.9343 | 5,076.35 |

---

### 2. Official NASA C-MAPSS FD001 Final Evaluation

| Metric | Official Score |
| :--- | :---: |
| **MAE** | **9.49 cycles** |
| **RMSE** | **12.90 cycles** |
| **$R^2$** | **0.9036** |
| **NASA Asymmetric Score** | **220.79** |

---

## 📂 Project Structure

```text
predictive-maintenance-rul/
│
├── app.py                    # Streamlit Interactive Web Application
│
├── data/
│   ├── raw/                  # Original NASA text files & documentation
│   ├── interim/              # Cleaned datasets
│   └── processed/            # Engineered features & train/val/test splits
│
├── logs/                     # Timestamped pipeline execution logs
│
├── src/
│   ├── config.py             # Central paths & hyperparameters
│   ├── logger.py             # Logging configuration
│   ├── exception.py          # CustomException with traceback details
│   ├── utils.py              # Serialization & file utilities
│   ├── data/                 # Data loader & integrity validation
│   ├── preprocessing/        # Sanitization & engine-wise partitioning
│   ├── features/             # Piecewise RUL, past rolling stats & trends
│   ├── models/               # Baseline, RF, XGBoost, LightGBM, CatBoost
│   ├── tuning/               # Optuna Bayesian hyperparameter optimization
│   ├── ensemble/             # Voting & GroupKFold OOF Stacking
│   ├── evaluation/           # Metrics & NASA PHM scoring
│   ├── explainability/       # Tree-SHAP feature attribution
│   ├── visualization/        # Publication-grade figures & residual plots
│   └── inference/            # Production engine inference class
│
├── models/
│   ├── individual/           # Tuned XGBoost, LightGBM, CatBoost
│   ├── ensemble/             # Voting & Stacking models
│   └── final/                # Winning model, feature config & metadata
│
├── results/
│   ├── metrics/              # CSV & JSON evaluation reports
│   ├── predictions/          # Prediction outputs
│   ├── figures/              # Visualizations
│   └── shap/                 # SHAP summary & bar plots
│
├── tests/                    # 15 Pytest unit tests
├── requirements.txt
├── README.md
├── conftest.py
├── .gitignore
└── main.py                   # Master entry point CLI
```

---

## 🚀 Quickstart & Commands

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Launch Streamlit Web UI
streamlit run app.py

# 3. Retrain Full Pipeline
python3 main.py

# 4. Run CLI Inference
python3 main.py --predict --engine-id 15

# 5. Run Unit Tests
pytest -v tests/
```
