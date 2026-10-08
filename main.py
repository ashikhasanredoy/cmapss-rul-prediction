import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    RUL_FILE,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_INDIVIDUAL_DIR,
    MODELS_ENSEMBLE_DIR,
    MODELS_FINAL_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    FIGURES_DIR,
    SHAP_DIR
)
from src.logger import logger
from src.utils import save_object, save_json, save_dataframe
from src.data.data_loader import load_train_data, load_official_test_data, load_official_rul_truth
from src.data.validation import validate_dataset, print_validation_report
from src.preprocessing.cleaning import clean_dataframe
from src.preprocessing.splitting import split_engines
from src.features.builder import (
    build_feature_dataframe,
    extract_features_and_target,
    extract_official_test_features
)
from src.models.baseline import get_mean_baseline_model, get_linear_regression_model
from src.models.random_forest import get_random_forest_model
from src.models.xgboost_model import get_xgboost_model
from src.models.lightgbm_model import get_lightgbm_model
from src.models.catboost_model import get_catboost_model
from src.tuning.optuna_tuning import tune_xgboost_optuna, tune_lightgbm_optuna, tune_catboost_optuna
from src.ensemble.voting import create_voting_ensemble
from src.ensemble.stacking import EngineGroupedStackingRegressor
from src.evaluation.evaluator import evaluate_model, create_leaderboard
from src.visualization.eda import plot_rul_distribution, plot_correlation_matrix, plot_sensor_trends
from src.visualization.predictions import (
    plot_actual_vs_predicted,
    plot_residual_analysis,
    plot_feature_importance,
    plot_model_comparison_bar
)
from src.explainability.shap_analysis import generate_shap_explanations
from src.inference.predict import RULPredictor


def prepare_datasets():
    train_raw = load_train_data(TRAIN_FILE)
    validation_report = validate_dataset(train_raw, 'Raw Train FD001')
    print_validation_report(validation_report)

    train_clean = clean_dataframe(train_raw)
    save_dataframe(INTERIM_DATA_DIR / 'train_cleaned.csv', train_clean)

    train_df, val_df, internal_test_df, dev_df = split_engines(train_clean)

    train_feat = build_feature_dataframe(train_df, is_train=True)
    val_feat = build_feature_dataframe(val_df, is_train=True)
    internal_test_feat = build_feature_dataframe(internal_test_df, is_train=True)
    dev_feat = build_feature_dataframe(dev_df, is_train=True)

    plot_rul_distribution(train_feat)
    plot_correlation_matrix(train_feat)
    plot_sensor_trends(train_clean, engine_ids=[1, 2, 3])

    X_train, y_train, train_groups = extract_features_and_target(train_feat)
    X_val, y_val, val_groups = extract_features_and_target(val_feat)
    X_internal_test, y_internal_test, test_groups = extract_features_and_target(internal_test_feat)
    X_dev, y_dev, dev_groups = extract_features_and_target(dev_feat)

    feature_names = list(X_train.columns)
    logger.info(f"Engineered {len(feature_names)} features per cycle")

    save_dataframe(PROCESSED_DATA_DIR / 'X_train.csv', X_train)
    save_dataframe(PROCESSED_DATA_DIR / 'X_val.csv', X_val)
    save_dataframe(PROCESSED_DATA_DIR / 'X_internal_test.csv', X_internal_test)
    save_dataframe(PROCESSED_DATA_DIR / 'X_dev.csv', X_dev)
    save_dataframe(PROCESSED_DATA_DIR / 'y_train.csv', y_train)
    save_dataframe(PROCESSED_DATA_DIR / 'y_val.csv', y_val)
    save_dataframe(PROCESSED_DATA_DIR / 'y_internal_test.csv', y_internal_test)
    save_dataframe(PROCESSED_DATA_DIR / 'y_dev.csv', y_dev)

    return (
        X_train, y_train,
        X_val, y_val,
        X_internal_test, y_internal_test,
        X_dev, y_dev, dev_groups,
        feature_names
    )


def train_and_select_model(X_train, y_train, X_val, y_val, X_internal_test, y_internal_test, X_dev, y_dev, dev_groups):
    candidate_models = {
        'Mean Baseline': get_mean_baseline_model(),
        'Linear Regression': get_linear_regression_model(),
        'Random Forest': get_random_forest_model(),
        'XGBoost (Default)': get_xgboost_model(),
        'LightGBM (Default)': get_lightgbm_model(),
        'CatBoost (Default)': get_catboost_model()
    }

    initial_val_evals = []
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        initial_val_evals.append(evaluate_model(y_val, val_pred, model_name=name))

    val_leaderboard = create_leaderboard(initial_val_evals)
    logger.info(f"Validation Leaderboard:\n{val_leaderboard.to_string(index=False)}")

    best_xgb_model, xgb_params = tune_xgboost_optuna(X_train, y_train, X_val, y_val, n_trials=15)
    best_lgb_model, lgb_params = tune_lightgbm_optuna(X_train, y_train, X_val, y_val, n_trials=15)
    best_cat_model, cat_params = tune_catboost_optuna(X_train, y_train, X_val, y_val, n_trials=12)

    save_object(MODELS_INDIVIDUAL_DIR / 'xgboost.pkl', best_xgb_model)
    save_object(MODELS_INDIVIDUAL_DIR / 'lightgbm.pkl', best_lgb_model)
    save_object(MODELS_INDIVIDUAL_DIR / 'catboost.pkl', best_cat_model)

    retrained_xgb = get_xgboost_model(**xgb_params).fit(X_dev, y_dev)
    retrained_lgb = get_lightgbm_model(**lgb_params).fit(X_dev, y_dev)
    retrained_cat = get_catboost_model(**cat_params).fit(X_dev, y_dev)

    voting_ensemble = create_voting_ensemble([
        ('xgb', retrained_xgb),
        ('lgb', retrained_lgb),
        ('cat', retrained_cat)
    ]).fit(X_dev, y_dev)

    stacking_ensemble = EngineGroupedStackingRegressor(
        base_models=[
            ('xgb', get_xgboost_model(**xgb_params)),
            ('lgb', get_lightgbm_model(**lgb_params)),
            ('cat', get_catboost_model(**cat_params))
        ]
    ).fit(X_dev, y_dev, groups=dev_groups, n_splits=5)

    save_object(MODELS_ENSEMBLE_DIR / 'voting.pkl', voting_ensemble)
    save_object(MODELS_ENSEMBLE_DIR / 'stacking.pkl', stacking_ensemble)

    final_candidates = {
        'Retrained XGBoost': retrained_xgb,
        'Retrained LightGBM': retrained_lgb,
        'Retrained CatBoost': retrained_cat,
        'Voting Ensemble': voting_ensemble,
        'Stacking Ensemble': stacking_ensemble
    }

    internal_test_evals = [
        evaluate_model(y_internal_test, model.predict(X_internal_test), model_name=name)
        for name, model in final_candidates.items()
    ]
    internal_test_leaderboard = create_leaderboard(internal_test_evals)
    logger.info(f"Internal Test Leaderboard:\n{internal_test_leaderboard.to_string(index=False)}")
    save_dataframe(METRICS_DIR / 'internal_test_metrics.csv', internal_test_leaderboard)

    best_model_name = internal_test_leaderboard.iloc[0]['Model']
    best_model_obj = final_candidates[best_model_name]
    logger.info(f"Champion Architecture: {best_model_name}")

    return best_model_obj, best_model_name, internal_test_leaderboard, retrained_xgb, (xgb_params, lgb_params, cat_params)


def evaluate_and_save_artifacts(best_model_obj, best_model_name, internal_test_leaderboard, X_internal_test, y_internal_test, retrained_xgb, tuned_params, feature_names):
    xgb_params, lgb_params, cat_params = tuned_params

    official_test_raw = load_official_test_data(TEST_FILE)
    official_rul_truth = load_official_rul_truth(RUL_FILE)
    official_test_clean = clean_dataframe(official_test_raw)
    official_test_feat = build_feature_dataframe(official_test_clean, is_train=False)

    X_test_last, test_engine_ids = extract_official_test_features(official_test_feat)
    y_test_pred = best_model_obj.predict(X_test_last)
    y_test_true = official_rul_truth['RUL'].values

    official_metrics = evaluate_model(y_test_true, y_test_pred, model_name=f"{best_model_name} (Official Test)")
    logger.info(f"Official Test Score: RMSE={official_metrics['RMSE']}, MAE={official_metrics['MAE']}, R2={official_metrics['R2']}, NASA Score={official_metrics['NASA_Score']}")

    internal_preds = pd.DataFrame({
        'actual_RUL': y_internal_test.values,
        'predicted_RUL': np.clip(best_model_obj.predict(X_internal_test), 0, None).round(2)
    })
    save_dataframe(PREDICTIONS_DIR / 'internal_test_predictions.csv', internal_preds)

    official_preds = pd.DataFrame({
        'engine_id': test_engine_ids,
        'actual_RUL': y_test_true,
        'predicted_RUL': np.clip(y_test_pred, 0, None).round(2)
    })
    save_dataframe(PREDICTIONS_DIR / 'official_test_predictions.csv', official_preds)

    plot_actual_vs_predicted(y_test_true, y_test_pred, title=f'Official Test: Actual vs Predicted RUL ({best_model_name})')
    plot_residual_analysis(y_test_true, y_test_pred)
    plot_model_comparison_bar(internal_test_leaderboard)

    if hasattr(best_model_obj, 'feature_importances_'):
        plot_feature_importance(feature_names, best_model_obj.feature_importances_)
    elif hasattr(retrained_xgb, 'feature_importances_'):
        plot_feature_importance(feature_names, retrained_xgb.feature_importances_)

    try:
        sample_idx = np.random.choice(X_internal_test.index, size=min(150, len(X_internal_test)), replace=False)
        shap_model = retrained_xgb if not hasattr(best_model_obj, 'feature_importances_') else best_model_obj
        generate_shap_explanations(shap_model, X_internal_test.loc[sample_idx])
    except Exception as e:
        logger.warning(f"SHAP explanation note: {e}")

    save_object(MODELS_FINAL_DIR / 'model.pkl', best_model_obj)

    feature_config = {
        'num_features': len(feature_names),
        'feature_names': feature_names,
        'rolling_windows': [5, 10, 20],
        'piecewise_rul_limit': 125
    }
    save_json(MODELS_FINAL_DIR / 'feature_config.json', feature_config)

    metadata = {
        'final_model_architecture': best_model_name,
        'dataset': 'NASA C-MAPSS FD001',
        'split_methodology': '70% Train / 15% Val / 15% Internal Test',
        'internal_test_metrics': internal_test_leaderboard.to_dict(orient='records'),
        'official_c_mapss_test_metrics': official_metrics,
        'tuned_hyperparameters': {
            'xgboost': xgb_params,
            'lightgbm': lgb_params,
            'catboost': cat_params
        }
    }
    save_json(MODELS_FINAL_DIR / 'metadata.json', metadata)
    save_json(METRICS_DIR / 'final_evaluation_metrics.json', metadata)


def run_pipeline():
    logger.info("Starting C-MAPSS FD001 RUL training pipeline")

    (
        X_train, y_train,
        X_val, y_val,
        X_internal_test, y_internal_test,
        X_dev, y_dev, dev_groups,
        feature_names
    ) = prepare_datasets()

    (
        best_model_obj,
        best_model_name,
        internal_test_leaderboard,
        retrained_xgb,
        tuned_params
    ) = train_and_select_model(
        X_train, y_train,
        X_val, y_val,
        X_internal_test, y_internal_test,
        X_dev, y_dev, dev_groups
    )

    evaluate_and_save_artifacts(
        best_model_obj,
        best_model_name,
        internal_test_leaderboard,
        X_internal_test,
        y_internal_test,
        retrained_xgb,
        tuned_params,
        feature_names
    )

    logger.info("Pipeline run finished successfully")


def main():
    parser = argparse.ArgumentParser(description='NASA Turbofan C-MAPSS RUL Prediction System')
    parser.add_argument('--train', action='store_true', help='Run full training pipeline')
    parser.add_argument('--predict', action='store_true', help='Run inference on an engine from the test set')
    parser.add_argument('--engine-id', type=int, default=1, help='Engine ID for inference (default: 1)')
    args = parser.parse_args()

    if args.predict:
        test_df = load_official_test_data()
        engine_df = test_df[test_df['engine_id'] == args.engine_id]

        if engine_df.empty:
            logger.error(f"Engine ID {args.engine_id} not found")
            sys.exit(1)

        predictor = RULPredictor()
        result = predictor.predict_engine(engine_df)

        logger.info(
            f"Engine #{args.engine_id} Prediction: RUL={result['predicted_rul_cycles']} cycles "
            f"(Cycle {result['current_cycle']} -> Est. Failure Cycle {result['estimated_failure_cycle']}) | "
            f"Status={result['health_status']}"
        )
    else:
        run_pipeline()


if __name__ == '__main__':
    main()
