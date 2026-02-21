import logging
import sys
from pathlib import Path

from src.config import Config
from src.eda import run_eda
from src.preprocessing import preprocess_test, preprocess_train
from src.models import train_all_models, get_feature_importance, save_best_model
from src.evaluation import evaluate_models, build_comparison_table, get_confusion_matrices
from src.visualization import plot_roc_curves, plot_pr_curves, plot_feature_importance, plot_confusion_matrices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def ensure_dirs(config: Config) -> None:
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    config.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    config.paths.models_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    config = Config()
    ensure_dirs(config)
    train_path = config.paths.data_dir / config.paths.train_file
    if not train_path.exists():
        logger.error("Train data not found at %s. Place cs-training.csv in data/", train_path)
        sys.exit(1)
    df, missing_report, outlier_report = run_eda(
        train_path,
        config.paths.figures_dir,
        target_column=config.data.target_column,
        drop_id=True,
    )
    X_train, y_train, feature_columns, scaler_ct = preprocess_train(
        df,
        config.data.target_column,
        id_column=config.data.id_column,
        missing_income_strategy=config.preprocessing.missing_income_strategy,
        missing_dependents_strategy=config.preprocessing.missing_dependents_strategy,
        lower_quantile=config.preprocessing.capping_quantile_low,
        upper_quantile=config.preprocessing.capping_quantile_high,
        scale_method=config.preprocessing.scale_method,
        balance_method=config.preprocessing.balance_method,
        random_state=config.data.random_state,
    )
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train, y_train, test_size=config.data.test_size, random_state=config.data.random_state, stratify=y_train
    )
    models, cv_results, _ = train_all_models(
        X_tr,
        y_tr,
        cv_folds=config.data.cv_folds,
        lr_c=config.model.logistic_regression_c,
        rf_n_estimators=config.model.random_forest_n_estimators,
        rf_max_depth=config.model.random_forest_max_depth,
        xgb_n_estimators=config.model.xgboost_n_estimators,
        xgb_max_depth=config.model.xgboost_max_depth,
        xgb_learning_rate=config.model.xgboost_learning_rate,
        lgb_n_estimators=config.model.lightgbm_n_estimators,
        lgb_max_depth=config.model.lightgbm_max_depth,
        lgb_learning_rate=config.model.lightgbm_learning_rate,
        random_state=config.data.random_state,
    )
    metrics_per_model, y_pred_proba_per_model, y_pred_per_model = evaluate_models(models, X_te, y_te)
    comparison_table = build_comparison_table(metrics_per_model)
    comparison_table.to_csv(config.paths.output_dir / "metrics_comparison.csv")
    feature_importance_per_model = {
        name: get_feature_importance(model, name, feature_columns) for name, model in models.items()
    }
    confusion_matrices = get_confusion_matrices(y_te, y_pred_per_model)
    plot_roc_curves(y_te, y_pred_proba_per_model, config.paths.figures_dir)
    plot_pr_curves(y_te, y_pred_proba_per_model, config.paths.figures_dir)
    plot_feature_importance(feature_importance_per_model, config.paths.figures_dir)
    plot_confusion_matrices(confusion_matrices, config.paths.figures_dir)
    best_name = save_best_model(models, cv_results, config.paths.models_dir, metric="test_roc_auc")
    try:
        import joblib
        joblib.dump(scaler_ct, config.paths.models_dir / "scaler.joblib")
        joblib.dump(feature_columns, config.paths.models_dir / "feature_columns.joblib")
    except Exception as e:
        logger.warning("Could not save scaler/features: %s", e)
    logger.info("Pipeline finished. Best model: %s", best_name)


if __name__ == "__main__":
    main()
