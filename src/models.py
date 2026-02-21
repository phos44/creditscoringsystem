import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def get_logistic_regression(c: float = 1.0, random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(C=c, random_state=random_state, max_iter=1000)


def get_random_forest(n_estimators: int = 100, max_depth: int = 10, random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)


def get_xgboost_classifier(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
    )


def get_lightgbm_classifier(
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        verbose=-1,
    )


def train_and_cross_validate(
    model: Any,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: List[str] = None,
) -> Tuple[Any, Dict[str, float], np.ndarray]:
    if scoring is None:
        scoring = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
    cv_metrics = {k: float(np.mean(v)) for k, v in scores.items() if k.startswith("test_")}
    model.fit(X, y)
    y_pred_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    logger.info("%s CV results: %s", model_name, cv_metrics)
    return model, cv_metrics, y_pred_proba


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    lr_c: float = 1.0,
    rf_n_estimators: int = 100,
    rf_max_depth: int = 10,
    xgb_n_estimators: int = 100,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.1,
    lgb_n_estimators: int = 100,
    lgb_max_depth: int = 6,
    lgb_learning_rate: float = 0.1,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    scoring = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    models = {}
    cv_results = {}
    cv_proba = {}
    lr = get_logistic_regression(c=lr_c, random_state=random_state)
    models["logistic_regression"], cv_results["logistic_regression"], cv_proba["logistic_regression"] = train_and_cross_validate(
        lr, "LogisticRegression", X_train, y_train, cv=cv_folds, scoring=scoring
    )
    rf = get_random_forest(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=random_state)
    models["random_forest"], cv_results["random_forest"], cv_proba["random_forest"] = train_and_cross_validate(
        rf, "RandomForest", X_train, y_train, cv=cv_folds, scoring=scoring
    )
    xgb_clf = get_xgboost_classifier(
        n_estimators=xgb_n_estimators, max_depth=xgb_max_depth, learning_rate=xgb_learning_rate, random_state=random_state
    )
    models["xgboost"], cv_results["xgboost"], cv_proba["xgboost"] = train_and_cross_validate(
        xgb_clf, "XGBoost", X_train, y_train, cv=cv_folds, scoring=scoring
    )
    lgb_clf = get_lightgbm_classifier(
        n_estimators=lgb_n_estimators, max_depth=lgb_max_depth, learning_rate=lgb_learning_rate, random_state=random_state
    )
    models["lightgbm"], cv_results["lightgbm"], cv_proba["lightgbm"] = train_and_cross_validate(
        lgb_clf, "LightGBM", X_train, y_train, cv=cv_folds, scoring=scoring
    )
    return models, cv_results, cv_proba


def get_feature_importance(model: Any, model_name: str, feature_names: List[str]) -> Dict[str, float]:
    if hasattr(model, "coef_") and model.coef_ is not None:
        imp = np.abs(model.coef_).ravel()
    elif hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        return {name: 0.0 for name in feature_names}
    n = min(len(imp), len(feature_names))
    return dict(zip(feature_names[:n], imp[:n].tolist()))


def save_best_model(models: Dict[str, Any], cv_results: Dict[str, Dict[str, float]], models_dir: Path, metric: str = "test_roc_auc") -> str:
    models_dir.mkdir(parents=True, exist_ok=True)
    best_name = max(cv_results.keys(), key=lambda k: cv_results[k].get(metric, 0))
    best_model = models[best_name]
    path = models_dir / "best_model.joblib"
    joblib.dump(best_model, path)
    logger.info("Saved best model (%s) to %s", best_name, path)
    return best_name
