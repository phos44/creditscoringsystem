import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_pr_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    try:
        from sklearn.metrics import auc
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        return float(auc(recall, precision))
    except Exception:
        return 0.0


def evaluate_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    metrics_per_model = {}
    y_pred_proba_per_model = {}
    y_pred_per_model = {}
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        metrics_per_model[name] = compute_metrics(y_test, y_pred, y_pred_proba)
        pr_auc = compute_pr_auc(y_test, y_pred_proba)
        metrics_per_model[name]["pr_auc"] = pr_auc
        y_pred_proba_per_model[name] = y_pred_proba
        y_pred_per_model[name] = y_pred
    return metrics_per_model, y_pred_proba_per_model, y_pred_per_model


def build_comparison_table(metrics_per_model: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for model_name, m in metrics_per_model.items():
        row = {"model": model_name, **m}
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.set_index("model")
    logger.info("Comparison table:\n%s", df.to_string())
    return df


def get_confusion_matrices(y_test: np.ndarray, y_pred_per_model: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: confusion_matrix(y_test, y_pred) for name, y_pred in y_pred_per_model.items()}
