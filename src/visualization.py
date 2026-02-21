import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba_per_model: Dict[str, np.ndarray],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_proba in y_pred_proba_per_model.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = figures_dir / "roc_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


def plot_pr_curves(
    y_true: np.ndarray,
    y_pred_proba_per_model: Dict[str, np.ndarray],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_proba in y_pred_proba_per_model.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = figures_dir / "precision_recall_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


def plot_feature_importance(
    feature_importance_per_model: Dict[str, Dict[str, float]],
    figures_dir: Path,
    top_n: int = 15,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for model_name, importance_dict in feature_importance_per_model.items():
        if not importance_dict:
            continue
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names = [x[0] for x in sorted_items]
        values = [x[1] for x in sorted_items]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(names)), values, align="center")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature importance: {model_name}")
        plt.tight_layout()
        safe_name = model_name.replace(" ", "_").lower()
        out_path = figures_dir / f"feature_importance_{safe_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", out_path)


def plot_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    n = len(confusion_matrices)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, square=True)
        ax.set_title(f"Confusion matrix: {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for idx in range(n, axes.size):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)
    plt.tight_layout()
    out_path = figures_dir / "confusion_matrices.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)
