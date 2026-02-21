import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def load_data(data_path: Path, drop_id: bool = True) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        if drop_id and "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"], errors="ignore")
        logger.info("Data loaded: %s rows, %s columns", len(df), len(df.columns))
        return df
    except Exception as e:
        logger.error("Failed to load data from %s: %s", data_path, e)
        raise


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    result = pd.DataFrame({"count": missing, "percent": missing_pct})
    result = result[result["count"] > 0].sort_values("count", ascending=False)
    logger.info("Missing values: %s", result.to_dict() if len(result) > 0 else "none")
    return result


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> Tuple[int, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    count = ((series < lower) | (series > upper)).sum()
    pct = count / len(series) * 100 if len(series) > 0 else 0
    return int(count), round(pct, 2)


def analyze_outliers(df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    results = []
    for col in numeric_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            count, pct = detect_outliers_iqr(df[col].dropna())
            results.append({"column": col, "outlier_count": count, "outlier_percent": pct})
    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.sort_values("outlier_count", ascending=False)
    logger.info("Outlier analysis completed for %s columns", len(result_df))
    return result_df


def plot_distributions(
    df: pd.DataFrame,
    numeric_columns: list,
    target_column: str,
    figures_dir: Path,
    ncols: int = 3,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    nrows = (len(numeric_columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)
    for idx, col in enumerate(numeric_columns):
        if col not in df.columns:
            continue
        row, col_idx = idx // ncols, idx % ncols
        ax = axes[row, col_idx]
        df[col].dropna().hist(ax=ax, bins=50, edgecolor="black", alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel("")
    for idx in range(len(numeric_columns), axes.size):
        row, col_idx = idx // ncols, idx % ncols
        axes[row, col_idx].set_visible(False)
    plt.tight_layout()
    out_path = figures_dir / "feature_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)

    if target_column in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[target_column].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Target distribution (SeriousDlqin2yrs)")
        ax.set_xlabel("Class")
        out_path_target = figures_dir / "target_distribution.png"
        plt.savefig(out_path_target, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", out_path_target)


def plot_correlation_heatmap(df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        logger.warning("Not enough numeric columns for correlation heatmap")
        return
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    plt.tight_layout()
    out_path = figures_dir / "correlation_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out_path)


def run_eda(
    data_path: Path,
    figures_dir: Path,
    target_column: str = "SeriousDlqin2yrs",
    drop_id: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_data(data_path, drop_id=drop_id)
    missing_report = analyze_missing_values(df)
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
    outlier_report = analyze_outliers(df, numeric_cols)
    plot_distributions(df, numeric_cols + ([target_column] if target_column in df.columns else []), target_column, figures_dir)
    plot_correlation_heatmap(df, figures_dir)
    return df, missing_report, outlier_report
