import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def drop_id_column(df: pd.DataFrame, id_column: str = "Unnamed: 0") -> pd.DataFrame:
    if id_column in df.columns:
        return df.drop(columns=[id_column], errors="ignore")
    return df


def fill_missing_values(
    df: pd.DataFrame,
    strategy_income: str = "median",
    strategy_dependents: str = "mode",
) -> pd.DataFrame:
    df = df.copy()
    if "MonthlyIncome" in df.columns and df["MonthlyIncome"].isnull().any():
        if strategy_income == "median":
            val = df["MonthlyIncome"].median()
        else:
            val = df["MonthlyIncome"].mean()
        df["MonthlyIncome"] = df["MonthlyIncome"].fillna(val)
        logger.info("Filled MonthlyIncome with %s: %s", strategy_income, val)
    if "NumberOfDependents" in df.columns and df["NumberOfDependents"].isnull().any():
        if strategy_dependents == "mode":
            val = df["NumberOfDependents"].mode()
            val = val[0] if len(val) > 0 else 0
        else:
            val = df["NumberOfDependents"].median()
        df["NumberOfDependents"] = df["NumberOfDependents"].fillna(val)
        logger.info("Filled NumberOfDependents with %s: %s", strategy_dependents, val)
    return df


def cap_outliers(
    series: pd.Series,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.Series:
    lower = series.quantile(lower_quantile)
    upper = series.quantile(upper_quantile)
    return series.clip(lower=lower, upper=upper)


def apply_capping(
    df: pd.DataFrame,
    numeric_columns: List[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = cap_outliers(df[col], lower_quantile, upper_quantile)
    logger.info("Applied capping to %s columns (quantiles %.2f, %.2f)", len(numeric_columns), lower_quantile, upper_quantile)
    return df


def get_feature_columns(df: pd.DataFrame, target_column: str, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = exclude or []
    candidates = [c for c in df.columns if c != target_column and c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return candidates


def build_scaler_pipeline(
    feature_columns: List[str],
    scale_method: str = "standard",
) -> ColumnTransformer:
    if scale_method == "standard":
        scaler = StandardScaler()
    else:
        scaler = StandardScaler()
    return ColumnTransformer(
        [("num", scaler, feature_columns)],
        remainder="passthrough",
    )


def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "smote",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=random_state, k_neighbors=5)
            X_res, y_res = sampler.fit_resample(X, y)
            logger.info("SMOTE: resampled to %s samples", len(y_res))
        except ImportError:
            logger.warning("imbalanced-learn not installed; skipping SMOTE")
            X_res, y_res = X, y
    elif method == "undersample":
        try:
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=random_state)
            X_res, y_res = sampler.fit_resample(X, y)
            logger.info("Undersample: resampled to %s samples", len(y_res))
        except ImportError:
            logger.warning("imbalanced-learn not installed; skipping undersample")
            X_res, y_res = X, y
    else:
        X_res, y_res = X, y
    return X_res, y_res


def preprocess_train(
    df: pd.DataFrame,
    target_column: str,
    id_column: str = "Unnamed: 0",
    missing_income_strategy: str = "median",
    missing_dependents_strategy: str = "mode",
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    scale_method: str = "standard",
    balance_method: str = "none",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], Any]:
    df = drop_id_column(df, id_column)
    df = fill_missing_values(df, missing_income_strategy, missing_dependents_strategy)
    feature_columns = get_feature_columns(df, target_column)
    df = apply_capping(df, feature_columns, lower_quantile, upper_quantile)
    X = df[feature_columns].values
    y = df[target_column].values
    if balance_method and balance_method != "none":
        X, y = balance_classes(X, y, balance_method, random_state)
    scaler_ct = build_scaler_pipeline(feature_columns, scale_method)
    X_scaled = scaler_ct.fit_transform(X)
    logger.info("Preprocessing train: X %s, y %s, features %s", X_scaled.shape, y.shape, feature_columns)
    return X_scaled, y, feature_columns, scaler_ct


def preprocess_test(
    df: pd.DataFrame,
    feature_columns: List[str],
    scaler_ct: Any,
    target_column: Optional[str] = None,
    id_column: str = "Unnamed: 0",
    missing_income_strategy: str = "median",
    missing_dependents_strategy: str = "mode",
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    df = drop_id_column(df, id_column)
    df = fill_missing_values(df, missing_income_strategy, missing_dependents_strategy)
    df = apply_capping(df, feature_columns, lower_quantile, upper_quantile)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_columns].values
    X_scaled = scaler_ct.transform(X)
    y = df[target_column].values if target_column and target_column in df.columns else None
    logger.info("Preprocessing test: X %s", X_scaled.shape)
    return X_scaled, y
