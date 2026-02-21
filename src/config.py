from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PathConfig:
    data_dir: Path = field(default_factory=lambda: Path("data"))
    train_file: str = "cs-training.csv"
    test_file: str = "cs-test.csv"
    output_dir: Path = field(default_factory=lambda: Path("output"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    figures_dir: Path = field(default_factory=lambda: Path("output/figures"))

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.models_dir = Path(self.models_dir)
        self.figures_dir = Path(self.figures_dir)


@dataclass
class DataConfig:
    target_column: str = "SeriousDlqin2yrs"
    id_column: str = "Unnamed: 0"
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5


@dataclass
class PreprocessingConfig:
    missing_income_strategy: str = "median"
    missing_dependents_strategy: str = "mode"
    capping_quantile_low: float = 0.01
    capping_quantile_high: float = 0.99
    scale_method: str = "standard"
    balance_method: str = "none"


@dataclass
class ModelConfig:
    logistic_regression_c: float = 1.0
    random_forest_n_estimators: int = 100
    random_forest_max_depth: int = 10
    xgboost_n_estimators: int = 100
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.1
    lightgbm_n_estimators: int = 100
    lightgbm_max_depth: int = 6
    lightgbm_learning_rate: float = 0.1


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
