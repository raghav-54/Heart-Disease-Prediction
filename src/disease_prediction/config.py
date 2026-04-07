from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the training pipeline."""

    data_path: Path
    target_column: str = "target"
    test_size: float = 0.2
    random_state: int = 42
    logistic_thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6)
    artifacts_dir: Path = Path("artifacts")
    model_path: Path = Path("artifacts/model.pkl")
    report_path: Path = Path("artifacts/metrics.json")
    feature_importance_path: Path = Path("artifacts/feature_importance.csv")
    rf_param_grid: dict[str, list[int]] = field(
        default_factory=lambda: {
            "n_estimators": [100, 200],
            "max_depth": [3, 4, 5],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [2, 4],
        }
    )

