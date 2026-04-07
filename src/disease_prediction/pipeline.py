import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from disease_prediction.config import PipelineConfig
from disease_prediction.data import load_dataset
from disease_prediction.evaluation import (
    classification_metrics,
    confusion,
    roc_auc,
    threshold_metrics,
)
from disease_prediction.features import engineer_features
from disease_prediction.models import (
    train_decision_tree,
    train_logistic_regression,
    tune_random_forest,
)


def _prepare_splits(
    df: pd.DataFrame, target_column: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def _to_json_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_training_pipeline(config: PipelineConfig) -> dict[str, Any]:
    df = load_dataset(config.data_path)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = _prepare_splits(
        df, config.target_column, config.test_size, config.random_state
    )

    logistic_model, scaler = train_logistic_regression(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    logistic_pred = logistic_model.predict(X_test_scaled)
    logistic_probs = logistic_model.predict_proba(X_test_scaled)[:, 1]

    decision_tree = train_decision_tree(X_train, y_train)
    dt_pred = decision_tree.predict(X_test)

    best_rf = tune_random_forest(X_train, y_train, config.rf_param_grid)
    rf_pred = best_rf.predict(X_test)
    rf_probs = best_rf.predict_proba(X_test)[:, 1]

    feature_importance = (
        pd.Series(best_rf.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
        .rename("importance")
    )

    metrics: dict[str, Any] = {
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target_distribution": df[config.target_column].value_counts(normalize=True).to_dict(),
        },
        "logistic_regression": {
            "classification_report": classification_metrics(y_test, logistic_pred),
            "threshold_analysis": threshold_metrics(
                y_test, logistic_probs, config.logistic_thresholds
            ),
        },
        "decision_tree": {
            "classification_report": classification_metrics(y_test, dt_pred),
        },
        "random_forest_tuned": {
            "classification_report": classification_metrics(y_test, rf_pred),
            "confusion_matrix": confusion(y_test, rf_pred),
            "roc_auc": roc_auc(y_test, rf_probs),
        },
    }

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_rf, config.model_path)
    feature_importance.to_csv(config.feature_importance_path, header=True)
    _to_json_file(config.report_path, metrics)

    return metrics

