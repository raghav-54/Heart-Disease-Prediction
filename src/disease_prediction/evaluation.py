from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    """Return classification report as a dictionary."""
    return classification_report(y_true, y_pred, output_dict=True)


def threshold_metrics(
    y_true: pd.Series, y_probs: np.ndarray, thresholds: tuple[float, ...]
) -> dict[str, dict[str, Any]]:
    """Evaluate predictions across probability thresholds."""
    metrics: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        metrics[str(threshold)] = classification_metrics(y_true, y_pred)
    return metrics


def confusion(y_true: pd.Series, y_pred: np.ndarray) -> list[list[int]]:
    """Return confusion matrix as serializable nested list."""
    return confusion_matrix(y_true, y_pred).tolist()


def roc_auc(y_true: pd.Series, y_probs: np.ndarray) -> float:
    """Compute ROC-AUC score."""
    return float(roc_auc_score(y_true, y_probs))

