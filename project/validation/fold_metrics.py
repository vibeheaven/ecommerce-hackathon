"""
Fold Metrics — Computes classification metrics (Macro F1, Precision, Recall)
by fold and calculates summary statistics (mean, std, worst fold).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute binary classification metrics with macro F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    # Class-specific F1s
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )

    return {
        "macro_f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": acc,
        "negative_f1": f1_class[0],
        "positive_f1": f1_class[1],
    }


def aggregate_fold_metrics(
    fold_results: list[dict[str, float]]
) -> dict[str, float]:
    """
    Aggregate metrics from multiple validation folds.
    Computes mean, standard deviation, and the minimum (worst fold) score.
    """
    metrics = ["macro_f1", "precision", "recall", "accuracy", "negative_f1", "positive_f1"]
    summary = {}

    for metric in metrics:
        values = [res[metric] for res in fold_results if metric in res]
        if not values:
            continue
        summary[f"{metric}_mean"] = np.mean(values)
        summary[f"{metric}_std"] = np.std(values)
        summary[f"{metric}_worst"] = np.min(values)

    return summary
