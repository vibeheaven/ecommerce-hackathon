"""Tests for validation modules."""
import pytest
import pandas as pd
import numpy as np
from project.validation.validator import Validator
from project.validation.fold_metrics import compute_metrics, aggregate_fold_metrics


def test_compute_metrics():
    """Test standard metric computation."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])  # 1 TP, 1 FN, 2 TN, 0 FP

    metrics = compute_metrics(y_true, y_pred)
    assert "macro_f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 0.75


def test_aggregate_fold_metrics():
    """Test fold metric aggregation calculation."""
    fold_results = [
        {"macro_f1": 0.90, "precision": 0.88, "recall": 0.92, "accuracy": 0.91},
        {"macro_f1": 0.92, "precision": 0.90, "recall": 0.94, "accuracy": 0.93},
    ]

    summary = aggregate_fold_metrics(fold_results)
    assert summary["macro_f1_mean"] == pytest.approx(0.91)
    assert summary["macro_f1_worst"] == 0.90
    assert "macro_f1_std" in summary


def test_validator_oof_evaluation():
    """Test full validator OOF evaluation pipeline."""
    # Create mock OOF DataFrame
    oof_df = pd.DataFrame({
        "label": [1, 1, 0, 0, 1, 1, 0, 0],
        "fold": [0, 0, 0, 0, 1, 1, 1, 1],
        "negative_type": ["positive", "positive", "random", "same_category",
                         "positive", "positive", "random", "same_category"],
        "query": ["tişört", "tişört", "çanta", "pantolon", "ceket", "ceket", "çanta", "şapka"],
        "title": ["A", "B", "C", "D", "E", "F", "G", "H"],
    })

    # Mock predictions
    probabilities = np.array([0.9, 0.8, 0.1, 0.4, 0.95, 0.85, 0.2, 0.3])
    threshold = 0.5

    config = {"validation": {"n_folds": 2}}
    validator = Validator(config)

    results = validator.evaluate_oof(oof_df, probabilities, threshold)

    assert "overall_macro_f1" in results
    assert "macro_f1_mean" in results
    assert "macro_f1_worst" in results
    assert "tn_rate_random" in results
    assert "scenario_f1_easy_mix" in results
    assert results["threshold"] == threshold
