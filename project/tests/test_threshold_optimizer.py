"""Tests for threshold_optimizer module."""
import pytest
import numpy as np
from project.training.threshold_optimizer import ThresholdOptimizer


def test_threshold_optimizer_search():
    """Test coarse and fine search logic on a clean distribution."""
    # Create simple probabilities and labels
    # Label is 1 for probabilities >= 0.58, 0 otherwise
    probabilities = np.array([0.1, 0.2, 0.3, 0.4, 0.57, 0.59, 0.7, 0.8, 0.9, 0.95])
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    folds = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    config = {
        "threshold": {
            "coarse_range": [0.20, 0.80],
            "coarse_step": 0.01,
            "fine_range_delta": 0.03,
            "fine_step": 0.001
        }
    }

    optimizer = ThresholdOptimizer(config)
    results = optimizer.optimize(y_true, probabilities, folds=folds)

    assert "best_threshold" in results
    assert "best_f1" in results
    assert results["best_f1"] == 1.0  # Perfect separation possible
    # Threshold should sit exactly between 0.57 and 0.59
    assert 0.57 < results["best_threshold"] < 0.59
    assert len(results["fold_thresholds"]) == 2
