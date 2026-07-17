"""
Threshold Optimizer — Grid search for the optimal classification threshold.
Performs coarse and fine grid search on Out-of-Fold (OOF) prediction probabilities.
"""
import numpy as np
import pandas as pd
from typing import Any

from project.utils.logging_utils import setup_logger
from project.validation.fold_metrics import compute_metrics

logger = setup_logger("threshold_optimizer")


class ThresholdOptimizer:
    """Optimizes classification threshold on OOF probabilities to maximize Macro F1."""

    def __init__(self, config: dict):
        self.config = config
        thresh_cfg = config.get("threshold", {})
        self.coarse_range = thresh_cfg.get("coarse_range", [0.20, 0.80])
        self.coarse_step = thresh_cfg.get("coarse_step", 0.01)
        self.fine_delta = thresh_cfg.get("fine_range_delta", 0.03)
        self.fine_step = thresh_cfg.get("fine_step", 0.001)

    def optimize(
        self,
        y_true: np.ndarray | pd.Series,
        probabilities: np.ndarray,
        folds: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Run coarse and fine grid search on OOF predictions.
        Also calculates fold-specific thresholds if fold array is provided.
        """
        logger.info("Starting threshold optimization grid search...")

        # 1. Coarse search
        coarse_thresholds = np.arange(self.coarse_range[0], self.coarse_range[1] + 1e-9, self.coarse_step)
        best_coarse_f1 = -1.0
        best_coarse_thresh = 0.50

        for t in coarse_thresholds:
            y_pred = (probabilities >= t).astype(int)
            res = compute_metrics(y_true, y_pred)
            f1 = res["macro_f1"]
            if f1 > best_coarse_f1:
                best_coarse_f1 = f1
                best_coarse_thresh = t

        logger.info(f"  Coarse search: best F1 = {best_coarse_f1:.4f} at threshold {best_coarse_thresh:.3f}")

        # 2. Fine search around best coarse threshold
        fine_start = max(self.coarse_range[0], best_coarse_thresh - self.fine_delta)
        fine_end = min(self.coarse_range[1], best_coarse_thresh + self.fine_delta)
        fine_thresholds = np.arange(fine_start, fine_end + 1e-9, self.fine_step)

        best_f1 = best_coarse_f1
        best_thresh = best_coarse_thresh

        for t in fine_thresholds:
            y_pred = (probabilities >= t).astype(int)
            res = compute_metrics(y_true, y_pred)
            f1 = res["macro_f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

        logger.info(f"  Fine search:   best F1 = {best_f1:.4f} at threshold {best_thresh:.4f}")

        # 3. Calculate fold-specific thresholds (to check stability/calibration)
        fold_thresholds = {}
        if folds is not None:
            unique_folds = np.unique(folds)
            for f in unique_folds:
                fold_mask = folds == f
                f_y_true = y_true[fold_mask] if isinstance(y_true, np.ndarray) else y_true.iloc[fold_mask]
                f_probs = probabilities[fold_mask]

                # Run simple coarse search for fold
                f_best_f1 = -1.0
                f_best_thresh = 0.50
                for t in coarse_thresholds:
                    f_y_pred = (f_probs >= t).astype(int)
                    f_f1 = compute_metrics(f_y_true, f_y_pred)["macro_f1"]
                    if f_f1 > f_best_f1:
                        f_best_f1 = f_f1
                        f_best_thresh = t
                fold_thresholds[int(f)] = f_best_thresh

            # Check standard deviation of fold thresholds
            f_thresh_vals = list(fold_thresholds.values())
            std_dev = np.std(f_thresh_vals)
            logger.info(f"  Fold thresholds: {fold_thresholds} (std: {std_dev:.4f})")
            if std_dev > 0.05:
                logger.warning("  [WARNING] High variance in fold thresholds! Check model calibration or negative sample distribution.")

        return {
            "best_threshold": float(best_thresh),
            "best_f1": float(best_f1),
            "fold_thresholds": fold_thresholds,
            "threshold_std": float(np.std(list(fold_thresholds.values()))) if fold_thresholds else 0.0,
        }
