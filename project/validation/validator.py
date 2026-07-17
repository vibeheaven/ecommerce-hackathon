"""
Validator — Main validation orchestrator.
Computes OOF metrics (with confusion matrix and per-class breakdown),
balanced scenario scores, and the negative-type report.
"""
import pandas as pd
import numpy as np
from typing import Any

from project.utils.logging_utils import setup_logger
from project.validation.fold_metrics import compute_metrics, aggregate_fold_metrics
from project.validation.negative_type_report import generate_negative_type_report
from project.validation.scenario_builder import ScenarioBuilder

logger = setup_logger("validator")


class Validator:
    """Main orchestrator for validation and threshold optimization."""

    def __init__(self, config: dict):
        self.config = config
        seed = int(config.get("project", {}).get("seed", 42))
        self.scenario_builder = ScenarioBuilder(seed=seed)

    def evaluate_oof(
        self,
        oof_df: pd.DataFrame,
        probabilities: np.ndarray,
        threshold: float,
    ) -> dict[str, Any]:
        """
        Evaluate out-of-fold predictions.

        Args:
            oof_df: DataFrame containing 'label', 'fold', and 'negative_type'
            probabilities: array of probabilities predicted by model (aligned to oof_df rows)
            threshold: classification threshold

        Returns:
            dict containing all metrics
        """
        if len(oof_df) != len(probabilities):
            raise ValueError(
                f"oof_df ({len(oof_df):,} rows) and probabilities ({len(probabilities):,}) "
                "are misaligned — refusing to evaluate."
            )

        logger.info("Evaluating Out-of-Fold predictions...")
        predictions = (probabilities >= threshold).astype(int)

        # 1. Overall metrics (with confusion matrix + per-class breakdown)
        overall = compute_metrics(oof_df["label"], predictions, include_confusion=True)
        logger.info(f"  Overall Macro F1: {overall['macro_f1']:.4f}")
        logger.info(f"  Precision:        {overall['precision']:.4f}")
        logger.info(f"  Recall:           {overall['recall']:.4f}")
        logger.info(f"  Accuracy:         {overall['accuracy']:.4f}")
        logger.info(
            f"  Confusion — TN: {overall['confusion_tn']:,}  FP: {overall['confusion_fp']:,}  "
            f"FN: {overall['confusion_fn']:,}  TP: {overall['confusion_tp']:,}"
        )
        logger.info(
            f"  Positive class — P: {overall['positive_precision']:.4f} "
            f"R: {overall['positive_recall']:.4f} F1: {overall['positive_f1']:.4f}"
        )
        logger.info(
            f"  Negative class — P: {overall['negative_precision']:.4f} "
            f"R: {overall['negative_recall']:.4f} F1: {overall['negative_f1']:.4f}"
        )

        # 2. Metrics by fold
        fold_results = []
        for f in sorted(oof_df["fold"].unique()):
            fold_mask = (oof_df["fold"] == f).to_numpy()
            f_metrics = compute_metrics(oof_df.loc[fold_mask, "label"], predictions[fold_mask])
            fold_results.append(f_metrics)
            logger.info(f"    Fold {f} Macro F1: {f_metrics['macro_f1']:.4f}")

        aggregated = aggregate_fold_metrics(fold_results)

        # 3. Negative type performance report
        neg_report = generate_negative_type_report(
            y_true=oof_df["label"],
            y_pred=predictions,
            negative_types=oof_df["negative_type"],
        )

        # 4. Multi-scenario evaluation (balanced)
        logger.info("Building and evaluating scenarios...")
        val_df_with_preds = oof_df.copy()
        val_df_with_preds["pred_prob"] = probabilities
        val_df_with_preds["pred_label"] = predictions

        scenarios = self.scenario_builder.build_scenarios(val_df_with_preds)
        scenario_scores = {}
        for name, sc_df in scenarios.items():
            sc_metrics = compute_metrics(sc_df["label"], sc_df["pred_label"])
            scenario_scores[f"scenario_f1_{name}"] = sc_metrics["macro_f1"]
            logger.info(f"    Scenario '{name}' F1: {sc_metrics['macro_f1']:.4f}")

        results = {
            "overall_macro_f1": overall["macro_f1"],
            "overall_precision": overall["precision"],
            "overall_recall": overall["recall"],
            "overall_accuracy": overall["accuracy"],
            "positive_f1": overall["positive_f1"],
            "negative_f1": overall["negative_f1"],
            "positive_precision": overall["positive_precision"],
            "positive_recall": overall["positive_recall"],
            "negative_precision": overall["negative_precision"],
            "negative_recall": overall["negative_recall"],
            "confusion_tn": overall["confusion_tn"],
            "confusion_fp": overall["confusion_fp"],
            "confusion_fn": overall["confusion_fn"],
            "confusion_tp": overall["confusion_tp"],
            **aggregated,
            **neg_report,
            **scenario_scores,
            "threshold": threshold,
        }

        return results
