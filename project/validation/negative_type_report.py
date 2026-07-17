"""
Negative Type Report — Evaluates model classification success by negative type.
Helps detect exactly which negative types (random, same-category, lexical hard, etc.)
the model is struggling to filter out.
"""
import pandas as pd
from typing import Any

from project.utils.logging_utils import setup_logger

logger = setup_logger("negative_type_report")


def generate_negative_type_report(
    y_true: pd.Series | list[int],
    y_pred: pd.Series | list[int],
    negative_types: pd.Series | list[str],
) -> dict[str, float]:
    """
    Compute accuracy (True Negative rate) for each negative type.
    Since all true negatives have label = 0, accuracy is the fraction of
    negatives correctly classified as 0.
    """
    df = pd.DataFrame({
        "label": y_true,
        "prediction": y_pred,
        "type": negative_types,
    })

    # Only look at true negatives (label=0)
    negatives = df[df["label"] == 0]
    if len(negatives) == 0:
        logger.warning("No negatives found in dataset to generate negative type report.")
        return {}

    report = {}
    logger.info("=" * 60)
    logger.info("Negative Type Performance Report (Accuracy / True Negative Rate)")
    logger.info("=" * 60)

    # Group by negative type
    grouped = negatives.groupby("type")
    for neg_type, group in grouped:
        correct = (group["prediction"] == 0).sum()
        total = len(group)
        accuracy = correct / total if total > 0 else 0.0
        report[f"tn_rate_{neg_type}"] = accuracy
        logger.info(f"  Type '{neg_type:<20}': {accuracy*100:6.2f}% correct ({correct:,} / {total:,})")

    logger.info("=" * 60)
    return report
