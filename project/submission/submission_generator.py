"""
Submission Generator — Creates versioned submission CSV files.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

from project.utils.logging_utils import setup_logger

logger = setup_logger("submission_generator")


def generate_submission(
    submission_pairs: pd.DataFrame,
    predictions: list[int] | pd.Series,
    output_dir: str | Path = "project/submission",
    version: int | None = None,
) -> Path:
    """
    Generate a versioned submission CSV file.

    Args:
        submission_pairs: DataFrame with 'id' column
        predictions: list/Series of 0/1 predictions (same length as submission_pairs)
        output_dir: directory to save submission files
        version: version number (auto-detected if None)

    Returns:
        Path to the generated submission file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect version
    if version is None:
        existing = list(output_dir.glob("submission_v*.csv"))
        if existing:
            versions = []
            for f in existing:
                try:
                    v = int(f.stem.split("_v")[1].split("_")[0])
                    versions.append(v)
                except (IndexError, ValueError):
                    pass
            version = max(versions) + 1 if versions else 0
        else:
            version = 0

    # Generate filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"submission_v{version:03d}_{timestamp}.csv"
    filepath = output_dir / filename

    # Create submission DataFrame
    submission = pd.DataFrame({
        "id": submission_pairs["id"],
        "prediction": predictions,
    })

    # Ensure integer predictions
    submission["prediction"] = submission["prediction"].astype(int)

    # Save
    submission.to_csv(filepath, index=False)

    # Stats
    pred_counts = submission["prediction"].value_counts().to_dict()
    total = len(submission)
    pos_count = pred_counts.get(1, 0)
    pos_rate = pos_count / total * 100 if total > 0 else 0

    logger.info(f"✓ Submission generated: {filepath}")
    logger.info(f"  Rows: {total:,}")
    logger.info(f"  Predictions: {pred_counts}")
    logger.info(f"  Predicted positive rate: {pos_rate:.2f}%")

    return filepath
