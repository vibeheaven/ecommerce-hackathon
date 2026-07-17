"""
Submission Validator — Validates submission CSV before upload.
9 mandatory checks. Blocks invalid submissions.
"""
import pandas as pd
from pathlib import Path

from project.utils.logging_utils import setup_logger

logger = setup_logger("submission_validator")


def validate_submission(
    submission_path: str | Path,
    sample_submission_path: str | Path | None = None,
    submission_pairs_path: str | Path | None = None,
    expected_row_count: int = 3_359_679,
) -> dict:
    """
    Validate submission CSV against all requirements.

    Returns dict with 'valid' bool and 'issues' list.
    """
    issues = []
    submission_path = Path(submission_path)

    # 1. File exists
    if not submission_path.exists():
        return {"valid": False, "issues": [f"File not found: {submission_path}"]}

    # 2. Load submission
    try:
        df = pd.read_csv(submission_path, dtype={"id": "str", "prediction": "str"})
    except Exception as e:
        return {"valid": False, "issues": [f"Cannot read CSV: {e}"]}

    # 3. Column check — must be exactly [id, prediction]
    expected_cols = ["id", "prediction"]
    has_id = "id" in df.columns
    has_pred = "prediction" in df.columns

    if list(df.columns) != expected_cols:
        issues.append(f"Wrong columns: {list(df.columns)}, expected {expected_cols}")

    # 4. Row count check
    if len(df) != expected_row_count:
        issues.append(f"Wrong row count: {len(df):,}, expected {expected_row_count:,}")

    # 5. Duplicate ID check
    if has_id:
        dup_count = df["id"].duplicated().sum()
        if dup_count > 0:
            issues.append(f"Duplicate IDs: {dup_count:,}")
    else:
        issues.append("Missing 'id' column")

    # 6. Prediction values — must be only 0 or 1
    if has_pred:
        try:
            predictions = df["prediction"].astype(int)
            invalid_vals = predictions[~predictions.isin([0, 1])]
            if len(invalid_vals) > 0:
                issues.append(f"Invalid prediction values (not 0/1): {len(invalid_vals):,}")
        except (ValueError, TypeError):
            issues.append("Prediction column contains non-integer values")
    else:
        issues.append("Missing 'prediction' column")

    # 7. NaN check
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        issues.append(f"NaN values found: {nan_count:,}")

    # 8. Check against sample submission IDs
    if sample_submission_path and has_id:
        sample = pd.read_csv(sample_submission_path, dtype={"id": "str"}, usecols=["id"])
        missing_ids = set(sample["id"]) - set(df["id"])
        extra_ids = set(df["id"]) - set(sample["id"])
        if missing_ids:
            issues.append(f"Missing IDs (in sample but not in submission): {len(missing_ids):,}")
        if extra_ids:
            issues.append(f"Extra IDs (in submission but not in sample): {len(extra_ids):,}")

        # 9. ID order check
        if list(df["id"]) != list(sample["id"]):
            issues.append("ID order does not match sample submission")

    # Report
    valid = len(issues) == 0
    if valid:
        logger.info("✓ Submission validation passed!")
        # Stats
        pred_counts = df["prediction"].astype(int).value_counts().to_dict()
        total = len(df)
        pos_count = pred_counts.get(1, 0)
        pos_rate = pos_count / total * 100 if total > 0 else 0
        logger.info(f"  Predictions: {pred_counts}")
        logger.info(f"  Predicted positive rate: {pos_rate:.2f}%")
    else:
        logger.error(f"✗ Submission validation FAILED ({len(issues)} issues):")
        for i, issue in enumerate(issues, 1):
            logger.error(f"  {i}. {issue}")

    return {"valid": valid, "issues": issues}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate submission CSV")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--expected-rows", type=int, default=3_359_679)
    args = parser.parse_args()

    result = validate_submission(
        submission_path=args.submission,
        sample_submission_path=args.sample,
        expected_row_count=args.expected_rows,
    )

    if not result["valid"]:
        exit(1)
