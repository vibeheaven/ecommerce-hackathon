"""
Data Validator — Validates data integrity before processing.
Checks: missing IDs, duplicates, empty fields, CSV parse issues.
"""
import pandas as pd
from pathlib import Path

from project.utils.logging_utils import setup_logger

logger = setup_logger("data_validator")


def validate_items(df: pd.DataFrame) -> list[str]:
    """Validate items DataFrame. Returns list of issues."""
    issues = []

    # Check required columns
    required = ["item_id", "title", "category", "brand", "gender", "age_group", "attributes"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    # Duplicate IDs
    dup_count = df["item_id"].duplicated().sum()
    if dup_count > 0:
        issues.append(f"Duplicate item_ids: {dup_count:,}")

    # Missing values
    for col in ["item_id", "title", "category"]:
        if col in df.columns:
            na_count = df[col].isna().sum()
            empty_count = (df[col] == "").sum() if df[col].dtype == "object" else 0
            total_missing = na_count + empty_count
            if total_missing > 0:
                issues.append(f"Missing/empty {col}: {total_missing:,}")

    # Brand/gender/age_group missing (warning level, not critical)
    for col in ["brand", "gender", "age_group"]:
        if col in df.columns:
            na_count = df[col].isna().sum()
            empty_count = (df[col] == "").sum() if df[col].dtype == "object" else 0
            total_missing = na_count + empty_count
            if total_missing > 0:
                logger.info(f"  [INFO] {col} missing/empty: {total_missing:,} ({total_missing/len(df)*100:.1f}%)")

    # Attributes missing
    if "attributes" in df.columns:
        na_count = df["attributes"].isna().sum()
        empty_count = (df["attributes"] == "").sum()
        total_missing = na_count + empty_count
        if total_missing > 0:
            logger.info(f"  [INFO] attributes missing/empty: {total_missing:,}")

    # CSV parse anomaly: gender/age_group containing category-like values
    if "gender" in df.columns:
        anomaly = df["gender"].str.contains("/", na=False).sum()
        if anomaly > 0:
            issues.append(f"Gender column contains '/' (possible CSV parse issue): {anomaly:,} rows")

    if "age_group" in df.columns:
        anomaly = df["age_group"].str.contains("/", na=False).sum()
        if anomaly > 0:
            issues.append(f"Age_group column contains '/' (possible CSV parse issue): {anomaly:,} rows")

    return issues


def validate_terms(df: pd.DataFrame) -> list[str]:
    """Validate terms DataFrame."""
    issues = []

    required = ["term_id", "query"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    dup_count = df["term_id"].duplicated().sum()
    if dup_count > 0:
        issues.append(f"Duplicate term_ids: {dup_count:,}")

    na_query = df["query"].isna().sum()
    if na_query > 0:
        issues.append(f"Missing queries: {na_query:,}")

    return issues


def validate_training_pairs(df: pd.DataFrame) -> list[str]:
    """Validate training_pairs DataFrame."""
    issues = []

    required = ["id", "term_id", "item_id", "label"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    dup_count = df["id"].duplicated().sum()
    if dup_count > 0:
        issues.append(f"Duplicate ids: {dup_count:,}")

    # Check label distribution
    if "label" in df.columns:
        label_dist = df["label"].value_counts().to_dict()
        logger.info(f"  Label distribution: {label_dist}")
        if set(label_dist.keys()) != {1}:
            issues.append(f"Unexpected labels (expected only 1): {label_dist}")

    return issues


def validate_submission_pairs(df: pd.DataFrame) -> list[str]:
    """Validate submission_pairs DataFrame."""
    issues = []

    required = ["id", "term_id", "item_id"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")

    dup_count = df["id"].duplicated().sum()
    if dup_count > 0:
        issues.append(f"Duplicate ids: {dup_count:,}")

    return issues


def validate_merged(df: pd.DataFrame, name: str) -> list[str]:
    """Validate merged DataFrame for join losses."""
    issues = []

    na_query = df["query"].isna().sum() if "query" in df.columns else 0
    if na_query > 0:
        issues.append(f"[{name}] Join loss on term_id: {na_query:,} rows missing query")

    na_title = df["title"].isna().sum() if "title" in df.columns else 0
    if na_title > 0:
        issues.append(f"[{name}] Join loss on item_id: {na_title:,} rows missing title")

    return issues


def run_full_validation(data: dict) -> dict:
    """
    Run full validation suite on all datasets.
    Returns dict with 'issues' list and 'passed' bool.
    """
    logger.info("=" * 60)
    logger.info("Running full data validation...")
    logger.info("=" * 60)

    all_issues = []

    logger.info("\n[1/6] Validating items...")
    issues = validate_items(data["items"])
    all_issues.extend(issues)
    for issue in issues:
        logger.warning(f"  ⚠ {issue}")
    if not issues:
        logger.info("  ✓ items.csv passed")

    logger.info("\n[2/6] Validating terms...")
    issues = validate_terms(data["terms"])
    all_issues.extend(issues)
    for issue in issues:
        logger.warning(f"  ⚠ {issue}")
    if not issues:
        logger.info("  ✓ terms.csv passed")

    logger.info("\n[3/6] Validating training_pairs...")
    issues = validate_training_pairs(data["training_pairs"])
    all_issues.extend(issues)
    for issue in issues:
        logger.warning(f"  ⚠ {issue}")
    if not issues:
        logger.info("  ✓ training_pairs.csv passed")

    logger.info("\n[4/6] Validating submission_pairs...")
    issues = validate_submission_pairs(data["submission_pairs"])
    all_issues.extend(issues)
    for issue in issues:
        logger.warning(f"  ⚠ {issue}")
    if not issues:
        logger.info("  ✓ submission_pairs.csv passed")

    logger.info("\n[5/6] Validating train_merged...")
    issues = validate_merged(data["train_merged"], "train_merged")
    all_issues.extend(issues)
    for issue in issues:
        logger.warning(f"  ⚠ {issue}")
    if not issues:
        logger.info("  ✓ train_merged passed (no join loss)")

    logger.info("\n[6/6] Validating submission_merged...")
    issues = validate_merged(data["submission_merged"], "submission_merged")
    all_issues.extend(issues)
    for issue in issues:
        logger.warning(f"  ⚠ {issue}")
    if not issues:
        logger.info("  ✓ submission_merged passed (no join loss)")

    logger.info("\n" + "=" * 60)
    if all_issues:
        logger.warning(f"Validation completed with {len(all_issues)} issues:")
        for i, issue in enumerate(all_issues, 1):
            logger.warning(f"  {i}. {issue}")
    else:
        logger.info("✓ All validations passed!")
    logger.info("=" * 60)

    return {"issues": all_issues, "passed": len(all_issues) == 0}


if __name__ == "__main__":
    import yaml
    import argparse
    from project.data.data_loader import load_all

    parser = argparse.ArgumentParser(description="Validate all data files")
    parser.add_argument("--config", default="project/configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data = load_all(".", config)
    result = run_full_validation(data)

    if not result["passed"]:
        exit(1)
