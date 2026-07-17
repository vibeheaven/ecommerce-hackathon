"""
Split Builder — Builds K-Fold splits based on normalized query hash.
Ensures zero query leakage between train and validation folds.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold

from project.utils.logging_utils import setup_logger
from project.utils.hashing import normalized_query_hash

logger = setup_logger("split_builder")


def build_splits(
    train_df: pd.DataFrame,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Assign each row of train_df to one of K folds using GroupKFold.
    The grouping is based on the normalized_query_hash of each query.

    Returns the input DataFrame with an added 'fold' column (0 to n_splits-1).
    """
    logger.info(f"Building {n_splits}-fold splits based on normalized query hash...")

    # Duplicate (term_id, item_id) pairs would leak identical rows across the
    # dataset; drop them before splitting.
    if {"term_id", "item_id"}.issubset(train_df.columns):
        dup_mask = train_df.duplicated(["term_id", "item_id"], keep="first")
        if dup_mask.any():
            logger.warning(f"  Dropping {dup_mask.sum():,} duplicate (term_id, item_id) pairs")
            train_df = train_df[~dup_mask].reset_index(drop=True)

    # Calculate normalized query hash if not present
    if "normalized_query_hash" not in train_df.columns:
        logger.info("  Generating normalized query hashes for split...")
        train_df["normalized_query_hash"] = train_df["query"].apply(normalized_query_hash)

    # Scikit-learn GroupKFold
    gkf = GroupKFold(n_splits=n_splits)

    # Initialize fold column
    train_df["fold"] = -1

    # Extract groups
    groups = train_df["normalized_query_hash"].values
    X = np.zeros(len(train_df))
    y = train_df["label"].values if "label" in train_df.columns else np.zeros(len(train_df))

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold_idx
        # Debug metrics per fold
        val_subset = train_df.iloc[val_idx]
        logger.info(
            f"  Fold {fold_idx}: {len(val_subset):,} rows, "
            f"{val_subset['normalized_query_hash'].nunique():,} unique query hashes"
        )

    # Verification: check query leakage
    for f in range(n_splits):
        train_queries = set(train_df[train_df["fold"] != f]["normalized_query_hash"])
        val_queries = set(train_df[train_df["fold"] == f]["normalized_query_hash"])
        leakage = train_queries.intersection(val_queries)
        if leakage:
            raise RuntimeError(
                f"Query leakage detected in fold {f}: {len(leakage)} overlapping query hashes. "
                "Refusing to save leaky splits."
            )
        logger.info(f"  ✓ Fold {f} has 0 query leakage.")

    return train_df


def save_splits(df: pd.DataFrame, output_path: str | Path):
    """Save splits DataFrame as parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"✓ Splits saved to {output_path}")


if __name__ == "__main__":
    import yaml
    import argparse
    from project.data.data_loader import load_all

    parser = argparse.ArgumentParser(description="Build train-validation splits")
    parser.add_argument("--config", default="project/configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data = load_all(".", config)
    train_splits = build_splits(data["train_merged"], n_splits=config["validation"]["n_folds"])

    out_dir = Path(config["data"]["processed_dir"])
    save_splits(train_splits, out_dir / "train_splits.parquet")
