"""
Build Sampled Dataset — Samples negatives for training and validation splits.
Builds train_set_sampled.parquet and val_set_sampled.parquet.
Supports `--max-positive-samples` to make a small subset for debug/smoke runs.
"""
import yaml
import argparse
import pandas as pd
from pathlib import Path

from project.utils.logging_utils import setup_logger
from project.data.data_loader import load_items
from project.negative_samples.negative_sampler import NegativeSampler

logger = setup_logger("build_sampled_dataset")


def main():
    parser = argparse.ArgumentParser(description="Build training & validation datasets with negatives")
    parser.add_argument("--config", default="project/configs/config.yaml")
    parser.add_argument("--neg-config", default="project/configs/negative_sampling.yaml")
    parser.add_argument("--max-positive-samples", type=int, default=None, help="Limit number of positives for quick runs")
    parser.add_argument("--val-fold", type=int, default=0, help="Fold index to use as validation")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.neg_config) as f:
        neg_config = yaml.safe_load(f)

    # 1. Load splits and items
    processed_dir = Path(config["data"]["processed_dir"])
    splits_path = processed_dir / "train_splits.parquet"
    if not splits_path.exists():
        logger.error(f"Splits file not found: {splits_path}. Run split_builder first.")
        return

    logger.info(f"Loading splits from {splits_path}...")
    splits_df = pd.read_parquet(splits_path)

    # Apply debug limit
    if args.max_positive_samples is not None:
        logger.info(f"Limiting input dataset to first {args.max_positive_samples:,} positive rows for debug.")
        splits_df = splits_df.head(args.max_positive_samples)

    raw_dir = Path(config["data"]["raw_dir"])
    items_path = raw_dir / config["data"]["files"]["items"]
    items_df = load_items(items_path)

    # 2. Divide train and val by fold
    val_fold_idx = args.val_fold
    logger.info(f"Dividing dataset: Fold {val_fold_idx} is Validation, others are Training.")
    train_splits = splits_df[splits_df["fold"] != val_fold_idx].copy()
    val_splits = splits_df[splits_df["fold"] == val_fold_idx].copy()

    logger.info(f"  Training positives:   {len(train_splits):,}")
    logger.info(f"  Validation positives: {len(val_splits):,}")

    # 3. Sample negatives
    sampler = NegativeSampler(config, neg_config, items_df, splits_df)

    logger.info("Sampling negatives for Training set...")
    train_sampled = sampler.build_dataset(train_splits)

    logger.info("Sampling negatives for Validation set...")
    val_sampled = sampler.build_dataset(val_splits)

    # 4. Save
    train_output = processed_dir / "train_set_sampled.parquet"
    val_output = processed_dir / "val_set_sampled.parquet"

    logger.info(f"Saving training set to {train_output} ({len(train_sampled):,} rows)...")
    train_sampled.to_parquet(train_output, index=False)

    logger.info(f"Saving validation set to {val_output} ({len(val_sampled):,} rows)...")
    val_sampled.to_parquet(val_output, index=False)

    logger.info("✓ Datasets successfully built!")


if __name__ == "__main__":
    main()
