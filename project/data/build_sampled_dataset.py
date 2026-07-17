"""
Build Sampled Dataset — Samples negatives for ALL folds into a single
fold-aware file: dataset_sampled.parquet.

The old two-file layout (train_set_sampled / val_set_sampled) hard-wired
fold 0 as validation; the single file lets run_trainer / run_cv pick any
fold, and lets threshold optimization run on true OOF predictions.
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
    parser = argparse.ArgumentParser(description="Build the fold-aware sampled dataset")
    parser.add_argument("--config", default="project/configs/config.yaml")
    parser.add_argument("--neg-config", default="project/configs/negative_sampling.yaml")
    parser.add_argument("--max-positive-samples", type=int, default=None, help="Limit number of positives for quick runs")
    parser.add_argument("--output", default="dataset_sampled.parquet")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.neg_config) as f:
        neg_config = yaml.safe_load(f)

    processed_dir = Path(config["data"]["processed_dir"])
    splits_path = processed_dir / "train_splits.parquet"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_path}. Run project.data.split_builder first."
        )

    logger.info(f"Loading splits from {splits_path}...")
    splits_df = pd.read_parquet(splits_path)

    if args.max_positive_samples is not None:
        logger.info(f"Limiting input dataset to first {args.max_positive_samples:,} positive rows for debug.")
        splits_df = splits_df.head(args.max_positive_samples)

    raw_dir = Path(config["data"]["raw_dir"])
    items_path = raw_dir / config["data"]["files"]["items"]
    items_df = load_items(items_path)

    # Load semantic neighbors when the embedding_hard strategy is enabled
    embedding_neighbors = None
    if neg_config.get("strategy_ratios", {}).get("embedding_hard", 0.0) > 0:
        from project.embeddings.build_embedding_neighbors import load_embedding_neighbors
        pos_by_item_id = {iid: i for i, iid in enumerate(items_df["item_id"].tolist())}
        embedding_neighbors = load_embedding_neighbors(processed_dir, pos_by_item_id)
        logger.info(f"Loaded embedding neighbors for {len(embedding_neighbors):,} queries")

    # Sample negatives for every positive row; each negative inherits the
    # fold of its anchor positive, so any fold can serve as validation.
    sampler = NegativeSampler(config, neg_config, items_df, splits_df, embedding_neighbors=embedding_neighbors)
    dataset = sampler.build_dataset(splits_df)

    output_path = processed_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving dataset to {output_path} ({len(dataset):,} rows)...")
    dataset.to_parquet(output_path, index=False)

    fold_summary = dataset.groupby("fold")["label"].agg(["count", "sum"])
    for fold, row in fold_summary.iterrows():
        logger.info(f"  Fold {fold}: {int(row['count']):,} rows ({int(row['sum']):,} positives)")

    logger.info("✓ Dataset successfully built!")


if __name__ == "__main__":
    main()
