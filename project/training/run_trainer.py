"""
Run Trainer — Launcher script for Cross Encoder Training.
Loads the fold-aware dataset_sampled.parquet and trains with the requested
validation fold held out.
"""
import yaml
import argparse
import pandas as pd
from pathlib import Path

from project.utils.logging_utils import setup_logger
from project.training.cross_encoder_trainer import train_model

logger = setup_logger("run_trainer")


def load_fold_split(dataset_path: Path, val_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the single sampled dataset into train/val by fold."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Sampled dataset not found: {dataset_path}. "
            "Run python -m project.data.build_sampled_dataset first."
        )
    logger.info(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)

    folds = sorted(df["fold"].unique())
    if val_fold not in folds:
        raise ValueError(f"val_fold={val_fold} not present in dataset (folds: {folds})")

    train_df = df[df["fold"] != val_fold].reset_index(drop=True)
    val_df = df[df["fold"] == val_fold].reset_index(drop=True)
    logger.info(
        f"  Fold {val_fold} held out — train: {len(train_df):,} rows, val: {len(val_df):,} rows"
    )
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description="Train sequence classification cross encoder")
    parser.add_argument("--config", default="project/configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config training.epochs")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--dataset", default="dataset_sampled.parquet")
    parser.add_argument("--run-tag", default="", help="Suffix for the model output dir (experiment name)")
    parser.add_argument("--debug", action="store_true", help="Run a quick smoke test training run")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["data"]["processed_dir"])
    train_df, val_df = load_fold_split(processed_dir / args.dataset, args.val_fold)

    model_dir, metrics = train_model(
        config=config,
        train_df=train_df,
        val_df=val_df,
        model_name=args.model,
        epochs=args.epochs,
        debug=args.debug,
        val_fold=args.val_fold,
        run_tag=args.run_tag,
    )

    logger.info(f"✓ Training finished! Best model stored in: {model_dir}")


if __name__ == "__main__":
    main()
