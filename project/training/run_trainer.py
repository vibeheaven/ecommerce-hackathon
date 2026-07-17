"""
Run Trainer — Launcher script for Cross Encoder Training.
Loads train_set_sampled.parquet & val_set_sampled.parquet and kicks off training.
"""
import yaml
import argparse
import pandas as pd
from pathlib import Path

from project.utils.logging_utils import setup_logger
from project.training.cross_encoder_trainer import train_model

logger = setup_logger("run_trainer")


def main():
    parser = argparse.ArgumentParser(description="Train sequence classification cross encoder")
    parser.add_argument("--config", default="project/configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--debug", action="store_true", help="Run a quick smoke test training run")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 1. Load sampled datasets
    processed_dir = Path(config["data"]["processed_dir"])
    train_path = processed_dir / "train_set_sampled.parquet"
    val_path = processed_dir / "val_set_sampled.parquet"

    if not train_path.exists() or not val_path.exists():
        logger.error(
            f"Sampled datasets not found. Run build_sampled_dataset.py first.\n"
            f"  Expected: {train_path} and {val_path}"
        )
        return

    logger.info(f"Loading training dataset from {train_path}...")
    train_df = pd.read_parquet(train_path)

    logger.info(f"Loading validation dataset from {val_path}...")
    val_df = pd.read_parquet(val_path)

    # 2. Train
    model_dir, metrics = train_model(
        config=config,
        train_df=train_df,
        val_df=val_df,
        model_name=args.model,
        epochs=args.epochs,
        debug=args.debug,
    )

    logger.info(f"✓ Training finished! Best model stored in: {model_dir}")


if __name__ == "__main__":
    main()
