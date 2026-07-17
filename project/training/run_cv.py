"""
Run CV — Trains one model per fold, assembles true Out-of-Fold predictions,
optimizes the final threshold ONLY on the OOF set, and writes a CV summary.

Usage:
    python -m project.training.run_cv --folds 0,1,2,3,4
    python -m project.training.run_cv --folds 0          # partial CV while iterating
"""
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from project.utils.logging_utils import setup_logger
from project.training.run_trainer import load_fold_split
from project.training.cross_encoder_trainer import train_model
from project.training.threshold_optimizer import ThresholdOptimizer
from project.validation.validator import Validator

logger = setup_logger("run_cv")


def main():
    parser = argparse.ArgumentParser(description="K-fold CV training with OOF threshold optimization")
    parser.add_argument("--config", default="project/configs/config.yaml")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--folds", default="0,1,2,3,4", help="Comma-separated folds to train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dataset", default="dataset_sampled.parquet")
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    folds = [int(f) for f in args.folds.split(",")]
    processed_dir = Path(config["data"]["processed_dir"])

    oof_frames = []
    oof_probs = []
    model_dirs = {}

    for fold in folds:
        logger.info("=" * 60)
        logger.info(f"Training fold {fold} ({folds.index(fold) + 1}/{len(folds)})...")
        logger.info("=" * 60)

        train_df, val_df = load_fold_split(processed_dir / args.dataset, fold)
        model_dir, _ = train_model(
            config=config,
            train_df=train_df,
            val_df=val_df,
            model_name=args.model,
            epochs=args.epochs,
            val_fold=fold,
            run_tag=args.run_tag,
        )
        model_dirs[fold] = model_dir

        probs = np.load(Path(model_dir) / "val_probs.npy")
        index = pd.read_parquet(Path(model_dir) / "val_index.parquet")
        oof_frames.append(index)
        oof_probs.append(probs)

    oof_df = pd.concat(oof_frames, ignore_index=True)
    oof_probs_arr = np.concatenate(oof_probs)

    logger.info("=" * 60)
    logger.info(f"OOF assembly complete: {len(oof_df):,} rows from folds {folds}")
    logger.info("=" * 60)

    # Threshold is chosen ONLY on OOF predictions
    threshold_opt = ThresholdOptimizer(config)
    opt_res = threshold_opt.optimize(
        oof_df["label"].to_numpy(), oof_probs_arr, folds=oof_df["fold"].to_numpy()
    )
    threshold = opt_res["best_threshold"]

    validator = Validator(config)
    metrics = validator.evaluate_oof(oof_df, oof_probs_arr, threshold)

    # Persist OOF artifacts for ensembling / analysis
    tag = f"_{args.run_tag}" if args.run_tag else ""
    oof_out = processed_dir / f"oof_{args.model}{tag}.parquet"
    oof_df = oof_df.copy()
    oof_df["pred_prob"] = oof_probs_arr
    oof_df.to_parquet(oof_out, index=False)

    summary = {
        "model": args.model,
        "run_tag": args.run_tag,
        "folds": folds,
        "oof_threshold": threshold,
        "threshold_per_fold": opt_res["fold_thresholds"],
        "threshold_std": opt_res["threshold_std"],
        "model_dirs": model_dirs,
        "metrics": metrics,
    }
    summary_path = processed_dir / f"cv_summary_{args.model}{tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"✓ CV complete. OOF Macro F1: {metrics['overall_macro_f1']:.4f} at threshold {threshold:.4f}")
    logger.info(f"  OOF predictions: {oof_out}")
    logger.info(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
