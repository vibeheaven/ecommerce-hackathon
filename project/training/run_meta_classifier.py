"""
Run Meta-Classifier — Trains a LightGBM Meta-Classifier using Cross-Encoder predictions
and structural/lexical features extracted from the validation fold (OOF).
Optimizes the threshold on LightGBM predictions and saves the model.
"""
import os
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from project.utils.logging_utils import setup_logger
from project.data.data_loader import load_items
from project.features.feature_extractor import FeatureExtractor
from project.training.threshold_optimizer import ThresholdOptimizer
from project.validation.validator import Validator

logger = setup_logger("run_meta_classifier")


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Meta-Classifier")
    parser.add_argument("--config", default="project/configs/config.yaml")
    parser.add_argument("--model-dir", default=None, help="Path to trained fold checkpoint (e.g. project/data/processed/model_BAAI-bge-reranker-v2-m3_fold0)")
    args = parser.parse_args()

    # Import lightgbm here
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm is not installed. Please run: pip install lightgbm")
        return

    with open(args.config) as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["data"]["processed_dir"])
    raw_dir = Path(config["data"]["raw_dir"])

    # Resolve model directory
    model_dir = args.model_dir
    if model_dir is None:
        # Auto-detect latest BAAI-bge-reranker-v2-m3 fold0 checkpoint
        candidates = [
            processed_dir / "model_BAAI-bge-reranker-v2-m3_fold0",
            processed_dir / "model_BAAI-bge-reranker-v2-m3",
            processed_dir / "model_xlm-roberta-base_fold0",
            processed_dir / "model_xlm-roberta-base",
        ]
        for c in candidates:
            if c.exists() and (c / "val_probs.npy").exists():
                model_dir = c
                break
        if model_dir is None:
            logger.error("No trained model directory with val_probs.npy found. Please run training first!")
            return

    model_dir = Path(model_dir)
    logger.info(f"Using Cross-Encoder validation outputs from: {model_dir}")

    # Load validation index and probabilities
    val_probs = np.load(model_dir / "val_probs.npy")
    val_index = pd.read_parquet(model_dir / "val_index.parquet")
    logger.info(f"Loaded validation outputs: {len(val_index):,} rows")

    # Load items metadata
    items_path = raw_dir / config["data"]["files"]["items"]
    items_df = load_items(items_path)

    # Initialize FeatureExtractor
    extractor = FeatureExtractor(items_df)

    # Extract lexical/structural features for validation set
    logger.info("Extracting features for validation pairs...")
    X_val_feats = extractor.extract_features(val_index)
    
    # Add Cross-Encoder probability as a feature
    X_val_feats["cross_encoder_prob"] = val_probs
    y_val = val_index["label"].to_numpy()

    # Train LightGBM Classifier
    logger.info("Training LightGBM Meta-Classifier...")
    train_data = lgb.Dataset(X_val_feats, label=y_val)
    
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "seed": 42,
        "verbose": -1,
    }
    
    # Simple training (we use whole validation split since it is OOF to the cross-encoder)
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=100,
    )
    logger.info("✓ LightGBM training complete!")

    # Predict and optimize threshold on the meta-classifier predictions
    meta_preds = gbm.predict(X_val_feats)
    
    threshold_opt = ThresholdOptimizer(config)
    # Simulate a single fold array
    dummy_folds = np.zeros(len(y_val))
    opt_res = threshold_opt.optimize(y_val, meta_preds, folds=dummy_folds)
    best_threshold = opt_res["best_threshold"]
    logger.info(f"Optimized Meta-Classifier Threshold: {best_threshold:.4f}")

    # Evaluate validation metrics
    val_index_with_pred = val_index.copy()
    val_index_with_pred["pred_prob"] = meta_preds
    
    validator = Validator(config)
    metrics = validator.evaluate_oof(val_index_with_pred, meta_preds, best_threshold)
    logger.info(f"Validation Macro F1 after Feature Fusion: {metrics['macro_f1']:.4f}")

    # Save meta-classifier model and metadata
    model_output_path = processed_dir / "meta_classifier_lgb.txt"
    gbm.save_model(str(model_output_path))
    
    meta_info = {
        "model_dir": str(model_dir),
        "best_threshold": best_threshold,
        "val_macro_f1": metrics["macro_f1"],
        "feature_names": list(X_val_feats.columns),
    }
    with open(processed_dir / "meta_classifier_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4)
        
    logger.info(f"✓ Meta-classifier saved to: {model_output_path}")
    logger.info(f"✓ Meta-classifier metadata saved to: {processed_dir / 'meta_classifier_meta.json'}")


if __name__ == "__main__":
    main()
