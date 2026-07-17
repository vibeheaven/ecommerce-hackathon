"""
Run Inference Meta — Generates test predictions by fusing Cross-Encoder scores
with lexical/structural features using the trained LightGBM Meta-Classifier.
Produces a validated, formatted Kaggle submission.
"""
import os
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from project.utils.logging_utils import setup_logger
from project.data.data_loader import load_all
from project.features.feature_extractor import FeatureExtractor
from project.inference.inference import run_inference
from project.submission.submission_generator import generate_submission
from project.submission.submission_validator import validate_submission

logger = setup_logger("run_inference_meta")


def main():
    parser = argparse.ArgumentParser(description="Run inference using the LightGBM Meta-Classifier")
    parser.add_argument("--config", default="project/configs/config.yaml")
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm is not installed. Please run: pip install lightgbm")
        return

    with open(args.config) as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["data"]["processed_dir"])
    raw_dir = Path(config["data"]["raw_dir"])

    # Load meta-classifier and metadata
    model_path = processed_dir / "meta_classifier_lgb.txt"
    meta_path = processed_dir / "meta_classifier_meta.json"
    
    if not model_path.exists() or not meta_path.exists():
        logger.error(f"Meta-classifier files not found at {model_path}. Please run run_meta_classifier.py first!")
        return

    gbm = lgb.Booster(model_file=str(model_path))
    with open(meta_path, encoding="utf-8") as f:
        meta_info = json.load(f)

    logger.info(f"Loaded Meta-Classifier. Cross-Encoder checkpoint: {meta_info['model_dir']}")
    logger.info(f"OOF Validation Macro F1 was: {meta_info['val_macro_f1']:.4f}")

    # 1. Load test data
    logger.info("Loading test pairs...")
    data = load_all(".", config)
    submission_merged = data["submission_merged"]
    items_df = data["items"]

    # 2. Get Cross-Encoder predictions on the test set
    logger.info("Predicting test probabilities with Cross-Encoder...")
    cross_encoder_prob = run_inference(
        submission_merged=submission_merged,
        model_dir=meta_info["model_dir"],
        batch_size=config["inference"]["batch_size"],
        max_length=config["inference"]["max_length"],
    )

    # 3. Extract lexical/structural features for the test set
    extractor = FeatureExtractor(items_df)
    logger.info("Extracting test set features...")
    X_test_feats = extractor.extract_features(submission_merged)
    
    # Add Cross-Encoder prob as a feature
    X_test_feats["cross_encoder_prob"] = cross_encoder_prob

    # 4. Predict with LightGBM
    logger.info("Running Meta-Classifier prediction...")
    test_probs = gbm.predict(X_test_feats)

    # 5. Apply optimized threshold
    threshold = meta_info["best_threshold"]
    predictions = (test_probs >= threshold).astype(int)
    logger.info(f"Prediction distribution: positives {predictions.mean()*100:.1f}%")

    # 6. Generate and validate submission
    submission_df = pd.DataFrame({
        "id": submission_merged["id"],
        "prediction": predictions
    })

    # Save submission file
    output_dir = Path(config["submission"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_file = output_dir / f"submission_meta_{timestamp}.csv"
    
    submission_df.to_csv(out_file, index=False)
    logger.info(f"✓ Submission file generated: {out_file}")

    # Validate
    validate_submission(out_file, config)
    logger.info("✓ Submission file successfully verified!")
    
    print("\n" + "="*80)
    print("Kaggle Submission Gönderim Komutunuz:")
    print(f"kaggle competitions submit -c trendyol-e-ticaret-yarismasi-2026-kaggle -f {out_file} -m \"Meta-Classifier BGE Reranker + Tabular Features\"")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
