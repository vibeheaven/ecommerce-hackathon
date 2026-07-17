"""
Run Inference Launcher — Runs predictions on the test dataset using the trained model
and generates the final submission file in Kaggle format.
"""
import yaml
import json
import argparse
import pandas as pd
from pathlib import Path

from project.utils.logging_utils import setup_logger
from project.data.data_loader import load_all
from project.inference.inference import run_inference
from project.submission.submission_generator import generate_submission
from project.submission.submission_validator import validate_submission

logger = setup_logger("run_inference")


def get_best_experiment_threshold(registry_path: Path, model_name: str) -> float:
    """Read the optimized threshold from the experiment registry."""
    if not registry_path.exists():
        logger.warning(f"Experiment registry not found at {registry_path}. Defaulting threshold to 0.35")
        return 0.35

    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        # Find best macro_f1 for the given model
        best_f1 = -1.0
        best_threshold = 0.35

        for entry in registry:
            if entry.get("model_name") == model_name:
                f1 = entry.get("metrics", {}).get("overall_macro_f1", 0.0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = entry.get("metrics", {}).get("threshold", 0.35)

        logger.info(f"Retrieved best threshold {best_threshold:.4f} (F1: {best_f1:.4f}) from registry for {model_name}.")
        return best_threshold
    except Exception as e:
        logger.error(f"Error reading registry: {e}. Defaulting threshold to 0.35")
        return 0.35


def main():
    parser = argparse.ArgumentParser(description="Run inference and generate submission")
    parser.add_argument("--config", default="project/configs/config.yaml")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--threshold", type=float, default=None, help="Force specific threshold. If None, reads from registry.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 1. Resolve paths
    processed_dir = Path(config["data"]["processed_dir"])
    model_dir = processed_dir / f"model_{args.model}"
    registry_path = Path(config["experiments"]["output_dir"]) / "experiment_registry.json"

    if not model_dir.exists():
        logger.error(f"Trained model not found at {model_dir}. Please train the model first.")
        return

    # 2. Get optimal threshold
    threshold = args.threshold
    if threshold is None:
        threshold = get_best_experiment_threshold(registry_path, args.model)

    # 3. Load merged submission dataset
    logger.info("Loading test pairs...")
    data = load_all(".", config)
    submission_merged = data["submission_merged"]

    # 4. Run model inference to get probability scores
    probabilities = run_inference(
        submission_merged=submission_merged,
        model_dir=model_dir,
        batch_size=config["inference"]["batch_size"],
        max_length=config["inference"]["max_length"],
    )

    # 5. Apply threshold to get binary predictions (0 or 1)
    predictions = (probabilities >= threshold).astype(int)

    # 6. Generate versioned submission CSV
    output_path = generate_submission(
        submission_pairs=data["submission_pairs"],
        predictions=predictions,
        output_dir=config["submission"]["output_dir"],
    )

    # 7. Validate format, length, ordering, and values
    raw_dir = Path(config["data"]["raw_dir"])
    sample_path = raw_dir / config["data"]["files"]["sample_submission"]
    result = validate_submission(
        submission_path=output_path,
        sample_submission_path=sample_path,
        expected_row_count=config["submission"]["required_row_count"],
    )

    if result["valid"]:
        logger.info(f"\n✓ Kaggle Submission file successfully generated and verified: {output_path}")
        logger.info(f"Run this command to upload directly to Kaggle:\n")
        logger.info(f"  kaggle competitions submit -c trendyol-e-ticaret-yarismasi-2026-kaggle -f {output_path} -m 'V1 Cross Encoder baseline'")
    else:
        logger.error("\n✗ Submission validation failed! Please check log output for issues.")


if __name__ == "__main__":
    main()
