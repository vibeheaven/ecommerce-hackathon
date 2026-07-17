"""
Run Inference Launcher — Runs predictions on the test dataset using one or
more trained fold models (logit-level ensemble) and generates the final
submission file in Kaggle format.

The threshold comes from the checkpoint's inference_meta.json (single model)
or from the OOF-optimized cv_summary (ensemble) — never from a hand-typed
default.
"""
import yaml
import json
import argparse
import numpy as np
from pathlib import Path

from project.utils.logging_utils import setup_logger
from project.data.data_loader import load_all
from project.inference.inference import run_inference, check_pair_text_version
from project.submission.submission_generator import generate_submission
from project.submission.submission_validator import validate_submission

logger = setup_logger("run_inference")


def resolve_model_dirs(processed_dir: Path, model: str, folds: str | None) -> list[Path]:
    """Resolve checkpoint directories for a single fold or a fold ensemble."""
    if folds is None:
        candidates = [processed_dir / f"model_{model}", processed_dir / f"model_{model}_fold0"]
        for c in candidates:
            if c.exists():
                return [c]
        raise FileNotFoundError(f"No trained model found at {candidates[0]} or {candidates[1]}")

    dirs = []
    for f in folds.split(","):
        f = int(f)
        d = processed_dir / (f"model_{model}_fold{f}" if f else f"model_{model}")
        if not d.exists():
            d_alt = processed_dir / f"model_{model}_fold{f}"
            if d_alt.exists():
                d = d_alt
            else:
                raise FileNotFoundError(f"Fold checkpoint not found: {d}")
        dirs.append(d)
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Run inference and generate submission")
    parser.add_argument("--config", default="project/configs/config.yaml")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--folds", default=None, help="Comma-separated folds to ensemble (prob averaging)")
    parser.add_argument("--threshold", type=float, default=None, help="Force specific threshold (otherwise read from checkpoint meta)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    processed_dir = Path(config["data"]["processed_dir"])
    model_dirs = resolve_model_dirs(processed_dir, args.model, args.folds)
    logger.info(f"Using checkpoints: {[str(d) for d in model_dirs]}")

    # Resolve threshold
    threshold = args.threshold
    if threshold is None:
        if len(model_dirs) == 1:
            meta = check_pair_text_version(model_dirs[0])
            threshold = meta["threshold"]
            logger.info(f"Threshold from checkpoint meta: {threshold:.4f} (val F1 {meta['macro_f1']:.4f})")
        else:
            cv_summary_path = processed_dir / f"cv_summary_{args.model}.json"
            if not cv_summary_path.exists():
                raise FileNotFoundError(
                    f"Ensembling requested but {cv_summary_path} not found. "
                    "Run project.training.run_cv first (it optimizes the threshold on OOF), "
                    "or pass --threshold explicitly."
                )
            with open(cv_summary_path, encoding="utf-8") as f:
                threshold = json.load(f)["oof_threshold"]
            logger.info(f"Threshold from OOF CV summary: {threshold:.4f}")

    logger.info("Loading test pairs...")
    data = load_all(".", config)
    submission_merged = data["submission_merged"]

    # Probability averaging over fold checkpoints
    prob_sum = None
    for model_dir in model_dirs:
        probabilities = run_inference(
            submission_merged=submission_merged,
            model_dir=model_dir,
            batch_size=config["inference"]["batch_size"],
            max_length=config["inference"]["max_length"],
        )
        prob_sum = probabilities if prob_sum is None else prob_sum + probabilities
    probabilities = prob_sum / len(model_dirs)

    predictions = (probabilities >= threshold).astype(int)
    logger.info(f"Prediction distribution: positives {predictions.mean()*100:.1f}%")

    output_path = generate_submission(
        submission_pairs=data["submission_pairs"],
        predictions=predictions,
        output_dir=config["submission"]["output_dir"],
    )

    raw_dir = Path(config["data"]["raw_dir"])
    sample_path = raw_dir / config["data"]["files"]["sample_submission"]
    result = validate_submission(
        submission_path=output_path,
        sample_submission_path=sample_path,
        expected_row_count=config["submission"]["required_row_count"],
    )

    if result["valid"]:
        logger.info(f"\n✓ Kaggle Submission file successfully generated and verified: {output_path}")
    else:
        logger.error("\n✗ Submission validation failed! Please check log output for issues.")


if __name__ == "__main__":
    main()
