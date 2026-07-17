"""
Inference Benchmark — Measures inference throughput on a 10K subset.
Projects total runtime for the full 3.36M test dataset.
"""
import time
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from project.utils.logging_utils import setup_logger
from project.data.data_loader import load_all
from project.inference.inference import run_inference

logger = setup_logger("benchmark")


def run_benchmark(
    config: dict,
    model_dir: str | Path,
    subset_size: int = 10_000,
) -> dict:
    """Run benchmark on a subset and save throughput/runtime metrics."""
    logger.info("=" * 60)
    logger.info(f"Running Inference Benchmark on {subset_size:,} pairs...")
    logger.info("=" * 60)

    # 1. Load data
    data = load_all(".", config)
    sub_df = data["submission_merged"].head(subset_size)

    # 2. Benchmark inference
    start_time = time.time()
    probs = run_inference(
        sub_df,
        model_dir=model_dir,
        batch_size=config["inference"]["batch_size"],
        max_length=config["inference"]["max_length"],
    )
    elapsed = time.time() - start_time

    # 3. Throughput metrics
    throughput = subset_size / elapsed if elapsed > 0 else 0.0
    total_pairs = config["submission"]["required_row_count"]
    projected_time_sec = total_pairs / throughput if throughput > 0 else 0.0
    projected_time_min = projected_time_sec / 60.0

    logger.info("=" * 60)
    logger.info("Benchmark Results:")
    logger.info(f"  Elapsed time for {subset_size:,} pairs: {elapsed:.2f} seconds")
    logger.info(f"  Throughput: {throughput:.2f} pairs/second")
    logger.info(f"  Estimated total time for {total_pairs:,} pairs: {projected_time_min:.2f} minutes")
    logger.info("=" * 60)

    # Save report
    report_path = Path("project/reports/inference_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    report_content = f"""# Inference Benchmark Report

- **Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Device**: `{device}`
- **Model Checkpoint**: `{model_dir}`
- **Benchmark Subset Size**: {subset_size:,}
- **Elapsed Time**: {elapsed:.2f} seconds
- **Throughput**: {throughput:.2f} pairs/second
- **Estimated Total Time for 3.36M**: **{projected_time_min:.2f} minutes** ({projected_time_sec:.2f} seconds)

## PyTorch / Hardware Stats
- CUDA Available: {torch.cuda.is_available()}
- MPS Available: {torch.backends.mps.is_available()}
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"✓ Inference benchmark report saved to {report_path}")

    return {
        "elapsed_sec": elapsed,
        "throughput_pairs_per_sec": throughput,
        "projected_time_min": projected_time_min,
        "device": device,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference benchmark")
    parser.add_argument("--config", default="project/configs/config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--subset", type=int, default=10_000)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_benchmark(config, args.model, args.subset)
