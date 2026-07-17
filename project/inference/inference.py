"""
Inference — Runs fast batch inference on 3.36M test pairs.
Uses dynamic padding, length-bucket batching, and mixed precision.
"""
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from project.utils.logging_utils import setup_logger
from project.inference.batch_builder import PairInferenceDataset, LengthBucketSampler, collate_fn_builder

logger = setup_logger("inference")


def run_inference(
    submission_merged: pd.DataFrame,
    model_dir: str | Path,
    batch_size: int = 128,
    max_length: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """
    Run sequence classification inference on merged pairs.

    Returns:
        np.ndarray of probability scores (0.0 to 1.0)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device for inference: {device}")

    # Load model and tokenizer
    logger.info(f"Loading checkpoint from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Pre-process columns
    queries = submission_merged["query"].fillna("").tolist()
    titles = submission_merged["title"].fillna("").tolist()
    categories = submission_merged["category"].fillna("").tolist()

    # Setup dataset & loader
    dataset = PairInferenceDataset(queries, titles, categories)
    sampler = LengthBucketSampler(dataset, tokenizer, batch_size, max_length)
    collate = collate_fn_builder(tokenizer, max_length)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,  # avoid multiprocessing issues on macOS/windows
    )

    # Placeholders for predictions and indices
    all_probs = []
    all_indices = []

    logger.info(f"Running batch inference on {len(dataset):,} pairs...")

    # Determine auto-cast device type
    device_type = "cuda" if "cuda" in device else "cpu"

    with torch.no_grad():
        # Get order of indices visited by sampler
        visited_indices = list(sampler)
        idx_ptr = 0

        # Loop loader
        for batch in tqdm(loader, desc="Inference"):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Mixed precision inference
            with torch.amp.autocast(device_type=device_type, dtype=torch.float16 if device_type == "cuda" else torch.float32):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)

            # Keep track of original indices
            batch_len = len(probs)
            all_indices.extend(visited_indices[idx_ptr : idx_ptr + batch_len])
            idx_ptr += batch_len

    # Sort back to original CSV order
    logger.info("Re-ordering predictions back to original sequence...")
    all_probs = np.array(all_probs)
    all_indices = np.array(all_indices)

    # Restore original order
    restored_probs = np.zeros(len(dataset))
    restored_probs[all_indices] = all_probs

    return restored_probs
