"""
Inference — Runs fast batch inference on 3.36M test pairs.
Uses dynamic padding, length-bucket batching, and mixed precision.
"""
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from project.utils.logging_utils import setup_logger
from project.utils.pair_text import build_pair_texts_from_frame, PAIR_TEXT_VERSION
from project.inference.batch_builder import PairInferenceDataset, LengthBucketSampler, collate_fn_builder

logger = setup_logger("inference")


def check_pair_text_version(model_dir: str | Path):
    """Refuse to serve a checkpoint trained with a different input format."""
    meta_path = Path(model_dir) / "inference_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Retrain with the v5 trainer (it saves threshold "
            "and input-format metadata next to the checkpoint)."
        )
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("pair_text_version") != PAIR_TEXT_VERSION:
        raise ValueError(
            f"Checkpoint {model_dir} was trained with input format "
            f"'{meta.get('pair_text_version')}' but the code builds '{PAIR_TEXT_VERSION}'. "
            "Retrain or check out the matching code version."
        )
    return meta


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
        np.ndarray of probability scores (0.0 to 1.0), aligned with the
        input DataFrame's row order.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device for inference: {device}")

    check_pair_text_version(model_dir)

    logger.info(f"Loading checkpoint from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    logger.info(f"Building pair texts ({PAIR_TEXT_VERSION})...")
    queries, products = build_pair_texts_from_frame(submission_merged)

    dataset = PairInferenceDataset(queries, products)
    sampler = LengthBucketSampler(dataset, batch_size)
    collate = collate_fn_builder(tokenizer, max_length)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,
    )

    all_probs = []
    logger.info(f"Running batch inference on {len(dataset):,} pairs...")

    device_type = "cuda" if "cuda" in device else "cpu"
    visited_indices = list(iter(sampler))
    idx_ptr = 0
    all_indices = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast(device_type=device_type, dtype=torch.float16 if device_type == "cuda" else torch.float32):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                probs = torch.sigmoid(logits).float().cpu().numpy()

            all_probs.extend(probs)
            batch_len = len(probs)
            all_indices.extend(visited_indices[idx_ptr : idx_ptr + batch_len])
            idx_ptr += batch_len

    logger.info("Re-ordering predictions back to original sequence...")
    restored_probs = np.zeros(len(dataset))
    restored_probs[np.array(all_indices)] = np.array(all_probs)

    return restored_probs
