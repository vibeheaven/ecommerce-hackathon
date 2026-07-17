"""
Cross Encoder Trainer — Fine-tunes XLM-RoBERTa / BERTürk sequence classification.
Supports: mixed precision (AMP), gradient accumulation, sample weights,
early stopping, cosine scheduling, and OOF threshold optimization per epoch.
"""
import os
import time
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

from project.utils.logging_utils import setup_logger
from project.training.sample_weighting import compute_weighted_loss
from project.training.threshold_optimizer import ThresholdOptimizer
from project.validation.validator import Validator
from project.experiments.tracker import ExperimentTracker

logger = setup_logger("cross_encoder_trainer")


class PairDataset(Dataset):
    """PyTorch Dataset representing query-product pairs and labels."""

    def __init__(
        self,
        queries: list[str],
        titles: list[str],
        categories: list[str],
        labels: list[int],
        weights: list[float],
    ):
        self.queries = queries
        self.titles = titles
        self.categories = categories
        self.labels = labels
        self.weights = weights

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "query": self.queries[idx],
            "title": self.titles[idx],
            "category": self.categories[idx],
            "label": self.labels[idx],
            "weight": self.weights[idx],
        }


def collate_fn_builder(tokenizer, max_length: int = 256):
    """Pads query-product texts dynamically to the longest item in the batch."""
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = []
        labels = []
        weights = []

        for item in batch:
            text = f"{item['query']} | {item['title']}"
            if item["category"]:
                text += f" | {item['category']}"
            texts.append(text)
            labels.append(item["label"])
            weights.append(item["weight"])

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32),
            "weights": torch.tensor(weights, dtype=torch.float32),
        }

    return collate_fn


def train_model(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = "xlm-roberta-base",
    epochs: int = 3,
    debug: bool = False,
    device: str | None = None,
) -> tuple[str, dict]:
    """
    Fine-tunes the cross encoder using OOF split.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    # Read hyperparameters
    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("learning_rate", 2e-5))
    batch_size = int(train_cfg.get("batch_size", 16))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 4))
    max_length = int(train_cfg.get("max_length", 256))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    patience = int(train_cfg.get("early_stopping_patience", 3))

    # Output paths
    model_save_dir = Path(config["data"]["processed_dir"]) / f"model_{model_name}"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer & model
    logger.info(f"Loading pretrained tokenizer & model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(device)

    # Prepare datasets
    logger.info("Preparing datasets...")
    train_dataset = PairDataset(
        queries=train_df["query"].fillna("").tolist(),
        titles=train_df["title"].fillna("").tolist(),
        categories=train_df["category"].fillna("").tolist(),
        labels=train_df["label"].tolist(),
        weights=train_df["sample_weight"].tolist(),
    )
    val_dataset = PairDataset(
        queries=val_df["query"].fillna("").tolist(),
        titles=val_df["title"].fillna("").tolist(),
        categories=val_df["category"].fillna("").tolist(),
        labels=val_df["label"].tolist(),
        weights=val_df["sample_weight"].tolist(),
    )

    collate = collate_fn_builder(tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collate)

    # Optimization
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    total_steps = len(train_loader) // grad_accum * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # AMP Scaler
    use_amp = train_cfg.get("fp16", True) and ("cuda" in device)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    device_type = "cuda" if "cuda" in device else "cpu"

    # Evaluators
    threshold_opt = ThresholdOptimizer(config)
    validator = Validator(config)

    # Early stopping state
    best_f1 = -1.0
    best_threshold = 0.5
    no_improvement_epochs = 0
    best_metrics = {}

    logger.info(f"Starting training for {epochs} epochs (Total steps: {total_steps})...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for step, batch in enumerate(train_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)

            # Mixed precision training
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = compute_weighted_loss(logits, labels, weights)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

            epoch_loss += loss.item() * grad_accum
            train_bar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})

            if debug and step >= 10:  # Break early for debug run
                break

        # --- Validation Epoch End ---
        model.eval()
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels = batch["labels"].numpy()

                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.squeeze(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()

                val_probs.extend(probs)
                val_labels.extend(batch_labels)

                if debug and len(val_probs) >= 100:
                    break

        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels)

        # Optimize threshold on val probabilities
        val_fold_df = val_df.head(len(val_probs)).copy()
        val_fold_df["label"] = val_labels
        # Generate temporary OOF metrics
        val_fold_df["fold"] = 0  # mock single fold

        opt_res = threshold_opt.optimize(val_labels, val_probs)
        epoch_threshold = opt_res["best_threshold"]

        # Run full validation scenario check
        metrics = validator.evaluate_oof(val_fold_df, val_probs, epoch_threshold)
        epoch_f1 = metrics["overall_macro_f1"]

        logger.info(f"Epoch {epoch+1} Results — Loss: {epoch_loss/len(train_loader):.4f} | F1: {epoch_f1:.4f} | Thresh: {epoch_threshold:.4f}")

        # Check improvement
        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_threshold = epoch_threshold
            best_metrics = metrics
            no_improvement_epochs = 0

            # Save best checkpoint
            logger.info(f"  ✓ Found new best model (F1: {best_f1:.4f}). Saving model to {model_save_dir}...")
            model.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)
        else:
            no_improvement_epochs += 1
            logger.info(f"  No improvement for {no_improvement_epochs} epoch(s). Best F1: {best_f1:.4f}")

        if no_improvement_epochs >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Save final OOF best metadata
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Best Macro F1: {best_f1:.4f} at threshold {best_threshold:.4f}")
    logger.info("=" * 60)

    # Log to registry
    tracker = ExperimentTracker(config)
    hyperparams = {
        "model": model_name,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "grad_accum": grad_accum,
        "max_length": max_length,
    }
    exp_id = tracker.log_experiment(
        model_name=model_name,
        metrics=best_metrics,
        hyperparams=hyperparams,
        notes="V1 baseline Deep Learning cross encoder model" if not debug else "Debug smoke test run",
    )

    return str(model_save_dir), best_metrics
