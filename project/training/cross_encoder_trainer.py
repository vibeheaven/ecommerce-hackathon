"""
Cross Encoder Trainer — Fine-tunes XLM-RoBERTa / BERTürk sequence classification.

v5 changes:
  * Deterministic: global seeding (python / numpy / torch / dataloader workers).
  * Model input built through project.utils.pair_text (single source of truth,
    shared with inference — no train/serve skew) and enriched with brand,
    category, gender, age group and whitelisted attributes.
  * Every epoch logs and persists the full evaluation: Macro F1, per-class
    P/R/F1, confusion matrix, negative-type TNR report, scenario scores and
    the optimized threshold — appended to <model_dir>/metrics_history.json.
  * The best checkpoint (by Macro F1) is saved together with
    <model_dir>/inference_meta.json (threshold + input-format version), so
    inference never has to guess the threshold from the experiment registry.
  * Validation predictions of the best epoch are saved for OOF/ensemble use.
"""
import json
import random

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Any
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

from project.utils.logging_utils import setup_logger
from project.utils.pair_text import build_pair_texts_from_frame, PAIR_TEXT_VERSION
from project.training.sample_weighting import compute_weighted_loss
from project.training.threshold_optimizer import ThresholdOptimizer
from project.validation.validator import Validator
from project.experiments.tracker import ExperimentTracker

logger = setup_logger("cross_encoder_trainer")


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Full determinism (deterministic kernels) costs ~20% speed; we settle for
    # seeded init + seeded shuffling, which makes runs comparable in practice.


class PairDataset(Dataset):
    """Query / product-text pairs with labels and sample weights."""

    def __init__(self, queries: list[str], products: list[str], labels: list[int], weights: list[float]):
        self.queries = queries
        self.products = products
        self.labels = labels
        self.weights = weights

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "query": self.queries[idx],
            "product": self.products[idx],
            "label": self.labels[idx],
            "weight": self.weights[idx],
        }


class PairCollator:
    """Tokenizes (query, product) as a text pair with dynamic padding.

    A class (not a closure) so DataLoader workers can pickle it under
    Python 3.14's forkserver start method.
    """

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [item["query"] for item in batch],
            [item["product"] for item in batch],
            padding=True,
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
            "weights": torch.tensor([item["weight"] for item in batch], dtype=torch.float32),
        }


def collate_fn_builder(tokenizer, max_length: int = 256) -> PairCollator:
    return PairCollator(tokenizer, max_length)


def _make_dataset(df: pd.DataFrame) -> PairDataset:
    queries, products = build_pair_texts_from_frame(df)
    return PairDataset(
        queries=queries,
        products=products,
        labels=df["label"].tolist(),
        weights=df["sample_weight"].tolist(),
    )


def train_model(
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = "xlm-roberta-base",
    epochs: int | None = None,
    debug: bool = False,
    device: str | None = None,
    val_fold: int = 0,
    run_tag: str = "",
) -> tuple[str, dict]:
    """Fine-tune the cross encoder; validate on the held-out fold each epoch."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    train_cfg = config.get("training", {})
    seed = int(config.get("project", {}).get("seed", 42))
    set_global_seed(seed + val_fold)

    lr = float(train_cfg.get("learning_rate", 2e-5))
    batch_size = int(train_cfg.get("batch_size", 16))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 4))
    max_length = int(train_cfg.get("max_length", 256))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    patience = int(train_cfg.get("early_stopping_patience", 3))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    pos_weight = float(train_cfg.get("pos_weight", 1.0))
    num_workers = int(train_cfg.get("num_workers", 2))
    if epochs is None:
        epochs = int(train_cfg.get("epochs", 5))

    suffix = f"_fold{val_fold}" if val_fold else ""
    if run_tag:
        suffix += f"_{run_tag}"
    model_save_dir = Path(config["data"]["processed_dir"]) / f"model_{model_name}{suffix}"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading pretrained tokenizer & model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(device)

    logger.info(f"Preparing datasets (input format: {PAIR_TEXT_VERSION})...")
    train_dataset = _make_dataset(train_df)
    val_dataset = _make_dataset(val_df)
    collate = collate_fn_builder(tokenizer, max_length)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + val_fold)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
        num_workers=num_workers, pin_memory=("cuda" in device), generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collate,
        num_workers=num_workers, pin_memory=("cuda" in device),
    )

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

    total_steps = max(1, len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = train_cfg.get("fp16", True) and ("cuda" in device)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    device_type = "cuda" if "cuda" in device else "cpu"

    threshold_opt = ThresholdOptimizer(config)
    validator = Validator(config)

    best_f1 = -1.0
    best_threshold = 0.5
    best_epoch = -1
    no_improvement_epochs = 0
    best_metrics: dict = {}
    metrics_history: list[dict] = []
    history_path = model_save_dir / "metrics_history.json"

    logger.info(f"Starting training for {epochs} epochs (Total optim steps: {total_steps})...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for step, batch in enumerate(train_bar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            weights = batch["weights"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = compute_weighted_loss(logits, labels, weights, pos_weight=pos_weight)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum
            train_bar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})

            if debug and step >= 10:
                break

        # --- Validation at epoch end ---
        model.eval()
        val_probs: list[float] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.sigmoid(outputs.logits.squeeze(-1)).float().cpu().numpy()
                val_probs.extend(probs.tolist())
                if debug and len(val_probs) >= 3200:
                    break

        val_probs_arr = np.asarray(val_probs)
        val_eval_df = val_df.iloc[: len(val_probs_arr)]

        opt_res = threshold_opt.optimize(val_eval_df["label"].to_numpy(), val_probs_arr)
        epoch_threshold = opt_res["best_threshold"]

        metrics = validator.evaluate_oof(val_eval_df, val_probs_arr, epoch_threshold)
        epoch_f1 = metrics["overall_macro_f1"]
        avg_loss = epoch_loss / max(len(train_loader), 1)

        logger.info(
            f"Epoch {epoch+1} Results — Loss: {avg_loss:.4f} | F1: {epoch_f1:.4f} | "
            f"Thresh: {epoch_threshold:.4f}"
        )

        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "threshold": epoch_threshold,
            **metrics,
        })
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(metrics_history, f, indent=2, ensure_ascii=False, default=float)

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_threshold = epoch_threshold
            best_metrics = metrics
            best_epoch = epoch + 1
            no_improvement_epochs = 0

            logger.info(f"  ✓ New best model (F1: {best_f1:.4f}). Saving to {model_save_dir}...")
            model.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)

            meta = {
                "base_model": model_name,
                "val_fold": val_fold,
                "best_epoch": best_epoch,
                "macro_f1": best_f1,
                "threshold": best_threshold,
                "pair_text_version": PAIR_TEXT_VERSION,
                "max_length": max_length,
                "seed": seed,
            }
            with open(model_save_dir / "inference_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            # Persist best-epoch validation predictions for OOF threshold / ensembling
            np.save(model_save_dir / "val_probs.npy", val_probs_arr)
            val_eval_df[["term_id", "item_id", "label", "negative_type", "fold"]].to_parquet(
                model_save_dir / "val_index.parquet", index=False
            )
        else:
            no_improvement_epochs += 1
            logger.info(f"  No improvement for {no_improvement_epochs} epoch(s). Best F1: {best_f1:.4f}")

        if no_improvement_epochs >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs (best epoch: {best_epoch}).")
            break

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Best Macro F1: {best_f1:.4f} at threshold {best_threshold:.4f} (epoch {best_epoch})")
    logger.info("=" * 60)

    tracker = ExperimentTracker(config)
    hyperparams = {
        "model": model_name,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "grad_accum": grad_accum,
        "max_length": max_length,
        "pos_weight": pos_weight,
        "val_fold": val_fold,
        "seed": seed,
        "pair_text_version": PAIR_TEXT_VERSION,
        "run_tag": run_tag,
    }
    tracker.log_experiment(
        model_name=f"{model_name}{suffix}",
        metrics=best_metrics,
        hyperparams=hyperparams,
        notes="Debug smoke test run" if debug else f"v5 pipeline — fold {val_fold}",
    )

    return str(model_save_dir), best_metrics
