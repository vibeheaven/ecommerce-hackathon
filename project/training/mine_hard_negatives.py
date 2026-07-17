"""
Mine Hard Negatives — Round-2 self-mining with the current best checkpoint.

For every unique training query, candidate items are gathered from the
hardest heuristic pools (lexical overlap + same category), filtered through
the probable-positive filter (label-noise control), and scored with the
trained cross encoder. Candidates the MODEL believes are positive
(prob >= --min-prob) but the heuristics confidently call negative are the
most informative training examples; they are appended to the sampled dataset
as negative_type='mined_hard' with a conservative sample weight.

Usage:
    python -m project.training.mine_hard_negatives \
        --model-dir project/data/processed/model_xlm-roberta-base_fold0 \
        --output dataset_sampled_mined.parquet
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from project.utils.logging_utils import setup_logger
from project.utils.text_cleaner import clean_index_text
from project.utils.pair_text import build_query_text, build_product_text
from project.data.data_loader import load_items
from project.negative_samples.negative_sampler import NegativeSampler
from project.inference.inference import check_pair_text_version
from project.inference.batch_builder import PairInferenceDataset, collate_fn_builder

logger = setup_logger("mine_hard_negatives")


def gather_candidates(
    sampler: NegativeSampler,
    queries_df: pd.DataFrame,
    per_query: int,
    seed: int,
) -> pd.DataFrame:
    """Collect PP-filtered hard candidates per unique query (no scoring yet)."""
    catalog = sampler.catalog
    rows = []

    for row in tqdm(queries_df.itertuples(index=False), total=len(queries_df), desc="Candidates"):
        query, category, fold, term_id = row.query, row.category, row.fold, row.term_id
        norm_q = clean_index_text(query)
        rng = random.Random(f"{seed}:mine:{norm_q}")

        exclude_pos = sampler.known_positive_pos.get(norm_q, set())
        used = sampler._used_negatives.get(norm_q, set())
        anchor_category = category.strip() if isinstance(category, str) else ""

        pool: list[int] = [pos for pos, _ in sampler._lexical_candidates(norm_q)[: per_query * 3]]
        group = catalog.cat_groups.get(anchor_category)
        if group is not None and len(group) > 0:
            n_extra = min(len(group), per_query * 3)
            pool += [int(group[i]) for i in rng.sample(range(len(group)), n_extra)]

        q_tokens = norm_q.split()
        accepted = 0
        seen = set()
        for pos in pool:
            if accepted >= per_query:
                break
            if pos in seen or pos in exclude_pos or pos in used:
                continue
            seen.add(pos)

            title_tokens = catalog.clean_titles[pos].split()
            category_match = (catalog.cat_clean[pos] == anchor_category) if anchor_category else None
            if sampler.filter.is_probable_positive(
                query=norm_q,
                product_title=catalog.clean_titles[pos],
                category_match=category_match,
                query_tokens=q_tokens,
                title_tokens=title_tokens,
            ):
                continue

            rows.append({
                "term_id": term_id,
                "fold": fold,
                "query": query,
                "pos": pos,
            })
            accepted += 1

    return pd.DataFrame(rows)


def score_candidates(
    candidates: pd.DataFrame,
    catalog,
    model_dir: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    """Score (query, candidate) pairs with the trained cross encoder."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    queries = [build_query_text(q) for q in candidates["query"].tolist()]
    products = [
        build_product_text(
            catalog.titles[pos], catalog.brands[pos], catalog.categories[pos],
            catalog.genders_raw[pos], catalog.age_groups[pos], catalog.attributes[pos],
        )
        for pos in candidates["pos"].tolist()
    ]

    dataset = PairInferenceDataset(queries, products)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_builder(tokenizer, max_length),
    )

    probs = []
    device_type = "cuda" if "cuda" in device else "cpu"
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring"):
            with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits.squeeze(-1)
            probs.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
    return np.asarray(probs)


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives with the current best model")
    parser.add_argument("--config", default="project/configs/config.yaml")
    parser.add_argument("--neg-config", default="project/configs/negative_sampling.yaml")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset", default="dataset_sampled.parquet")
    parser.add_argument("--output", default="dataset_sampled_mined.parquet")
    parser.add_argument("--candidates-per-query", type=int, default=40)
    parser.add_argument("--min-prob", type=float, default=0.60, help="Model prob above which a heuristic negative counts as 'hard'")
    parser.add_argument("--max-per-query", type=int, default=3)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.neg_config) as f:
        neg_config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_pair_text_version(args.model_dir)

    processed_dir = Path(config["data"]["processed_dir"])
    dataset_path = processed_dir / args.dataset
    dataset = pd.read_parquet(dataset_path)
    positives = dataset[dataset["label"] == 1]

    # One anchor row per unique query (majority category of its positives)
    queries_df = (
        positives.groupby("query", as_index=False)
        .agg(category=("category", lambda s: s.mode().iat[0] if len(s.mode()) else ""),
             fold=("fold", "first"), term_id=("term_id", "first"))
    )
    logger.info(f"Mining over {len(queries_df):,} unique queries...")

    raw_dir = Path(config["data"]["raw_dir"])
    items_df = load_items(raw_dir / config["data"]["files"]["items"])
    sampler = NegativeSampler(config, neg_config, items_df, positives)

    # Register existing negatives so mining doesn't duplicate them
    negs = dataset[dataset["label"] == 0]
    for query, item_id in zip(negs["query"].tolist(), negs["item_id"].tolist()):
        pos = sampler.catalog.pos_by_item_id.get(item_id)
        if pos is not None:
            sampler._used_negatives.setdefault(clean_index_text(query), set()).add(pos)

    seed = int(config.get("project", {}).get("seed", 42))
    candidates = gather_candidates(sampler, queries_df, args.candidates_per_query, seed)
    logger.info(f"Collected {len(candidates):,} PP-clean candidates. Scoring with {args.model_dir}...")

    probs = score_candidates(
        candidates, sampler.catalog, args.model_dir,
        batch_size=int(config["inference"]["batch_size"]),
        max_length=int(config["inference"]["max_length"]),
        device=device,
    )
    candidates = candidates.assign(prob=probs)

    hard = candidates[candidates["prob"] >= args.min_prob]
    hard = (
        hard.sort_values("prob", ascending=False)
        .groupby("query", group_keys=False)
        .head(args.max_per_query)
    )
    logger.info(
        f"Mined {len(hard):,} hard negatives "
        f"({len(hard) / max(len(candidates), 1) * 100:.1f}% of scored candidates, "
        f"min_prob={args.min_prob})"
    )

    catalog = sampler.catalog
    mined_rows = []
    for row in hard.itertuples(index=False):
        pos = row.pos
        mined_rows.append({
            "term_id": row.term_id,
            "item_id": catalog.item_ids[pos],
            "query": row.query,
            "title": catalog.titles[pos],
            "category": catalog.categories[pos],
            "brand": catalog.brands[pos],
            "gender": catalog.genders_raw[pos],
            "age_group": catalog.age_groups[pos],
            "attributes": catalog.attributes[pos],
            "label": 0,
            # Conservative weight — mined labels carry heuristic (not human) certainty
            "sample_weight": 0.8,
            "negative_type": "mined_hard",
            "fold": row.fold,
        })

    mined_df = pd.DataFrame(mined_rows)
    combined = pd.concat([dataset, mined_df], ignore_index=True)
    output_path = processed_dir / args.output
    combined.to_parquet(output_path, index=False)

    logger.info(f"✓ Wrote {output_path} ({len(combined):,} rows, +{len(mined_df):,} mined_hard)")
    logger.info("  Next: python -m project.training.run_trainer --dataset "
                f"{args.output} --run-tag mined")


if __name__ == "__main__":
    main()
