"""
Build Embedding Neighbors — Precomputes semantic nearest-neighbor candidates
for the embedding_hard negative strategy.

Encodes unique training queries and all item titles with a multilingual
sentence-transformer, then retrieves the top-K most similar items per query
via FAISS (inner product over normalized vectors). Own positives are NOT
excluded here — the sampler handles exclusions and the probable-positive
filter at sampling time.

Output: <processed_dir>/embedding_neighbors.parquet
    columns: query_norm (str), item_id (str), sim (float32)

Usage:
    python -m project.embeddings.build_embedding_neighbors --top-k 30
Then set strategy_ratios.embedding_hard > 0 and rebuild the dataset:
    python -m project.data.build_sampled_dataset
"""
import argparse

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from project.utils.logging_utils import setup_logger
from project.utils.text_cleaner import clean_index_text
from project.data.data_loader import load_items

logger = setup_logger("build_embedding_neighbors")

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_embedding_neighbors(processed_dir: Path, pos_by_item_id: dict) -> dict[str, list[tuple[int, float]]]:
    """Load neighbors parquet into the sampler's expected format."""
    path = processed_dir / "embedding_neighbors.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run python -m project.embeddings.build_embedding_neighbors first."
        )
    df = pd.read_parquet(path)
    neighbors: dict[str, list[tuple[int, float]]] = {}
    for query_norm, group in df.groupby("query_norm"):
        entries = []
        for item_id, sim in zip(group["item_id"], group["sim"]):
            pos = pos_by_item_id.get(item_id)
            if pos is not None:
                entries.append((pos, float(sim)))
        neighbors[query_norm] = entries
    return neighbors


def main():
    parser = argparse.ArgumentParser(description="Precompute semantic neighbors for embedding_hard negatives")
    parser.add_argument("--config", default="project/configs/config.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-items", type=int, default=None, help="Debug: limit item count")
    args = parser.parse_args()

    # Imported here so the rest of the pipeline works without these packages
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])

    splits_path = processed_dir / "train_splits.parquet"
    if not splits_path.exists():
        raise FileNotFoundError(f"{splits_path} not found. Run split_builder first.")
    splits_df = pd.read_parquet(splits_path)
    queries = sorted({clean_index_text(q) for q in splits_df["query"].dropna() if q})
    logger.info(f"Encoding {len(queries):,} unique queries...")

    items_df = load_items(raw_dir / config["data"]["files"]["items"])
    if args.max_items:
        items_df = items_df.head(args.max_items)
    titles = items_df["title"].fillna("").tolist()
    item_ids = items_df["item_id"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)

    query_emb = model.encode(
        queries, batch_size=args.batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    logger.info(f"Encoding {len(titles):,} item titles (this is the long part)...")
    item_emb = model.encode(
        titles, batch_size=args.batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    logger.info("Building FAISS index and searching...")
    index = faiss.IndexFlatIP(item_emb.shape[1])
    index.add(item_emb)
    sims, ids = index.search(query_emb, args.top_k)

    rows = {
        "query_norm": np.repeat(queries, args.top_k),
        "item_id": [item_ids[i] for i in ids.ravel()],
        "sim": sims.ravel().astype(np.float32),
    }
    out_df = pd.DataFrame(rows)
    out_path = processed_dir / "embedding_neighbors.parquet"
    out_df.to_parquet(out_path, index=False)
    logger.info(f"✓ Wrote {out_path} ({len(out_df):,} rows, top-{args.top_k} per query)")


if __name__ == "__main__":
    main()
