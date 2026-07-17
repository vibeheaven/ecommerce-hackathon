"""
Heuristic Baseline (V0) — Fast rule-based relevance scoring.

Purpose:
  - Validate submission pipeline end-to-end
  - Test CSV format and Kaggle upload
  - Establish reference score for cross encoder
  - Detect join errors

Features used:
  - Query-title token overlap
  - Brand match
  - Category token overlap
  - Gender conflict detection
  - Color conflict detection (query-dependent)
  - BM25-like TF-IDF similarity
  - Weighted scoring + threshold
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re

from project.utils.text_cleaner import clean_index_text, clean_model_text, clean_brand, clean_gender
from project.utils.attribute_parser import parse_attributes, get_color
from project.utils.logging_utils import setup_logger

logger = setup_logger("heuristic_baseline")


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization on cleaned text."""
    if not text:
        return []
    return text.split()


def token_overlap_ratio(query_tokens: list[str], title_tokens: list[str]) -> float:
    """Fraction of query tokens found in title."""
    if not query_tokens:
        return 0.0
    title_set = set(title_tokens)
    matches = sum(1 for t in query_tokens if t in title_set)
    return matches / len(query_tokens)


def jaccard_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Jaccard similarity between two token sets."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def brand_in_query(query_index: str, brand: str | None) -> bool:
    """Check if brand name appears in query."""
    if not brand or not isinstance(brand, str):
        return False
    brand_clean = clean_index_text(brand)
    if not brand_clean:
        return False
    return brand_clean in query_index


def category_overlap(query_tokens: list[str], category: str | None) -> float:
    """Token overlap between query and category hierarchy."""
    if not category or not isinstance(category, str):
        return 0.0
    cat_clean = clean_index_text(category.replace("/", " "))
    cat_tokens = tokenize(cat_clean)
    return token_overlap_ratio(query_tokens, cat_tokens)


def detect_gender_conflict(query_index: str, product_gender: str | None) -> bool:
    """
    Detect gender conflict: query specifies one gender, product is another.
    Only triggers if query mentions a gender explicitly.
    """
    product_gender = clean_gender(product_gender)
    if product_gender == "unknown" or product_gender == "unisex":
        return False

    query_genders = []
    if "erkek" in query_index and "kadın" not in query_index:
        query_genders.append("erkek")
    elif "kadın" in query_index or "kadin" in query_index:
        query_genders.append("kadın")

    if not query_genders:
        return False  # Query doesn't mention gender

    return product_gender not in query_genders


def detect_color_conflict(query_index: str, attributes: str | None) -> bool:
    """
    Detect color conflict: query specifies a color, product has a different color.
    Only triggers if query mentions a color explicitly (query-dependent).
    """
    colors = [
        "siyah", "beyaz", "kirmizi", "mavi", "yesil", "sari", "pembe",
        "mor", "turuncu", "gri", "kahverengi", "lacivert", "bej",
        "krem", "bordo", "haki", "ekru", "antrasit",
    ]

    query_colors = [c for c in colors if c in query_index]
    if not query_colors:
        return False  # Query doesn't mention color

    if not attributes or not isinstance(attributes, str):
        return False

    parsed = parse_attributes(attributes)
    product_color = get_color(parsed)
    if not product_color:
        return False

    product_color_index = clean_index_text(product_color)

    # Check if any query color matches product color
    for qc in query_colors:
        if qc in product_color_index:
            return False  # Match found, no conflict

    return True  # Query specifies color, product has different color


def score_pair(
    query: str | None,
    title: str | None,
    category: str | None,
    brand: str | None,
    gender: str | None,
    attributes: str | None,
) -> float:
    """
    Compute heuristic relevance score for a query-product pair.
    Returns float between 0.0 and 1.0.
    """
    if not query or not title:
        return 0.0

    query_index = clean_index_text(query)
    title_index = clean_index_text(title)

    query_tokens = tokenize(query_index)
    title_tokens = tokenize(title_index)

    if not query_tokens:
        return 0.0

    # Feature 1: Token overlap (strongest signal)
    overlap = token_overlap_ratio(query_tokens, title_tokens)

    # Feature 2: Jaccard similarity
    jaccard = jaccard_similarity(query_tokens, title_tokens)

    # Feature 3: Brand match
    brand_match = 1.0 if brand_in_query(query_index, brand) else 0.0

    # Feature 4: Category overlap
    cat_overlap = category_overlap(query_tokens, category)

    # Feature 5: Gender conflict (penalty)
    gender_penalty = -0.3 if detect_gender_conflict(query_index, gender) else 0.0

    # Feature 6: Color conflict (penalty, query-dependent)
    color_penalty = -0.2 if detect_color_conflict(query_index, attributes) else 0.0

    # Weighted combination
    score = (
        0.40 * overlap
        + 0.20 * jaccard
        + 0.15 * brand_match
        + 0.15 * cat_overlap
        + gender_penalty
        + color_penalty
    )

    return max(0.0, min(1.0, score))


def run_heuristic_baseline(
    submission_merged: pd.DataFrame,
    threshold: float = 0.35,
    max_samples: int | None = None,
) -> pd.Series:
    """
    Run heuristic baseline on all submission pairs.

    Args:
        submission_merged: DataFrame with query, title, category, brand, gender, attributes
        threshold: decision threshold for binary prediction
        max_samples: maximum number of samples to process

    Returns:
        Series of 0/1 predictions
    """
    df_to_process = submission_merged
    if max_samples is not None:
        df_to_process = submission_merged.head(max_samples)

    logger.info(f"Running heuristic baseline on {len(df_to_process):,} pairs...")
    logger.info(f"  Threshold: {threshold}")

    scores = []
    for idx, row in df_to_process.iterrows():
        s = score_pair(
            query=row.get("query"),
            title=row.get("title"),
            category=row.get("category"),
            brand=row.get("brand"),
            gender=row.get("gender"),
            attributes=row.get("attributes"),
        )
        scores.append(s)

        if (idx + 1) % 500_000 == 0:
            logger.info(f"  Processed {idx + 1:,} / {len(df_to_process):,}")

    scores = pd.Series(scores)
    predictions = (scores >= threshold).astype(int)

    pos_count = predictions.sum()
    total = len(predictions)
    logger.info(f"  Scores — min: {scores.min():.4f}, max: {scores.max():.4f}, mean: {scores.mean():.4f}")
    logger.info(f"  Predictions — positive: {pos_count:,} ({pos_count/total*100:.2f}%), negative: {total - pos_count:,}")

    return predictions


if __name__ == "__main__":
    import yaml
    import argparse
    from project.data.data_loader import load_all
    from project.submission.submission_generator import generate_submission
    from project.submission.submission_validator import validate_submission

    parser = argparse.ArgumentParser(description="Generate V0 heuristic baseline submission")
    parser.add_argument("--config", default="project/configs/config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load data
    data = load_all(".", config)

    # Run heuristic
    predictions = run_heuristic_baseline(
        data["submission_merged"],
        threshold=args.threshold,
        max_samples=args.max_samples
    )

    # Slice submission_pairs to match predictions size
    sub_pairs = data["submission_pairs"]
    expected_rows = config["submission"]["required_row_count"]
    if args.max_samples is not None:
        sub_pairs = sub_pairs.head(args.max_samples)
        expected_rows = args.max_samples

    # Generate submission
    output_path = generate_submission(
        submission_pairs=sub_pairs,
        predictions=predictions,
        output_dir=config["submission"]["output_dir"],
        version=0,
    )

    # Validate
    raw_dir = Path(config["data"]["raw_dir"])
    sample_path = raw_dir / config["data"]["files"]["sample_submission"]
    result = validate_submission(
        submission_path=output_path,
        sample_submission_path=sample_path if args.max_samples is None else None,
        expected_row_count=expected_rows,
    )

    if result["valid"]:
        logger.info(f"\n✓ V0 baseline ready: {output_path}")
        if args.max_samples is None:
            logger.info("Next step: kaggle competitions submit -c trendyol-e-ticaret-yarismasi-2026-kaggle -f <file> -m 'V000 heuristic baseline'")
    else:
        logger.error("\n✗ Submission validation failed!")
        exit(1)
