"""
Data Loader — Loads and merges all CSV files into unified DataFrames.
Handles CSV quoting issues and memory-efficient loading.
"""
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

from project.utils.logging_utils import setup_logger

logger = setup_logger("data_loader")


@dataclass
class DataPaths:
    """Resolved paths to all data files."""
    items: Path
    terms: Path
    training_pairs: Path
    submission_pairs: Path
    sample_submission: Path


def resolve_paths(base_dir: str, config: dict) -> DataPaths:
    """Resolve data file paths from config."""
    raw_dir = Path(base_dir) / config["data"]["raw_dir"]
    files = config["data"]["files"]
    return DataPaths(
        items=raw_dir / files["items"],
        terms=raw_dir / files["terms"],
        training_pairs=raw_dir / files["training_pairs"],
        submission_pairs=raw_dir / files["submission_pairs"],
        sample_submission=raw_dir / files["sample_submission"],
    )


def load_items(path: Path) -> pd.DataFrame:
    """Load items.csv with proper quoting for attribute field."""
    logger.info(f"Loading items from {path}")
    df = pd.read_csv(
        path,
        dtype={
            "item_id": "str",
            "title": "str",
            "category": "str",
            "brand": "str",
            "gender": "str",
            "age_group": "str",
            "attributes": "str",
        },
        quotechar='"',
        escapechar=None,
        on_bad_lines="warn",
        engine="python",
    )
    logger.info(f"  Loaded {len(df):,} items, columns: {list(df.columns)}")
    return df


def load_terms(path: Path) -> pd.DataFrame:
    """Load terms.csv."""
    logger.info(f"Loading terms from {path}")
    df = pd.read_csv(path, dtype={"term_id": "str", "query": "str"})
    logger.info(f"  Loaded {len(df):,} terms")
    return df


def load_training_pairs(path: Path) -> pd.DataFrame:
    """Load training_pairs.csv."""
    logger.info(f"Loading training pairs from {path}")
    df = pd.read_csv(
        path,
        dtype={"id": "str", "term_id": "str", "item_id": "str", "label": "int8"},
    )
    logger.info(f"  Loaded {len(df):,} training pairs, labels: {df['label'].value_counts().to_dict()}")
    return df


def load_submission_pairs(path: Path) -> pd.DataFrame:
    """Load submission_pairs.csv."""
    logger.info(f"Loading submission pairs from {path}")
    df = pd.read_csv(
        path, dtype={"id": "str", "term_id": "str", "item_id": "str"}
    )
    logger.info(f"  Loaded {len(df):,} submission pairs")
    return df


def load_sample_submission(path: Path) -> pd.DataFrame:
    """Load sample_submission.csv."""
    logger.info(f"Loading sample submission from {path}")
    df = pd.read_csv(path, dtype={"id": "str", "prediction": "int8"})
    logger.info(f"  Loaded {len(df):,} rows")
    return df


def merge_training_data(
    training_pairs: pd.DataFrame,
    terms: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge training_pairs with terms and items into a unified DataFrame.
    Every row contains: id, term_id, item_id, label, query, title, category,
    brand, gender, age_group, attributes
    """
    logger.info("Merging training data...")

    # Merge with terms
    merged = training_pairs.merge(terms, on="term_id", how="left")
    missing_terms = merged["query"].isna().sum()
    if missing_terms > 0:
        logger.warning(f"  {missing_terms:,} training pairs have missing term_id join")

    # Merge with items
    merged = merged.merge(items, on="item_id", how="left")
    missing_items = merged["title"].isna().sum()
    if missing_items > 0:
        logger.warning(f"  {missing_items:,} training pairs have missing item_id join")

    logger.info(f"  Merged training data: {len(merged):,} rows, {len(merged.columns)} columns")
    return merged


def merge_submission_data(
    submission_pairs: pd.DataFrame,
    terms: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.DataFrame:
    """Merge submission_pairs with terms and items."""
    logger.info("Merging submission data...")

    merged = submission_pairs.merge(terms, on="term_id", how="left")
    missing_terms = merged["query"].isna().sum()
    if missing_terms > 0:
        logger.warning(f"  {missing_terms:,} submission pairs have missing term_id join")

    merged = merged.merge(items, on="item_id", how="left")
    missing_items = merged["title"].isna().sum()
    if missing_items > 0:
        logger.warning(f"  {missing_items:,} submission pairs have missing item_id join")

    logger.info(f"  Merged submission data: {len(merged):,} rows")
    return merged


def load_all(base_dir: str, config: dict) -> dict:
    """
    Load all data files and return merged datasets.

    Returns dict with keys:
        - items, terms, training_pairs, submission_pairs, sample_submission
        - train_merged, submission_merged
    """
    paths = resolve_paths(base_dir, config)

    items = load_items(paths.items)
    terms = load_terms(paths.terms)
    training_pairs = load_training_pairs(paths.training_pairs)
    submission_pairs = load_submission_pairs(paths.submission_pairs)
    sample_submission = load_sample_submission(paths.sample_submission)

    train_merged = merge_training_data(training_pairs, terms, items)
    submission_merged = merge_submission_data(submission_pairs, terms, items)

    return {
        "items": items,
        "terms": terms,
        "training_pairs": training_pairs,
        "submission_pairs": submission_pairs,
        "sample_submission": sample_submission,
        "train_merged": train_merged,
        "submission_merged": submission_merged,
    }


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="Load and merge all data files")
    parser.add_argument("--config", default="project/configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data = load_all(".", config)
    print(f"\n{'='*60}")
    print(f"Items:            {len(data['items']):>12,}")
    print(f"Terms:            {len(data['terms']):>12,}")
    print(f"Training pairs:   {len(data['training_pairs']):>12,}")
    print(f"Submission pairs: {len(data['submission_pairs']):>12,}")
    print(f"Train merged:     {len(data['train_merged']):>12,}")
    print(f"Submission merged:{len(data['submission_merged']):>12,}")
    print(f"{'='*60}")
