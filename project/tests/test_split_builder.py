"""Tests for split_builder module."""
import pytest
import pandas as pd
from project.data.split_builder import build_splits


def test_build_splits_no_leakage():
    """Test that build_splits creates folds with zero query leakage."""
    # Create sample training data with duplicate queries and different items
    df = pd.DataFrame({
        "query": [
            "siyah ceket", "siyah ceket", "kırmızı mont", "kırmızı mont",
            "mavi kot", "mavi kot", "yeşil şapka", "yeşil şapka",
            "sarı tişört", "sarı tişört"
        ],
        "item_id": [f"ITEM_{i}" for i in range(10)],
        "label": [1] * 10
    })

    # Run split
    n_splits = 3
    result = build_splits(df, n_splits=n_splits)

    # 1. Verify fold values are between 0 and n_splits-1
    assert set(result["fold"].unique()) == set(range(n_splits))

    # 2. Verify complete query isolation between train/val in each fold
    for f in range(n_splits):
        train_queries = set(result[result["fold"] != f]["query"])
        val_queries = set(result[result["fold"] == f]["query"])

        # Intersection must be empty (0 leakage)
        assert train_queries.isdisjoint(val_queries)


def test_build_splits_case_insensitivity():
    """Test that queries differing only by casing/whitespace are grouped together."""
    df = pd.DataFrame({
        "query": [
            "Siyah Ceket", "siyah  ceket", "Kırmızı Mont", "kırmızı mont"
        ],
        "item_id": [f"ITEM_{i}" for i in range(4)],
        "label": [1] * 4
    })

    # Split into 2 folds
    result = build_splits(df, n_splits=2)

    # Both "Siyah Ceket" variations must belong to the same fold
    ceket_folds = result[result["query"].str.lower().str.contains("ceket")]["fold"].unique()
    assert len(ceket_folds) == 1

    # Both "Kırmızı Mont" variations must belong to the same fold
    mont_folds = result[result["query"].str.lower().str.contains("mont")]["fold"].unique()
    assert len(mont_folds) == 1
