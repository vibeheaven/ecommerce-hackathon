"""Tests for data_loader module using mock CSV data."""
import pytest
import pandas as pd
from pathlib import Path
from project.data.data_loader import (
    load_items, load_terms, load_training_pairs,
    load_submission_pairs, load_sample_submission,
    merge_training_data, merge_submission_data,
)


@pytest.fixture
def mock_raw_dir(tmp_path):
    """Create mock CSV files in a temporary directory."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # 1. items.csv
    items_data = pd.DataFrame({
        "item_id": ["ITEM_1", "ITEM_2"],
        "title": ["Erkek Oversize Tişört", "Kadın Çanta"],
        "category": ["Giyim/Tişört", "Aksesuar/Çanta"],
        "brand": ["Defacto", "Mango"],
        "gender": ["Erkek", "Kadın"],
        "age_group": ["Yetişkin", "Yetişkin"],
        "attributes": ["renk: siyah, materyal: pamuk", "renk: kırmızı"],
    })
    items_data.to_csv(raw_dir / "items.csv", index=False)

    # 2. terms.csv
    terms_data = pd.DataFrame({
        "term_id": ["TERM_1", "TERM_2"],
        "query": ["siyah tişört", "kırmızı çanta"],
    })
    terms_data.to_csv(raw_dir / "terms.csv", index=False)

    # 3. training_pairs.csv
    training_data = pd.DataFrame({
        "id": ["TRN_1", "TRN_2"],
        "term_id": ["TERM_1", "TERM_2"],
        "item_id": ["ITEM_1", "ITEM_2"],
        "label": [1, 1],
    })
    training_data.to_csv(raw_dir / "training_pairs.csv", index=False)

    # 4. submission_pairs.csv
    submission_data = pd.DataFrame({
        "id": ["TST_1", "TST_2"],
        "term_id": ["TERM_1", "TERM_2"],
        "item_id": ["ITEM_1", "ITEM_2"],
    })
    submission_data.to_csv(raw_dir / "submission_pairs.csv", index=False)

    # 5. sample_submission.csv
    sample_sub = pd.DataFrame({
        "id": ["TST_1", "TST_2"],
        "prediction": [0, 0],
    })
    sample_sub.to_csv(raw_dir / "sample_submission.csv", index=False)

    return raw_dir


def test_load_all_files(mock_raw_dir):
    """Test individual file loading functions."""
    items = load_items(mock_raw_dir / "items.csv")
    assert len(items) == 2
    assert "item_id" in items.columns
    assert items.loc[0, "brand"] == "Defacto"

    terms = load_terms(mock_raw_dir / "terms.csv")
    assert len(terms) == 2
    assert "term_id" in terms.columns

    train_pairs = load_training_pairs(mock_raw_dir / "training_pairs.csv")
    assert len(train_pairs) == 2
    assert train_pairs.loc[0, "label"] == 1

    sub_pairs = load_submission_pairs(mock_raw_dir / "submission_pairs.csv")
    assert len(sub_pairs) == 2

    sample_sub = load_sample_submission(mock_raw_dir / "sample_submission.csv")
    assert len(sample_sub) == 2
    assert sample_sub.loc[0, "prediction"] == 0


def test_merge_data(mock_raw_dir):
    """Test training and submission data merging."""
    items = load_items(mock_raw_dir / "items.csv")
    terms = load_terms(mock_raw_dir / "terms.csv")
    train_pairs = load_training_pairs(mock_raw_dir / "training_pairs.csv")
    sub_pairs = load_submission_pairs(mock_raw_dir / "submission_pairs.csv")

    # Merge train
    train_merged = merge_training_data(train_pairs, terms, items)
    assert len(train_merged) == 2
    assert "query" in train_merged.columns
    assert "title" in train_merged.columns
    assert train_merged.loc[0, "query"] == "siyah tişört"
    assert train_merged.loc[0, "title"] == "Erkek Oversize Tişört"

    # Merge submission
    sub_merged = merge_submission_data(sub_pairs, terms, items)
    assert len(sub_merged) == 2
    assert "query" in sub_merged.columns
    assert "title" in sub_merged.columns
    assert sub_merged.loc[1, "query"] == "kırmızı çanta"
    assert sub_merged.loc[1, "title"] == "Kadın Çanta"
