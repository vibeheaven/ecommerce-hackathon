"""Tests for negative_sampler and related modules."""
import pytest
import pandas as pd
from project.negative_samples.negative_sampler import NegativeSampler
from project.negative_samples.confidence_scorer import ConfidenceScorer
from project.negative_samples.probable_positive_filter import ProbablePositiveFilter


@pytest.fixture
def sample_configs():
    config = {
        "validation": {"n_folds": 5, "group_key": "normalized_query_hash"},
        "negative_sampling": {
            "positive_negative_ratio": 3,
            "confidence": {
                "discard_below": 0.80,
                "weight_tiers": [
                    {"min": 0.95, "max": 1.00, "weight": 1.0},
                    {"min": 0.90, "max": 0.95, "weight": 0.8},
                    {"min": 0.80, "max": 0.90, "weight": 0.5}
                ]
            }
        }
    }
    neg_config = {
        "positive_negative_ratio": 3,
        "strategy_ratios": {
            "random": 0.10,
            "cross_category": 0.10,
            "same_category": 0.20,
            "same_brand": 0.10,
            "lexical_hard": 0.05,
            "attribute_conflict": 0.05,
            "cross_query": 0.15,
            "embedding_hard": 0.0
        },
        "confidence": {
            "discard_below": 0.80,
            "weight_tiers": [
                {"min": 0.95, "max": 1.00, "weight": 1.0},
                {"min": 0.90, "max": 0.95, "weight": 0.8},
                {"min": 0.80, "max": 0.90, "weight": 0.5}
            ]
        },
        "probable_positive": {
            "title_contains_query_threshold": 0.85,
            "lexical_similarity_threshold": 0.90
        }
    }
    return config, neg_config


@pytest.fixture
def mock_dataset():
    items = pd.DataFrame({
        "item_id": ["ITEM_1", "ITEM_2", "ITEM_3", "ITEM_4"],
        "title": ["Erkek Oversize Siyah Tişört", "Kadın Kırmızı Çanta", "Mavi Kot Pantolon", "Yeşil Deri Ceket"],
        "category": ["Giyim > Tişört", "Aksesuar > Çanta", "Giyim > Pantolon", "Giyim > Ceket"],
        "brand": ["Defacto", "Mango", "Levi's", "Zara"],
        "gender": ["erkek", "kadın", "unisex", "erkek"],
        "age_group": ["yetişkin", "yetişkin", "genç", "yetişkin"],
        "attributes": ["renk: siyah", "renk: kırmızı", "renk: mavi", "renk: yeşil, materyal: deri"]
    })

    train = pd.DataFrame({
        "query": ["siyah tişört", "kırmızı çanta"],
        "item_id": ["ITEM_1", "ITEM_2"],
        "title": ["Erkek Oversize Siyah Tişört", "Kadın Kırmızı Çanta"],
        "category": ["Giyim > Tişört", "Aksesuar > Çanta"],
        "brand": ["Defacto", "Mango"],
        "gender": ["erkek", "kadın"],
        "age_group": ["yetişkin", "yetişkin"],
        "attributes": ["renk: siyah", "renk: kırmızı"],
        "label": [1, 1],
        "fold": [0, 0]
    })
    return items, train


def test_confidence_scorer(sample_configs):
    _, neg_config = sample_configs
    scorer = ConfidenceScorer(neg_config)

    # Test individual scores
    assert scorer.score_negative("random", "siyah tişört", "kırmızı çanta") == 1.0
    assert scorer.score_negative("cross_category", "siyah tişört", "kırmızı çanta") == 0.99
    assert scorer.score_negative("attribute_conflict", "siyah tişört", "kırmızı çanta", has_attribute_conflict=True) == 0.98

    # Test sample weights
    assert scorer.get_sample_weight(1.0) == 1.0
    assert scorer.get_sample_weight(0.93) == 0.8
    assert scorer.get_sample_weight(0.85) == 0.5
    assert scorer.get_sample_weight(0.70) == 0.0  # discarded


def test_probable_positive_filter(sample_configs):
    _, neg_config = sample_configs
    pp_filter = ProbablePositiveFilter(neg_config)

    # 1. Exact query match in title -> probable positive
    assert pp_filter.is_probable_positive("siyah ceket", "Siyah Ceket Pantolon") is True

    # 2. Token overlap >= 0.85 -> probable positive
    assert pp_filter.is_probable_positive("siyah oversize ceket", "oversize siyah ceket") is True

    # 3. Low overlap -> not probable positive
    assert pp_filter.is_probable_positive("siyah ceket", "mavi kot pantolon") is False


def test_embedding_hard_requires_neighbors(sample_configs, mock_dataset):
    """embedding_hard > 0 without a neighbors file must fail loudly, not silently produce 0."""
    config, neg_config = sample_configs
    items, train = mock_dataset
    neg_config = {**neg_config, "strategy_ratios": {**neg_config["strategy_ratios"], "embedding_hard": 0.25}}

    with pytest.raises(ValueError, match="embedding_hard"):
        NegativeSampler(config, neg_config, items, train)


def test_negative_sampler(sample_configs, mock_dataset):
    config, neg_config = sample_configs
    items, train = mock_dataset

    sampler = NegativeSampler(config, neg_config, items, train)

    # Test single query sampling
    negs = sampler.sample_negatives_for_query("siyah tişört", "Giyim > Tişört", "Defacto")
    assert len(negs) > 0
    for neg in negs:
        assert neg["label"] == 0
        assert neg["sample_weight"] in [0.5, 0.8, 1.0]
        assert neg["item_id"] != "ITEM_1"  # exclude positive

    # Test full dataset build
    dataset = sampler.build_dataset(train)
    assert len(dataset) > 0
    # Positive label check
    assert 1 in dataset["label"].values
    assert 0 in dataset["label"].values
