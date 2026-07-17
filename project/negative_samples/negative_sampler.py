"""
Negative Sampler — Orchestrates all negative sampling strategies.
Integrates Probable Positive Filter and Confidence Scorer.
"""
import pandas as pd
from tqdm import tqdm
from typing import Any

from project.utils.logging_utils import setup_logger
from project.negative_samples.probable_positive_filter import ProbablePositiveFilter
from project.negative_samples.confidence_scorer import ConfidenceScorer

# Import strategies
from project.negative_samples.strategies.random_negative import RandomNegativeStrategy
from project.negative_samples.strategies.cross_category_negative import CrossCategoryNegativeStrategy
from project.negative_samples.strategies.same_category_negative import SameCategoryNegativeStrategy
from project.negative_samples.strategies.same_brand_negative import SameBrandNegativeStrategy
from project.negative_samples.strategies.lexical_hard_negative import LexicalHardNegativeStrategy
from project.negative_samples.strategies.attribute_conflict_negative import AttributeConflictNegativeStrategy
from project.negative_samples.strategies.cross_query_negative import CrossQueryNegativeStrategy

logger = setup_logger("negative_sampler")


class NegativeSampler:
    """Orchestrates negative sampling for train and validation splits."""

    def __init__(self, config: dict, neg_config: dict, items_df: pd.DataFrame, train_df: pd.DataFrame):
        self.config = config
        self.neg_config = neg_config
        self.items_df = items_df

        # Initialize filter and scorer
        self.filter = ProbablePositiveFilter(neg_config)
        self.scorer = ConfidenceScorer(neg_config)

        # Initialize strategies
        logger.info("Initializing negative sampling strategies...")
        self.strat_random = RandomNegativeStrategy(items_df)
        self.strat_cross_cat = CrossCategoryNegativeStrategy(items_df)
        self.strat_same_cat = SameCategoryNegativeStrategy(items_df)
        self.strat_same_brand = SameBrandNegativeStrategy(items_df)
        self.strat_lexical_hard = LexicalHardNegativeStrategy(items_df)
        self.strat_attr_conflict = AttributeConflictNegativeStrategy(items_df)
        self.strat_cross_query = CrossQueryNegativeStrategy(train_df, items_df)

        # Ratio and counts settings
        self.ratio = neg_config.get("positive_negative_ratio", 3)
        self.strategy_ratios = neg_config.get("strategy_ratios", {})

        # Known positives dictionary mapping query/term_id -> set of positive item_ids
        logger.info("Building known positives lookup dictionary...")
        self.known_positives = {}
        for _, row in train_df.iterrows():
            q_key = row.get("query")
            if q_key:
                if q_key not in self.known_positives:
                    self.known_positives[q_key] = set()
                self.known_positives[q_key].add(row.get("item_id"))

    def sample_negatives_for_query(
        self,
        query: str,
        query_category: str | None = None,
        query_brand: str | None = None,
    ) -> list[dict]:
        """
        Sample a set of negative items for a single query.
        Returns a list of dicts representing the negatives.
        """
        negatives = []
        exclude_item_ids = set(self.known_positives.get(query, []))

        # Determine target number of samples for each active strategy
        total_target = self.ratio
        sampled_count = 0

        # Strategies and their specific weights/ratios
        # Sum of active V1 ratios in negative_sampling.yaml:
        # random: 0.10, cross_category: 0.10, same_category: 0.20, same_brand: 0.10,
        # lexical_hard: 0.05, attribute_conflict: 0.05, cross_query: 0.15, embedding_hard: 0.25 (V2+)
        # For V1, we scale the active strategy ratios to sum to 1.0
        active_strats = [
            ("random", self.strat_random.sample),
            ("cross_category", self.strat_cross_cat.sample),
            ("same_category", self.strat_same_cat.sample),
            ("same_brand", self.strat_same_brand.sample),
            ("lexical_hard", self.strat_lexical_hard.sample),
            ("attribute_conflict", self.strat_attr_conflict.sample),
            ("cross_query", self.strat_cross_query.sample),
        ]

        active_ratios = {name: self.strategy_ratios.get(name, 0.0) for name, _ in active_strats}
        ratio_sum = sum(active_ratios.values())
        if ratio_sum == 0:
            ratio_sum = 1.0

        # Calculate exact integer counts for each strategy
        strategy_counts = {}
        for name in active_ratios.keys():
            normalized_ratio = active_ratios[name] / ratio_sum
            strategy_counts[name] = max(1, round(total_target * normalized_ratio))

        # Perform sampling
        for name, sample_func in active_strats:
            count = strategy_counts.get(name, 1)
            if count <= 0:
                continue

            # Standard argument signature checking
            kwargs = {"query": query, "exclude_item_ids": exclude_item_ids, "n_samples": count}
            if name in ("random", "cross_category", "same_category", "attribute_conflict"):
                kwargs["query_category"] = query_category
            if name == "same_brand":
                kwargs["query_brand"] = query_brand

            try:
                strat_samples = sample_func(**kwargs)
                for item in strat_samples:
                    # Apply probable positive filter
                    is_pp = self.filter.is_probable_positive(
                        query=query,
                        product_title=item["title"],
                        product_category=item["category"],
                        product_brand=item["brand"],
                    )
                    if is_pp:
                        continue  # Skip probable positive

                    # Scorer & sample weight
                    confidence = self.scorer.score_negative(
                        negative_type=item["negative_type"],
                        query=query,
                        title=item["title"],
                    )
                    weight = self.scorer.get_sample_weight(confidence)
                    if weight == 0.0:
                        continue  # Discard below threshold

                    item["label"] = 0
                    item["sample_weight"] = weight
                    item["query"] = query
                    negatives.append(item)
            except Exception as e:
                logger.error(f"Error sampling {name} for query '{query}': {e}")

        # Limit to self.ratio if we got slightly more due to rounding
        return negatives[:total_target]

    def build_dataset(self, split_df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct the complete dataset by combining positive pairs
        with dynamically sampled negative pairs.
        """
        logger.info(f"Building dataset on {len(split_df):,} positive pairs...")
        rows = []

        for idx, row in tqdm(split_df.iterrows(), total=len(split_df)):
            # 1. Add positive pair
            rows.append({
                "term_id": row.get("term_id"),
                "item_id": row.get("item_id"),
                "query": row.get("query"),
                "title": row.get("title"),
                "category": row.get("category"),
                "brand": row.get("brand"),
                "gender": row.get("gender"),
                "age_group": row.get("age_group"),
                "attributes": row.get("attributes"),
                "label": 1,
                "sample_weight": 1.0,
                "negative_type": "positive",
                "fold": row.get("fold", -1),
            })

            # 2. Sample negatives
            negatives = self.sample_negatives_for_query(
                query=row.get("query"),
                query_category=row.get("category"),
                query_brand=row.get("brand"),
            )

            for neg in negatives:
                rows.append({
                    "term_id": row.get("term_id"),  # Keep same query term_id
                    "item_id": neg["item_id"],
                    "query": neg["query"],
                    "title": neg["title"],
                    "category": neg["category"],
                    "brand": neg["brand"],
                    "gender": neg["gender"],
                    "age_group": neg.get("age_group"),
                    "attributes": neg["attributes"],
                    "label": 0,
                    "sample_weight": neg["sample_weight"],
                    "negative_type": neg["negative_type"],
                    "fold": row.get("fold", -1),
                })

        return pd.DataFrame(rows)


if __name__ == "__main__":
    import yaml
    import argparse
    from project.data.data_loader import load_items

    parser = argparse.ArgumentParser(description="Run negative sampler smoke test")
    parser.add_argument("--config", default="project/configs/config.yaml")
    parser.add_argument("--neg-config", default="project/configs/negative_sampling.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.neg-config) as f:
        neg_config = yaml.safe_load(f)

    # Simple smoke test on mock splits
    items_path = Path(config["data"]["raw_dir"]) / config["data"]["files"]["items"]
    items_df = load_items(items_path).head(1000)

    mock_train = pd.DataFrame({
        "query": ["tişört", "çanta"],
        "item_id": ["ITEM_1", "ITEM_2"],
        "title": ["Erkek Tişört", "Kadın Çanta"],
        "category": ["Giyim", "Aksesuar"],
        "brand": ["Defacto", "Mango"],
        "gender": ["erkek", "kadın"],
        "age_group": ["yetişkin", "yetişkin"],
        "attributes": ["", ""],
        "label": [1, 1],
        "fold": [0, 0]
    })

    sampler = NegativeSampler(config, neg_config, items_df, mock_train)
    dataset = sampler.build_dataset(mock_train)
    print(f"\nGenerated dataset rows: {len(dataset)}")
    print(dataset[["query", "title", "label", "negative_type", "sample_weight"]].head(10))
