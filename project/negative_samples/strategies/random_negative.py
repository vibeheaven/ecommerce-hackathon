"""
Random Negative Strategy — Selects completely random items from the catalog.
These items are from different top-level categories.
"""
import random
import pandas as pd

class RandomNegativeStrategy:
    """Random negative sampler."""

    def __init__(self, items_df: pd.DataFrame):
        self.items_df = items_df
        # Index items by item_id for fast lookup
        self.item_ids = items_df["item_id"].values
        self.categories = items_df["category"].values

    def sample(
        self,
        query: str,
        exclude_item_ids: set[str],
        query_category: str | None = None,
        n_samples: int = 1,
    ) -> list[dict]:
        """
        Sample random negative items.
        Ensures they are not in exclude_item_ids and from a different top-level category if query_category is set.
        """
        samples = []
        max_attempts = 100
        attempts = 0

        # Get parent category of query
        query_parent = query_category.split(">")[0].strip() if query_category else None

        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            idx = random.randint(0, len(self.item_ids) - 1)
            item_id = self.item_ids[idx]

            if item_id in exclude_item_ids:
                continue

            # Category filter
            if query_parent:
                item_cat = self.categories[idx]
                item_parent = item_cat.split(">")[0].strip() if isinstance(item_cat, str) else None
                if item_parent == query_parent:
                    continue  # Skip if same top category to preserve "randomness"

            # Valid random item found
            row = self.items_df.iloc[idx]
            samples.append({
                "item_id": item_id,
                "title": row.get("title"),
                "category": row.get("category"),
                "brand": row.get("brand"),
                "gender": row.get("gender"),
                "attributes": row.get("attributes"),
                "negative_type": "random",
                "negative_confidence": 1.0,
            })

        return samples
