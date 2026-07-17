"""
Cross Query Negative Strategy — Samples positive items from other queries.
"""
import random
import pandas as pd

class CrossQueryNegativeStrategy:
    """Samples negatives by selecting positive items of other queries."""

    def __init__(self, train_df: pd.DataFrame, items_df: pd.DataFrame):
        self.items_df = items_df
        # Index all items by item_id for fast lookup
        self.items_by_id = {}
        for idx, row in items_df.iterrows():
            self.items_by_id[row.get("item_id")] = idx

        # Get all positive item_ids in the training split
        # In the training split, all labels are 1
        self.all_positive_items = list(train_df["item_id"].unique())

    def sample(
        self,
        query: str,
        exclude_item_ids: set[str],
        n_samples: int = 1,
    ) -> list[dict]:
        """Sample positive items belonging to other queries."""
        samples = []
        if not self.all_positive_items:
            return samples

        max_attempts = 100
        attempts = 0

        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            item_id = random.choice(self.all_positive_items)

            if item_id in exclude_item_ids:
                continue

            idx = self.items_by_id.get(item_id)
            if idx is None:
                continue

            row = self.items_df.iloc[idx]
            samples.append({
                "item_id": item_id,
                "title": row.get("title"),
                "category": row.get("category"),
                "brand": row.get("brand"),
                "gender": row.get("gender"),
                "attributes": row.get("attributes"),
                "negative_type": "cross_query",
                "negative_confidence": 0.95,
            })

        return samples
