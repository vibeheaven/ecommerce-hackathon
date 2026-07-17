"""
Cross Category Negative Strategy — Selects items from a different category.
"""
import random
import pandas as pd

class CrossCategoryNegativeStrategy:
    """Samples negatives from a completely different parent category."""

    def __init__(self, items_df: pd.DataFrame):
        self.items_df = items_df
        # Group items by their top-level parent category
        self.parent_categories = {}
        for idx, row in items_df.iterrows():
            cat = row.get("category")
            parent = cat.split(">")[0].strip() if isinstance(cat, str) else "unknown"
            if parent not in self.parent_categories:
                self.parent_categories[parent] = []
            self.parent_categories[parent].append(idx)

        self.all_parents = list(self.parent_categories.keys())

    def sample(
        self,
        query: str,
        exclude_item_ids: set[str],
        query_category: str | None = None,
        n_samples: int = 1,
    ) -> list[dict]:
        """Sample items from a parent category different from query_category."""
        samples = []
        query_parent = query_category.split(">")[0].strip() if query_category else None

        # Filter candidate parent categories
        candidate_parents = [p for p in self.all_parents if p != query_parent]
        if not candidate_parents:
            candidate_parents = self.all_parents

        max_attempts = 100
        attempts = 0

        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            # Select random parent category
            parent = random.choice(candidate_parents)
            indices = self.parent_categories[parent]
            if not indices:
                continue

            idx = random.choice(indices)
            row = self.items_df.iloc[idx]
            item_id = row.get("item_id")

            if item_id in exclude_item_ids:
                continue

            samples.append({
                "item_id": item_id,
                "title": row.get("title"),
                "category": row.get("category"),
                "brand": row.get("brand"),
                "gender": row.get("gender"),
                "attributes": row.get("attributes"),
                "negative_type": "cross_category",
                "negative_confidence": 0.99,
            })

        return samples
