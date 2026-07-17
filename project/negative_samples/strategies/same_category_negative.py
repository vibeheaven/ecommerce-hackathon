"""
Same Category Negative Strategy — Selects items from the exact same category.
These are much harder negatives because the product type matches.
"""
import random
import pandas as pd

class SameCategoryNegativeStrategy:
    """Samples negatives from the exact same category as the query."""

    def __init__(self, items_df: pd.DataFrame):
        self.items_df = items_df
        # Group items by their exact clean category path
        self.category_groups = {}
        for idx, row in items_df.iterrows():
            cat = row.get("category")
            cat_clean = cat.strip() if isinstance(cat, str) else "unknown"
            if cat_clean not in self.category_groups:
                self.category_groups[cat_clean] = []
            self.category_groups[cat_clean].append(idx)

    def sample(
        self,
        query: str,
        exclude_item_ids: set[str],
        query_category: str | None = None,
        n_samples: int = 1,
    ) -> list[dict]:
        """Sample items from the same category path, excluding positive items."""
        samples = []
        if not query_category:
            return samples

        query_cat_clean = query_category.strip()
        indices = self.category_groups.get(query_cat_clean, [])
        if not indices:
            return samples

        max_attempts = 100
        attempts = 0

        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
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
                "negative_type": "same_category",
                "negative_confidence": 0.94,
            })

        return samples
