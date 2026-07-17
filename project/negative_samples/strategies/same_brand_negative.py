"""
Same Brand Negative Strategy — Selects items from the exact same brand.
"""
import random
import pandas as pd
from project.utils.text_cleaner import clean_brand

class SameBrandNegativeStrategy:
    """Samples negatives from the exact same brand as the query/product."""

    def __init__(self, items_df: pd.DataFrame):
        self.items_df = items_df
        # Group items by their clean brand name
        self.brand_groups = {}
        for idx, row in items_df.iterrows():
            brand = row.get("brand")
            brand_clean = clean_brand(brand)
            if not brand_clean:
                continue
            if brand_clean not in self.brand_groups:
                self.brand_groups[brand_clean] = []
            self.brand_groups[brand_clean].append(idx)

    def sample(
        self,
        query: str,
        exclude_item_ids: set[str],
        query_brand: str | None = None,
        n_samples: int = 1,
    ) -> list[dict]:
        """Sample items from the same brand group, excluding positive items."""
        samples = []
        brand_clean = clean_brand(query_brand)
        if not brand_clean:
            return samples

        indices = self.brand_groups.get(brand_clean, [])
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
                "negative_type": "same_brand",
                "negative_confidence": 0.93,
            })

        return samples
