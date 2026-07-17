"""
Attribute Conflict Negative Strategy — Samples same-category items with
conflicting attributes (e.g. gender mismatch or color mismatch).
"""
import random
import pandas as pd
from project.utils.text_cleaner import clean_index_text, clean_gender
from project.utils.attribute_parser import parse_attributes, get_color

class AttributeConflictNegativeStrategy:
    """Samples negatives with explicit color or gender conflicts."""

    def __init__(self, items_df: pd.DataFrame):
        self.items_df = items_df
        # Group items by category to find matching product types
        self.category_groups = {}
        for idx, row in items_df.iterrows():
            cat = row.get("category")
            cat_clean = cat.strip() if isinstance(cat, str) else "unknown"
            if cat_clean not in self.category_groups:
                self.category_groups[cat_clean] = []
            self.category_groups[cat_clean].append(idx)

        # Pre-defined conflict sets
        self.genders = ["erkek", "kadın"]
        self.colors = [
            "siyah", "beyaz", "kirmizi", "mavi", "yesil", "sari", "pembe",
            "mor", "turuncu", "gri", "kahverengi", "lacivert", "bej"
        ]

    def _get_query_attributes(self, query_clean: str) -> dict[str, list[str]]:
        """Extract explicit color or gender mentions from the query."""
        tokens = query_clean.split()
        q_genders = []
        if "erkek" in tokens:
            q_genders.append("erkek")
        if "kadin" in tokens or "kız" in query_clean or "kiz" in tokens:
            q_genders.append("kadın")

        q_colors = [c for c in self.colors if c in tokens]

        return {"gender": q_genders, "color": q_colors}

    def sample(
        self,
        query: str,
        exclude_item_ids: set[str],
        query_category: str | None = None,
        n_samples: int = 1,
    ) -> list[dict]:
        """Sample same-category items that conflict with query attributes."""
        samples = []
        if not query_category:
            return samples

        query_clean = clean_index_text(query)
        q_attrs = self._get_query_attributes(query_clean)

        # Only sample if there is at least one query attribute to conflict with
        if not q_attrs["gender"] and not q_attrs["color"]:
            return samples

        indices = self.category_groups.get(query_category.strip(), [])
        if not indices:
            return samples

        # Shuffle category indices to get varied samples
        indices_shuffled = list(indices)
        random.shuffle(indices_shuffled)

        max_attempts = 100
        attempts = 0

        for idx in indices_shuffled:
            attempts += 1
            if attempts > max_attempts or len(samples) >= n_samples:
                break

            row = self.items_df.iloc[idx]
            item_id = row.get("item_id")

            if item_id in exclude_item_ids:
                continue

            # Check for gender conflict
            gender_conflict = False
            if q_attrs["gender"]:
                item_gender = clean_gender(row.get("gender"))
                if item_gender != "unknown" and item_gender != "unisex":
                    if item_gender not in q_attrs["gender"]:
                        gender_conflict = True

            # Check for color conflict
            color_conflict = False
            if q_attrs["color"]:
                attrs_str = row.get("attributes")
                if isinstance(attrs_str, str):
                    parsed = parse_attributes(attrs_str)
                    prod_color = get_color(parsed)
                    if prod_color:
                        prod_color_clean = clean_index_text(prod_color)
                        # Check if none of the query colors match product color
                        if not any(qc in prod_color_clean for qc in q_attrs["color"]):
                            color_conflict = True

            if gender_conflict or color_conflict:
                samples.append({
                    "item_id": item_id,
                    "title": row.get("title"),
                    "category": row.get("category"),
                    "brand": row.get("brand"),
                    "gender": row.get("gender"),
                    "attributes": row.get("attributes"),
                    "negative_type": "attribute_conflict",
                    "negative_confidence": 0.98,
                })

        return samples
