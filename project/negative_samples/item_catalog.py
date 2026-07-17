"""
Item Catalog — Precomputed, vectorized view over items.csv used by the
negative sampler. Built once; all strategies share it.

Replaces the per-strategy `iterrows()` scans of the old implementation
(7 strategies × 1M items) with a single pass and O(1) lookups.
"""
import numpy as np
import pandas as pd
import time

from project.utils.logging_utils import setup_logger
from project.utils.text_cleaner import clean_index_text, clean_brand, clean_gender
from project.utils.attribute_parser import parse_attributes, get_color

logger = setup_logger("item_catalog")

# Cap on inverted-index posting list length. Very common tokens ("set", "cm")
# carry little ranking signal; a seeded subsample keeps per-query candidate
# generation bounded without meaningfully changing the candidate pool.
_MAX_POSTING = 30_000

_MIN_TOKEN_LEN = 3


class ItemCatalog:
    """Column-oriented item store with category/brand/parent groups and a token index."""

    def __init__(self, items_df: pd.DataFrame, seed: int = 42):
        t0 = time.time()
        n = len(items_df)
        logger.info(f"Building item catalog for {n:,} items...")

        self.item_ids: list = items_df["item_id"].tolist()
        self.titles: list = items_df["title"].fillna("").tolist()
        self.categories: list = items_df["category"].fillna("").tolist()
        self.brands: list = items_df["brand"].fillna("").tolist()
        self.genders_raw: list = items_df["gender"].fillna("").tolist() if "gender" in items_df.columns else [""] * n
        self.age_groups: list = items_df["age_group"].fillna("").tolist() if "age_group" in items_df.columns else [""] * n
        self.attributes: list = items_df["attributes"].fillna("").tolist() if "attributes" in items_df.columns else [""] * n

        self.pos_by_item_id = {iid: i for i, iid in enumerate(self.item_ids)}

        # Cleaned/derived columns
        self.clean_titles: list[str] = [clean_index_text(t) for t in self.titles]
        self.clean_genders: list[str] = [clean_gender(g) for g in self.genders_raw]

        cat_clean = [c.strip() for c in self.categories]
        parents = [c.split("/")[0].split(">")[0].strip() if c else "unknown" for c in cat_clean]
        self.cat_clean = cat_clean
        self.parents = parents

        # Group positions
        self.cat_groups: dict[str, np.ndarray] = self._group_positions(cat_clean)
        self.parent_groups: dict[str, np.ndarray] = self._group_positions(parents)
        self.brand_groups: dict[str, np.ndarray] = self._group_positions(
            [clean_brand(b) for b in self.brands]
        )
        self.all_parents = [p for p in self.parent_groups.keys() if p and p != "unknown"]

        # Inverted index over title tokens (full catalog, capped posting lists)
        rng = np.random.default_rng(seed)
        raw_index: dict[str, list[int]] = {}
        for pos, title in enumerate(self.clean_titles):
            for tok in set(title.split()):
                if len(tok) >= _MIN_TOKEN_LEN:
                    raw_index.setdefault(tok, []).append(pos)

        self.inverted_index: dict[str, np.ndarray] = {}
        capped = 0
        for tok, positions in raw_index.items():
            arr = np.asarray(positions, dtype=np.int32)
            if len(arr) > _MAX_POSTING:
                arr = rng.choice(arr, size=_MAX_POSTING, replace=False)
                capped += 1
            self.inverted_index[tok] = arr

        # Lazy caches
        self._color_cache: dict[int, str | None] = {}

        logger.info(
            f"  Catalog ready in {time.time() - t0:.1f}s — "
            f"{len(self.cat_groups):,} categories, {len(self.brand_groups):,} brands, "
            f"{len(self.inverted_index):,} indexed tokens ({capped} capped at {_MAX_POSTING:,})"
        )

    @staticmethod
    def _group_positions(keys: list[str]) -> dict[str, np.ndarray]:
        groups: dict[str, list[int]] = {}
        for pos, key in enumerate(keys):
            if key:
                groups.setdefault(key, []).append(pos)
        return {k: np.asarray(v, dtype=np.int32) for k, v in groups.items()}

    def color_of(self, pos: int) -> str | None:
        """Parsed (cleaned, ASCII) color attribute of an item, cached."""
        if pos not in self._color_cache:
            color = get_color(parse_attributes(self.attributes[pos]))
            self._color_cache[pos] = clean_index_text(color) if color else None
        return self._color_cache[pos]

    def record(self, pos: int, negative_type: str, confidence: float) -> dict:
        """Materialize an item row as a negative-candidate dict."""
        return {
            "item_id": self.item_ids[pos],
            "title": self.titles[pos],
            "category": self.categories[pos],
            "brand": self.brands[pos],
            "gender": self.genders_raw[pos],
            "age_group": self.age_groups[pos],
            "attributes": self.attributes[pos],
            "negative_type": negative_type,
            "negative_confidence": confidence,
        }
