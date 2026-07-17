"""
Probable Positive Filter — Filters out negative candidates that are highly
likely to be positive (to prevent label noise / pollution).
"""
from typing import Any

from project.utils.text_cleaner import clean_index_text
from project.utils.logging_utils import setup_logger

logger = setup_logger("probable_positive_filter")


class ProbablePositiveFilter:
    """Checks if a query-product pair is a probable positive."""

    def __init__(self, config: dict):
        self.config = config
        pp_cfg = config.get("probable_positive", {})
        self.title_overlap_thresh = pp_cfg.get("title_contains_query_threshold", 0.85)
        self.lexical_thresh = pp_cfg.get("lexical_similarity_threshold", 0.90)

    def is_probable_positive(
        self,
        query: str,
        product_title: str,
        product_category: str | None = None,
        product_brand: str | None = None,
        semantic_sim: float | None = None,
    ) -> bool:
        """
        Check if query and product have extremely high similarity or exact matches,
        meaning this product is highly likely relevant and shouldn't be labeled as 0.
        """
        if not query or not product_title:
            return False

        q_clean = clean_index_text(query)
        t_clean = clean_index_text(product_title)

        q_tokens = q_clean.split()
        t_tokens = t_clean.split()

        if not q_tokens or not t_tokens:
            return False

        # 1. Exact query match in title (as substring or token overlap)
        if q_clean in t_clean:
            # If the entire query is a substring of the title, it's highly likely related
            return True

        # Token overlap ratio (query tokens in title)
        t_set = set(t_tokens)
        matches = sum(1 for tok in q_tokens if tok in t_set)
        overlap = matches / len(q_tokens) if q_tokens else 0.0

        if overlap >= self.title_overlap_thresh:
            return True

        # Jaccard lexical similarity
        intersection = set(q_tokens) & t_set
        union = set(q_tokens) | t_set
        jaccard = len(intersection) / len(union) if union else 0.0
        if jaccard >= self.lexical_thresh:
            return True

        # 2. Semantic similarity (V2+)
        if semantic_sim is not None:
            pp_cfg = self.config.get("probable_positive", {})
            sem_thresh = pp_cfg.get("semantic_similarity_threshold", 0.92)
            if semantic_sim >= sem_thresh:
                return True

        return False
