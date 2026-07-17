"""
Probable Positive Filter — Filters out negative candidates that are highly
likely to be positive (to prevent label noise / pollution).
"""
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
        self.require_category_match = pp_cfg.get("require_category_match", True)

    def is_probable_positive(
        self,
        query: str,
        product_title: str,
        product_category: str | None = None,
        product_brand: str | None = None,
        semantic_sim: float | None = None,
        category_match: bool | None = None,
        query_tokens: list[str] | None = None,
        title_tokens: list[str] | None = None,
    ) -> bool:
        """
        Check if query and product have extremely high similarity or exact matches,
        meaning this product is highly likely relevant and shouldn't be labeled as 0.

        `category_match` (optional): whether the candidate shares the anchor
        positive's category. When require_category_match is on and the
        categories are known to differ, the token-overlap heuristics are
        skipped (a title containing the query words but from another category
        — e.g. "elbise askısı" for query "elbise" — is a valid negative).
        The full-query-substring rule is applied regardless.

        `query_tokens` / `title_tokens` allow callers to pass precomputed
        cleaned tokens to avoid re-cleaning in hot loops.
        """
        if not query or not product_title:
            return False

        if query_tokens is None:
            query_tokens = clean_index_text(query).split()
        if title_tokens is None:
            title_tokens = clean_index_text(product_title).split()

        if not query_tokens or not title_tokens:
            return False

        q_clean = " ".join(query_tokens)
        t_clean = " ".join(title_tokens)

        # 1. Entire query appearing verbatim in the title is always suspicious.
        if q_clean in t_clean:
            return True

        # 2. Overlap-based rules — only when the category is not known to differ.
        overlap_rules_active = not (self.require_category_match and category_match is False)

        if overlap_rules_active:
            t_set = set(title_tokens)
            matches = sum(1 for tok in query_tokens if tok in t_set)
            overlap = matches / len(query_tokens)
            if overlap >= self.title_overlap_thresh:
                return True

            q_set = set(query_tokens)
            union = q_set | t_set
            jaccard = len(q_set & t_set) / len(union) if union else 0.0
            if jaccard >= self.lexical_thresh:
                return True

        # 3. Semantic similarity (used by embedding/mining flows)
        if semantic_sim is not None:
            sem_thresh = self.config.get("probable_positive", {}).get(
                "semantic_similarity_threshold", 0.92
            )
            if semantic_sim >= sem_thresh:
                return True

        return False
