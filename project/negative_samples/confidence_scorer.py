"""
Confidence Scorer — Computes confidence scores for negative samples.
Confidence is used to weigh samples during training or discard low-confidence negatives.
"""
from project.utils.logging_utils import setup_logger

logger = setup_logger("confidence_scorer")


class ConfidenceScorer:
    """Computes a confidence score (0.0 to 1.0) for a negative pair."""

    def __init__(self, config: dict):
        # Accept both the negative_sampling.yaml root and a wrapping config dict.
        conf_cfg = config.get("confidence") or config.get("negative_sampling", {}).get("confidence", {})
        if not conf_cfg:
            logger.warning("No 'confidence' section found in config — using built-in defaults.")
        self.discard_below = conf_cfg.get("discard_below", 0.80)
        self.weight_tiers = conf_cfg.get("weight_tiers", [
            {"min": 0.95, "max": 1.00, "weight": 1.0},
            {"min": 0.90, "max": 0.95, "weight": 0.8},
            {"min": 0.80, "max": 0.90, "weight": 0.5},
        ])
        # Optional type-based weighting: emphasizes hard negative types instead
        # of the tier scheme (which conflates label certainty with difficulty
        # and ends up training hardest examples with the lowest weight).
        self.type_weights = config.get("type_weights") or {}

    def score_negative(
        self,
        negative_type: str,
        query: str,
        title: str,
        lexical_similarity: float | None = None,
        semantic_similarity: float | None = None,
        has_attribute_conflict: bool = False,
    ) -> float:
        """
        Assign a confidence score based on negative type and similarities.
        Higher score = more confident that the pair is truly negative.
        """
        if negative_type == "random":
            return 1.0
        if negative_type == "cross_category":
            return 0.99

        if negative_type == "same_category":
            if lexical_similarity is not None:
                return max(0.80, min(0.96, 1.0 - lexical_similarity * 0.5))
            return 0.94

        if negative_type == "same_brand":
            if lexical_similarity is not None:
                return max(0.80, min(0.95, 1.0 - lexical_similarity * 0.5))
            return 0.93

        if negative_type == "cross_query":
            if lexical_similarity is not None:
                return max(0.80, min(0.97, 1.0 - lexical_similarity * 0.5))
            return 0.95

        if negative_type == "attribute_conflict":
            # A hard conflict (query says "kırmızı", product is "mavi") is a
            # near-certain negative regardless of lexical overlap.
            return 0.98 if has_attribute_conflict else 0.85

        if negative_type == "lexical_hard":
            if lexical_similarity is not None:
                return max(0.80, min(0.90, 1.05 - lexical_similarity * 0.5))
            return 0.82

        if negative_type in ("embedding_hard", "mined_hard"):
            if semantic_similarity is not None:
                return max(0.80, min(0.96, 1.0 - (semantic_similarity - 0.5)))
            return 0.82

        return 0.85

    def get_sample_weight(self, confidence: float, negative_type: str | None = None) -> float:
        """
        Get training sample weight. Returns 0.0 if the sample should be discarded.

        When `type_weights` is configured and a negative_type is given, the
        weight is type_weight × mild noise discount (0.9 below 0.90 confidence).
        Otherwise falls back to the confidence-tier scheme.
        """
        if confidence < self.discard_below:
            return 0.0

        if negative_type is not None and self.type_weights:
            base = self.type_weights.get(negative_type, 1.0)
            noise_discount = 1.0 if confidence >= 0.90 else 0.9
            return round(base * noise_discount, 3)

        for tier in self.weight_tiers:
            if tier["min"] <= confidence <= tier["max"]:
                return tier["weight"]

        return 0.0
