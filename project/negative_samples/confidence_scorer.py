"""
Confidence Scorer — Computes confidence scores for negative samples.
Confidence is used to weigh samples during training or discard low-confidence negatives.
"""
from typing import Any

from project.utils.text_cleaner import clean_index_text

class ConfidenceScorer:
    """Computes a confidence score (0.0 to 1.0) for a negative pair."""

    def __init__(self, config: dict):
        self.config = config
        self.discard_below = config.get("negative_sampling", {}).get("confidence", {}).get("discard_below", 0.80)

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
        # 1. Easy negatives are extremely high confidence
        if negative_type == "random":
            return 1.0
        if negative_type == "cross_category":
            return 0.99

        # 2. Same category / Same brand are high confidence but have minor risk
        if negative_type == "same_category":
            # If lexical similarity is low, we are very confident
            if lexical_similarity is not None:
                return max(0.80, min(0.96, 1.0 - lexical_similarity))
            return 0.94

        if negative_type == "same_brand":
            if lexical_similarity is not None:
                return max(0.80, min(0.95, 1.0 - lexical_similarity))
            return 0.93

        # 3. Cross query negatives
        if negative_type == "cross_query":
            # Very strong signal since it's positive for another query, but we still verify overlap
            if lexical_similarity is not None:
                return max(0.80, min(0.97, 1.0 - lexical_similarity))
            return 0.95

        # 4. Attribute conflict is extremely high confidence if query mentioned attribute
        if negative_type == "attribute_conflict":
            return 0.98 if has_attribute_conflict else 0.85

        # 5. Lexical hard is tricky because token overlap is high
        if negative_type == "lexical_hard":
            if lexical_similarity is not None:
                # High overlap reduces negative confidence
                return max(0.70, min(0.90, 1.1 - lexical_similarity))
            return 0.82

        # 6. Embedding hard (semantic) negative (V2+)
        if negative_type == "embedding_hard":
            if semantic_similarity is not None:
                # Lower semantic similarity = higher negative confidence
                return max(0.70, min(0.96, 1.0 - (semantic_similarity - 0.5) * 2))
            return 0.80

        return 0.85

    def get_sample_weight(self, confidence: float) -> float:
        """
        Get training sample weight based on confidence tiers.
        Returns 0.0 if the sample should be discarded.
        """
        if confidence < self.discard_below:
            return 0.0

        # Map tiers from config
        tiers = self.config.get("negative_sampling", {}).get("confidence", {}).get("weight_tiers", [])
        if not tiers:
            # Fallback default tiers
            if confidence >= 0.95:
                return 1.0
            if confidence >= 0.90:
                return 0.8
            if confidence >= 0.80:
                return 0.5
            return 0.0

        for tier in tiers:
            if tier["min"] <= confidence <= tier["max"]:
                return tier["weight"]

        return 0.0
